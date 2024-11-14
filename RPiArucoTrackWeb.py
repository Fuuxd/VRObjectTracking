#file equivalent to arucoTrackWeb.py intended for headless use in an RPI4,M.B

import cv2
import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation
import asyncio
import websockets
import threading
from queue import Queue

# Global queue for passing tracking data to WebSocket thread
tracking_queue = Queue()

def create_pipeline():
    pipeline = dai.Pipeline()
    
    # Create color camera node
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(1280, 720)
    cam.setInterleaved(False)
    cam.setFps(60)
    
    # Create output node
    xout = pipeline.createXLinkOut()
    xout.setStreamName("preview")
    cam.preview.link(xout.input)
    
    return pipeline

async def websocket_server():
    async def send_tracking_data(websocket, path):
        try:
            while True:
                if not tracking_queue.empty():
                    data = tracking_queue.get()
                    message = f"x:{data['x']:.6f},y:{data['y']:.6f},z:{data['z']:.6f},rx:{data['rx']:.6f},ry:{data['ry']:.6f},rz:{data['rz']:.6f}"
                    print(f"Sending: {message}")
                    await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")

    async with websockets.serve(send_tracking_data, "localhost", 8080):
        print("WebSocket server started on ws://localhost:8080")
        await asyncio.Future()  # run forever

def run_websocket_server():
    asyncio.run(websocket_server())

def main():
    # Initialize ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    
    # Initialize device
    device = dai.Device()
    pipeline = create_pipeline()
    device.startPipeline(pipeline)

    # Start WebSocket server in a separate thread
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()

    rgb_camsocket = dai.CameraBoardSocket.CAM_A
    calib_data = device.readCalibration()
    
    camera_matrix = np.array(calib_data.getCameraIntrinsics(rgb_camsocket, 1920, 1080))
    dist_coeffs = np.array(calib_data.getDistortionCoefficients(rgb_camsocket)[:5]).reshape((1,5))
    
    # Get output queue
    preview_queue = device.getOutputQueue("preview", maxSize=1, blocking=False)
    
    while True:
        preview_frame = preview_queue.get()
        frame = preview_frame.getCvFrame()
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        
        if ids is not None:

            # Calculate pose for each marker
            for i in range(len(ids)):
                # Get rotation and translation vectors
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
                
                # Convert rotation vector to euler angles
                rot_matrix = cv2.Rodrigues(rvec)[0]
                r = Rotation.from_matrix(rot_matrix)
                euler_angles = r.as_euler('xyz', degrees=True)
                
                # Get marker position
                marker_position = tvec[0][0]
                
                # Send data to WebSocket queue
                tracking_data = {
                    'x': marker_position[0],
                    'y': marker_position[1],
                    'z': marker_position[2],
                    'rx': euler_angles[0],
                    'ry': euler_angles[1],
                    'rz': euler_angles[2]
                }
                tracking_queue.put(tracking_data)
                
    device.close()

if __name__ == "__main__":
    while True:
        try:
            main()  # Call your main function here
        except Exception as e:
            print(f"An error occurred: {e}")