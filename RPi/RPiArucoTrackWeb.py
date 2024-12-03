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
    cam.setFps(30)
    
    # Create stereo depth node
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    spatialCalculator = pipeline.createSpatialLocationCalculator()
    
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setCamera("right")
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setSubpixel(True)  # Improves depth precision
    
    # Configure spatial location calculator
    spatialCalculator.inputConfig.setWaitForMessage(False)
    config = dai.SpatialLocationCalculatorConfigData()
    config.roi = dai.Rect(dai.Point2f(0, 0), dai.Point2f(1, 1))  # Default ROI for initialization
    spatialCalculator.initialConfig.addROI(config)
    
    # Output and input nodes
    xoutColor = pipeline.createXLinkOut()
    xoutSpatialData = pipeline.createXLinkOut()
    xinSpatialCalcConfig = pipeline.createXLinkIn()
    xoutColor.setStreamName("preview")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
    
    # Linking
    cam.preview.link(xoutColor.input)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(spatialCalculator.inputDepth)
    spatialCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialCalculator.inputConfig)
    
    return pipeline

async def websocket_server():
    async def send_tracking_data(websocket):
        try:
            while True:
                if not tracking_queue.empty():
                    data = tracking_queue.get()
                    message = f"id:{data['id']},x:{data['x']:.6f},y:{data['y']:.6f},z:{data['z']:.6f},rx:{data['rx']:.6f},ry:{data['ry']:.6f},rz:{data['rz']:.6f}"
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
    pipeline = create_pipeline()
    device = dai.Device(pipeline)

    preview_queue = device.getOutputQueue("preview", maxSize=1, blocking=False)
    spatial_data_queue = device.getOutputQueue("spatialData", maxSize=1, blocking=False)
    spatial_calc_config_in_queue = device.getInputQueue("spatialCalcConfig")

    # Start WebSocket server in a separate thread
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()

    rgb_camsocket = dai.CameraBoardSocket.CAM_A
    calib_data = device.readCalibration()
    
    camera_matrix = np.array(calib_data.getCameraIntrinsics(rgb_camsocket, 1920, 1080))
    dist_coeffs = np.array(calib_data.getDistortionCoefficients(rgb_camsocket)[:5]).reshape((1,5))
    
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
                roi = corners[i][0]
                x_min, y_min = roi[:, 0].min() / frame.shape[1], roi[:, 1].min() / frame.shape[0]
                x_max, y_max = roi[:, 0].max() / frame.shape[1], roi[:, 1].max() / frame.shape[0]
                
                # Configure spatial location
                roi_config = dai.SpatialLocationCalculatorConfigData()
                roi_config.roi = dai.Rect(dai.Point2f(x_min, y_min), dai.Point2f(x_max, y_max))
                
                spatial_calc_config = dai.SpatialLocationCalculatorConfig()
                spatial_calc_config.addROI(roi_config)
                spatial_calc_config_in_queue.send(spatial_calc_config)
                
                # Get spatial data
                spatial_data = spatial_data_queue.get().getSpatialLocations()
                if spatial_data:
                    loc = spatial_data[0].spatialCoordinates
                    marker_position = (loc.x / 1000, loc.y / 1000, loc.z / 1000)  # Convert to meters
                
                # Send data to WebSocket queue
                tracking_data = {
                    'id': ids[i][0],
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
    main()