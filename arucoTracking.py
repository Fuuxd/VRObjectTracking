import cv2
import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

def create_pipeline():
    pipeline = dai.Pipeline()
    
    # Create color camera node
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(1280, 720)
    cam.setInterleaved(False)
    cam.setFps(30)
    
    # Create output node
    xout = pipeline.createXLinkOut()
    xout.setStreamName("preview")
    cam.preview.link(xout.input)
    
    return pipeline

def main():
    # Initialize ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    
    # Initialize device
    device = dai.Device()
    pipeline = create_pipeline()
    device.startPipeline(pipeline)


    rgb_camsocket = dai.CameraBoardSocket.CAM_A
    calib_data = device.readCalibration()
    
    camera_matrix = np.array( calib_data.getCameraIntrinsics(rgb_camsocket, 1920, 1080))
    dist_coeffs = np.array(calib_data.getDistortionCoefficients(rgb_camsocket)[:5]).reshape((1,5))
    
    # Get output queue
    preview_queue = device.getOutputQueue("preview", maxSize=1, blocking=False)
    
    while True:
        preview_frame = preview_queue.get()
        frame = preview_frame.getCvFrame()
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        
        if ids is not None:
            # Draw markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate pose for each marker
            
            for i in range(len(ids)):
                # Get rotation and translation vectors
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
                
                # Draw axis
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                
                # Convert rotation vector to euler angles
                rot_matrix = cv2.Rodrigues(rvec)[0]
                r = Rotation.from_matrix(rot_matrix)
                euler_angles = r.as_euler('xyz', degrees=True)
                
                # Get marker position
                marker_position = tvec[0][0]
                
                # Display position and rotation
                text_position = tuple(corners[i][0][0].astype(int))
                cv2.putText(frame, f"ID: {ids[i][0]}", 
                           (text_position[0], text_position[1] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Pos: {marker_position[0]:.2f}, {marker_position[1]:.2f}, {marker_position[2]:.2f}", 
                           (text_position[0], text_position[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Rot: {euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}", 
                           (text_position[0], text_position[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("ArUco Marker Detection", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    device.close()

if __name__ == "__main__":
    main()