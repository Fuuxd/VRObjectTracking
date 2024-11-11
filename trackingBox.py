import cv2
import numpy as np
import depthai as dai
from math import degrees
import time

def create_pipeline():
    """Create and configure the DepthAI pipeline"""
    pipeline = dai.Pipeline()
    
    # Define sources
    cam_rgb = pipeline.createColorCamera()
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    spatial_calc = pipeline.createSpatialLocationCalculator()
    
    # Define outputs
    xout_rgb = pipeline.createXLinkOut()
    xout_depth = pipeline.createXLinkOut()
    xout_spatial = pipeline.createXLinkOut()
    
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    xout_spatial.setStreamName("spatial")
    
    # Properties for RGB - adjusted to multiple of 16
    cam_rgb.setPreviewSize(320, 320)  # Changed from 300x300
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    
    # Properties for mono cameras
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    
    # Configure stereo
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(cam_rgb.getPreviewWidth(), cam_rgb.getPreviewHeight())
    
    # Configure spatial calculator
    spatial_calc.inputConfig.setWaitForMessage(False)
    spatial_calc.inputDepth.setBlocking(False)
    
    # Create ROI for spatial calculator
    roi = dai.SpatialLocationCalculatorConfigData()
    roi.roi = dai.Rect(dai.Point2f(0.48, 0.48), dai.Point2f(0.52, 0.52))
    config = dai.SpatialLocationCalculatorConfig()
    config.addROI(roi)
    spatial_calc.initialConfig.addROI(roi)
    
    # Linking
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(spatial_calc.inputDepth)
    spatial_calc.out.link(xout_spatial.input)
    stereo.depth.link(xout_depth.input)
    
    return pipeline

def get_object_pose(frame, depth_frame, spatial_data):
    """Extract 6-DoF data from detected object"""
    # For this example, we'll use ArUco markers
    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50, 4)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3  # Increase to make detection stricter
    parameters.adaptiveThreshWinSizeMax = 23  # Increase for higher sensitivity control
    parameters.adaptiveThreshConstant = 7    # Modify this value to control thresholding
    parameters.minMarkerPerimeterRate = 0.05  # Decrease for stricter perimeter detection
    parameters.maxMarkerPerimeterRate = 4.0   # Increase to limit marker size detection
    parameters.perspectiveRemovePixelPerCell = 4
    
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # Get rotation and translation vectors
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 
            0.05,  # Marker size in meters
            camera_matrix,  # Need to be calibrated for your OAK-D
            dist_coeffs     # Need to be calibrated for your OAK-D
        )
        
        # Draw bounding box for each detected marker
        for corner in corners:
            # Convert the corners to integer tuples for cv2.rectangle
            top_left = tuple(map(int, corner[0][0]))
            bottom_right = tuple(map(int, corner[0][2]))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Get spatial coordinates from depth
        x = spatial_data.spatialCoordinates.x
        y = spatial_data.spatialCoordinates.y
        z = spatial_data.spatialCoordinates.z
        
        # Convert rotation vector to Euler angles
        rot_mat = cv2.Rodrigues(rvecs[0])[0]
        euler_angles = cv2.RQDecomp3x3(rot_mat)[0]
        
        return {
            'position': {
                'x': x / 1000.0,  # Convert to meters
                'y': y / 1000.0,
                'z': z / 1000.0
            },
            'rotation': {
                'roll': degrees(euler_angles[0]),
                'pitch': degrees(euler_angles[1]),
                'yaw': degrees(euler_angles[2])
            }
        }
    return None

# Using the test camera matrix and distortion coefficients
camera_matrix = np.array([
    [858.5, 0.0, 640.0],
    [0.0, 858.5, 360.0],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

def main():
    # Create pipeline
    pipeline = create_pipeline()
    
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Get output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        q_spatial = device.getOutputQueue(name="spatial", maxSize=4, blocking=False)
        
        while True:
            # Get the newest data from the device
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()
            in_spatial = q_spatial.get()
            
            # Convert the frames
            frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()
            
            # Check if there are any spatial locations
            spatial_data = in_spatial.getSpatialLocations()
            if spatial_data:
                spatial_data = spatial_data[0]
                
                # Get pose data
                pose = get_object_pose(frame, depth_frame, spatial_data)
                
                if pose:
                    print(f"Position: {pose['position']}")
                    print(f"Rotation: {pose['rotation']}")
            
            # Display the frame with bounding box
            cv2.imshow("RGB", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    main()
