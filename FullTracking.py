import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import depthai as dai
import cv2
from torchvision import transforms
from ultralytics import YOLO
import time

# Assuming PoseNet and PoseRefineNet are in the same directory
from lib.network import PoseNet, PoseRefineNet

class DenseFusionPredictor:
    def __init__(self, model_path, refiner_path, num_objects=21, num_points=1000):
        self.num_points = num_points
        self.num_objects = num_objects
        
        # Camera parameters (OAK-D Lite default calibration - adjust these based on your calibration)
        self.cam_cx = 312.9869
        self.cam_cy = 241.3109
        self.cam_fx = 1066.778
        self.cam_fy = 1067.487
        self.cam_scale = 10000.0
        
        # Initialize normalization transform
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
        
        # Load models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = self._load_estimator(model_path)
        self.refiner = self._load_refiner(refiner_path)
        
        # Initialize depth mapping
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

    def _load_estimator(self, model_path):
        estimator = PoseNet(num_points=self.num_points, num_obj=self.num_objects)
        estimator.load_state_dict(torch.load(model_path, map_location=self.device))
        estimator.to(self.device)
        estimator.eval()
        return estimator

    def _load_refiner(self, refiner_path):
        refiner = PoseRefineNet(num_points=self.num_points, num_obj=self.num_objects)
        refiner.load_state_dict(torch.load(refiner_path, map_location=self.device))
        refiner.to(self.device)
        refiner.eval()
        return refiner

    def create_pipeline(self):
        pipeline = dai.Pipeline()

        # Define sources and outputs
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        camRgb = pipeline.create(dai.node.ColorCamera)

        # Create outputs
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutDepth.setStreamName("depth")

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        camRgb.setPreviewSize(640, 480)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        # Stereo depth configuration
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        camRgb.preview.link(xoutRgb.input)
        stereo.depth.link(xoutDepth.input)

        return pipeline

    def process_frame(self, color_frame, depth_frame, bbox, obj_id):
        """Process a single frame and return pose estimation"""
        rmin, rmax, cmin, cmax = bbox
        
        # Prepare depth mask
        mask_depth = (depth_frame != 0).astype(np.uint8)
        
        # Get points for pose estimation
        choose = mask_depth[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

        # Prepare point cloud
        depth_masked = depth_frame[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        
        pt2 = depth_masked / self.cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        # Prepare image
        img_masked = color_frame[:, :, :3]
        img_masked = np.transpose(img_masked, (2, 0, 1))
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]

        # Convert to torch tensors
        cloud = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(np.array([choose]))
        img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
        index = torch.LongTensor([obj_id - 1])

        # Move to device
        cloud = Variable(cloud).to(self.device)
        choose = Variable(choose).to(self.device)
        img_masked = Variable(img_masked).to(self.device)
        index = Variable(index).to(self.device)

        # Reshape for network
        cloud = cloud.view(1, self.num_points, 3)
        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

        # Get initial pose estimation
        with torch.no_grad():
            pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)
            pred_c = pred_c.view(1, self.num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(self.num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (cloud.view(self.num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

        return my_r, my_t

def create_yolo_pipeline(pipeline):
    """Add YOLO nodes to the pipeline"""
    # Neural Network node for YOLO
    yolo_det_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    
    # Set path to your YOLO blob file
    yolo_det_nn.setBlobPath("yolov8n_openvino_2022.1_6shave.blob")
    yolo_det_nn.setConfidenceThreshold(0.5)
    yolo_det_nn.setNumClasses(80)  # Adjust based on your model
    yolo_det_nn.setCoordinateSize(4)
    yolo_det_nn.setAnchors([])  # YOLOv8 doesn't use anchors
    yolo_det_nn.setIouThreshold(0.5)
    
    # Create YOLO output
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("detections")
    
    # Link YOLO to the RGB camera
    camRgb = pipeline.createColorCamera()
    camRgb.preview.link(yolo_det_nn.input)
    yolo_det_nn.out.link(nnOut.input)
    
    return nnOut

def frame_norm(frame, bbox):
    """Convert bounding box coordinates to pixel values"""
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def main():
    # Initialize DenseFusion predictor
    predictor = DenseFusionPredictor(
        model_path='pose_model.pth',
        refiner_path='pose_refine_model.pth'
    )
    
    # Create and start pipeline
    pipeline = predictor.create_pipeline()
    
    # Add YOLO detection to pipeline
    nnOut = create_yolo_pipeline(pipeline)
    
    # Object class mapping (adjust based on your needs)
    class_map = {
        0: 1,  # Map YOLO class 0 (person) to your model's class 1
        # Add more mappings as needed
    }
    
    with dai.Device(pipeline) as device:
        # Get output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()
            in_nn = q_nn.get()
            
            if in_rgb is not None:
                rgb_frame = in_rgb.getCvFrame()
                depth_frame = in_depth.getFrame()
                
                # Get detections
                detections = in_nn.detections
                
                # Process each detection
                for detection in detections:
                    # Get bounding box
                    bbox = frame_norm(rgb_frame, (detection.xmin, detection.ymin, 
                                                detection.xmax, detection.ymax))
                    
                    # Convert to format expected by DenseFusion
                    rmin = bbox[1]  # ymin
                    rmax = bbox[3]  # ymax
                    cmin = bbox[0]  # xmin
                    cmax = bbox[2]  # xmax
                    
                    # Add padding to bbox if needed
                    padding = 10
                    rmin = max(0, rmin - padding)
                    rmax = min(rgb_frame.shape[0], rmax + padding)
                    cmin = max(0, cmin - padding)
                    cmax = min(rgb_frame.shape[1], cmax + padding)
                    
                    # Map YOLO class to your model's class ID
                    if detection.label in class_map:
                        obj_id = class_map[detection.label]
                        
                        try:
                            # Get pose estimation
                            rotation, translation = predictor.process_frame(
                                rgb_frame, depth_frame, [rmin, rmax, cmin, cmax], obj_id
                            )
                            
                            # Visualization
                            cv2.rectangle(rgb_frame, (cmin, rmin), (cmax, rmax), (0, 255, 0), 2)
                            
                            # Draw axes to show pose
                            R = cv2.Rodrigues(rotation)[0]
                            tvec = translation.reshape(3, 1)
                            
                            # Project 3D axes onto the image
                            axis_length = 0.1  # 10cm
                            points_3d = np.float32([[0, 0, 0], 
                                                  [axis_length, 0, 0], 
                                                  [0, axis_length, 0], 
                                                  [0, 0, axis_length]])
                            
                            camera_matrix = np.array([[predictor.cam_fx, 0, predictor.cam_cx],
                                                    [0, predictor.cam_fy, predictor.cam_cy],
                                                    [0, 0, 1]], dtype=np.float32)
                            
                            dist_coeffs = np.zeros((4, 1))
                            
                            points_2d, _ = cv2.projectPoints(points_3d, R, tvec, 
                                                           camera_matrix, dist_coeffs)
                            
                            # Draw the axes
                            origin = tuple(points_2d[0].ravel().astype(int))
                            point_x = tuple(points_2d[1].ravel().astype(int))
                            point_y = tuple(points_2d[2].ravel().astype(int))
                            point_z = tuple(points_2d[3].ravel().astype(int))
                            
                            cv2.line(rgb_frame, origin, point_x, (0, 0, 255), 2)  # X-axis: Red
                            cv2.line(rgb_frame, origin, point_y, (0, 255, 0), 2)  # Y-axis: Green
                            cv2.line(rgb_frame, origin, point_z, (255, 0, 0), 2)  # Z-axis: Blue
                            
                            # Add text with position information
                            pos_text = f"X:{translation[0]:.2f} Y:{translation[1]:.2f} Z:{translation[2]:.2f}"
                            cv2.putText(rgb_frame, pos_text, (cmin, rmin - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                        except Exception as e:
                            print(f"Error processing detection: {e}")
                            continue
                
                # Show the frame
                cv2.imshow("RGB", rgb_frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == "__main__":
    main()