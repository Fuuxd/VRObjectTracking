import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import depthai as dai
import cv2
from torchvision import transforms

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
        camRgb = pipeline.createColorCamera()
        spatialCalcHost = pipeline.createXLinkOut()
        rgbOut = pipeline.createXLinkOut()
        depthOut = pipeline.createXLinkOut()

        spatialCalcHost.setStreamName("spatialData")
        rgbOut.setStreamName("rgb")
        depthOut.setStreamName("depth")

        # Properties
        camRgb.setPreviewSize(640, 480)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Linking
        camRgb.preview.link(rgbOut.input)

        # Create stereo depth
        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.depth.link(depthOut.input)

        return pipeline

    def process_frame(self, color_frame, depth_frame, bbox, obj_id):
        """
        Process a single frame to estimate pose.
        
        Args:
            color_frame: RGB image as numpy array
            depth_frame: Depth image as numpy array (uint16)
            bbox: Bounding box as [rmin, rmax, cmin, cmax]
            obj_id: Object ID corresponding to the training set
            
        Returns:
            tuple: (rotation as quaternion, translation vector)
        """
        # Convert frames to torch tensors and normalize
        img_tensor = torch.from_numpy(color_frame).float().cuda() / 255.0
        
        # Convert depth from uint16 to float32 before creating tensor
        depth_frame = depth_frame.astype(np.float32)
        depth_tensor = torch.from_numpy(depth_frame).cuda()
        
        # Extract features from the bounded region
        rmin, rmax, cmin, cmax = bbox
        choose = depth_tensor[rmin:rmax, cmin:cmax].contiguous().view(-1)
        
        # Get points from the depth image
        cam_points = self.get_model_points(choose, depth_tensor[rmin:rmax, cmin:cmax])
        
        # Forward pass through PoseNet
        with torch.no_grad():
            points, choose, img_feat = self.get_features(img_tensor, depth_tensor, bbox)
            pred_r, pred_t = self.pose_net(points, choose, img_feat)
            
            # Refine prediction if refiner is available
            if self.pose_refine_net is not None:
                for ite in range(0, 2):
                    pred_r, pred_t = self.pose_refine_net(points, choose, img_feat, pred_r, pred_t)
        
        # Convert rotation matrix to quaternion
        pred_r = pred_r.cpu().numpy()
        r = Rotation.from_matrix(pred_r)
        quaternion = r.as_quat()
        
        # Get translation
        translation = pred_t.cpu().numpy().flatten()
        
        return quaternion, translation

def main():
    # Initialize DenseFusion predictor
    predictor = DenseFusionPredictor(
        model_path='path/to/pose_model.pth',
        refiner_path='path/to/pose_refiner.pth'
    )
    
    # Create and start pipeline
    pipeline = predictor.create_pipeline()
    with dai.Device(pipeline) as device:
        # Get output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        while True:
            if q_rgb.has():
                rgb_frame = q_rgb.get().getCvFrame()
                depth_frame = q_depth.get().getFrame()
                
                # Here you would add object detection to get bounding boxes
                # For example purposes, using a static bbox
                bbox = [100, 300, 100, 300]  # [rmin, rmax, cmin, cmax]
                obj_id = 1  # Object ID from your training set
                
                # Get pose estimation
                rotation, translation = predictor.process_frame(
                    rgb_frame, depth_frame, bbox, obj_id
                )
                
                # Visualization code would go here
                cv2.imshow("RGB", rgb_frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == "__main__":
    main()