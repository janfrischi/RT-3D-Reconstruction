import numpy as np
import pyzed.sl as sl
import cv2
import torch
import open3d as o3d

from ultralytics import YOLO

class ZEDCameraSystem:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load the pre-trained YOLOv11 model and move it to the device
        self.model = YOLO("yolo11m-seg.pt").to(self.device)

        # Initialize the ZED camera objects
        self.zed1 = sl.Camera()
        self.zed2 = sl.Camera()

        # Set the serial numbers of the cameras
        self.sn_cam1 = 33137761
        self.sn_cam2 = 36829049

        # Set the initialization parameters for camera 1
        self.init_params1 = sl.InitParameters()
        self.init_params1.set_from_serial_number(self.sn_cam1)
        self.init_params1.camera_resolution = sl.RESOLUTION.HD720
        self.init_params1.camera_fps = 30
        self.init_params1.depth_mode = sl.DEPTH_MODE.NEURAL  # TODO: Check what if NEURAL and NEURAL_PLUS differ in performance
        self.init_params1.depth_minimum_distance = 0.4
        self.init_params1.coordinate_units = sl.UNIT.METER

        # Set the initialization parameters for camera 2
        self.init_params2 = sl.InitParameters()
        self.init_params2.set_from_serial_number(self.sn_cam2)
        self.init_params2.camera_resolution = sl.RESOLUTION.HD720
        self.init_params2.camera_fps = 30
        self.init_params2.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params2.depth_minimum_distance = 0.4
        self.init_params2.coordinate_units = sl.UNIT.METER

        # Check if the cameras were successfully opened
        self.err1 = self.zed1.open(self.init_params1)
        if self.err1 != sl.ERROR_CODE.SUCCESS:
            print(f"Error opening ZED camera 1: {self.err1}")
            exit(1)

        self.err2 = self.zed2.open(self.init_params2)
        if self.err2 != sl.ERROR_CODE.SUCCESS:
            print(f"Error opening ZED camera 2: {self.err2}")
            exit(1)

        # Get the camera calibration parameters
        self.calibration_params1 = self.zed1.get_camera_information().camera_configuration.calibration_parameters
        self.fx1, self.fy1 = self.calibration_params1.left_cam.fx, self.calibration_params1.left_cam.fy
        self.cx1, self.cy1 = self.calibration_params1.left_cam.cx, self.calibration_params1.left_cam.cy

        self.calibration_params2 = self.zed2.get_camera_information().camera_configuration.calibration_parameters
        self.fx2, self.fy2 = self.calibration_params2.left_cam.fx, self.calibration_params2.left_cam.fy
        self.cx2, self.cy2 = self.calibration_params2.left_cam.cx, self.calibration_params2.left_cam.cy

        # Define the transformation matrices from the chessboard to the camera frames
        # These matrices can be obtained from the extrinsic calibration process
        self.T_chess_cam1 = np.array([[0.6653, 0.4827, -0.5696, 0.5868],
                                      [-0.7466, 0.4314, -0.5065, 0.7718],
                                      [0.0012, 0.7622, 0.6473, -0.7245],
                                      [0.0000, 0.0000, 0.0000, 1.0000]])

        self.T_chess_cam2 = np.array([[0.3981, -0.6302, 0.6666, -0.5739],
                                      [0.9173, 0.2688, -0.2937, 0.3581],
                                      [0.0059, 0.7284, 0.6851, -0.6835],
                                      [0.0000, 0.0000, 0.0000, 1.0000]])

        self.T_robot_chess = np.array([[-1.0000, 0.0000, 0.0000, 0.3580],
                                       [0.0000, 1.0000, 0.0000, 0.0300],
                                       [0.0000, 0.0000, -1.0000, 0.0060],
                                       [0.0000, 0.0000, 0.0000, 1.0000]])

        # Calculate the transformation matrices from the robot frame to the camera frames
        self.T_robot_cam1 = np.dot(self.T_robot_chess, self.T_chess_cam1)
        self.T_robot_cam2 = np.dot(self.T_robot_chess, self.T_chess_cam2)

        # Extract the rotation matrices and translation vectors from the transformation matrices
        self.rotation_robot_cam1 = self.T_robot_cam1[:3, :3]
        self.rotation_robot_cam2 = self.T_robot_cam2[:3, :3]

        self.origin_cam1 = self.T_robot_cam1[:3, 3]
        self.origin_cam2 = self.T_robot_cam2[:3, 3]

        self.distance_cam1 = np.linalg.norm(self.origin_cam1 - np.array([0, 0, 0]))
        self.distance_cam2 = np.linalg.norm(self.origin_cam2 - np.array([0, 0, 0]))

        print(f"Distance from robot frame to camera frame 1: {self.distance_cam1:.4f} meters")
        print(f"Distance from robot frame to camera frame 2: {self.distance_cam2:.4f} meters")