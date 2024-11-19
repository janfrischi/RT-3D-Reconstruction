# object_detection_and_point_cloud_processing.py
import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import open3d as o3d
import open3d.core as o3c

from ultralytics import YOLO

class CameraManager:
    """ Manages the initialization, retrieval and closing of the ZED cameras."""
    def __init__(self, sn_cam1, sn_cam2, init_params1, init_params2):
        # Initialize the ZED camera objects
        self.zed1 = sl.Camera()
        self.zed2 = sl.Camera()
        self.sn_cam1 = sn_cam1
        self.sn_cam2 = sn_cam2
        self.init_params1 = self.get_init_params(sn_cam1, init_params1)
        self.init_params2 = self.get_init_params(sn_cam2, init_params2)
        self.open_cameras()
        self._get_calibration_params()

    def get_init_params(self, serial_number, custom_params):
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)
        init_params.camera_resolution = getattr(sl.RESOLUTION, custom_params["resolution"])
        init_params.camera_fps = custom_params["fps"]
        init_params.depth_mode = getattr(sl.DEPTH_MODE, custom_params["depth_mode"])
        init_params.depth_minimum_distance = custom_params["min_distance"]
        init_params.coordinate_units = getattr(sl.UNIT, custom_params["units"])
        return init_params

    def open_cameras(self):
        err1 = self.zed1.open(self.init_params1)
        if err1 != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Error opening ZED camera 1: {err1}")

        err2 = self.zed2.open(self.init_params2)
        if err2 != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Error opening ZED camera 2: {err2}")

    def _get_calibration_params(self):
        self.calibration_params1 = self.zed1.get_camera_information().camera_configuration.calibration_parameters
        self.calibration_params2 = self.zed2.get_camera_information().camera_configuration.calibration_parameters

    def retrieve_images_and_depths(self):
        image1, depth1 = sl.Mat(), sl.Mat()
        image2, depth2 = sl.Mat(), sl.Mat()
        if self.zed1.grab() == sl.ERROR_CODE.SUCCESS and self.zed2.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed1.retrieve_image(image1, sl.VIEW.LEFT)
            self.zed2.retrieve_image(image2, sl.VIEW.LEFT)
            self.zed1.retrieve_measure(depth1, sl.MEASURE.DEPTH)
            self.zed2.retrieve_measure(depth2, sl.MEASURE.DEPTH)
            return image1, depth1, image2, depth2
        else:
            return None, None, None, None

    def close(self):
        self.zed1.close()
        self.zed2.close()


class YOLOModel:
    """ Manages the YOLO model for object detection and tracking."""
    def __init__(self, model_path, device):
        self.model = YOLO(model_path).to(device)

    def track(self, frame, device):
        return self.model.track(
            source=frame,
            imgsz=640,
            classes=[39, 41],
            persist=True,
            retina_masks=True,
            conf=0.4,
            device=device,
            tracker="trackers/botsort.yaml"
        )

class PointCloudProcessor:
    @staticmethod
    def erode_mask(mask, iterations=1):
        kernel = np.ones((12, 12), np.uint8)
        return cv2.erode(mask, kernel, iterations=iterations)

    @staticmethod
    def convert_mask_to_3d_points(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
        mask_indices = torch.tensor(mask_indices, device=device)
        u_coords = mask_indices[:, 1]
        v_coords = mask_indices[:, 0]
        depth_values = depth_map[v_coords, u_coords].to(device)
        valid_mask = (depth_values > 0) & ~torch.isnan(depth_values)
        u_coords = u_coords[valid_mask]
        v_coords = v_coords[valid_mask]
        depth_values = depth_values[valid_mask]
        x_coords = (u_coords - cx) * depth_values / fx
        y_coords = (v_coords - cy) * depth_values / fy
        z_coords = depth_values
        return torch.stack((x_coords, y_coords, z_coords), dim=-1)

    @staticmethod
    def downsample_point_cloud(point_cloud, voxel_size=0.005):
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(point_cloud, device=o3c.Device("CUDA:0")))
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return downsampled_pcd.point.positions.cpu().numpy()

    @staticmethod
    def filter_outliers_sor(point_cloud, nb_neighbors=20, std_ratio=5):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return np.asarray(filtered_pcd.points)

    @staticmethod
    def calculate_centroid(point_cloud):
        return np.mean(point_cloud, axis=0)

    @staticmethod
    def fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.1):
        pcs1 = []
        pcs2 = []
        class_dict1 = {}
        class_dict2 = {}
        for pc, class_id in point_clouds_camera1:
            if class_id not in class_dict1:
                class_dict1[class_id] = []
            class_dict1[class_id].append(pc)
        for pc, class_id in point_clouds_camera2:
            if class_id not in class_dict2:
                class_dict2[class_id] = []
            class_dict2[class_id].append(pc)
        fused_point_clouds = []
        for class_id in set(class_dict1.keys()).union(class_dict2.keys()):
            pcs1 = class_dict1.get(class_id, [])
            pcs2 = class_dict2.get(class_id, [])
            if len(pcs1) == 1 and len(pcs2) == 1:
                fused_pc = PointCloudProcessor.filter_outliers_sor(np.vstack((pcs1[0], pcs2[0])))
                fused_point_clouds.append((fused_pc, class_id))
            else:
                for pc1 in pcs1:
                    best_distance = float('inf')
                    best_match = None
                    centroid1 = PointCloudProcessor.calculate_centroid(pc1)
                    for pc2 in pcs2:
                        centroid2 = PointCloudProcessor.calculate_centroid(pc2)
                        distance = np.linalg.norm(centroid1 - centroid2)
                        if distance < best_distance and distance < distance_threshold:
                            best_distance = distance
                            best_match = pc2
                    if best_match is not None:
                        fused_pc = PointCloudProcessor.filter_outliers_sor(np.vstack((pc1, best_match)))
                        fused_point_clouds.append((fused_pc, class_id))
                        pcs2 = [pc for pc in pcs2 if not np.array_equal(pc, best_match)]
                    else:
                        fused_point_clouds.append((pc1, class_id))
                for pc2 in pcs2:
                    fused_point_clouds.append((pc2, class_id))

        return class_dict1, class_dict2, pcs1, pcs2, fused_point_clouds

# Manages the main workflow of the application
class MainApp:
    def __init__(self, model_path, sn_cam1, sn_cam2, color_map, class_names, T_chess_cam1, T_chess_cam2, T_robot_chess, init_params1, init_params2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.color_map = color_map
        self.class_names = class_names

        # Initialize the camera manager, YOLO model, and point cloud processor instances
        self.camera_manager = CameraManager(sn_cam1, sn_cam2, init_params1, init_params2)
        self.yolo_model = YOLOModel(model_path, self.device)
        self.pc_processor = PointCloudProcessor()

        # Get the calibration parameters and the transformation matrices
        self.T_robot_cam1, self.T_robot_cam2 = self._get_transformations(T_chess_cam1, T_chess_cam2, T_robot_chess)
        self.rotation_robot_cam1 = self.T_robot_cam1[:3, :3]
        self.rotation_robot_cam2 = self.T_robot_cam2[:3, :3]
        self.origin_cam1 = self.T_robot_cam1[:3, 3]
        self.origin_cam2 = self.T_robot_cam2[:3, 3]
        self._init_display()

    def _get_transformations(self, T_chess_cam1, T_chess_cam2, T_robot_chess):
        T_robot_cam1 = np.dot(T_robot_chess, T_chess_cam1)
        T_robot_cam2 = np.dot(T_robot_chess, T_chess_cam2)
        return T_robot_cam1, T_robot_cam2

    def _init_display(self):
        cv2.namedWindow("YOLO11 Segmentation+Tracking")
        self.fps_values = []
        self.frame_count = 0

    def run(self):
        # Initialize the key variable and the point clouds lists
        key = ''
        point_clouds_camera1 = []
        point_clouds_camera2 = []

        # Main loop to process the images and point clouds
        while key != ord('q'):
            start_time = time.time()
            image1, depth1, image2, depth2 = self.camera_manager.retrieve_images_and_depths()
            if image1 is None or image2 is None:
                continue
            frame1 = cv2.cvtColor(image1.get_data(), cv2.COLOR_BGRA2BGR)
            frame2 = cv2.cvtColor(image2.get_data(), cv2.COLOR_BGRA2BGR)
            results1 = self.yolo_model.track(frame1, self.device)
            results2 = self.yolo_model.track(frame2, self.device)
            zed_depth_np1 = depth1.get_data()
            zed_depth_np2 = depth2.get_data()
            if zed_depth_np1 is None or zed_depth_np2 is None:
                continue
            annotated_frame1 = results1[0].plot(line_width=2, font_size=18)
            annotated_frame2 = results2[0].plot(line_width=2, font_size=18)
            masks1 = results1[0].masks
            masks2 = results2[0].masks
            class_ids1 = results1[0].boxes.cls.cpu().numpy()
            class_ids2 = results2[0].boxes.cls.cpu().numpy()
            self._process_masks(masks1, class_ids1, zed_depth_np1, point_clouds_camera1, self.rotation_robot_cam1, self.origin_cam1, self.camera_manager.calibration_params1)
            self._process_masks(masks2, class_ids2, zed_depth_np2, point_clouds_camera2, self.rotation_robot_cam2, self.origin_cam2, self.camera_manager.calibration_params2)
            dict1, dict2, pcs_1, pcs_2, fused_pc = self.pc_processor.fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.3)
            self._display_frames(annotated_frame1, annotated_frame2, frame1, frame2, start_time)
            yield fused_pc
            point_clouds_camera1.clear()
            point_clouds_camera2.clear()
            pcs_1.clear()
            pcs_2.clear()
            fused_pc.clear()
            key = cv2.waitKey(1)
        self.camera_manager.close()

    def _process_masks(self, masks, class_ids, zed_depth_np, point_clouds_camera, rotation_robot_cam, origin_cam, calibration_params):
        if masks is not None:
            depth_map = torch.from_numpy(zed_depth_np).to(self.device)
            for i, mask in enumerate(masks.data):
                mask = mask.cpu().numpy()
                mask = self.pc_processor.erode_mask(mask, iterations=1)
                mask_indices = np.argwhere(mask > 0)
                with torch.amp.autocast('cuda'):
                    points_3d = self.pc_processor.convert_mask_to_3d_points(mask_indices, depth_map, calibration_params.left_cam.cx, calibration_params.left_cam.cy, calibration_params.left_cam.fx, calibration_params.left_cam.fy)
                if points_3d.size(0) > 0:
                    point_cloud_cam = points_3d.cpu().numpy()
                    point_cloud_cam_transformed = np.dot(rotation_robot_cam, point_cloud_cam.T).T + origin_cam
                    point_cloud_cam_transformed = self.pc_processor.downsample_point_cloud(point_cloud_cam_transformed, voxel_size=0.01)
                    point_clouds_camera.append((point_cloud_cam_transformed, int(class_ids[i])))

    def _display_frames(self, annotated_frame1, annotated_frame2, frame1, frame2, start_time):
        self.frame_count += 1
        fps = 1.0 / (time.time() - start_time)
        self.fps_values.append(fps)
        if len(self.fps_values) > 10:
            self.fps_values.pop(0)
        avg_fps = sum(self.fps_values) / len(self.fps_values)
        if annotated_frame1 is not None:
            cv2.putText(annotated_frame1, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            height, width, _ = frame1.shape
            cv2.resizeWindow("YOLO11 Segmentation+Tracking", width, height)
            cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame1)
        if annotated_frame2 is not None:
            cv2.putText(annotated_frame2, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            height, width, _ = frame2.shape
            cv2.resizeWindow("YOLO11 Segmentation+Tracking", width, height)
            cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame2)
        combined_frame = cv2.hconcat([annotated_frame1, annotated_frame2])
        combined_frame = cv2.resize(combined_frame, (combined_frame.shape[1] // 2, combined_frame.shape[0] // 2))
        cv2.imshow("YOLO11 Segmentation+Tracking", combined_frame)