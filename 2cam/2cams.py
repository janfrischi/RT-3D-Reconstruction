import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import open3d as o3d
import random
from ultralytics import YOLO

# Define a color map for different classes
color_map = {
    0: [15, 82, 186],  # Person - sapphire
    39: [255, 255, 0],  # Bottle - yellow
    41: [63, 224, 208],  # Cup - turquoise
    62: [255, 0, 255],  # Laptop - magenta
    64: [0, 0, 128],  # Mouse - navy
    66: [255, 0, 0]  # Keyboard - red
}

# Define class names for the detected objects
class_names = {0: "Person",
               39: "Bottle",
               41: "Cup",
               62: "Laptop",
               64: "Mouse",
               66: "Keyboard",
               73: "Book"}

def erode_mask(mask, iterations=1):
    kernel = np.ones((8, 8), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask

# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
    mask_indices = torch.tensor(mask_indices, device=device)
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]
    depth_values = depth_map[v_coords, u_coords].to(device)
    valid_mask = depth_values > 0
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depth_values = depth_values[valid_mask]
    x_coords = (u_coords - cx) * depth_values / fx
    y_coords = (v_coords - cy) * depth_values / fy
    z_coords = depth_values
    return torch.stack((x_coords, y_coords, z_coords), dim=-1)

def random_sample_pointcloud(pc, fraction):
    n_points = pc.shape[0]
    sample_size = int(n_points * fraction)
    if sample_size <= 0:
        return pc
    indices = random.sample(range(n_points), sample_size)
    return pc[indices]

def process_camera(zed, model, device, vis, coordinate_frame, transformation_matrix, window_name):
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy = calibration_params.left_cam.fx, calibration_params.left_cam.fy
    cx, cy = calibration_params.left_cam.cx, calibration_params.left_cam.cy
    img_width, img_height = calibration_params.left_cam.image_size.width, calibration_params.left_cam.image_size.height

    image = sl.Mat()
    depth = sl.Mat()
    cv2.namedWindow(window_name)

    fps_values = []
    frame_count = 0
    update_frequency = 5

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        start_time = time.time()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        depth_retrieval_result = zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        if depth_retrieval_result != sl.ERROR_CODE.SUCCESS:
            print(f"Error retrieving depth: {depth_retrieval_result}")
            return None, None, []

        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model.track(
            source=frame,
            imgsz=640,
            max_det=10,
            classes=[39, 64, 73],
            half=True,
            persist=True,
            retina_masks=True,
            conf=0.3,
            device=device,
            tracker="ultralytics/cfg/trackers/bytetrack.yaml"
        )

        zed_depth_np = depth.get_data()
        if zed_depth_np is None:
            print("Error: Depth map is empty")
            return None, None, []

        annotated_frame = results[0].plot(line_width=2, font_size=18)
        masks = results[0].masks
        class_ids = results[0].boxes.cls.cpu().numpy()

        point_clouds = []

        for i, mask in enumerate(masks.data):
            mask = mask.cpu().numpy()
            mask = erode_mask(mask, iterations=1)
            mask_indices = np.argwhere(mask > 0)

            depth_map = torch.from_numpy(zed_depth_np).to(device)

            with torch.cuda.amp.autocast():
                points_3d = get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy)

            if points_3d.size(0) > 0:
                point_cloud_cam = points_3d.cpu().numpy()
                point_cloud_np_transformed = np.dot(transformation_matrix[:3, :3], point_cloud_cam.T).T + transformation_matrix[:3, 3]
                point_clouds.append(point_cloud_np_transformed)
                class_id = int(class_ids[i])
                print(f"Detected object: {class_names.get(class_id, 'Unknown')}")

        if point_clouds and frame_count % update_frequency == 0:
            vis.clear_geometries()
            for i, pc in enumerate(point_clouds):
                sampled_pc = random_sample_pointcloud(pc, fraction=0.05)
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_pc))
                class_id = int(class_ids[i])
                color = np.array(color_map.get(class_id, [1, 1, 1])) / 255.0
                pcd.paint_uniform_color(color)
                vis.add_geometry(pcd)

            vis.add_geometry(coordinate_frame)
            vis.poll_events()
            vis.update_renderer()

        frame_count += 1

        fps = 1.0 / (time.time() - start_time)
        fps_values.append(fps)

        if len(fps_values) > 10:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)

        cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        height, width, _ = frame.shape
        cv2.resizeWindow(window_name, width, height)
        cv2.imshow(window_name, annotated_frame)

        return cv2.waitKey(1), annotated_frame, point_clouds

    return None, None, []

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO("models/yolo11l-seg.pt").to(device)

    zed1 = sl.Camera()
    zed2 = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = 0.4
    init_params.coordinate_units = sl.UNIT.METER

    T_chess_cam1 = np.array([[0.6577, 0.4860, -0.5756, 0.5863],
                             [-0.7533, 0.4243, -0.5025, 0.7735],
                             [-0.0000, 0.7641, 0.6451, -0.7238],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_chess_cam2 = np.array([[-0.3884, -0.5786, 0.7172, -0.6803],
                             [0.9215, -0.2497, 0.2976, -0.1952],
                             [0.0068, 0.7765, 0.6301, -0.6902],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_robot_chess = np.array([[-1, 0, 0, 0.3580],
                              [0, 1, 0, 0.0300],
                              [0, 0, -1, 0.0060],
                              [0, 0, 0, 1]])

    T_robot_cam1 = np.dot(T_robot_chess, T_chess_cam1)
    T_robot_cam2 = np.dot(T_robot_chess, T_chess_cam2)

    rotation_robot_cam1 = T_robot_cam1[:3, :3]
    rotation_robot_cam2 = T_robot_cam2[:3, :3]

    origin_robot = np.array([0, 0, 0])
    origin_cam1 = T_robot_cam1[:3, 3]
    origin_cam2 = T_robot_cam2[:3, 3]

    distance_cam1 = np.linalg.norm(origin_cam1 - origin_robot)
    distance_cam2 = np.linalg.norm(origin_cam2 - origin_robot)

    print(f"Distance from robot frame to camera frame 1: {distance_cam1:.4f} meters")
    print(f"Distance from robot frame to camera frame 2: {distance_cam2:.4f} meters")

    robot_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cube = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
    cube.translate([-0.025, -0.025, -0.025])
    cube.paint_uniform_color([1, 0, 0])
    camera_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    camera_frame1.transform(T_robot_cam1)
    camera_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    camera_frame2.transform(T_robot_cam2)

    err1 = zed1.open(init_params)
    if err1 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 1: {err1}")
        exit(1)

    err2 = zed2.open(init_params)
    if err2 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 2: {err2}")
        exit(1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Visualization-Pointcloud', width=1280, height=720)
    vis.add_geometry(robot_base_frame)
    vis.add_geometry(camera_frame1)
    vis.add_geometry(camera_frame2)

    key = ''
    point_clouds1 = []
    point_clouds2 = []
    while key != ord('q'):
        key1, frame1, pc1 = process_camera(zed1, model, device, vis, camera_frame1, T_robot_cam1, "YOLO11 Segmentation+Tracking - Camera 1")
        key2, frame2, pc2 = process_camera(zed2, model, device, vis, camera_frame2, T_robot_cam2, "YOLO11 Segmentation+Tracking - Camera 2")

        if key1 == ord('q') or key2 == ord('q'):
            key = ord('q')

        if frame1 is not None:
            cv2.imshow("YOLO11 Segmentation+Tracking - Camera 1", frame1)
        if frame2 is not None:
            cv2.imshow("YOLO11 Segmentation+Tracking - Camera 2", frame2)

        point_clouds1.extend(pc1)
        point_clouds2.extend(pc2)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Capturing static image of the scene")
            class_ids = np.array([0, 39, 41, 62, 64, 66])
            captured_point_clouds = []
            for i, pc in enumerate(point_clouds1 + point_clouds2):
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
                class_id = int(class_ids[i])
                color = np.array(color_map.get(class_id, [1, 1, 1])) / 255.0
                pcd.paint_uniform_color(color)
                captured_point_clouds.append(pcd)
            captured_point_clouds.append(robot_base_frame)
            captured_point_clouds.append(camera_frame1)
            captured_point_clouds.append(camera_frame2)
            captured_point_clouds.append(cube)
            o3d.visualization.draw_geometries(captured_point_clouds)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed1.close()
    zed2.close()
    vis.destroy_window()
    cv2.destroyAllWindows()

    return point_clouds1, point_clouds2

if __name__ == "__main__":
    point_clouds1, point_clouds2 = main()
    print("Point clouds from Camera 1:", point_clouds1)
    print("Point clouds from Camera 2:", point_clouds2)

    # Clear the point clouds
    point_clouds1.clear()
    point_clouds2.clear()