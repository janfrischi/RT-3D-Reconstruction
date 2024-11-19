import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import plotly.graph_objects as go
import random

from ultralytics import YOLO

# This script uses YOLO 11 on two camera instances to detect objects and plot their 3D point clouds in a 3D plot using Plotly

# Define a color map for different classes
color_map = {
    0: [15, 82, 186],  # Person - sapphire
    39: [255, 255, 0],  # Bottle - yellow
    41: [63, 224, 208],  # Cup - turquoise
    62: [255, 0, 255],  # Laptop - magenta
    64: [0, 0, 128],  # Mouse - navy
    66: [255, 0, 0],  # Keyboard - red
    73: [0, 255, 0]   # Book - green
}

# Define class names for the detected objects
class_names = {0: "Person", 39: "Bottle", 41: "Cup", 62: "Laptop", 64: "Mouse", 66: "Keyboard", 73: "Book"}

def erode_mask(mask, iterations=1):
    kernel = np.ones((14, 14), np.uint8)
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

def add_coordinate_frame(fig, origin, rotation, name_prefix):
    axis_length = 0.1
    x_axis = np.dot(rotation, np.array([axis_length, 0, 0])) + origin
    y_axis = np.dot(rotation, np.array([0, axis_length, 0])) + origin
    z_axis = np.dot(rotation, np.array([0, 0, axis_length])) + origin

    fig.add_trace(go.Scatter3d(
        x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]], z=[origin[2], x_axis[2]],
        mode='lines',
        line=dict(color='red', width=5),
        name=f'{name_prefix} X-axis',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]], z=[origin[2], y_axis[2]],
        mode='lines',
        line=dict(color='green', width=5),
        name=f'{name_prefix} Y-axis',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[origin[0], z_axis[0]], y=[origin[1], z_axis[1]], z=[origin[2], z_axis[2]],
        mode='lines',
        line=dict(color='blue', width=5),
        name=f'{name_prefix} Z-axis',
        showlegend=False
    ))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO("yolo11l-seg.pt").to(device)

    zed1 = sl.Camera()
    zed2 = sl.Camera()
    init_params1 = sl.InitParameters()
    init_params2 = sl.InitParameters()
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 60
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_units = sl.UNIT.METER
    init_params2.camera_resolution = sl.RESOLUTION.HD720
    init_params2.camera_fps = 60
    init_params2.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params2.depth_minimum_distance = 0.4
    init_params2.coordinate_units = sl.UNIT.METER

    # Set the serial numbers of the cameras
    sn_cam1 = 33137761
    sn_cam2 = 36829049
    init_params1.set_from_serial_number(sn_cam1)
    init_params2.set_from_serial_number(sn_cam2)

    err1 = zed1.open(init_params1)
    if err1 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 1: {err1}")
        exit(1)

    err2 = zed2.open(init_params2)
    if err2 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 2: {err2}")
        exit(1)

    calibration_params1 = zed1.get_camera_information().camera_configuration.calibration_parameters
    calibration_params2 = zed2.get_camera_information().camera_configuration.calibration_parameters
    fx1, fy1 = calibration_params1.left_cam.fx, calibration_params1.left_cam.fy
    cx1, cy1 = calibration_params1.left_cam.cx, calibration_params1.left_cam.cy
    fx2, fy2 = calibration_params2.left_cam.fx, calibration_params2.left_cam.fy
    cx2, cy2 = calibration_params2.left_cam.cx, calibration_params2.left_cam.cy

    T_chess_cam1 = np.array([[0.6746, 0.4777, -0.5628, 0.5761],
                             [-0.7382, 0.4367, -0.5142, 0.7767],
                             [0.0002, 0.7623, 0.6472, -0.7242],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_chess_cam2 = np.array([[0.5949, -0.5125, 0.6193, -0.6283],
                             [0.8038, 0.3728, -0.4637, 0.5383],
                             [0.0068, 0.7736, 0.6337, -0.6596],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_robot_chess = np.array([[-1.0000, 0.0000, 0.0000, 0.3580],
                              [0.0000, 1.0000, 0.0000, 0.0300],
                              [0.0000, 0.0000, -1.0000, 0.0060],
                              [0.0000, 0.0000, 0.0000, 1.0000]])

    T_robot_cam1 = np.dot(T_robot_chess, T_chess_cam1)
    T_robot_cam2 = np.dot(T_robot_chess, T_chess_cam2)

    rotation_robot_cam1 = T_robot_cam1[:3, :3]
    rotation_robot_cam2 = T_robot_cam2[:3, :3]

    origin_cam1 = T_robot_cam1[:3, 3]
    origin_cam2 = T_robot_cam2[:3, 3]

    print(f"Rotation matrix from robot frame to camera frame 1:\n{rotation_robot_cam1}")
    print(f"Rotation matrix from robot frame to camera frame 2:\n{rotation_robot_cam2}")
    print(f"Translation vector from robot frame to camera frame 1: {origin_cam1}")
    print(f"Translation vector from robot frame to camera frame 2: {origin_cam2}")

    distance_cam1 = np.linalg.norm(origin_cam1 - np.array([0, 0, 0]))
    distance_cam2 = np.linalg.norm(origin_cam2 - np.array([0, 0, 0]))

    print(f"Distance from robot frame to camera frame 1: {distance_cam1:.4f} meters")
    print(f"Distance from robot frame to camera frame 2: {distance_cam2:.4f} meters")

    image1 = sl.Mat()
    depth1 = sl.Mat()
    image2 = sl.Mat()
    depth2 = sl.Mat()
    key = ''
    print("Press 'q' to quit the video feed.")

    cv2.namedWindow("YOLO11 Segmentation+Tracking")

    fps_values = []
    frame_count = 0
    update_frequency = 5

    while key != ord('q'):
        start_time = time.time()

        if zed1.grab() == sl.ERROR_CODE.SUCCESS and zed2.grab() == sl.ERROR_CODE.SUCCESS:
            zed1.retrieve_image(image1, sl.VIEW.LEFT)
            zed2.retrieve_image(image2, sl.VIEW.LEFT)
            depth_retrieval_result1 = zed1.retrieve_measure(depth1, sl.MEASURE.DEPTH)
            depth_retrieval_result2 = zed2.retrieve_measure(depth2, sl.MEASURE.DEPTH)
            if depth_retrieval_result1 != sl.ERROR_CODE.SUCCESS or depth_retrieval_result2 != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result1}, {depth_retrieval_result2}")
                continue

            frame1 = image1.get_data()
            frame2 = image2.get_data()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)

            results1 = model.track(
                source=frame1,
                imgsz=640,
                max_det=20,
                classes=[0, 39, 41, 62, 64, 66, 73],
                half=True,
                persist=True,
                retina_masks=True,
                conf=0.2,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            results2 = model.track(
                source=frame2,
                imgsz=640,
                max_det=20,
                classes=[0, 39, 41, 62, 64, 66, 73],
                half=True,
                persist=True,
                retina_masks=True,
                conf=0.2,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            zed_depth_np1 = depth1.get_data()
            zed_depth_np2 = depth2.get_data()
            if zed_depth_np1 is None or zed_depth_np2 is None:
                print("Error: Depth map is empty")
                continue

            annotated_frame1 = results1[0].plot(line_width=2, font_size=18)
            annotated_frame2 = results2[0].plot(line_width=2, font_size=18)

            masks1 = results1[0].masks
            masks2 = results2[0].masks
            class_ids1 = results1[0].boxes.cls.cpu().numpy()
            class_ids2 = results2[0].boxes.cls.cpu().numpy()
            point_clouds1 = []
            point_clouds2 = []

            for i, mask in enumerate(masks1.data):
                mask = mask.cpu().numpy()
                mask = erode_mask(mask, iterations=1)
                mask_indices = np.argwhere(mask > 0)
                depth_map = torch.from_numpy(zed_depth_np1).to(device)

                with torch.cuda.amp.autocast():
                    points_3d = get_3d_points_torch(mask_indices, depth_map, cx1, cy1, fx1, fy1)

                if points_3d.size(0) > 0:
                    point_cloud_cam1 = points_3d.cpu().numpy()
                    point_cloud_np_transformed = np.dot(rotation_robot_cam1, point_cloud_cam1.T).T + origin_cam1
                    point_clouds1.append((point_cloud_np_transformed, int(class_ids1[i])))
                    print(f"Class ID: {class_ids1[i]} ({class_names[class_ids1[i]]}) in Camera Frame 1")

            for i, mask in enumerate(masks2.data):
                mask = mask.cpu().numpy()
                mask = erode_mask(mask, iterations=1)
                mask_indices = np.argwhere(mask > 0)
                depth_map = torch.from_numpy(zed_depth_np2).to(device)

                with torch.cuda.amp.autocast():
                    points_3d = get_3d_points_torch(mask_indices, depth_map, cx2, cy2, fx2, fy2)

                if points_3d.size(0) > 0:
                    point_cloud_cam2 = points_3d.cpu().numpy()
                    point_cloud_np_transformed = np.dot(rotation_robot_cam2, point_cloud_cam2.T).T + origin_cam2
                    point_clouds2.append((point_cloud_np_transformed, int(class_ids2[i])))
                    print(f"Class ID: {class_ids2[i]} ({class_names[class_ids2[i]]}) in Camera Frame 2")

            if (point_clouds1 or point_clouds2) and frame_count % update_frequency == 0:
                fig = go.Figure()

                # Add robot base coordinate frame
                fig.add_trace(go.Scatter3d(
                    x=[0, 0.1], y=[0, 0], z=[0, 0],
                    mode='lines',
                    line=dict(color='red', width=5),
                    name='Robot X-axis'
                ))
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[0, 0.1], z=[0, 0],
                    mode='lines',
                    line=dict(color='green', width=5),
                    name='Robot Y-axis'
                ))
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[0, 0], z=[0, 0.1],
                    mode='lines',
                    line=dict(color='blue', width=5),
                    name='Robot Z-axis'
                ))

                # Add the camera positions
                fig.add_trace(go.Scatter3d(
                    x=[origin_cam1[0]], y=[origin_cam1[1]], z=[origin_cam1[2]],
                    mode='markers+text',
                    marker=dict(size=5, color='red'),
                    textposition= "top center",
                    name='Camera 1',
                    showlegend=False
                ))

                fig.add_trace(go.Scatter3d(
                    x=[origin_cam2[0]], y=[origin_cam2[1]], z=[origin_cam2[2]],
                    mode='markers+text',
                    marker=dict(size=5, color='blue'),
                    textposition= "top center",
                    name='Camera 2',
                    showlegend=False
                ))

                # Add camera coordinate frames
                add_coordinate_frame(fig, origin_cam1, rotation_robot_cam1, 'Camera 1')
                add_coordinate_frame(fig, origin_cam2, rotation_robot_cam2, 'Camera 2')

                for pc, class_id in point_clouds1 + point_clouds2:
                    sampled_pc = random_sample_pointcloud(pc, fraction=0.5)
                    color = np.array(color_map.get(class_id, [255, 255, 255])) / 255.0
                    fig.add_trace(go.Scatter3d(
                        x=sampled_pc[:, 0], y=sampled_pc[:, 1], z=sampled_pc[:, 2],
                        mode='markers',
                        marker=dict(size=2, color=f'rgb({color[0]*255},{color[1]*255},{color[2]*255})'),
                        name=class_names[class_id]
                    ))

                fig.update_layout(
                    title=dict(
                        text="3D Point Cloud",
                        x=0.5,
                        xanchor='center'
                    ),
                    legend = dict(
                        title="Classes",
                        font=dict(
                            family="Courier",
                            size=12,
                            color="black"
                        ),
                        bgcolor="LightSteelBlue",
                        bordercolor="Black",
                        borderwidth=2
                    ),
                    scene_camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    scene=dict(
                        xaxis=dict(range=[-0.5, 1]),
                        yaxis=dict(range=[-0.75,1]),
                        zaxis=dict(range=[0, 1])
                    )
                )

                fig.show()

            frame_count += 1

            fps = 1.0 / (time.time() - start_time)
            fps_values.append(fps)

            if len(fps_values) > 10:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            if annotated_frame1 is not None:
                cv2.putText(annotated_frame1, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                height, width, _ = frame1.shape
                cv2.resizeWindow("YOLO11 Segmentation+Tracking", width // 2, height // 2)
                cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame1)

            if annotated_frame2 is not None:
                cv2.putText(annotated_frame2, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                height, width, _ = frame2.shape
                cv2.resizeWindow("YOLO11 Segmentation+Tracking", width // 2, height // 2)
                cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame2)

            # Concatenate the frames horizontally
            combined_frame = cv2.hconcat([annotated_frame1, annotated_frame2])
            cv2.imshow("YOLO11 Segmentation+Tracking", combined_frame)

            key = cv2.waitKey(1)

    zed1.close()
    zed2.close()

if __name__ == "__main__":
    main()