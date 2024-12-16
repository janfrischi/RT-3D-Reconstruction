from vision_pipeline_utils import *

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
class_names = {0: "Person",
               39: "Bottle",
               41: "Cup",
               62: "Laptop",
               64: "Mouse",
               66: "Keyboard",
               73: "Book"}

# Dictionary to store cumulative timings for Benchmarking
timings = {
    "Frame Retrieval": [],
    "Depth Retrieval": [],
    "Point Cloud Processing": [],
    "YOLO11 Inference": [],
    "Mask Processing": [],
    "Point Cloud Fusion": [],
    "Subtraction": [],
    "Total Time per Iteration": []
}

def main():

    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pre-trained YOLOv11 model and move it to the device
    model = YOLO("models/yolo11x-seg.pt").to(device)

    # Initialize the CSV file to store the results
    fps_log_file = "../fps_log.csv"

    # Create or overwrite the CSV file and write the header
    with open(fps_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "FPS"])

    # Initialize the ZED camera objects
    zed1 = sl.Camera()
    zed2 = sl.Camera()

    # Set the serial numbers of the cameras
    sn_cam1 = 33137761
    sn_cam2 = 36829049

    # Set the initialization parameters for camera 1
    init_params1 = sl.InitParameters()
    init_params1.set_from_serial_number(sn_cam1)
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 30
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_units = sl.UNIT.METER

    # Set the initialization parameters for camera 2
    init_params2 = sl.InitParameters()
    init_params2.set_from_serial_number(sn_cam2)
    init_params2.camera_resolution = sl.RESOLUTION.HD720
    init_params2.camera_fps = 30
    init_params2.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params2.depth_minimum_distance = 0.4
    init_params2.coordinate_units = sl.UNIT.METER

    # Check if the cameras were successfully opened
    err1 = zed1.open(init_params1)
    if err1 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 1: {err1}")
        exit(1)

    err2 = zed2.open(init_params2)
    if err2 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 2: {err2}")
        exit(1)

    # Get the camera calibration parameters
    calibration_params1 = zed1.get_camera_information().camera_configuration.calibration_parameters
    fx1, fy1 = calibration_params1.left_cam.fx, calibration_params1.left_cam.fy
    cx1, cy1 = calibration_params1.left_cam.cx, calibration_params1.left_cam.cy

    calibration_params2 = zed2.get_camera_information().camera_configuration.calibration_parameters
    fx2, fy2 = calibration_params2.left_cam.fx, calibration_params2.left_cam.fy
    cx2, cy2 = calibration_params2.left_cam.cx, calibration_params2.left_cam.cy

    # Define the transformation matrices from the chessboard to the camera frames

    T_chess_cam1 = np.array([[0.8401, 0.3007, -0.4515, 0.5914],
                             [-0.5424, 0.4647, -0.6999, 1.1329],
                             [-0.0006, 0.8329, 0.5535, -0.7193],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_chess_cam2 = np.array([[0.4608, -0.4105, 0.7869, -0.8716],
                             [0.8874, 0.2010, -0.4148, 0.8318],
                             [0.0121, 0.8895, 0.4569, -0.7200],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    # This transformation matrix is given by the geometry of the mount and the chessboard
    T_robot_chess = np.array([[-1.0000, 0.0000, 0.0000, 0.3580],
                              [0.0000, 1.0000, 0.0000, 0.0300],
                              [0.0000, 0.0000, -1.0000, 0.0060],
                              [0.0000, 0.0000, 0.0000, 1.0000]])

    # Calculate the transformation matrices from the robot frame to the camera frames
    T_robot_cam1 = np.dot(T_robot_chess, T_chess_cam1)
    T_robot_cam2 = np.dot(T_robot_chess, T_chess_cam2)

    # Extract the rotation matrices and translation vectors from the transformation matrices
    rotation_robot_cam1 = T_robot_cam1[:3, :3]
    rotation_robot_cam2 = T_robot_cam2[:3, :3]
    origin_cam1 = T_robot_cam1[:3, 3]
    origin_cam2 = T_robot_cam2[:3, 3]

    # Convert them to tensors
    rotation1_torch = torch.tensor(rotation_robot_cam1, dtype=torch.float32, device=device)
    origin1_torch = torch.tensor(origin_cam1, dtype=torch.float32, device=device)
    rotation2_torch = torch.tensor(rotation_robot_cam2, dtype=torch.float32, device=device)
    origin2_torch = torch.tensor(origin_cam2, dtype=torch.float32, device=device)

    # Calculate the distance from the robot frame to the camera frames
    distance_cam1 = np.linalg.norm(origin_cam1 - np.array([0, 0, 0]))
    distance_cam2 = np.linalg.norm(origin_cam2 - np.array([0, 0, 0]))

    print(f"Distance from robot frame to camera frame 1: {distance_cam1:.4f} meters")
    print(f"Distance from robot frame to camera frame 2: {distance_cam2:.4f} meters")

    # Set the resolution for the point clouds
    resolution = sl.Resolution(640, 360)

    # Initialize the image and depth map objects for both cameras
    image1 = sl.Mat()
    depth1 = sl.Mat()
    image2 = sl.Mat()
    depth2 = sl.Mat()

    # Create point cloud objects to hold data of the workspace
    point_cloud1_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud2_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # Initialize the key variable to check for the 'q' key press
    key = ''

    # Create a window to display the output
    cv2.namedWindow("YOLO11 Segmentation+Tracking")
    fps_values = []
    frame_count = 0

    # Initialize lists to store the point clouds of the reconstructed objects from both cameras, each point cloud is a tuple of the point cloud and the class ID
    point_clouds_camera1 = []
    point_clouds_camera2 = []

    # Main loop to capture and process images from both cameras
    while key != ord('q'):
        start_time = time.time()
        # Step 1: Frame retrieval
        if zed1.grab() == sl.ERROR_CODE.SUCCESS and zed2.grab() == sl.ERROR_CODE.SUCCESS:
            frame1, frame2 = retrieve_frames(zed1, zed2, image1, image2, timings)

            # Step 2: Depth Map retrieval
            depth_retrieval_result1, depth_retrieval_result2, zed_depth_np1, zed_depth_np2 = retrieve_depth_maps(zed1, zed2, depth1, depth2, timings)
            # Check if the depth maps were successfully retrieved
            if depth_retrieval_result1 != sl.ERROR_CODE.SUCCESS or depth_retrieval_result2 != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result1}, {depth_retrieval_result2}")
                continue

            # Step 3: Point Cloud retrieval and processing
            fused_point_cloud_ws = process_point_clouds(zed1, zed2, point_cloud1_ws, point_cloud2_ws, resolution,
                                                        rotation1_torch, origin1_torch, rotation2_torch, origin2_torch,
                                                        device, timings)


            # Step 4: YOLOv11 Inference
            annotated_frame1, annotated_frame2, masks1, masks2, class_ids1, class_ids2 = perform_yolo_inference(model, frame1, frame2, device, timings)

            # Step 5: Mask Processing
            start_time_processing_masks = time.time()
            process_masks(masks1, class_ids1, zed_depth_np1, cx1, cy1, fx1, fy1, rotation1_torch, origin1_torch, device,
                          point_clouds_camera1)
            process_masks(masks2, class_ids2, zed_depth_np2, cx2, cy2, fx2, fy2, rotation2_torch, origin2_torch, device,
                          point_clouds_camera2)
            end_time_processing_masks = time.time()
            timings["Mask Processing"].append(end_time_processing_masks - start_time_processing_masks)

            # Step 6: Point Cloud Fusion
            pcs1, pcs2, fused_pc_objects_concatenated = fuse_point_clouds(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.3, timings=timings)

            # Step 7: Point Cloud Subtraction
            workspace_pc_subtracted = subtract_point_clouds(fused_point_cloud_ws, fused_pc_objects_concatenated, distance_threshold=0.3, timings=timings)

            # Call the process_iteration function to display the results
            frame_count, fps_values = process_iteration_end(start_time, timings, frame_count, fps_values, fps_log_file,
                                                            annotated_frame1, annotated_frame2, point_clouds_camera1,
                                                            point_clouds_camera2, pcs1, pcs2)

            # Wait for the 'q' key press to exit the loop
            key = cv2.waitKey(1)

    zed1.close()
    zed2.close()


if __name__ == "__main__":
    main()