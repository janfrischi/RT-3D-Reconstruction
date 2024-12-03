if points_3d_cam1.size(0) > 0:
    # Transform points using torch.mm for GPU acceleration
    rotation_robot_cam1_torch = torch.tensor(rotation_robot_cam1, dtype=torch.float32,
                                             device=points_3d_cam1.device)
    origin_cam1_torch = torch.tensor(origin_cam1, dtype=torch.float32, device=points_3d_cam1.device)

    # Perform transformation on the GPU
    point_cloud_cam1_transformed = torch.mm(points_3d_cam1,
                                            rotation_robot_cam1_torch.T) + origin_cam1_torch

    point_cloud_cam1_downsampled = downsample_point_cloud_gpu(point_cloud_cam1_transformed, voxel_size=0.01)

    # Move transformed points to CPU for further processing
    point_cloud_cam1_downsampled_cpu = point_cloud_cam1_downsampled.cpu().numpy()

    # Add the downsampled point cloud and class ID to this camera's point cloud list
    point_clouds_camera1.append((point_cloud_cam1_downsampled_cpu, int(class_ids1[i])))
    print(f"Class ID: {class_ids1[i]} ({class_names[class_ids1[i]]}) in Camera Frame 1")