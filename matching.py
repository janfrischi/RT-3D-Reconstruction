import numpy as np
import open3d as o3d

# Generate a sample point cloud (e.g., points on a sphere)
num_points = 1000
radius = 1.0
points = []
for _ in range(num_points):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points.append([x, y, z])

# Convert the list of points to a NumPy array
points = np.array(points)

# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Define the voxel size
voxel_size = 0.1

# Perform voxelization
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# Visualize the voxel grid
o3d.visualization.draw_geometries([voxel_grid])
