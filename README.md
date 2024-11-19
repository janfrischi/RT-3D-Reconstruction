# Vision-Based Object Detection and Point Cloud Processing

This project implements a vision-based pipeline for object detection, tracking, and point cloud processing using YOLO for segmentation and the ZED stereo camera system for depth sensing. The main objective of this project is to detect and track specific objects in real-time, generate 3D point clouds for these objects, and process the point clouds to remove noise and improve data quality.

## Features

- **Real-Time Object Detection and Tracking**: Uses the YOLO model for object detection and tracking in each camera frame.
- **Depth-Based Point Cloud Generation**: Generates 3D point clouds of detected objects by using depth information from ZED stereo cameras.
- **Point Cloud Processing**: Includes functionalities to filter outliers and fuse point clouds from multiple camera perspectives.
- **Modular Architecture**: The project is organized into multiple classes to manage cameras, object detection models, and point cloud processing.

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `opencv-python`
  - `torch` (with CUDA support if using a GPU)
  - `pyzed` (for interfacing with ZED cameras)
  - `open3d` (for point cloud processing)
  - `ultralytics` (for YOLO model)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/janfrischi/RT-3D-Reconstruction.git
   cd RT_3D_Reconstruction
   ```

2. **Install required libraries**:
   ```bash
   pip install numpy opencv-python torch pyzed open3d ultralytics
   ```

3. **Download the YOLO model**: Place the model file `yolo11l-seg.pt` in the `models/` directory.

## File Structure

- `main.py`: Initializes the YOLO model, ZED cameras, and transformation matrices, then runs the main application.
- `object_detection_and_point_cloud_processing.py`: Contains classes for managing cameras, YOLO model tracking, point cloud processing, and the main application loop.

## Usage

To run the project, use the following command:

```bash
python main.py
```

## Configuration

### Camera Configuration

- **Camera Serial Numbers**: The serial numbers of the ZED cameras (`sn_cam1` and `sn_cam2`) need to be specified in `main.py` to correctly initialize the cameras.
- **Transformation Matrices**:
  - `T_chess_cam1`, `T_chess_cam2`: Transformations from the chessboard to each camera's frame.
  - `T_robot_chess`: Transformation from the robot's base frame to the chessboard.
  
- **Camera Parameters**:
  - `resolution`: Camera resolution (e.g., "HD720").
  - `fps`: Frames per second.
  - `depth_mode`: Depth mode (e.g., "NEURAL").
  - `min_distance`: Minimum distance for depth sensing.
  - `units`: Depth measurement units (e.g., "METER").

### Object Detection and Tracking Configuration

The `YOLOModel` class initializes the YOLO model for object detection and tracking with the following parameters:
- `imgsz`: Image size for the model.
- `classes`: List of class IDs to detect and track.
- `conf`: Confidence threshold for detections.

### Class Definitions

#### `CameraManager`

Manages the initialization, retrieval, and closing of ZED stereo cameras.
- **Methods**:
  - `retrieve_images_and_depths()`: Captures images and depth maps from both cameras.
  - `close()`: Closes both camera connections.

#### `YOLOModel`

Encapsulates the YOLO object detection and tracking functionality.
- **Methods**:
  - `track()`: Runs the YOLO model to detect and track objects in a given frame.

#### `PointCloudProcessor`

Handles point cloud operations, including mask erosion, point cloud generation, downsampling, outlier filtering, and point cloud fusion.
- **Static Methods**:
  - `erode_mask()`: Erodes the mask to reduce noise.
  - `convert_mask_to_3d_points()`: Converts mask indices to 3D points using depth map values.
  - `downsample_point_cloud()`: Downsamples a point cloud to reduce the number of points.
  - `filter_outliers_sor()`: Filters outliers in the point cloud using statistical outlier removal.
  - `fuse_point_clouds_centroid()`: Fuses point clouds from two cameras based on centroid distance.

#### `MainApp`

Manages the main workflow of the application, coordinating camera retrieval, object detection, point cloud generation, and display.
- **Methods**:
  - `run()`: Main loop to retrieve images, detect objects, generate point clouds, process data, and display annotated frames.
  - `_process_masks()`: Processes segmentation masks to generate and transform 3D points for each detected object.
  - `_display_frames()`: Displays annotated frames with object detections and frame rate.

### Sample Configuration in `main.py`

```python
model_path = "models/yolo11l-seg.pt"
sn_cam1 = 33137761
sn_cam2 = 36829049

# Color map and class names
color_map = {
    0: [15, 82, 186],
    39: [255, 255, 0],
    41: [63, 224, 208],
    62: [255, 0, 255],
    64: [0, 0, 128],
    66: [255, 0, 0],
    73: [0, 255, 0]
}
class_names = {
    0: "Person",
    39: "Bottle",
    41: "Cup",
    62: "Laptop",
    64: "Mouse",
    66: "Keyboard",
    73: "Book"
}
```

### Transformations

Transformation matrices are used to calibrate the position of cameras and objects:
- `T_chess_cam1`, `T_chess_cam2`: Transformations from chessboard to camera frame.
- `T_robot_chess`: Transformation from robot frame to chessboard.

### Example Usage in `main.py`

```python
app = MainApp(model_path, sn_cam1, sn_cam2, color_map, class_names, T_chess_cam1, T_chess_cam2, T_robot_chess, init_params1, init_params2)
for fused_pc in app.run():
    # Process each fused point cloud as needed
    pass
```

## Key Components and Functionalities

1. **Object Detection and Tracking**:
   - YOLO model detects specific objects in frames from both cameras.
   
2. **Depth Map Conversion to 3D Points**:
   - The depth information from the ZED stereo cameras is used to convert 2D mask pixels into 3D coordinates.
   
3. **Point Cloud Processing**:
   - After generating point clouds from detected objects, the points are downsampled and outliers are removed.
   - Point clouds from both cameras are fused based on centroid distance to generate a comprehensive 3D representation.

4. **Display and Visualization**:
   - Annotated frames with bounding boxes and object labels are displayed, showing the objects detected in real-time.
   - The frame rate (FPS) is displayed on each frame for performance monitoring.

## Notes

- The code is designed to run in real-time, with optimizations for using CUDA if available.
- The application runs in a loop until the `q` key is pressed.

## Troubleshooting

- **Camera Initialization Error**: Ensure that the correct serial numbers are provided for the ZED cameras.
- **CUDA Errors**: Make sure that your GPU drivers and CUDA toolkit are properly installed if using CUDA.
- **Model Not Found**: Verify that the YOLO model file is in the specified path (`models/yolo11l-seg.pt`).