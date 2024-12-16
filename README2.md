# **Real-Time Vision Pipeline for 3D Object Detection, Segmentation, and Reconstruction**

This repository implements a modular, real-time vision pipeline for object detection, segmentation, and 3D reconstruction using **YOLOv11** and **ZED Stereo Cameras**. The project leverages GPU acceleration and advanced computer vision techniques to create a scalable solution for collaborative robotics, workspace analysis, and dynamic object tracking.

## **Table of Contents**

1. [Features and Functionalities](#features-and-functionalities)
   - [Real-Time Object Detection, Segmentation, and Tracking](#1-real-time-object-detection-segmentation-and-tracking)
   - [3D Reconstruction from Depth Maps](#2-3d-reconstruction-from-depth-maps)
   - [Advanced Point Cloud Processing](#3-advanced-point-cloud-processing)
   - [Interactive Visualization](#4-interactive-visualization)
   - [Modular and Extensible Design](#5-modular-and-extensible-design)
   - [Performance Monitoring](#6-performance-monitoring)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Usage](#usage)
   - [Running the Pipeline](#running-the-pipeline)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Core Functions](#core-functions)
6. [Notes](#notes)
7. [Troubleshooting](#troubleshooting)
   - [Common Issues and Solutions](#common-issues-and-solutions)


## **Features and Functionalities**

### **1. Real-Time Object Detection, Segmentation, and Tracking**
- Utilizes **YOLOv11** for detecting and segmenting objects of interest (e.g., bottles, cups, laptops) in frames from stereo cameras.
- Tracks object IDs across frames to ensure consistency.

### **2. 3D Reconstruction from Depth Maps**
- Converts 2D segmentation masks into 3D point clouds using ZED stereo camera depth maps and camera intrinsics.
- Improves accuracy and efficiency through depth-to-3D mapping.

### **3. Advanced Point Cloud Processing**
- **Downsampling**: Reduces point cloud density using voxel grid filtering.
- **Outlier Removal**: Removes statistical outliers to enhance point cloud quality.
- **Fusion**: Combines point clouds from multiple cameras based on centroid distances to create a unified 3D representation.

### **4. Interactive Visualization**
- Displays annotated video frames with bounding boxes, segmentation masks, and object labels.
- Visualizes 3D point clouds of detected objects and the workspace using Open3D.
- Displays real-time frame rate (FPS) for performance monitoring.

### **5. Modular and Extensible Design**
- Organized into reusable components for:
  - Camera management
  - Object detection and segmentation
  - Point cloud generation and fusion
- Easily extendable for future functionalities.

### **6. Performance Monitoring**
- Logs real-time FPS and timing metrics (e.g., frame retrieval, YOLO inference, point cloud processing) to CSV files for benchmarking.

## **Getting Started**

### **Prerequisites**
1. **Hardware**:
   - NVIDIA RTX 4090 or equivalent GPU.
   - ZED Stereo Cameras.

2. **Software**:
   - Python >= 3.8
   - CUDA-enabled GPU drivers.

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/vision-pipeline.git
   cd vision-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install numpy pyzed-python opencv-python torch ultralytics open3d
   ```

3. Download the YOLOv11 model:
   - Ensure the YOLOv11 model (`yolo11x-seg.pt`) is stored in the `models/` directory.
   - You can download it from the [Ultralytics repository](https://github.com/ultralytics).

4. Connect ZED cameras and set up the environment.

## **Usage**

### **Running the Pipeline**
Run the main script to start the real-time pipeline:
```bash
python main.py
```

## **Performance Benchmarking**

The pipeline tracks the following metrics for each frame:
- **Frame Retrieval Time**: Time taken to retrieve frames from cameras.
- **Depth Retrieval Time**: Time for generating depth maps.
- **YOLO Inference Time**: Time for object detection and segmentation.
- **Point Cloud Processing Time**: Time to process and fuse point clouds.
- **Overall Loop Time**: Total time per frame.

All timings are logged in `timings.csv` for analysis.

## **Core Functions**

1. **`convert_mask_to_3d_points()`**: Converts 2D segmentation masks into 3D coordinates using depth maps.
2. **`downsample_point_cloud_gpu()`**: Downsamples point clouds using voxel grid filtering on the GPU.
3. **`crop_point_cloud_gpu()`**: Crops point clouds to specified boundaries.
4. **`fuse_point_clouds_centroid()`**: Fuses point clouds from two cameras based on centroid distance.
5. **`subtract_point_clouds_gpu()`**: Subtracts object point clouds from the workspace.

For a detailed description of functions, see the **[Core Functions](#core-functions)** section.


## **Notes**
- The code is designed to run in real-time, with optimizations for using CUDA if available.
- The application runs in a loop until the `q` key is pressed.


## **Troubleshooting**

### **Common Issues and Solutions**:

1. **Camera Initialization Error**:
   - Ensure the correct serial numbers are provided for your ZED cameras in the code.

2. **CUDA Errors**:
   - Verify your GPU drivers and CUDA toolkit are properly installed.
   - Ensure PyTorch is installed with CUDA support.

3. **Model Not Found**:
   - Ensure the YOLO model file is stored at the specified path (`models/yolo11x-seg.pt`).
