# **Real-Time Vision Pipeline for 3D Object Detection, Segmentation, and Reconstruction**

This repository implements a modular, real-time vision pipeline for object detection, segmentation, and 3D reconstruction using **YOLO11** and a set of **ZED Stereo Cameras**. The project leverages GPU acceleration and advanced computer vision techniques to create a scalable solution for collaborative robotics, workspace analysis, and dynamic object tracking.

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
   - [Available Scripts](#available-scripts)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Core Functions](#core-functions)
6. [Notes](#notes)
7. [Troubleshooting](#troubleshooting)
   - [Common Issues and Solutions](#common-issues-and-solutions)

## **Features and Functionalities**

### **1. Real-Time Object Detection, Segmentation, and Tracking**
- Utilizes **YOLO11** for detecting and segmenting objects of interest (e.g., bottles, cups, laptops) in frames from stereo cameras.
- Tracks objects and their IDs across frames to ensure consistency. The pipeline supports two tracking algorithms:
  - **ByteTrack**: A lightweight, real-time object tracker that employs a simple yet effective online and real-time tracking algorithm.
  - **DeepSORT**: A deep learning-based object tracker that combines appearance features and motion information for robust tracking.

### **2. 3D Reconstruction from Depth Maps**
- Converts 2D segmentation masks into 3D point clouds using ZED stereo camera depth maps and camera intrinsics.
- Improves accuracy and efficiency through depth-to-3D mapping.

### **3. Advanced Point Cloud Processing**
- **Down sampling**: Reduces point cloud density using voxel grid filtering.
- **Outlier Removal**: Removes statistical outliers to enhance point cloud quality.
- **Cropping**: Crops point clouds to specific regions of interest.
- **Transformation**: Transforms point clouds to a common reference frame for fusion.
- **Fusion**: Combines point clouds from multiple cameras based on centroid distances to create a unified 3D representation.
- **Subtraction**: Removes the reconstructed object point cloud from the workspace.

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
- Logs real-time FPS and timing metrics for each pipeline stage.
- Enables performance analysis and optimization for specific setups.
## **Getting Started**

### **Prerequisites**
1. **Hardware**:
   - NVIDIA RTX 4090 or equivalent GPU.
   - ZED Stereo Cameras.

2. **Software**:
   - Python >= 3.8
   - CUDA-enabled GPU drivers
   - **ZED SDK and Python API**:
     - Install the [ZED SDK](https://www.stereolabs.com/zed-sdk/) and its Python API, which are required for camera access and depth map generation.
     - Detailed installation instructions can be found on the [Stereolabs documentation page](https://www.stereolabs.com/docs/).

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/janfrischi/RT-3D-Reconstruction.git
   cd your-directory
   ```

2. Install required dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv11 model:
   - Ensure the YOLOv11 model (`yolo11x-seg.pt`) is stored in the `models/` directory.
   - You can download it from the [Ultralytics repository](https://github.com/ultralytics).

4. Connect ZED cameras and set up the environment.
   - Ensure the ZED cameras are connected and recognized by the system. Use USB 3.0 10GB/s ports for optimal performance.
   - Set up the ZED SDK and Python API for camera access.
   - Make sure the camera serial numbers are correctly set in the code.

## **Usage**

### **Available Scripts**
This repository provides multiple scripts tailored to specific hardware configurations and use cases:

1. **`2cams.py`**:
   - Runs the vision pipeline with two ZED cameras.
   - Streamlined version that imports all necessary functions from the vision_pipeline_utils.py file. This script is preconfigured with default parameters, such as voxel_size=0.005, making it ready to use out of the box.

2. **`2cams_mask_cpu.py`**:
   - Optimized for running the vision pipeline with CPU-based mask processing.
   - Ideal when GPU resources are limited.

3. **`2cams_mask_gpu.py`**:
   - Optimized for GPU-based mask processing, leveraging CUDA for enhanced performance.
   - Recommended for real-time applications where high-speed processing is required.

4. **`visualizer_fps.py`**: 
   - Uses the fps_log.csv file to visualize the frame rate over time.

5. **`visualizer_performance.py`**:
   - Uses the timings.csv file to visualize the performance metrics over time.r3oy

## **Performance Benchmarking**

The pipeline tracks the following metrics for each frame:
- **Frame Retrieval Time**: Time taken to retrieve frames from cameras.
- **Depth Retrieval Time**: Time for generating depth maps.
- **YOLO Inference Time**: Time for object detection and segmentation.
- **Point Cloud Processing Time**: Time to process and fuse point clouds.
- **Overall Loop Time**: Total time per frame.

All timings are logged in `timings.csv` for analysis.

---

## **Core Functions**

1. **`convert_mask_to_3d_points()`**: Converts 2D segmentation masks into 3D coordinates using depth maps.
2. **`downsample_point_cloud_gpu()`**: Downsamples point clouds using voxel grid filtering on the GPU.
3. **`crop_point_cloud_gpu()`**: Crops point clouds to specified boundaries.
4. **`fuse_point_clouds_centroid()`**: Fuses point clouds from two cameras based on centroid distance.
5. **`subtract_point_clouds_gpu()`**: Subtracts object point clouds from the workspace.

For a detailed description of functions, see the **[Core Functions](#core-functions)** section.

---

## **Notes**
- The code is designed to run in real-time, with optimizations for using CUDA if available.
- The application runs in a loop until the `q` key is pressed.

---

## **Troubleshooting**

### **Common Issues and Solutions**:

1. **Camera Initialization Error**:
   - Ensure the correct serial numbers are provided for your ZED cameras in the code.

2. **CUDA Errors**:
   - Verify your GPU drivers and CUDA toolkit are properly installed.
   - Ensure PyTorch is installed with CUDA support.

3. **Model Not Found**:
   - Ensure the YOLO model file is stored at the specified path (`models/yolo11x-seg.pt`).
