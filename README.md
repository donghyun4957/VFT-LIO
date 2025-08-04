# VFT-LIO: Visual Feature Tracking for Robust LiDAR Inertial Odometry Under Repetitive Patterns

This repository contains the implementation of **VFT-LIO**, a front-end SLAM module that enhances odometry estimation in repetitive environments such as tunnels and bridges. The method leverages **SuperPoint**, a deep learning-based visual feature extraction method, on LiDAR intensity images and refines pose estimation using ICP.

## Features
- Imaging LiDAR
- Visual feature extraction from LiDAR intensity images
- Conversion of extracted visual feature to corresponding point cloud
- ICP pose refinement

## System Pipeline
The figure below illustrates the full pipeline of VFT-LIO, including feature extraction, point cloud association, and ICP-based refinement.

<p align="center">
  <img src="doc/Framework.png" width="700" height="200"/>
</p>

## Example Results

We provide example results using our proposed method in repetitive environments such as tunnels and bridges.
As shown below, the system demonstrates robust odometry estimation even in highly repetitive environments, such as tunnels and bridges.

<p align="center">
  <img src="doc/tunnel01.png" width="350" height="300"/>
  <img src="doc/tunnel02.png" width="350" height="300"/>
</p>

<p align="center">
  <img src="doc/bridge01.png" width="350" height="300"/>
  <img src="doc/bridge02.png" width="350" height="300"/>
</p>

## Installation

```
cd ~/catkin_ws/src
git clone https://github.com/donghyun4957/VFT-LIO.git
cd ..
catkin_make
```

## How to Use

Pre-trained SuperPoint weights are provided at  
`pretrained/superPointNet_checkpoint.pth.tar`

To run the system with the pre-trained weights, simply execute the following commands:

```bash
roslaunch liosam run.launch
rosrun lio_sam visual_tracking.py

💡 **Note:** A sample dataset recorded in a repetitive environment (e.g., tunnel or bridge)  
will be provided.
```

## Modification Notice

This repository is a modified version of [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM),  
with a front-end changed for robust odometry in repetitive environments.  
SuperPoint is integrated for feature extraction from LiDAR intensity images.

## Citations

If you use this code, please cite the following works:

### 🔹 LIO-SAM

```bibtex
@inproceedings{liosam2020shan,
  title={LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping},
  author={Shan, Tixiao and Englot, Brendan and Meyers, Drew and Wang, Wei and Ratti, Carlo and Rus, Daniela},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
  organization={IEEE}
}
```

### 🔹 SuperPoint
```bibtex
@inproceedings{detone2018superpoint,
  title={SuperPoint: Self-Supervised Interest Point Detection and Description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={CVPR Workshops},
  year={2018}
}
```