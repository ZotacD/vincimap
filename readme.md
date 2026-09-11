# VinciMap

**Metric 3D reconstruction from video using photogrammetry and 3D Gaussian Splatting**

VinciMap is a five-student engineering project developed at ESILV Nantes as part of the P2IP program.

The project explores a low-cost approach to creating measurable 3D environments from conventional video acquisition. The current software pipeline converts video frames into a COLMAP reconstruction and trains a 3D Gaussian Splatting model for visualization and further metric processing.

 Best Project Execution Award – ESILV Nantes P2IP  
 five-student engineering team   
 
## Project Objectives

The main objectives of VinciMap were to:

- reconstruct a real environment from conventional video acquisition;
- generate a detailed 3D representation using photogrammetry and 3D Gaussian Splatting;
- recover a real-world metric scale;
- investigate the integration of distance-sensor data to improve metric consistency;
- provide a simple and accessible alternative to more expensive 3D mapping solutions.

The objective was not only to generate visually realistic 3D scenes, but also
to preserve sufficient metric consistency to perform measurements inside the
reconstructed environment.

## Current Status

### Available in the public repository

- Video frame extraction
- COLMAP sparse reconstruction
- Sparse sub-model merging
- COLMAP model conversion
- PLY point-cloud generation
- 3D Gaussian Splatting training
- 3D visualization

### Final prototype developments not yet fully reflected in this branch

- Distance-sensor integration
- Metric scale calibration
- LiDAR-assisted measurement refinement
- Final measurement-validation workflow

## Processing Pipeline

Video acquisition
        ↓
Frame extraction with FFmpeg
        ↓
COLMAP sparse reconstruction
        ↓
Sparse sub-model merging
        ↓
COLMAP model conversion
        ↓
Point-cloud export
        ↓
3D Gaussian Splatting training
        ↓
3D visualization
        ↓
Metric scaling / measurement

The pipeline is partially automated through `main.py`.

Main processing stages currently include:

1. Video frame extraction
2. COLMAP sparse reconstruction
3. Sparse model merging
4. COLMAP TXT model generation
5. PLY point-cloud generation
6. Gaussian Splatting training

> **Project result:** The final prototype reached approximately ±1.5 mm overall
> measurement accuracy under the tested conditions.
>
> The latest metric-calibration developments are not yet fully reflected in the
> current public branch.

## System Architecture

The project was designed as a complete acquisition-to-reconstruction pipeline,
combining hardware integration, data acquisition, photogrammetry and 3D
reconstruction.

## Hardware

- Raspberry Pi 5
- LiDAR 
- Camera / smartphone imaging system
- Custom mechanical sensor mounts
- Power and communication interfaces

## Mechanical Integration

Custom mechanical supports were designed to integrate the computing unit,
sensors and imaging system while taking into account payload, sensor
orientation, accessibility and mechanical constraints.

## Data Acquisition

The VinciMap project combines visual acquisition with external distance
measurements used for metric calibration.

The current public software branch mainly contains the video-to-3D
reconstruction pipeline. The latest distance-sensor and metric-calibration
developments from the final prototype are not yet fully integrated into
this branch.

## Photogrammetry

COLMAP is used to estimate camera poses and reconstruct the geometry required
by the 3D reconstruction pipeline.

Video → Frames → Feature extraction → Feature matching
→ Camera pose estimation → Sparse reconstruction

## 3D Gaussian Splatting

3D Gaussian Splatting is used to generate a detailed and photorealistic
representation of the captured environment.

The reconstruction is then combined with metric information in order to move
beyond visualization and enable measurements within the scene.

## Metric Calibration

Standard photogrammetric and Gaussian Splatting reconstructions do not
inherently provide a reliable real-world scale.

VinciMap therefore uses external distance information to establish and refine
the metric scale of the reconstructed scene.

## Results

After the final calibration and reconstruction improvements, the system
achieved an overall measurement accuracy of approximately **±1.5 mm**
under the tested conditions.

## Engineering Challenges

Several technical challenges were encountered during development:

- maintaining metric consistency in the 3D reconstruction
- integrating multiple sensors on a constrained platform
- reducing payload and mechanical interference
- selecting appropriate sensor orientations
- managing heterogeneous acquisition data
- improving reconstruction quality
- validating measurements against physical references

## Software & Technologies

### 3D Reconstruction
- COLMAP / pycolmap
- 3D Gaussian Splatting
- gsplat

### Computer Vision & Data Processing
- Python
- OpenCV
- NumPy
- FFmpeg

### GPU Computing
- PyTorch
- CUDA

### Visualization
- nerfview
- PLY point clouds

### Embedded System
- Raspberry Pi 5

## Current Limitations

- reconstruction quality depends on acquisition conditions
- reflective or textureless surfaces may reduce reconstruction quality
- measurement accuracy depends on calibration quality
- computationally intensive reconstruction stages are performed off-board

## Future Work

Potential developments include:

- improved sensor synchronization
- automated metric calibration
- real-time or near-real-time reconstruction
- improved LiDAR / image fusion
- autonomous acquisition planning
- larger-scale mapping
- improved measurement tools

## Development Requirements

Before running VinciMap, install:

- [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)
- [CUDA 12.4.0](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- [COLMAP 3.13.0 (version CUDA)](https://github.com/colmap/colmap/releases/tag/3.13.0)
- [FFmpeg](https://ffmpeg.org/download.html)
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Installation

Create a Python 3.10 virtual environment:

```bash
py -3.10 -m venv venv
```

### Activate the virtual environment:

```bash
venv\Scripts\activate
```

### Install the dependencies:
> **Important:** Follow the installation steps in the order shown below. If the environment becomes corrupted or an installation step fails, delete the `venv` folder and restart the installation process.
PyTorch must be installed before the remaining dependencies:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

Additional dependencies:

```bash
python -m pip install ninja numpy jaxtyping rich
```

```bash
python -m pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
python -m pip install --no-build-isolation -r requirements.txt
```

### Run the application:

```bash
python main.py
```
