# VinciMap

**Metric 3D mapping using LiDAR, photogrammetry and 3D Gaussian Splatting**

VinciMap is a five-student engineering project developed to explore a fast,
flexible and accessible approach to metric 3D mapping.

 Best Project Execution Award – ESILV Nantes P2IP  
 
## Project Overview

VinciMap combines several acquisition and reconstruction technologies to create
measurable 3D representations of real environments.

The system explores the combination of:

- LiDAR / Time-of-Flight sensing
- Photogrammetry
- 3D Gaussian Splatting
- Embedded data acquisition
- Metric calibration
- 3D reconstruction and visualization

The objective was not only to generate visually realistic 3D scenes, but also
to preserve sufficient metric consistency to perform measurements inside the
reconstructed environment.

Sensors / Camera
       ↓
Data Acquisition
       ↓
Raspberry Pi 5
       ↓
Images + Distance Data
       ↓
    COLMAP
       ↓
Camera poses / sparse reconstruction
       ↓
3D Gaussian Splatting
       ↓
Metric calibration
       ↓
Measurable 3D environment

## System Architecture

The project was designed as a complete acquisition-to-reconstruction pipeline,
combining hardware integration, data acquisition, photogrammetry and 3D
reconstruction.

# Hardware

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

The acquisition system collects visual information together with distance
measurements from the onboard sensors.

Particular attention was paid to:

- sensor positioning
- acquisition consistency
- field of view
- calibration
- data organization

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

- Python
- COLMAP
- CUDA
- gsplat / 3D Gaussian Splatting
- Raspberry Pi
- Git / GitHub

## Usage

### 1. Prepare the input data
...

### 2. Run COLMAP
...

### 3. Train the 3D Gaussian Splatting model
...

### 4. Apply metric calibration
...

### 5. Visualize and measure
...

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


## Prérequis dev

Avant de lancer le projet, installer les dépendances suivantes :

- [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)
- [CUDA 12.4.0](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- [COLMAP 3.13.0 (version CUDA)](https://github.com/colmap/colmap/releases/tag/3.13.0)
- [FFmpeg](https://ffmpeg.org/download.html)
- Dernière version de C++. Voir ["Microsoft C++ Build Tools"](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

## Installation

Créer un environnement virtuel Python 3.10 :

```bash
python -3.10 -m venv venv
```

### Activer l’environnement virtuel :

```bash
venv/scripts/activate
```

### Installer les dépendances :

*Attention, l'éxécution doit se faire __dans l'odre__, supprimer le dossier venv si vous rencontrez une erreur et recommencer lé début de l'installation*

PyTorch doit être préinstallé pour installer les dépendances complémentaires :

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

Dépendances complémentaires :

```bash
python -m install ninja numpy jaxtyping rich
```

```bash
python -m install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
python -m pip install --no-build-isolation -r requirements.txt
```

### Lancer l’application :

```bash
python main.py
```
