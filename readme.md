# Vincimap

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

### Lancer l’application :

```bash
python main.py
```