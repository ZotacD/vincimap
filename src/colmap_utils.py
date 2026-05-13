import subprocess
from pathlib import Path

"""
Vérifie si COLMAP est installé
et accessible depuis le système.
"""
def colmap_installed(colmap_path: str = "colmap") -> bool:
    try:
        subprocess.run(
            [colmap_path, "-h"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


"""
Génère un fichier projet COLMAP
dans le dossier accessible via ``data_path``.

Crée le dossier ``data_path`` s'il n'existe pas.
"""
def project_generator(data_path: str, filename: str = "project.ini", colmap_path: str = "colmap") -> None:
    output_dir = Path(data_path)
    output_file = output_dir / filename

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                colmap_path,
                "project_generator",
                "--output_path",
                str(output_file),
            ],
            check=True,
            text=True,
        )

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(
            "COLMAP error during project generation :\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )
        raise


"""
Crée une base de données COLMAP
accessible via ``database_path``.

Crée le dossier parent de ``database_path`` s'il n'existe pas.
"""
def database_creator(database_path: str, colmap_path: str = "colmap") -> None:
    output_file = Path(database_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                colmap_path,
                "database_creator",
                "--database_path",
                str(output_file),
            ],
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(
            "COLMAP error during database creation :\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )
        raise


"""
Reconstruit automatiquement une scène 3D
dans l'espace de travail accessible via ``workspace_path``
à partir des images accessibles via ``images_path``.

Crée le dossier ``workspace_path`` s'il n'existe pas.
"""
def automatic_reconstructor(
    workspace_path: str,
    images_path: str,
    project_path: str,
    colmap_path: str = "colmap",
) -> None:
    output_dir = Path(workspace_path)
    input_dir = Path(images_path)
    project_file = Path(project_path)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found : {input_dir}")

    if not project_file.is_file():
        raise FileNotFoundError(f"Project file not found : {project_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                colmap_path,
                "automatic_reconstructor",
                "--project_path",
                str(project_file),
                "--workspace_path",
                str(output_dir),
                "--dense",
                "false",
                "--data_type",
                "video",
            ],
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(
            "COLMAP error during automatic reconstruction :\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )
        raise


"""
Transforme le modèle accessible via ``input_path``
en un modèle accessible via ``output_path``
au format ``output_type``.

Crée le dossier parent de ``output_path`` s'il n'existe pas.
"""
def model_converter(input_path: str, output_path: str, output_type: str, colmap_path: str = "colmap") -> None:
    input_dir = Path(input_path)
    output_file = Path(output_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"Model not found : {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                colmap_path,
                "model_converter",
                "--input_path",
                str(input_dir),
                "--output_path",
                str(output_file),
                "--output_type",
                output_type,
            ],
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(
            "COLMAP error during model conversion :\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )
        raise
    
"""
Fusionne les modèles accessibles via ``input_path1``
et ``input_path2``
en un modèle accessible via ``output_path``.

Crée le dossier ``output_path`` s'il n'existe pas.
"""
def model_merger(input_path1: str, input_path2: str, output_path: str, colmap_path: str = "colmap") -> None:
    input_dir1 = Path(input_path1)
    input_dir2 = Path(input_path2)
    output_dir = Path(output_path)

    if not input_dir1.exists():
        raise FileNotFoundError(f"Model not found : {input_dir1}")

    if not input_dir2.exists():
        raise FileNotFoundError(f"Model not found : {input_dir2}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                colmap_path,
                "model_merger",
                "--input_path1",
                str(input_dir1),
                "--input_path2",
                str(input_dir2),
                "--output_path",
                str(output_dir),
            ],
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(
            "COLMAP error during model merge :\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )
        raise
