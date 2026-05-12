# TODO: Script complet qui prend une vidéo et qui la transforme en modèle 3D ply.
#
# Les étapes suivantes doivent être respectées (en suivant le google doc "Colmap") :
#
# - Vérifier si cuda installé + version demandée
# - Intégrer colmap dans la codebase ou vérifier si colmap installé + version demandée ?
# - Vérifier ffmpeg installé 
#
# - Détecter frame_rate de la vidéo
# 
# - Transformer la vidéo en images via Min(frame_rate, 12fps)
# - Stocker dans le dossier "images"
#
# - Redimensionner les images de "images" par un facteur 4
# - Stocker dans le dossier "images_4" (utile pour le training du modèle splatté avec --data_factor 4)
#
# - Appeler colmap et éxécuter automatic_reconstruction via option sparse (ATTENTION, résultat non déterministe. On peut fixer un seed si on veut un résultat déterministe)
#
# - Appeler colmap et éxécuter model_converter pour convertir sparse sous format txt (utile pour le training du modèle splatté)
# - Appeler colmap et éxécuter model_converter pour convertir sparse sous format ply (visualisation nuage de points)
#
# - Lancer simple_trainer avec les options adéquats (des options peuvent être spécifiées par l'utilisateur)
#                                   
# Une fois cela fais, voir pour l'intégration des données Lidar
#
# Idée : L'utilisateur choisis des faces (ou itération sur toutes les faces d'un axe), 
# un modèle compare la face avec la vrai photo, récupère l'ID de la photo et récupère les points Lidar associés
# 
# Ensuite, correction des gaussiennes affichées sur le même plan avec les points Lidar associés 

import argparse
import src.ffmpeg_utils as ffmpeg_utils
import src.colmap_utils as colmap_utils
import torch
from pathlib import Path
import math
import shutil
import src.simple_trainer as simple_trainer

STEP_ORDER = [
    "extract_video_images",
    "reconstruct_sparse_model",
    "merge_sparse_submodels",
    "create_sparse_txt_model",
    "create_scatter_plot_ply",
    "run_gaussian_training",
]

# Arguments

def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=False,
        default=None,
        help="Path to the input video file, name of the video will be used as workspace name.",
    )
    parser.add_argument(
        "--video_frame_rate",
        type=int,
        required=False,
        default=12,
        help="Frame rate used to extract images from the input video.",
    )
    parser.add_argument(
        "--distances_path",
        type=str,
        required=False,
        default=None,
        help="Path to the input distances TXT file.",
    )
    parser.add_argument(
        "--trainer-data_factor",
        type=int,
        required=False,
        default=1,
        help="Scaling factor applied to the training images.",
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        required=False,
        default="colmap",
        help="Path to the COLMAP executable.",
    )
    parser.add_argument(
        "--workspace_path",
        type=str,
        required=False,
        default=None,
        help="Path to an existing workspace.",
    )
    parser.add_argument(
        "--step",
        type=str,
        required=False,
        default=STEP_ORDER[0],
        choices=STEP_ORDER,
        help="Pipeline step to run from an existing workspace.",
    )

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def build_trainer_args(unknown_args: list[str], ) -> list[str]:
    trainer_args = []

    for arg in unknown_args:
        if arg in {"default", "mcmc"}:
            trainer_args.append(arg)
        elif arg.startswith("--trainer-"):
            trainer_args.append("--" + arg[len("--trainer-"):])
        else:
            trainer_args.append(arg)

    if not trainer_args or trainer_args[0] not in {"default", "mcmc"}:
        trainer_args.insert(0, "default")

    return trainer_args

# Etapes

def extract_video_images(
    video_path: str,
    images_path: Path,
    video_name: str,
    video_frame_rate: int,
    trainer_data_factor: int,
) -> Path:
    scaled_images_path = Path(images_path)

    ffmpeg_utils.video_to_images(
        video_path,
        images_path,
        video_name,
        video_frame_rate,
    )

    if trainer_data_factor > 1:
        scaled_images_path = Path(f"{images_path}_{trainer_data_factor}")
        ffmpeg_utils.images_to_scaled_images(images_path, scaled_images_path)

    return scaled_images_path

def reconstruct_sparse_model(
    data_path: Path,
    images_path: Path,
    colmap_path: str,
) -> Path:
    colmap_utils.automatic_reconstructor(
        data_path,
        images_path,
        colmap_path=colmap_path,
    )

    return Path(data_path) / "sparse"

def merge_sparse_submodels(
    sparse_path: Path,
    colmap_path: str,
) -> Path:
    submodels = sorted(
        path for path in sparse_path.iterdir()
        if path.is_dir() and path.name.isdigit()
    )

    if not submodels:
        raise FileNotFoundError(f"No sparse submodels found in : {sparse_path}")

    merged_model_path = submodels[0]

    for submodel_path in submodels[1:]:
        print(f'Merging sparse {submodel_path.name} with sparse {merged_model_path.name}')
        colmap_utils.model_merger(
            str(merged_model_path),
            str(submodel_path),
            str(merged_model_path),
            colmap_path=colmap_path,
        )
        shutil.rmtree(submodel_path)

    return merged_model_path

def create_scatter_plot_ply(
    sparse_path: Path,
    sparse_0_path: Path,
    colmap_path: str,
) -> None:
    colmap_utils.model_converter(
        sparse_0_path,
        Path(sparse_path) / "scatter_plot.ply",
        "PLY",
        colmap_path=colmap_path,
    )

def create_sparse_txt_model(
    sparse_path: Path,
    merged_model_path: Path,
    colmap_path: str,
) -> Path:
    sparse_0_bin_path = Path(sparse_path) / "0_bin"
    sparse_0_txt_path = Path(sparse_path) / "0_txt"
    sparse_0_txt_path.mkdir(parents=True, exist_ok=True)
    sparse_0_path = Path(sparse_path) / "0"

    merged_model_path.rename(sparse_0_bin_path)

    colmap_utils.model_converter(
        str(sparse_0_bin_path),
        str(sparse_0_txt_path),
        "TXT",
        colmap_path=colmap_path,
    )

    if sparse_0_path.exists():
        shutil.rmtree(sparse_0_path)

    shutil.copytree(sparse_0_txt_path, sparse_0_path)

    return sparse_0_path

def run_gaussian_training(
    workspace_path: Path,
    data_path: Path,
    trainer_data_factor: int,
    trainer_args: list[str],
) -> None:
    result_path = Path(workspace_path) / "result"

    trainer_args.extend(["--data_dir", str(data_path)])
    trainer_args.extend(["--data_factor", str(trainer_data_factor)])
    trainer_args.extend(["--result_dir", str(result_path)])

    simple_trainer.run_cli_args(trainer_args)

# Pipeline

def run_pipeline_from_step(step: str,
    workspace_path: Path,
    data_path: Path,
    images_path: Path,
    sparse_path: Path,
    video_path: str,
    video_name: str,
    video_frame_rate: int,
    trainer_data_factor: int,
    trainer_args: list[str],
    colmap_path: str) -> None:
    start_index = STEP_ORDER.index(step)
    steps_to_run = STEP_ORDER[start_index:]

    merged_model_path = None
    sparse_0_path = None

    for current_step in steps_to_run:
        if current_step == "extract_video_images":
            extract_video_images(
                video_path=video_path,
                images_path=images_path,
                video_name=video_name,
                video_frame_rate=video_frame_rate,
                trainer_data_factor=trainer_data_factor,
            )

        elif current_step == "reconstruct_sparse_model":
            reconstruct_sparse_model(
                data_path=data_path,
                images_path=images_path,
                colmap_path=colmap_path,
            )

        elif current_step == "merge_sparse_submodels":
            merged_model_path = merge_sparse_submodels(
                sparse_path=sparse_path,
                colmap_path=colmap_path,
            )

        elif current_step == "create_sparse_txt_model":
            if merged_model_path is None:
                if (sparse_path / "0_bin").exists():
                    merged_model_path = sparse_path / "0_bin"
                elif (sparse_path / "0").exists():
                    merged_model_path = sparse_path / "0"
                else:
                    raise FileNotFoundError(
                        f"No merged model found in : {sparse_path}"
                    )

            sparse_0_path = create_sparse_txt_model(
                sparse_path=sparse_path,
                merged_model_path=merged_model_path,
                colmap_path=colmap_path,
            )

        elif current_step == "create_scatter_plot_ply":
            create_scatter_plot_ply(
                sparse_path=sparse_path,
                sparse_0_path=sparse_0_path,
                colmap_path=colmap_path,
            )

        elif current_step == "run_gaussian_training":
            run_gaussian_training(
                workspace_path=workspace_path,
                data_path=data_path,
                trainer_data_factor=trainer_data_factor,
                trainer_args=trainer_args,
            )

def main(args, unknown_args):
    if args.workspace_path is not None and args.video_path is not None:
        raise ValueError(
            "Arguments --workspace_path and --video_path cannot be provided together."
        )

    trainer_args = build_trainer_args(unknown_args)

    trainer_data_factor = args.trainer_data_factor
    trainer_args.extend(['--data_factor', trainer_data_factor])
    colmap_path = args.colmap_path
    step = args.step

    if args.workspace_path is not None:
        workspace_path = Path(args.workspace_path)
        video_path = args.video_path
        video_name = workspace_path.name
    else:
        if args.video_path is None:
            raise ValueError(
                "Argument --video_path is required when --workspace_path is not provided."
            )

        video_path = args.video_path
        video_name = Path(video_path).stem
        workspace_path = Path("workspaces") / video_name
        workspace_path.mkdir(parents=True, exist_ok=False)

    data_path = workspace_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    images_path = data_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    sparse_path = data_path / "sparse"

    if video_path is not None:
        video_frame_rate = min(
            math.ceil(ffmpeg_utils.frame_rate_from_video(video_path)),
            args.video_frame_rate,
        )
    else:
        video_frame_rate = args.video_frame_rate

    if video_path is None and step == "extract_video_images":
        raise ValueError(
            "Argument --video_path is required to run step extract_video_images."
        )

    run_pipeline_from_step(
        step=step,
        workspace_path=workspace_path,
        data_path=data_path,
        images_path=images_path,
        sparse_path=sparse_path,
        video_path=video_path,
        video_name=video_name,
        video_frame_rate=video_frame_rate,
        trainer_data_factor=trainer_data_factor,
        trainer_args=trainer_args,
        colmap_path=colmap_path,
    )

if __name__ == '__main__':
    args, unknown_args = load_args()

    if not ffmpeg_utils.ffmpeg_installed():
        raise EnvironmentError(
            "FFmpeg is not installed or not accessible from the system. "
            "Please install FFmpeg and make sure it is available in the PATH."
        )

    if not colmap_utils.colmap_installed(args.colmap_path):
        raise EnvironmentError(
            "COLMAP is not installed or not accessible from the system. "
            "Please install COLMAP and make sure it is available in the PATH or pass it with --colmap_path."
        )

    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA is not available from PyTorch. "
            "Please check that CUDA is installed correctly, your NVIDIA drivers are installed, "
            "and the required binaries are available in the PATH."
        )

    main(args, unknown_args)
