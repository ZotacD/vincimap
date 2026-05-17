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
import csv
import os
import src.ffmpeg_utils as ffmpeg_utils
import src.colmap_utils as colmap_utils
import torch
from pathlib import Path
import math
import shutil
import src.simple_trainer as simple_trainer

STEP_ORDER = [
    "extract_video_images",
    "link_image_with_distances",
    "reconstruct_sparse_model",
    "merge_sparse_submodels",
    "decrease_noise",
    "create_sparse_txt_model",
    "create_scatter_plot_ply",
    "run_gaussian_training",
]


def full_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def project_relative_path(path: str | Path) -> Path:
    path = full_path(path)
    project_path = full_path(Path.cwd())
    try:
        return path.relative_to(project_path)
    except ValueError:
        try:
            return Path(os.path.relpath(path, project_path))
        except ValueError:
            return path


def config_path_value(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else full_path(Path.cwd() / path)


def executable_config_value(path: str, workspace_path: Path) -> str | Path:
    executable_path = Path(path).expanduser()
    if executable_path.parent != Path(".") or executable_path.is_absolute():
        return executable_path.resolve()
    return path


def copy_file_to_workspace(source: str | Path, workspace_path: Path) -> Path:
    source_path = full_path(source)
    output_path = full_path(workspace_path) / source_path.name
    if source_path != output_path:
        shutil.copy2(source_path, output_path)
    return output_path


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
        "--video-fov",
        type=float,
        required=False,
        default=70.0,
        help="Video horizontal field of view in degrees. Distances outside 0 +/- FOV/2 are ignored.",
    )
    parser.add_argument(
        "--distances_path",
        type=str,
        required=True,
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

def build_prefixed_args(unknown_args: list[str], prefix: str) -> list[str]:
    prefixed_args = []
    index = 0

    while index < len(unknown_args):
        arg = unknown_args[index]

        if arg.startswith(prefix):
            prefixed_args.append("--" + arg[len(prefix):])

            if index + 1 < len(unknown_args) and not unknown_args[index + 1].startswith("--"):
                prefixed_args.append(unknown_args[index + 1])
                index += 1

        index += 1

    return prefixed_args

def build_trainer_args(unknown_args: list[str]) -> list[str]:
    trainer_args = [
        arg for arg in unknown_args
        if arg in {"default", "mcmc"}
    ]
    trainer_args.extend(build_prefixed_args(unknown_args, "--trainer-"))

    if not trainer_args or trainer_args[0] not in {"default", "mcmc"}:
        trainer_args.insert(0, "default")

    return trainer_args

def angle_in_video_fov(angle_deg: float, video_fov: float) -> bool:
    if video_fov >= 360.0:
        return True

    wrapped_angle = (angle_deg + 180.0) % 360.0 - 180.0
    return abs(wrapped_angle) <= video_fov / 2.0

def find_angle_column_index(header: list[str]) -> int:
    for index, column_name in enumerate(header):
        if column_name.strip().lower() == "angle(deg)":
            return index

    for index, column_name in enumerate(header):
        if "angle" in column_name.strip().lower():
            return index

    raise ValueError("Distances file must contain an angle column.")


def find_distance_column_index(header: list[str]) -> int:
    for index, column_name in enumerate(header):
        if column_name.strip().lower() == "distance(mm)":
            return index

    for index, column_name in enumerate(header):
        if "distance" in column_name.strip().lower():
            return index

    raise ValueError("Distances file must contain a distance column.")


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

def link_image_with_distances(
    distances_path: str,
    data_path: Path,
    images_path: Path,
    video_duration: float | None,
    video_frame_rate: int,
    video_fov: float,
) -> Path:
    input_distances_path = Path(distances_path)
    output_distances_path = Path(data_path) / "distances.txt"

    if not input_distances_path.is_file():
        raise FileNotFoundError(f"Distances file not found : {input_distances_path}")

    image_names = sorted(
        path.name for path in images_path.iterdir()
    )

    if not image_names:
        raise FileNotFoundError(f"No images found in : {images_path}")

    with input_distances_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        sample = input_file.read(4096)
        input_file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t;,")
        except csv.Error:
            dialect = csv.excel_tab

        rows = [
            row for row in csv.reader(input_file, dialect)
            if any(cell.strip() for cell in row)
        ]

    if not rows:
        raise ValueError(f"Distances file is empty : {input_distances_path}")

    header = rows[0]
    distance_rows = rows[1:]

    if not distance_rows:
        raise ValueError(f"Distances file has no data rows : {input_distances_path}")
    if video_fov <= 0.0 or video_fov > 360.0:
        raise ValueError(f"Video FOV must be in ]0, 360], got {video_fov}.")

    angle_index = find_angle_column_index(header)
    distance_index = find_distance_column_index(header)
    distance_rows = [
        row for row in distance_rows
        if (
            len(row) > max(angle_index, distance_index)
            and angle_in_video_fov(float(row[angle_index]), video_fov)
        )
    ]

    if not distance_rows:
        raise ValueError(
            f"No distance rows remain after applying video FOV {video_fov} deg."
        )

    if video_duration is not None:
        expected_image_count = max(1, math.ceil(video_duration * video_frame_rate))
    else:
        expected_image_count = len(image_names)

    image_count = min(expected_image_count, len(image_names))
    distances_per_image = len(distance_rows) / image_count

    output_distances_path.parent.mkdir(parents=True, exist_ok=True)

    with output_distances_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(
            output_file,
            delimiter=dialect.delimiter,
            lineterminator="\n",
        )
        writer.writerow(["angle", "distance", "image_id"])

        for row_index, row in enumerate(distance_rows):
            image_index = min(
                int(row_index / distances_per_image),
                image_count - 1,
            )
            writer.writerow([row[angle_index], row[distance_index], image_names[image_index]])

    return output_distances_path

def reconstruct_sparse_model(
    data_path: Path,
    images_path: Path,
    colmap_path: str,
) -> Path:
    database_path = Path(data_path) / "database.db"

    sparse_path = Path(data_path) / "sparse"
    sparse_path.mkdir(exist_ok=True)

    configs_path = create_workspace_configs(
        data_path,
        database_path,
        images_path,
        sparse_path,
    )
    feature_extractor_config_path = configs_path / "feature_extractor.ini"
    sequential_matcher_config_path = configs_path / "sequential_matcher.ini"
    mapper_config_path = configs_path / "mapper.ini"

    if not database_path.exists():
        colmap_utils.database_creator(database_path, colmap_path=colmap_path)

    colmap_utils.feature_extractor(
        feature_extractor_config_path,
        colmap_path=colmap_path,
    )

    colmap_utils.sequential_matcher(
        sequential_matcher_config_path,
        colmap_path=colmap_path,
    )

    colmap_utils.mapper(
        mapper_config_path,
        colmap_path=colmap_path,
    )

    return sparse_path

def create_workspace_configs(
    data_path: Path,
    database_path: Path,
    images_path: Path,
    sparse_path: Path,
    video_path: str | None = None,
    video_frame_rate: int | None = None,
    video_fov: float | None = None,
    distances_path: str | None = None,
    colmap_path: str | None = None,
) -> Path:
    data_path = full_path(data_path)
    database_path = full_path(database_path)
    images_path = full_path(images_path)
    sparse_path = full_path(sparse_path)
    configs_path = Path(data_path) / "configs"
    configs_path.mkdir(parents=True, exist_ok=True)

    workspace_path = full_path(configs_path.parent.parent)
    result_path = workspace_path / "result"
    linked_distances_path = Path(data_path) / "distances.txt"

    configs_to_create = [
        (
            Path("configs") / "colmap" / "feature_extractor.ini",
            configs_path / "feature_extractor.ini",
            None,
            {
                "database_path": project_relative_path(database_path),
                "image_path": project_relative_path(images_path),
            },
        ),
        (
            Path("configs") / "colmap" / "sequential_matcher.ini",
            configs_path / "sequential_matcher.ini",
            None,
            {
                "database_path": project_relative_path(database_path),
            },
        ),
        (
            Path("configs") / "colmap" / "mapper.ini",
            configs_path / "mapper.ini",
            None,
            {
                "database_path": project_relative_path(database_path),
                "image_path": project_relative_path(images_path),
                "output_path": project_relative_path(sparse_path),
            },
        ),
        (
            Path("configs") / "simple_trainer" / "trainer.ini",
            configs_path / "trainer.ini",
            "trainer",
            {
                "data_dir": project_relative_path(data_path),
                "result_dir": project_relative_path(result_path),
            },
        ),
        (
            Path("configs") / "distances_computer" / "distances_computer.ini",
            configs_path / "distances_computer.ini",
            "distances_computer",
            {
                "distances_path": project_relative_path(linked_distances_path),
            },
        ),
    ]

    if video_path is not None and distances_path is not None and colmap_path is not None:
        configs_to_create.append(
            (
                Path("configs") / "workspace" / "workspace.ini",
                workspace_path / "workspace.ini",
                "workspace",
                {
                    "video_path": project_relative_path(video_path),
                    "video_frame_rate": video_frame_rate,
                    "video_fov": video_fov,
                    "distances_path": project_relative_path(distances_path),
                    "colmap_path": executable_config_value(colmap_path, workspace_path),
                },
            )
        )

    for template_path, output_path, section_name, values in configs_to_create:
        write_workspace_config(
            template_path,
            output_path,
            values,
            section_name=section_name,
        )

    return configs_path

def write_workspace_config(
    template_path: Path,
    output_path: Path,
    values: dict[str, object],
    section_name: str | None = None,
) -> None:
    config_content = Path(template_path).read_text(encoding="utf-8")
    config_content = apply_workspace_config_values(
        config_content,
        values,
        section_name=section_name,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as output_file:
        output_file.write(config_content)

def apply_workspace_config_values(
    config_content: str,
    values: dict[str, object],
    section_name: str | None = None,
) -> str:
    value_lines = [f"{key} = {value}\n" for key, value in values.items()]

    if section_name is None:
        return "".join(value_lines) + config_content

    lines = config_content.splitlines(keepends=True)
    section_header = f"[{section_name}]"
    section_start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().lower() == section_header.lower()
        ),
        None,
    )

    if section_start is None:
        suffix = "" if config_content.endswith("\n") else "\n"
        return (
            config_content
            + suffix
            + "\n"
            + section_header
            + "\n"
            + "".join(value_lines)
        )

    section_end = next(
        (
            index
            for index in range(section_start + 1, len(lines))
            if lines[index].lstrip().startswith("[")
        ),
        len(lines),
    )

    dynamic_keys = set(values)
    section_lines = [
        line
        for line in lines[section_start + 1:section_end]
        if line.split("=", 1)[0].strip() not in dynamic_keys
    ]

    return "".join(
        lines[:section_start + 1]
        + value_lines
        + section_lines
        + lines[section_end:]
    )

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

    merged_sparse_0_path = submodels[0]

    for submodel_path in submodels[1:]:
        print(f'Merging sparse {submodel_path.name} with sparse {merged_sparse_0_path.name}')
        colmap_utils.model_merger(
            str(merged_sparse_0_path),
            str(submodel_path),
            str(merged_sparse_0_path),
            colmap_path=colmap_path,
        )
        shutil.rmtree(submodel_path)

    return merged_sparse_0_path

def decrease_noise(
    data_path: Path,
    merged_sparse_0_path: Path,
    colmap_path: str,
) -> Path:
    data_path = full_path(data_path)
    merged_sparse_0_path = full_path(merged_sparse_0_path)
    workspace_path = data_path.parent
    configs_path = Path(data_path) / "configs"
    configs_path.mkdir(parents=True, exist_ok=True)

    point_filtering_config_path = configs_path / "point_filtering.ini"
    bundle_adjuster_config_path = configs_path / "bundle_adjuster.ini"

    write_workspace_config(
        Path("configs") / "colmap" / "point_filtering.ini",
        point_filtering_config_path,
        {
            "input_path": project_relative_path(merged_sparse_0_path),
            "output_path": project_relative_path(merged_sparse_0_path),
        },
    )

    colmap_utils.point_filtering(
        point_filtering_config_path,
        colmap_path=colmap_path,
    )

    write_workspace_config(
        Path("configs") / "colmap" / "bundle_adjuster.ini",
        bundle_adjuster_config_path,
        {
            "input_path": project_relative_path(merged_sparse_0_path),
            "output_path": project_relative_path(merged_sparse_0_path),
        },
    )
    colmap_utils.bundle_adjuster(
        bundle_adjuster_config_path,
        colmap_path=colmap_path,
    )

    return merged_sparse_0_path

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
    merged_sparse_0_path: Path,
    colmap_path: str,
) -> Path:
    sparse_0_bin_path = Path(sparse_path) / "0_bin"
    sparse_0_txt_path = Path(sparse_path) / "0_txt"
    sparse_0_path = Path(sparse_path) / "0"

    for path in [sparse_0_bin_path, sparse_0_txt_path]:
        if path.exists():
            shutil.rmtree(path)

    if not merged_sparse_0_path.exists():
        raise FileNotFoundError(
            f"Merged sparse model not found before conversion: {merged_sparse_0_path}"
        )

    merged_sparse_0_path.rename(sparse_0_bin_path)
    sparse_0_txt_path.mkdir(parents=True)

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
    distances_path: str,
    video_duration: float | None,
    video_frame_rate: int,
    video_fov: float,
    trainer_data_factor: int,
    trainer_args: list[str],
    colmap_path: str) -> None:
    start_index = STEP_ORDER.index(step)
    steps_to_run = STEP_ORDER[start_index:]

    merged_sparse_0_path = None
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

        elif current_step == "link_image_with_distances":
            link_image_with_distances(
                distances_path=distances_path,
                data_path=data_path,
                images_path=images_path,
                video_duration=video_duration,
                video_frame_rate=video_frame_rate,
                video_fov=video_fov,
            )

        elif current_step == "reconstruct_sparse_model":
            reconstruct_sparse_model(
                data_path=data_path,
                images_path=images_path,
                colmap_path=colmap_path,
            )

        elif current_step == "merge_sparse_submodels":
            merged_sparse_0_path = merge_sparse_submodels(
                sparse_path=sparse_path,
                colmap_path=colmap_path,
            )

        elif current_step == "decrease_noise":
            if merged_sparse_0_path is None:
                if (sparse_path / "0").exists():
                    merged_sparse_0_path = sparse_path / "0"
                else:
                    raise FileNotFoundError(
                        f"No merged model found in : {sparse_path}"
                    )

            merged_sparse_0_path = decrease_noise(
                data_path=data_path,
                merged_sparse_0_path=merged_sparse_0_path,
                colmap_path=colmap_path,
            )

        elif current_step == "create_sparse_txt_model":
            if merged_sparse_0_path is None:
                if (sparse_path / "0_bin").exists():
                    merged_sparse_0_path = sparse_path / "0_bin"
                elif (sparse_path / "0").exists():
                    merged_sparse_0_path = sparse_path / "0"
                else:
                    raise FileNotFoundError(
                        f"No merged model found in : {sparse_path}"
                    )

            sparse_0_path = create_sparse_txt_model(
                sparse_path=sparse_path,
                merged_sparse_0_path=merged_sparse_0_path,
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
    trainer_args.extend(["--data_factor", str(trainer_data_factor)])
    colmap_path = args.colmap_path
    step = args.step

    if args.workspace_path is not None:
        workspace_path = full_path(args.workspace_path)
        video_path = args.video_path
        video_name = workspace_path.name
    else:
        if args.video_path is None:
            raise ValueError(
                "Argument --video_path is required when --workspace_path is not provided."
            )

        video_path = args.video_path
        video_name = Path(video_path).stem
        workspace_path = full_path(Path("workspaces") / video_name)
        workspace_path.mkdir(parents=True, exist_ok=False)

    data_path = workspace_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    images_path = data_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    sparse_path = data_path / "sparse"

    video_duration = None

    if video_path is not None:
        video_path = copy_file_to_workspace(video_path, workspace_path)
        distances_path = copy_file_to_workspace(args.distances_path, workspace_path)
        video_frame_rate = min(
            math.ceil(ffmpeg_utils.frame_rate_from_video(video_path)),
            args.video_frame_rate,
        )
        video_duration = ffmpeg_utils.duration_from_video(video_path)
        create_workspace_configs(
            data_path=data_path,
            database_path=data_path / "database.db",
            images_path=images_path,
            sparse_path=sparse_path,
            video_path=video_path,
            video_frame_rate=video_frame_rate,
            video_fov=args.video_fov,
            distances_path=distances_path,
            colmap_path=colmap_path,
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
        distances_path=str(config_path_value(distances_path)) if video_path is not None else args.distances_path,
        video_duration=video_duration,
        video_frame_rate=video_frame_rate,
        video_fov=args.video_fov,
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
