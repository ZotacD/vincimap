import argparse
import csv
import dataclasses
import math
import os
import shutil
from pathlib import Path

from configobj import ConfigObj

import src.colmap_utils as colmap_utils
import src.ffmpeg_utils as ffmpeg_utils
import torch
import src.simple_trainer as simple_trainer


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["create_workspace", "run_colmap", "run_gaussian_training"],
        help="Action to run.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=False,
        default=None,
        help="Path to the input video file. Required by create_workspace.",
    )
    parser.add_argument(
        "--distances_path",
        type=str,
        required=False,
        default=None,
        help="Path to the input distances TXT file. Required by create_workspace.",
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        required=False,
        default="colmap",
        help="Path to the COLMAP executable. Saved in workspace.ini by create_workspace.",
    )
    parser.add_argument(
        "--workspace_path",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to the workspace. Optional for create_workspace, required by "
            "run_colmap and run_gaussian_training."
        ),
    )
    return parser.parse_args()


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


def executable_from_config(path: str) -> str:
    executable_path = Path(path).expanduser()
    if executable_path.parent == Path(".") and not executable_path.is_absolute():
        return path
    return str(config_path_value(executable_path))


def copy_file_to_workspace(source: str | Path, workspace_path: Path) -> Path:
    source_path = full_path(source)
    output_path = full_path(workspace_path) / source_path.name
    if source_path != output_path:
        shutil.copy2(source_path, output_path)
    return output_path


def paths(workspace_path: Path) -> tuple[Path, Path, Path, Path]:
    workspace_path = full_path(workspace_path)
    data_path = workspace_path / "data"
    return data_path, data_path / "images", data_path / "sparse", workspace_path / "configs"


def config(path: Path) -> ConfigObj:
    return ConfigObj(str(path), encoding="utf-8", list_values=False, write_empty_values=True)


def section(path: Path, name: str):
    return config(path)[name]


def workspace_section(workspace_path: Path):
    return section(workspace_path / "configs" / "workspace.ini", "workspace")


def write_config(
    template_path: Path,
    output_path: Path,
    values: dict[str, object],
    section_name: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config(template_path)
    target = cfg if section_name is None else cfg.setdefault(section_name, {})

    for key, value in values.items():
        target[key] = str(value)

    cfg.filename = str(output_path)
    cfg.write()


def create_workspace_configs(
    workspace_path: Path,
    video_path: str,
    distances_path: str,
    colmap_path: str,
) -> Path:
    workspace_path = full_path(workspace_path)
    data_path, images_path, sparse_path, configs_path = paths(workspace_path)
    result_path = workspace_path / "result"
    database_path = data_path / "database.db"

    configs = [
        (
            Path("configs") / "workspace.ini",
            configs_path / "workspace.ini",
            "workspace",
            {
                "video_path": project_relative_path(video_path),
                "distances_path": project_relative_path(distances_path),
                "colmap_path": executable_config_value(colmap_path, workspace_path),
            },
        ),
        (
            Path("configs") / "trainer.ini",
            configs_path / "trainer.ini",
            "trainer",
            {
                "data_dir": project_relative_path(data_path),
                "result_dir": project_relative_path(result_path),
            },
        ),
        (
            Path("configs") / "distances_computer.ini",
            configs_path / "distances_computer.ini",
            "distances_computer",
            {"distances_path": project_relative_path(data_path / "distances.txt")},
        ),
        (
            Path("configs") / "colmap" / "feature_extractor.ini",
            configs_path / "colmap" / "feature_extractor.ini",
            None,
            {
                "database_path": project_relative_path(database_path),
                "image_path": project_relative_path(images_path),
            },
        ),
        (
            Path("configs") / "colmap" / "sequential_matcher.ini",
            configs_path / "colmap" / "sequential_matcher.ini",
            None,
            {"database_path": project_relative_path(database_path)},
        ),
        (
            Path("configs") / "colmap" / "mapper.ini",
            configs_path / "colmap" / "mapper.ini",
            None,
            {
                "database_path": project_relative_path(database_path),
                "image_path": project_relative_path(images_path),
                "output_path": project_relative_path(sparse_path),
            },
        ),
        (
            Path("configs") / "colmap" / "point_filtering.ini",
            configs_path / "colmap" / "point_filtering.ini",
            None,
            {},
        ),
        (
            Path("configs") / "colmap" / "bundle_adjuster.ini",
            configs_path / "colmap" / "bundle_adjuster.ini",
            None,
            {},
        ),
    ]

    for template, output, section_name, values in configs:
        write_config(Path(template), output, values, section_name)

    return configs_path


def angle_in_fov(angle: float, fov: float) -> bool:
    return fov >= 360.0 or abs((angle + 180.0) % 360.0 - 180.0) <= fov / 2.0


def link_images_with_distances(
    distances_path: str,
    data_path: Path,
    images_path: Path,
    duration: float,
    fps: int,
    fov: float,
) -> None:
    image_names = sorted(path.name for path in images_path.iterdir())

    with Path(distances_path).open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t;,")
        except csv.Error:
            dialect = csv.excel_tab
        rows = [row for row in csv.reader(f, dialect) if any(cell.strip() for cell in row)]

    header, data_rows = rows[0], rows[1:]
    angle_col = "angle"
    distance_col = "distance"
    data_rows = [
        row for row in data_rows
        if (
            len(row) > max(angle_col, distance_col)
            and angle_in_fov(float(row[angle_col]), fov)
        )
    ]

    image_count = min(max(1, math.ceil(duration * fps)), len(image_names))
    rows_per_image = len(data_rows) / image_count

    with (data_path / "distances.txt").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=dialect.delimiter, lineterminator="\n")
        writer.writerow(["angle", "distance", "image_id"])
        for i, row in enumerate(data_rows):
            writer.writerow(
                [
                    row[angle_col],
                    row[distance_col],
                    image_names[min(int(i / rows_per_image), image_count - 1)],
                ]
            )


def action_create_workspace(args: argparse.Namespace) -> None:
    if not ffmpeg_utils.ffmpeg_installed():
        raise EnvironmentError(
            "FFmpeg is not installed or not accessible from the system. "
            "Please install FFmpeg and make sure it is available in the PATH."
        )
    
    if not args.video_path or not args.distances_path:
        raise ValueError("--video_path and --distances_path are required.")

    source_video_path = full_path(args.video_path)
    workspace_path = full_path(args.workspace_path or Path("workspaces") / source_video_path.stem)
    data_path, images_path, sparse_path, configs_path = paths(workspace_path)
    for path in [data_path, images_path, sparse_path, configs_path]:
        path.mkdir(parents=True, exist_ok=True)

    video_path = copy_file_to_workspace(source_video_path, workspace_path)
    distances_path = copy_file_to_workspace(args.distances_path, workspace_path)

    create_workspace_configs(
        workspace_path,
        video_path=video_path,
        distances_path=distances_path,
        colmap_path=args.colmap_path,
    )

    workspace = workspace_section(workspace_path)
    trainer = section(configs_path / "trainer.ini", "trainer")
    fps = min(math.ceil(ffmpeg_utils.frame_rate_from_video(video_path)), workspace.as_int("video_frame_rate"))
    duration = ffmpeg_utils.duration_from_video(video_path)
    data_factor = trainer.as_int("data_factor")

    write_config(
        configs_path / "workspace.ini",
        configs_path / "workspace.ini",
        {"video_frame_rate": fps},
        "workspace",
    )
    ffmpeg_utils.video_to_images(video_path, images_path, video_path.stem, fps)
    if data_factor > 1:
        ffmpeg_utils.images_to_scaled_images(images_path, Path(f"{images_path}_{data_factor}"))
    link_images_with_distances(
        distances_path=config_path_value(workspace["distances_path"]),
        data_path=data_path,
        images_path=images_path,
        duration=duration,
        fps=fps,
        fov=workspace.as_float("video_fov"),
    )

def action_run_colmap(args: argparse.Namespace) -> None:
    workspace_path = full_path(args.workspace_path)
    data_path, images_path, sparse_path, configs_path = paths(workspace_path)
    colmap_path = executable_from_config(workspace_section(workspace_path)["colmap_path"])
    colmap_configs_path = configs_path / "colmap"

    if not colmap_utils.colmap_installed(colmap_path):
        raise EnvironmentError(
            "COLMAP is not installed or not accessible from the system. "
            "Please install COLMAP and make sure it is available in the PATH or pass it with --colmap_path."
        )

    database_path = data_path / "database.db"
    colmap_utils.database_creator(database_path, colmap_path=colmap_path)

    colmap_utils.feature_extractor(colmap_configs_path / "feature_extractor.ini", colmap_path=colmap_path)
    colmap_utils.sequential_matcher(colmap_configs_path / "sequential_matcher.ini", colmap_path=colmap_path)
    colmap_utils.mapper(colmap_configs_path / "mapper.ini", colmap_path=colmap_path)

    submodels = sorted(path for path in sparse_path.iterdir() if path.is_dir() and path.name.isdigit())
    merged = submodels[0]
    for submodel in submodels[1:]:
        colmap_utils.model_merger(str(merged), str(submodel), str(merged), colmap_path=colmap_path)
        shutil.rmtree(submodel)

    for name in ["point_filtering", "bundle_adjuster"]:
        cfg_path = colmap_configs_path / f"{name}.ini"
        write_config(
            Path(f"configs/colmap/{name}.ini"),
            cfg_path,
            {
                "input_path": project_relative_path(merged),
                "output_path": project_relative_path(merged),
            },
        )
        getattr(colmap_utils, name)(cfg_path, colmap_path=colmap_path)

    bin_path, txt_path, sparse_0 = sparse_path / "0_bin", sparse_path / "0_txt", sparse_path / "0"
    merged.rename(bin_path)
    txt_path.mkdir(parents=True)
    colmap_utils.model_converter(str(bin_path), str(txt_path), "TXT", colmap_path=colmap_path)
    shutil.copytree(txt_path, sparse_0)
    colmap_utils.model_converter(sparse_0, sparse_path / "scatter_plot.ply", "PLY", colmap_path=colmap_path)


def value_to_args(name: str, value: str) -> list[str]:
    value = value.strip()
    return [] if not value else [f"--{name}", *[part.strip() for part in value.split(",") if part.strip()]]


def bool_value(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "on"}:
        return True
    if normalized in {"false", "no", "off"}:
        return False
    return None


def trainer_bool_defaults() -> dict[str, bool]:
    return {
        field.name: field.default
        for field in dataclasses.fields(simple_trainer.Config)
        if field.type is bool and isinstance(field.default, bool)
    }


def bool_to_args(name: str, value: bool, default: bool | None) -> list[str]:
    if value:
        return [] if default is True else [f"--{name}"]
    return [] if default is False else [f"--no-{name}"]


def trainer_strategy_options(workspace_path: Path) -> dict[str, object]:
    _, _, _, configs_path = paths(workspace_path)
    cfg = config(configs_path / "trainer.ini")
    trainer = cfg["trainer"]
    strategy = trainer.get("strategy", "default").strip() or "default"
    section_name = f"{strategy}_strategy"

    if section_name not in cfg:
        return {}

    return {strategy: cfg[section_name]}


def training_args(workspace_path: Path) -> list[str]:
    workspace_path = full_path(workspace_path)
    _, _, _, configs_path = paths(workspace_path)
    trainer = section(configs_path / "trainer.ini", "trainer")
    strategy = trainer.get("strategy", "default").strip() or "default"
    args = [strategy]
    bool_defaults = trainer_bool_defaults()

    for key, value in trainer.items():
        if key != "strategy":
            if key in {"data_dir", "result_dir"} and value.strip():
                value = str(config_path_value(value))
            parsed_bool = bool_value(value)
            if parsed_bool is not None and key in bool_defaults:
                args.extend(bool_to_args(key, parsed_bool, bool_defaults[key]))
            else:
                args.extend(value_to_args(key, value))

    return args


def action_run_gaussian_training(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA is not available from PyTorch. "
            "Please check that CUDA is installed correctly, your NVIDIA drivers are installed, "
            "and the required binaries are available in the PATH."
        )

    workspace_path = full_path(args.workspace_path)
    simple_trainer.run_cli_args(
        training_args(workspace_path),
        strategy_options=trainer_strategy_options(workspace_path),
    )


def main() -> None:
    args = load_args()
    
    actions = {
        "create_workspace": action_create_workspace,
        "run_colmap": action_run_colmap,
        "run_gaussian_training": action_run_gaussian_training,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
