import subprocess
from pathlib import Path


def colmap_installed(
    colmap_path: str = "colmap",
) -> bool:
    """Check whether COLMAP is installed and accessible from the system."""

    try:
        subprocess.run(
            [
                colmap_path,
                "-h",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return True

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return False


def automatic_reconstructor(
    workspace_path: str,
    images_path: str,
    colmap_path: str = "colmap",
) -> None:
    """Automatically reconstruct a 3D scene from images.

    The reconstruction is written to ``workspace_path`` from the images stored
    in ``images_path``. The workspace directory is created if necessary.
    """

    output_dir = Path(
        workspace_path
    )

    input_dir = Path(
        images_path
    )

    if not input_dir.is_dir():

        raise FileNotFoundError(
            f"Images folder not found: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        subprocess.run(
            [
                colmap_path,
                "automatic_reconstructor",
                "--workspace_path",
                str(output_dir),
                "--image_path",
                str(input_dir),
                "--dense",
                "false",
                "--data_type",
                "video",
            ],
            check=True,
            text=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as e:

        print(
            "COLMAP error during automatic reconstruction:\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )

        raise


def model_converter(
    input_path: str,
    output_path: str,
    output_type: str,
    colmap_path: str = "colmap",
) -> None:
    """Convert a COLMAP model to another output format.

    ``input_path`` points to the source model,
    ``output_path`` to the converted model,
    and ``output_type`` defines the target format.
    """

    input_dir = Path(
        input_path
    )

    output_file = Path(
        output_path
    )

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Model not found: {input_dir}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as e:

        print(
            "COLMAP error during model conversion:\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )

        raise


def model_merger(
    input_path1: str,
    input_path2: str,
    output_path: str,
    colmap_path: str = "colmap",
) -> None:
    """Merge two COLMAP models into a single output model.

    The output directory is created automatically if it does not exist.
    """

    input_dir1 = Path(
        input_path1
    )

    input_dir2 = Path(
        input_path2
    )

    output_dir = Path(
        output_path
    )

    if not input_dir1.exists():

        raise FileNotFoundError(
            f"Model not found: {input_dir1}"
        )

    if not input_dir2.exists():

        raise FileNotFoundError(
            f"Model not found: {input_dir2}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as e:

        print(
            "COLMAP error during model merge:\n"
            f"{e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else e}"
        )

        raise
