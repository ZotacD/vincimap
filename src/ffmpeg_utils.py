import subprocess
import ffmpeg
from pathlib import Path

"""
Vérifie si FFmpeg est installé
et accessible depuis le système.
"""
def ffmpeg_installed() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    
"""
Transforme la vidéo accessible via ``video_path``
en une série d'images extraites de celle-ci.

Crée le dossier ``images_path`` s'il n'existe pas.
"""
def video_to_images(video_path: str, images_path: str, image_name: str, fps: int = 12) -> None:
    input_video = Path(video_path)
    output_dir = Path(images_path)

    if not input_video.is_file():
        raise FileNotFoundError(f"Video not found : {input_video}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / f"{image_name}_%06d.jpg"

    try:
        (
            ffmpeg
            .input(str(input_video))
            .filter("fps", fps=fps)
            .output(str(output_pattern), start_number=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        print(f"FFmpeg error during image extraction :\n{stderr}")

"""
Transforme les images accessibles via ``images_path``
en une série d'images redimensionnées.

Crée le dossier ``scaled_images_path`` s'il n'existe pas.
"""
def images_to_scaled_images(images_path: str, scaled_images_path: str) -> None:
    input_dir = Path(images_path)
    output_dir = Path(scaled_images_path)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found : {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    input_images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not input_images:
        raise FileNotFoundError(f"No images found in : {input_dir}")

    for input_image in input_images:
        output_image = output_dir / input_image.name

        try:
            (
                ffmpeg
                .input(str(input_image))
                .filter("scale", "iw/2", "ih/2")
                .output(str(output_image))
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except Exception as e:
            print(
                f"FFmpeg error during resizing {input_image.name} : {e}"
            )
        
"""
Retourne le frame rate de la vidéo
accessible via ``video_path``.
"""
def frame_rate_from_video(video_path: str) -> float:
    input_video = Path(video_path)

    if not input_video.is_file():
        raise FileNotFoundError(f"Video not found : {input_video}")

    try:
        probe = ffmpeg.probe(str(input_video))
    except Exception as e:
        print(f"FFmpeg error during video probing : {e}")

    video_stream = next(
        (stream for stream in probe["streams"] if stream.get("codec_type") == "video"),
        None,
    )

    if video_stream is None:
        print(f"No video stream found in : {input_video}")

    frame_rate = video_stream.get("r_frame_rate")

    if not frame_rate:
        print(f"Frame rate not found for video : {input_video}")

    try:
        numerator, denominator = frame_rate.split("/")
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as e:
        print(
            f"Invalid frame rate returned by FFmpeg for video : {input_video}"
        )