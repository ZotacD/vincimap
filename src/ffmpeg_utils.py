import subprocess
import ffmpeg
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

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
        raise

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

    input_images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
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
        except ffmpeg.Error as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            print(
                f"FFmpeg error during resizing {input_image.name} :\n{stderr}"
            )
            raise
        
"""
Cree des masques COLMAP a partir des images accessibles via ``images_path``.

Par defaut, chaque masque est entierement blanc. Les rectangles optionnels
dans ``black_boxes`` sont remplis en noir pour exclure ces zones.

COLMAP attend un fichier ``nom_image.ext.png`` pour l'image ``nom_image.ext``.
"""
def images_to_masks(
    images_path: str,
    masks_path: str,
    black_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    input_dir = Path(images_path)
    output_dir = Path(masks_path)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found : {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not input_images:
        raise FileNotFoundError(f"No images found in : {input_dir}")

    for input_image in input_images:
        output_mask = output_dir / f"{input_image.name}.png"

        try:
            stream = (
                ffmpeg
                .input(str(input_image))
                .filter("format", "gray")
                .filter("lut", y=255)
            )

            for x, y, width, height in black_boxes or []:
                stream = stream.filter(
                    "drawbox",
                    x=x,
                    y=y,
                    w=width,
                    h=height,
                    color="black",
                    t="fill",
                )

            (
                stream
                .output(str(output_mask), vframes=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            print(
                f"FFmpeg error during mask creation {input_image.name} :\n{stderr}"
            )
            raise


"""
Retourne la taille d'une image sous la forme ``(width, height)``.
"""
def image_size(image_path: str) -> tuple[int, int]:
    input_image = Path(image_path)

    if not input_image.is_file():
        raise FileNotFoundError(f"Image not found : {input_image}")

    try:
        probe = ffmpeg.probe(str(input_image))
    except ffmpeg.Error as e:
        print(f"FFmpeg error during image probing : {e}")
        raise

    image_stream = next(
        (stream for stream in probe["streams"] if stream.get("codec_type") == "video"),
        None,
    )

    if image_stream is None:
        raise ValueError(f"No image stream found in : {input_image}")

    return int(image_stream["width"]), int(image_stream["height"])


"""
Retourne le frame rate de la video
accessible via ``video_path``.
"""
def frame_rate_from_video(video_path: str) -> float:
    input_video = Path(video_path)

    if not input_video.is_file():
        raise FileNotFoundError(f"Video not found : {input_video}")

    try:
        probe = ffmpeg.probe(str(input_video))
    except ffmpeg.Error as e:
        print(f"FFmpeg error during video probing : {e}")
        raise

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
            "Invalid frame rate returned by FFmpeg for video : "
            f"{input_video} ({e})"
        )
        raise

"""
Retourne la durée de la vidéo en secondes.
"""
def duration_from_video(video_path: str) -> float:
    input_video = Path(video_path)

    if not input_video.is_file():
        raise FileNotFoundError(f"Video not found : {input_video}")

    try:
        probe = ffmpeg.probe(str(input_video))
    except ffmpeg.Error as e:
        print(f"FFmpeg error during video probing : {e}")
        raise

    duration = probe.get("format", {}).get("duration")

    if duration is None:
        video_stream = next(
            (
                stream for stream in probe.get("streams", [])
                if stream.get("codec_type") == "video"
            ),
            None,
        )

        if video_stream is not None:
            duration = video_stream.get("duration")

    if duration is None:
        raise ValueError(f"Duration not found for video : {input_video}")

    try:
        return float(duration)
    except ValueError as e:
        print(
            "Invalid duration returned by FFmpeg for video : "
            f"{input_video} ({e})"
        )
        raise
