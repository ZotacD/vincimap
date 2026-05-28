# Ce module calcule une echelle "gsplat units per meter" a partir de distances mesurees.
# Les mesures peuvent venir de n'importe quel outil si le fichier contient:
# - une colonne image, par exemple "image_id" ou "image";
# - une colonne angle, par exemple "angle" ou "angle(deg)";
# - une colonne distance, par exemple "distance" ou "distance(mm)".
#
# Principe general:
# 1. Lire le fichier de distances separe par tabulations.
# 2. Identifier les colonnes image, angle et distance, puis convertir les distances en metres.
# 3. Regrouper les mesures par image et agreger les doublons d'angle avec une mediane.
# 4. Pour chaque image, transformer les centres des splats du repere monde vers le repere camera.
# 5. Projeter les splats dans l'image et garder seulement ceux visibles dans le cadre.
# 6. Calculer, pour chaque splat visible, sa distance camera et ses angles horizontal/vertical.
# 7. Aligner les angles mesures avec la convention camera via offset configurable.
# 8. Associer chaque direction mesuree aux splats proches selon les tolerances angulaires.
# 9. Garder, pour chaque direction mesuree, le splat candidat le plus proche.
# 10. Calculer le ratio distance_gsplat / distance_mesuree_m pour obtenir un scale gsplat/m.
# 11. Ignorer les images avec trop peu de correspondances pour eviter un scale instable.
# 12. Agreger les scales par image et tous les ratios pour produire le scale global.
# 13. Ecrire le resultat dans un JSON par step et dans un JSON "latest".

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from configobj import ConfigObj
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor

def _load_distance_measurements(
    distances_path: str,
) -> Dict[str, List[Tuple[float, float]]]:
    # Lit le fichier de distances avec prise en charge des BOM UTF-8.
    with open(distances_path, "r", encoding="utf-8-sig", newline="") as f:
        # Construit un lecteur CSV indexe par nom de colonne.
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Distances file has no header: {distances_path}")

        # Retrouve les colonnes utiles meme si leurs noms varient legerement.
        angle_key = "angle"
        distance_key = "distance"
        image_key = "image_id"

        # Regroupe les mesures par image pour pouvoir les comparer aux cameras.
        grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in reader:
            # Ignore les lignes sans image associee.
            image_name = row[image_key].strip()
            if not image_name:
                continue

            # Convertit les distances en metres et garde seulement les valeurs valides.
            angle_deg = float(row[angle_key])
            distance = float(row[distance_key])
            distance_m = distance / 1000.0
            if distance_m > 0.0 and math.isfinite(distance_m):
                grouped[image_name].append((angle_deg, distance_m))

    return grouped


def _aggregate_distance_samples(
    samples: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    # Regroupe les mesures qui correspondent au meme angle.
    by_angle: Dict[float, List[float]] = defaultdict(list)
    for angle_deg, distance_m in samples:
        by_angle[round(angle_deg, 4)].append(distance_m)

    # Prend la mediane par angle pour reduire l'effet des doublons et valeurs extremes.
    angles = []
    distances = []
    for angle_deg in sorted(by_angle):
        angles.append(angle_deg)
        distances.append(float(np.median(by_angle[angle_deg])))

    # Retourne deux tableaux alignes: angles en degres, distances en metres.
    return np.asarray(angles, dtype=np.float32), np.asarray(distances, dtype=np.float32)


def _wrap_degrees_tensor(angles: Tensor) -> Tensor:
    # Ramene les angles dans l'intervalle [-180, 180[ pour comparer les directions.
    return torch.remainder(angles + 180.0, 360.0) - 180.0


class DistancesComputer:
    @classmethod
    def from_workspace_config(
        cls,
        *,
        data_dir: str,
        result_dir: str,
        stats_dir: str,
        world_rank: int,
        splats: torch.nn.ParameterDict,
        parser,
        near_plane: float,
    ) -> "DistancesComputer":
        config_path = Path(data_dir).parent / "configs" / "distances_computer.ini"
        config = ConfigObj(str(config_path), encoding="utf-8", list_values=False, write_empty_values=True)
        distances_config = config["distances_computer"]
        workspace_config = ConfigObj(
            str(config_path.parent / "workspace.ini"),
            encoding="utf-8",
            list_values=False,
            write_empty_values=True,
        )
        workspace = workspace_config["workspace"] if "workspace" in workspace_config else {}

        distances_path = distances_config.get("distances_path", "").strip()
        if not distances_path:
            distances_path = str(Path(data_dir) / "distances.txt")
        elif not Path(distances_path).is_absolute():
            repo_relative_path = Path.cwd() / distances_path
            workspace_relative_path = Path(data_dir).parent / distances_path
            if repo_relative_path.exists() or Path(distances_path).parts[:1] == ("workspaces",):
                distances_path = str(repo_relative_path)
            else:
                distances_path = str(workspace_relative_path)

        return cls(
            enabled=distances_config.as_bool("enabled") if "enabled" in distances_config else True,
            distances_path=distances_path,
            data_dir=data_dir,
            result_dir=result_dir,
            stats_dir=stats_dir,
            world_rank=world_rank,
            splats=splats,
            parser=parser,
            near_plane=near_plane,
            video_fov_deg=workspace.as_float("video_fov") if "video_fov" in workspace else 180.0,
            pivot_deg=distances_config.as_float("pivot_deg") if "pivot_deg" in distances_config else 0.0,
            angle_tolerance_deg=distances_config.as_float("angle_tolerance_deg") if "angle_tolerance_deg" in distances_config else 1.0,
            line_tolerance_deg=distances_config.as_float("line_tolerance_deg") if "line_tolerance_deg" in distances_config else 2.0,
            min_splat_matched_per_image=distances_config.as_int("min_splat_matched_per_image") if "min_splat_matched_per_image" in distances_config else 3,
            offset_x_m=distances_config.as_float("offset_x_m") if "offset_x_m" in distances_config else 0.0,
            offset_y_m=distances_config.as_float("offset_y_m") if "offset_y_m" in distances_config else 0.0,
            offset_z_m=distances_config.as_float("offset_z_m") if "offset_z_m" in distances_config else 0.0,
            angle_offset_deg=distances_config.as_float("angle_offset_deg") if "angle_offset_deg" in distances_config else 0.0,
        )

    def __init__(
        self,
        *,
        enabled: bool,
        distances_path: str,
        data_dir: str,
        result_dir: str,
        stats_dir: str,
        world_rank: int,
        splats: torch.nn.ParameterDict,
        parser,
        near_plane: float,
        video_fov_deg: float,
        pivot_deg: float,
        angle_tolerance_deg: float,
        line_tolerance_deg: float,
        min_splat_matched_per_image: int,
        offset_x_m: float,
        offset_y_m: float,
        offset_z_m: float,
        angle_offset_deg: float,
    ) -> None:
        self.enabled = enabled
        self.distances_path = distances_path
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.stats_dir = stats_dir
        self.world_rank = world_rank
        self.splats = splats
        self.parser = parser
        self.near_plane = near_plane
        self.video_fov_deg = video_fov_deg
        self.pivot_deg = pivot_deg
        self.angle_tolerance_deg = angle_tolerance_deg
        self.line_tolerance_deg = line_tolerance_deg
        self.min_splat_matched_per_image = min_splat_matched_per_image
        self.offset_x_m = offset_x_m
        self.offset_y_m = offset_y_m
        self.offset_z_m = offset_z_m
        self.angle_offset_deg = angle_offset_deg

    @torch.no_grad()
    def compute_scale(self, step: int) -> Optional[Dict]:
        """Compute an independent gsplat-units-per-meter scale from measured distances."""
        # Le calcul est optionnel et ne doit etre fait que par le process principal.
        if not self.enabled or self.world_rank != 0:
            return None

        if not os.path.exists(self.distances_path):
            print(
                f"[Distance scale] Distances file not found, skipping: {self.distances_path}"
            )
            return None

        # Valide les parametres de matching avant de lancer un calcul couteux.
        if self.angle_tolerance_deg <= 0.0:
            raise ValueError("angle_tolerance_deg must be > 0.")
        if self.line_tolerance_deg <= 0.0:
            raise ValueError("line_tolerance_deg must be > 0.")
        if self.video_fov_deg <= 0.0:
            raise ValueError("video_fov must be > 0.")

        print(f"[Distance scale] Computing gsplat/meter scale from {self.distances_path}")

        # Charge les mesures et prepare les centres des splats en coordonnees homogenes.
        self.current_step = step
        measurements_by_image = _load_distance_measurements(self.distances_path)
        means = self.splats["means"].detach()
        ones = torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype)
        means_h = torch.cat([means, ones], dim=-1)

        image_results, all_ratios = self._collect_matches(measurements_by_image, means_h)
        if not image_results:
            print("[Distance scale] No image had enough matched measurement directions.")
            return None

        payload = self._build_payload(step, image_results, all_ratios)
        self._write_payload(step, payload)

        print(
            "[Distance scale] "
            f"{payload['global_distance_scale_gsplat_per_meter']:.6f} gsplat/m "
            f"from {payload['num_images']} images and "
            f"{payload['num_matched_directions']} directions."
        )
        return payload

    def _collect_matches(
        self,
        measurements_by_image: Dict[str, List[Tuple[float, float]]],
        means_h: Tensor,
    ) -> Tuple[List[Dict], List[float]]:
        # Collecte les correspondances mesures/splats en une seule passe.
        image_results = []
        all_ratios = []
        means = means_h[:, :3]

        for image_index, image_name in enumerate(self.parser.image_names):
            # Recupere les mesures de l'image, avec fallback sur le basename.
            samples = measurements_by_image.get(image_name)
            if samples is None:
                samples = measurements_by_image.get(os.path.basename(image_name))
            if not samples:
                continue

            # Agrege les distances par angle avant le matching.
            measured_angles_np, measured_distances_m_np = _aggregate_distance_samples(
                samples
            )
            if measured_angles_np.size == 0:
                continue

            # Transforme les centres de splats du monde vers le repere camera.
            camtoworld = torch.from_numpy(self.parser.camtoworlds[image_index]).to(
                device=means.device, dtype=means.dtype
            )
            worldtocam = torch.linalg.inv(camtoworld)
            points_cam = (means_h @ worldtocam.T)[:, :3]

            visible_points_cam = self._visible_points_for_image(image_index, points_cam)
            if visible_points_cam is None:
                continue

            image_result, ratios = self._match_image_measurements(
                image_name,
                visible_points_cam,
                measured_angles_np,
                measured_distances_m_np,
            )
            if image_result is None:
                continue

            projected_line = self._projected_line_for_image(image_index)
            image_result["projected_line"] = projected_line
            preview_path = self._write_projected_line_preview(
                image_index, image_name, projected_line
            )
            if preview_path is not None:
                image_result["projected_line_preview_path"] = preview_path
            all_ratios.extend(float(value) for value in ratios)
            image_results.append(image_result)

        return image_results, all_ratios

    def _visible_points_for_image(
        self, image_index: int, points_cam: Tensor
    ) -> Optional[Tensor]:
        # Recupere les intrinseques pour projeter les splats dans l'image.
        camera_id = self.parser.camera_ids[image_index]
        K_np = self.parser.Ks_dict[camera_id]
        width, height = self.parser.imsize_dict[camera_id]
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])

        # Projette les points et ne garde que ceux visibles dans le frame.
        z = points_cam[:, 2]
        projected_x = fx * points_cam[:, 0] / z + cx
        projected_y = fy * points_cam[:, 1] / z + cy
        visible = (
            (z > self.near_plane)
            & (projected_x >= 0.0)
            & (projected_x < width)
            & (projected_y >= 0.0)
            & (projected_y < height)
        )
        if not torch.any(visible):
            return None
        return points_cam[visible]

    def _projected_line_for_image(self, image_index: int) -> Dict:
        camera_id = self.parser.camera_ids[image_index]
        K_np = self.parser.Ks_dict[camera_id]
        width, height = self.parser.imsize_dict[camera_id]
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        reference_fov_deg = self._camera_horizontal_fov(width, cx, fx)

        center_points = self._project_scan_line_points(
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            fixed_angle_deg=0.0,
            reference_fov_deg=reference_fov_deg,
        )
        lower_points = self._project_scan_line_points(
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            fixed_angle_deg=-self.line_tolerance_deg,
            reference_fov_deg=reference_fov_deg,
        )
        upper_points = self._project_scan_line_points(
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            fixed_angle_deg=self.line_tolerance_deg,
            reference_fov_deg=reference_fov_deg,
        )
        degree_scale = self._project_degree_scale(
            width, height, fx, fy, cx, cy, reference_fov_deg
        )

        return {
            "points": center_points,
            "tolerance_lines": {
                "negative": lower_points,
                "positive": upper_points,
            },
            "center": {
                "x": cx,
                "y": cy,
            },
            "line_tolerance_deg": self.line_tolerance_deg,
            "video_fov_deg": self.video_fov_deg,
            "reference_fov_deg": reference_fov_deg,
            "degree_scale": degree_scale,
        }

    @staticmethod
    def _camera_horizontal_fov(width: int, cx: float, fx: float) -> float:
        half_width = min(cx, float(width) - cx)
        return math.degrees(2.0 * math.atan(half_width / fx))

    def _project_scan_line_points(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        *,
        fixed_angle_deg: float,
        reference_fov_deg: float,
    ) -> List[Dict]:
        pivot = math.radians(self.pivot_deg)
        scan_angles_deg = np.linspace(-89.0, 89.0, 2001, dtype=np.float64)
        horizontal_deg = (
            scan_angles_deg * math.cos(pivot) - fixed_angle_deg * math.sin(pivot)
        )
        vertical_deg = (
            scan_angles_deg * math.sin(pivot) + fixed_angle_deg * math.cos(pivot)
        )
        horizontal_angles = np.deg2rad(horizontal_deg)
        vertical_angles = np.deg2rad(vertical_deg)

        x = np.tan(horizontal_angles)
        y = np.tan(vertical_angles) * np.sqrt((x * x) + 1.0)
        projected_x = (fx * x) + cx
        projected_y = (fy * y) + cy

        inside = (
            (projected_x >= 0.0)
            & (projected_x < width)
            & (projected_y >= 0.0)
            & (projected_y < height)
        )
        indices = np.flatnonzero(inside)
        if indices.size < 2:
            return []

        p0 = np.asarray([projected_x[indices[0]], projected_y[indices[0]]])
        p1 = np.asarray([projected_x[indices[-1]], projected_y[indices[-1]]])
        center = self._project_scan_line_center(fx, fy, cx, cy, fixed_angle_deg)
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return []

        direction = direction / length
        half_length = min(
            float(np.linalg.norm(center - p0)),
            float(np.linalg.norm(center - p1)),
        )
        fov_ratio = min(self.video_fov_deg / reference_fov_deg, 1.0)
        half_length *= fov_ratio
        cropped_p0 = center - direction * half_length
        cropped_p1 = center + direction * half_length

        return [
            {
                "x": float(cropped_p0[0]),
                "y": float(cropped_p0[1]),
            },
            {
                "x": float(cropped_p1[0]),
                "y": float(cropped_p1[1]),
            },
        ]

    def _project_scan_line_center(
        self, fx: float, fy: float, cx: float, cy: float, fixed_angle_deg: float
    ) -> np.ndarray:
        pivot = math.radians(self.pivot_deg)
        horizontal_deg = -fixed_angle_deg * math.sin(pivot)
        vertical_deg = fixed_angle_deg * math.cos(pivot)
        horizontal_angle = math.radians(horizontal_deg)
        vertical_angle = math.radians(vertical_deg)
        x = math.tan(horizontal_angle)
        y = math.tan(vertical_angle) * math.sqrt((x * x) + 1.0)
        return np.asarray([(fx * x) + cx, (fy * y) + cy], dtype=np.float64)

    def _project_degree_scale(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        reference_fov_deg: float,
    ) -> Dict:
        vertical_reference_fov_deg = self._camera_vertical_fov(height, cy, fy)
        fov_ratio = min(self.video_fov_deg / reference_fov_deg, 1.0)
        horizontal_limit_deg = min(self.video_fov_deg, reference_fov_deg) * 0.5
        vertical_limit_deg = vertical_reference_fov_deg * fov_ratio * 0.5

        x_ticks = []
        for angle_deg in self._degree_tick_values(horizontal_limit_deg):
            x = (fx * math.tan(math.radians(angle_deg))) + cx
            if 0.0 <= x < width:
                x_ticks.append({"degree": angle_deg, "x": float(x)})

        y_ticks = []
        for angle_deg in self._degree_tick_values(vertical_limit_deg):
            y = (fy * math.tan(math.radians(angle_deg))) + cy
            if 0.0 <= y < height:
                y_ticks.append({"degree": angle_deg, "y": float(y)})

        return {
            "x": x_ticks,
            "y": y_ticks,
            "horizontal_limit_deg": horizontal_limit_deg,
            "vertical_limit_deg": vertical_limit_deg,
        }

    @staticmethod
    def _camera_vertical_fov(height: int, cy: float, fy: float) -> float:
        half_height = min(cy, float(height) - cy)
        return math.degrees(2.0 * math.atan(half_height / fy))

    @staticmethod
    def _degree_tick_values(limit_deg: float) -> List[float]:
        if limit_deg <= 0.0:
            return [0.0]
        step = 5.0 if limit_deg <= 30.0 else 10.0
        start = math.ceil(-limit_deg / step) * step
        end = math.floor(limit_deg / step) * step
        values = []
        current = start
        while current <= end + 1e-6:
            values.append(0.0 if abs(current) < 1e-6 else current)
            current += step
        return values

    def _write_projected_line_preview(
        self, image_index: int, image_name: str, projected_line: Dict
    ) -> Optional[str]:
        image_path = Path(self.data_dir) / "images" / os.path.basename(image_name)
        if not image_path.exists():
            return None

        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            return None
        draw = ImageDraw.Draw(image)

        for points in projected_line.get("tolerance_lines", {}).values():
            self._draw_projected_line(draw, points, color=(255, 255, 0), thickness=2)
        self._draw_projected_line(
            draw, projected_line.get("points", []), color=(255, 0, 0), thickness=3
        )
        self._draw_degree_scale(draw, image.size, projected_line.get("degree_scale"))

        center = projected_line.get("center")
        if center:
            x, y = int(round(center["x"])), int(round(center["y"]))
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 255, 255))
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 0, 0), width=1)

        output_dir = (
            Path(self.result_dir)
            / "distance_projected_lines"
            / f"step_{self.current_step:04d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / os.path.basename(image_name)
        image.save(output_path, quality=95)
        return str(output_path)

    @staticmethod
    def _draw_projected_line(
        draw: ImageDraw.ImageDraw,
        points: List[Dict],
        *,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        if len(points) != 2:
            return
        p0 = (int(round(points[0]["x"])), int(round(points[0]["y"])))
        p1 = (int(round(points[1]["x"])), int(round(points[1]["y"])))
        draw.line((p0, p1), fill=color, width=thickness)

    @staticmethod
    def _draw_degree_scale(
        draw: ImageDraw.ImageDraw, image_size: Tuple[int, int], degree_scale: Optional[Dict]
    ) -> None:
        if not degree_scale:
            return

        width, height = image_size
        x_axis_y = height - 34
        y_axis_x = 34
        color = (30, 30, 30)
        label_color = (255, 255, 255)
        label_bg = (0, 0, 0)

        draw.line((0, x_axis_y, width, x_axis_y), fill=color, width=1)
        for tick in degree_scale.get("x", []):
            x = int(round(tick["x"]))
            degree = int(round(tick["degree"]))
            draw.line((x, x_axis_y - 8, x, x_axis_y + 8), fill=color, width=2)
            label = f"{degree:+d}deg"
            label_pos = (x - 18, x_axis_y + 10)
            draw.rectangle(
                (label_pos[0] - 2, label_pos[1] - 1, label_pos[0] + 40, label_pos[1] + 12),
                fill=label_bg,
            )
            draw.text(label_pos, label, fill=label_color)

        draw.line((y_axis_x, 0, y_axis_x, height), fill=color, width=1)
        for tick in degree_scale.get("y", []):
            y = int(round(tick["y"]))
            degree = int(round(tick["degree"]))
            draw.line((y_axis_x - 8, y, y_axis_x + 8, y), fill=color, width=2)
            label = f"{degree:+d}deg"
            label_pos = (y_axis_x + 12, y - 6)
            draw.rectangle(
                (label_pos[0] - 2, label_pos[1] - 1, label_pos[0] + 40, label_pos[1] + 12),
                fill=label_bg,
            )
            draw.text(label_pos, label, fill=label_color)

    def _match_image_measurements(
        self,
        image_name: str,
        visible_points_cam: Tensor,
        measured_angles_np: np.ndarray,
        measured_distances_m_np: np.ndarray,
    ) -> Tuple[Optional[Dict], np.ndarray]:
        gs_distances = torch.linalg.norm(visible_points_cam, dim=-1)
        # Convertit les mesures en tenseurs sur le meme device que les splats.
        measured_angles = torch.from_numpy(measured_angles_np).to(
            device=visible_points_cam.device, dtype=visible_points_cam.dtype
        )
        measured_distances_m = torch.from_numpy(measured_distances_m_np).to(
            device=visible_points_cam.device, dtype=visible_points_cam.dtype
        )

        # Applique l'offset d'angle pour aligner les mesures avec la camera.
        measured_angles_camera = _wrap_degrees_tensor(
            measured_angles + self.angle_offset_deg
        )
        pivot = math.radians(self.pivot_deg)
        scan_weight = math.cos(pivot)
        vertical_scan_weight = math.sin(pivot)

        measured_scan_angles = measured_angles_camera
        measured_fixed_angle = 0.0

        offset = self._distance_offset(
            device=visible_points_cam.device, dtype=visible_points_cam.dtype
        )
        if torch.all(offset == 0.0):
            # Sans offset, les directions mesurees partent du centre optique camera.
            horizontal_angles, vertical_angles = self._camera_angles(visible_points_cam)
            splat_scan_angles, splat_fixed_angles = self._scan_angles(
                horizontal_angles, vertical_angles, scan_weight, vertical_scan_weight
            )
            scan_diff = torch.abs(
                _wrap_degrees_tensor(
                    splat_scan_angles[:, None] - measured_scan_angles[None, :]
                )
            )
            fixed_diff = torch.abs(
                _wrap_degrees_tensor(splat_fixed_angles - measured_fixed_angle)
            )
            candidates = (
                (scan_diff <= self.angle_tolerance_deg)
                & (fixed_diff[:, None] <= self.line_tolerance_deg)
            )
            candidate_scores = gs_distances[:, None]
        else:
            # Avec offset, l'angle mesure part du capteur decale. Comme le scale
            # est inconnu, on resout la position metrique compatible avec chaque
            # distance mesuree avant de comparer les directions.
            ray_scales = self._ray_scales(
                visible_points_cam, measured_distances_m, offset
            )
            sensor_x = ray_scales * visible_points_cam[:, 0, None] - offset[0]
            sensor_y = ray_scales * visible_points_cam[:, 1, None] - offset[1]
            sensor_z = ray_scales * visible_points_cam[:, 2, None] - offset[2]
            horizontal_angles, vertical_angles = self._camera_angles_from_components(
                sensor_x, sensor_y, sensor_z
            )
            splat_scan_angles, splat_fixed_angles = self._scan_angles(
                horizontal_angles, vertical_angles, scan_weight, vertical_scan_weight
            )
            scan_diff = torch.abs(
                _wrap_degrees_tensor(splat_scan_angles - measured_scan_angles[None, :])
            )
            fixed_diff = torch.abs(
                _wrap_degrees_tensor(splat_fixed_angles - measured_fixed_angle)
            )
            candidates = (
                torch.isfinite(ray_scales)
                & (scan_diff <= self.angle_tolerance_deg)
                & (fixed_diff <= self.line_tolerance_deg)
            )
            candidate_scores = ray_scales * gs_distances[:, None]

        # Pour chaque direction mesuree, garde le splat candidat le plus proche.
        inf = torch.full_like(scan_diff, float("inf"))
        candidate_distances = torch.where(candidates, candidate_scores, inf)
        matched = candidate_distances.min(dim=0)
        matched_indices = matched.indices

        # Elimine les directions sans match et les distances mesurees invalides.
        valid = torch.isfinite(matched.values) & (measured_distances_m > 0.0)
        if not torch.any(valid):
            return None, np.asarray([], dtype=np.float64)

        # Le ratio distance gsplat / distance metre donne un scale gsplat par metre.
        matched_points = visible_points_cam[matched_indices[valid]]
        ratios_tensor = self._scale_ratios(matched_points, measured_distances_m[valid])
        valid_ratios = torch.isfinite(ratios_tensor) & (ratios_tensor > 0.0)
        if not torch.any(valid_ratios):
            return None, np.asarray([], dtype=np.float64)

        ratios = ratios_tensor[valid_ratios].cpu().numpy()
        measured_distances_valid = measured_distances_m[valid][valid_ratios].cpu().numpy()
        gs_distances_valid = gs_distances[matched_indices[valid]][valid_ratios].cpu().numpy()
        matched_angles = measured_angles_np[valid.cpu().numpy()][
            valid_ratios.cpu().numpy()
        ]

        # Ignore les images avec trop peu de directions matchees pour etre robustes.
        if ratios.size < self.min_splat_matched_per_image:
            return None, ratios

        return (
            {
                "image_id": image_name,
                "matched_directions": int(ratios.size),
                "scale_gsplat_per_meter": float(np.median(ratios)),
                "mean_scale_gsplat_per_meter": float(np.mean(ratios)),
                "std_scale_gsplat_per_meter": float(np.std(ratios)),
                "median_measured_distance_m": float(np.median(measured_distances_valid)),
                "median_gsplat_distance": float(np.median(gs_distances_valid)),
                "matched_measurement_angles_deg": [
                    float(angle) for angle in matched_angles.tolist()
                ],
            },
            ratios,
        )

    def _distance_offset(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.tensor(
            [self.offset_x_m, self.offset_y_m, self.offset_z_m],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _camera_angles(points_cam: Tensor) -> Tuple[Tensor, Tensor]:
        return DistancesComputer._camera_angles_from_components(
            points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
        )

    @staticmethod
    def _camera_angles_from_components(
        x: Tensor, y: Tensor, z: Tensor
    ) -> Tuple[Tensor, Tensor]:
        horizontal_angles = torch.rad2deg(torch.atan2(x, z))
        vertical_angles = torch.rad2deg(torch.atan2(y, torch.sqrt(x * x + z * z)))
        return horizontal_angles, vertical_angles

    @staticmethod
    def _scan_angles(
        horizontal_angles: Tensor,
        vertical_angles: Tensor,
        scan_weight: float,
        vertical_scan_weight: float,
    ) -> Tuple[Tensor, Tensor]:
        scan_angles = (
            horizontal_angles * scan_weight + vertical_angles * vertical_scan_weight
        )
        fixed_angles = (
            -horizontal_angles * vertical_scan_weight + vertical_angles * scan_weight
        )
        return scan_angles, fixed_angles

    @staticmethod
    def _ray_scales(
        points_cam: Tensor,
        measured_distances_m: Tensor,
        offset: Tensor,
    ) -> Tensor:
        a = torch.sum(points_cam * points_cam, dim=-1)
        b = -2.0 * torch.sum(points_cam * offset[None, :], dim=-1)
        c = torch.sum(offset * offset) - measured_distances_m * measured_distances_m
        discriminant = b[:, None] * b[:, None] - 4.0 * a[:, None] * c[None, :]
        valid = (a[:, None] > 0.0) & (discriminant >= 0.0)
        safe_discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_discriminant = torch.sqrt(safe_discriminant)
        root_1 = (-b[:, None] + sqrt_discriminant) / (2.0 * a[:, None])
        root_2 = (-b[:, None] - sqrt_discriminant) / (2.0 * a[:, None])
        ray_scale = torch.where(root_1 > 0.0, root_1, root_2)
        return torch.where(
            valid & (ray_scale > 0.0),
            ray_scale,
            torch.full_like(ray_scale, float("nan")),
        )

    @staticmethod
    def _ray_scales_for_pairs(
        points_cam: Tensor,
        measured_distances_m: Tensor,
        offset: Tensor,
    ) -> Tensor:
        a = torch.sum(points_cam * points_cam, dim=-1)
        b = -2.0 * torch.sum(points_cam * offset[None, :], dim=-1)
        c = torch.sum(offset * offset) - measured_distances_m * measured_distances_m
        discriminant = b * b - 4.0 * a * c
        valid = (a > 0.0) & (discriminant >= 0.0)
        safe_discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_discriminant = torch.sqrt(safe_discriminant)
        root_1 = (-b + sqrt_discriminant) / (2.0 * a)
        root_2 = (-b - sqrt_discriminant) / (2.0 * a)
        ray_scale = torch.where(root_1 > 0.0, root_1, root_2)
        return torch.where(
            valid & (ray_scale > 0.0),
            ray_scale,
            torch.full_like(ray_scale, float("nan")),
        )

    def _scale_ratios(self, points_cam: Tensor, measured_distances_m: Tensor) -> Tensor:
        offset = self._distance_offset(device=points_cam.device, dtype=points_cam.dtype)
        if torch.all(offset == 0.0):
            return torch.linalg.norm(points_cam, dim=-1) / measured_distances_m

        ray_scale = self._ray_scales_for_pairs(
            points_cam, measured_distances_m, offset
        )
        return torch.where(
            torch.isfinite(ray_scale),
            1.0 / ray_scale,
            torch.full_like(ray_scale, float("nan")),
        )

    def _build_payload(
        self, step: int, image_results: List[Dict], all_ratios: List[float]
    ) -> Dict:
        # Agrege les resultats finaux sur les images et sur toutes les directions matchees.
        image_scales = np.asarray(
            [result["scale_gsplat_per_meter"] for result in image_results],
            dtype=np.float64,
        )
        all_ratios_np = np.asarray(all_ratios, dtype=np.float64)

        # Prepare le JSON de sortie avec le scale global, les reglages et le detail image.
        return {
            "step": int(step),
            "unit": "gsplat_units_per_meter",
            "global_distance_scale_gsplat_per_meter": float(np.median(image_scales)),
            "global_distance_scale_all_directions_median": float(
                np.median(all_ratios_np)
            ),
            "mean_image_scale_gsplat_per_meter": float(np.mean(image_scales)),
            "std_image_scale_gsplat_per_meter": float(np.std(image_scales)),
            "num_images": len(image_results),
            "num_matched_directions": int(all_ratios_np.size),
            "distances_path": self.distances_path,
            "matching": {
                "pivot_deg": self.pivot_deg,
                "angle_tolerance_deg": self.angle_tolerance_deg,
                "line_tolerance_deg": self.line_tolerance_deg,
                "video_fov_deg": self.video_fov_deg,
                "min_splat_matched_per_image": self.min_splat_matched_per_image,
                "offset_x_m": self.offset_x_m,
                "offset_y_m": self.offset_y_m,
                "offset_z_m": self.offset_z_m,
                "angle_offset_deg": self.angle_offset_deg,
            },
            "images": image_results,
        }

    def _write_payload(self, step: int, payload: Dict) -> None:
        # Ecrit un fichier par step et un fichier latest facilement consommable.
        step_path = f"{self.stats_dir}/distance_scale_step{step:04d}.json"
        latest_path = f"{self.result_dir}/distance_scale.json"
        for output_path in [step_path, latest_path]:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
