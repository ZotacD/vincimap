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
# 7. Aligner les angles mesures avec la convention camera via signe et offset configurables.
# 8. Associer chaque direction mesuree aux splats proches selon les tolerances angulaires.
# 9. Garder, pour chaque direction mesuree, le splat candidat le plus proche.
# 10. Calculer le ratio distance_gsplat / distance_mesuree_m pour obtenir un scale gsplat/m.
# 11. Ignorer les images avec trop peu de correspondances pour eviter un scale instable.
# 12. Agreger les scales par image et tous les ratios pour produire le scale global.
# 13. Ecrire le resultat dans un JSON par step et dans un JSON "latest".

import csv
import configparser
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def _first_matching_column(fieldnames: List[str], candidates: List[str]) -> str:
    # Normalise les noms de colonnes pour accepter des variations de casse/espaces.
    normalized = {name.strip().lower(): name for name in fieldnames}

    # Cherche d'abord une correspondance exacte avec les noms attendus.
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    # Sinon, accepte une colonne qui contient le mot attendu dans son nom.
    for name in fieldnames:
        lowered = name.strip().lower()
        if any(candidate in lowered for candidate in candidates):
            return name

    raise ValueError(f"Missing one of columns {candidates} in distances file.")


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
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        distances_config = config["distances_computer"]

        distances_path = distances_config.get("distances_path", fallback="").strip()
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
            enabled=distances_config.getboolean("enabled", fallback=True),
            distances_path=distances_path,
            result_dir=result_dir,
            stats_dir=stats_dir,
            world_rank=world_rank,
            splats=splats,
            parser=parser,
            near_plane=near_plane,
            angle_offset_deg=distances_config.getfloat("angle_offset_deg", fallback=0.0),
            angle_sign=distances_config.getint("angle_sign", fallback=1),
            angle_tolerance_deg=distances_config.getfloat(
                "angle_tolerance_deg",
                fallback=0.5,
            ),
            vertical_angle_deg=distances_config.getfloat("vertical_angle_deg", fallback=0.0),
            vertical_tolerance_deg=distances_config.getfloat(
                "vertical_tolerance_deg",
                fallback=1.0,
            ),
            min_matches_per_image=distances_config.getint(
                "min_matches_per_image",
                fallback=3,
            ),
        )

    def __init__(
        self,
        *,
        enabled: bool,
        distances_path: str,
        result_dir: str,
        stats_dir: str,
        world_rank: int,
        splats: torch.nn.ParameterDict,
        parser,
        near_plane: float,
        angle_offset_deg: float,
        angle_sign: int,
        angle_tolerance_deg: float,
        vertical_angle_deg: float,
        vertical_tolerance_deg: float,
        min_matches_per_image: int,
    ) -> None:
        self.enabled = enabled
        self.distances_path = distances_path
        self.result_dir = result_dir
        self.stats_dir = stats_dir
        self.world_rank = world_rank
        self.splats = splats
        self.parser = parser
        self.near_plane = near_plane
        self.angle_offset_deg = angle_offset_deg
        self.angle_sign = -1.0 if angle_sign < 0 else 1.0
        self.angle_tolerance_deg = angle_tolerance_deg
        self.vertical_angle_deg = vertical_angle_deg
        self.vertical_tolerance_deg = vertical_tolerance_deg
        self.min_matches_per_image = min_matches_per_image

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
        if self.vertical_tolerance_deg <= 0.0:
            raise ValueError("vertical_tolerance_deg must be > 0.")

        print(f"[Distance scale] Computing gsplat/meter scale from {self.distances_path}")

        # Charge les mesures et prepare les centres des splats en coordonnees homogenes.
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

    def _match_image_measurements(
        self,
        image_name: str,
        visible_points_cam: Tensor,
        measured_angles_np: np.ndarray,
        measured_distances_m_np: np.ndarray,
    ) -> Tuple[Optional[Dict], np.ndarray]:
        # Calcule distances et angles depuis le centre camera.
        gs_distances = torch.linalg.norm(visible_points_cam, dim=-1)
        horizontal_angles = torch.rad2deg(
            torch.atan2(visible_points_cam[:, 0], visible_points_cam[:, 2])
        )
        vertical_angles = torch.rad2deg(
            torch.atan2(
                visible_points_cam[:, 1],
                torch.linalg.norm(visible_points_cam[:, [0, 2]], dim=-1),
            )
        )

        # Convertit les mesures en tenseurs sur le meme device que les splats.
        measured_angles = torch.from_numpy(measured_angles_np).to(
            device=visible_points_cam.device, dtype=visible_points_cam.dtype
        )
        measured_distances_m = torch.from_numpy(measured_distances_m_np).to(
            device=visible_points_cam.device, dtype=visible_points_cam.dtype
        )

        # Applique l'offset/sign d'angle pour aligner les mesures avec la camera.
        measured_angles_camera = _wrap_degrees_tensor(
            self.angle_sign * (measured_angles + self.angle_offset_deg)
        )

        # Calcule les ecarts angulaires entre chaque splat visible et chaque direction mesuree.
        horizontal_diff = torch.abs(
            _wrap_degrees_tensor(
                horizontal_angles[:, None] - measured_angles_camera[None, :]
            )
        )
        vertical_diff = torch.abs(vertical_angles - float(self.vertical_angle_deg))
        candidates = (
            (horizontal_diff <= self.angle_tolerance_deg)
            & (vertical_diff[:, None] <= self.vertical_tolerance_deg)
        )

        # Pour chaque direction mesuree, garde le splat candidat le plus proche.
        inf = torch.full_like(horizontal_diff, float("inf"))
        matched_distances = torch.where(candidates, gs_distances[:, None], inf).min(
            dim=0
        ).values

        # Elimine les directions sans match et les distances mesurees invalides.
        valid = torch.isfinite(matched_distances) & (measured_distances_m > 0.0)
        if not torch.any(valid):
            return None, np.asarray([], dtype=np.float64)

        # Le ratio distance gsplat / distance metre donne un scale gsplat par metre.
        ratios = (matched_distances[valid] / measured_distances_m[valid]).cpu().numpy()
        measured_distances_valid = measured_distances_m[valid].cpu().numpy()
        gs_distances_valid = matched_distances[valid].cpu().numpy()
        matched_angles = measured_angles_np[valid.cpu().numpy()]

        # Ignore les images avec trop peu de directions matchees pour etre robustes.
        if ratios.size < self.min_matches_per_image:
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
                "angle_offset_deg": self.angle_offset_deg,
                "angle_sign": -1 if self.angle_sign < 0 else 1,
                "angle_tolerance_deg": self.angle_tolerance_deg,
                "vertical_angle_deg": self.vertical_angle_deg,
                "vertical_tolerance_deg": self.vertical_tolerance_deg,
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
