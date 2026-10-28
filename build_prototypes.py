"""
ParticleVision V4
=================

Prototype Ingest

Turns 10 hand-sorted folders of confident example crops
(one folder per DEFECT_TAXONOMY class, ~10-15 single-particle images
each) into the labeled prototype set classification.py trains/matches
against. This is the ONLY manual-labeling step in v4, a one-time
sitting, not an ongoing review queue.

No SAM2 needed here (deliberately, to keep this script self-contained
and fast): each crop's particle mask is found with a lightweight
Otsu-threshold + largest-central-component heuristic, which works well
on SEM's typical bright-particle-on-dark-background contrast for a
crop the user has already centered on one particle. DINOv2 embeddings
still use the SAME extractor/preprocessing as the main pipeline (via
dinov2_features.py), so prototype embeddings are directly comparable
to production particle embeddings at inference time.

microns_per_pixel is a single value applied to the WHOLE prototype
set, these crops should come from the same magnification/imaging
conditions as your main dataset.
"""

from __future__ import annotations

import os

# Must be set before torch/xgboost are imported anywhere in the
# process.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse

import cv2
import joblib
import numpy as np
import pandas as pd

from data_models import BoundingBox, Centroid, Particle
from morphology import MorphologyExtractor, apply_calibration
from physics_formulas import PhysicsDescriptorExtractor, compute_named_physics_formulas
from physics_formulas_v2 import compute_v2_features
from dinov2_features import crop_particle_patch
from attachment import compute_attachments
from dinov2_features import build_dinov2_extractor
from segmentation import split_touching_particles
from utils import list_images, load_rgb, logger
import config


PROTOTYPES_FILENAME = "prototypes.joblib"
PROTOTYPES_SUMMARY_FILENAME = "prototypes_summary.csv"


def _seed_everything(seed: int = 42):
    """Duplicated from main.py (not imported from it) to avoid a circular
    import — main.py itself imports from this module at load time."""

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

_FOLDER_NAME_ALIASES = {
    "perfect_sphere": "Perfect_Spherical",
    "oversized_sphere": "Oversized_Spherical",
    "bonded_satellite": "Molten_Satellite",
}


def _normalize_folder_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_class_folder_name(folder_name: str):
    """Returns the matching config.DEFECT_TAXONOMY class name for a
    given folder name, or None if it doesn't match any class."""

    normalized = _normalize_folder_name(folder_name)

    if normalized in _FOLDER_NAME_ALIASES:
        return _FOLDER_NAME_ALIASES[normalized]

    for class_name in config.DEFECT_TAXONOMY:
        if _normalize_folder_name(class_name) == normalized:
            return class_name

    return None


# ============================================================
# ROUGH PARTICLE MASK(S)
# ============================================================

def find_particle_components(image_rgb: np.ndarray, max_components: int = 4) -> list[np.ndarray]:

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask_bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = image_rgb.shape[:2]
    min_area = 0.01 * h * w

    best_components: list[np.ndarray] = []
    best_total_area = None

    for candidate_binary in (mask_bright, mask_dark):

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_binary.astype(np.uint8), connectivity=8
        )

        raw_components = []

        for i in range(1, n_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            raw_components.append((area, labels == i))

        if not raw_components:
            continue

        raw_components.sort(key=lambda t: t[0], reverse=True)
        raw_components = raw_components[:max_components]

        split_components = []

        for _, comp_mask in raw_components:
        
            split_components.extend(
                split_touching_particles(
                    comp_mask, min_distance=6, min_component_area=20,
                    min_peak_prominence_ratio=0.15,
                )
            )

        total_area = sum(np.count_nonzero(m) for m in split_components)

        # Prefer the polarity whose kept components cover LESS of the
        # frame, that's the particle(s), not the background.
        if best_total_area is None or total_area < best_total_area:
            best_total_area = total_area
            best_components = split_components

    return best_components


def _make_particle_from_mask(mask: np.ndarray, particle_id: int, image_name: str) -> Particle | None:

    mask_u8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)

    if M["m00"] != 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2, y + h / 2

    return Particle(
        particle_id=particle_id,
        image_name=image_name,
        mask=mask.astype(bool),
        contour=contour,
        bbox=BoundingBox(x=x, y=y, width=w, height=h),
        centroid=Centroid(x=cx, y=cy),
    )


# ============================================================
# BUILD
# ============================================================

def build_prototype_set(
    examples_dir: str,
    output_dir: str,
    microns_per_pixel: float,
    device: str = "cpu",
    dinov2_model: str = None,
) -> pd.DataFrame:

    _seed_everything()

    os.makedirs(output_dir, exist_ok=True)

    class_names = set(config.DEFECT_TAXONOMY)

    raw_dirs = [
        d for d in os.listdir(examples_dir)
        if os.path.isdir(os.path.join(examples_dir, d))
    ]

    found_dirs: dict[str, list[str]] = {}
    unknown = []

    for d in raw_dirs:

        resolved = resolve_class_folder_name(d)

        if resolved is None:
            unknown.append(d)
        else:
            found_dirs.setdefault(resolved, []).append(os.path.join(examples_dir, d))

    if unknown:
        logger.warning(
            f"Folder name(s) don't match any class, skipped: {sorted(unknown)} "
            f"— expected one of {sorted(class_names)}"
        )

    missing = class_names - set(found_dirs)
    if missing:
        logger.warning(f"No folder found for: {sorted(missing)} — those classes will have zero prototypes.")

    # dinov2_model is a real parameter, not a module-constant patch:
    # build_dinov2_extractor binds its DEFAULT_MODEL default at import
    # time, so reassigning dinov2_features.DEFAULT_MODEL afterward has
    # NO effect.
    dinov2_extractor = (
        build_dinov2_extractor(device=device, model_name=dinov2_model)
        if dinov2_model
        else build_dinov2_extractor(device=device)
    )
    morphology_extractor = MorphologyExtractor()
    physics_extractor = PhysicsDescriptorExtractor()

    records = []
    _crop_cache = {}
    particle_id = 1

    for class_name in config.DEFECT_TAXONOMY:

        folders = found_dirs.get(class_name)

        if not folders:
            continue

        image_paths = [p for folder in folders for p in list_images(folder)]

        if len(folders) > 1:
            logger.info(f"{class_name}: merging {len(image_paths)} example(s) from {len(folders)} folders.")

        if len(image_paths) < config.CLASSIFICATION.min_prototype_examples:
            logger.warning(
                f"{class_name}: only {len(image_paths)} example(s), below the "
                f"recommended minimum ({config.CLASSIFICATION.min_prototype_examples}) — "
                "this class's prototypes will be less reliable."
            )

        for image_path in image_paths:

            image_rgb = load_rgb(str(image_path))

            masks = find_particle_components(image_rgb)

            if not masks:
                logger.warning(f"Could not isolate a particle in {image_path}, skipped.")
                continue

            image_basename = os.path.basename(str(image_path))
            crop_particles = []

            for mask in masks:

                p = _make_particle_from_mask(mask, particle_id, image_basename)

                if p is not None:
                    crop_particles.append(p)
                    particle_id += 1

            if not crop_particles:
                logger.warning(f"No usable contour in {image_path}, skipped.")
                continue

            for p in crop_particles:
                morphology_extractor.extract(p)
                apply_calibration(p, microns_per_pixel)
                physics_extractor.extract(p)
                compute_named_physics_formulas(p, config.PHYSICS)

            h, w = image_rgb.shape[:2]
            crop_area = float(h * w)
            center = np.array([w / 2.0, h / 2.0])

            MIN_SUBJECT_FRACTION = 0.02      # 2% of the crop
            CENTRE_RADIUS = 0.35 * min(h, w)  # "near the middle"

            def _area(p):
                return float(p.morphology.get("area") or 0.0)

            substantial = [p for p in crop_particles
                           if _area(p) >= MIN_SUBJECT_FRACTION * crop_area]
            if not substantial:
                substantial = crop_particles

            central = [p for p in substantial
                       if np.linalg.norm(np.array([p.centroid.x, p.centroid.y]) - center)
                       <= CENTRE_RADIUS]
            candidates = central or substantial

            if class_name == "Molten_Satellite" and len(candidates) > 1:
                primary = min(candidates, key=_area)
            else:
                primary = max(candidates, key=_area)

            if len(crop_particles) > 1:
                compute_attachments(
                    crop_particles,
                    contact_gap_px=config.ATTACHMENT.contact_gap_px,
                    min_contact_fraction=config.ATTACHMENT.min_contact_fraction,
                    host_size_ratio=config.ATTACHMENT.host_size_ratio,
                )
            try:
                _crop_cache[int(primary.particle_id)] = crop_particle_patch(
                    image_rgb, primary,
                    padding_ratio=config.DINOV2.padding_ratio if hasattr(config, "DINOV2") else 0.2,
                    min_pad_px=4, mask_background=False,
                )
            except Exception as _exc:
                logger.warning(f"crop cache failed for {image_path}: {_exc}")

            try:
                _v2_cache = compute_v2_features(primary, config.PHYSICS)
            except Exception as _exc:
                logger.warning(f"v2 physics failed for {image_path}: {_exc}")
                _v2_cache = {}

            embeddings = dinov2_extractor.extract_augmented_embeddings(primary, image_rgb)

            attachment_snapshot = {
                "is_attached": primary.attachment.is_attached,
                "has_host": primary.attachment.host_particle_id is not None,
                "attached_count": len(primary.attachment.attached_particle_ids),
                "boundary_contact_fraction": primary.attachment.boundary_contact_fraction,
                "nearest_neighbor_distance_px": primary.attachment.nearest_neighbor_distance_px,
            }

            for view_index, embedding in enumerate(embeddings):

                records.append({
                    "class_label": class_name,
                    "source_path": str(image_path),
                    "source_group_id": primary.particle_id,
                    "augmentation_view": view_index,
                    "particle_id": primary.particle_id,
                    "size_um": primary.size_um,
                    "dinov2_embedding": embedding,
                    "morphology": dict(primary.morphology.values),
                    "physics": dict(primary.physics.values),
                    "physics_v2": _v2_cache,
                    "attachment": attachment_snapshot,
                })

    if not records:
        raise ValueError(
            f"No usable prototype examples found under {examples_dir}. "
            "Check folder names match the taxonomy and images contain a clear particle."
        )

    joblib.dump(records, os.path.join(output_dir, PROTOTYPES_FILENAME))

    # Crops saved separately from prototypes.joblib: they are bulky and
    # only the fine-tuned ViT needs them, so keeping them out keeps the
    # prototype file cheap for every other consumer to load.
    if _crop_cache:
        joblib.dump(_crop_cache, os.path.join(output_dir, "prototype_crops.joblib"))
        logger.info(f"Cached {len(_crop_cache)} particle crops for ViT training.")

    summary = (
        pd.DataFrame([{"class_label": r["class_label"]} for r in records])
        .value_counts()
        .reset_index(name="count")
        .sort_values("class_label")
    )
    summary.to_csv(os.path.join(output_dir, PROTOTYPES_SUMMARY_FILENAME), index=False)

    logger.info(f"Built {len(records)} prototype example(s) across {summary.shape[0]} class(es).")
    logger.info(f"Saved to {os.path.join(output_dir, PROTOTYPES_FILENAME)}")

    for _, row in summary.iterrows():
        flag = " (LOW)" if row["count"] < config.CLASSIFICATION.min_prototype_examples else ""
        logger.info(f"  {row['class_label']}: {row['count']}{flag}")

    return summary


def load_prototype_set(output_dir: str):

    path = os.path.join(output_dir, PROTOTYPES_FILENAME)

    if not os.path.exists(path):
        return None

    return joblib.load(path)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Build the v4 prototype set from hand-sorted example folders.")

    parser.add_argument("--examples_dir", required=True, help="Folder containing 10 subfolders, one per defect class.")
    parser.add_argument("--output_dir", required=True, help="Where prototypes.joblib / prototypes_summary.csv are saved.")
    parser.add_argument("--microns_per_pixel", type=float, required=True, help="Calibration for ALL prototype crops (same magnification as main dataset).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dinov2_model", default=None,
        help="DINOv2 backbone (e.g. dinov2_vits14, dinov2_vitb14, dinov2_vitl14). "
             "Larger = richer features, slower. Defaults to dinov2_features.DEFAULT_MODEL.",
    )

    args = parser.parse_args()

    build_prototype_set(
        examples_dir=args.examples_dir,
        output_dir=args.output_dir,
        microns_per_pixel=args.microns_per_pixel,
        device=args.device,
        dinov2_model=args.dinov2_model,
    )


if __name__ == "__main__":
    main()


__all__ = ["build_prototype_set", "load_prototype_set", "find_particle_mask", "PROTOTYPES_FILENAME"]
