"""
ParticleVision V4
=================

Pipeline Orchestration (CLI)

One run: segment every image (SAM2) -> attachment detection ->
morphology + physics-constrained features (calibrated to real microns)
-> DINOv2 embeddings -> classify (unified XGBoost / prototype few-shot
model, trained folders) -> explainability (SHAP /
nearest-prototype) -> reusability score.

"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  

import argparse
import datetime
import json

import cv2
import numpy as np
import pandas as pd

import config
from utils import list_images, load_rgb, logger, ensure_dir, Timer
from preprocessing import build_preprocessor, get_calibration, load_manual_calibration
from segmentation import build_segmenter
from morphology import MorphologyExtractor, apply_calibration
from physics_formulas import PhysicsDescriptorExtractor, compute_named_physics_formulas
from attachment import compute_attachments
from dinov2_features import build_dinov2_extractor
from data_models import Particle


MASKS_DIRNAME = "masks"


def save_particle_label_map(particles, image_shape, output_dir: str, image_name: str, full_image_shape=None):
    """
    Persists the REAL per-pixel segmentation masks for one image as a
    single compact label map (0 = background, N = particle_id N),
    keyed by (image_name, particle_id) — the same composite key
    features.csv already uses. Without this, anything previewing
    results after the fact (gui_app.py's Image Preview) has only
    bbox/centroid to work with and has to fake a shape (an ellipse) —
    which looks nothing like the real, often irregular SAM2 mask and
    defeats the point of a shape-sensitive taxonomy (Agglomerate,
    Deformed_Particle, Clip_Cap are literally defined by not being a
    clean ellipse). Saved once per image at segmentation time, when
    the real masks exist, not reconstructed later.

    `image_shape` must match what segmentation actually ran on
    (preprocessor.process()'s output, i.e. particle.mask's own shape)
    — NOT the original file's shape, which can be taller by the
    microscope info-bar strip height that gets cropped before
    segmentation. If `full_image_shape` (the ORIGINAL, pre-crop shape)
    is given, the map is padded back to it (zeros for the removed
    strip, at the bottom — remove_information_strip always crops from
    the bottom, keeping the top-left origin identical) so a preview
    that reloads the original file lines up pixel-for-pixel without
    needing to redo strip detection itself.
    """

    label_map = np.zeros(image_shape[:2], dtype=np.uint16)

    for particle in particles:
       
        label_map[particle.mask.astype(bool)] = particle.particle_id

    if full_image_shape is not None and full_image_shape[:2] != label_map.shape:

        full_h, full_w = full_image_shape[:2]
        padded = np.zeros((full_h, full_w), dtype=np.uint16)
        padded[: label_map.shape[0], : label_map.shape[1]] = label_map
        label_map = padded

    masks_dir = os.path.join(output_dir, MASKS_DIRNAME)
    os.makedirs(masks_dir, exist_ok=True)

    base = os.path.splitext(image_name)[0]
    np.savez_compressed(os.path.join(masks_dir, f"{base}_labels.npz"), labels=label_map)


def load_particle_label_map(output_dir: str, image_name: str):

    base = os.path.splitext(image_name)[0]
    path = os.path.join(output_dir, MASKS_DIRNAME, f"{base}_labels.npz")

    if not os.path.exists(path):
        return None

    return np.load(path)["labels"]

from build_prototypes import build_prototype_set, load_prototype_set, PROTOTYPES_FILENAME
import classification
from classification import train_and_compare, load_classifier, classify_particles
from explainability import explain_predictions
from reusability import ReusabilityScorer


# ============================================================
# REPRODUCIBILITY
# ============================================================

def write_run_config(args: argparse.Namespace, output_dir: str):

    payload = dict(vars(args))
    payload["timestamp"] = datetime.datetime.now().isoformat()
    payload["version"] = config.VERSION

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ============================================================
# PARTICLE -> ROW  (clean, unprefixed columns for features.csv)
# ============================================================

def particle_to_row(particle: Particle) -> dict:

    row = {
        "particle_id": particle.particle_id,
        "image_name": particle.image_name,
        "bbox_x": particle.bbox.x,
        "bbox_y": particle.bbox.y,
        "bbox_width": particle.bbox.width,
        "bbox_height": particle.bbox.height,
        "centroid_x": particle.centroid.x,
        "centroid_y": particle.centroid.y,
        "size_um": particle.size_um,
        "defect_label": particle.prediction.label,
        "confidence": particle.prediction.confidence,
        "prediction_model": particle.prediction.model_name,
        "is_attached": particle.attachment.is_attached,
        "host_particle_id": particle.attachment.host_particle_id,
        "attached_count": len(particle.attachment.attached_particle_ids),
        "boundary_contact_fraction": particle.attachment.boundary_contact_fraction,
        "nearest_neighbor_distance_px": particle.attachment.nearest_neighbor_distance_px,
 
        "touches_border": particle.metadata.get("touches_border", False),
        "border_fraction": particle.metadata.get("border_fraction", 0.0),
        "border_truncated": particle.metadata.get("border_truncated", False),
    }

    row.update(particle.morphology.values)
    row.update(particle.physics.values)

    if "microns_per_pixel" in particle.metadata:
        row["microns_per_pixel"] = particle.metadata["microns_per_pixel"]

    return row


# ============================================================
# SEGMENTATION + FEATURE PASS (one image)
# ============================================================

def process_image(
    image_path,
    segmenter,
    preprocessor,
    morphology_extractor,
    physics_extractor,
    dinov2_extractor,
    manual_calibration,
    fov_width_um,
    scale_bar_um=None,
    output_dir=None,
):

    image_name = os.path.basename(str(image_path))

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        logger.warning(f"Could not read {image_path}, skipped.")
        return []

    calibration = get_calibration(
        image_bgr, image_name, manual_calibration,
        fov_width_um=fov_width_um, scale_bar_um=scale_bar_um,
        image_path=image_path,
    )

    image_rgb, raw_rgb, _prep_stats = preprocessor.process(image_bgr)

    particles, seg_stats = segmenter.segment(image_rgb, image_name=image_name, raw_image_rgb=raw_rgb)

    logger.info(
        f"{image_name}: {seg_stats.kept_masks} particle(s) kept "
        f"(from {seg_stats.raw_masks} raw mask(s))."
    )

    if not particles:
        return []

    if output_dir is not None:
        save_particle_label_map(
            particles, image_rgb.shape, output_dir, image_name,
            full_image_shape=image_bgr.shape,
        )

    for particle in particles:
        morphology_extractor.extract(particle)
        apply_calibration(particle, calibration.microns_per_pixel)
        physics_extractor.extract(particle)
        compute_named_physics_formulas(particle, config.PHYSICS)

    compute_attachments(
        particles,
        contact_gap_px=config.ATTACHMENT.contact_gap_px,
        min_contact_fraction=config.ATTACHMENT.min_contact_fraction,
        host_size_ratio=config.ATTACHMENT.host_size_ratio,
    )

    dinov2_extractor.extract_batch(particles, image_rgb)

    return particles


# ============================================================
# MAIN PIPELINE
# ============================================================

def _seed_everything(seed: int = 42):
 

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_pipeline(args: argparse.Namespace):

    _seed_everything()

    ensure_dir(args.output_dir)
    write_run_config(args, args.output_dir)

    manual_calibration = (
        load_manual_calibration(args.calibration_csv) if args.calibration_csv else {}
    )

    with Timer("Loading SAM2"):
        segmenter = build_segmenter(
            checkpoint_path=args.checkpoint,
            model_cfg=args.model_cfg,
            device=args.device,
            min_area=config.SEGMENTATION.min_particle_area_px,
            border_margin=config.SEGMENTATION.border_margin,
            edge_filtering=config.SEGMENTATION.edge_filtering,
            points_per_side=config.SEGMENTATION.points_per_side,
            pred_iou_thresh=config.SEGMENTATION.pred_iou_thresh,
            stability_score_thresh=config.SEGMENTATION.stability_score_thresh,
        )

    preprocessor = build_preprocessor()
    dinov2_extractor = build_dinov2_extractor(device=args.device)
    morphology_extractor = MorphologyExtractor()
    physics_extractor = PhysicsDescriptorExtractor()

    image_paths = list_images(args.input_dir)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    all_particles = []

    for image_path in image_paths:
        with Timer(f"Segmenting {os.path.basename(str(image_path))}"):
            particles = process_image(
                image_path, segmenter, preprocessor,
                morphology_extractor, physics_extractor, dinov2_extractor,
                manual_calibration, args.fov_width_um,
                scale_bar_um=args.scale_bar_um,
                output_dir=args.output_dir,
            )
            all_particles.extend(particles)

    logger.info(f"Total particles across {len(image_paths)} image(s): {len(all_particles)}")

    # ---------------- classifier: train (if requested/needed) or load ----------------

    classifier = None
    prototype_records = None

    prototypes_path = os.path.join(args.output_dir, PROTOTYPES_FILENAME)

    if args.examples_dir:

        with Timer("Building prototype set from example folders"):
            build_prototype_set(
                examples_dir=args.examples_dir,
                output_dir=args.output_dir,
                microns_per_pixel=args.prototype_microns_per_pixel,
                device=args.device,
            )

    if os.path.exists(prototypes_path):
        prototype_records = load_prototype_set(args.output_dir)

    existing_classifier = load_classifier(args.output_dir)

    if existing_classifier is not None and not args.retrain_classifier:

        classifier = existing_classifier
        logger.info("Loaded existing trained classifier.")

    elif prototype_records is not None:

        with Timer("Training unified classifier"):
            train_and_compare(prototype_records, args.output_dir)

        classifier = load_classifier(args.output_dir)

    else:
        logger.warning(
            "No trained classifier and no prototype examples available — "
            "particles will be left unclassified. Provide --examples_dir "
            "(your 10 labeled example folders) to train one."
        )

    if classifier is not None and all_particles:

        with Timer("Classifying particles"):
            classify_particles(all_particles, classifier)

        with Timer("Explaining predictions"):
            explain_predictions(all_particles, classifier, args.output_dir, prototype_records=prototype_records)

    # ---------------- export ----------------

    features_df = pd.DataFrame([particle_to_row(p) for p in all_particles])
    features_csv = os.path.join(args.output_dir, "features.csv")
    features_df.to_csv(features_csv, index=False)
    logger.info(f"Wrote {features_csv} ({len(features_df)} particle(s)).")

    if classifier is not None and "defect_label" in features_df.columns and features_df["defect_label"].notna().any():

        scorer = ReusabilityScorer()
        scored_df = scorer.score_dataframe(features_df)
        scored_csv = os.path.join(args.output_dir, "features_with_reusability.csv")
        scored_df.to_csv(scored_csv, index=False)

        summary_df = scorer.batch_summary(scored_df)
        summary_df.to_csv(os.path.join(args.output_dir, "reusability_summary.csv"), index=False)

        logger.info(f"Wrote {scored_csv} and reusability_summary.csv")

    logger.info("Pipeline complete.")

    return features_df


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="ParticleVision V4 pipeline.")

    parser.add_argument("--input_dir", required=True, help="Folder of SEM images.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint (.pt).")
    parser.add_argument("--model_cfg", required=True, help="SAM2 model config (.yaml).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    parser.add_argument("--calibration_csv", default=None, help="Optional manual per-image calibration CSV.")
    parser.add_argument("--fov_width_um", type=float, default=None, help="Field-of-view width in microns (simplest calibration source).")
    parser.add_argument("--scale_bar_um", type=float, default=None,
                        help="Value in microns of the burned-in scale bar (e.g. 10). The bar's pixel length is measured directly — no OCR — so this is far more reliable than the OCR fallback.")

    parser.add_argument("--examples_dir", default=None, help="Folder of 10 labeled example subfolders (see build_prototypes.py) — builds/updates the prototype set and (re)trains the classifier.")
    parser.add_argument("--prototype_microns_per_pixel", type=float, default=None, help="Calibration for --examples_dir crops (required if --examples_dir is given).")
    parser.add_argument("--retrain_classifier", action="store_true", help="Retrain even if a trained classifier already exists in output_dir.")

    return parser


def main():

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.examples_dir and args.prototype_microns_per_pixel is None:
        parser.error("--prototype_microns_per_pixel is required when --examples_dir is given.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
