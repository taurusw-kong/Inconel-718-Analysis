#!/usr/bin/env python3
"""
Re-classify an existing pipeline run WITHOUT re-segmenting.

WHY THIS EXISTS
    main.py runs segmentation -> features -> classification in one pass,
    and reuses whatever trained classifier it finds in the output
    directory. If that classifier is stale (trained before a taxonomy
    change, a feature-exclusion change, or a retrain), every prediction
    in the run reflects the old model while the expensive segmentation
    is perfectly fine.

    Re-running main.py to fix this re-segments from scratch — hours of
    SAM2 work thrown away to redo a classification step that takes
    seconds. This script instead reloads the cached per-image label maps
    (output_dir/masks/*_labels.npz), rebuilds the Particle objects and
    their features from them, and applies a chosen classifier.

    Segmentation is NOT re-run and its results are NOT modified.

USAGE
    python3 reclassify_from_masks.py \
        --run_dir   /path/to/output_sem2 \
        --classifier_dir /path/to/output/classifier \
        --input_dir /path/to/SEM2 \
        --scale_bar_um 10.0
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob
import logging

import cv2
import numpy as np
import pandas as pd

import config
from data_models import Particle, BoundingBox, Centroid
from segmentation import estimate_border_truncation
from preprocessing import build_preprocessor, get_calibration
from morphology import MorphologyExtractor, apply_calibration
from physics_formulas import PhysicsDescriptorExtractor, compute_named_physics_formulas
from attachment import compute_attachments
from dinov2_features import build_dinov2_extractor
from classification import load_classifier, classify_particles
from explainability import explain_predictions
from reusability import ReusabilityScorer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def particles_from_label_map(label_map: np.ndarray, image_name: str,
                             image_shape=None):
    """
    Rebuild Particle objects from a saved integer label map.

    Label 0 is background; every other value is one particle instance,
    exactly as save_particle_label_map wrote it.
    """
    particles = []

    for label in np.unique(label_map):
        if label == 0:
            continue

        mask = label_map == label
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue

        # morphology.py works from the CONTOUR, not the mask, so it has to
        # be recovered here exactly as segmentation.py first produced it:
        # external contour only, CHAIN_APPROX_NONE (every boundary pixel
        # kept — approximation would quietly change perimeter, circularity
        # and fractal dimension relative to the original run).
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        particle = Particle(
            particle_id=int(label),
            image_name=image_name,
            mask=mask,
            bbox=BoundingBox(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            centroid=Centroid(float(xs.mean()), float(ys.mean())),
            contour=contour,
        )

        # classify_particles() routes particles whose metadata says
        # border_truncated to the Edge_Truncated sentinel instead of the
        # model, because a particle clipped by the frame has unreliable
        # morphology. That metadata is set during segmentation, so it has
        # to be recomputed here — without it every truncated particle
        # would silently receive a real (and meaningless) class label.
        touches, fraction, truncated = estimate_border_truncation(
            mask, image_shape if image_shape is not None else label_map.shape,
        )
        particle.metadata["touches_border"] = touches
        particle.metadata["border_fraction"] = fraction
        particle.metadata["border_truncated"] = truncated

        particles.append(particle)

    return particles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Existing output dir containing masks/")
    ap.add_argument("--classifier_dir", required=True, help="Directory holding the CORRECT trained classifier")
    ap.add_argument("--input_dir", required=True, help="Folder of the original SEM images")
    ap.add_argument("--scale_bar_um", type=float, default=None)
    ap.add_argument("--fov_width_um", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    masks_dir = os.path.join(args.run_dir, "masks")
    label_files = sorted(glob.glob(os.path.join(masks_dir, "*_labels.npz")))
    if not label_files:
        raise SystemExit(f"No cached label maps in {masks_dir}")

    logger.info(f"Found {len(label_files)} cached label map(s) — segmentation will NOT be re-run.")

    # load_classifier expects the RUN directory (it looks for classifier/
    # inside), so accept either that or the classifier dir itself.
    classifier = load_classifier(args.classifier_dir)
    if classifier is None:
        classifier = load_classifier(os.path.dirname(args.classifier_dir.rstrip("/")))
    if classifier is None:
        raise SystemExit(f"No trained classifier found at {args.classifier_dir}")

    logger.info(f"Classifier: best={classifier['best_model_name']} "
                f"classes={list(classifier['label_encoder'].classes_)}")

    preprocessor = build_preprocessor()
    morphology_extractor = MorphologyExtractor()
    physics_extractor = PhysicsDescriptorExtractor()
    dinov2_extractor = build_dinov2_extractor(device=args.device)

    all_particles = []

    for i, lf in enumerate(label_files, 1):
        base = os.path.basename(lf).replace("_labels.npz", "")

        matches = glob.glob(os.path.join(args.input_dir, base + ".*"))
        if not matches:
            logger.warning(f"No source image for {base}, skipped.")
            continue
        image_path = matches[0]

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logger.warning(f"Could not read {image_path}, skipped.")
            continue

        calibration = get_calibration(
            image_bgr, os.path.basename(image_path), {},
            fov_width_um=args.fov_width_um, scale_bar_um=args.scale_bar_um,
            image_path=image_path,
        )

        image_rgb, _raw, _stats = preprocessor.process(image_bgr)

        label_map = np.load(lf)["labels"]

        # save_particle_label_map pads back to the FULL image height so the
        # map aligns with the original file; the preprocessed image has the
        # info bar cropped off. Trim to match before rebuilding masks, or
        # every mask would be offset relative to the pixels it describes.
        if label_map.shape != image_rgb.shape[:2]:
            label_map = label_map[:image_rgb.shape[0], :image_rgb.shape[1]]

        particles = particles_from_label_map(
            label_map, os.path.basename(image_path), image_rgb.shape)
        if not particles:
            continue

        for p in particles:
            morphology_extractor.extract(p)
            apply_calibration(p, calibration.microns_per_pixel)
            physics_extractor.extract(p)
            compute_named_physics_formulas(p, config.PHYSICS)

        compute_attachments(
            particles,
            contact_gap_px=config.ATTACHMENT.contact_gap_px,
            min_contact_fraction=config.ATTACHMENT.min_contact_fraction,
            host_size_ratio=config.ATTACHMENT.host_size_ratio,
        )

        dinov2_extractor.extract_batch(particles, image_rgb)
        all_particles.extend(particles)

        if i % 10 == 0 or i == len(label_files):
            logger.info(f"  {i}/{len(label_files)} images  ({len(all_particles)} particles)")

    logger.info(f"Rebuilt {len(all_particles)} particle(s) from cached masks.")

    classify_particles(all_particles, classifier)
    logger.info("Classified with the supplied classifier.")

    try:
        explain_predictions(all_particles, classifier, args.run_dir)
        logger.info("Wrote refreshed SHAP explanations.")
    except Exception as exc:
        logger.warning(f"Explainability step failed (non-fatal): {exc}")

    from main import particle_to_row
    features_df = pd.DataFrame([particle_to_row(p) for p in all_particles])
    out_csv = os.path.join(args.run_dir, "features.csv")
    features_df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(features_df)} particles).")

    if "defect_label" in features_df.columns and features_df["defect_label"].notna().any():
        scorer = ReusabilityScorer()
        scored_df = scorer.score_dataframe(features_df)
        scored_df.to_csv(os.path.join(args.run_dir, "features_with_reusability.csv"), index=False)
        scorer.batch_summary(scored_df).to_csv(
            os.path.join(args.run_dir, "reusability_summary.csv"), index=False)
        logger.info("Wrote features_with_reusability.csv and reusability_summary.csv")

    logger.info("Reclassification complete — segmentation untouched.")


if __name__ == "__main__":
    main()
