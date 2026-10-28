"""
ParticleVision V4
=================

Segmentation Validation

Everything downstream (morphology, physics features, attachment,
classification) depends on SAM2's masks being trustworthy — this
script measures that directly, independent of classification, exactly
as the manuscript's Section 5.1 asks for (segmentation performance
reported on its own, not folded into classification accuracy).

Usage:
    python validate_segmentation.py --input_dir <folder with 2-3 images> \\
        --checkpoint <sam2.pt> --model_cfg <sam2.yaml> --device cpu

For each image: SAM2 segments it first, then a window opens showing
the image. For each of a handful of particles (mix of easy, isolated
ones and hard, touching ones — pick a variety):

    - left-click to add a point to your hand-drawn outline
    - 'c' to close the polygon and score it against SAM2's own mask
    - 'u' to undo the last point
    - 'n' to move to the next image
    - 'q' / ESC to stop early and save what you have

Real IoU between your outline and whichever SAM2 mask it best
overlaps is saved to segmentation_iou_report.csv, plus a printed
summary. Low-IoU cases point at exactly which segmentation.py
parameter to retune (touching particles under-split -> min_distance /
min_peak_prominence_ratio; particles missed or merged with background
-> pred_iou_thresh / stability_score_thresh).
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from segmentation import build_segmenter, mask_iou
from utils import list_images, load_rgb, logger


IOU_WEAK_THRESHOLD = 0.5


# ============================================================
# CORE LOGIC (unit-testable independent of the interactive UI)
# ============================================================

def polygon_to_mask(points: list, image_shape) -> np.ndarray:
    """points: list of (x, y) pixel coordinates, in order, defining a closed polygon."""

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if len(points) < 3:
        return mask.astype(bool)

    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)

    return mask.astype(bool)


def best_matching_particle(hand_mask: np.ndarray, particles: list):
    """
    Returns (particle, iou) for whichever segmented particle's mask
    has the highest IoU with the hand-drawn outline — i.e. "which SAM2
    mask were you actually outlining." None if no particle overlaps
    at all (SAM2 missed this particle entirely — itself a real,
    reportable failure, recorded with iou=0.0 and particle=None).
    """

    best_particle = None
    best_iou = 0.0

    for particle in particles:

        iou = mask_iou(hand_mask, particle.mask)

        if iou > best_iou:
            best_iou = iou
            best_particle = particle

    return best_particle, best_iou


def score_outline(points: list, image_shape, particles: list) -> dict:

    hand_mask = polygon_to_mask(points, image_shape)
    particle, iou = best_matching_particle(hand_mask, particles)

    return {
        "hand_drawn_area_px": int(np.count_nonzero(hand_mask)),
        "matched_particle_id": particle.particle_id if particle else None,
        "sam2_area_px": int(np.count_nonzero(particle.mask)) if particle else 0,
        "iou": iou,
    }


def summarize(rows: list) -> dict:

    df = pd.DataFrame(rows)

    if df.empty:
        return {"n": 0}

    return {
        "n": len(df),
        "mean_iou": df["iou"].mean(),
        "median_iou": df["iou"].median(),
        "min_iou": df["iou"].min(),
        "n_weak": int((df["iou"] < IOU_WEAK_THRESHOLD).sum()),
        "n_missed": int((df["matched_particle_id"].isna()).sum()),
    }


# ============================================================
# INTERACTIVE UI (cv2 window + mouse callback)
# ============================================================

class _OutlineSession:

    def __init__(self, image_bgr, particles, image_name):

        self.image_bgr = image_bgr
        self.particles = particles
        self.image_name = image_name
        self.points = []
        self.rows = []
        self.window_name = f"Outline particles — {image_name}  (c=close, u=undo, n=next image, q=quit)"

    def _on_mouse(self, event, x, y, flags, _param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def _redraw(self):

        canvas = self.image_bgr.copy()

        for x, y in self.points:
            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)

        if len(self.points) >= 2:
            cv2.polylines(canvas, [np.array(self.points, dtype=np.int32)], False, (0, 0, 255), 1)

        cv2.imshow(self.window_name, canvas)

    def run(self) -> bool:
        """Returns True to continue to the next image, False to quit entirely."""

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:

            self._redraw()
            key = cv2.waitKey(20) & 0xFF

            if key == ord("u") and self.points:
                self.points.pop()

            elif key == ord("c") and len(self.points) >= 3:

                result = score_outline(self.points, self.image_bgr.shape, self.particles)
                result["image_name"] = self.image_name
                self.rows.append(result)

                logger.info(f"Outline scored: IoU = {result['iou']:.3f}")
                self.points = []

            elif key == ord("n") or key == ord("q") or key == 27:

                cv2.destroyWindow(self.window_name)
                return key != ord("q") and key != 27

        return True


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Hand-outline a few particles per image and measure real SAM2 IoU.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    segmenter = build_segmenter(checkpoint_path=args.checkpoint, model_cfg=args.model_cfg, device=args.device)

    image_paths = list_images(args.input_dir)

    if not image_paths:
        raise FileNotFoundError(f"No images in {args.input_dir}")

    all_rows = []

    for image_path in image_paths:

        image_rgb = load_rgb(str(image_path))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        particles, stats = segmenter.segment(image_rgb, image_name=os.path.basename(str(image_path)))
        logger.info(f"{image_path}: SAM2 found {len(particles)} particle(s) — now outline a handful by hand.")

        session = _OutlineSession(image_bgr, particles, os.path.basename(str(image_path)))
        keep_going = session.run()
        all_rows.extend(session.rows)

        if not keep_going:
            break

    cv2.destroyAllWindows()

    if not all_rows:
        logger.warning("No outlines were scored — nothing to report.")
        return

    report_df = pd.DataFrame(all_rows)
    report_path = os.path.join(args.output_dir, "segmentation_iou_report.csv")
    report_df.to_csv(report_path, index=False)

    summary = summarize(all_rows)

    print()
    print("=" * 60)
    print("SEGMENTATION VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Outlines scored : {summary['n']}")
    print(f"Mean IoU        : {summary['mean_iou']:.3f}")
    print(f"Median IoU      : {summary['median_iou']:.3f}")
    print(f"Min IoU         : {summary['min_iou']:.3f}")
    print(f"Weak (<{IOU_WEAK_THRESHOLD}) : {summary['n_weak']} / {summary['n']}")
    print(f"Missed entirely : {summary['n_missed']} / {summary['n']}")
    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    main()


__all__ = ["polygon_to_mask", "best_matching_particle", "score_outline", "summarize"]
