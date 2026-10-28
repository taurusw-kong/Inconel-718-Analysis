"""
ParticleVision V4
=================

Classification Overlay + Legend

"""

from __future__ import annotations

import cv2
import numpy as np

from data_models import Particle
from config import CLASSIFICATION, VISUALIZATION, BORDER_TRUNCATED_LABEL, BORDER_TRUNCATED_COLOR



LABEL_COLORS = dict(CLASSIFICATION.colors)
LABEL_COLORS[BORDER_TRUNCATED_LABEL] = BORDER_TRUNCATED_COLOR

UNCLASSIFIED_COLOR = (140, 140, 140)


# ============================================================
# ROUNDED RECTANGLE 
# ============================================================

def draw_rounded_rectangle(image, pt1, pt2, color, radius, thickness=-1):
    """
    Standard OpenCV recipe for a rounded rect: a plain rectangle for
    the body plus 4 filled corner arcs (cv2 has no native rounded-rect
    primitive). thickness=-1 fills; any positive value draws an
    outline of that width instead.
    """

    x1, y1 = pt1
    x2, y2 = pt2

    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    radius = max(radius, 0)

    if thickness < 0:

        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)

        for cx, cy in (
            (x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius), (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, (cx, cy), radius, color, -1)

    else:

        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)

        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


# ============================================================
# OVERLAY
# ============================================================

def particle_color(particle: Particle):
    return LABEL_COLORS.get(particle.prediction.label, UNCLASSIFIED_COLOR)


def draw_dashed_contour(image, contour, color, dash_length=None, gap_length=None, thickness=1):

    dash_length = dash_length if dash_length is not None else VISUALIZATION.border_dash_length
    gap_length = gap_length if gap_length is not None else VISUALIZATION.border_gap_length
    period = max(dash_length + gap_length, 1)

    pts = contour.reshape(-1, 2).astype(np.float64)

    if len(pts) < 2:
        return

    pts = np.vstack([pts, pts[0]])
    dist = 0.0

    for i in range(len(pts) - 1):

        p0, p1 = pts[i], pts[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))

        if seg_len == 0:
            continue

        direction = (p1 - p0) / seg_len
        traveled = 0.0

        while traveled < seg_len:

            phase = dist % period
            in_dash = phase < dash_length
            remain_in_phase = (dash_length - phase) if in_dash else (period - phase)
            step = min(remain_in_phase, seg_len - traveled)

            start_pt = p0 + direction * traveled
            end_pt = p0 + direction * (traveled + step)

            if in_dash:
                cv2.line(
                    image,
                    (int(round(start_pt[0])), int(round(start_pt[1]))),
                    (int(round(end_pt[0])), int(round(end_pt[1]))),
                    color, thickness, cv2.LINE_AA,
                )

            traveled += step
            dist += step


def draw_classification_overlay(image_rgb, particles, alpha=None, show_outline=True, on_black=False):
    """
    One flat, translucent color per predicted defect class — every
    particle of the same class gets exactly the same color, unchanged
    from before. Publication figure / GUI Image Preview material.

    Edge-truncated particles (see config.BORDER_TRUNCATED_LABEL) are
    the one exception: a cut-off particle was never fully measured, so
    filling it in the same style as a real classification overstates
    what's actually known about it. These get NO fill — just a dashed
    orange outline, so they're still visible as detections without
    looking like a confident class prediction.
    """

    alpha = alpha if alpha is not None else VISUALIZATION.overlay_alpha

    if on_black:
        overlay = np.zeros_like(image_rgb)
        blend_alpha = 1.0
    else:
        overlay = image_rgb.copy()
        blend_alpha = alpha

    normal = [p for p in particles if p.prediction.label != BORDER_TRUNCATED_LABEL]
    truncated = [p for p in particles if p.prediction.label == BORDER_TRUNCATED_LABEL]

    colored = np.zeros_like(image_rgb)

    for particle in normal:
        colored[particle.mask.astype(bool)] = particle_color(particle)

    overlay = cv2.addWeighted(overlay, 1.0, colored, blend_alpha, 0)

    if show_outline:
        for particle in normal:
            if particle.contour is not None:
                draw_dashed_contour(overlay, particle.contour, particle_color(particle), thickness=1)

    for particle in truncated:
        if particle.contour is not None:
            draw_dashed_contour(overlay, particle.contour, BORDER_TRUNCATED_COLOR, thickness=1)

    return overlay


# ============================================================
# LEGEND (rounded-corner swatches)
# ============================================================

def draw_legend(labels=None, swatch=34, pad=10, font_scale=0.55, corner_radius=8):

    labels = labels or list(LABEL_COLORS.keys())

    width = 340
    height = pad + len(labels) * (swatch + pad)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    y = pad

    for label in labels:

        color = LABEL_COLORS.get(label, UNCLASSIFIED_COLOR)

        draw_rounded_rectangle(
            canvas, (pad, y), (pad + swatch, y + swatch), color, corner_radius, thickness=-1,
        )

        cv2.putText(
            canvas, label.replace("_", " "),
            (pad * 2 + swatch, y + swatch - 8),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA,
        )

        y += swatch + pad

    return canvas


# ============================================================
# COMPACT GRID LEGEND (composited directly onto a corner of an image)
# ============================================================

def draw_compact_legend_grid(
    image,
    labels=None,
    columns=3,
    swatch=14,
    pad=8,
    font_scale=0.34,
    corner_radius=4,
    position="bottom-left",
    margin=10,
    backing_alpha=0.72,
):
    """
    Composites a small, fixed-size N-column legend grid directly onto
    a corner of `image` (in place of a separate widget) — e.g. every
    class in a 3x3 grid for a 9-class taxonomy. Cell width is sized
    from the actual rendered width of the LONGEST label at this font
    scale, so labels never overlap regardless of how long any class
    name is. A translucent dark backing panel sits behind the whole
    grid so swatches/text stay readable against any image content.
    """

    labels = labels or list(LABEL_COLORS.keys())

    rows = int(np.ceil(len(labels) / columns))

  
    max_text_w = max(
        cv2.getTextSize(label.replace("_", " "), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for label in labels
    )

    cell_w = pad + swatch + pad // 2 + max_text_w + pad
    cell_h = swatch + pad

    grid_w = cell_w * columns + pad
    grid_h = cell_h * rows + pad

    h, w = image.shape[:2]

    if position == "bottom-left":
        x0, y0 = margin, h - grid_h - margin
    elif position == "bottom-right":
        x0, y0 = w - grid_w - margin, h - grid_h - margin
    elif position == "top-left":
        x0, y0 = margin, margin
    else:
        x0, y0 = w - grid_w - margin, margin

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x0 + grid_w), min(h, y0 + grid_h)

  
    backing = image.copy()
    draw_rounded_rectangle(backing, (x0, y0), (x1, y1), (18, 18, 18), 10, thickness=-1)
    cv2.addWeighted(backing, backing_alpha, image, 1 - backing_alpha, 0, dst=image)

    for i, label in enumerate(labels):

        col = i % columns
        row = i // columns

        cx = x0 + pad + col * cell_w
        cy = y0 + pad // 2 + row * cell_h

        color = LABEL_COLORS.get(label, UNCLASSIFIED_COLOR)

        draw_rounded_rectangle(
            image, (cx, cy), (cx + swatch, cy + swatch), color, corner_radius, thickness=-1,
        )

        cv2.putText(
            image, label.replace("_", " "),
            (cx + swatch + pad // 2, cy + swatch - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return image


__all__ = [
    "LABEL_COLORS", "UNCLASSIFIED_COLOR",
    "draw_rounded_rectangle", "particle_color", "draw_dashed_contour",
    "draw_classification_overlay", "draw_legend", "draw_compact_legend_grid",
]
