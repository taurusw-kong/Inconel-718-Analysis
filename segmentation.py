"""
ParticleVision V2
=================

Segmentation Engine

Author: Taurus W. Kong

This module performs:
- SAM2 loading
- Automatic mask generation
- Duplicate removal
- Hole filling
- Boundary smoothing
- Watershed splitting
- Background rejection
- Adaptive filtering
- Quality scoring
- Particle object creation

"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import (
    SAM2AutomaticMaskGenerator,
)

from data_models import (
    BoundingBox,
    Centroid,
    Particle,
)


# ============================================================
# STATISTICS
# ============================================================

@dataclass
class SegmentationStatistics:

    raw_masks: int = 0

    kept_masks: int = 0

    rejected_small: int = 0

    rejected_overlap: int = 0

    rejected_background: int = 0

    rejected_quality: int = 0

    rejected_artifact: int = 0

    rejected_contained: int = 0

    split_particles: int = 0


# ============================================================
# MODEL LOADER
# ============================================================

def resolve_sam2_config_name(model_cfg):

    if not os.path.isabs(model_cfg):
        return model_cfg

    if not os.path.exists(model_cfg):
       
        return model_cfg

    try:
        import sam2 as _sam2_pkg

        package_dir = os.path.dirname(os.path.abspath(_sam2_pkg.__file__))

    except Exception:
        return model_cfg

    abs_cfg = os.path.abspath(model_cfg)

    if abs_cfg.startswith(package_dir + os.sep):

        relative = os.path.relpath(abs_cfg, package_dir)

        # Hydra config names always use forward slashes, regardless
        # of OS.
        relative = relative.replace(os.sep, "/")

        if relative != model_cfg:

            print(
                f"[INFO] --model_cfg was an absolute path; using the "
                f"path relative to the sam2 package instead: {relative}"
            )

        return relative


    return model_cfg


def load_sam2_model(

    checkpoint_path,

    model_cfg,

    device="cuda",

    points_per_side=32,

    pred_iou_thresh=0.86,

    stability_score_thresh=0.92,

    min_mask_region_area=25,

):

    print(f"[INFO] Loading SAM2 model on {device}")

    model_cfg = resolve_sam2_config_name(model_cfg)

    sam_model = build_sam2(

        model_cfg,

        checkpoint_path,

        device=device,

    )

    generator = SAM2AutomaticMaskGenerator(

        model=sam_model,

        points_per_side=points_per_side,

        pred_iou_thresh=pred_iou_thresh,

        stability_score_thresh=stability_score_thresh,

   
        crop_n_layers=0,

        crop_n_points_downscale_factor=2,

        min_mask_region_area=min_mask_region_area,

    )

    print("[INFO] SAM2 ready.")

    return generator


# ============================================================
# BASIC UTILITIES
# ============================================================

def mask_iou(

    mask1,

    mask2,

):

    intersection = np.logical_and(

        mask1,

        mask2,

    ).sum()

    union = np.logical_or(

        mask1,

        mask2,

    ).sum()

    if union == 0:

        return 0.0

    return float(intersection / union)


def mask_containment(small_mask, large_mask):
  

    small_area = np.count_nonzero(small_mask)

    if small_area == 0:
        return 0.0

    intersection = np.count_nonzero(
        np.logical_and(small_mask, large_mask)
    )

    return float(intersection / small_area)


def contour_center(contour):

    M = cv2.moments(contour)

    if M["m00"] == 0:

        x, y, w, h = cv2.boundingRect(contour)

        return np.array([

            x + w / 2,

            y + h / 2,

        ])

    return np.array([

        M["m10"] / M["m00"],

        M["m01"] / M["m00"],

    ])


def contour_area(contour):

    return float(

        cv2.contourArea(

            contour

        )

    )


def contour_perimeter(contour):

    return float(

        cv2.arcLength(

            contour,

            True,

        )

    )


# ============================================================
# ARTIFACT REJECTION
# ============================================================

def is_artifact_shape(
    contour,
    image_shape,
    border_margin=5,
    edge_filtering=False,
    extent_threshold=0.94,
):


    h, w = image_shape[:2]

    x, y, cw, ch = cv2.boundingRect(contour)

    area = float(cv2.contourArea(contour))

    bbox_area = float(cw * ch)

    extent = area / bbox_area if bbox_area > 0 else 0.0

    if extent >= extent_threshold:
        return True

    if edge_filtering:

        touches_border = (
            x <= border_margin
            or y <= border_margin
            or (x + cw) >= (w - border_margin)
            or (y + ch) >= (h - border_margin)
        )

        if touches_border:
            return True

    return False


def estimate_border_truncation(mask, image_shape, truncation_threshold=0.15):


    h, w = image_shape[:2]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return False, 0.0, False

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    on_border = (
        (contour[:, 0] <= 0) | (contour[:, 0] >= w - 1) |
        (contour[:, 1] <= 0) | (contour[:, 1] >= h - 1)
    )

    border_fraction = float(np.mean(on_border)) if len(contour) > 0 else 0.0
    touches_border = border_fraction > 0.0

    return touches_border, border_fraction, border_fraction >= truncation_threshold


# ============================================================
# MASK CLEANING
# ============================================================

def remove_duplicate_masks(
    masks: List[dict],
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.90,
):

    if len(masks) <= 1:
        return masks, 0, 0

    masks = sorted(
        masks,
        key=lambda x: x["area"],
        reverse=True,
    )

    keep = []

    removed_duplicate = 0

    removed_contained = 0

    for candidate in masks:

        duplicate = False

        contained = False

        for accepted in keep:

            iou = mask_iou(
                candidate["segmentation"],
                accepted["segmentation"],
            )

            if iou >= iou_threshold:

                duplicate = True

                removed_duplicate += 1

                break

            containment = mask_containment(
                candidate["segmentation"],
                accepted["segmentation"],
            )

            if containment >= containment_threshold:

                contained = True

                removed_contained += 1

                break

        if not duplicate and not contained:

            keep.append(candidate)

    return keep, removed_duplicate, removed_contained


def remove_contained_final_particles(particles, containment_threshold: float = 0.85):

    if len(particles) <= 1:
        return particles, 0

    ordered = sorted(particles, key=lambda p: np.count_nonzero(p.mask), reverse=True)

    keep = []
    removed = 0

    for candidate in ordered:

        contained = False

        for accepted in keep:

            if mask_containment(candidate.mask, accepted.mask) >= containment_threshold:
                contained = True
                removed += 1
                break

        if not contained:
            keep.append(candidate)

    keep.sort(key=lambda p: p.particle_id)

    return keep, removed


def _contact_arc_fraction(child_seg, parent_seg, max_gap_px):
  
    contours, _ = cv2.findContours(
        child_seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    outside_parent = (~parent_seg).astype(np.uint8)
    dist = cv2.distanceTransform(outside_parent, cv2.DIST_L2, 5)

    at_contact = dist[contour[:, 1], contour[:, 0]] <= max_gap_px
    contact_arc_px = float(np.count_nonzero(at_contact))

    child_area = float(np.count_nonzero(child_seg))
    child_diameter_px = 2.0 * np.sqrt(child_area / np.pi) if child_area > 0 else 0.0

    return contact_arc_px / child_diameter_px if child_diameter_px > 0 else 0.0


def merge_into_whole_particles(
    masks,
    image_shape,
    bridge_gap_px=2,
    min_area=80,
    max_child_ratio=0.45,
    raw_image_rgb=None,
    min_bridge_intensity=40,
    min_bridge_textured_fraction=0.35,
    min_contact_fraction=0.15,
    max_particle_area_fraction=0.35,
):
   
    ordered = sorted(masks, key=lambda m: np.count_nonzero(m["segmentation"]), reverse=True)

    image_area = image_shape[0] * image_shape[1]
    max_absolute_area = max_particle_area_fraction * image_area

    kept = []            # list of boolean masks, largest first
    kept_sources = []    # the mask_dict each kept mask came from
    kept_original_area = []  # each entry's area at FIRST creation, never updated after merges

    kernel = None
    if bridge_gap_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * bridge_gap_px + 1, 2 * bridge_gap_px + 1),
        )

    gray, texture_map, texture_threshold = None, None, None
    if kernel is not None and raw_image_rgb is not None:
        gray = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2GRAY)
        texture_map = compute_local_texture_map(gray)
        texture_threshold = float(np.percentile(texture_map, 30))

    for mask_dict in ordered:

        seg = mask_dict["segmentation"].astype(bool)
        area = np.count_nonzero(seg)

        if area < min_area:
            continue

        probe = seg
        if kernel is not None:
            probe = cv2.dilate(seg.astype(np.uint8), kernel).astype(bool)

        absorbed = False

        for i, parent in enumerate(kept):

     
            if area > kept_original_area[i] * max_child_ratio:
                continue

            parent_current_area = np.count_nonzero(parent)
            if parent_current_area + area > max_absolute_area:
                continue

            direct_touch = np.count_nonzero(seg & parent) > 0
            bridged_only = (not direct_touch) and np.count_nonzero(probe & parent) > 0

            if not (direct_touch or bridged_only):
                continue

            if bridged_only and gray is not None:

             
                bridge_pixels = probe & ~seg & ~parent

                if np.count_nonzero(bridge_pixels) > 0:

                    bridge_intensity = float(np.mean(gray[bridge_pixels]))
                    bridge_textured = float(np.mean(
                        texture_map[bridge_pixels] >= texture_threshold
                    ))

                    if (bridge_intensity < min_bridge_intensity or
                            bridge_textured < min_bridge_textured_fraction):
                        continue

        
            contact_fraction = _contact_arc_fraction(seg, parent, bridge_gap_px + 1)

            if contact_fraction < min_contact_fraction:
                continue

            kept[i] = parent | seg
            absorbed = True
            break

        if not absorbed:
            kept.append(seg)
            kept_sources.append(mask_dict)
            kept_original_area.append(area)

    return _split_any_disconnected_masks(kept, min_area)


def _split_any_disconnected_masks(masks, min_area):

    result = []

    for mask in masks:

        n_labels, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

        if n_labels <= 2:  # background (0) + exactly one component
            result.append(mask)
            continue

        for label in range(1, n_labels):

            piece = components == label

            if np.count_nonzero(piece) >= min_area:
                result.append(piece)

    return _resolve_overlaps(result, min_area)


def _resolve_overlaps(masks, min_area):
   
    ordered = sorted(masks, key=lambda m: np.count_nonzero(m), reverse=True)

    claimed = np.zeros_like(ordered[0]) if ordered else None
    resolved = []

    for mask in ordered:

        exclusive = mask & ~claimed

        n_labels, components = cv2.connectedComponents(exclusive.astype(np.uint8), connectivity=8)

        for label in range(1, n_labels):

            piece = components == label

            if np.count_nonzero(piece) >= min_area:
                resolved.append(piece)

        claimed = claimed | mask

    return resolved


def best_source_mask_for(component, masks):
    
    best, best_overlap = None, 0

    for mask_dict in masks:

        overlap = np.count_nonzero(component & mask_dict["segmentation"].astype(bool))

        if overlap > best_overlap:
            best_overlap = overlap
            best = mask_dict

    return best if best is not None else {"predicted_iou": np.nan, "stability_score": np.nan, "area": np.nan}


def fill_mask_holes(
    mask,
    max_hole_area=200,
):

    mask = mask.astype(np.uint8)

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )

    flood_inv = cv2.bitwise_not(flood)

    holes = flood_inv & (
        (mask == 0).astype(np.uint8) * 255
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes,
        connectivity=8,
    )

    output = mask.copy()

    for i in range(1, n_labels):

        if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:

            output[labels == i] = 1

    return output.astype(bool)


def compute_raw_hole_stats(mask):

    mask = mask.astype(np.uint8)

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )

    flood_inv = cv2.bitwise_not(flood)

    holes = flood_inv & (
        (mask == 0).astype(np.uint8) * 255
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes,
        connectivity=8,
    )

    hole_count = 0
    hole_area = 0.0
    largest_hole_area = 0.0

    for i in range(1, n_labels):

        area = float(stats[i, cv2.CC_STAT_AREA])

        hole_count += 1
        hole_area += area
        largest_hole_area = max(largest_hole_area, area)

    mask_area = float(np.count_nonzero(mask))

    hole_area_ratio = hole_area / mask_area if mask_area > 0 else 0.0

    return {

        "raw_hole_count": hole_count,

        "raw_hole_area_px": hole_area,

        "raw_largest_hole_area_px": largest_hole_area,

        "raw_hole_area_ratio": hole_area_ratio,

    }


def smooth_mask(
    mask,
    kernel_size=3,
):

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (kernel_size, kernel_size),

    )

    mask = mask.astype(np.uint8)

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel,

    )

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel,

    )

    return mask.astype(bool)


# ============================================================
# BACKGROUND FILTER
# ============================================================

def compute_local_texture_map(gray, ksize=9):
   
    gray_f = gray.astype(np.float32)

    mean = cv2.boxFilter(gray_f, -1, (ksize, ksize))
    sq_mean = cv2.boxFilter(gray_f ** 2, -1, (ksize, ksize))

    local_var = np.clip(sq_mean - mean ** 2, 0, None)

    return np.sqrt(local_var)


def remove_background_masks(

    masks,

    image_rgb,

    min_mean_intensity=8,

    max_background_fraction=0.98,

    min_textured_fraction=0.35,

    texture_percentile=30,

):

    gray = cv2.cvtColor(

        image_rgb,

        cv2.COLOR_RGB2GRAY,

    )

    texture_map = compute_local_texture_map(gray)

    texture_threshold = float(np.percentile(texture_map, texture_percentile))

    filtered = []

    rejected = 0

    for m in masks:

        seg = m["segmentation"]

        pixels = gray[seg]

        if len(pixels) == 0:

            rejected += 1

            continue

        mean_intensity = float(

            np.mean(pixels)

        )

        background_fraction = float(

            np.mean(

                pixels < 5

            )

        )

        if mean_intensity < min_mean_intensity:

            rejected += 1

            continue

        if background_fraction > max_background_fraction:

            rejected += 1

            continue

     
        textured_fraction = float(

            np.mean(

                texture_map[seg] >= texture_threshold

            )

        )

        if textured_fraction < min_textured_fraction:

            rejected += 1

            continue

        filtered.append(m)

    return filtered, rejected


# ============================================================
# ADAPTIVE PARAMETERS
# ============================================================

def estimate_particle_density(

    masks,

    image_shape,

):

    h, w = image_shape[:2]

    image_area = h * w

    particle_area = sum(

        m["area"]

        for m in masks

    )

    return particle_area / image_area


def adaptive_parameters(

    masks,

    image_shape,

):

    coverage = estimate_particle_density(

        masks,

        image_shape,

    )

    params = {

        "hole_area": 200,

        "min_distance": 12,

        "quality_threshold": 0.45,

    }

    if coverage < 0.10:

        params["min_distance"] = 8

    elif coverage > 0.35:

        params["min_distance"] = 18

    if coverage > 0.50:

        params["quality_threshold"] = 0.55

    return params


# ============================================================
# WATERSHED SPLITTING
# ============================================================

def split_touching_particles(
    mask,
    min_distance=12,
    min_component_area=80,
    min_peak_prominence_ratio=0.55,
):


    mask = mask.astype(bool)

    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask,
    )

    if len(coords) <= 1:
        return [mask]

    peak_values = distance[coords[:, 0], coords[:, 1]]

    max_distance = float(peak_values.max())

    prominence_threshold = max_distance * min_peak_prominence_ratio

    significant = peak_values >= prominence_threshold

    coords = coords[significant]

    if len(coords) <= 1:
      
        return [mask]

    markers = np.zeros_like(
        distance,
        dtype=np.int32,
    )

    for i, (r, c) in enumerate(coords):

        markers[r, c] = i + 1

    markers = ndi.label(markers)[0]

    labels = watershed(
        -distance,
        markers,
        mask=mask,
    )

    components = []

    for label in np.unique(labels):

        if label == 0:
            continue

        component = labels == label

        if component.sum() < min_component_area:
            continue

        components.append(component)

    if len(components) == 0:
        return [mask]

    return components


# ============================================================
# MASK QUALITY
# ============================================================

def compute_mask_quality(
    mask_dict,
    contour,
):

    predicted_iou = mask_dict.get(
        "predicted_iou",
        0.0,
    )

    stability = mask_dict.get(
        "stability_score",
        0.0,
    )

    quality = (

        0.55 * predicted_iou +

        0.45 * stability

    )

    return float(quality)


# ============================================================
# PARTICLE CREATION
# ============================================================

def create_particle(

    particle_id,

    image_name,

    component,

    contour,

    mask_dict,

    raw_hole_stats=None,

):

    x, y, w, h = cv2.boundingRect(contour)

    M = cv2.moments(contour)

    if M["m00"] != 0:

        cx = M["m10"] / M["m00"]

        cy = M["m01"] / M["m00"]

    else:

        cx = x + w / 2

        cy = y + h / 2

    quality = compute_mask_quality(

        mask_dict,

        contour,

    )

    particle = Particle(

        particle_id=particle_id,

        image_name=image_name,

        mask=component,

        contour=contour,

        bbox=BoundingBox(

            x=x,

            y=y,

            width=w,

            height=h,

        ),

        centroid=Centroid(

            x=cx,

            y=cy,

        ),

        metadata={

            "predicted_iou": mask_dict.get(
                "predicted_iou",
                np.nan,
            ),

            "stability_score": mask_dict.get(
                "stability_score",
                np.nan,
            ),

            "sam2_area": mask_dict.get(
                "area",
                np.nan,
            ),

            "quality_score": quality,

            **(raw_hole_stats or {}),

        },

    )

    return particle, quality


# ============================================================
# SEGMENTATION ENGINE
# ============================================================

class SAM2Segmenter:

    def __init__(
        self,
        mask_generator,
        min_area=80,
        adaptive_segmentation=True,
        border_margin=5,
        edge_filtering=False,
        max_particle_area_fraction=0.35,
        whole_particle_mode=True,
        bridge_gap_px=2,
    ):

        self.mask_generator = mask_generator

        self.min_area = min_area

        self.adaptive_segmentation = adaptive_segmentation

        self.border_margin = border_margin

        self.edge_filtering = edge_filtering

        self.max_particle_area_fraction = max_particle_area_fraction
        self.whole_particle_mode = whole_particle_mode

        self.bridge_gap_px = bridge_gap_px

    def segment(
        self,
        image_rgb,
        image_name="image",
        raw_image_rgb=None,
    ):

        stats = SegmentationStatistics()

        masks = self.mask_generator.generate(
            image_rgb
        )

        stats.raw_masks = len(masks)

        masks, removed_duplicate, removed_contained = remove_duplicate_masks(
            masks
        )

        stats.rejected_overlap = removed_duplicate

        stats.rejected_contained = removed_contained

     
        masks, rejected = remove_background_masks(
            masks,
            raw_image_rgb if raw_image_rgb is not None else image_rgb,
        )

        stats.rejected_background = rejected

        if self.adaptive_segmentation:

            params = adaptive_parameters(
                masks,
                image_rgb.shape,
            )

        else:

            params = {

                "hole_area": 200,

                "min_distance": 12,

                "quality_threshold": 0.45,

            }

        particles = []

        particle_id = 1

        if self.whole_particle_mode:

            components = merge_into_whole_particles(
                masks,
                image_rgb.shape,
                bridge_gap_px=self.bridge_gap_px,
                min_area=self.min_area,
                raw_image_rgb=raw_image_rgb if raw_image_rgb is not None else image_rgb,
                max_particle_area_fraction=self.max_particle_area_fraction,
            )

            image_area = image_rgb.shape[0] * image_rgb.shape[1]

            for component in components:

                if np.count_nonzero(component) > self.max_particle_area_fraction * image_area:
                    stats.rejected_artifact += 1
                    continue

                source = best_source_mask_for(component, masks)

                raw_hole_stats = compute_raw_hole_stats(component)

                filled = fill_mask_holes(component, max_hole_area=params["hole_area"])
                filled = smooth_mask(filled).astype(np.uint8)

                contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)

                epsilon = max(0.001 * contour_perimeter(contour), 0.5)
                contour = cv2.approxPolyDP(contour, epsilon, True)

                if is_artifact_shape(
                    contour, image_rgb.shape,
                    border_margin=self.border_margin,
                    edge_filtering=self.edge_filtering,
                ):
                    stats.rejected_artifact += 1
                    continue

                particle, quality = create_particle(
                    particle_id, image_name, filled.astype(bool), contour, source,
                    raw_hole_stats=raw_hole_stats,
                )

                if quality < params["quality_threshold"]:
                    stats.rejected_quality += 1
                    continue

                touches_border, border_fraction, is_truncated = estimate_border_truncation(
                    filled.astype(bool), image_rgb.shape,
                )
                particle.metadata["touches_border"] = touches_border
                particle.metadata["border_fraction"] = border_fraction
                particle.metadata["border_truncated"] = is_truncated

                particles.append(particle)
                particle_id += 1

            stats.kept_masks = len(particles)

            return particles, stats

        for mask_dict in sorted(

            masks,

            key=lambda x: x["area"],

            reverse=True,

        ):

            
            raw_hole_stats = compute_raw_hole_stats(
                mask_dict["segmentation"],
            )

            seg = fill_mask_holes(

                mask_dict["segmentation"],

                max_hole_area=params["hole_area"],

            )

            seg = smooth_mask(seg)

            if np.count_nonzero(seg) < self.min_area:

                stats.rejected_small += 1

                continue

            image_area = image_rgb.shape[0] * image_rgb.shape[1]

            if np.count_nonzero(seg) > self.max_particle_area_fraction * image_area:

                stats.rejected_artifact += 1

                continue

            split_masks = split_touching_particles(

                seg,

                min_distance=params["min_distance"],

            )

            if len(split_masks) > 1:

                stats.split_particles += (

                    len(split_masks) - 1

                )

            for component in split_masks:

                component = component.astype(np.uint8)

                contours, _ = cv2.findContours(

                    component,

                    cv2.RETR_EXTERNAL,

                    cv2.CHAIN_APPROX_NONE,

                )

                if len(contours) == 0:

                    continue

                contour = max(

                    contours,

                    key=cv2.contourArea,

                )

                epsilon = max(

                    0.001 * contour_perimeter(

                        contour

                    ),

                    0.5,

                )

                contour = cv2.approxPolyDP(

                    contour,

                    epsilon,

                    True,

                )

                if is_artifact_shape(

                    contour,

                    image_rgb.shape,

                    border_margin=self.border_margin,

                    edge_filtering=self.edge_filtering,

                ):

                    stats.rejected_artifact += 1

                    continue

                particle, quality = create_particle(

                    particle_id,

                    image_name,

                    component,

                    contour,

                    mask_dict,

                    raw_hole_stats=raw_hole_stats,

                )

                if quality < params["quality_threshold"]:

                    stats.rejected_quality += 1

                    continue

                touches_border, border_fraction, is_truncated = estimate_border_truncation(
                    component, image_rgb.shape,
                )
                particle.metadata["touches_border"] = touches_border
                particle.metadata["border_fraction"] = border_fraction
                particle.metadata["border_truncated"] = is_truncated

                particles.append(

                    particle

                )

                particle_id += 1

        particles, removed_final_dup = remove_contained_final_particles(particles)

        stats.rejected_overlap += removed_final_dup

        stats.kept_masks = len(

            particles

        )

        return particles, stats


# ============================================================
# PUBLIC API
# ============================================================

    def generate_masks(
        self,
        image_rgb,
    ):
       

        return self.mask_generator.generate(image_rgb)

    def segment_single_mask(
        self,
        mask_dict,
        image_name="image",
    ):
       

        params = {

            "hole_area": 200,

            "min_distance": 12,

            "quality_threshold": 0.45,

        }

        seg = fill_mask_holes(
            mask_dict["segmentation"],
            max_hole_area=params["hole_area"],
        )

        seg = smooth_mask(seg)

        particles = []

        particle_id = 1

        split_masks = split_touching_particles(
            seg,
            min_distance=params["min_distance"],
        )

        for component in split_masks:

            component = component.astype(np.uint8)

            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            if len(contours) == 0:
                continue

            contour = max(
                contours,
                key=cv2.contourArea,
            )

            epsilon = max(
                0.001 * contour_perimeter(contour),
                0.5,
            )

            contour = cv2.approxPolyDP(
                contour,
                epsilon,
                True,
            )

            particle, quality = create_particle(
                particle_id,
                image_name,
                component,
                contour,
                mask_dict,
            )

            if quality < params["quality_threshold"]:
                continue

            touches_border, border_fraction, is_truncated = estimate_border_truncation(
                component, image_rgb.shape,
            )
            particle.metadata["touches_border"] = touches_border
            particle.metadata["border_fraction"] = border_fraction
            particle.metadata["border_truncated"] = is_truncated

            particles.append(
                particle
            )

            particle_id += 1

        return particles


# ============================================================
# FACTORY
# ============================================================

def build_segmenter(
    checkpoint_path,
    model_cfg,
    device="cpu",
    min_area=80,
    border_margin=5,
    edge_filtering=False,
    max_particle_area_fraction=0.35,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
):
   

    generator = load_sam2_model(

        checkpoint_path=checkpoint_path,

        model_cfg=model_cfg,

        device=device,

        points_per_side=points_per_side,

        pred_iou_thresh=pred_iou_thresh,

        stability_score_thresh=stability_score_thresh,

    )

    return SAM2Segmenter(

        mask_generator=generator,

        min_area=min_area,

        adaptive_segmentation=True,

        border_margin=border_margin,

        edge_filtering=edge_filtering,

        max_particle_area_fraction=max_particle_area_fraction,

    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    "SegmentationStatistics",

    "SAM2Segmenter",

    "build_segmenter",

    "load_sam2_model",

]
