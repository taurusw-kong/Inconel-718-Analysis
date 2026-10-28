"""
ParticleVision V4
=================

Morphology Feature Extraction Engine

Extracts geometric/shape descriptors from segmented particle masks:
size, circularity, solidity, aspect ratio, boundary roughness, radial
asymmetry (comet-tail signature), and localized-neck/dent detection
(satellite-attachment signal). These are geometric primitives only,
computed the same way for every particle regardless of class, v4's
classifier (classification.py) is what turns them into a prediction,
not this module.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from scipy.spatial.distance import pdist
from scipy import stats as sp_stats

from skimage.measure import label, regionprops

from data_models import Particle


class MorphologyExtractor:
    """
    Extract morphology descriptors from a particle mask.
    """

    def __init__(self):
        pass

    def extract(self, particle: Particle):

        if particle.mask is None:
            raise ValueError("Particle mask is None.")

        if particle.contour is None:
            raise ValueError("Particle contour is None.")

        mask = particle.mask.astype(np.uint8)

        contour = particle.contour

        self._geometry(mask, contour, particle)
        self._shape(contour, particle)
        self._feret(contour, particle)
        self._convexity_defects(contour, particle)
        self._holes(mask, particle)
        self._boundary(contour, particle)
        self._radial_profile(contour, particle)
        self._neck_and_dent(particle)
        self._derived(mask, particle)

        return particle

    # ==========================================================
    # GEOMETRY
    # ==========================================================

    def _geometry(self, mask, contour, particle):

        area = float(cv2.contourArea(contour))

        perimeter = float(
            cv2.arcLength(contour, True)
        )

        x, y, w, h = cv2.boundingRect(contour)

        bbox_area = float(w * h)

        equivalent_diameter = (
            np.sqrt(4.0 * area / np.pi)
            if area > 0
            else np.nan
        )

        extent = (
            area / bbox_area
            if bbox_area > 0
            else np.nan
        )

        particle.morphology.set("area", area)
        particle.morphology.set("perimeter", perimeter)
        particle.morphology.set("equivalent_diameter", equivalent_diameter)
       
        particle.morphology.set("bbox_x", x)
        particle.morphology.set("bbox_y", y)
        particle.morphology.set("bbox_width", w)
        particle.morphology.set("bbox_height", h)
        particle.morphology.set("bbox_area", bbox_area)
        particle.morphology.set("extent", extent)

    # ==========================================================
    # SHAPE DESCRIPTORS
    # ==========================================================

    def _shape(self, contour, particle):

        area = particle.morphology.get("area")
        perimeter = particle.morphology.get("perimeter")

        circularity = (
            (4.0 * np.pi * area) / (perimeter ** 2)
            if perimeter > 0
            else np.nan
        )

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perimeter = float(cv2.arcLength(hull, True))

        solidity = area / hull_area if hull_area > 0 else np.nan
        convexity = hull_perimeter / perimeter if perimeter > 0 else np.nan
        compactness = (perimeter ** 2) / area if area > 0 else np.nan

        if len(contour) >= 5:

            ellipse = cv2.fitEllipse(contour)
            (_, _), (major, minor), angle = ellipse

            if major < minor:
                major, minor = minor, major

            aspect_ratio = major / minor if minor > 0 else np.nan
            eccentricity = np.sqrt(1.0 - (minor ** 2) / (major ** 2))

        else:

            major = np.nan
            minor = np.nan
            angle = np.nan
            aspect_ratio = np.nan
            eccentricity = np.nan

        particle.morphology.set("circularity", circularity)
        particle.morphology.set("solidity", solidity)
        particle.morphology.set("convexity", convexity)
        particle.morphology.set("compactness", compactness)
        particle.morphology.set("major_axis", major)
        particle.morphology.set("minor_axis", minor)
        particle.morphology.set("aspect_ratio", aspect_ratio)
        particle.morphology.set("eccentricity", eccentricity)
        particle.morphology.set("orientation", angle)

    # ==========================================================
    # FERET DIAMETERS
    # ==========================================================

    def _feret(self, contour, particle):

        points = contour.reshape(-1, 2).astype(np.float32)

        if len(points) < 2:
            particle.morphology.set("max_feret", np.nan)
            particle.morphology.set("min_feret", np.nan)
            particle.morphology.set("feret_ratio", np.nan)
            return

        max_feret = float(np.max(pdist(points)))

        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect

        min_feret = float(min(w, h))
        feret_ratio = max_feret / min_feret if min_feret > 0 else np.nan

        particle.morphology.set("max_feret", max_feret)
        particle.morphology.set("min_feret", min_feret)
        particle.morphology.set("feret_ratio", feret_ratio)

    # ==========================================================
    # CONVEX HULL & CONVEXITY DEFECTS
    # ==========================================================

    def _convexity_defects(self, contour, particle):

        hull = cv2.convexHull(contour, returnPoints=False)

        defect_count = 0
        defect_depth_sum = 0.0
        defect_depth_max = 0.0
        defect_depths = []

        if hull is not None and len(hull) >= 3 and len(contour) >= 4:

            try:

                hull_sorted = np.sort(hull, axis=0)

                defects = cv2.convexityDefects(contour, hull_sorted)

            except cv2.error:

                defects = None

            if defects is not None:

                defects = np.asarray(defects).reshape(-1, 4)

                defect_count = len(defects)

                for d in defects:

                    depth = float(d[3]) / 256.0
                    defect_depths.append(depth)
                    defect_depth_sum += depth
                    defect_depth_max = max(defect_depth_max, depth)

        defect_depth_mean = (
            defect_depth_sum / defect_count if defect_count > 0 else 0.0
        )

        particle.morphology.set("convexity_defect_count", defect_count)
        particle.morphology.set("convexity_defect_mean_depth", defect_depth_mean)
        particle.morphology.set("convexity_defect_max_depth", defect_depth_max)

       
        particle.metadata["_defect_depths"] = defect_depths

    # ==========================================================
    # HOLE / POROSITY FEATURES
    # ==========================================================

    def _holes(self, mask, particle):

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )

        area = particle.morphology.get("area")

        hole_count = 0
        hole_area = 0.0
        largest_hole = 0.0

        if hierarchy is not None:

            hierarchy = hierarchy[0]

            for i, h in enumerate(hierarchy):

                if h[3] != -1:

                    a = float(cv2.contourArea(contours[i]))
                    hole_count += 1
                    hole_area += a
                    largest_hole = max(largest_hole, a)

        hole_ratio = hole_area / area if area > 0 else np.nan
        largest_hole_ratio = largest_hole / area if area > 0 else np.nan

        particle.morphology.set("hole_count", hole_count)
        particle.morphology.set("hole_area", hole_area)
        particle.morphology.set("hole_area_ratio", hole_ratio)
        particle.morphology.set("largest_hole_area", largest_hole)
        particle.morphology.set("largest_hole_ratio", largest_hole_ratio)

    # ==========================================================
    # BOUNDARY ROUGHNESS & CURVATURE
    # ==========================================================

    def _boundary(self, contour, particle):

        area = particle.morphology.get("area")
        perimeter = particle.morphology.get("perimeter")

        hull = cv2.convexHull(contour)
        hull_perimeter = float(cv2.arcLength(hull, True))

        boundary_complexity = (
            perimeter / np.sqrt(area) if area > 0 else np.nan
        )

        roughness_index = (
            perimeter / hull_perimeter if hull_perimeter > 0 else np.nan
        )

        pts = contour.reshape(-1, 2).astype(np.float64)

        curvature = []

        if len(pts) >= 5:

            for i in range(len(pts)):

                p0 = pts[(i - 1) % len(pts)]
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]

                v1 = p0 - p1
                v2 = p2 - p1

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 0 and n2 > 0:

                    c = np.dot(v1, v2) / (n1 * n2)
                    c = np.clip(c, -1.0, 1.0)
                    curvature.append(np.arccos(c))

        if len(curvature):

            curvature = np.asarray(curvature)
            c_mean = float(np.mean(curvature))
            c_std = float(np.std(curvature))
            c_max = float(np.max(curvature))
            c_min = float(np.min(curvature))
            waviness = c_std / c_mean if c_mean > 0 else np.nan

        else:

            c_mean = np.nan
            c_std = np.nan
            c_max = np.nan
            c_min = np.nan
            waviness = np.nan

        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        particle.morphology.set("boundary_complexity", boundary_complexity)
        particle.morphology.set("roughness_index", roughness_index)
        particle.morphology.set("polygon_vertices", len(approx))
        particle.morphology.set("curvature_mean", c_mean)
        particle.morphology.set("curvature_std", c_std)
        particle.morphology.set("curvature_max", c_max)
        particle.morphology.set("curvature_min", c_min)
        particle.morphology.set("boundary_waviness", waviness)

    # ==========================================================
    # RADIAL PROFILE (asymmetry / "comet tail" signature)
    # ==========================================================

    def _radial_profile(self, contour, particle):
       
        pts = contour.reshape(-1, 2).astype(np.float64)

        if len(pts) < 5:
            particle.morphology.set("radial_skewness", np.nan)
            particle.morphology.set("radial_kurtosis", np.nan)
            particle.morphology.set("radial_tail_ratio", np.nan)
            return

        centroid = pts.mean(axis=0)
        distances = np.linalg.norm(pts - centroid, axis=1)

        median_d = float(np.median(distances))
        max_d = float(np.max(distances))

        skewness = float(sp_stats.skew(distances, bias=False))
        kurtosis = float(sp_stats.kurtosis(distances, bias=False))
        tail_ratio = max_d / median_d if median_d > 0 else np.nan

        particle.morphology.set("radial_skewness", skewness)
        particle.morphology.set("radial_kurtosis", kurtosis)
        particle.morphology.set("radial_tail_ratio", tail_ratio)

    # ==========================================================
    # NECK / DENT LOCALISATION
    # ==========================================================

    def _neck_and_dent(self, particle):
     
        depths = particle.metadata.pop("_defect_depths", [])
        equiv_d = particle.morphology.get("equivalent_diameter")

        if len(depths) == 0 or not equiv_d or equiv_d <= 0:
            particle.morphology.set("neck_depth_ratio", 0.0)
            particle.morphology.set("concavity_locality", np.nan)
            particle.morphology.set("significant_defect_count", 0.0)
            return

        depths = np.asarray(depths, dtype=float)
        max_depth = float(depths.max())
        mean_depth = float(depths.mean())

        neck_depth_ratio = max_depth / equiv_d
        concavity_locality = (
            max_depth / mean_depth if mean_depth > 0 else np.nan
        )

        significant_threshold = max(1.0, 0.015 * equiv_d)

        significant_defect_count = float(
            np.sum(depths > significant_threshold)
        )

        particle.morphology.set("neck_depth_ratio", neck_depth_ratio)
        particle.morphology.set("concavity_locality", concavity_locality)
        particle.morphology.set(
            "significant_defect_count", significant_defect_count
        )

    # ==========================================================
    # FRACTAL & DERIVED FEATURES
    # ==========================================================

    def _derived(self, mask, particle):

        area = particle.morphology.get("area")
        circularity = particle.morphology.get("circularity")
        solidity = particle.morphology.get("solidity")
        major = particle.morphology.get("major_axis")
        minor = particle.morphology.get("minor_axis")
        roughness = particle.morphology.get("roughness_index")
        waviness = particle.morphology.get("boundary_waviness")

        elongation = (
            major / minor
            if (not np.isnan(major) and not np.isnan(minor) and minor > 0)
            else np.nan
        )

        circularity_error = (
            abs(1.0 - circularity) if not np.isnan(circularity) else np.nan
        )

        shape_factor = (
            solidity * circularity
            if (not np.isnan(solidity) and not np.isnan(circularity))
            else np.nan
        )

        surface_irregularity = (
            roughness * waviness
            if (not np.isnan(roughness) and not np.isnan(waviness))
            else np.nan
        )

        mask_pixels = int(np.count_nonzero(mask))
        fill_ratio = area / mask_pixels if mask_pixels > 0 else np.nan

        particle.morphology.set("elongation_index", elongation)
        particle.morphology.set("circularity_error", circularity_error)
        particle.morphology.set("shape_factor", shape_factor)
        particle.morphology.set("surface_irregularity", surface_irregularity)
        particle.morphology.set("mask_pixels", mask_pixels)
        particle.morphology.set("fill_ratio", fill_ratio)

        particle.metadata["morphology_feature_count"] = len(
            particle.morphology.values
        )


# ==========================================================
# CALIBRATION (pixels -> real microns)
# ==========================================================

def apply_calibration(particle: Particle, microns_per_pixel: float):
   

    if microns_per_pixel is None or microns_per_pixel <= 0:
        return particle

    equiv_px = particle.morphology.get("equivalent_diameter")

    if equiv_px is not None and not np.isnan(equiv_px):
        particle.size_um = float(equiv_px * microns_per_pixel)

    for px_name, um_name in (
        ("equivalent_diameter", "equivalent_diameter_um"),
        ("max_feret", "max_feret_um"),
        ("min_feret", "min_feret_um"),
        ("major_axis", "major_axis_um"),
        ("minor_axis", "minor_axis_um"),
    ):
        px_value = particle.morphology.get(px_name)

        if px_value is not None and not np.isnan(px_value):
            particle.morphology.set(um_name, px_value * microns_per_pixel)

    particle.metadata["microns_per_pixel"] = microns_per_pixel

    return particle
