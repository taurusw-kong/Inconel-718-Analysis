"""
ParticleVision V4 — Physics Feature Set v2
==========================================

"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ============================================================
# GEOMETRIC PRIMITIVES
# ============================================================

def _distance_transform(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def inscribed_radius_px(mask: np.ndarray) -> float:
    """Largest inscribed circle radius = the particle's 'core' radius R."""
    dt = _distance_transform(mask)
    return float(dt.max()) if dt.size else 0.0


def neck_radius_px(mask: np.ndarray, max_erode: int = 60):
   
    m = mask.astype(np.uint8)
    if m.sum() < 30:
        return None, None, None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = m.copy()

    for step in range(1, max_erode + 1):
        eroded = cv2.erode(eroded, kernel, iterations=1)
        if eroded.sum() == 0:
            return None, None, None

        n_lab, labels = cv2.connectedComponents(eroded)
        # n_lab counts background, so >2 means the mask has split
        if n_lab - 1 >= 2:
            areas = [(labels == i).sum() for i in range(1, n_lab)]
            areas.sort(reverse=True)
            big, small = areas[0], areas[1]
            if small < 8:                 # noise speck, keep eroding
                continue
            r_large = math.sqrt(big / math.pi)
            r_small = math.sqrt(small / math.pi)
            return float(step), float(r_small), float(r_large)

    return None, None, None


def radial_signature(contour, centroid, n_samples: int = 128):
    """Radius as a function of angle, resampled uniformly."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    cx, cy = centroid
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    r = np.hypot(dx, dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    order = np.argsort(theta)
    theta, r = theta[order], r[order]
    grid = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    return np.interp(grid, theta, r, period=2 * np.pi)


def granulometry_fractions(mask: np.ndarray, scales=(0.10, 0.20, 0.35)):
 
    R = inscribed_radius_px(mask)
    m = mask.astype(np.uint8)
    total = float(m.sum())
    out = []

    for frac in scales:
        rad = max(int(round(R * frac)), 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        removed = (total - float(opened.sum())) / total if total > 0 else np.nan
        out.append(removed)

    return out


# ============================================================
# FEATURE COMPUTATION
# ============================================================

def compute_v2_features(particle, constants, use_dimensional: bool = True) -> dict:

    f = {}
    mask = particle.mask
    if mask is None or particle.contour is None:
        return f

    mpp = particle.metadata.get("microns_per_pixel")
    mask_u8 = mask.astype(np.uint8)
    area_px = float(mask_u8.sum())
    if area_px <= 0:
        return f

    # ---------- shared geometry ----------
    R_px = inscribed_radius_px(mask)
    neck_px, r_small_px, r_large_px = neck_radius_px(mask)

    hull = cv2.convexHull(particle.contour)
    hull_area = float(cv2.contourArea(hull)) or np.nan
    solidity = area_px / hull_area if hull_area and hull_area > 0 else np.nan

    cx, cy = particle.centroid.x, particle.centroid.y
    sig = radial_signature(particle.contour, (cx, cy))
    sig_mean = float(np.mean(sig)) if sig.size else np.nan

    # ============================================================
    # GROUP A — dimensionless, no material constants
    # ============================================================


    if neck_px is not None and R_px > 0:
        f["ratio_neck_to_radius"] = neck_px / R_px
        f["ratio_pair_diameter"] = (r_small_px / r_large_px) if r_large_px else np.nan
        f["ratio_has_neck"] = 1.0
    else:
        f["ratio_neck_to_radius"] = 0.0      # no neck detected = single body
        f["ratio_pair_diameter"] = np.nan
        f["ratio_has_neck"] = 0.0

    # Multi-scale protrusion mass (satellites vs major lobes)
    g = granulometry_fractions(mask)
    f["ratio_granulometry_fine"] = g[0]
    f["ratio_granulometry_mid"] = g[1]
    f["ratio_granulometry_coarse"] = g[2]
 
    f["ratio_fine_to_coarse"] = (g[0] / g[2]) if g[2] and g[2] > 1e-6 else np.nan

   
    if sig.size and sig_mean and sig_mean > 0:
        spec = np.abs(np.fft.rfft(sig - sig_mean)) / (len(sig) * sig_mean)
        f["ratio_fourier_elongation"] = float(spec[2]) if len(spec) > 2 else np.nan
        f["ratio_fourier_triangularity"] = float(spec[3]) if len(spec) > 3 else np.nan
        f["ratio_fourier_roughness"] = float(np.sum(spec[5:20])) if len(spec) > 5 else np.nan
        f["ratio_radial_cv"] = float(np.std(sig) / sig_mean)
        rough = f.get("ratio_fourier_roughness")
        elong = f.get("ratio_fourier_elongation")
        f["ratio_rough_to_elong"] = (rough / elong) if elong and elong > 1e-9 else np.nan
    else:
        for k in ("ratio_fourier_elongation", "ratio_fourier_triangularity",
                  "ratio_fourier_roughness", "ratio_radial_cv", "ratio_rough_to_elong"):
            f[k] = np.nan


    eps = (1.0 - solidity) if solidity == solidity else np.nan
    f["ratio_void_fraction"] = eps
    f["ratio_coordination_est"] = (math.pi / eps) if eps and eps > 0.02 else np.nan

 
    perim = float(cv2.arcLength(particle.contour, True)) or np.nan
    wadell = (2.0 * math.sqrt(math.pi * area_px) / perim) if perim and perim > 0 else np.nan
    f["ratio_wadell_sphericity"] = wadell

    # ============================================================
    # GROUP B — dimensional, uses config.PHYSICS constants
    # ============================================================
    if use_dimensional:
        gamma = getattr(constants, "surface_tension_n_per_m", np.nan)

        if mpp and mpp > 0:
            R_m = R_px * mpp * 1e-6
            neck_m = (neck_px * mpp * 1e-6) if neck_px is not None else None

      
            if neck_m and neck_m > 0 and R_m > 0:
                f["dim_sintering_stress_pa"] = gamma * (1.0 / neck_m - 1.0 / R_m)
            else:
                f["dim_sintering_stress_pa"] = np.nan

  
            if r_small_px and r_large_px:
                d1 = 2 * r_small_px * mpp * 1e-6
                d2 = 2 * r_large_px * mpp * 1e-6
                d_harm = 2 * d1 * d2 / (d1 + d2) if (d1 + d2) > 0 else np.nan
                f["dim_capillary_bond_force_n"] = math.pi * gamma * d_harm
            else:
                f["dim_capillary_bond_force_n"] = np.nan

            d_eq_m = 2.0 * math.sqrt(area_px / math.pi) * mpp * 1e-6
            F = f.get("dim_capillary_bond_force_n")
            if (eps == eps) and eps and eps > 0.02 and d_eq_m > 0 and F == F and F:
                f["dim_rumpf_strength_pa"] = (9.0 / 8.0) * ((1 - eps) / eps) * F / (d_eq_m ** 2)
            else:
                f["dim_rumpf_strength_pa"] = np.nan
        else:
            for k in ("dim_sintering_stress_pa", "dim_capillary_bond_force_n",
                      "dim_rumpf_strength_pa"):
                f[k] = np.nan

    return f


__all__ = ["compute_v2_features", "neck_radius_px", "inscribed_radius_px",
           "radial_signature", "granulometry_fractions"]
