"""
ParticleVision V4
=================

Physics-Constrained Feature Derivation

"""

from __future__ import annotations

import math

import numpy as np

from data_models import Particle


# ============================================================
# COMPOSITE SHAPE DESCRIPTORS (carried over from v3, validated)
# ============================================================

class PhysicsDescriptorExtractor:

    def __init__(self):
        pass

    def extract(self, particle: Particle):

        m = particle.morphology

        circularity = m.get("circularity")
        solidity = m.get("solidity")
        aspect_ratio = m.get("aspect_ratio")
        feret_ratio = m.get("feret_ratio")
        convexity = m.get("convexity")
        roughness = m.get("roughness_index")
        waviness = m.get("boundary_waviness")

        raw_hole_ratio = particle.metadata.get("raw_hole_area_ratio")
        hole_ratio = (
            raw_hole_ratio
            if raw_hole_ratio is not None
            else m.get("hole_area_ratio")
        )

        defect_count = m.get("significant_defect_count")
        defect_depth = m.get("convexity_defect_mean_depth")
        elongation = m.get("elongation_index")
        fractal = m.get("fractal_dimension")
        curvature_std = m.get("curvature_std")
        circularity_error = m.get("circularity_error")
        surface_irregularity = m.get("surface_irregularity")
        equiv_diameter = m.get("equivalent_diameter")

        neck_depth_ratio = m.get("neck_depth_ratio")
        concavity_locality = m.get("concavity_locality")
        radial_tail_ratio = m.get("radial_tail_ratio")
        min_feret = m.get("min_feret")
        max_feret = m.get("max_feret")

        defect_count_norm = (
            min(defect_count, 6.0) / 6.0
            if defect_count is not None and not np.isnan(defect_count)
            else np.nan
        )

        defect_depth_norm = (
            defect_depth / equiv_diameter
            if (
                defect_depth is not None and not np.isnan(defect_depth)
                and equiv_diameter and not np.isnan(equiv_diameter)
                and equiv_diameter > 0
            )
            else np.nan
        )

        sphericity = np.nanmean([circularity, solidity, convexity])
        elongation_score = np.nanmean([aspect_ratio, feret_ratio, elongation])
        porosity_index = np.nanmean([hole_ratio, circularity_error])
        rough_surface_index = np.nanmean([roughness, curvature_std, surface_irregularity])
        dendritic_index = np.nanmean([rough_surface_index, defect_count_norm, fractal])
        fragmentation_index = np.nanmean([circularity_error, elongation_score, defect_depth_norm])

        quality_score = np.nanmean([
            sphericity,
            1.0 / (1.0 + porosity_index) if not np.isnan(porosity_index) else np.nan,
            1.0 / (1.0 + rough_surface_index) if not np.isnan(rough_surface_index) else np.nan,
        ])

        neck_score = neck_depth_ratio if neck_depth_ratio is not None else np.nan
        dent_locality = concavity_locality

        tail_excess = (
            (radial_tail_ratio - 1.0)
            if radial_tail_ratio is not None and not np.isnan(radial_tail_ratio)
            else np.nan
        )
        comet_score = tail_excess

        flatness_index = (
            1.0 - (min_feret / max_feret)
            if (
                min_feret is not None and max_feret is not None
                and not np.isnan(min_feret) and not np.isnan(max_feret)
                and max_feret > 0
            )
            else np.nan
        )

        particle.physics.set("sphericity_score", sphericity)
        particle.physics.set("elongation_score", elongation_score)
        particle.physics.set("porosity_index", porosity_index)
        particle.physics.set("rough_surface_index", rough_surface_index)
        particle.physics.set("dendritic_index", dendritic_index)
        particle.physics.set("fragmentation_index", fragmentation_index)
        particle.physics.set("quality_score", quality_score)
        particle.physics.set("neck_score", neck_score)
        particle.physics.set("dent_locality", dent_locality)
        particle.physics.set("comet_score", comet_score)
        particle.physics.set("flatness_index", flatness_index)

        return particle


# ============================================================
# NAMED PHYSICS EQUATIONS (calibration-driven, new in v4)
# ============================================================

def compute_named_physics_formulas(particle: Particle, constants) -> Particle:
    """
    Requires morphology.apply_calibration(particle, microns_per_pixel)
    to have already run — reads particle.metadata["microns_per_pixel"]
    and particle.size_um. If calibration hasn't been applied, every
    value here is set to NaN rather than silently computed on raw
    pixel counts mislabeled as microns.
    """

    microns_per_pixel = particle.metadata.get("microns_per_pixel")

    if microns_per_pixel is None or microns_per_pixel <= 0:
        for name in (
            "laplace_pore_radius_um", "laplace_pressure_pa",
            "weber_number", "griffith_crack_length_um",
            "griffith_critical_stress_pa",
        ):
            particle.physics.set(name, np.nan)
        return particle

    # ---------------------------------------------------
    # 1. LAPLACE PRESSURE — porosity plausibility (dP = 2*gamma/r)
    # ---------------------------------------------------


    largest_hole_area_px = particle.metadata.get("raw_largest_hole_area_px")

    if largest_hole_area_px is None:
        largest_hole_area_px = particle.morphology.get("largest_hole_area")

    if largest_hole_area_px is not None and not np.isnan(largest_hole_area_px) and largest_hole_area_px > 0:

        largest_hole_area_um2 = largest_hole_area_px * (microns_per_pixel ** 2)
        pore_radius_um = math.sqrt(largest_hole_area_um2 / math.pi)
        pore_radius_m = pore_radius_um * 1e-6

        laplace_pressure_pa = (
            2.0 * constants.surface_tension_n_per_m / pore_radius_m
            if pore_radius_m > 0
            else np.nan
        )

    else:
        pore_radius_um = 0.0
        laplace_pressure_pa = np.nan  # no measurable pore -> not applicable

    particle.physics.set("laplace_pore_radius_um", pore_radius_um)
    particle.physics.set("laplace_pressure_pa", laplace_pressure_pa)

    # ---------------------------------------------------
    # 2. WEBER NUMBER — satellite-formation-likelihood proxy
    #    (We = rho * v^2 * d / gamma), reference-condition velocity.
    # ---------------------------------------------------

    if particle.size_um is not None and not np.isnan(particle.size_um) and particle.size_um > 0:

        d_m = particle.size_um * 1e-6

        weber_number = (
            constants.melt_density_kg_per_m3
            * (constants.reference_ejection_velocity_m_per_s ** 2)
            * d_m
            / constants.surface_tension_n_per_m
        )

    else:
        weber_number = np.nan

    particle.physics.set("weber_number", weber_number)

    # ---------------------------------------------------
    # 3. GRIFFITH / K_IC CRACK RISK
    #    (sigma_critical = K_IC / sqrt(pi * c))
    # ---------------------------------------------------


    defect_depth_px = particle.morphology.get("convexity_defect_max_depth")

    if defect_depth_px is not None and not np.isnan(defect_depth_px) and defect_depth_px > 0:

        crack_length_um = defect_depth_px * microns_per_pixel
        crack_length_m = crack_length_um * 1e-6

        k_ic_pa_sqrt_m = constants.fracture_toughness_mpa_sqrt_m * 1e6

        critical_stress_pa = (
            k_ic_pa_sqrt_m / math.sqrt(math.pi * crack_length_m)
            if crack_length_m > 0
            else np.nan
        )

    else:
        crack_length_um = 0.0
        critical_stress_pa = np.nan

    particle.physics.set("griffith_crack_length_um", crack_length_um)
    particle.physics.set("griffith_critical_stress_pa", critical_stress_pa)

    return particle


__all__ = ["PhysicsDescriptorExtractor", "compute_named_physics_formulas"]
