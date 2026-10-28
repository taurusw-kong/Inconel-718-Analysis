"""
ParticleVision V4 Configuration
================================
Central configuration file for the entire framework.

Everything a non-programmer should be able to tune lives here as plain,
flat, editable values, segmentation thresholds, the defect taxonomy
(names + colors + severity), physics constants, and the reusability
score weights. gui_app.py reads and writes this module's dataclasses
directly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------
# PROJECT
# ---------------------------------------------------------------------

PROJECT_NAME = "ParticleVision V4"
VERSION = "4.0.0"


# ---------------------------------------------------------------------
# SEGMENTATION
# ---------------------------------------------------------------------

@dataclass
class SegmentationConfig:

    points_per_side: int = 48
    pred_iou_thresh: float = 0.75
    stability_score_thresh: float = 0.85
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 2

    min_particle_area_px: int = 80
    border_margin: int = 5
    edge_filtering: bool = False


# ---------------------------------------------------------------------
# MORPHOLOGY
# ---------------------------------------------------------------------

@dataclass
class MorphologyConfig:

    smoothing_kernel: int = 3
    gaussian_kernel: int = 5
    min_contour_points: int = 20


# ---------------------------------------------------------------------
# ATTACHMENT DETECTION
# ---------------------------------------------------------------------


@dataclass
class AttachmentConfig:

    contact_gap_px: int = 3

    min_contact_fraction: float = 0.03

   


# ---------------------------------------------------------------------
# DEFECT TAXONOMY  (Inconel-718 / powder-bed reuse classification)
# ---------------------------------------------------------------------


DEFECT_TAXONOMY: List[str] = [
    "Perfect_Spherical",     # 20-30 um, circularity/solidity ~1, unattached
    "Oversized_Spherical",   # > 30 um, circularity/solidity ~1, unattached
    "Elongated_Particle",    # high aspect ratio, unattached
    "Clip_Cap",               # partial/truncated sphere ("cap") shape
    "Molten_Satellite",      # < 3 um, attached to a host (circular or not — see note)
    "Bonded_Particle",       # > 3 um, fused to one or more comparable particles
    "Agglomerate",           # multi-particle fused cluster
    "Deformed_Particle",     # irregular shape, unattached
]



DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = {
 
    "Perfect_Spherical":   (0, 131, 0),      # green
    "Oversized_Spherical": (237, 161, 0),    # yellow
    "Elongated_Particle":  (138, 154, 31),   # olive
    "Clip_Cap":            (227, 73, 72),    # red
    "Molten_Satellite":    (42, 120, 214),   # blue
    "Bonded_Particle":     (74, 58, 167),    # violet
    "Agglomerate":         (168, 63, 174),   # purple-pink
    "Deformed_Particle":   (160, 90, 32),    # rust brown
}

DEFECT_SEVERITY: Dict[str, float] = {
    "Perfect_Spherical":   0,
    "Oversized_Spherical": 15,
    "Elongated_Particle":  35,
    "Clip_Cap":            55,
    "Molten_Satellite":    35,   # midpoint of the pre-merge 25 (circular) / 45 (irregular) priors
    "Bonded_Particle":     55,
    "Agglomerate":         80,
    "Deformed_Particle":   40,
}

BORDER_TRUNCATED_LABEL = "Edge_Truncated"
BORDER_TRUNCATED_COLOR: Tuple[int, int, int] = (255, 140, 0)  # orange (RGB)


@dataclass
class ClassificationConfig:

    labels: List[str] = field(default_factory=lambda: list(DEFECT_TAXONOMY))

    colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFECT_COLORS)
    )

    severity: Dict[str, float] = field(
        default_factory=lambda: dict(DEFECT_SEVERITY)
    )


    perfect_spherical_min_um: float = 20.0
    perfect_spherical_max_um: float = 30.0
    oversized_spherical_min_um: float = 30.0
    satellite_max_um: float = 3.0

    
    min_prototype_examples: int = 8

    confidence_threshold: float = 0.35


# ---------------------------------------------------------------------
# PHYSICS-CONSTRAINED FEATURE CONSTANTS
# ---------------------------------------------------------------------


@dataclass
class PhysicsConstants:

    # Laplace pressure (dP = 2*gamma/r) — surface tension of molten
    # Inconel 718 near its melting range. 
    surface_tension_n_per_m: float = 1.80

    # Weber number (We = rho*v^2*d/gamma) — molten Inconel 718 density
    # near the melting point (lower than the ~8190 kg/m^3 room- temperature solid density).
    melt_density_kg_per_m3: float = 7400.0

    reference_ejection_velocity_m_per_s: float = 5.0

    # Griffith/critical-stress term (sigma_critical = sqrt(2*E*gamma /
    # (pi*c)), or the K_IC-based equivalent sigma_critical = K_IC /
    # sqrt(pi*c)) — Young's modulus of Inconel 718.
    youngs_modulus_pa: float = 200.0e9

    fracture_toughness_mpa_sqrt_m: float = 100.0


# ---------------------------------------------------------------------
# REUSABILITY SCORE
# ---------------------------------------------------------------------


@dataclass
class ReusabilityConfig:

    alpha: float = 0.55   # weight on defect-class severity
    beta: float = 0.25    # weight on population PSD shift
    gamma: float = 0.20   # weight on surface-quality degradation

    
    reference_d10_um: float | None = None
    reference_d50_um: float | None = None
    reference_d90_um: float | None = None


# ---------------------------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------------------------

@dataclass
class MLConfig:

    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5


    dinov2_pca_components: int = 16
    # k for the prototype/few-shot classifier's nearest-neighbor vote.
    prototype_k: int = 3


# ---------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------

@dataclass
class VisualizationConfig:

    overlay_alpha: float = 0.40
    contour_thickness: int = 2
    font_scale: float = 0.40

    border_dash_length: int = 6
    border_gap_length: int = 5

    clean_mask_no_text: bool = True


    legend_swatch_corner_radius: float = 0.15


# ---------------------------------------------------------------------
# EXPORT SINGLETONS
# ---------------------------------------------------------------------

SEGMENTATION = SegmentationConfig()
MORPHOLOGY = MorphologyConfig()
ATTACHMENT = AttachmentConfig()
CLASSIFICATION = ClassificationConfig()
PHYSICS = PhysicsConstants()
REUSABILITY = ReusabilityConfig()
ML = MLConfig()
VISUALIZATION = VisualizationConfig()
