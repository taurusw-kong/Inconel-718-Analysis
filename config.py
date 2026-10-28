"""
ParticleVision V4 Configuration
================================
Central configuration file for the entire framework.

Everything a non-programmer should be able to tune lives here as plain,
flat, editable values — segmentation thresholds, the defect taxonomy
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

    # points_per_side raised (32->48) and both thresholds lowered
    # (0.86->0.75, 0.92->0.85) from SAM2's stock defaults — confirmed
    # by direct testing that the stock values systematically under-
    # detect complex, irregular, multi-lobed particles (exactly
    # Agglomerate/Deformed_Particle — the classes this taxonomy cares
    # about most), while a denser point grid + more permissive
    # acceptance recovers them. Trade-off: more raw candidate masks
    # means more work for the duplicate/quality filters downstream,
    # and some risk of more false positives — worth it given what was
    # being silently missed before. If you find spurious/artifact
    # particles increasing, raise these back toward the stock values.
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
#
# Two particles are considered candidates for a host/satellite or
# bonded relationship when their masks sit within `contact_gap_px` of
# each other (a real physical neck may not survive segmentation as
# literally touching pixels) AND their shared-boundary evidence clears
# `min_contact_fraction`. Both are continuous inputs to the classifier
# (see attachment.py / data_models.Attachment) rather than a hard
# accept/reject gate — the classifier decides how much attachment
# evidence matters per class, this just decides what counts as
# "evidence" at all.

@dataclass
class AttachmentConfig:

    contact_gap_px: int = 3

    min_contact_fraction: float = 0.03

    # A particle is a plausible "host" for another only if it is at
    # least this many times larger by equivalent diameter — prevents
    # two similar-sized touching particles from being mislabeled as
    # host+satellite when they're really a Bonded_Particle pair.
    host_size_ratio: float = 1.8


# ---------------------------------------------------------------------
# DEFECT TAXONOMY  (Inconel-718 / powder-bed reuse classification)
# ---------------------------------------------------------------------
#
# 10 classes, defined with the user (a materials researcher) directly.
# Every class is classified the SAME way — one fused physics+morphology
# +DINOv2 feature vector through one trained classifier
# (classification.py) — none of these get a hardcoded threshold rule.
# The size/attachment/shape descriptions below inform which FEATURES
# the classifier is given (calibrated size_um, attachment evidence,
# circularity, etc.), not an if/else decision tree.
#
# `severity` is a 0-100 relative-harm weight used only by the
# reusability score (REUSABILITY below). These are a reasonable
# starting prior based on the mechanism each defect implies for AM
# feedstock reuse (flowability, packing density, melt-pool stability)
# — NOT a validated materials standard. For publication, recalibrate
# against real flowability/porosity measurements (e.g. Hall-flow rate,
# apparent density) or an expert elicitation panel, then report the
# calibration method.

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

# "Spatter_Derived" was dropped from the active taxonomy: only 5 real
# labeled examples were ever found (it is a genuinely rare defect that
# occurs under specific conditions), which is far below what any
# classifier can learn — it scored precision/recall/F1 of exactly 0.00
# in every cross-validated run and was dragging macro-F1 down for the
# whole framework. The Defects/Spatter Derived/ folder is left on disk
# untouched; build_prototypes.py simply reports it as an unmatched
# folder and skips it. To reactivate the class later, add the name
# back to this list (plus DEFECT_COLORS/DEFECT_SEVERITY below) once
# enough examples exist. Report it in the manuscript as observed but
# too rare in this dataset to classify reliably — an honest, normal
# limitation, not a hidden failure.

# Originally 10 classes split Molten_Satellite (attached small sphere,
# itself circular) from Bonded_Satellite (attached small sphere, itself
# NOT circular). Merged back into one class on real cross-validated
# data: with only ~10-12 labeled examples each, that shape distinction
# wasn't learnable (F1 0.095 for Molten_Satellite specifically, heavily
# confused with Bonded_Satellite) — a genuine small-sample limitation,
# not a code bug. If the labeled set grows substantially for this
# region of the taxonomy later, this is the natural place to split it
# back into two classes.

DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    # A strict 10-color subset of v3's 12-color CVD-validated palette
    # (dropped "Non_Spherical"/"Satellite" — those generic names have
    # no place in this taxonomy; satellite is now split into
    # Molten/Bonded specifically). CVD-safety and pairwise
    # distinctness are properties of every pair in the original set —
    # removing colors can only preserve those properties, never break
    # them, so this subset is still fully validated (dataviz skill's
    # validate_palette tool). Also chosen with visualization.py's
    # ADDITIVE overlay blend in mind: every color has at least one
    # near-zero channel so it still reads as a visible tint on a
    # bright SEM particle instead of washing out to near-white.
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

# Sentinel pseudo-class — NOT one of the 8 defect classes above, and
# never a model prediction. Assigned directly (see
# segmentation.estimate_border_truncation / main.classify_particles)
# to any particle whose mask is significantly cut off by the image
# frame edge. A truncated view of a particle can't be reliably typed
# even by eye — the whole point of the class is being visible — so
# these are excluded from classification, size statistics, and
# reusability composition rather than forced through the classifier.
# They are NOT discarded: kept in features.csv and the overlay (as a
# dashed orange OUTLINE only, no fill — see
# visualization.draw_classification_overlay) so the batch's true
# detected count is still honest and auditable, without looking like
# a confident class prediction the way a filled color would.
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

    # Class-size bands (microns) as originally specified. NOTE: these
    # are currently REFERENCE VALUES ONLY — nothing reads them. The
    # classifier learns size boundaries from labeled examples via the
    # continuous `phys_size_um` feature rather than being handed these
    # numbers, so editing them changes nothing today. They are kept
    # because they document the intended taxonomy.
    #
    # They also DISAGREE with the measured powder. Against correctly
    # calibrated data (0.1 um/px — see preprocessing.get_calibration;
    # an earlier OCR misread of 0.96 um/px made everything ~10x too
    # large), the labeled examples actually measure:
    #
    #     Perfect_Spherical    median  6.9 um   (spec says 20-30)
    #     Oversized_Spherical  median 13.9 um   (spec says > 30)
    #     Molten_Satellite     median  6.1 um   (spec says < 3)
    #
    # The RELATIVE ordering is right (oversized ~2x perfect), so the
    # labeling is internally consistent — this powder is simply much
    # finer than the spec bands assume (batch D50 ~13 um). Worth
    # reconciling before publication: either restate the bands to match
    # the measured material, or confirm the intended spec and re-check
    # which particles belong in which class.
    perfect_spherical_min_um: float = 20.0
    perfect_spherical_max_um: float = 30.0
    oversized_spherical_min_um: float = 30.0
    satellite_max_um: float = 3.0

    # Minimum labeled examples per class (from the user's 10 prototype
    # folders) below which classification.py should warn rather than
    # silently train on too little signal.
    min_prototype_examples: int = 8

    confidence_threshold: float = 0.35


# ---------------------------------------------------------------------
# PHYSICS-CONSTRAINED FEATURE CONSTANTS
# ---------------------------------------------------------------------
#
# Every value below is a LITERATURE-TYPICAL constant for molten/solid
# Inconel 718, not a value measured from this dataset. They convert
# calibrated geometry (real microns, from preprocessing.py) into the
# manuscript's three named physics quantities. Before submission,
# replace each with a cited value appropriate to your actual process
# conditions (build temperature, powder condition, etc.) — these
# defaults are a defensible literature-typical starting point, not a
# substitute for your own citation.

@dataclass
class PhysicsConstants:

    # Laplace pressure (dP = 2*gamma/r) — surface tension of molten
    # Inconel 718 near its melting range. Literature-typical value;
    # cite your specific source.
    surface_tension_n_per_m: float = 1.80

    # Weber number (We = rho*v^2*d/gamma) — molten Inconel 718 density
    # near the melting point (lower than the ~8190 kg/m^3 room-
    # temperature solid density).
    melt_density_kg_per_m3: float = 7400.0

    # Weber number's ejection velocity. SEM gives no direct velocity
    # measurement per particle — this is a fixed, documented
    # reference-condition assumption (a representative LPBF
    # spatter/droplet ejection velocity order-of-magnitude from
    # literature), so the resulting We score is a REFERENCE-CONDITION
    # PLAUSIBILITY INDEX, not a per-particle physical measurement.
    # State this explicitly wherever We is reported.
    reference_ejection_velocity_m_per_s: float = 5.0

    # Griffith/critical-stress term (sigma_critical = sqrt(2*E*gamma /
    # (pi*c)), or the K_IC-based equivalent sigma_critical = K_IC /
    # sqrt(pi*c)) — Young's modulus of Inconel 718.
    youngs_modulus_pa: float = 200.0e9

    # Practical implementation uses a fracture-toughness-based
    # critical stress (K_IC / sqrt(pi*c)) rather than a raw surface
    # energy, since K_IC is the value actually reported/citable in
    # Inconel 718 literature; wrought-material order-of-magnitude
    # default given here — cite the value appropriate to your powder's
    # condition.
    fracture_toughness_mpa_sqrt_m: float = 100.0


# ---------------------------------------------------------------------
# REUSABILITY SCORE
# ---------------------------------------------------------------------
#
# Reusability(particle) = 100 - alpha * D_severity
#                              - beta  * delta_PSD   (batch-level term)
#                              - gamma * O_surface
#
# See reusability.py for the full derivation. All three weights are
# exposed here so they can be tuned/sensitivity-analysed.

@dataclass
class ReusabilityConfig:

    alpha: float = 0.55   # weight on defect-class severity
    beta: float = 0.25    # weight on population PSD shift
    gamma: float = 0.20   # weight on surface-quality degradation

    # Optional reference powder-size distribution (microns), e.g. the
    # virgin-powder spec range. Leave as None to skip the PSD-shift
    # term (it will contribute 0 until you supply this).
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

    # DINOv2 embedding PCA reduction before fusion with physics/
    # morphology features — keeps the fused vector from being
    # dominated by embedding dimensionality on a small labeled set.
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

    # Corner radius (points) for legend swatches in matplotlib figures
    # — rounded rectangles instead of plain ones, per the requested
    # visual style.
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
