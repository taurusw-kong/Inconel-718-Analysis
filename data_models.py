"""
ParticleVision V4
=================

Core data models used throughout the framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ---------------------------------------------------------------------
# BASIC GEOMETRY
# ---------------------------------------------------------------------

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Centroid:
    x: float
    y: float


# ---------------------------------------------------------------------
# MORPHOLOGY FEATURES
# ---------------------------------------------------------------------

@dataclass
class MorphologyFeatures:

    values: Dict[str, float] = field(default_factory=dict)

    def set(self, name: str, value: float):
        self.values[name] = float(value)

    def get(self, name: str, default=np.nan):
        return self.values.get(name, default)


# ---------------------------------------------------------------------
# PHYSICS FEATURES
# ---------------------------------------------------------------------

@dataclass
class PhysicsFeatures:

    values: Dict[str, float] = field(default_factory=dict)

    def set(self, name: str, value: float):
        self.values[name] = float(value)

    def get(self, name: str, default=np.nan):
        return self.values.get(name, default)


# ---------------------------------------------------------------------
# ATTACHMENT (v4 — first-class, not inferred indirectly)
# ---------------------------------------------------------------------

@dataclass
class Attachment:
    """
    Whether this particle is physically attached (shares a neck / sits
    close enough to be a host+satellite pair) to another particle in the
    same image, and to which one.

    `is_attached` / `host_particle_id` describe this particle's role as
    the (usually smaller) attached member; `attached_particle_ids` lists
    every other particle attached TO this one (relevant when this
    particle is itself the larger host of one or more satellites, or a
    member of a multi-particle bonded/agglomerate cluster).

    Computed by attachment.py from touching/near-touching mask
    geometry — see that module for the actual detection logic.
    """

    is_attached: bool = False

    host_particle_id: Optional[int] = None

    attached_particle_ids: List[int] = field(default_factory=list)

    # Distance (pixels) between this particle's centroid and its
    # nearest neighbor's centroid, regardless of whether that neighbor
    # counts as "attached" — kept as a continuous feature since the
    # unified classifier learns from continuous inputs rather than a
    # hard threshold.
    nearest_neighbor_distance_px: float = float("nan")

    # Fraction of this particle's own boundary that sits within the
    # touching tolerance of another particle's boundary — a continuous
    # "how attached" signal, not just a yes/no.
    boundary_contact_fraction: float = 0.0


# ---------------------------------------------------------------------
# CLASSIFICATION RESULT
# ---------------------------------------------------------------------

@dataclass
class PredictionResult:

    label: str = "Unknown"

    confidence: float = 0.0

    scores: Dict[str, float] = field(default_factory=dict)

    explanation: List[str] = field(default_factory=list)

    # v4: which model produced this prediction ("xgboost" or
    # "prototype_few_shot") — the unified classifier trains/compares
    # both, and the GUI/explainability layer need to know which one a
    # given prediction came from to fetch the right kind of
    # explanation (SHAP vs. nearest-prototype).
    model_name: Optional[str] = None


# ---------------------------------------------------------------------
# PARTICLE OBJECT
# ---------------------------------------------------------------------

@dataclass
class Particle:

    particle_id: int

    image_name: str

    mask: np.ndarray

    contour: Optional[np.ndarray] = None

    bbox: Optional[BoundingBox] = None

    centroid: Optional[Centroid] = None

    morphology: MorphologyFeatures = field(
        default_factory=MorphologyFeatures
    )

    physics: PhysicsFeatures = field(
        default_factory=PhysicsFeatures
    )

    # Frozen DINOv2 embedding of this particle's crop (see
    # dinov2_features.py). A single dense vector, not a named-scalar
    # bag like `morphology`/`physics` above, so it gets its own field
    # rather than living in `.values`. None until an extractor has
    # actually run on this particle.
    dinov2_embedding: Optional[np.ndarray] = None

    # v4: calibrated real-world size (equivalent_diameter converted
    # via preprocessing.py's microns_per_pixel for this image). None
    # until calibration has been applied — every physics-constrained
    # feature and the classifier's fused vector depend on this being
    # real microns, not raw pixels, so a missing value here should be
    # treated as "this particle can't be trusted yet," not silently
    # defaulted to 0.
    size_um: Optional[float] = None

    # v4: first-class attachment relationship (see Attachment above).
    attachment: Attachment = field(default_factory=Attachment)

    prediction: PredictionResult = field(
        default_factory=PredictionResult
    )

    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# IMAGE RESULT
# ---------------------------------------------------------------------

@dataclass
class ImageAnalysis:

    image_name: str

    particles: List[Particle] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
