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


    is_attached: bool = False

    host_particle_id: Optional[int] = None

    attached_particle_ids: List[int] = field(default_factory=list)


    nearest_neighbor_distance_px: float = float("nan")

    # Fraction of this particle's own boundary that sits within the
    # touching tolerance of another particle's boundary 
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
    # rather than living in `.values`.
    dinov2_embedding: Optional[np.ndarray] = None

    # v4: calibrated real-world size (equivalent_diameter converted
    # via preprocessing.py's microns_per_pixel for this image). None
    # until calibration has been applied, every physics-constrained
    # feature and the classifier's fused vector depend on this being
    # real microns, not raw pixels, so a missing value here should be
    # treated as "this particle can't be trusted yet," not silently
    # defaulted to 0.
    size_um: Optional[float] = None

  
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
