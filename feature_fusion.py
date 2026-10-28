"""
ParticleVision V4
=================

Feature Fusion

Combines every particle's morphology descriptors, physics-constrained
features (composite shape scores, calibrated size, attachment
evidence, the 3 named physics equations), and self-supervised DINOv2
embedding into ONE fused feature vector — the single input the
unified classifier (classification.py) trains/predicts on for all 10
classes. No feature group is treated specially per class.

    morph_<name>   — pure shape descriptors (morphology.py)
    phys_<name>    — composite physics-constrained scores
                     (physics_formulas.py) + calibrated size +
                     attachment evidence
    dinov2_pc<i>   — PCA-reduced self-supervised embedding dimensions
"""

from __future__ import annotations

from typing import List

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data_models import Particle


FUSION_PIPELINE_FILENAME = "fusion_pipeline.joblib"


# ============================================================
# FLATTENING (Particle / prototype record -> one flat dict)
# ============================================================

def _attachment_flat_from_dataclass(attachment) -> dict:

    return {
        "phys_attach_is_attached": float(attachment.is_attached),
        "phys_attach_has_host": float(attachment.host_particle_id is not None),
        "phys_attach_count": float(len(attachment.attached_particle_ids)),
        "phys_attach_contact_fraction": attachment.boundary_contact_fraction,
        "phys_attach_nn_distance_px": attachment.nearest_neighbor_distance_px,
    }


def _attachment_flat_from_dict(attachment: dict) -> dict:

    return {
        "phys_attach_is_attached": float(attachment.get("is_attached", False)),
        "phys_attach_has_host": float(attachment.get("has_host", False)),
        "phys_attach_count": float(attachment.get("attached_count", 0)),
        "phys_attach_contact_fraction": attachment.get("boundary_contact_fraction", np.nan),
        "phys_attach_nn_distance_px": attachment.get("nearest_neighbor_distance_px", np.nan),
    }


def flatten_particle(particle: Particle) -> dict:
    """Flattens a production Particle (in-memory pipeline object)."""

    flat = {f"morph_{k}": v for k, v in particle.morphology.values.items()}
    flat.update({f"phys_{k}": v for k, v in particle.physics.values.items()})
    flat["phys_size_um"] = particle.size_um if particle.size_um is not None else np.nan
    flat.update(_attachment_flat_from_dataclass(particle.attachment))

    return flat


def flatten_prototype_record(record: dict) -> dict:
    """Flattens a record from build_prototypes.py's prototypes.joblib."""

    flat = {f"morph_{k}": v for k, v in record["morphology"].items()}
    flat.update({f"phys_{k}": v for k, v in record["physics"].items()})
    flat["phys_size_um"] = record.get("size_um", np.nan)
    flat.update(_attachment_flat_from_dict(record.get("attachment", {})))

    return flat

EXCLUDED_SCALAR_FEATURES = {
    "morph_largest_hole_ratio", "morph_curvature_max", "phys_porosity_index",
    "morph_equivalent_diameter", "morph_max_feret", "morph_min_feret",
    "morph_major_axis", "morph_minor_axis",
    "morph_bbox_x", "morph_bbox_y",
    "morph_mask_pixels", "morph_bbox_width", "morph_bbox_height", "morph_bbox_area",
}


def _drop_excluded_features(scalar_df: pd.DataFrame) -> pd.DataFrame:
    return scalar_df.drop(columns=[c for c in EXCLUDED_SCALAR_FEATURES if c in scalar_df.columns])


def prototypes_to_frame(records: List[dict]):


    scalar_rows = [flatten_prototype_record(r) for r in records]
    scalar_df = _drop_excluded_features(pd.DataFrame(scalar_rows))

    embeddings = np.stack([r["dinov2_embedding"] for r in records])
    labels = np.array([r["class_label"] for r in records])
    group_ids = np.array([
        r.get("source_group_id", i) for i, r in enumerate(records)
    ])

    return scalar_df, embeddings, labels, group_ids


def particles_to_frame(particles: List[Particle]):
    """
    -> (scalar_df, embeddings [N x D]) for production particles that
    already have a dinov2_embedding set.
    """

    usable = [p for p in particles if p.dinov2_embedding is not None]

    scalar_rows = [flatten_particle(p) for p in usable]
    scalar_df = _drop_excluded_features(pd.DataFrame(scalar_rows))

    embeddings = (
        np.stack([p.dinov2_embedding for p in usable])
        if usable else np.empty((0, 0))
    )

    return scalar_df, embeddings, usable


# ============================================================
# FUSION PIPELINE (fit once on prototypes, reused at inference)
# ============================================================

class FusionPipeline:
 

    def __init__(self, dinov2_pca_components: int = 16, random_state: int = 42,
                 use_physics: bool = True, use_morphology: bool = True, use_dinov2: bool = True):

        self.dinov2_pca_components = dinov2_pca_components
        self.random_state = random_state
        self.use_physics = use_physics
        self.use_morphology = use_morphology
        self.use_dinov2 = use_dinov2

        self.scalar_columns_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.embedding_pca_ = None
        self.embedding_scaler_ = None
        self.fitted_ = False

    def _select_columns(self, scalar_df: pd.DataFrame) -> pd.DataFrame:

        keep = list(scalar_df.columns)

        if not self.use_physics:
            keep = [c for c in keep if not c.startswith("phys_")]

        if not self.use_morphology:
            keep = [c for c in keep if not c.startswith("morph_")]

        return scalar_df[keep]

    def fit_transform(self, scalar_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:

        scalar_df = self._select_columns(scalar_df)
        self.scalar_columns_ = list(scalar_df.columns)

        blocks = []

        if self.scalar_columns_:

       
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            scalar_imputed = self.imputer_.fit_transform(scalar_df.values)

            self.scaler_ = StandardScaler()
            scalar_scaled = self.scaler_.fit_transform(scalar_imputed)

            blocks.append(scalar_scaled)

        if self.use_dinov2 and embeddings.size > 0:

            n_components = min(
                self.dinov2_pca_components, embeddings.shape[1],
                max(2, embeddings.shape[0] - 1),
            )

            self.embedding_pca_ = PCA(n_components=n_components, random_state=self.random_state)
            emb_reduced = self.embedding_pca_.fit_transform(embeddings)

            self.embedding_scaler_ = StandardScaler()
            emb_scaled = self.embedding_scaler_.fit_transform(emb_reduced)

            blocks.append(emb_scaled)

        self.fitted_ = True

        return np.hstack(blocks)

    def transform(self, scalar_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:

        if not self.fitted_:
            raise RuntimeError("FusionPipeline.transform called before fit_transform.")

        blocks = []

        if self.scalar_columns_:

            scalar_df = self._select_columns(scalar_df)
            scalar_df = scalar_df.reindex(columns=self.scalar_columns_)

            scalar_imputed = self.imputer_.transform(scalar_df.values)
            scalar_scaled = self.scaler_.transform(scalar_imputed)

            blocks.append(scalar_scaled)

        if self.use_dinov2 and self.embedding_pca_ is not None and embeddings.size > 0:

            expected_dim = self.embedding_pca_.n_features_in_

            if embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"DINOv2 embedding size mismatch: got {embeddings.shape[1]}, "
                    f"but this classifier was trained on {expected_dim}. You changed "
                    f"the DINOv2 backbone without rebuilding. Re-run build_prototypes.py "
                    f"(with the same --dinov2_model) and retrain, or switch the backbone back."
                )

            emb_reduced = self.embedding_pca_.transform(embeddings)
            emb_scaled = self.embedding_scaler_.transform(emb_reduced)
            blocks.append(emb_scaled)

        return np.hstack(blocks)

    def feature_names(self) -> List[str]:

        names = list(self.scalar_columns_)

        if self.use_dinov2 and self.embedding_pca_ is not None:
            names += [f"dinov2_pc{i}" for i in range(self.embedding_pca_.n_components_)]

        return names

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FusionPipeline":
        return joblib.load(path)


__all__ = [
    "FusionPipeline", "FUSION_PIPELINE_FILENAME",
    "flatten_particle", "flatten_prototype_record",
    "prototypes_to_frame", "particles_to_frame",
]
