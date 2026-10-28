"""
ParticleVision V4
=================

Explainability

Every classified particle gets an explanation, regardless of which
model predicted it (no class or model gets special treatment):

- Tree-model predictions (XGBoost, RandomForest, ExtraTrees,
  GradientBoosting — classification.TREE_MODEL_NAMES): SHAP feature
  attribution — which fused features (physics-constrained or
  morphological or DINOv2) pushed the prediction toward its predicted
  class, and by how much. Standard, citable global/local explanation
  method for tree models, and shap.TreeExplainer supports all of
  these natively through the same interface.
- Prototype few-shot predictions: nearest-labeled-example explanation
  — the k prototypes (by fused-feature distance) that the prediction
  is actually based on. Inherently interpretable by construction; a
  case-based complement to SHAP's feature-based story.
- The remaining models (LogisticRegression, KNN, SVM, MLP) don't get
  a per-particle explanation here — SHAP's TreeExplainer doesn't apply
  to them, and the generic KernelExplainer is too slow to run
  per-particle on a live pipeline. 

"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import shap

from data_models import Particle
from feature_fusion import particles_to_frame
from classification import MODEL_PROTOTYPE, TREE_MODEL_NAMES


SHAP_EXPLANATIONS_FILENAME = "shap_explanations.csv"
PROTOTYPE_EXPLANATIONS_FILENAME = "prototype_explanations.csv"

TOP_K_SHAP_FEATURES = 5


# ============================================================
# SHAP (any tree model — classification.TREE_MODEL_NAMES)
# ============================================================

def explain_tree_model_predictions(
    particles: List[Particle],
    classifier: dict,
    top_k: int = TOP_K_SHAP_FEATURES,
) -> pd.DataFrame:
 

    all_rows = []

    for model_name in TREE_MODEL_NAMES:

        targets = [p for p in particles if p.prediction.model_name == model_name]

        if not targets:
            continue

        all_rows.extend(_explain_one_tree_model(targets, classifier, model_name, top_k))

    return pd.DataFrame(all_rows)


def _explain_one_tree_model(targets: List[Particle], classifier: dict, model_name: str, top_k: int) -> list:

    scalar_df, embeddings, usable = particles_to_frame(targets)

    if not usable:
        return []

    X = classifier["fusion"].transform(scalar_df, embeddings)
    feature_names = classifier["fusion"].feature_names()
    label_encoder = classifier["label_encoder"]

    model = classifier["models"][model_name]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # (n_samples, n_features, n_classes)

    rows = []

    for i, particle in enumerate(usable):

        predicted_index = list(label_encoder.classes_).index(particle.prediction.label)

        per_feature = shap_values[i, :, predicted_index]

        order = np.argsort(-np.abs(per_feature))[:top_k]

        explanation_strings = []

        for rank, feat_idx in enumerate(order, start=1):

            feature_name = feature_names[feat_idx]
            value = float(per_feature[feat_idx])

            rows.append({
                "model": model_name,
                "particle_id": particle.particle_id,
                "image_name": particle.image_name,
                "predicted_label": particle.prediction.label,
                "rank": rank,
                "feature": feature_name,
                "shap_value": value,
            })

            sign = "+" if value >= 0 else ""
            explanation_strings.append(f"{feature_name}: {sign}{value:.3f}")

        particle.prediction.explanation = explanation_strings

    return rows


# ============================================================
# NEAREST-PROTOTYPE (few-shot)
# ============================================================

def explain_prototype_predictions(
    particles: List[Particle],
    classifier: dict,
    prototype_records: List[dict],
    k: int = 3,
) -> pd.DataFrame:
 
    targets = [p for p in particles if p.prediction.model_name == MODEL_PROTOTYPE]

    if not targets:
        return pd.DataFrame()

    scalar_df, embeddings, usable = particles_to_frame(targets)

    if not usable:
        return pd.DataFrame()

    X = classifier["fusion"].transform(scalar_df, embeddings)
    model = classifier["models"][MODEL_PROTOTYPE]

    rows = []

    for i, particle in enumerate(usable):

        idx, distances = model.neighbors(X[i], k=k)

        explanation_strings = []

        for rank, (proto_idx, dist) in enumerate(zip(idx, distances), start=1):

            record = prototype_records[proto_idx]

            rows.append({
                "particle_id": particle.particle_id,
                "image_name": particle.image_name,
                "predicted_label": particle.prediction.label,
                "neighbor_rank": rank,
                "neighbor_class": record["class_label"],
                "neighbor_source_path": record["source_path"],
                "distance": float(dist),
            })

            explanation_strings.append(
                f"#{rank} nearest: {record['class_label']} ({os.path.basename(record['source_path'])}, d={dist:.2f})"
            )

        particle.prediction.explanation = explanation_strings

    return pd.DataFrame(rows)


# ============================================================
# COMBINED ENTRY POINT
# ============================================================

def explain_predictions(
    particles: List[Particle],
    classifier: dict,
    output_dir: str,
    prototype_records: List[dict] = None,
) -> dict:
    """
    Runs whichever explanation method applies to each particle
    (based on particle.prediction.model_name, set by
    classification.classify_particles) and saves both tidy tables.
    """

    shap_df = explain_tree_model_predictions(particles, classifier)

    prototype_df = pd.DataFrame()
    if prototype_records is not None:
        prototype_df = explain_prototype_predictions(particles, classifier, prototype_records)

    if not shap_df.empty:
        shap_df.to_csv(os.path.join(output_dir, SHAP_EXPLANATIONS_FILENAME), index=False)

    if not prototype_df.empty:
        prototype_df.to_csv(os.path.join(output_dir, PROTOTYPE_EXPLANATIONS_FILENAME), index=False)

    return {"shap": shap_df, "prototype": prototype_df}


__all__ = [
    "explain_tree_model_predictions", "explain_prototype_predictions", "explain_predictions",
    "SHAP_EXPLANATIONS_FILENAME", "PROTOTYPE_EXPLANATIONS_FILENAME",
]
