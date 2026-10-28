"""
ParticleVision V4
=================

Quantification (population-proportion estimation)

Reusability analysis asks a different question than classification.
Classification asks "what is THIS particle?" — quantification asks
"what FRACTION of this batch is each defect type?". They are separate
statistical targets, and the second is far more tolerant of an
imperfect classifier: individual mistakes cancel out in aggregate as
long as the classifier's error PATTERN is known.

Simply counting predicted labels ("Classify and Count") is biased,
and the bias does NOT shrink as we add particles.

IMPORTANT: the confusion matrix must come from HELD-OUT predictions
(cross-validated out-of-fold), never from the training fit, a matrix
built on data the model memorized would understate its true error rate
and the correction would be wrong in the same optimistic direction.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


CORRECTION_MATRIX_FILENAME = "quantification_matrix.csv"
POPULATION_FILENAME = "population_estimate.csv"


# ============================================================
# BUILDING THE CORRECTION
# ============================================================

def build_correction_matrix(y_true, y_pred, class_names) -> pd.DataFrame:
   

    from sklearn.metrics import confusion_matrix

    n = len(class_names)
    index = {c: i for i, c in enumerate(class_names)}

    y_true_idx = np.array([index[c] for c in y_true])
    y_pred_idx = np.array([index[c] for c in y_pred])

    # confusion_matrix is [true, pred]; transpose to [pred, true]
    counts = confusion_matrix(y_true_idx, y_pred_idx, labels=range(n)).astype(float)

    per_true_total = counts.sum(axis=1, keepdims=True)
    per_true_total[per_true_total == 0] = 1.0

    M = (counts / per_true_total).T

    return pd.DataFrame(M, index=class_names, columns=class_names)


def save_correction_matrix(matrix: pd.DataFrame, output_dir: str):
    matrix.to_csv(os.path.join(output_dir, CORRECTION_MATRIX_FILENAME))


def load_correction_matrix(output_dir: str):

    path = os.path.join(output_dir, CORRECTION_MATRIX_FILENAME)

    if not os.path.exists(path):
        return None

    return pd.read_csv(path, index_col=0)


# ============================================================
# APPLYING THE CORRECTION
# ============================================================

def adjust_proportions(raw_proportions: np.ndarray, matrix: pd.DataFrame) -> np.ndarray:

    M = matrix.values

    adjusted, *_ = np.linalg.lstsq(M, np.asarray(raw_proportions, dtype=float), rcond=None)

    adjusted = np.clip(adjusted, 0.0, None)

    total = adjusted.sum()

    return adjusted / total if total > 0 else adjusted


def estimate_population(
    predicted_labels,
    matrix: pd.DataFrame,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
  

    class_names = list(matrix.index)
    index = {c: i for i, c in enumerate(class_names)}

    labels = [str(l) for l in predicted_labels if str(l) in index]
    n = len(labels)

    if n == 0:
        return pd.DataFrame()

    counts = np.zeros(len(class_names))
    for l in labels:
        counts[index[l]] += 1

    raw = counts / n
    corrected = adjust_proportions(raw, matrix)

    rng = np.random.default_rng(random_state)
    label_idx = np.array([index[l] for l in labels])

    boot = np.zeros((n_bootstrap, len(class_names)))

    for b in range(n_bootstrap):
        sample = rng.choice(label_idx, size=n, replace=True)
        c = np.bincount(sample, minlength=len(class_names)).astype(float) / n
        boot[b] = adjust_proportions(c, matrix)

    lower = np.percentile(boot, 2.5, axis=0)
    upper = np.percentile(boot, 97.5, axis=0)

    return pd.DataFrame({
        "class_label": class_names,
        "n_predicted": counts.astype(int),
        "raw_proportion": raw,
        "corrected_proportion": corrected,
        "ci95_low": lower,
        "ci95_high": upper,
    })


# ============================================================
# ACROSS REUSE CYCLES
# ============================================================

def compare_cycles(cycle_predictions: dict, matrix: pd.DataFrame, **kwargs) -> pd.DataFrame:
  

    frames = []

    for cycle_name, labels in cycle_predictions.items():

        estimate = estimate_population(labels, matrix, **kwargs)

        if estimate.empty:
            continue

        estimate.insert(0, "cycle", cycle_name)
        estimate["n_particles"] = len([l for l in labels])
        frames.append(estimate)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "CORRECTION_MATRIX_FILENAME", "POPULATION_FILENAME",
    "build_correction_matrix", "save_correction_matrix", "load_correction_matrix",
    "adjust_proportions", "estimate_population", "compare_cycles",
]
