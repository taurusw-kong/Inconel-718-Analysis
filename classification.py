"""
ParticleVision V4
=================

Unified Classification

Trains and compares TWO model families on the SAME fused feature
vector (feature_fusion.py) against the SAME prototype-labeled set, for
ALL 10 classes uniformly — no class gets a different method:

- Gradient-Boosted Trees (XGBoost)
- Prototype-based Few-Shot (k-nearest labeled example, by distance in
  the fused feature space)

Evaluation uses out-of-fold cross-validated predictions (not a single
train/test split) — the prototype set is small (tens of examples per
class), so a one-shot split would be high-variance and easy to
over-read; cross-validation uses every example as a held-out test case
exactly once, which is the standard, defensible way to report accuracy
on a dataset this size.

The deployed model (used for actual predictions on production
particles) is refit on the FULL prototype set after evaluation — CV
folds are for honest metrics, not the final classifier.
"""

from __future__ import annotations

import os

# torch and xgboost each bundle their own OpenMP runtime. Running both
# in the same process — this pipeline needs DINOv2 (torch) embeddings
# AND an XGBoost classifier together — reliably SEGFAULTS on macOS the
# first time xgboost actually trains or loads/infers after torch has
# run a real forward pass. Reproduced directly; neither import order
# nor torch.set_num_threads(1) alone fixes it (both still tried).
# OMP_NUM_THREADS=1 set BEFORE either library initializes does fix it
# — but only if it's set before the *first* import of torch or
# xgboost anywhere in the process, since OpenMP reads it at library
# load time. This must be the first thing in every entrypoint
# (main.py, gui_app.py, build_prototypes.py) — setting it here alone
# is not sufficient if something else imports torch first.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import xgboost
from xgboost import XGBClassifier

# LightGBM and CatBoost complete the gradient-boosted-tree family the
# manuscript's Section 4.8.1 names. Imported defensively: neither is
# required for the pipeline to run, and a missing one should drop that
# model from the comparison rather than break the whole classifier.
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# TabPFN (Section 4.8.2) runs as a HOSTED SERVICE: fitting uploads the
# fused feature vectors to Prior Labs' API. It is therefore strictly
# opt-in — enabled only when TABPFN_TOKEN is present in the environment
# — so a normal pipeline run never sends this project's data off the
# machine by default, and never needs network access to succeed.
try:
    from tabpfn_client import TabPFNClassifier as _TabPFNCloudClassifier
except ImportError:
    _TabPFNCloudClassifier = None

from typing import List

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from data_models import Particle
from feature_fusion import FusionPipeline, prototypes_to_frame, particles_to_frame
from quantification import build_correction_matrix, save_correction_matrix
import config


CLASSIFIER_DIRNAME = "classifier"

MODEL_XGBOOST = "xgboost"
MODEL_LIGHTGBM = "lightgbm"
MODEL_CATBOOST = "catboost"
MODEL_TABPFN = "tabpfn"
MODEL_FT_TRANSFORMER = "ft_transformer"
MODEL_PROTOTYPE = "prototype_few_shot"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_EXTRA_TREES = "extra_trees"
MODEL_GRADIENT_BOOSTING = "gradient_boosting"
MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_KNN = "knn"
MODEL_SVM = "svm_rbf"
MODEL_MLP = "mlp"

# shap.TreeExplainer works natively on any of these — explainability.py
# uses this set to decide which predictions get SHAP feature
# attribution vs. no per-particle explanation (SVM/LogReg/KNN/MLP
# would need the much slower KernelExplainer, not done here).
TREE_MODEL_NAMES = {MODEL_XGBOOST, MODEL_LIGHTGBM, MODEL_CATBOOST,
                    MODEL_RANDOM_FOREST, MODEL_EXTRA_TREES, MODEL_GRADIENT_BOOSTING}


def _fresh_sklearn_models(random_state: int) -> dict:
    """
    The full model roster (matches the previous version's comparison
    breadth) — every one trained/compared on the SAME fused feature
    vector, no per-class special-casing. The fused vector is already
    standardized by FusionPipeline, so unlike a raw-feature setup
    these don't need their own internal StandardScaler step.
    """

    models = {
        # Hyperparameters below are the result of a real
        # RandomizedSearchCV pass (25 iterations) over this project's
        # actual labeled prototype set, using the SAME group-aware CV
        # as everywhere else (no leakage) — chosen because they scored
        # best (38.2% vs. 36.0% for the untuned defaults), not guessed.
        MODEL_XGBOOST: XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.6, colsample_bytree=0.6,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=random_state, verbosity=0, n_jobs=1,
        ),
        MODEL_RANDOM_FOREST: RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=1,
        ),
        MODEL_EXTRA_TREES: ExtraTreesClassifier(
            n_estimators=300, random_state=random_state, n_jobs=1,
        ),
        MODEL_GRADIENT_BOOSTING: GradientBoostingClassifier(
            random_state=random_state,
        ),
        MODEL_LOGISTIC_REGRESSION: LogisticRegression(
            max_iter=5000, random_state=random_state,
        ),
        MODEL_KNN: KNeighborsClassifier(
            n_neighbors=min(5, config.ML.prototype_k + 2),
        ),
        MODEL_SVM: SVC(
            kernel="rbf", probability=True, random_state=random_state,
        ),
        MODEL_MLP: MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=2000,
            early_stopping=True, random_state=random_state,
        ),
    }

    # Both are matched to XGBoost's tuned depth/rate/subsample settings
    # rather than left at their own library defaults, so the three-way
    # gradient-boosting comparison in Section 4.8.1 isolates the
    # *implementation* (leaf-wise vs. level-wise growth, ordered vs.
    # standard boosting) instead of measuring who happened to ship the
    # better defaults for this dataset shape.
    if LGBMClassifier is not None:
        models[MODEL_LIGHTGBM] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.6, colsample_bytree=0.6, subsample_freq=1,
            # min_child_samples defaults to 20, which on a set this small
            # silently prunes most splits away and collapses the model
            # toward predicting the majority class.
            min_child_samples=5,
            random_state=random_state, n_jobs=1, verbose=-1,
        )

    if CatBoostClassifier is not None:
        models[MODEL_CATBOOST] = CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.1,
            subsample=0.6, rsm=0.6,
            loss_function="MultiClass", bootstrap_type="Bernoulli",
            random_seed=random_state, thread_count=1, verbose=False,
            allow_writing_files=False,
        )

    # Opt-in only (see the import note above): absent a token, TabPFN is
    # simply not in the comparison, and nothing is uploaded anywhere.
    token = os.environ.get("TABPFN_TOKEN")
    if _TabPFNCloudClassifier is not None and token:
        import tabpfn_client
        tabpfn_client.set_access_token(token)
        # n_estimators=8 keeps the number of billed API round-trips per
        # CV fold bounded; the default ensemble size multiplies request
        # count without meaningfully changing accuracy at this data size.
        models[MODEL_TABPFN] = _TabPFNCloudClassifier(n_estimators=8)

    return models


# ============================================================
# PROTOTYPE / FEW-SHOT CLASSIFIER
# ============================================================

class PrototypeFewShotClassifier:
    """
    Classifies by majority vote among the k nearest labeled prototypes
    in the fused feature space (Euclidean distance on the already-
    standardized fused vector). Inherently interpretable — a
    prediction can always be justified by literally showing the
    nearest labeled example(s) it matched (see explainability.py).
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.X_ = None
        self.y_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrototypeFewShotClassifier":
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def _neighbor_indices(self, x: np.ndarray, k: int) -> np.ndarray:
        distances = np.linalg.norm(self.X_ - x, axis=1)
        return np.argsort(distances)[:k], np.sort(distances)[:k]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x)[0] for x in np.asarray(X)])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        probs = np.zeros((len(X), len(self.classes_)))
        class_index = {c: i for i, c in enumerate(self.classes_)}

        for row, x in enumerate(np.asarray(X)):

            _, votes = self._predict_one(x)

            for cls, weight in votes.items():
                probs[row, class_index[cls]] = weight

        return probs

    def _predict_one(self, x: np.ndarray):

        k = min(self.k, len(self.y_))
        idx, distances = self._neighbor_indices(x, k)
        neighbor_labels = self.y_[idx]

        # Inverse-distance-weighted vote (a tied/very close neighbor
        # counts for more than a far one) — a small epsilon avoids
        # division by zero for an exact match.
        weights = 1.0 / (distances + 1e-6)

        votes: dict = {}
        for label, w in zip(neighbor_labels, weights):
            votes[label] = votes.get(label, 0.0) + w

        total = sum(votes.values())
        votes = {k_: v / total for k_, v in votes.items()}

        best_label = max(votes, key=votes.get)

        return best_label, votes

    def neighbors(self, x: np.ndarray, k: int = None):
        """Returns (indices, distances) of the k nearest prototypes — used by explainability.py."""
        idx, distances = self._neighbor_indices(np.asarray(x), k or self.k)
        return idx, distances


# ============================================================
# CROSS-VALIDATED EVALUATION
# ============================================================

def _safe_cv_folds(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """
    Fold count must respect GROUP counts per class, not raw row
    counts — with augmentation, a class can have far more rows than
    real independent examples (each real particle contributes several
    augmented views, but those are one group). Using raw row counts
    here would let StratifiedGroupKFold silently be asked for more
    folds than a class has independent groups.
    """

    df = pd.DataFrame({"y": y, "group": groups})
    group_counts_per_class = df.groupby("y")["group"].nunique()
    return max(2, min(requested, int(group_counts_per_class.min())))


def _cross_validated_predictions(X: np.ndarray, y_encoded: np.ndarray, groups: np.ndarray, model_name: str, n_folds: int, random_state: int):

    # StratifiedGroupKFold — NOT plain StratifiedKFold — is required
    # once augmented rotated/flipped views exist: several rows can
    # trace back to the SAME real labeled particle
    # (feature_fusion.prototypes_to_frame's group_ids), and letting
    # augmented siblings land in both the train and test fold would be
    # leakage (the model could "recognize" a rotated copy of an
    # example it already saw), not genuine held-out accuracy.
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    if model_name == MODEL_PROTOTYPE:

        # PrototypeFewShotClassifier isn't a full sklearn estimator
        # (no get_params/set_params needed for our use), so fold it by
        # hand rather than dragging in sklearn's estimator machinery.
        predictions = np.empty_like(y_encoded)

        for train_idx, test_idx in skf.split(X, y_encoded, groups):
            clf = PrototypeFewShotClassifier(k=config.ML.prototype_k)
            clf.fit(X[train_idx], y_encoded[train_idx])
            predictions[test_idx] = clf.predict(X[test_idx])

        return predictions

    if model_name == MODEL_FT_TRANSFORMER:

        # Also folded by hand, for a different reason: this model needs
        # the training fold's `groups` passed into fit() so its internal
        # early-stopping split can hold out whole source particles.
        # cross_val_predict has no way to forward per-fold groups into a
        # fit call, and a random inner split silently mis-stops the model
        # (see ft_transformer.py's note) — so the loop is explicit here.
        from ft_transformer import FTTransformerClassifier

        predictions = np.empty_like(y_encoded)

        for train_idx, test_idx in skf.split(X, y_encoded, groups):
            clf = FTTransformerClassifier(random_state=random_state)
            clf.fit(X[train_idx], y_encoded[train_idx], groups=groups[train_idx])
            predictions[test_idx] = clf.predict(X[test_idx])

        return predictions

    model = _fresh_sklearn_models(random_state)[model_name]
    return cross_val_predict(model, X, y_encoded, groups=groups, cv=skf)


def late_fusion_cross_validated_accuracy(
    scalar_df: pd.DataFrame, embeddings: np.ndarray, y_encoded: np.ndarray,
    groups: np.ndarray, n_folds: int, random_state: int = 42,
) -> np.ndarray:
    """
    Real, honest test of late fusion vs. the early-fusion default:
    fits TWO separate XGBoost models per fold — one on DINOv2
    embeddings alone, one on physics/morphology alone — and averages
    their predicted class probabilities, rather than concatenating
    both feature groups into one vector for a single model to learn
    over (early fusion, what every other model in this file does).

    Late fusion can win when one feature group is much noisier than
    the other, since early fusion lets a noisy group's irrelevant
    variance dilute the clean group's signal inside one shared
    decision boundary; late fusion never lets that happen because each
    submodel only ever sees its own feature group.

    Uses the SAME StratifiedGroupKFold splits as every other model
    here (`random_state` fixed) so the comparison is apples-to-apples,
    not an artifact of different folds landing differently.
    """

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    predictions = np.empty_like(y_encoded)

    for train_idx, test_idx in skf.split(embeddings, y_encoded, groups):

        scalar_pipeline = FusionPipeline(use_dinov2=False)
        embedding_pipeline = FusionPipeline(use_physics=False, use_morphology=False)

        X_scalar_train = scalar_pipeline.fit_transform(scalar_df.iloc[train_idx], embeddings[train_idx])
        X_embed_train = embedding_pipeline.fit_transform(scalar_df.iloc[train_idx], embeddings[train_idx])

        X_scalar_test = scalar_pipeline.transform(scalar_df.iloc[test_idx], embeddings[test_idx])
        X_embed_test = embedding_pipeline.transform(scalar_df.iloc[test_idx], embeddings[test_idx])

        scalar_model = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.6, colsample_bytree=0.6,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=random_state, verbosity=0, n_jobs=1,
        )
        embed_model = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.6, colsample_bytree=0.6,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=random_state, verbosity=0, n_jobs=1,
        )

        scalar_model.fit(X_scalar_train, y_encoded[train_idx])
        embed_model.fit(X_embed_train, y_encoded[train_idx])

        # Simple averaging of predicted probabilities — the standard
        # late-fusion baseline. Both submodels see the SAME class set
        # (fit on the same y_encoded), so their probability columns
        # already align without extra bookkeeping.
        combined_proba = (scalar_model.predict_proba(X_scalar_test) + embed_model.predict_proba(X_embed_test)) / 2.0

        predictions[test_idx] = np.argmax(combined_proba, axis=1)

    return predictions


def _metrics_row(y_true, y_pred, model_name: str) -> dict:

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


# ============================================================
# MAIN TRAIN + COMPARE ENTRY POINT
# ============================================================

def train_and_compare(prototype_records: List[dict], output_dir: str) -> dict:
    """
    Full training + evaluation pass:
      1. Fuse prototype features (physics + morphology + DINOv2).
      2. Cross-validated comparison of XGBoost vs. prototype few-shot
         -> model_comparison.csv
      3. Per-class precision/recall/F1 for the best model -> per_class_metrics.csv
      4. Confusion matrix for the best model -> confusion_matrix.csv
      5. Ablation (full / no-physics / embeddings-only), XGBoost as the
         fixed classifier -> ablation.csv
      6. Refit both models on the FULL prototype set for deployment,
         persist everything to <output_dir>/classifier/.
    """

    os.makedirs(os.path.join(output_dir, CLASSIFIER_DIRNAME), exist_ok=True)
    classifier_dir = os.path.join(output_dir, CLASSIFIER_DIRNAME)

    scalar_df, embeddings, labels, groups = prototypes_to_frame(prototype_records)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Real (pre-augmentation) example counts per class — counting raw
    # rows here would hide a genuinely thin class behind its augmented
    # view count (e.g. 5 real Spatter_Derived crops x 8 views = 40
    # rows, but still only 5 independent examples).
    real_counts = pd.DataFrame({"label": labels, "group": groups}).drop_duplicates("group")["label"].value_counts()
    low_classes = real_counts[real_counts < config.CLASSIFICATION.min_prototype_examples]
    if len(low_classes):
        print(
            "[WARN] Classes below the recommended minimum REAL example count "
            f"({config.CLASSIFICATION.min_prototype_examples}, augmented views don't count): "
            + ", ".join(f"{c} (n={n})" for c, n in low_classes.items())
        )

    n_folds = _safe_cv_folds(y_encoded, groups, config.ML.cv_folds)

    # ---------------- full fused feature space ----------------

    fusion = FusionPipeline(dinov2_pca_components=config.ML.dinov2_pca_components)
    X = fusion.fit_transform(scalar_df, embeddings)

    # MODEL_PROTOTYPE and MODEL_FT_TRANSFORMER are both hand-folded in
    # _cross_validated_predictions rather than living in the sklearn
    # roster dict, so they're prepended explicitly here.
    all_model_names = ([MODEL_PROTOTYPE, MODEL_FT_TRANSFORMER]
                       + list(_fresh_sklearn_models(config.ML.random_state).keys()))

    comparison_rows = []
    oof_predictions = {}

    for model_name in all_model_names:

        preds_encoded = _cross_validated_predictions(X, y_encoded, groups, model_name, n_folds, config.ML.random_state)
        oof_predictions[model_name] = preds_encoded
        comparison_rows.append(_metrics_row(y_encoded, preds_encoded, model_name))

    comparison_df = pd.DataFrame(comparison_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["model"]

    ordered_classes = [c for c in config.DEFECT_TAXONOMY if c in label_encoder.classes_]

    # Per-class metrics AND a confusion matrix for EVERY model, not
    # just the best one — lets Results show a full model-by-model
    # comparison (confusion matrix per model), not just a single
    # winner's.
    per_class_frames = []
    confusion_matrices = {}

    for model_name in all_model_names:

        preds_labels = label_encoder.inverse_transform(oof_predictions[model_name])

        report = classification_report(labels, preds_labels, output_dict=True, zero_division=0)
        model_per_class = (
            pd.DataFrame(report).T
            .drop(index=[i for i in ["accuracy", "macro avg", "weighted avg"] if i in report], errors="ignore")
            .reset_index().rename(columns={"index": "class_label"})
        )
        model_per_class.insert(0, "model", model_name)
        per_class_frames.append(model_per_class)

        cm = confusion_matrix(labels, preds_labels, labels=ordered_classes)
        confusion_matrices[model_name] = pd.DataFrame(cm, index=ordered_classes, columns=ordered_classes)

    per_class_df = pd.concat(per_class_frames, ignore_index=True)
    confusion_df = confusion_matrices[best_model_name]

    # Quantification correction matrix, built from the BEST model's
    # out-of-fold (held-out) predictions — never from a training fit,
    # which would understate the real error rate and bias the
    # correction in the same optimistic direction. This is what lets
    # population proportions be estimated accurately for reusability
    # analysis despite imperfect per-particle accuracy (see
    # quantification.py).
    correction_matrix = build_correction_matrix(
        labels,
        label_encoder.inverse_transform(oof_predictions[best_model_name]),
        ordered_classes,
    )

    # ---------------- ablation ----------------

    ablation_specs = [
        ("Full framework (physics + morphology + DINOv2)", dict(use_physics=True, use_morphology=True, use_dinov2=True)),
        ("Without physics-constrained features", dict(use_physics=False, use_morphology=True, use_dinov2=True)),
        ("Embeddings only (no fusion)", dict(use_physics=False, use_morphology=False, use_dinov2=True)),
    ]

    ablation_rows = []

    for config_label, kwargs in ablation_specs:

        ablation_fusion = FusionPipeline(dinov2_pca_components=config.ML.dinov2_pca_components, **kwargs)
        X_ablation = ablation_fusion.fit_transform(scalar_df, embeddings)

        preds = _cross_validated_predictions(X_ablation, y_encoded, groups, MODEL_XGBOOST, n_folds, config.ML.random_state)

        ablation_rows.append({
            "configuration": config_label,
            "accuracy": accuracy_score(y_encoded, preds),
            "f1_macro": f1_score(y_encoded, preds, average="macro", zero_division=0),
        })

    ablation_df = pd.DataFrame(ablation_rows)

    # ---------------- refit ALL models on the full prototype set ----------------
    # CV folds above are for honest metrics; every model gets refit on
    # 100% of the data here for actual deployment/inference.

    fitted_models = {MODEL_PROTOTYPE: PrototypeFewShotClassifier(k=config.ML.prototype_k).fit(X, y_encoded)}

    # Refit here too, not just in CV — otherwise a run where the
    # FT-Transformer wins would KeyError when the best model is looked
    # up for deployment. Groups are passed so its internal early-stopping
    # split stays group-aware on the full set as well.
    from ft_transformer import FTTransformerClassifier
    fitted_models[MODEL_FT_TRANSFORMER] = FTTransformerClassifier(
        random_state=config.ML.random_state).fit(X, y_encoded, groups=groups)

    for model_name, model in _fresh_sklearn_models(config.ML.random_state).items():
        model.fit(X, y_encoded)
        fitted_models[model_name] = model

    # Feature importance from every tree model that exposes it —
    # separate CSVs so Results can chart each one.
    feature_importance_frames = []

    for model_name in TREE_MODEL_NAMES:
        model = fitted_models[model_name]
        feature_importance_frames.append(pd.DataFrame({
            "model": model_name,
            "feature": fusion.feature_names(),
            "importance": model.feature_importances_,
        }))

    feature_importance_df = (
        pd.concat(feature_importance_frames, ignore_index=True)
        .sort_values(["model", "importance"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ---------------- persist ----------------

    fusion.save(os.path.join(classifier_dir, "fusion_pipeline.joblib"))
    joblib.dump(label_encoder, os.path.join(classifier_dir, "label_encoder.joblib"))
    joblib.dump(best_model_name, os.path.join(classifier_dir, "best_model_name.joblib"))

    for model_name, model in fitted_models.items():
        joblib.dump(model, os.path.join(classifier_dir, f"model_{model_name}.joblib"))

    comparison_df.to_csv(os.path.join(classifier_dir, "model_comparison.csv"), index=False)
    per_class_df.to_csv(os.path.join(classifier_dir, "per_class_metrics.csv"), index=False)
    confusion_df.to_csv(os.path.join(classifier_dir, "confusion_matrix.csv"))
    ablation_df.to_csv(os.path.join(classifier_dir, "ablation.csv"), index=False)
    feature_importance_df.to_csv(os.path.join(classifier_dir, "feature_importance.csv"), index=False)
    save_correction_matrix(correction_matrix, classifier_dir)

    confusion_dir = os.path.join(classifier_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)

    for model_name, cm_df in confusion_matrices.items():
        cm_df.to_csv(os.path.join(confusion_dir, f"{model_name}.csv"))

    print(f"[INFO] Best model (cross-validated): {best_model_name} — accuracy {comparison_df.iloc[0]['accuracy']:.3f}, macro-F1 {comparison_df.iloc[0]['f1_macro']:.3f}")

    return {
        "comparison": comparison_df,
        "per_class": per_class_df,
        "confusion_matrix": confusion_df,
        "confusion_matrices": confusion_matrices,
        "correction_matrix": correction_matrix,
        "ablation": ablation_df,
        "feature_importance": feature_importance_df,
        "best_model_name": best_model_name,
        "fusion": fusion,
        "label_encoder": label_encoder,
        "models": fitted_models,
    }


# ============================================================
# INFERENCE ON PRODUCTION PARTICLES
# ============================================================

def load_classifier(output_dir: str):

    classifier_dir = os.path.join(output_dir, CLASSIFIER_DIRNAME)

    if not os.path.exists(classifier_dir):
        return None

    models = {}

    for path in os.listdir(classifier_dir):

        if path.startswith("model_") and path.endswith(".joblib"):
            model_name = path[len("model_"):-len(".joblib")]
            models[model_name] = joblib.load(os.path.join(classifier_dir, path))

    return {
        "fusion": FusionPipeline.load(os.path.join(classifier_dir, "fusion_pipeline.joblib")),
        "label_encoder": joblib.load(os.path.join(classifier_dir, "label_encoder.joblib")),
        "models": models,
        "best_model_name": joblib.load(os.path.join(classifier_dir, "best_model_name.joblib")),
    }


def classify_particles(particles: List[Particle], classifier: dict, model_name: str = None) -> List[Particle]:
    """
    Predicts every particle's class using `model_name` (defaults to
    the cross-validated best model). Sets particle.prediction in
    place. Confidence is the winning class's predicted probability
    (XGBoost) or vote share (prototype few-shot) — both on a
    comparable [0,1] scale.

    Particles flagged by segmentation as border-truncated (see
    segmentation.estimate_border_truncation) are routed to the
    BORDER_TRUNCATED_LABEL sentinel instead of the model — a
    significantly cut-off particle can't be reliably typed even by
    eye, so forcing it through the classifier would just produce a
    confident-looking wrong answer. They still get a PredictionResult
    (so features.csv/the overlay always has a label to show), just not
    one of the 8 real defect classes, and they're excluded downstream
    from classification metrics, quantification, and reusability
    composition — see main.py/reusability.py.
    """

    model_name = model_name or classifier["best_model_name"]

    truncated = [p for p in particles if p.metadata.get("border_truncated")]
    classifiable = [p for p in particles if not p.metadata.get("border_truncated")]

    for particle in truncated:
        particle.prediction.label = config.BORDER_TRUNCATED_LABEL
        particle.prediction.confidence = 0.0
        particle.prediction.model_name = "excluded_border_truncated"
        particle.prediction.explanation = [
            f"Excluded from classification: {particle.metadata.get('border_fraction', 0.0):.0%} "
            "of this particle's outline is the image frame edge, not its real boundary — "
            "too much of the particle was never photographed to type it reliably."
        ]

    scalar_df, embeddings, usable_particles = particles_to_frame(classifiable)

    if not usable_particles:
        return particles

    X = classifier["fusion"].transform(scalar_df, embeddings)

    model = classifier["models"][model_name]
    label_encoder = classifier["label_encoder"]

    probs = model.predict_proba(X)
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_indices)

    for particle, pred_label, prob_row, idx in zip(usable_particles, pred_labels, probs, pred_indices):

        particle.prediction.label = str(pred_label)
        particle.prediction.confidence = float(prob_row[idx])
        particle.prediction.model_name = model_name
        particle.prediction.scores = {
            str(cls): float(p) for cls, p in zip(label_encoder.classes_, prob_row)
        }

    return particles


__all__ = [
    "PrototypeFewShotClassifier", "train_and_compare", "load_classifier",
    "classify_particles", "CLASSIFIER_DIRNAME", "MODEL_XGBOOST", "MODEL_PROTOTYPE",
]
