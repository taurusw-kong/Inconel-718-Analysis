#!/usr/bin/env python3
"""
Independent verification of the fine-tuned DINOv2/ViT accuracy.

The headline number (0.6937) is far above every frozen-embedding model,
so it should not be trusted on one run's say-so. This script re-derives
it from scratch and applies the standard checks that would expose the
ways such a number is usually wrong.

WHAT IT RUNS

  1. REPRODUCE (seed 42)
     Re-runs the exact evaluation and prints PER-FOLD accuracy, not just
     the pooled figure. One lucky fold cannot hide inside a mean here.

  2. DIFFERENT SPLITS (seeds 7, 123)
     Re-runs with different fold boundaries. A real effect survives
     re-splitting; a number that only exists for one partition of the
     data does not.

  3. SHUFFLED-LABEL CONTROL  <-- the decisive test
     Labels are randomly permuted WITHIN the group structure, so the
     model is asked to learn a mapping that genuinely does not exist.
     An honest pipeline must collapse to chance (1/8 = 12.5%). If it
     scores meaningfully above chance on shuffled labels, something is
     leaking and the real number is not trustworthy either. This is the
     single most informative check in the script.

  4. PER-PARTICLE PREDICTIONS -> CSV
     Every particle's true label, predicted label, and source image
     path, so the result can be checked BY EYE against the actual
     crops rather than taken on faith.

USAGE
    python3 verify_vit_accuracy.py                  # all checks
    python3 verify_vit_accuracy.py --quick          # reproduce + shuffle only
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold

from vit_finetune import cross_validated_vit, build_crop_tensors

PROTOTYPES = "output/prototypes.joblib"
CHANCE_CLASSES = 8


def per_fold_report(records, seed, square_crop=True):
    """Re-run CV and report each fold separately as well as pooled."""
    y_true, y_pred = cross_validated_vit(
        records, n_folds=5, random_state=seed,
        verbose=False, square_crop=square_crop,
    )

    groups = np.array([r["source_group_id"] for r in records])
    labels = np.array([r["class_label"] for r in records])

    # Recreate the same fold assignment to attribute predictions to folds.
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dummy = np.zeros((len(labels), 1))

    fold_accs = []
    for _, test_idx in skf.split(dummy, labels, groups):
        fold_accs.append(accuracy_score(y_true[test_idx], y_pred[test_idx]))

    return y_true, y_pred, fold_accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Run only the reproduce + shuffled-label checks.")
    ap.add_argument("--out", default="output/vit_verification")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = joblib.load(PROTOTYPES)
    print(f"Loaded {len(records)} prototype records "
          f"({len(set(r['source_group_id'] for r in records))} real particles)\n")

    summary = []

    # ---------------- 1. reproduce ----------------
    print("=" * 66)
    print("1. REPRODUCE  (seed 42, the reported configuration)")
    print("=" * 66, flush=True)

    y_true, y_pred, folds = per_fold_report(records, seed=42)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    for i, fa in enumerate(folds, 1):
        print(f"   fold {i}: {fa:.4f}")
    print(f"   spread: {min(folds):.4f} - {max(folds):.4f}  (sd {np.std(folds):.4f})")
    print(f"   POOLED ACCURACY: {acc:.4f}   macro-F1: {f1:.4f}")
    summary.append(("reproduce_seed42", acc, f1))

    pd.DataFrame({
        "source_path": [r["source_path"] for r in records],
        "source_group_id": [r["source_group_id"] for r in records],
        "augmentation_view": [r.get("augmentation_view", 0) for r in records],
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
    }).to_csv(os.path.join(args.out, "per_particle_predictions.csv"), index=False)
    print(f"   -> per_particle_predictions.csv written (check these by eye)")

    classes = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(args.out, "vit_confusion_matrix.csv"))
    print("\n" + classification_report(y_true, y_pred, zero_division=0))

    # ---------------- 2. shuffled-label control ----------------
    print("=" * 66)
    print("2. SHUFFLED-LABEL CONTROL  (must collapse to ~12.5% chance)")
    print("=" * 66, flush=True)

    rng = np.random.default_rng(0)

    # Permute labels at the GROUP level, so all 8 augmented views of one
    # particle keep a single (wrong) label. Shuffling row-by-row instead
    # would break that consistency and make the task artificially harder
    # than chance, hiding leakage rather than exposing it.
    groups = np.array([r["source_group_id"] for r in records])
    unique_groups = np.unique(groups)
    group_label = {g: records[int(np.flatnonzero(groups == g)[0])]["class_label"]
                   for g in unique_groups}
    shuffled_vals = rng.permutation([group_label[g] for g in unique_groups])
    shuffled_map = dict(zip(unique_groups, shuffled_vals))

    shuffled_records = []
    for r in records:
        r2 = dict(r)
        r2["class_label"] = shuffled_map[r["source_group_id"]]
        shuffled_records.append(r2)

    ys_true, ys_pred = cross_validated_vit(
        shuffled_records, n_folds=5, random_state=42,
        verbose=False, square_crop=True,
    )
    shuf_acc = accuracy_score(ys_true, ys_pred)
    print(f"   SHUFFLED ACCURACY: {shuf_acc:.4f}   (chance = {1/CHANCE_CLASSES:.4f})")
    verdict = ("PASS - no leakage detected" if shuf_acc < 0.20
               else "FAIL - scores above chance on random labels, investigate")
    print(f"   VERDICT: {verdict}")
    summary.append(("shuffled_labels", shuf_acc, float("nan")))

    # ---------------- 3. different splits ----------------
    if not args.quick:
        for seed in (7, 123):
            print("=" * 66)
            print(f"3. DIFFERENT SPLIT  (seed {seed})")
            print("=" * 66, flush=True)
            yt, yp, fl = per_fold_report(records, seed=seed)
            a = accuracy_score(yt, yp)
            f = f1_score(yt, yp, average="macro", zero_division=0)
            print(f"   folds: {' '.join(f'{x:.3f}' for x in fl)}")
            print(f"   POOLED ACCURACY: {a:.4f}   macro-F1: {f:.4f}")
            summary.append((f"seed_{seed}", a, f))

    # ---------------- summary ----------------
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    df = pd.DataFrame(summary, columns=["check", "accuracy", "macro_f1"])
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.out, "verification_summary.csv"), index=False)
    print(f"\nAll outputs in {args.out}")
    print("VERIFYDONE")


if __name__ == "__main__":
    main()
