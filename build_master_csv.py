#!/usr/bin/env python3
"""
Consolidate everything the figures need into ONE tidy CSV.

WHY
    paper_figures.py currently reads six different files (features.csv,
    prototypes.joblib, confusion_matrix.csv, model_comparison.csv,
    ablation.csv, shap_explanations.csv) and re-runs t-SNE on every
    call. That means regenerating a chart requires the whole pipeline
    output to be present and re-does expensive work each time.

    This script flattens all of it into master_data.csv once. After
    that, plot_from_master.py needs nothing but that single file.

FORMAT — long/tidy, because the sources have incompatible shapes
    A per-particle table (thousands of rows x dozens of columns), an
    8x8 confusion matrix, and a 13-row model table cannot share a wide
    layout. Long format holds all of them losslessly:

        block, entity, key, value

        block   which dataset the row belongs to
        entity  the row identity within that block
        key     the column/field name
        value   the cell (numeric or text, stored as text)

    plot_from_master.py pivots each block back to its natural shape.

    t-SNE coordinates are computed HERE and stored, so plotting is
    instant and — more importantly — reproducible: every regenerated
    contour plot uses the same projection instead of a fresh stochastic
    one.

USAGE
    python3 build_master_csv.py --run_dir /path/to/output6 \
                                --classifier_dir /path/to/output6 \
                                --out master_data.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


# Per-particle fields the charts actually consume. Kept to a short list
# on purpose: dumping all ~90 morphology columns would inflate the file
# several-fold for data no figure reads.
PARTICLE_FIELDS = [
    "image_name", "defect_label", "prediction_model", "confidence",
    "size_um", "is_attached", "boundary_contact_fraction",
    "circularity", "solidity", "aspect_ratio", "convexity", "eccentricity",
    "area", "perimeter", "equivalent_diameter",
]


def rows_from_particles(run_dir, out):
    feats = os.path.join(run_dir, "features.csv")
    reuse = os.path.join(run_dir, "features_with_reusability.csv")
    if not os.path.exists(feats):
        print(f"  [skip] no features.csv in {run_dir}")
        return

    df = pd.read_csv(feats)

    # Reusability lives in a second file; merge it in so the plotting
    # script never needs to know there were two.
    if os.path.exists(reuse):
        r = pd.read_csv(reuse)
        for col in ("reusability_score", "D_severity"):
            if col in r.columns and len(r) == len(df):
                df[col] = r[col].values

    fields = [c for c in PARTICLE_FIELDS + ["reusability_score", "D_severity"]
              if c in df.columns]

    for i, row in df.iterrows():
        entity = f"{row.get('image_name','?')}#{row.get('particle_id', i)}"
        for c in fields:
            v = row[c]
            if pd.isna(v):
                continue
            out.append(("particle", entity, c, v))
    print(f"  particles: {len(df)} rows x {len(fields)} fields")


def rows_from_matrix(path, block, out):
    if not os.path.exists(path):
        print(f"  [skip] missing {os.path.basename(path)}")
        return
    df = pd.read_csv(path, index_col=0)
    for true_c, r in df.iterrows():
        for pred_c, v in r.items():
            out.append((block, str(true_c), str(pred_c), v))

    # Record the row order explicitly. Pivoting the long form back sorts
    # entities alphabetically, which would silently re-order the chord
    # diagram's arcs away from the taxonomy order used everywhere else.
    for i, c in enumerate(df.index):
        out.append(("meta", f"{block}_class_order", str(i), str(c)))

    print(f"  {block}: {df.shape[0]}x{df.shape[1]}")


def rows_from_table(path, block, id_col, out):
    if not os.path.exists(path):
        print(f"  [skip] missing {os.path.basename(path)}")
        return
    df = pd.read_csv(path)
    if id_col not in df.columns:
        id_col = df.columns[0]
    for _, r in df.iterrows():
        for c in df.columns:
            if c == id_col or pd.isna(r[c]):
                continue
            out.append((block, str(r[id_col]), c, r[c]))
    print(f"  {block}: {len(df)} rows")


def rows_from_shap(run_dir, out):
    p = os.path.join(run_dir, "shap_explanations.csv")
    if not os.path.exists(p):
        print("  [skip] no shap_explanations.csv")
        return
    df = pd.read_csv(p)
    for i, r in df.iterrows():
        e = str(i)
        for c in ("model", "feature", "shap_value", "predicted_label", "particle_id"):
            if c in df.columns and not pd.isna(r[c]):
                out.append(("shap", e, c, r[c]))
    print(f"  shap: {len(df)} attributions")


def rows_from_embedding(run_dir, out, perplexity=30.0):
    """t-SNE the DINOv2 prototype embeddings ONCE and store the 2D coords."""
    p = os.path.join(run_dir, "prototypes.joblib")
    if not os.path.exists(p):
        print("  [skip] no prototypes.joblib (contour plot will be unavailable)")
        return
    import joblib
    from sklearn.manifold import TSNE

    recs = joblib.load(p)
    emb = np.stack([r["dinov2_embedding"] for r in recs])
    labels = [r["class_label"] for r in recs]

    print(f"  embedding: running t-SNE on {emb.shape[0]}x{emb.shape[1]} (once)...")
    xy = TSNE(n_components=2, perplexity=min(perplexity, (len(emb) - 1) / 3),
              init="pca", random_state=42).fit_transform(emb)

    for i, (x, y) in enumerate(xy):
        out.append(("embedding", str(i), "tsne_x", x))
        out.append(("embedding", str(i), "tsne_y", y))
        out.append(("embedding", str(i), "class_label", labels[i]))
    print(f"  embedding: {len(xy)} points stored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--classifier_dir", default=None,
                    help="dir holding classifier/ tables (default: run_dir)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = args.run_dir
    clf = os.path.join(args.classifier_dir or run, "classifier")
    out_path = args.out or os.path.join(run, "master_data.csv")

    rows = []
    print(f"building master CSV from {run}")
    rows_from_particles(run, rows)
    rows_from_matrix(os.path.join(clf, "confusion_matrix.csv"), "confusion", rows)
    rows_from_table(os.path.join(clf, "model_comparison.csv"), "model_comparison", "model", rows)
    rows_from_table(os.path.join(clf, "ablation.csv"), "ablation", "configuration", rows)
    rows_from_table(os.path.join(clf, "per_class_metrics.csv"), "per_class", "class_label", rows)
    rows_from_shap(run, rows)
    rows_from_embedding(run, rows)

    df = pd.DataFrame(rows, columns=["block", "entity", "key", "value"])
    df.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path}  ({len(df):,} rows, {size_mb:.1f} MB)")
    print("\nblocks:")
    for b, n in df.block.value_counts().items():
        print(f"  {b:18s} {n:>8,} rows")


if __name__ == "__main__":
    main()
