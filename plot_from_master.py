#!/usr/bin/env python3
"""
Generate any figure from master_data.csv alone.

Nothing else is needed — no pipeline output, no model files, no SAM2,
no re-running t-SNE. Point it at the CSV and ask for a chart.

    python3 plot_from_master.py --csv master_data.csv --chart all
    python3 plot_from_master.py --csv master_data.csv --chart sankey
    python3 plot_from_master.py --csv master_data.csv --chart contour --outdir ~/figs

HOW IT WORKS
    master_data.csv is long/tidy (block, entity, key, value). This
    script pivots each block back to the wide shape the chart functions
    in paper_figures.py already expect, writes those to a scratch
    directory, and calls the existing, tested chart code. Reusing that
    code rather than reimplementing it means a fix to a chart benefits
    both entry points.

    The one exception is the contour chart: master_data.csv stores
    PRE-COMPUTED t-SNE coordinates, so this script draws them directly
    instead of re-projecting. That makes repeated plots both instant and
    identical, which re-running a stochastic t-SNE would not be.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_figures as pf


# ============================================================
# LONG -> WIDE
# ============================================================

def load_blocks(csv_path):
    df = pd.read_csv(csv_path, dtype={"block": str, "entity": str, "key": str})
    return {b: g for b, g in df.groupby("block")}


def pivot(block_df, numeric=False):
    """entity x key table, values as strings unless numeric is asked for."""
    w = block_df.pivot_table(index="entity", columns="key", values="value",
                             aggfunc="first")
    if numeric:
        w = w.apply(pd.to_numeric, errors="coerce")
    return w


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def materialize(blocks, workdir):
    """
    Rebuild the intermediate CSVs the paper_figures chart functions read.
    Returns a dict of the paths that could be produced.
    """
    os.makedirs(os.path.join(workdir, "classifier"), exist_ok=True)
    paths = {}

    # ---- per-particle table (sankey, parallel, beeswarm, treemap) ----
    if "particle" in blocks:
        w = pivot(blocks["particle"]).reset_index()
        for c in w.columns:
            if c not in ("entity", "image_name", "defect_label", "prediction_model"):
                w[c] = _to_num(w[c])

        # is_attached round-trips through CSV as "True"/"False" text;
        # the sankey compares it against the string "True", so normalize
        # rather than leaving it as whatever pandas inferred.
        if "is_attached" in w.columns:
            w["is_attached"] = w["is_attached"].astype(str).str.strip().str.capitalize()

        w["particle_id"] = w["entity"].str.split("#").str[-1]
        p = os.path.join(workdir, "features.csv")
        w.to_csv(p, index=False)
        paths["features"] = p

        if "reusability_score" in w.columns:
            p2 = os.path.join(workdir, "features_with_reusability.csv")
            w.to_csv(p2, index=False)
            paths["reusability"] = p2

    # ---- confusion matrix (chord) ----
    if "confusion" in blocks:
        w = pivot(blocks["confusion"], numeric=True)

        # Restore the taxonomy row/column order recorded by
        # build_master_csv; without it the pivot returns alphabetical
        # order and the chord arcs come out rotated.
        order = None
        if "meta" in blocks:
            m = blocks["meta"]
            m = m[m.entity == "confusion_class_order"]
            if len(m):
                order = [v for _, v in sorted(
                    ((int(k), v) for k, v in zip(m["key"], m["value"])))]
        if order and set(order) == set(w.index):
            w = w.reindex(index=order, columns=order)
        else:
            w = w.reindex(index=w.columns.tolist())
        p = os.path.join(workdir, "classifier", "confusion_matrix.csv")
        w.to_csv(p)
        paths["confusion"] = p

    # ---- model comparison (bump) ----
    if "model_comparison" in blocks:
        w = pivot(blocks["model_comparison"], numeric=True).reset_index()
        w = w.rename(columns={"entity": "model"}).sort_values("accuracy", ascending=False)
        p = os.path.join(workdir, "classifier", "model_comparison.csv")
        w.to_csv(p, index=False)
        paths["model_comparison"] = p

    # ---- ablation (dumbbell) ----
    if "ablation" in blocks:
        w = pivot(blocks["ablation"], numeric=True).reset_index()
        w = w.rename(columns={"entity": "configuration"})
        p = os.path.join(workdir, "classifier", "ablation.csv")
        w.to_csv(p, index=False)
        paths["ablation"] = p

    # ---- SHAP ----
    if "shap" in blocks:
        w = pivot(blocks["shap"]).reset_index(drop=True)
        if "shap_value" in w.columns:
            w["shap_value"] = _to_num(w["shap_value"])
        p = os.path.join(workdir, "shap_explanations.csv")
        w.to_csv(p, index=False)
        paths["shap"] = p

    return paths


# ============================================================
# CONTOUR FROM STORED t-SNE
# ============================================================

def chart_contour_stored(blocks, basename, outdir, scale):
    """Same figure as paper_figures.chart_contour, minus the t-SNE step."""
    from scipy.stats import gaussian_kde
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = pivot(blocks["embedding"])
    xy = np.column_stack([_to_num(w["tsne_x"]).values, _to_num(w["tsne_y"]).values])
    labels = w["class_label"].astype(str).values

    ok = np.isfinite(xy).all(axis=1)
    xy, labels = xy[ok], labels[ok]

    W, H = 1500, 1150
    L, R_, T, B = 90, 330, 150, 90
    pw, ph = W - L - R_, H - T - B

    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    px_, py_ = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    x0, x1, y0, y1 = x0 - px_, x1 + px_, y0 - py_, y1 + py_

    sx = lambda v: L + pw * (v - x0) / (x1 - x0)
    sy = lambda v: T + ph * (1 - (v - y0) / (y1 - y0))

    classes = sorted(set(labels))
    svg = pf.svg_open(W, H, "DINOv2 Embedding Space by Defect Class",
                      f"stored t-SNE projection of {len(xy)} embeddings "
                      f"with per-class kernel-density contours")

    for i, (a, b) in enumerate(xy):
        c = pf.color_for(labels[i], classes.index(labels[i]))
        svg.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="2.6" fill="{c}" opacity="0.30"/>')

    gx = np.linspace(x0, x1, 120)
    gy = np.linspace(y0, y1, 120)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.vstack([GX.ravel(), GY.ravel()])
    fig, ax = plt.subplots()

    for idx, cls in enumerate(classes):
        sel = xy[labels == cls]
        if len(sel) < 5:
            continue
        try:
            Z = gaussian_kde(sel.T)(pts).reshape(GX.shape)
        except np.linalg.LinAlgError:
            continue
        col = pf.color_for(cls, idx)
        for segs in ax.contour(GX, GY, Z, levels=[Z.max() * 0.45]).allsegs:
            for seg in segs:
                if len(seg) < 3:
                    continue
                d = "M " + " L ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in seg) + " Z"
                svg.append(f'<path d="{d}" fill="{col}" fill-opacity="0.16" '
                           f'stroke="{col}" stroke-width="3" opacity="0.9"/>')
    plt.close(fig)

    ly = T + 6
    for idx, cls in enumerate(classes):
        n = int((labels == cls).sum())
        svg.append(f'<rect x="{W-R_+50}" y="{ly-11}" width="26" height="13" rx="4" '
                   f'fill="{pf.color_for(cls, idx)}"/>')
        svg.append(f'<text x="{W-R_+86}" y="{ly}" class="nvalue" font-size="13">'
                   f'{pf.esc(pf.pretty(cls))} ({n})</text>')
        ly += 30

    svg.append(f'<text x="{W/2}" y="{H-28}" text-anchor="middle" class="sub" font-size="12.5">'
               'Shaded region = each class&#8217;s dense core (45% of peak density).</text>')
    svg.append("</svg>")
    pf.write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# CLI
# ============================================================

CHARTS = ["sankey", "chord", "bump", "dumbbell", "parallel",
          "contour", "beeswarm", "shap", "treemap"]


def main():
    ap = argparse.ArgumentParser(
        description="Generate figures from a single master_data.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--csv", required=True, help="master_data.csv")
    ap.add_argument("--chart", default="all", choices=CHARTS + ["all", "list"])
    ap.add_argument("--outdir", default="figures_from_csv")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    blocks = load_blocks(args.csv)

    if args.chart == "list":
        print("blocks present:", ", ".join(sorted(blocks)))
        print("charts available with this file:")
        for c in CHARTS:
            need = {"sankey": "particle", "chord": "confusion", "bump": "model_comparison",
                    "dumbbell": "ablation", "parallel": "particle", "contour": "embedding",
                    "beeswarm": "particle", "shap": "shap", "treemap": "particle"}[c]
            print(f"  {c:10s} {'OK' if need in blocks else 'missing block: ' + need}")
        return

    wanted = CHARTS if args.chart == "all" else [args.chart]
    os.makedirs(args.outdir, exist_ok=True)

    with tempfile.TemporaryDirectory() as work:
        paths = materialize(blocks, work)

        jobs = {
            "sankey":   lambda: pf.chart_sankey(paths["features"], paths["reusability"],
                                                "fig_sankey", args.outdir, args.scale),
            "chord":    lambda: pf.chart_chord(paths["confusion"], "fig_chord", args.outdir, args.scale),
            "bump":     lambda: pf.chart_bump(paths["model_comparison"], "fig_bump", args.outdir, args.scale),
            "dumbbell": lambda: pf.chart_dumbbell(paths["ablation"], "fig_dumbbell", args.outdir, args.scale),
            "parallel": lambda: pf.chart_parallel(paths["features"], "fig_parallel", args.outdir, args.scale),
            "beeswarm": lambda: pf.chart_beeswarm(paths["reusability"], "fig_beeswarm", args.outdir, args.scale),
            "shap":     lambda: pf.chart_shap(paths["shap"], "fig_shap", args.outdir, args.scale),
            "treemap":  lambda: pf.chart_treemap(paths["features"], "fig_treemap", args.outdir, args.scale),
            "contour":  lambda: chart_contour_stored(blocks, "fig_contour", args.outdir, args.scale),
        }

        for c in wanted:
            print(f"[{c}]")
            try:
                jobs[c]()
            except KeyError as e:
                print(f"  skipped — master CSV has no data for this chart ({e})")
            except SystemExit as e:
                print(f"  skipped — {e}")
            except Exception as e:
                print(f"  FAILED — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
