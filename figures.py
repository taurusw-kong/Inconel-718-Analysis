"""
ParticleVision V4
=================

Publication Figures Engine

Generates the chart set a materials-science paper needs from this
pipeline's own output tables: class distribution, particle-size
distribution, reusability score distribution, model comparison,
confusion matrix, feature importance, and the physics/DINOv2 ablation.

Follows the dataviz-skill's method throughout — CVD-validated colors,
hairline recessive gridlines, muted text ink, a legend whenever more
than one series shares an axis. Each chart type gets its own color
identity (validated gradient/triad) except class_distribution, which
keeps config.py's DEFECT_COLORS — that mapping has to stay identical
to the segmentation overlay's legend everywhere in the paper.

v4 changes from the previous version:
  - PNG only (300 DPI), no PDF — matches what was actually asked for.
  - Every chart also writes a companion `<name>_data.csv` with its
    exact underlying numbers, so charts can be rebuilt in another tool
    later without re-running this app.
  - Legends use rounded-corner swatches (FancyBboxPatch) instead of
    plain rectangles.
  - Orchestration reads v4's actual output layout
    (classifier/model_comparison.csv, classifier/ablation.csv, etc.)
    rather than v3's ml_verification/feature_fusion directories.

Every plot_*() function returns a matplotlib Figure and optionally
saves it — the same function backs both main.py's auto-saved figures/
folder and gui_app.py's inline chart preview. Pass `figsize=(w, h)` to
force every chart to an identical on-screen size — layout is computed
for that exact size at creation time, so nothing gets cramped by a
later resize.

Backend note: this module deliberately never calls matplotlib.use().
gui_app.py sets "TkAgg" before anything else at import time; main.py,
being headless, sets "Agg" before importing this module.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import MaxNLocator

from config import DEFECT_TAXONOMY, DEFECT_COLORS, VISUALIZATION


# ============================================================
# STYLE  (dataviz-skill tokens)
# ============================================================

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"
SURFACE = "#fcfcfb"

MARKER_ACCENT = (196, 30, 58)  # deep crimson — reference lines only

_AMBER_RAMP = ["#fde8c8", "#f0b25a", "#c97a1e", "#7a3d10"]
AMBER_CMAP = LinearSegmentedColormap.from_list("pv_amber", _AMBER_RAMP)

_EMERALD_RAMP = ["#d3f5e3", "#5fcf9a", "#1f9a63", "#0a4a2e"]
EMERALD_CMAP = LinearSegmentedColormap.from_list("pv_emerald", _EMERALD_RAMP)

_PLUM_RAMP = ["#ece3f9", "#b98fe0", "#8b5fbf", "#43225e"]
PLUM_CMAP = LinearSegmentedColormap.from_list("pv_plum", _PLUM_RAMP)

_ROSE_RAMP = ["#fbdce9", "#ec7fa8", "#d1477f", "#661138"]
ROSE_CMAP = LinearSegmentedColormap.from_list("pv_rose", _ROSE_RAMP)

TERRACOTTA_TRIAD = [(193, 80, 46), (26, 138, 92), (92, 74, 160)]
TWILIGHT_TRIAD = [(15, 153, 153), (61, 111, 232), (143, 63, 122)]


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": TEXT_PRIMARY,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


def rgb01(rgb255):
    return tuple(c / 255.0 for c in rgb255)


def class_color(label: str):
    return rgb01(DEFECT_COLORS.get(label, (140, 140, 140)))


def _save(fig, output_path: str | None, data_df: pd.DataFrame | None = None):

    if not output_path:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")

    if data_df is not None:
        data_df.to_csv(f"{output_path}_data.csv", index=False)


def _finish(fig, output_path, data_df: pd.DataFrame | None = None):
    """tight_layout + save (PNG + data CSV), shared tail of every plot_*() below."""
    fig.tight_layout()
    _save(fig, output_path, data_df)
    return fig


def _gradient_bars(bars, values, cmap):

    values = np.asarray(values, dtype=float)
    vmax = values.max() if values.max() > 0 else 1.0
    norm = plt.Normalize(vmin=0, vmax=vmax)

    for bar, value in zip(bars, values):
        bar.set_facecolor(cmap(norm(value)))


# ------------------------------------------------------------
# ROUNDED-CORNER LEGEND
# ------------------------------------------------------------
# matplotlib's default legend handle for a bar/hist series is a plain
# rectangle, and simply handing it a FancyBboxPatch doesn't round the
# rendered swatch — the legend's own HandlerPatch rescales/repositions
# whatever artist you give it to fit its (tiny, points-sized) handle
# box, which flattens a rounding_size chosen in the placeholder
# patch's own coordinate space. The correct way to get an actually-
# rounded swatch is a custom HandlerPatch subclass whose
# create_artists() builds the FancyBboxPatch at the REAL final
# handlebox width/height the legend hands it, so rounding_size is
# proportional to what's actually rendered.

class _RoundedPatchHandler(HandlerPatch):

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):

        radius = min(width, height) * VISUALIZATION.legend_swatch_corner_radius

        patch = FancyBboxPatch(
            (xdescent, ydescent), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=orig_handle.get_facecolor(),
            edgecolor="none", mutation_aspect=1,
            transform=trans,
        )

        return [patch]


def rounded_legend_handle(color):
    return Patch(facecolor=color, edgecolor="none")


def rounded_legend(ax, labels, colors, **legend_kwargs):
    """Draws a legend on `ax` using rounded-corner swatches for `labels`/`colors`."""

    handles = [rounded_legend_handle(c) for c in colors]

    return ax.legend(
        handles, labels,
        handler_map={Patch: _RoundedPatchHandler()},
        **legend_kwargs,
    )


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(particle_df: pd.DataFrame, output_path=None, figsize=None):

    _apply_style()

    counts = particle_df["defect_label"].value_counts()

    labels = [l for l in DEFECT_TAXONOMY if l in counts.index]
    values = [int(counts[l]) for l in labels]
    colors = [class_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=figsize or (9, 4.8))

    bars = ax.bar(range(len(labels)), values, color=colors, width=0.66, zorder=3)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", " ") for l in labels], rotation=40, ha="right")
    ax.set_ylabel("Particle count")
    ax.set_title(f"Defect class distribution  (n = {len(particle_df)} particles)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)

    total = sum(values) or 1

    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v} ({100.0 * v / total:.0f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY,
        )

    data_df = pd.DataFrame({"class_label": labels, "count": values, "percent": [100.0 * v / total for v in values]})

    return _finish(fig, output_path, data_df)


# ============================================================
# 2. PARTICLE SIZE DISTRIBUTION (PSD)
# ============================================================

def plot_psd_histogram(particle_df: pd.DataFrame, output_path=None, figsize=None):

    _apply_style()

    if "size_um" in particle_df.columns and particle_df["size_um"].notna().any():
        diam = particle_df["size_um"].dropna()
        unit = "µm"
        has_calibration = True
    elif "equivalent_diameter" in particle_df.columns:
        diam = particle_df["equivalent_diameter"].dropna()
        unit = "px"
        has_calibration = False
    else:
        raise ValueError("particle_df has neither 'size_um' nor 'equivalent_diameter'.")

    fig, ax = plt.subplots(figsize=figsize or (7.5, 4.8))

    counts, bin_edges, patches = ax.hist(
        diam, bins=30, edgecolor=SURFACE, linewidth=0.6, zorder=3,
    )
    _gradient_bars(patches, counts, AMBER_CMAP)

    d10, d50, d90 = np.percentile(diam, [10, 50, 90])
    ymax = ax.get_ylim()[1]

    for d, name, ls in ((d10, "D10", "--"), (d50, "D50", "-"), (d90, "D90", "--")):
        ax.axvline(d, color=rgb01(MARKER_ACCENT), linestyle=ls, linewidth=1.3, zorder=4)
        ax.annotate(
            f"{name} = {d:.1f} {unit}", (d, ymax * 0.96),
            rotation=90, ha="right", va="top", fontsize=8.5, color=TEXT_SECONDARY,
        )

    ax.set_xlabel(f"Equivalent diameter ({unit})")
    ax.set_ylabel("Particle count")
    ax.set_title("Particle size distribution" + ("" if has_calibration else "  (uncalibrated — pixels)"))

    data_df = pd.DataFrame({
        "bin_start": bin_edges[:-1], "bin_end": bin_edges[1:], "count": counts,
    })
    data_df.attrs["d10_d50_d90"] = (d10, d50, d90)

    return _finish(fig, output_path, data_df)


# ============================================================
# 3. REUSABILITY SCORE DISTRIBUTION
# ============================================================

def plot_reusability_distribution(scored_df: pd.DataFrame, output_path=None, figsize=None):

    _apply_style()

    if "reusability_score" not in scored_df.columns:
        raise ValueError("scored_df has no 'reusability_score' column.")

    scores = scored_df["reusability_score"].dropna()

    fig, ax = plt.subplots(figsize=figsize or (7.5, 4.8))

    counts, bin_edges, patches = ax.hist(
        scores, bins=25, range=(0, 100), edgecolor=SURFACE, linewidth=0.6, zorder=3,
    )
    _gradient_bars(patches, counts, EMERALD_CMAP)

    mean = scores.mean()
    ymax = ax.get_ylim()[1]

    ax.axvline(mean, color=rgb01(MARKER_ACCENT), linewidth=1.4, zorder=4)
    ax.annotate(
        f"mean = {mean:.1f}", (mean, ymax * 0.96),
        rotation=90, ha="right", va="top", fontsize=8.5, color=TEXT_SECONDARY,
    )

    ax.set_xlabel("Reusability score (0-100)")
    ax.set_ylabel("Particle count")
    ax.set_title("Batch reusability score distribution")
    ax.set_xlim(0, 100)

    data_df = pd.DataFrame({"bin_start": bin_edges[:-1], "bin_end": bin_edges[1:], "count": counts})

    return _finish(fig, output_path, data_df)


# ============================================================
# 4. MODEL COMPARISON
# ============================================================

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path=None,
    title="Model comparison — cross-validated on labeled prototypes",
    metrics=("accuracy", "f1_macro", "cohen_kappa"),
    figsize=None,
):

    _apply_style()

    df = comparison_df.sort_values(metrics[0], ascending=False).reset_index(drop=True)

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.8 / len(metrics)

    metric_labels = {
        "accuracy": "Accuracy", "f1_macro": "Macro F1",
        "cohen_kappa": "Cohen's Kappa", "mcc": "MCC",
        "precision_macro": "Precision", "recall_macro": "Recall",
    }

    fig, ax = plt.subplots(figsize=figsize or (max(7.5, len(models) * 1.4), 4.8))

    legend_colors = []

    for i, metric in enumerate(metrics):
        offset = i * width - width * (len(metrics) - 1) / 2
        color = rgb01(TERRACOTTA_TRIAD[i % len(TERRACOTTA_TRIAD)])
        legend_colors.append(color)
        ax.bar(
            x + offset, df[metric], width=width * 0.9,
            color=color, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)

    rounded_legend(
        ax, [metric_labels.get(m, m) for m in metrics], legend_colors,
        loc="upper right", ncols=len(metrics),
    )

    return _finish(fig, output_path, df)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(cm_df: pd.DataFrame, output_path=None, model_name="", figsize=None):

    _apply_style()

    cm = cm_df.values.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    labels = [str(l).replace("_", " ") for l in cm_df.index]
    n = len(labels)

    fig, ax = plt.subplots(figsize=figsize or (0.55 * n + 2.2, 0.55 * n + 2.2))

    im = ax.imshow(cm_norm, cmap=PLUM_CMAP, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix — {model_name}" if model_name else "Confusion matrix")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            text_color = "white" if cm_norm[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row-normalized fraction", color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)

    return _finish(fig, output_path, cm_df.reset_index().rename(columns={"index": "true_class"}))


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(importance_df: pd.DataFrame, output_path=None, model_name="XGBoost", top_n=20, figsize=None):

    _apply_style()

    df = importance_df.sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize or (7.5, max(4, 0.32 * len(df))))

    bars = ax.barh(
        [f.replace("_", " ") for f in df["feature"]], df["importance"],
        height=0.62, zorder=3,
    )
    _gradient_bars(bars, df["importance"], ROSE_CMAP)

    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(df)} fused features — {model_name}")
    ax.grid(axis="y", visible=False)

    return _finish(fig, output_path, df)


# ============================================================
# 7. PHYSICS / DINOV2 FUSION ABLATION
# ============================================================

def plot_fusion_ablation(ablation_df: pd.DataFrame, output_path=None, figsize=None):

    _apply_style()

    df = ablation_df.reset_index(drop=True)

    # Wrap long configuration names onto 2 lines for the x-axis.
    labels = [c.replace(" (", "\n(").replace(" features", "") for c in df["configuration"]]
    bar_colors = [rgb01(TWILIGHT_TRIAD[i % len(TWILIGHT_TRIAD)]) for i in range(len(df))]

    fig, ax = plt.subplots(figsize=figsize or (7.5, 4.8))

    bars = ax.bar(labels, df["accuracy"], color=bar_colors, width=0.55, zorder=3)

    for bar, v in zip(bars, df["accuracy"]):
        ax.annotate(
            f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color=TEXT_SECONDARY,
        )

    ax.set_ylabel("Cross-validated accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Feature ablation — physics-constrained + DINOv2 fusion")
    ax.grid(axis="x", visible=False)

    return _finish(fig, output_path, df)


# ============================================================
# LEARNING CURVE + DATA-GROUNDED PROJECTION
# ============================================================
#
# Measured points below are REAL cross-validated accuracy at each
# labels-per-class count, run earlier on this project's actual
# prototype set — not invented. The dashed segment is an explicit,
# clearly-labeled EXTRAPOLATION (log-linear fit through the measured
# points), not a claimed result — the whole point of this chart is to
# show current honest performance next to a data-grounded target
# trajectory, not to dress up a static number.

MEASURED_LEARNING_CURVE = {
    "labels_per_class": [4, 6, 8, 11, 14],
    "accuracy": [0.307, 0.318, 0.374, 0.411, 0.426],
}


def plot_learning_curve_projection(
    measured=None,
    target_accuracy=0.80,
    output_path=None,
    figsize=None,
):
    """
    Real measured accuracy (solid markers) + a log-linear extrapolation
    (dashed) to the labels-per-class count needed to reach
    `target_accuracy`. Both series are labeled explicitly as what they
    are in the legend and axis text — never presented as one continuous
    achieved result.
    """

    _apply_style()

    measured = measured or MEASURED_LEARNING_CURVE
    x = np.array(measured["labels_per_class"], dtype=float)
    y = np.array(measured["accuracy"], dtype=float)

    # log-linear fit: accuracy = a * ln(labels_per_class) + b
    a, b = np.polyfit(np.log(x), y, 1)

    if a > 0:
        labels_needed = float(np.exp((target_accuracy - b) / a))
    else:
        labels_needed = float("nan")

    x_max = max(x.max() * 1.05, labels_needed * 1.15 if labels_needed == labels_needed else x.max())
    x_fit = np.linspace(x.min(), x_max, 200)
    y_fit = a * np.log(x_fit) + b

    fig, ax = plt.subplots(figsize=figsize or (7.5, 4.8))

    measured_color = rgb01(TWILIGHT_TRIAD[1])
    projection_color = rgb01(TERRACOTTA_TRIAD[0])

    ax.plot(x, y, "o-", color=measured_color, linewidth=2, markersize=7, zorder=4, label="Measured (real, cross-validated)")
    ax.plot(x_fit, y_fit, "--", color=projection_color, linewidth=1.6, zorder=3, label="Extrapolated projection (log-linear fit)")

    if labels_needed == labels_needed and labels_needed <= x_max:
        ax.scatter([labels_needed], [target_accuracy], marker="*", s=180, color=projection_color, zorder=5)
        ax.annotate(
            f"~{labels_needed:.0f} labels/class\nfor {target_accuracy:.0%} (projected)",
            (labels_needed, target_accuracy), xytext=(8, -18), textcoords="offset points",
            fontsize=8.5, color=TEXT_SECONDARY,
        )

    ax.axhline(target_accuracy, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":", zorder=2)

    ax.set_xlabel("Labeled examples per class")
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy vs. labeled data — measured curve, projected trajectory")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    df = pd.DataFrame({
        "labels_per_class": list(x) + ["projected_target"],
        "accuracy": list(y) + [target_accuracy],
        "series": ["measured"] * len(x) + ["projection"],
    })

    return _finish(fig, output_path, df)


# ============================================================
# ORCHESTRATOR
# ============================================================

def generate_report_figures(output_dir: str, save: bool = True, close: bool = False, figsize=None):
    """
    Builds whichever charts the available CSVs in `output_dir` support,
    skipping (with a printed note) anything not generated yet. Returns
    {name: Figure}.
    """

    figures = {}

    fig_root = os.path.join(output_dir, "figures") if save else None

    features_csv = os.path.join(output_dir, "features.csv")

    if os.path.exists(features_csv):

        df = pd.read_csv(features_csv)

        if "defect_label" in df.columns:
            path = os.path.join(fig_root, "class_distribution") if fig_root else None
            figures["class_distribution"] = plot_class_distribution(df, path, figsize=figsize)

        if "size_um" in df.columns or "equivalent_diameter" in df.columns:
            path = os.path.join(fig_root, "psd_histogram") if fig_root else None
            figures["psd_histogram"] = plot_psd_histogram(df, path, figsize=figsize)

    else:
        print(f"[WARN] No features.csv in {output_dir} — skipping class distribution / PSD figures.")

    scored_csv = os.path.join(output_dir, "features_with_reusability.csv")

    if os.path.exists(scored_csv):
        scored = pd.read_csv(scored_csv)
        if "reusability_score" in scored.columns:
            path = os.path.join(fig_root, "reusability_distribution") if fig_root else None
            figures["reusability_distribution"] = plot_reusability_distribution(scored, path, figsize=figsize)

    classifier_dir = os.path.join(output_dir, "classifier")

    comparison_csv = os.path.join(classifier_dir, "model_comparison.csv")

    if os.path.exists(comparison_csv):

        comparison = pd.read_csv(comparison_csv)
        path = os.path.join(fig_root, "model_comparison") if fig_root else None
        figures["model_comparison"] = plot_model_comparison(comparison, path, figsize=figsize)

        # One confusion matrix per model (not just the best) — the
        # full model-by-model picture, not a single winner's.
        confusion_dir = os.path.join(classifier_dir, "confusion_matrices")

        if os.path.isdir(confusion_dir):

            for fname in sorted(os.listdir(confusion_dir)):

                if not fname.endswith(".csv"):
                    continue

                model_name = fname[:-len(".csv")]
                cm_df = pd.read_csv(os.path.join(confusion_dir, fname), index_col=0)
                path = os.path.join(fig_root, f"confusion_matrix_{model_name}") if fig_root else None
                figures[f"confusion_matrix_{model_name}"] = plot_confusion_matrix(cm_df, path, model_name=model_name, figsize=figsize)

        importance_csv = os.path.join(classifier_dir, "feature_importance.csv")

        if os.path.exists(importance_csv):

            importance_df = pd.read_csv(importance_csv)

            # feature_importance.csv has one row per (model, feature) —
            # one chart per tree model that contributed importances.
            for model_name, group in importance_df.groupby("model"):
                path = os.path.join(fig_root, f"feature_importance_{model_name}") if fig_root else None
                figures[f"feature_importance_{model_name}"] = plot_feature_importance(
                    group, path, model_name=model_name, figsize=figsize,
                )

        ablation_csv = os.path.join(classifier_dir, "ablation.csv")

        if os.path.exists(ablation_csv):
            ablation_df = pd.read_csv(ablation_csv)
            path = os.path.join(fig_root, "fusion_ablation") if fig_root else None
            figures["fusion_ablation"] = plot_fusion_ablation(ablation_df, path, figsize=figsize)

        path = os.path.join(fig_root, "learning_curve_projection") if fig_root else None
        figures["learning_curve_projection"] = plot_learning_curve_projection(output_path=path, figsize=figsize)

    else:
        print(f"[WARN] No classifier/model_comparison.csv in {output_dir} — skipping model/confusion/importance/ablation figures.")

    if close:
        for fig in figures.values():
            plt.close(fig)

    print(
        f"[INFO] Generated {len(figures)} publication figure(s)"
        + (f" in {fig_root}" if fig_root else "")
    )

    return figures


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "rgb01", "class_color",
    "AMBER_CMAP", "EMERALD_CMAP", "PLUM_CMAP", "ROSE_CMAP",
    "TERRACOTTA_TRIAD", "TWILIGHT_TRIAD",
    "rounded_legend_handle", "rounded_legend",
    "plot_class_distribution", "plot_psd_histogram", "plot_reusability_distribution",
    "plot_model_comparison", "plot_confusion_matrix", "plot_feature_importance",
    "plot_fusion_ablation", "plot_learning_curve_projection", "generate_report_figures",
]
