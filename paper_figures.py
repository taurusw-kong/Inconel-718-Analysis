#!/usr/bin/env python3
"""
ParticleVision V4 — Publication Figure Generator
================================================

Standalone, dependency-light generator for the *non-traditional* journal
figures (Sankey, chord, bump, dumbbell, parallel coordinates). You point it
at a CSV, it writes an SVG (always) and a PNG (if cairosvg is available).

Nothing here invents data: every value plotted comes from the CSV you pass.

USAGE
-----
    python3 paper_figures.py sankey   --features features.csv \
                                      --reusability features_with_reusability.csv
    python3 paper_figures.py chord    --csv classifier/confusion_matrix.csv
    python3 paper_figures.py bump     --csv classifier/model_comparison.csv
    python3 paper_figures.py dumbbell --csv classifier/ablation.csv
    python3 paper_figures.py parallel --csv features.csv
    python3 paper_figures.py all      --dir output

Options common to every chart:
    --out NAME     output basename (default: fig_<chart>)
    --outdir DIR   where to write (default: ./paper_figures)
    --scale N      PNG pixel scale factor (default 2 => ~2x the SVG viewBox)

WHICH CSV FEEDS WHICH FIGURE  (paths relative to your output/ directory)
-----------------------------------------------------------------------
    sankey    features.csv  +  features_with_reusability.csv
    chord     classifier/confusion_matrix.csv
    bump      classifier/model_comparison.csv
    dumbbell  classifier/ablation.csv
    parallel  features.csv

PNG RENDERING
-------------
SVG is written with the standard library alone. PNG additionally needs:
    brew install cairo
    pip3 install cairosvg
If cairosvg is missing the script still writes the SVG and tells you so.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

# The first five charts are pure stdlib. The statistical ones (contour,
# beeswarm, SHAP, treemap) need numpy for density/quantile work; it is
# already a hard dependency of the pipeline this reads output from, so
# importing it here adds nothing new to install.
import numpy as np


# ============================================================
# THEME
# ============================================================

# ------------------------------------------------------------------
# THEME
# ------------------------------------------------------------------
# Journals print on white and generally reject dark-background figures:
# they consume ink, reproduce poorly in greyscale, and look out of place
# beside typeset text. LIGHT is therefore the default; DARK remains for
# screen use and presentations.
#
# The palette is unchanged between themes -- only the ground, ink and
# gridlines swap -- so a figure keeps its identity across both.

THEME = "light"

_THEMES = {
    "light": dict(bg="#ffffff", bg_mid="#ffffff", ink="#1a1a1a",
                  grid="#000000", grid_op=0.12, dot_op=0.0, sub_op=0.75),
    "dark":  dict(bg="#0a0c18", bg_mid="#161a2e", ink="#eef2ff",
                  grid="#ffffff", grid_op=0.10, dot_op=0.05, sub_op=0.6),
}

def set_theme(name: str):
    global THEME, BG_DARK, BG_MID, INK, GRID, GRID_OP, DOT_OP, SUB_OP
    THEME = name if name in _THEMES else "light"
    t = _THEMES[THEME]
    BG_DARK, BG_MID, INK = t["bg"], t["bg_mid"], t["ink"]
    GRID, GRID_OP, DOT_OP, SUB_OP = t["grid"], t["grid_op"], t["dot_op"], t["sub_op"]

BG_DARK = BG_MID = INK = GRID = None
GRID_OP = DOT_OP = SUB_OP = 0.0
set_theme("light")

# CVD-safe categorical palette, matched to the pipeline overview figure.
CLASS_COLOR = {
    "Bonded_Particle":     "#ff6ec7",
    "Clip_Cap":            "#ffd93d",
    "Molten_Satellite":    "#ff5e5e",
    "Elongated_Particle":  "#5ec8ff",
    "Agglomerate":         "#b83ff0",
    "Oversized_Spherical": "#3ddc97",
    "Deformed_Particle":   "#ff9f1c",
    "Perfect_Spherical":   "#a78bfa",
    "Spatter_Derived":     "#7cfcff",
}
FALLBACK_CYCLE = ["#ff6ec7", "#ffd93d", "#ff5e5e", "#5ec8ff", "#b83ff0",
                  "#3ddc97", "#ff9f1c", "#a78bfa", "#7cfcff", "#f472b6"]

BUCKET_COLOR = {"Reusable": "#34d399", "Marginal": "#fbbf24", "Non-Reusable": "#f87171"}
ATTACH_COLOR = {"Attached": "#a78bfa", "Isolated": "#7cfcff"}


def pretty(name: str) -> str:
    """Snake_case class/model id -> human label for axis and node text."""
    special = {
        "Oversized_Spherical": "Oversized Sphere",
        "Perfect_Spherical": "Perfect Sphere",
        "Clip_Cap": "Clip-Cap",
        "f1_macro": "Macro F1",
        "precision_macro": "Precision",
        "recall_macro": "Recall",
        "cohen_kappa": "Cohen κ",
        "mcc": "MCC",
        "accuracy": "Accuracy",
        "svm_rbf": "SVM (RBF)",
        "knn": "k-NN",
        "mlp": "MLP",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "ft_transformer": "FT-Transformer",
        "tabpfn": "TabPFN",
        "gradient_boosting": "Gradient Boosting",
        "prototype_few_shot": "Prototype Few-Shot",
        "dinov2_vit_finetuned": "Fine-tuned DINOv2/ViT",
    }
    if name in special:
        return special[name]
    return name.replace("_", " ").title()


def color_for(name: str, index: int = 0) -> str:
    return CLASS_COLOR.get(name, FALLBACK_CYCLE[index % len(FALLBACK_CYCLE)])


def esc(text) -> str:
    """XML-escape a label so stray & or < can't corrupt the SVG."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _shared_defs():
    return f"""
  <radialGradient id="bg" cx="50%" cy="35%" r="80%">
    <stop offset="0%" stop-color="{BG_MID}"/>
    <stop offset="100%" stop-color="{BG_DARK}"/>
  </radialGradient>
  <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="6" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <pattern id="dots" width="26" height="26" patternUnits="userSpaceOnUse">
    <circle cx="2" cy="2" r="1.1" fill="{GRID}" opacity="{DOT_OP}"/>
  </pattern>
  <style>
    .title   {{ font-family: Georgia, 'Times New Roman', serif; font-weight: 700; fill: {INK}; }}
    .sub     {{ font-family: Helvetica, Arial, sans-serif; fill: {INK}; opacity: {SUB_OP}; }}
    .colhead {{ font-family: Georgia, serif; font-weight: 700; font-size: 15px; fill: {INK};
                opacity: 0.75; letter-spacing: 1.5px; }}
    .nlabel  {{ font-family: Helvetica, Arial, sans-serif; font-size: 14.5px; fill: {INK}; font-weight: 600; }}
    .nvalue  {{ font-family: Helvetica, Arial, sans-serif; font-size: 12.5px; fill: {INK}; opacity: 0.65; }}
    .tick    {{ font-family: Helvetica, Arial, sans-serif; font-size: 12.5px; fill: {INK}; opacity: 0.55; }}
  </style>
"""


def svg_open(w: int, h: int, title: str, subtitle: str) -> list:
    out = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
           "<defs>", _shared_defs(), "</defs>",
           f'<rect width="{w}" height="{h}" fill="{BG_DARK}"/>']
    if DOT_OP > 0:
        out.append(f'<rect width="{w}" height="{h}" fill="url(#dots)"/>')
    if title:
        out.append(f'<text x="{w/2}" y="46" text-anchor="middle" class="title" '
                   f'font-size="26">{esc(title)}</text>')
    if subtitle:
        out.append(f'<text x="{w/2}" y="72" text-anchor="middle" class="sub" '
                   f'font-size="14">{esc(subtitle)}</text>')
    return out


def write_out(svg_parts: list, basename: str, outdir: str, w: int, h: int, scale: float):
    os.makedirs(outdir, exist_ok=True)
    svg_path = os.path.join(outdir, basename + ".svg")
    with open(svg_path, "w") as f:
        f.write("\n".join(svg_parts))
    print(f"  wrote {svg_path}")

    try:
        import cairosvg
    except Exception:
        print("  (PNG skipped — `pip3 install cairosvg` and `brew install cairo` to enable)")
        return
    png_path = os.path.join(outdir, basename + ".png")
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path,
                         output_width=int(w * scale), output_height=int(h * scale))
        print(f"  wrote {png_path}  ({int(w*scale)}x{int(h*scale)} px)")
    except OSError as exc:
        # cairocffi raises OSError when the native cairo library isn't on the
        # dynamic-linker path — the usual macOS case, where Homebrew installs
        # to /opt/homebrew/lib but Python doesn't look there by default.
        print(f"  (PNG skipped — cairo library not loadable: {exc})")
        print("   try:  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python3 paper_figures.py ...")



# Frame-truncated particles are detected but deliberately not classified,
# because a partial outline gives unreliable morphology. They are reported
# as a count in the text; including them in figures would add a category
# that is an artefact of framing rather than a defect type, and would
# dominate several charts at ~52% of detections.
TRUNCATED_MODEL = "excluded_border_truncated"
TRUNCATED_LABEL = "Edge_Truncated"


def drop_truncated(rows):
    """Remove frame-truncated detections from a list of csv.DictReader rows."""
    return [r for r in rows
            if r.get("prediction_model") != TRUNCATED_MODEL
            and r.get("defect_label") != TRUNCATED_LABEL]


def read_csv(path: str) -> list:
    if not os.path.exists(path):
        sys.exit(f"ERROR: CSV not found: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# SHARED GEOMETRY HELPERS
# ============================================================

def stack_nodes(items, total_height, gap):
    """items: [(key, value)] -> {key: (y0, y1, value)}, sized proportionally."""
    n = len(items)
    total_val = sum(v for _, v in items) or 1
    usable = total_height - gap * max(n - 1, 0)
    out, y = {}, 0.0
    for k, v in items:
        h = usable * (v / total_val)
        out[k] = (y, y + h, v)
        y += h + gap
    return out


def ribbon(x0, y0a, y0b, x1, y1a, y1b, color, opacity=0.45):
    """Horizontal cubic-bezier ribbon between two vertical edges."""
    xm = (x0 + x1) / 2
    return (f'<path d="M {x0:.1f},{y0a:.1f} C {xm:.1f},{y0a:.1f} {xm:.1f},{y1a:.1f} '
            f'{x1:.1f},{y1a:.1f} L {x1:.1f},{y1b:.1f} C {xm:.1f},{y1b:.1f} '
            f'{xm:.1f},{y0b:.1f} {x0:.1f},{y0b:.1f} Z" fill="{color}" opacity="{opacity}"/>')


def polar(cx, cy, r, theta):
    return cx + r * math.cos(theta), cy + r * math.sin(theta)


# ============================================================
# 1. SANKEY  —  features.csv + features_with_reusability.csv
# ============================================================

def reusability_bucket(score: float) -> str:
    if score >= 80:
        return "Reusable"
    if score >= 65:
        return "Marginal"
    return "Non-Reusable"


def chart_sankey(features_csv, reusability_csv, basename, outdir, scale):
    rows = drop_truncated(read_csv(features_csv))
    reuse_rows = drop_truncated(read_csv(reusability_csv))

    # Stage B: attachment geometry, split per predicted class
    bc = defaultdict(lambda: [0, 0])          # class -> [attached, isolated]
    for r in rows:
        bc[r["defect_label"]][0 if r.get("is_attached") == "True" else 1] += 1

    class_order = [c for c, _ in sorted(bc.items(), key=lambda kv: -sum(kv[1]))]
    class_totals = {c: sum(v) for c, v in bc.items()}
    total = len(rows)
    total_attached = sum(v[0] for v in bc.values())
    total_isolated = sum(v[1] for v in bc.values())

    # Stage D: reusability outcome, split per predicted class
    cd = defaultdict(lambda: defaultdict(int))
    for r in reuse_rows:
        raw = r.get("reusability_score", "")
        if raw in ("", None):
            continue
        cd[r["defect_label"]][reusability_bucket(float(raw))] += 1

    bucket_order = ["Reusable", "Marginal", "Non-Reusable"]
    bucket_totals = {b: sum(cd[c][b] for c in class_order) for b in bucket_order}

    W, H = 1700, 1000
    TOP, BOT, NODE_W, GAP = 150, 60, 26, 6
    BAND = H - TOP - BOT
    COL = {"A": 90, "B": 470, "C": 900, "D": 1560}

    nodeA = (0.0, float(BAND), total)
    nodeB = stack_nodes([("Attached", total_attached), ("Isolated", total_isolated)], BAND, GAP)
    nodeC = stack_nodes([(c, class_totals[c]) for c in class_order], BAND, GAP)
    nodeD = stack_nodes([(b, bucket_totals[b]) for b in bucket_order], BAND, GAP)

    def yt(v):
        return TOP + v

    svg = svg_open(W, H,
                   "Particle Flow Through the Physics-Constrained Classification Pipeline",
                   f"Real distribution across {total} classified particles "
                   f"· {os.path.basename(features_csv)} + {os.path.basename(reusability_csv)}")

    for label, x in [("SEGMENTED", COL["A"]), ("ATTACHMENT GEOMETRY", COL["B"]),
                     ("PREDICTED DEFECT CLASS", COL["C"]), ("REUSABILITY OUTCOME", COL["D"])]:
        svg.append(f'<text x="{x + NODE_W/2}" y="{TOP - 26}" text-anchor="middle" '
                   f'class="colhead">{label}</text>')

    # A -> B
    cursor = nodeA[0]
    for key, (by0, by1, bval) in nodeB.items():
        seg = (nodeA[1] - nodeA[0]) * (bval / nodeA[2])
        svg.append(ribbon(COL["A"] + NODE_W, yt(cursor), yt(cursor + seg),
                          COL["B"], yt(by0), yt(by1), ATTACH_COLOR[key], 0.45))
        cursor += seg

    # B -> C
    b_cursor = {k: nodeB[k][0] for k in nodeB}
    for c in class_order:
        cy0, cy1, cval = nodeC[c]
        c_cursor = cy0
        for key, n in [("Attached", bc[c][0]), ("Isolated", bc[c][1])]:
            if not n:
                continue
            by0, by1, bval = nodeB[key]
            seg_b = (by1 - by0) * (n / bval)
            seg_c = (cy1 - cy0) * (n / cval)
            svg.append(ribbon(COL["B"] + NODE_W, yt(b_cursor[key]), yt(b_cursor[key] + seg_b),
                              COL["C"], yt(c_cursor), yt(c_cursor + seg_c),
                              color_for(c), 0.42))
            b_cursor[key] += seg_b
            c_cursor += seg_c

    # C -> D
    d_cursor = {k: nodeD[k][0] for k in nodeD}
    for c in class_order:
        cy0, cy1, cval = nodeC[c]
        c_cursor = cy0
        for b in bucket_order:
            n = cd[c][b]
            if not n:
                continue
            dy0, dy1, dval = nodeD[b]
            seg_c = (cy1 - cy0) * (n / cval)
            seg_d = (dy1 - dy0) * (n / dval)
            svg.append(ribbon(COL["C"] + NODE_W, yt(c_cursor), yt(c_cursor + seg_c),
                              COL["D"], yt(d_cursor[b]), yt(d_cursor[b] + seg_d),
                              color_for(c), 0.42))
            c_cursor += seg_c
            d_cursor[b] += seg_d

    def node(x, y0, y1, val, label, fill, side="right", glow=False):
        h = max(y1 - y0, 2)
        f = ' filter="url(#glow)"' if glow else ""
        svg.append(f'<rect x="{x}" y="{yt(y0):.1f}" width="{NODE_W}" height="{h:.1f}" '
                   f'rx="6" fill="{fill}"{f}/>')
        ty = yt((y0 + y1) / 2)
        if side == "right":
            svg.append(f'<text x="{x + NODE_W + 10}" y="{ty - 4:.1f}" class="nlabel">{esc(label)}</text>')
            svg.append(f'<text x="{x + NODE_W + 10}" y="{ty + 13:.1f}" class="nvalue">n = {val}</text>')
        else:
            svg.append(f'<text x="{x - 10}" y="{ty - 4:.1f}" text-anchor="end" class="nlabel">{esc(label)}</text>')
            svg.append(f'<text x="{x - 10}" y="{ty + 13:.1f}" text-anchor="end" class="nvalue">n = {val}</text>')

    node(COL["A"], nodeA[0], nodeA[1], nodeA[2], "Segmented Particles", "#3ddc97", "right", glow=True)
    for k in ["Attached", "Isolated"]:
        node(COL["B"], *nodeB[k], k, ATTACH_COLOR[k])
    for c in class_order:
        node(COL["C"], *nodeC[c], pretty(c), color_for(c))
    for b in bucket_order:
        node(COL["D"], *nodeD[b], b, BUCKET_COLOR[b], side="left")

    svg.append(f'<text x="{W/2}" y="{H-22}" text-anchor="middle" class="sub" font-size="12.5">'
               'Reusability buckets: Reusable ≥ 80 · Marginal 65–79.9 · '
               'Non-Reusable &lt; 65 (physics-weighted score)</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 2. CHORD  —  classifier/confusion_matrix.csv
# ============================================================

def chart_chord(csv_path, basename, outdir, scale):
    with open(csv_path, newline="") as f:
        reader = list(csv.reader(f))
    header = reader[0][1:]
    matrix, labels = [], []
    for row in reader[1:]:
        labels.append(row[0])
        matrix.append([float(v) for v in row[1:]])

    n = len(labels)
    correct = sum(matrix[i][i] for i in range(n))
    grand = sum(sum(r) for r in matrix)

    # Arc size = total MISCLASSIFICATIONS touching a class (as true or predicted).
    # Diagonal is excluded on purpose: this figure is about confusion structure,
    # not overall accuracy, which the subtitle reports numerically instead.
    err_weight = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                err_weight[i] += matrix[i][j]
                err_weight[j] += matrix[i][j]
    total_err = sum(err_weight) or 1

    W, H = 1500, 1250
    CX, CY, R, TH = W / 2, 690, 400, 34
    GAP_A = 0.022

    svg = svg_open(W, H, "Inter-Class Confusion Structure",
                   f"Chord width = misclassifications between class pairs · "
                   f"overall accuracy {correct/grand:.1%} · {os.path.basename(csv_path)}")

    usable = 2 * math.pi - GAP_A * n
    angles, a = {}, -math.pi / 2
    for i, lab in enumerate(labels):
        span = usable * (err_weight[i] / total_err)
        angles[i] = (a, a + span)
        a += span + GAP_A

    # ---- ribbons (each off-diagonal cell, drawn once per direction) ----
    sub_cursor = {i: angles[i][0] for i in range(n)}
    flows = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                flows.append((matrix[i][j], i, j))
    flows.sort(reverse=True)          # big flows first so small ones stay visible on top

    # With 8 classes there are up to 56 off-diagonal flows; drawn at uniform
    # opacity they average into an unreadable hairball. Scaling opacity by
    # magnitude keeps every flow present (nothing is hidden) while letting the
    # dominant confusion pairs — the actual finding — read at a glance.
    max_flow = max(f[0] for f in flows) if flows else 1.0
    min_flow = min(f[0] for f in flows) if flows else 0.0
    flow_span = (max_flow - min_flow) or 1.0

    for val, i, j in flows:
        op = 0.10 + 0.45 * ((val - min_flow) / flow_span)
        span_i = (angles[i][1] - angles[i][0]) * (val / err_weight[i])
        span_j = (angles[j][1] - angles[j][0]) * (val / err_weight[j])
        a0, a1 = sub_cursor[i], sub_cursor[i] + span_i
        b0, b1 = sub_cursor[j], sub_cursor[j] + span_j
        sub_cursor[i] += span_i
        sub_cursor[j] += span_j

        r = R - TH
        p0 = polar(CX, CY, r, a0)
        p1 = polar(CX, CY, r, a1)
        q0 = polar(CX, CY, r, b0)
        q1 = polar(CX, CY, r, b1)
        # Quadratic beziers bowed toward the centre give the classic chord look.
        svg.append(
            f'<path d="M {p0[0]:.1f},{p0[1]:.1f} '
            f'A {r:.1f},{r:.1f} 0 0,1 {p1[0]:.1f},{p1[1]:.1f} '
            f'Q {CX:.1f},{CY:.1f} {q1[0]:.1f},{q1[1]:.1f} '
            f'A {r:.1f},{r:.1f} 0 0,1 {q0[0]:.1f},{q0[1]:.1f} '
            f'Q {CX:.1f},{CY:.1f} {p0[0]:.1f},{p0[1]:.1f} Z" '
            f'fill="{color_for(labels[i], i)}" opacity="{op:.2f}"/>')

    # ---- outer arcs + labels ----
    for i, lab in enumerate(labels):
        a0, a1 = angles[i]
        outer0, outer1 = polar(CX, CY, R, a0), polar(CX, CY, R, a1)
        inner1, inner0 = polar(CX, CY, R - TH, a1), polar(CX, CY, R - TH, a0)
        large = 1 if (a1 - a0) > math.pi else 0
        svg.append(
            f'<path d="M {outer0[0]:.1f},{outer0[1]:.1f} '
            f'A {R},{R} 0 {large},1 {outer1[0]:.1f},{outer1[1]:.1f} '
            f'L {inner1[0]:.1f},{inner1[1]:.1f} '
            f'A {R-TH},{R-TH} 0 {large},0 {inner0[0]:.1f},{inner0[1]:.1f} Z" '
            f'fill="{color_for(lab, i)}" opacity="0.92"/>')

        mid = (a0 + a1) / 2
        lx, ly = polar(CX, CY, R + 26, mid)
        deg = math.degrees(mid)
        flip = 90 < (deg % 360) < 270
        rot = deg + 180 if flip else deg
        anchor = "end" if flip else "start"
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="nlabel" '
                   f'transform="rotate({rot:.1f} {lx:.1f} {ly:.1f})">{esc(pretty(lab))}</text>')

    svg.append(f'<text x="{CX}" y="{CY-6}" text-anchor="middle" class="title" font-size="20">'
               f'{int(grand - correct)}</text>')
    svg.append(f'<text x="{CX}" y="{CY+16}" text-anchor="middle" class="nvalue">'
               'misclassifications</text>')
    svg.append(f'<text x="{W/2}" y="{H-24}" text-anchor="middle" class="sub" font-size="12.5">'
               'Arc length scales with total confusions involving that class; ribbon opacity '
               'scales with flow size; diagonal (correct predictions) omitted</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 3. BUMP  —  classifier/model_comparison.csv
# ============================================================

def chart_bump(csv_path, basename, outdir, scale):
    rows = read_csv(csv_path)
    # Only rank on metrics every model actually reports. A model with a
    # blank cell (e.g. a metric that was never computed for it) would
    # otherwise crash the sort or, worse, be silently ranked as zero.
    def _complete(col):
        return all(str(r.get(col, "")).strip() not in ("", "nan", "None") for r in rows)

    metrics = [c for c in ["accuracy", "f1_macro", "precision_macro", "recall_macro",
                           "cohen_kappa", "mcc"]
               if c in rows[0] and _complete(c)]
    if not metrics:
        sys.exit("ERROR: no metric is populated for every model")
    models = [r["model"] for r in rows]

    ranks = {}
    for m in metrics:
        ordered = sorted(rows, key=lambda r: -float(r[m]))
        for pos, r in enumerate(ordered):
            ranks[(r["model"], m)] = pos + 1

    W, H = 1500, 950
    L, R_, T, B = 330, 300, 150, 90
    plot_w, plot_h = W - L - R_, H - T - B
    n_models = len(models)

    def x_of(i):
        return L + plot_w * (i / max(len(metrics) - 1, 1))

    def y_of(rank):
        return T + plot_h * ((rank - 1) / max(n_models - 1, 1))

    svg = svg_open(W, H, "Model Ranking Consistency Across Evaluation Metrics",
                   f"Rank 1 = best · {n_models} models · {os.path.basename(csv_path)}")

    for i, m in enumerate(metrics):
        svg.append(f'<line x1="{x_of(i):.1f}" y1="{T-16}" x2="{x_of(i):.1f}" y2="{T+plot_h+16}" '
                   f'stroke="{GRID}" stroke-width="1" opacity="{GRID_OP}"/>')
        svg.append(f'<text x="{x_of(i):.1f}" y="{T-28}" text-anchor="middle" class="colhead" '
                   f'font-size="13">{esc(pretty(m))}</text>')

    for rank in range(1, n_models + 1):
        svg.append(f'<text x="30" y="{y_of(rank)+5:.1f}" class="tick" font-size="12">#{rank}</text>')

    for idx, r in enumerate(rows):
        model = r["model"]
        col = FALLBACK_CYCLE[idx % len(FALLBACK_CYCLE)]
        pts = [(x_of(i), y_of(ranks[(model, m)])) for i, m in enumerate(metrics)]

        d = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"
        for k in range(1, len(pts)):
            x0, y0 = pts[k-1]
            x1, y1 = pts[k]
            xm = (x0 + x1) / 2
            d += f" C {xm:.1f},{y0:.1f} {xm:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
        svg.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="3.5" opacity="0.85"/>')

        for (px, py) in pts:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{col}" '
                       f'stroke="{BG_DARK}" stroke-width="2.5"/>')

        svg.append(f'<text x="{L-26}" y="{pts[0][1]+5:.1f}" text-anchor="end" class="nlabel" '
                   f'fill="{col}">{esc(pretty(model))}</text>')
        svg.append(f'<text x="{pts[-1][0]+18:.1f}" y="{pts[-1][1]+5:.1f}" class="nvalue">'
                   f'{float(r[metrics[0]]):.3f}</text>')

    svg.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" class="sub" font-size="12.5">'
               'A flat line = a model that holds its rank across every metric; crossings reveal '
               'metric-specific trade-offs a summary table hides</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 4. DUMBBELL  —  classifier/ablation.csv
# ============================================================

def chart_dumbbell(csv_path, basename, outdir, scale):
    rows = read_csv(csv_path)
    baseline = float(rows[0]["accuracy"])
    base_name = rows[0]["configuration"]

    W = 1500
    row_h = 108
    T, B, L, R_ = 165, 120, 620, 150
    H = T + row_h * len(rows) + B
    plot_w = W - L - R_

    vals = [float(r["accuracy"]) for r in rows] + [float(r["f1_macro"]) for r in rows]
    lo = max(min(vals) - 0.04, 0.0)
    hi = min(max(vals) + 0.04, 1.0)

    def x_of(v):
        return L + plot_w * ((v - lo) / (hi - lo))

    svg = svg_open(W, H, "Ablation: Contribution of Each Feature Block",
                   f"Accuracy vs. macro-F1 per configuration · {os.path.basename(csv_path)}")

    for k in range(6):
        v = lo + (hi - lo) * k / 5
        gx = x_of(v)
        svg.append(f'<line x1="{gx:.1f}" y1="{T-30}" x2="{gx:.1f}" y2="{H-B+10}" '
                   f'stroke="{GRID}" stroke-width="1" opacity="{GRID_OP}"/>')
        svg.append(f'<text x="{gx:.1f}" y="{T-40}" text-anchor="middle" class="tick">{v:.2f}</text>')

    bx = x_of(baseline)
    svg.append(f'<line x1="{bx:.1f}" y1="{T-30}" x2="{bx:.1f}" y2="{H-B+10}" '
               f'stroke="#34d399" stroke-width="2" stroke-dasharray="7,6" opacity="0.55"/>')
    svg.append(f'<text x="{bx+8:.1f}" y="{H-B+30}" class="nvalue" fill="#34d399">'
               f'full-framework baseline</text>')

    for i, r in enumerate(rows):
        cy = T + row_h * i + row_h / 2
        acc, f1 = float(r["accuracy"]), float(r["f1_macro"])
        delta = acc - baseline
        xa, xf = x_of(acc), x_of(f1)

        svg.append(f'<line x1="{L-30}" y1="{cy+row_h/2-4:.1f}" x2="{W-R_+40}" '
                   f'y2="{cy+row_h/2-4:.1f}" stroke="{GRID}" stroke-width="1" opacity="{GRID_OP*0.6:.3f}"/>')

        label = r["configuration"]
        if len(label) > 46:
            label = label[:44] + "…"
        svg.append(f'<text x="{L-56}" y="{cy-4:.1f}" text-anchor="end" class="nlabel">{esc(label)}</text>')

        if i == 0:
            tag, tcol = "reference", "#34d399"
        else:
            tag = f"{delta:+.3f} vs. baseline"
            tcol = "#34d399" if delta > 0 else ("#f87171" if delta < 0 else INK)
        svg.append(f'<text x="{L-56}" y="{cy+16:.1f}" text-anchor="end" class="nvalue" '
                   f'fill="{tcol}">{esc(tag)}</text>')

        svg.append(f'<line x1="{min(xa,xf):.1f}" y1="{cy:.1f}" x2="{max(xa,xf):.1f}" y2="{cy:.1f}" '
                   f'stroke="{INK}" stroke-width="4" opacity="0.30" stroke-linecap="round"/>')
        svg.append(f'<circle cx="{xf:.1f}" cy="{cy:.1f}" r="11" fill="#a78bfa" '
                   f'stroke="{BG_DARK}" stroke-width="2.5"/>')
        svg.append(f'<circle cx="{xa:.1f}" cy="{cy:.1f}" r="11" fill="#fbbf24" '
                   f'stroke="{BG_DARK}" stroke-width="2.5"/>')
        svg.append(f'<text x="{xa:.1f}" y="{cy-22:.1f}" text-anchor="middle" class="nvalue">{acc:.3f}</text>')
        svg.append(f'<text x="{xf:.1f}" y="{cy+32:.1f}" text-anchor="middle" class="nvalue">{f1:.3f}</text>')

    lx, ly = L, H - 52
    svg.append(f'<circle cx="{lx}" cy="{ly}" r="9" fill="#fbbf24"/>')
    svg.append(f'<text x="{lx+16}" y="{ly+5}" class="nvalue">Accuracy</text>')
    svg.append(f'<circle cx="{lx+130}" cy="{ly}" r="9" fill="#a78bfa"/>')
    svg.append(f'<text x="{lx+146}" y="{ly+5}" class="nvalue">Macro F1</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 5. PARALLEL COORDINATES  —  features.csv
# ============================================================

PARALLEL_AXES = [
    ("circularity", "Circularity"),
    ("solidity", "Solidity"),
    ("aspect_ratio", "Aspect Ratio"),
    ("convexity", "Convexity"),
    ("boundary_contact_fraction", "Contact Fraction"),
    ("eccentricity", "Eccentricity"),
]


def chart_parallel(csv_path, basename, outdir, scale):
    rows = drop_truncated(read_csv(csv_path))
    axes = [(k, lab) for k, lab in PARALLEL_AXES if k in rows[0]]
    if not axes:
        sys.exit(f"ERROR: none of the expected morphology columns found in {csv_path}")

    records = []
    for r in rows:
        try:
            vals = [float(r[k]) for k, _ in axes]
        except (ValueError, KeyError):
            continue
        if any(v != v for v in vals):      # skip NaN rows
            continue
        records.append((r.get("defect_label", "Unknown"), vals))

    if not records:
        sys.exit("ERROR: no complete rows to plot")

    n_ax = len(axes)
    mins = [min(rec[1][i] for rec in records) for i in range(n_ax)]
    maxs = [max(rec[1][i] for rec in records) for i in range(n_ax)]

    W, H = 1600, 950
    L, R_, T, B = 140, 300, 175, 110
    plot_w, plot_h = W - L - R_, H - T - B

    def x_of(i):
        return L + plot_w * (i / max(n_ax - 1, 1))

    def y_of(i, v):
        span = (maxs[i] - mins[i]) or 1.0
        return T + plot_h * (1 - (v - mins[i]) / span)

    classes = sorted({c for c, _ in records})
    svg = svg_open(W, H, "Physics & Morphology Signature per Defect Class",
                   f"{len(records)} particles across {len(classes)} classes "
                   f"· {os.path.basename(csv_path)}")

    for i, (_, lab) in enumerate(axes):
        gx = x_of(i)
        svg.append(f'<line x1="{gx:.1f}" y1="{T}" x2="{gx:.1f}" y2="{T+plot_h}" '
                   f'stroke="{GRID}" stroke-width="1.5" opacity="{GRID_OP*2:.3f}"/>')
        svg.append(f'<text x="{gx:.1f}" y="{T-22}" text-anchor="middle" class="colhead" '
                   f'font-size="13">{esc(lab)}</text>')
        svg.append(f'<text x="{gx:.1f}" y="{T-6}" text-anchor="middle" class="tick" '
                   f'font-size="11">{maxs[i]:.2f}</text>')
        svg.append(f'<text x="{gx:.1f}" y="{T+plot_h+18}" text-anchor="middle" class="tick" '
                   f'font-size="11">{mins[i]:.2f}</text>')

    # Individual particles as a faint "cloud" — they show spread and outliers
    # but at 186 overlapping lines they cannot carry the message on their own.
    for cls, vals in records:
        idx = classes.index(cls)
        pts = [(x_of(i), y_of(i, v)) for i, v in enumerate(vals)]
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        svg.append(f'<path d="{d}" fill="none" stroke="{color_for(cls, idx)}" '
                   f'stroke-width="1.4" opacity="0.10"/>')

    # Per-class MEDIAN path drawn bold on top: this is the actual "signature"
    # the figure is about. Median (not mean) so a single mis-segmented outlier
    # cannot drag a class profile away from where its particles really sit.
    def median(xs):
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    for idx, cls in enumerate(classes):
        members = [vals for c, vals in records if c == cls]
        med = [median([m[i] for m in members]) for i in range(n_ax)]
        pts = [(x_of(i), y_of(i, v)) for i, v in enumerate(med)]
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        svg.append(f'<path d="{d}" fill="none" stroke="{BG_DARK}" '
                   f'stroke-width="7" opacity="0.85" stroke-linejoin="round"/>')
        svg.append(f'<path d="{d}" fill="none" stroke="{color_for(cls, idx)}" '
                   f'stroke-width="3.6" opacity="0.95" stroke-linejoin="round"/>')
        for (px, py) in pts:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" '
                       f'fill="{color_for(cls, idx)}" stroke="{BG_DARK}" stroke-width="2"/>')

    ly = T + 6
    for idx, cls in enumerate(classes):
        count = sum(1 for c, _ in records if c == cls)
        svg.append(f'<rect x="{W-R_+40}" y="{ly-11}" width="26" height="13" rx="4" '
                   f'fill="{color_for(cls, idx)}"/>')
        svg.append(f'<text x="{W-R_+76}" y="{ly}" class="nvalue" font-size="13">'
                   f'{esc(pretty(cls))} ({count})</text>')
        ly += 30

    svg.append(f'<text x="{W/2}" y="{H-30}" text-anchor="middle" class="sub" font-size="12.5">'
               'Bold line = per-class median signature; faint lines = individual particles. '
               'Each axis independently min–max scaled.</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 6. CONTOUR — DINOv2 embedding space (prototypes.joblib)
# ============================================================

def chart_contour(joblib_path, basename, outdir, scale, perplexity: float = 30.0):
    """
    t-SNE projection of the frozen DINOv2 embeddings, with a per-class
    kernel-density contour drawn over it.

    This is the direct visual test of the project's central numerical
    result — a dozen classifiers of very different families all landing
    within ~1 accuracy point of each other. If the class densities
    overlap heavily here, the shared ceiling is a property of the
    REPRESENTATION, and no change of classifier can fix it. Separated
    densities would mean the opposite. Either way the figure answers a
    question the accuracy table can only hint at.

    Contour paths come from matplotlib's marching-squares implementation
    rather than a hand-rolled one (correct topology, handles saddle
    points), and are then re-emitted as SVG in this module's style.
    """
    import joblib
    from scipy.stats import gaussian_kde
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = joblib.load(joblib_path)
    emb = np.stack([r["dinov2_embedding"] for r in records])
    labels = np.array([r["class_label"] for r in records])

    print(f"  t-SNE on {emb.shape[0]}x{emb.shape[1]} embeddings...", flush=True)
    xy = TSNE(n_components=2, perplexity=min(perplexity, (len(emb) - 1) / 3),
              init="pca", random_state=42).fit_transform(emb)

    W, H = 1500, 1150
    L, R_, T, B = 90, 330, 150, 90
    pw, ph = W - L - R_, H - T - B

    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    padx, pady = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady

    def sx(v):
        return L + pw * (v - x0) / (x1 - x0)

    def sy(v):
        return T + ph * (1 - (v - y0) / (y1 - y0))

    classes = sorted(set(labels))
    svg = svg_open(W, H, "DINOv2 Embedding Space by Defect Class",
                   f"t-SNE projection of {len(emb)} frozen embeddings with per-class "
                   f"kernel-density contours")

    # faint individual points first, contours drawn over them
    for i, (px, py) in enumerate(xy):
        c = color_for(labels[i], classes.index(labels[i]))
        svg.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="2.6" '
                   f'fill="{c}" opacity="0.30"/>')

    grid_n = 120
    gx = np.linspace(x0, x1, grid_n)
    gy = np.linspace(y0, y1, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.vstack([GX.ravel(), GY.ravel()])

    fig, ax = plt.subplots()

    for idx, cls in enumerate(classes):
        pts = xy[labels == cls]
        if len(pts) < 5:
            continue
        try:
            kde = gaussian_kde(pts.T)
        except np.linalg.LinAlgError:
            # degenerate (collinear) class cloud — no density to draw
            continue
        Z = kde(grid_pts).reshape(GX.shape)

        # ONE level per class, not several. Two levels x 8 classes puts
        # 16 closed curves on one panel, which obscures the very overlap
        # the figure exists to show. A single filled dense-core region
        # per class makes the overlap structure read immediately.
        cs = ax.contour(GX, GY, Z, levels=[Z.max() * 0.45])

        col = color_for(cls, idx)
        for segs in cs.allsegs:
            for seg in segs:
                if len(seg) < 3:
                    continue
                d = "M " + " L ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in seg) + " Z"
                svg.append(f'<path d="{d}" fill="{col}" fill-opacity="0.16" '
                           f'stroke="{col}" stroke-width="3" opacity="0.9"/>')

    plt.close(fig)

    ly = T + 6
    for idx, cls in enumerate(classes):
        n = int(np.sum(labels == cls))
        svg.append(f'<rect x="{W-R_+50}" y="{ly-11}" width="26" height="13" rx="4" '
                   f'fill="{color_for(cls, idx)}"/>')
        svg.append(f'<text x="{W-R_+86}" y="{ly}" class="nvalue" font-size="13">'
                   f'{esc(pretty(cls))} ({n})</text>')
        ly += 30

    svg.append(f'<text x="{W/2}" y="{H-28}" text-anchor="middle" class="sub" font-size="12.5">'
               'Shaded region = each class&#8217;s dense core (45% of peak density). Overlapping '
               'cores mark classes the frozen representation does not separate.</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 7. BEESWARM — reusability score per class
# ============================================================

def chart_beeswarm(csv_path, basename, outdir, scale,
                   value_col: str = "reusability_score",
                   group_col: str = "defect_label"):
    """
    Every particle as one dot, positioned by its score and nudged
    vertically to avoid overlap. Unlike a bar of class means, this shows
    the full within-class spread — which for this data is wide enough
    that the means alone are actively misleading.
    """
    rows = drop_truncated(read_csv(csv_path))

    data = defaultdict(list)
    for r in rows:
        raw = r.get(value_col, "")
        if raw in ("", None):
            continue
        data[r.get(group_col, "Unknown")].append(float(raw))

    if not data:
        sys.exit(f"ERROR: no usable '{value_col}' values in {csv_path}")

    groups = sorted(data, key=lambda g: -np.median(data[g]))
    all_vals = [v for vs in data.values() for v in vs]
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.05, hi + span * 0.05

    W = 1500
    row_h = 108
    L, R_, T, B = 300, 120, 165, 120
    H = T + row_h * len(groups) + B
    pw = W - L - R_

    def sx(v):
        return L + pw * (v - lo) / (hi - lo)

    svg = svg_open(W, H, "Reusability Score Distribution by Defect Class",
                   f"{len(all_vals)} particles, one dot each · {os.path.basename(csv_path)}")

    for k in range(6):
        v = lo + (hi - lo) * k / 5
        gx = sx(v)
        svg.append(f'<line x1="{gx:.1f}" y1="{T-30}" x2="{gx:.1f}" y2="{H-B+10}" '
                   f'stroke="{GRID}" stroke-width="1" opacity="{GRID_OP}"/>')
        svg.append(f'<text x="{gx:.1f}" y="{T-40}" text-anchor="middle" class="tick">{v:.0f}</text>')

    rng = np.random.default_rng(42)

    for gi, g in enumerate(groups):
        cy = T + row_h * gi + row_h / 2
        vals = sorted(data[g])
        col = color_for(g, gi)

        svg.append(f'<text x="{L-30}" y="{cy-4:.1f}" text-anchor="end" class="nlabel">'
                   f'{esc(pretty(g))}</text>')
        svg.append(f'<text x="{L-30}" y="{cy+16:.1f}" text-anchor="end" class="nvalue">'
                   f'n = {len(vals)} · median {np.median(vals):.1f}</text>')

        # Vertical offset by local crowding: dots sharing an x-neighborhood
        # fan out symmetrically instead of stacking into one opaque blob.
        placed = []
        for v in vals:
            x = sx(v)
            near = sum(1 for px in placed if abs(px - x) < 9)
            placed.append(x)
            direction = 1 if near % 2 else -1
            offset = direction * (near // 2 + 1) * 6.5 if near else 0
            offset = max(min(offset, row_h / 2 - 12), -(row_h / 2 - 12))
            jitter = rng.uniform(-1.5, 1.5)
            svg.append(f'<circle cx="{x:.1f}" cy="{cy + offset + jitter:.1f}" r="5" '
                       f'fill="{col}" opacity="0.75" stroke="{BG_DARK}" stroke-width="1"/>')

        med = sx(float(np.median(vals)))
        svg.append(f'<line x1="{med:.1f}" y1="{cy-row_h/2+14:.1f}" x2="{med:.1f}" '
                   f'y2="{cy+row_h/2-14:.1f}" stroke="{INK}" stroke-width="2.5" opacity="0.8"/>')

    svg.append(f'<text x="{W/2}" y="{H-34}" text-anchor="middle" class="sub" font-size="12.5">'
               'White bar = class median. Dots fan vertically only to avoid overlap; '
               'vertical position carries no meaning.</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 8. SHAP BEESWARM — shap_explanations.csv
# ============================================================

def chart_shap(csv_path, basename, outdir, scale, model: str = "xgboost", top_n: int = 14):
    """
    Standard SHAP summary beeswarm: one row per feature, one dot per
    per-particle attribution, x = signed SHAP value (push toward or away
    from the predicted class).
    """
    rows = [r for r in read_csv(csv_path) if r.get("model", model) == model]
    if not rows:
        sys.exit(f"ERROR: no rows for model '{model}' in {csv_path}")

    by_feature = defaultdict(list)
    for r in rows:
        by_feature[r["feature"]].append(float(r["shap_value"]))

    ranked = sorted(by_feature.items(),
                    key=lambda kv: -np.mean(np.abs(kv[1])))[:top_n]

    W = 1500
    row_h = 62
    L, R_, T, B = 430, 130, 165, 110
    H = T + row_h * len(ranked) + B
    pw = W - L - R_

    all_vals = [v for _, vs in ranked for v in vs]
    lim = max(abs(min(all_vals)), abs(max(all_vals))) * 1.05 or 1.0

    def sx(v):
        return L + pw * (v + lim) / (2 * lim)

    svg = svg_open(W, H, "SHAP Feature Attribution",
                   f"model: {pretty(model)} · top {len(ranked)} features by mean |SHAP| "
                   f"· {os.path.basename(csv_path)}")

    zero = sx(0.0)
    svg.append(f'<line x1="{zero:.1f}" y1="{T-30}" x2="{zero:.1f}" y2="{H-B+10}" '
               f'stroke="{INK}" stroke-width="2" opacity="0.35"/>')
    for k in range(5):
        v = -lim + 2 * lim * k / 4
        svg.append(f'<text x="{sx(v):.1f}" y="{T-40}" text-anchor="middle" class="tick">'
                   f'{v:+.2f}</text>')

    rng = np.random.default_rng(42)

    for fi, (feature, vals) in enumerate(ranked):
        cy = T + row_h * fi + row_h / 2
        col = FALLBACK_CYCLE[fi % len(FALLBACK_CYCLE)]

        label = feature if len(feature) <= 34 else feature[:32] + "…"
        svg.append(f'<text x="{L-24}" y="{cy-2:.1f}" text-anchor="end" class="nlabel" '
                   f'font-size="13">{esc(label)}</text>')
        svg.append(f'<text x="{L-24}" y="{cy+15:.1f}" text-anchor="end" class="nvalue" '
                   f'font-size="11">mean |SHAP| {np.mean(np.abs(vals)):.3f}</text>')

        placed = []
        for v in sorted(vals):
            x = sx(v)
            near = sum(1 for px in placed if abs(px - x) < 7)
            placed.append(x)
            direction = 1 if near % 2 else -1
            off = direction * (near // 2 + 1) * 4.5 if near else 0
            off = max(min(off, row_h / 2 - 8), -(row_h / 2 - 8))
            svg.append(f'<circle cx="{x:.1f}" cy="{cy + off + rng.uniform(-1,1):.1f}" '
                       f'r="4" fill="{col}" opacity="0.72"/>')

    svg.append(f'<text x="{W/2}" y="{H-32}" text-anchor="middle" class="sub" font-size="12.5">'
               'Right of the centre line = the feature pushed the prediction toward its class; '
               'left = away. Spread shows how consistently it acts.</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# 9. CIRCULAR TREEMAP — defect origin hierarchy
# ============================================================

# The manuscript's Section 2.1 promises a taxonomy "categorized by
# origin", but config.DEFECT_TAXONOMY is a flat list — the grouping
# exists only in prose. This is that grouping made explicit, by the
# physical mechanism that produces each defect.
DEFECT_ORIGIN = {
    "Atomization": ["Perfect_Spherical", "Oversized_Spherical", "Molten_Satellite"],
    "Thermal / Bonding": ["Bonded_Particle", "Agglomerate"],
    "Mechanical": ["Deformed_Particle", "Elongated_Particle", "Clip_Cap"],
}
ORIGIN_COLOR = {"Atomization": "#3ddc97", "Thermal / Bonding": "#ff6ec7", "Mechanical": "#5ec8ff"}


def chart_treemap(csv_path, basename, outdir, scale, group_col: str = "defect_label"):
    """
    Two-level circular treemap: outer circle per origin mechanism,
    inner circle per defect class, area proportional to particle count.

    Areas (not radii) carry the counts, so r scales with sqrt(n) — a
    class with 4x the particles gets 4x the area, not 4x the radius,
    which would overstate it by a factor of four.
    """
    rows = drop_truncated(read_csv(csv_path))

    counts = defaultdict(int)
    for r in rows:
        counts[r.get(group_col, "Unknown")] += 1

    class_to_origin = {c: o for o, cs in DEFECT_ORIGIN.items() for c in cs}
    origins = defaultdict(dict)
    for cls, n in counts.items():
        origins[class_to_origin.get(cls, "Other")][cls] = n

    origin_totals = {o: sum(cs.values()) for o, cs in origins.items()}
    total = sum(origin_totals.values())

    W, H = 1500, 1180
    CX, CY, R_OUTER = W / 2, 640, 400

    svg = svg_open(W, H, "Defect Population by Origin Mechanism",
                   f"{total} classified particles · area proportional to count "
                   f"· {os.path.basename(csv_path)}")

    ordered = sorted(origin_totals, key=lambda o: -origin_totals[o])
    n_groups = len(ordered)

    # Parent circles sized by sqrt(share), then placed on a ring whose
    # radius leaves room for the largest of them.
    parent_r = {o: R_OUTER * 0.52 * np.sqrt(origin_totals[o] / total) for o in ordered}
    ring_r = R_OUTER - max(parent_r.values()) - 10

    for gi, origin in enumerate(ordered):
        angle = -np.pi / 2 + 2 * np.pi * gi / n_groups
        px, py = (CX + ring_r * np.cos(angle), CY + ring_r * np.sin(angle)) if n_groups > 1 else (CX, CY)
        pr = parent_r[origin]
        col = ORIGIN_COLOR.get(origin, FALLBACK_CYCLE[gi])

        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{pr:.1f}" fill="{col}" '
                   f'opacity="0.16" stroke="{col}" stroke-width="2.5"/>')

        children = sorted(origins[origin].items(), key=lambda kv: -kv[1])
        child_total = sum(n for _, n in children)
        n_child = len(children)

        for ci, (cls, n) in enumerate(children):
            cr = pr * 0.62 * np.sqrt(n / child_total)
            if n_child == 1:
                cx_, cy_ = px, py
            else:
                a = -np.pi / 2 + 2 * np.pi * ci / n_child
                orbit = pr - cr - 6
                cx_, cy_ = px + orbit * np.cos(a), py + orbit * np.sin(a)

            ccol = color_for(cls, ci)
            svg.append(f'<circle cx="{cx_:.1f}" cy="{cy_:.1f}" r="{cr:.1f}" '
                       f'fill="{ccol}" opacity="0.88"/>')
            svg.append(f'<text x="{cx_:.1f}" y="{cy_:.1f}" text-anchor="middle" '
                       f'class="nvalue" font-size="15" fill="{BG_DARK}" '
                       f'font-weight="700">{n}</text>')

        ly = py - pr - 16
        svg.append(f'<text x="{px:.1f}" y="{ly:.1f}" text-anchor="middle" class="nlabel" '
                   f'font-size="17" fill="{col}">{esc(origin)}</text>')
        svg.append(f'<text x="{px:.1f}" y="{ly+19:.1f}" text-anchor="middle" class="nvalue">'
                   f'{origin_totals[origin]} particles · {origin_totals[origin]/total:.0%}</text>')

    ly = H - 190
    for cls in sorted(counts, key=lambda c: -counts[c]):
        pass
    lx = 80
    for idx, cls in enumerate(sorted(counts, key=lambda c: -counts[c])):
        col = color_for(cls, idx)
        svg.append(f'<rect x="{lx}" y="{H-96}" width="22" height="12" rx="4" fill="{col}"/>')
        svg.append(f'<text x="{lx+28}" y="{H-86}" class="nvalue" font-size="12">'
                   f'{esc(pretty(cls))}</text>')
        lx += 175
        if lx > W - 180:
            lx = 80

    svg.append(f'<text x="{W/2}" y="{H-40}" text-anchor="middle" class="sub" font-size="12.5">'
               'Outer ring = origin mechanism (Section 2.1 taxonomy); inner circles = defect '
               'classes. Radius scales as sqrt(count) so AREA encodes the count.</text>')
    svg.append("</svg>")
    write_out(svg, basename, outdir, W, H, scale)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate publication figures from ParticleVision CSV outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("chart", choices=["sankey", "chord", "bump", "dumbbell", "parallel",
                                      "contour", "beeswarm", "shap", "treemap", "all"])
    ap.add_argument("--csv", help="input CSV (chord / bump / dumbbell / parallel)")
    ap.add_argument("--features", help="features.csv (sankey)")
    ap.add_argument("--reusability", help="features_with_reusability.csv (sankey)")
    ap.add_argument("--dir", help="output/ directory — used by `all` to find every CSV")
    ap.add_argument("--out", help="output basename (default: fig_<chart>)")
    ap.add_argument("--outdir", default="paper_figures", help="output directory")
    ap.add_argument("--scale", type=float, default=2.0, help="PNG scale factor (default 2)")
    ap.add_argument("--theme", choices=["light", "dark"], default="light",
                    help="light = white ground for journal print (default)")
    args = ap.parse_args()
    set_theme(args.theme)

    if args.chart == "all":
        if not args.dir:
            sys.exit("ERROR: `all` needs --dir pointing at your output/ directory")
        d = args.dir
        jobs = [
            ("sankey",   lambda: chart_sankey(os.path.join(d, "features.csv"),
                                              os.path.join(d, "features_with_reusability.csv"),
                                              "fig_sankey", args.outdir, args.scale)),
            ("chord",    lambda: chart_chord(os.path.join(d, "classifier", "confusion_matrix.csv"),
                                             "fig_chord", args.outdir, args.scale)),
            ("bump",     lambda: chart_bump(os.path.join(d, "classifier", "model_comparison.csv"),
                                            "fig_bump", args.outdir, args.scale)),
            ("dumbbell", lambda: chart_dumbbell(os.path.join(d, "classifier", "ablation.csv"),
                                                "fig_dumbbell", args.outdir, args.scale)),
            ("parallel", lambda: chart_parallel(os.path.join(d, "features.csv"),
                                                "fig_parallel", args.outdir, args.scale)),
            ("contour",  lambda: chart_contour(os.path.join(d, "prototypes.joblib"),
                                               "fig_contour", args.outdir, args.scale)),
            ("beeswarm", lambda: chart_beeswarm(os.path.join(d, "features_with_reusability.csv"),
                                                "fig_beeswarm", args.outdir, args.scale)),
            ("shap",     lambda: chart_shap(os.path.join(d, "shap_explanations.csv"),
                                            "fig_shap", args.outdir, args.scale)),
            ("treemap",  lambda: chart_treemap(os.path.join(d, "features.csv"),
                                               "fig_treemap", args.outdir, args.scale)),
        ]
        for name, fn in jobs:
            print(f"[{name}]")
            try:
                fn()
            except SystemExit as e:
                print(f"  skipped: {e}")
        return

    out = args.out or f"fig_{args.chart}"

    if args.chart == "sankey":
        if not (args.features and args.reusability):
            sys.exit("ERROR: sankey needs --features features.csv --reusability features_with_reusability.csv")
        chart_sankey(args.features, args.reusability, out, args.outdir, args.scale)
    else:
        if not args.csv:
            sys.exit(f"ERROR: {args.chart} needs --csv <path>")
        {"chord": chart_chord, "bump": chart_bump,
         "dumbbell": chart_dumbbell, "parallel": chart_parallel,
         "contour": chart_contour, "beeswarm": chart_beeswarm,
         "shap": chart_shap, "treemap": chart_treemap}[args.chart](
            args.csv, out, args.outdir, args.scale)


if __name__ == "__main__":
    main()
