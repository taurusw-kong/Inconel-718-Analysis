#!/usr/bin/env python3

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy import stats


def load_master_particles(path):
    d = pd.read_csv(path)
    p = d[d.block == "particle"].pivot_table(
        index="entity", columns="key", values="value", aggfunc="first")
    for c in ("reusability_score", "D_severity", "size_um"):
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    return p


def load_features(path):
    d = pd.read_csv(path)
    d["entity"] = d.image_name.astype(str) + "#" + d.particle_id.astype(str)
    return d.set_index("entity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="our features_with_reusability.csv")
    ap.add_argument("--theirs", required=True, help="their master CSV")
    ap.add_argument("--label_a", default="OURS")
    ap.add_argument("--label_b", default="THEIRS")
    ap.add_argument("--out_prefix", default="implementation_comparison")
    args = ap.parse_args()

    A = load_features(args.ours)
    B = load_master_particles(args.theirs)

    A = A[A.prediction_model != "excluded_border_truncated"]
    shared = sorted(set(A.index) & set(B.dropna(subset=["reusability_score"]).index))
    A, B = A.loc[shared], B.loc[shared]
    cls = A.defect_label.values

    a_r, b_r = A.reusability_score.values, B.reusability_score.values
    a_s, b_s = A.D_severity.values, B.D_severity.values

    LA, LB = args.label_a, args.label_b

    # ---------------- per-class table ----------------
    rows = []
    for c in sorted(set(cls)):
        m = cls == c
        rows.append({
            "defect_class": c,
            "n": int(m.sum()),
            f"reuse_mean_{LA}": a_r[m].mean(),
            f"reuse_mean_{LB}": b_r[m].mean(),
            "reuse_delta": b_r[m].mean() - a_r[m].mean(),
            f"sev_median_{LA}": np.median(a_s[m]),
            f"sev_median_{LB}": np.median(b_s[m]),
            "sev_delta": np.median(b_s[m]) - np.median(a_s[m]),
        })
    t = pd.DataFrame(rows)


    t[f"rank_{LA}"] = t[f"reuse_mean_{LA}"].rank(ascending=False).astype(int)
    t[f"rank_{LB}"] = t[f"reuse_mean_{LB}"].rank(ascending=False).astype(int)
    t["rank_shift"] = t[f"rank_{LB}"] - t[f"rank_{LA}"]
    t = t.sort_values(f"rank_{LA}")

    print("=" * 92)
    print(f"PER-CLASS COMPARISON   ({LA} vs {LB})   n = {len(shared)} particles")
    print("=" * 92)
    print(t.round(2).to_string(index=False))

    # ---------------- agreement ----------------
    rho_cls, p_cls = stats.spearmanr(t[f"reuse_mean_{LA}"], t[f"reuse_mean_{LB}"])
    pear = np.corrcoef(a_r, b_r)[0, 1]
    spear = stats.spearmanr(a_r, b_r).statistic
    sev_pear = np.corrcoef(a_s, b_s)[0, 1]

    print("\n" + "=" * 92)
    print("AGREEMENT")
    print("=" * 92)
    print(f"  class-level rank correlation (Spearman) : {rho_cls:+.4f}  (p={p_cls:.3g})")
    print(f"  classes whose rank changed at all       : {(t.rank_shift != 0).sum()} of {len(t)}")
    print(f"  per-particle reusability  Pearson       : {pear:+.4f}")
    print(f"  per-particle reusability  Spearman      : {spear:+.4f}")
    print(f"  per-particle severity     Pearson       : {sev_pear:+.4f}")
    d = b_r - a_r
    print(f"  mean offset ({LB} - {LA})               : {d.mean():+.3f}  (sd {d.std():.3f})")

    # ---------------- definitional check ----------------
    print("\n" + "=" * 92)
    print("DEFINITIONAL CHECK — a defect-free class should have zero severity")
    print("=" * 92)
    for c in ("Perfect_Spherical",):
        m = cls == c
        if m.sum():
            print(f"  {c} severity  {LA}: {a_s[m].min():.2f}-{a_s[m].max():.2f}"
                  f"   {LB}: {b_s[m].min():.2f}-{b_s[m].max():.2f}")

    t.to_csv(f"{args.out_prefix}_per_class.csv", index=False)
    pd.DataFrame([{
        "class_rank_spearman": rho_cls, "ranks_changed": int((t.rank_shift != 0).sum()),
        "particle_pearson": pear, "particle_spearman": spear,
        "severity_pearson": sev_pear, "mean_offset": d.mean(), "offset_sd": d.std(),
        "n_particles": len(shared),
    }]).to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nwrote {args.out_prefix}_per_class.csv and {args.out_prefix}_summary.csv")


if __name__ == "__main__":
    main()
