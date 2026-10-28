"""
ParticleVision V4
=================

Reusability Score Engine

    Reusability = 100 - alpha * D_severity - beta * delta_PSD - gamma * O_surface

This is a *self-derived* composite index (not a published/validated
standard) meant to give a single, defensible 0-100 number for "how
reusable is this powder lot for another AM build", built entirely out
of quantities this pipeline already measures. It is designed to be
transparent and re-derivable — every term below is explained, and all
three weights (alpha, beta, gamma) live in `config.REUSABILITY` so you
can run a sensitivity analysis, which a reviewer will expect if you
present a self-derived index in a paper.

Terms
-----
D_severity (0-100, per particle)
    severity_weight[predicted_class] * classification_confidence.
    severity_weight is a relative-harm ranking of the 10 defect
    classes (config.DEFECT_SEVERITY) reflecting the AM literature's
    general consensus on what hurts flowability/packing/melt-pool
    stability most (e.g. large agglomerates and splats are worse than
    a single oversized-but-round particle). *You should recalibrate
    these weights against your own flowability (Hall/Carney flow) or
    porosity data before treating the resulting score as quantitative*
    — see the docstring in config.py.

delta_PSD (0-100, one value per batch, shared by all particles in it)
    Normalised deviation of the *measured* particle-size distribution
    (D10/D50/D90 of equivalent diameter) from a *reference* PSD you
    supply (e.g. the virgin powder's certified spec window). This
    captures "spec creep" — a lot dominated by fines or coarse
    particles is less reusable even if every individual particle looks
    geometrically fine. If no reference PSD is supplied, this term is
    0 for every particle (i.e. the score falls back to a purely
    per-particle geometric assessment) and a warning is attached to
    the batch summary.

O_surface (0-100, per particle)
    Population-percentile-ranked composite of roughness_index,
    porosity_index (hole content), and boundary_waviness. Using
    percentile rank (rather than a fixed physical threshold) makes
    this term robust across different SEM magnifications/pixel scales
    — it answers "how rough is this particle relative to the rest of
    THIS batch", which is what matters for a reuse decision on a
    specific powder lot.

The final score is clipped to [0, 100]. A batch-level summary
(mean, distribution, class-wise breakdown) is also produced so it can
be dropped straight into a results table/figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from config import REUSABILITY, CLASSIFICATION
except Exception:  # pragma: no cover
    REUSABILITY = None
    CLASSIFICATION = None


DEFAULT_SEVERITY = {
    "Perfect_Spherical": 0,
    "Oversized_Spherical": 15,
    "Elongated_Particle": 35,
    "Clip_Cap": 55,
    "Molten_Satellite": 35,
    "Bonded_Particle": 55,
    "Agglomerate": 80,
    "Deformed_Particle": 40,
}


def _percentile_rank_0_100(values):
    """
    Rank-based 0-100 scaling, robust to outliers and unknown units.
    NaNs are assigned the batch mean rank (50) so they don't distort
    the rest of the distribution.
    """

    values = np.asarray(values, dtype=float)

    valid = ~np.isnan(values)

    ranks = np.full(values.shape, 50.0)

    if valid.sum() >= 2:

        ranks[valid] = (
            (rankdata(values[valid]) - 1)
            / (valid.sum() - 1)
            * 100.0
        )

    elif valid.sum() == 1:

        ranks[valid] = 50.0

    return ranks


class ReusabilityScorer:

    def __init__(
        self,
        alpha=None,
        beta=None,
        gamma=None,
        severity=None,
        reference_d10_um=None,
        reference_d50_um=None,
        reference_d90_um=None,
    ):

        cfg = REUSABILITY

        self.alpha = alpha if alpha is not None else (cfg.alpha if cfg else 0.55)
        self.beta = beta if beta is not None else (cfg.beta if cfg else 0.25)
        self.gamma = gamma if gamma is not None else (cfg.gamma if cfg else 0.20)

        self.severity = dict(
            severity
            or (CLASSIFICATION.severity if CLASSIFICATION else DEFAULT_SEVERITY)
        )

        self.reference_d10_um = (
            reference_d10_um
            if reference_d10_um is not None
            else (cfg.reference_d10_um if cfg else None)
        )
        self.reference_d50_um = (
            reference_d50_um
            if reference_d50_um is not None
            else (cfg.reference_d50_um if cfg else None)
        )
        self.reference_d90_um = (
            reference_d90_um
            if reference_d90_um is not None
            else (cfg.reference_d90_um if cfg else None)
        )

    # --------------------------------------------------------

    def _delta_psd(self, diameters_um):

        if (
            self.reference_d10_um is None
            or self.reference_d50_um is None
            or self.reference_d90_um is None
            or diameters_um is None
            or len(diameters_um) < 5
        ):
            return 0.0, False

        d10 = float(np.percentile(diameters_um, 10))
        d50 = float(np.percentile(diameters_um, 50))
        d90 = float(np.percentile(diameters_um, 90))

        deviations = []

        for measured, reference in (
            (d10, self.reference_d10_um),
            (d50, self.reference_d50_um),
            (d90, self.reference_d90_um),
        ):
            if reference and reference > 0:
                deviations.append(abs(measured - reference) / reference)

        if not deviations:
            return 0.0, False

        delta_psd = float(np.mean(deviations)) * 100.0

        return min(delta_psd, 100.0), True

    # --------------------------------------------------------

    def score_dataframe(self, particle_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score an entire particle table (as produced by
        `ParticleVisionPipeline.dataframe` / `main.py`). Adds columns:

            D_severity, O_surface, delta_PSD, reusability_score

        Batch grouping is by `image_name` if present (i.e. delta_PSD
        and the O_surface percentile ranks are computed per-image),
        otherwise over the whole table.
        """

        df = particle_df.copy()

        if "image_name" in df.columns:
            groups = df.groupby("image_name", sort=False)
        else:
            df["__batch__"] = "all"
            groups = df.groupby("__batch__", sort=False)

        out_frames = []

        for _, group in groups:

            group = group.copy()

            # --- D_severity -------------------------------------
            confidence = group.get("confidence", pd.Series(1.0, index=group.index))
            label = group.get("defect_label", pd.Series("Unclassified", index=group.index))

            severity_weight = label.map(self.severity).fillna(50.0)

            group["D_severity"] = severity_weight * confidence

            # --- O_surface ---------------------------------------
            components = []

            for col in ("roughness_index", "porosity_index", "boundary_waviness"):
                if col in group.columns:
                    components.append(_percentile_rank_0_100(group[col].values))

            if components:
                group["O_surface"] = np.mean(components, axis=0)
            else:
                group["O_surface"] = 50.0

            # --- delta_PSD (batch-level, one value shared by group)
            if "equivalent_diameter" in group.columns:

                diam_px = group["equivalent_diameter"].values

                if "microns_per_pixel" in group.columns:
                    mpp = group["microns_per_pixel"].dropna()
                    mpp = float(mpp.iloc[0]) if len(mpp) else None
                else:
                    mpp = None

                diam_um = diam_px * mpp if mpp else None

                delta_psd, used = self._delta_psd(diam_um)

            else:
                delta_psd, used = 0.0, False

            group["delta_PSD"] = delta_psd
            group["_psd_reference_used"] = used

            # --- final score --------------------------------------
            score = (
                100.0
                - self.alpha * group["D_severity"]
                - self.beta * group["delta_PSD"]
                - self.gamma * group["O_surface"]
            )

            group["reusability_score"] = score.clip(lower=0.0, upper=100.0)

            # Border-truncated particles (see
            # segmentation.estimate_border_truncation /
            # config.BORDER_TRUNCATED_LABEL) have no real defect
            # classification, so D_severity=0 for them is an artifact
            # of confidence=0, not a genuine "perfect" reading — giving
            # them a reusability_score would silently pull the batch
            # mean toward 100. NaN them out here so every downstream
            # aggregate (.mean()/.median()/.std(), all pandas-default
            # NaN-skipping) excludes them automatically without special
            # casing at every call site.
            if "border_truncated" in group.columns:
                truncated_mask = group["border_truncated"].astype(bool)
                group.loc[truncated_mask, ["D_severity", "O_surface", "reusability_score"]] = np.nan

            out_frames.append(group)

        result = pd.concat(out_frames, axis=0).sort_index()

        if "__batch__" in result.columns:
            result = result.drop(columns="__batch__")

        return result

    # --------------------------------------------------------

    def batch_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        One row per image (or one row total if no image_name column):
        mean/median/std reusability score, and defect-class fractions.
        """

        if "image_name" not in scored_df.columns:
            scored_df = scored_df.copy()
            scored_df["image_name"] = "all"

        rows = []

        for image_name, group in scored_df.groupby("image_name", sort=False):

            if "border_truncated" in group.columns:
                truncated_mask = group["border_truncated"].astype(bool)
            else:
                truncated_mask = pd.Series(False, index=group.index)

            classified = group.loc[~truncated_mask]

            row = {
                "image_name": image_name,
                # Honest, auditable count split — see
                # segmentation.estimate_border_truncation. n_particles
                # is every mask this image produced; n_classified is
                # what composition/reusability stats below are actually
                # computed over.
                "n_particles": len(group),
                "n_classified": len(classified),
                "n_border_truncated": int(truncated_mask.sum()),
                "reusability_mean": group["reusability_score"].mean(),
                "reusability_median": group["reusability_score"].median(),
                "reusability_std": group["reusability_score"].std(),
                "delta_PSD": group["delta_PSD"].iloc[0] if len(group) else np.nan,
                "psd_reference_used": bool(group["_psd_reference_used"].iloc[0])
                if len(group)
                else False,
            }

            # Composition fractions are computed over CLASSIFIED
            # particles only — a border-truncated particle isn't a
            # defect type, so it must not dilute the real class
            # proportions the reusability narrative is built on.
            if "defect_label" in classified.columns and len(classified):
                fractions = (
                    classified["defect_label"].value_counts(normalize=True) * 100.0
                )
                for label, pct in fractions.items():
                    row[f"pct_{label}"] = pct

            rows.append(row)

        return pd.DataFrame(rows)


# ============================================================
# FACTORY
# ============================================================

def build_reusability_scorer(**kwargs):
    return ReusabilityScorer(**kwargs)


__all__ = [
    "ReusabilityScorer",
    "build_reusability_scorer",
]
