"""
ParticleVision V4 — Fine-tuned ViT production inference
=======================================================

Wires the Section 4.8.4 model (vit_finetune.py) into the pipeline's
prediction path, so a run can be classified by the fine-tuned ViT
(cross-validated ~0.70) instead of the fused-vector classifier (~0.40).

THE CROP-CONSISTENCY REQUIREMENT
    The ViT sees pixels, so its accuracy only transfers if inference
    crops are produced the same way training crops were. Training uses
    the crops build_prototypes.py cached via crop_particle_patch();
    this module calls that identical function on production particles
    with the identical padding. Anything else silently shifts the input
    distribution and the measured accuracy would not apply.

WHAT IS PERSISTED
    vit_classifier/
        vit_state.pt        backbone + head weights
        vit_meta.joblib     class order, scalar names, scalar mean/std,
                            architecture args, crop settings

    The scalar standardization statistics are saved with the model
    because they are part of the fitted transform: recomputing them
    from production data would apply a different transform than the one
    the network was trained under.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import torch

from dinov2_features import crop_particle_patch, resize_for_model, to_model_tensor
from vit_finetune import (
    DINOv2ClassifierHead, _pick_device, _centre_square,
    SCALAR_FEATURES, SCALAR_FEATURES_V2,
)

VIT_DIRNAME = "vit_classifier"


# ============================================================
# SAVE / LOAD
# ============================================================

def save_vit_bundle(model, classes, scalar_names, scalar_mu, scalar_sd,
                    output_dir, *, arch: dict, crop_size: int = 224,
                    padding_ratio: float = 0.2, square_crop: bool = True):
    d = os.path.join(output_dir, VIT_DIRNAME)
    os.makedirs(d, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(d, "vit_state.pt"))
    joblib.dump({
        "classes": list(classes),
        "scalar_names": list(scalar_names),
        "scalar_mu": np.asarray(scalar_mu, dtype=np.float64),
        "scalar_sd": np.asarray(scalar_sd, dtype=np.float64),
        "arch": dict(arch),
        "crop_size": crop_size,
        "padding_ratio": padding_ratio,
        "square_crop": square_crop,
    }, os.path.join(d, "vit_meta.joblib"))
    return d


def load_vit_bundle(output_dir: str):
    """Accepts either the run dir or the vit_classifier dir itself."""
    d = output_dir
    if not os.path.exists(os.path.join(d, "vit_meta.joblib")):
        d = os.path.join(output_dir, VIT_DIRNAME)
    meta_path = os.path.join(d, "vit_meta.joblib")
    if not os.path.exists(meta_path):
        return None

    meta = joblib.load(meta_path)
    device = _pick_device()

    model = DINOv2ClassifierHead(
        n_classes=len(meta["classes"]),
        model_name=meta["arch"].get("model_name", "dinov2_vits14"),
        n_unfrozen_blocks=meta["arch"].get("n_unfrozen_blocks", 4),
        dropout=meta["arch"].get("dropout", 0.2),
        device=str(device),
        n_scalar=len(meta["scalar_names"]),
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(d, "vit_state.pt"), map_location=device))
    model.eval()

    meta["model"] = model
    meta["device"] = device
    return meta


# ============================================================
# FEATURE EXTRACTION FROM A LIVE Particle
# ============================================================

def _scalar_from_particle(particle, scalar_names):
    """
    Build the scalar vector for a production Particle.

    vit_finetune's SCALAR_FEATURES read prototype RECORD dicts; a live
    Particle stores the same quantities on its dataclasses, so the
    mapping is done explicitly here rather than by duck-typing — a
    silently-missing field would become a zero after standardization
    and quietly degrade predictions instead of failing.
    """
    morph = getattr(particle, "morphology", None)
    morph_vals = getattr(morph, "values", {}) if morph is not None else {}
    att = getattr(particle, "attachment", None)
    v2 = (particle.metadata or {}).get("physics_v2", {}) or {}

    lookup = {
        "size_um": particle.size_um,
        "equivalent_diameter_um": morph_vals.get("equivalent_diameter_um"),
        "is_attached": float(bool(getattr(att, "is_attached", False))) if att else 0.0,
        "contact_fraction": getattr(att, "boundary_contact_fraction", None) if att else None,
        "attached_count": float(len(getattr(att, "attached_particle_ids", []) or [])) if att else 0.0,
        "v2_neck_to_radius": v2.get("ratio_neck_to_radius"),
        "v2_pair_diameter": v2.get("ratio_pair_diameter"),
        "v2_has_neck": v2.get("ratio_has_neck"),
        "v2_granulometry_fine": v2.get("ratio_granulometry_fine"),
        "v2_fourier_elongation": v2.get("ratio_fourier_elongation"),
        "v2_fourier_roughness": v2.get("ratio_fourier_roughness"),
        "v2_rough_to_elong": v2.get("ratio_rough_to_elong"),
    }

    out = []
    for name in scalar_names:
        v = lookup.get(name)
        out.append(float(v) if v is not None and v == v else np.nan)
    return out


@torch.no_grad()
def classify_particles_vit(particles, image_rgb, bundle, batch_size: int = 32):
    """
    Predicts a class for each particle using the fine-tuned ViT and
    writes it onto particle.prediction, matching what
    classification.classify_particles produces so downstream code
    (features.csv, reusability, figures) is unchanged.

    Border-truncated particles are skipped for the same reason the
    fused-vector path skips them: a particle clipped by the frame has
    unreliable geometry, and a confident label on it would be worse
    than an explicit sentinel.
    """
    import config as _config

    model, device = bundle["model"], bundle["device"]
    classes = bundle["classes"]
    mu, sd = bundle["scalar_mu"], bundle["scalar_sd"]
    names = bundle["scalar_names"]

    target, skipped = [], []
    for p in particles:
        (skipped if (p.metadata or {}).get("border_truncated") else target).append(p)

    # Mutate the existing PredictionResult rather than replacing it,
    # matching classification.classify_particles — anything already
    # holding a reference to it keeps seeing the live value.
    for p in skipped:
        p.prediction.label = _config.BORDER_TRUNCATED_LABEL
        p.prediction.confidence = 0.0
        p.prediction.model_name = "excluded_border_truncated"

    if not target:
        return particles

    tensors, scal = [], []
    for p in target:
        crop = crop_particle_patch(
            image_rgb, p,
            padding_ratio=bundle.get("padding_ratio", 0.2),
            min_pad_px=4, mask_background=False,
        )
        if bundle.get("square_crop", True):
            crop = _centre_square(crop)
        tensors.append(to_model_tensor(resize_for_model(crop, bundle.get("crop_size", 224))))
        scal.append(_scalar_from_particle(p, names))

    x_all = torch.stack(tensors)

    z = (np.asarray(scal, dtype=np.float64) - mu) / np.where(sd > 1e-8, sd, 1.0)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    s_all = torch.tensor(z, dtype=torch.float32)

    for start in range(0, len(target), batch_size):
        xb = x_all[start:start + batch_size].to(device)
        sb = s_all[start:start + batch_size].to(device) if names else None
        probs = torch.softmax(model(xb, sb), dim=1).cpu().numpy()

        for p, row in zip(target[start:start + batch_size], probs):
            k = int(np.argmax(row))
            p.prediction.label = str(classes[k])
            p.prediction.confidence = float(row[k])
            p.prediction.model_name = "dinov2_vit_finetuned"
            p.prediction.scores = {str(c): float(v) for c, v in zip(classes, row)}

    return particles


__all__ = ["save_vit_bundle", "load_vit_bundle", "classify_particles_vit", "VIT_DIRNAME"]
