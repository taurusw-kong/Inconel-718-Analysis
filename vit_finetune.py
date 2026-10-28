"""
ParticleVision V4
=================

Fine-tuned DINOv2 / ViT Classification Head  (manuscript Section 4.8.4)
----------------------------------------------------------------------

Every other model in this project classifies on a FROZEN DINOv2
embedding: the visual representation is fixed, and only the decision
boundary is learned. Those models have all converged to within about
one accuracy point of each other (gradient-boosted trees, a tabular
transformer, a kernel machine, and a cloud foundation model alike),
which is evidence that the representation — not the classifier — is
the binding constraint.

This module tests that directly by unfreezing the backbone, so the
representation itself adapts to powder-defect morphology. It is the
only model in the comparison that can change what the features *are*
rather than how they are combined.

DESIGN NOTES

Partial unfreezing, not full.
    Only the last `n_unfrozen_blocks` transformer blocks plus the head
    are trained. Full fine-tuning of a 21M-parameter ViT against ~220
    real labeled particles would overfit almost immediately; the early
    blocks hold generic low-level visual structure that is already
    appropriate for SEM imagery and does not need to move.

Group-aware everything.
    Augmented views (4 rotations x 2 flips) of one real particle share
    a source_group_id. Both the outer cross-validation AND the inner
    early-stopping split hold out whole groups, so a rotated copy of a
    training particle can never appear in any evaluation set. This
    mirrors classification.py exactly, so the resulting accuracy is
    directly comparable to every other model's.

Augmentation is applied on the fly, matching dinov2_features.py's
    `extract_augmented_embeddings` (same 4 rotations, same horizontal
    flip, same letterbox-then-resize, same ImageNet normalization), so
    the fine-tuned model sees the same views the frozen-embedding
    models were built from.

ONE DOCUMENTED DIFFERENCE FROM THE FROZEN PATH
    build_prototypes.py re-segments each labeled crop and re-crops the
    detected particle with padding before embedding it. This module
    instead loads the labeled crop file directly. Those crops are
    already tight single-particle images (they were cut by hand for
    exactly that purpose), so the difference is small — but it is a
    real difference, and it is stated here rather than buried, because
    it is the one place this model's input is not bit-identical to the
    others'.
"""

from __future__ import annotations

import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from dinov2_features import (
    load_dinov2_model, resize_for_model, to_model_tensor, DEFAULT_MODEL,
)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# DATA
# ============================================================

def _augmented_views(crop: np.ndarray):
    """
    The same 8 views dinov2_features.extract_augmented_embeddings
    produces, in the same order, so view i here means view i there.
    """
    views = []
    for angle in (0, 90, 180, 270):
        rotated = crop
        if angle == 90:
            rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(crop, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        views.append(rotated)
        views.append(cv2.flip(rotated, 1))
    return views


def _centre_square(image: np.ndarray) -> np.ndarray:
    """
    Largest centred square of the image.

    This is the framing control. The labeled crops were cut by hand,
    and crop RECTANGLE SIZE turns out to correlate with class (measured:
    a random forest reads the class from crop width/height alone at
    ~30% vs 12.5% chance, because bigger particles were naturally given
    bigger rectangles). Letterboxing already destroys absolute size, but
    aspect ratio and how tightly the particle was framed survive it.
    Taking a centred square removes the aspect-ratio channel and
    standardizes framing shape across classes, so a model trained this
    way cannot be reading the annotator's cropping habits.
    """
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


# Physical quantities the resized crop cannot express. Deliberately a
# SHORT list: each entry is information the image genuinely lost to
# preprocessing, not a general dump of the tabular features (which would
# make this a fusion model rather than a test of the visual backbone).
SCALAR_FEATURES = [
    ("size_um", lambda r: r.get("size_um")),
    ("equivalent_diameter_um", lambda r: r.get("morphology", {}).get("equivalent_diameter_um")),
    ("is_attached", lambda r: float(bool(r.get("attachment", {}).get("is_attached", False)))),
    ("contact_fraction", lambda r: r.get("attachment", {}).get("boundary_contact_fraction")),
    ("attached_count", lambda r: float(r.get("attachment", {}).get("attached_count", 0) or 0)),
]

# Physics v2 additions (physics_formulas_v2.py). Kept as a SEPARATE list
# so the ViT can be run with and without them and the difference
# attributed cleanly. The neck ratio is the one with a specific
# hypothesis behind it: Frenkel/Kuczynski two-sphere theory predicts it
# separates an early-stage neck (satellite on a host) from advanced
# coalescence (bonded pair) — the largest measured confusion in the
# frozen-scalar run.
SCALAR_FEATURES_V2 = [
    ("v2_neck_to_radius", lambda r: (r.get("physics_v2") or {}).get("ratio_neck_to_radius")),
    ("v2_pair_diameter", lambda r: (r.get("physics_v2") or {}).get("ratio_pair_diameter")),
    ("v2_has_neck", lambda r: (r.get("physics_v2") or {}).get("ratio_has_neck")),
    ("v2_granulometry_fine", lambda r: (r.get("physics_v2") or {}).get("ratio_granulometry_fine")),
    ("v2_fourier_elongation", lambda r: (r.get("physics_v2") or {}).get("ratio_fourier_elongation")),
    ("v2_fourier_roughness", lambda r: (r.get("physics_v2") or {}).get("ratio_fourier_roughness")),
    ("v2_rough_to_elong", lambda r: (r.get("physics_v2") or {}).get("ratio_rough_to_elong")),
]

# Toggled by cross_validated_vit(use_v2_scalars=...)
_ACTIVE_SCALARS = list(SCALAR_FEATURES)


def _scalar_vector(record) -> list:
    out = []
    for _name, getter in _ACTIVE_SCALARS:
        try:
            v = getter(record)
        except Exception:
            v = None
        out.append(float(v) if v is not None and v == v else np.nan)
    return out


def build_crop_tensors(records, crop_size: int = 224, verbose: bool = True,
                       square_crop: bool = False):
    """
    Loads each unique labeled crop ONCE, precomputes its 8 augmented
    view tensors, and returns arrays aligned to the prototype records.

    Returns (tensors, labels, groups, kept_index) where `kept_index`
    maps back into `records` — a record whose source image can't be
    read is dropped from all four, so nothing silently misaligns.
    """
    by_path = {}
    tensors, labels, groups, kept_index, scalars = [], [], [], [], []
    missing = set()

    for i, rec in enumerate(records):
        path = rec["source_path"]

        if path in missing:
            continue

        if path not in by_path:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                missing.add(path)
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if square_crop:
                rgb = _centre_square(rgb)
            by_path[path] = [
                to_model_tensor(resize_for_model(v, crop_size))
                for v in _augmented_views(rgb)
            ]

        views = by_path[path]
        view_idx = int(rec.get("augmentation_view", 0)) % len(views)

        tensors.append(views[view_idx])
        labels.append(rec["class_label"])
        groups.append(rec["source_group_id"])
        kept_index.append(i)
        scalars.append(_scalar_vector(rec))

    if verbose:
        print(f"[vit] {len(by_path)} unique crops -> {len(tensors)} view tensors"
              + (f"  ({len(missing)} unreadable source paths skipped)" if missing else ""))

    return (torch.stack(tensors), np.array(labels),
            np.array(groups), np.array(kept_index),
            np.asarray(scalars, dtype=np.float32))


# ============================================================
# MODEL
# ============================================================

class DINOv2ClassifierHead(nn.Module):
    """
    DINOv2 backbone (partially unfrozen) + linear classification head,
    optionally with physical scalars concatenated before the head.

    WHY THE SCALARS ARE NEEDED
        Crops are letterboxed to a square and resized to 224x224 before
        they reach the network, which normalizes away ABSOLUTE SIZE: a
        25 um sphere and a 40 um sphere arrive as pixel-identical
        images. Several classes in this taxonomy are separated mainly
        by calibrated size and by whether the particle is attached to a
        host — exactly the information that normalization discards.
        Measured on the frozen run, the two largest error sources were
        Bonded <-> Molten Satellite and Perfect -> Oversized Sphere,
        both size-separated pairs.

        Feeding size and attachment evidence in alongside the visual
        embedding restores that channel without giving the model
        anything it could not legitimately measure from the image plus
        its scale calibration.
    """

    def __init__(self, n_classes: int, model_name: str = DEFAULT_MODEL,
                 n_unfrozen_blocks: int = 4, dropout: float = 0.2,
                 device: str = "cpu", n_scalar: int = 0):
        super().__init__()

        self.n_scalar = n_scalar
        self.backbone, embed_dim = load_dinov2_model(model_name, device)

        # Freeze everything, then re-enable only the last N blocks and
        # the final norm. Freezing first and selectively unfreezing (as
        # opposed to the reverse) means a backbone whose block count
        # differs from what we assume fails safe — fully frozen — rather
        # than silently training more than intended.
        for param in self.backbone.parameters():
            param.requires_grad = False

        blocks = getattr(self.backbone, "blocks", [])
        for block in list(blocks)[-n_unfrozen_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + n_scalar),
            nn.Dropout(dropout),
            nn.Linear(embed_dim + n_scalar, n_classes),
        )

    def forward(self, x, scalars=None):
        features = self.backbone(x)
        if self.n_scalar and scalars is not None:
            features = torch.cat([features, scalars], dim=1)
        return self.head(features)


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def _train_one_fold(tensors, y_idx, groups, train_idx, n_classes, device, scalars=None,
                    *, model_name, n_unfrozen_blocks, lr, head_lr, weight_decay,
                    batch_size, max_epochs, patience, val_fraction,
                    random_state, verbose):

    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)

    # group-aware inner split for early stopping
    tr_groups = groups[train_idx]
    unique = rng.permutation(np.unique(tr_groups))
    n_val_target = max(int(len(train_idx) * val_fraction), n_classes)

    held, count = [], 0
    for g in unique:
        if count >= n_val_target:
            break
        held.append(g)
        count += int(np.sum(tr_groups == g))

    val_mask = np.isin(tr_groups, held)
    if val_mask.all() or not val_mask.any():
        perm = rng.permutation(len(train_idx))
        inner_val = train_idx[perm[:n_val_target]]
        inner_tr = train_idx[perm[n_val_target:]]
    else:
        inner_val = train_idx[val_mask]
        inner_tr = train_idx[~val_mask]

    # Standardize the scalars using TRAINING-FOLD statistics only. Using
    # whole-dataset mean/std would let the held-out particles influence
    # the transform applied to themselves — small, but exactly the kind
    # of leak that inflates a cross-validated score.
    n_scalar = 0
    scal_t = None
    if scalars is not None and scalars.size:
        n_scalar = scalars.shape[1]
        tr = scalars[inner_tr]
        mu = np.nanmean(tr, axis=0)
        sd = np.nanstd(tr, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        z = (np.nan_to_num(scalars, nan=np.nan) - mu) / sd
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        scal_t = torch.tensor(z, dtype=torch.float32, device=device)

    model = DINOv2ClassifierHead(
        n_classes, model_name=model_name,
        n_unfrozen_blocks=n_unfrozen_blocks, device=str(device),
        n_scalar=n_scalar,
    ).to(device)

    # Backbone gets a much smaller learning rate than the fresh head:
    # the pretrained weights are already good and only need nudging,
    # while the head starts from random init and must move fast. One
    # shared rate would either wreck the backbone or starve the head.
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()
    y_t = torch.tensor(y_idx, device=device)

    best_loss, best_state, bad = float("inf"), None, 0

    for epoch in range(max_epochs):

        model.train()
        order = rng.permutation(len(inner_tr))

        for start in range(0, len(inner_tr), batch_size):
            idx = inner_tr[order[start:start + batch_size]]
            xb = tensors[idx].to(device)
            sb = scal_t[idx] if scal_t is not None else None
            opt.zero_grad()
            loss = loss_fn(model(xb, sb), y_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(inner_val), batch_size):
                idx = inner_val[start:start + batch_size]
                xb = tensors[idx].to(device)
                sb = scal_t[idx] if scal_t is not None else None
                val_losses.append(loss_fn(model(xb, sb), y_t[idx]).item() * len(idx))
        val_loss = sum(val_losses) / max(len(inner_val), 1)

        if verbose:
            print(f"    epoch {epoch:3d}  val_loss {val_loss:.4f}", flush=True)

        if val_loss < best_loss - 1e-4:
            best_loss, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop @ epoch {epoch} (best {best_loss:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scal_t


@torch.no_grad()
def _predict(model, tensors, idx, device, batch_size, scal_t=None):
    model.eval()
    out = []
    for start in range(0, len(idx), batch_size):
        sel = idx[start:start + batch_size]
        xb = tensors[sel].to(device)
        sb = scal_t[sel] if scal_t is not None else None
        out.append(model(xb, sb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def cross_validated_vit(records, n_folds: int = 5, *,
                        model_name: str = DEFAULT_MODEL,
                        n_unfrozen_blocks: int = 4,
                        lr: float = 1e-5, head_lr: float = 1e-3,
                        weight_decay: float = 1e-4,
                        batch_size: int = 32, max_epochs: int = 30,
                        patience: int = 5, val_fraction: float = 0.15,
                        crop_size: int = 224, random_state: int = 42,
                        verbose: bool = True, square_crop: bool = False,
                        use_scalars: bool = True, use_v2_scalars: bool = False):
    """
    Group-aware cross-validated evaluation of the fine-tuned ViT.

    Returns (y_true_labels, y_pred_labels) as string class labels, so
    the caller can compute exactly the same metrics as every other
    model without knowing anything about the encoding used here.
    """
    global _ACTIVE_SCALARS
    _ACTIVE_SCALARS = list(SCALAR_FEATURES) + (list(SCALAR_FEATURES_V2) if use_v2_scalars else [])

    device = _pick_device()
    if verbose:
        print(f"[vit] device: {device}")

    tensors, labels, groups, _, scalars = build_crop_tensors(
        records, crop_size, verbose, square_crop)
    if not use_scalars:
        scalars = None
    elif verbose:
        print(f"[vit] + {scalars.shape[1]} physical scalars: "
              f"{[n for n, _ in _ACTIVE_SCALARS]}")

    encoder = LabelEncoder()
    y_idx = encoder.fit_transform(labels)
    n_classes = len(encoder.classes_)

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    predictions = np.empty_like(y_idx)

    for fold, (train_idx, test_idx) in enumerate(skf.split(tensors.numpy()[:, :1, :1, :1],
                                                           y_idx, groups), 1):
        if verbose:
            print(f"[vit] fold {fold}/{n_folds}  train={len(train_idx)} test={len(test_idx)}",
                  flush=True)

        model, scal_t = _train_one_fold(
            tensors, y_idx, groups, train_idx, n_classes, device, scalars,
            model_name=model_name, n_unfrozen_blocks=n_unfrozen_blocks,
            lr=lr, head_lr=head_lr, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience,
            val_fraction=val_fraction, random_state=random_state, verbose=verbose,
        )

        predictions[test_idx] = _predict(model, tensors, test_idx, device, batch_size, scal_t)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    return encoder.inverse_transform(y_idx), encoder.inverse_transform(predictions)


__all__ = ["cross_validated_vit", "DINOv2ClassifierHead", "build_crop_tensors"]
