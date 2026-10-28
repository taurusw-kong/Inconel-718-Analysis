
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import joblib
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vit_finetune import (
    DINOv2ClassifierHead, _pick_device, _centre_square,
    SCALAR_FEATURES, SCALAR_FEATURES_V2, _train_one_fold,
)
import vit_finetune
from dinov2_features import resize_for_model, to_model_tensor
from vit_inference import save_vit_bundle
from sklearn.preprocessing import LabelEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototypes_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--use_v2_scalars", action="store_true")
    ap.add_argument("--square_crop", action="store_true", default=True)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    records = joblib.load(os.path.join(args.prototypes_dir, "prototypes.joblib"))
    crops = joblib.load(os.path.join(args.prototypes_dir, "prototype_crops.joblib"))
    print(f"[train] {len(records)} records, {len(crops)} cached crops")

    vit_finetune._ACTIVE_SCALARS = list(SCALAR_FEATURES) + (
        list(SCALAR_FEATURES_V2) if args.use_v2_scalars else [])
    scalar_names = [n for n, _ in vit_finetune._ACTIVE_SCALARS]
    print(f"[train] scalars: {scalar_names}")

    # Build tensors from the CACHED crops (not the source files), so the
    # training distribution matches inference exactly.
    from vit_finetune import _augmented_views, _scalar_vector

    view_cache, tensors, labels, groups, scal = {}, [], [], [], []
    for rec in records:
        gid = int(rec["source_group_id"])
        crop = crops.get(gid)
        if crop is None:
            continue
        if gid not in view_cache:
            base = _centre_square(crop) if args.square_crop else crop
            view_cache[gid] = [to_model_tensor(resize_for_model(v, 224))
                               for v in _augmented_views(base)]
        views = view_cache[gid]
        tensors.append(views[int(rec.get("augmentation_view", 0)) % len(views)])
        labels.append(rec["class_label"])
        groups.append(gid)
        scal.append(_scalar_vector(rec))

    tensors = torch.stack(tensors)
    labels = np.array(labels)
    groups = np.array(groups)
    scalars = np.asarray(scal, dtype=np.float32)
    print(f"[train] tensors {tuple(tensors.shape)}  scalars {scalars.shape}")

    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    device = _pick_device()
    print(f"[train] device {device}, {len(enc.classes_)} classes")

    
    all_idx = np.arange(len(y))
    model, scal_t = _train_one_fold(
        tensors, y, groups, all_idx, len(enc.classes_), device, scalars,
        model_name="dinov2_vits14", n_unfrozen_blocks=4,
        lr=1e-5, head_lr=1e-3, weight_decay=1e-4,
        batch_size=32, max_epochs=args.max_epochs, patience=5,
        val_fraction=0.15, random_state=args.random_state, verbose=True,
    )

    mu = np.nanmean(scalars, axis=0)
    sd = np.nanstd(scalars, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    mu[~np.isfinite(mu)] = 0.0

    d = save_vit_bundle(
        model, list(enc.classes_), scalar_names, mu, sd, args.output_dir,
        arch=dict(model_name="dinov2_vits14", n_unfrozen_blocks=4, dropout=0.2),
        crop_size=224, padding_ratio=0.2, square_crop=args.square_crop,
    )
    print(f"[train] saved bundle -> {d}")
    print("TRAINVITDONE")


if __name__ == "__main__":
    main()
