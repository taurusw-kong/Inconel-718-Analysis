"""
ParticleVision V2
=================

DINOv2 Self-Supervised Feature Extraction Engine

Computes a frozen DINOv2 embedding for every segmented particle. This
is the piece named in PROJECT_STATE.md's "central gap": the paper
claims a DINOv2 self-supervised + morphology + physics fusion
architecture, and until this module existed there was no learned
representation anywhere in the pipeline — only rule-based geometry
(morphology.py, physics_descriptors.py, defect_classification.py).

This module only produces embeddings. It does NOT fuse them with
morphology/physics or train a classifier — that is the next TODO item
("feature fusion layer") and depends on this one.

Design notes
------------
- Crops come from `image_rgb` (the CLAHE-enhanced image segmentation
  actually ran on), using each particle's own `bbox`, NOT from the
  colored classification overlay — the whole point is for DINOv2 to
  see real particle texture/morphology, not the taxonomy's own paint.
- A crop is padded around the tight bounding box (`padding_ratio`) so
  the model sees a little context beyond the exact mask edge — useful
  signal for contact-dependent classes (Molten_Satellite,
  Bonded_Particle) where whether/how a particle touches its neighbor
  matters. Set `mask_background=True` to instead blank out everything
  outside the particle's own mask if you want pure per-particle
  texture with zero neighbor contamination — that discards the
  contact signal, so it's off by default.
- Crops are letterboxed to square (constant neutral-gray fill, never
  data-dependent) before resizing to `crop_size`, so a particle's
  aspect ratio isn't distorted going into a square ViT input.
- Most particle crops here are small (tens of pixels) and DINOv2 was
  pretrained on natural photos at 224px+ — expect the upsampling to
  blur fine texture on the smallest particles. This is a known
  limitation, not a bug; if it matters for the paper, it's worth
  reporting crop-size statistics alongside the embeddings.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from data_models import BoundingBox, Particle


# ============================================================
# MODEL REGISTRY
# ============================================================

# Every official torch.hub entrypoint in facebookresearch/dinov2,
# mapped to its output embedding dimensionality. "_reg" variants add
# register tokens (Darcet et al., 2023) — generally cleaner attention
# maps, drop-in compatible. Default is the smallest plain model: this
# pipeline already runs on CPU/Apple-Silicon MPS at 50-90s/image for
# segmentation alone (see PROJECT_STATE.md), so vits14 keeps the
# extra cost from becoming the new bottleneck. Swap to vitb14/vitl14
# later if the fusion classifier's accuracy needs it and the hardware
# can afford it.
DINOV2_MODELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}

DEFAULT_MODEL = "dinov2_vits14"

# ImageNet normalization statistics — what every DINOv2 checkpoint
# was pretrained with.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Neutral fill for letterbox padding / off-image overflow. Deliberately
# a fixed constant rather than each crop's own mean color, so the fill
# itself never carries information that could leak into the embedding
# (e.g. a crop's mean brightness correlating with defect severity).
_LETTERBOX_FILL = (114, 114, 114)


# ============================================================
# MODEL LOADING
# ============================================================

def load_dinov2_model(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """
    Load a frozen DINOv2 backbone via torch.hub.

    First call on a machine needs internet access (downloads weights
    into ~/.cache/torch/hub, same cache torch.hub always uses) — after
    that it's local and offline. If you're on an air-gapped machine,
    run this once elsewhere and copy that cache folder over.
    """

    if model_name not in DINOV2_MODELS:
        raise ValueError(
            f"Unknown DINOv2 model '{model_name}'. Choose one of: "
            f"{sorted(DINOV2_MODELS)}"
        )

    # torch and xgboost each bundle their own OpenMP runtime; running
    # both multi-threaded in the same process (this pipeline needs
    # DINOv2 embeddings AND an XGBoost classifier together) reliably
    # SEGFAULTS on macOS the first time xgboost actually trains/infers
    # after torch has run a real forward pass — reproduced directly,
    # and neither import order nor KMP_DUPLICATE_LIB_OK fixes it.
    # Restricting torch to a single thread avoids the conflict
    # (confirmed) — classification.py's XGBClassifier calls pair this
    # with n_jobs=1 on its side.
    torch.set_num_threads(1)

    print(
        f"[INFO] Loading DINOv2 backbone '{model_name}' on {device} "
        f"(first run downloads weights — needs internet once)."
    )

    try:

        model = torch.hub.load("facebookresearch/dinov2", model_name)

    except Exception as exc:

        # torch.hub collapses every download failure into the same
        # generic "no internet connection" RuntimeError, even when the
        # real cause is unrelated — most commonly an SSL certificate
        # verification failure, which on macOS almost always means the
        # official python.org installer's certificate-install step was
        # never run (it doesn't hook into the system trust store on its
        # own). That's a one-time local fix, not a connectivity problem,
        # so it gets its own diagnosis here instead of the misleading
        # generic message below.
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):

            raise RuntimeError(
                f"Failed to load DINOv2 model '{model_name}': SSL "
                "certificate verification failed (CERTIFICATE_VERIFY_"
                "FAILED). This is NOT a missing-internet problem — it's "
                "a one-time macOS + python.org setup step that was never "
                "run. Fix it by running the certificate installer that "
                "ships with your Python install, e.g. from Terminal:\n"
                '  open "/Applications/Python 3.14/Install Certificates.command"\n'
                "(look in /Applications for the folder matching your "
                "Python version if 3.14 isn't right), then re-run this "
                f"command. Original error: {exc}"
            ) from exc

        raise RuntimeError(
            f"Failed to load DINOv2 model '{model_name}' via torch.hub. "
            "This needs an internet connection the first time it runs "
            "(to download weights into ~/.cache/torch/hub). Check your "
            "connection and try again, or pre-download on a machine with "
            f"internet and copy that cache folder over. Original error: "
            f"{exc}"
        ) from exc

    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    model.to(device)

    print("[INFO] DINOv2 ready.")

    return model, DINOV2_MODELS[model_name]


# ============================================================
# CROPPING
# ============================================================

def compute_padded_bbox(
    bbox: BoundingBox,
    image_shape: Tuple[int, int],
    padding_ratio: float = 0.2,
    min_pad_px: int = 4,
) -> Tuple[int, int, int, int]:
    """
    Expand a particle's tight bounding box by a size-relative margin,
    clipped to the image bounds. Returns (x0, y0, x1, y1).
    """

    h_img, w_img = image_shape[:2]

    pad = max(padding_ratio * max(bbox.width, bbox.height), min_pad_px)

    x0 = int(round(bbox.x - pad))
    y0 = int(round(bbox.y - pad))
    x1 = int(round(bbox.x + bbox.width + pad))
    y1 = int(round(bbox.y + bbox.height + pad))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w_img, x1)
    y1 = min(h_img, y1)

    return x0, y0, x1, y1


def letterbox_square(image: np.ndarray, fill_value=_LETTERBOX_FILL) -> np.ndarray:
    """
    Pad a crop to a square canvas (constant fill, centered) without
    distorting its aspect ratio. A no-op if already square.
    """

    h, w = image.shape[:2]

    if h == w:
        return image

    size = max(h, w)

    canvas = np.full((size, size, 3), fill_value, dtype=np.uint8)

    y0 = (size - h) // 2
    x0 = (size - w) // 2

    canvas[y0:y0 + h, x0:x0 + w] = image

    return canvas


def crop_particle_patch(
    image_rgb: np.ndarray,
    particle: Particle,
    padding_ratio: float = 0.2,
    min_pad_px: int = 4,
    mask_background: bool = False,
) -> np.ndarray:
    """
    Crop this particle's patch out of `image_rgb` (mask + bbox, with
    consistent padding) — never from a colored overlay.
    """

    if particle.bbox is None:
        raise ValueError("Particle bbox is None.")

    x0, y0, x1, y1 = compute_padded_bbox(
        particle.bbox,
        image_rgb.shape,
        padding_ratio=padding_ratio,
        min_pad_px=min_pad_px,
    )

    crop = image_rgb[y0:y1, x0:x1].copy()

    if mask_background:

        if particle.mask is None:
            raise ValueError("Particle mask is None.")

        mask_crop = particle.mask[y0:y1, x0:x1].astype(bool)

        fill = np.array(_LETTERBOX_FILL, dtype=np.uint8)

        crop = np.where(mask_crop[..., None], crop, fill).astype(np.uint8)

    return crop


def resize_for_model(crop: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Letterbox to square, then resize to (crop_size, crop_size).
    Upsampling (the common case — most particle crops are far smaller
    than 224px) uses cubic interpolation; downsampling (rare — only
    unusually large particles) uses area interpolation to avoid
    aliasing.
    """

    squared = letterbox_square(crop)

    interpolation = (
        cv2.INTER_CUBIC if squared.shape[0] < crop_size else cv2.INTER_AREA
    )

    return cv2.resize(
        squared,
        (crop_size, crop_size),
        interpolation=interpolation,
    )


def to_model_tensor(image_uint8: np.ndarray) -> torch.Tensor:
    """
    uint8 HWC RGB crop -> normalized float32 CHW tensor, using the
    ImageNet statistics DINOv2 was pretrained with.
    """

    x = image_uint8.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))

    return torch.from_numpy(x)


# ============================================================
# EXTRACTOR
# ============================================================

class DINOv2FeatureExtractor:
    """
    Extract a frozen DINOv2 embedding per particle crop.

    Mirrors the extract()-per-particle shape of MorphologyExtractor /
    PhysicsDescriptorExtractor, but (unlike those) needs the source
    `image_rgb` alongside each particle, since a Particle doesn't
    carry a reference to the image it came from.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        crop_size: int = 224,
        padding_ratio: float = 0.2,
        min_pad_px: int = 4,
        mask_background: bool = False,
        batch_size: int = 16,
    ):

        if crop_size % 14 != 0:
            raise ValueError(
                f"crop_size must be a multiple of DINOv2's patch size "
                f"(14) — got {crop_size}."
            )

        self.model_name = model_name
        self.device = device
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self.min_pad_px = min_pad_px
        self.mask_background = mask_background
        self.batch_size = batch_size

        self.model, self.embedding_dim = load_dinov2_model(model_name, device)

    # --------------------------------------------------------

    def _prepare_tensor(self, image_rgb: np.ndarray, particle: Particle) -> torch.Tensor:

        crop = crop_particle_patch(
            image_rgb,
            particle,
            padding_ratio=self.padding_ratio,
            min_pad_px=self.min_pad_px,
            mask_background=self.mask_background,
        )

        crop = resize_for_model(crop, self.crop_size)

        return to_model_tensor(crop)

    def extract_augmented_embeddings(
        self, particle: Particle, image_rgb: np.ndarray, rotations=(0, 90, 180, 270), flip: bool = True,
    ) -> List[np.ndarray]:
        """
        For a small labeled prototype set (build_prototypes.py), one
        embedding per real example isn't much for a classifier to
        learn from. SEM particles have no inherent "up" — a real
        Agglomerate rotated 90 degrees is still that same real
        Agglomerate — so embedding several rotated/mirrored views of
        the SAME crop is legitimate augmentation, not fabricated data:
        every view comes from one real labeled particle, and the
        crop's own morphology/physics features (rotation-invariant —
        circularity, solidity, area, aspect ratio) are correctly
        reused unchanged across all of them by the caller; only the
        embedding differs per view, since DINOv2 itself has no
        built-in rotational invariance.

        Returns one embedding per (rotation x flip-state) combination
        — up to len(rotations) * (2 if flip else 1). Caller is
        responsible for keeping every returned embedding tagged with
        the SAME source-particle group id, so cross-validation can
        keep them together in one fold (splitting augmented siblings
        across train/test would leak — see classification.py).
        """

        crop = crop_particle_patch(
            image_rgb, particle,
            padding_ratio=self.padding_ratio, min_pad_px=self.min_pad_px,
            mask_background=self.mask_background,
        )

        views = []

        for angle in rotations:

            rotated = crop

            if angle == 90:
                rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(crop, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

            views.append(rotated)

            if flip:
                views.append(cv2.flip(rotated, 1))

        tensors = [to_model_tensor(resize_for_model(v, self.crop_size)) for v in views]

        return self._embed_tensors(tensors)

    def _embed_tensors(self, tensors: Sequence[torch.Tensor]) -> List[np.ndarray]:

        embeddings: List[np.ndarray] = []

        for start in range(0, len(tensors), self.batch_size):

            chunk = torch.stack(
                list(tensors[start:start + self.batch_size])
            ).to(self.device)

            with torch.no_grad():
                out = self.model(chunk)

            embeddings.extend(out.detach().cpu().numpy())

        return embeddings

    # --------------------------------------------------------

    def extract(self, particle: Particle, image_rgb: np.ndarray) -> Particle:
        """
        Embed a single particle. Prefer `extract_batch` when embedding
        every particle from one image — one shared forward pass is far
        faster than calling this in a loop.
        """

        return self.extract_batch([particle], image_rgb)[0]

    def extract_batch(
        self,
        particles: List[Particle],
        image_rgb: np.ndarray,
    ) -> List[Particle]:
        """
        Embed every particle segmented from the same `image_rgb` in
        batched forward passes (chunked by `batch_size`).
        """

        if len(particles) == 0:
            return particles

        tensors = [
            self._prepare_tensor(image_rgb, particle) for particle in particles
        ]

        embeddings = self._embed_tensors(tensors)

        for particle, embedding in zip(particles, embeddings):

            particle.dinov2_embedding = embedding

            particle.metadata["dinov2_model"] = self.model_name
            particle.metadata["dinov2_dim"] = self.embedding_dim

        return particles

    def extract_for_analysis(self, analysis, image_rgb: np.ndarray):
        """
        Convenience wrapper: embed every particle in an ImageAnalysis.
        """

        self.extract_batch(analysis.particles, image_rgb)

        return analysis


# ============================================================
# FACTORY
# ============================================================

def build_dinov2_extractor(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    crop_size: int = 224,
    padding_ratio: float = 0.2,
    mask_background: bool = False,
    batch_size: int = 16,
) -> DINOv2FeatureExtractor:

    return DINOv2FeatureExtractor(
        model_name=model_name,
        device=device,
        crop_size=crop_size,
        padding_ratio=padding_ratio,
        mask_background=mask_background,
        batch_size=batch_size,
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "DINOV2_MODELS",
    "DEFAULT_MODEL",
    "load_dinov2_model",
    "compute_padded_bbox",
    "letterbox_square",
    "crop_particle_patch",
    "resize_for_model",
    "to_model_tensor",
    "DINOv2FeatureExtractor",
    "build_dinov2_extractor",
]
