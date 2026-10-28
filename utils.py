"""
ParticleVision V2
=================
Common utility functions.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger("ParticleVision")


# ---------------------------------------------------------------------
# TIMER
# ---------------------------------------------------------------------

class Timer:

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        logger.info(f"Started: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        logger.info(f"Finished: {self.name} ({elapsed:.2f} s)")


# ---------------------------------------------------------------------
# FILES
# ---------------------------------------------------------------------

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(folder: str):

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    folder = Path(folder)

    return sorted(
        [
            p for p in folder.iterdir()
            if p.suffix.lower() in exts
        ]
    )


# ---------------------------------------------------------------------
# IMAGE IO
# ---------------------------------------------------------------------

def load_rgb(path: str):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: str, image):

    cv2.imwrite(
        path,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )


# ---------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------

def euclidean_distance(p1, p2):

    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def safe_divide(a, b):

    if b == 0:
        return np.nan

    return a / b


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------

def normalize(value, min_value, max_value):

    if max_value == min_value:
        return 0.0

    return (value - min_value) / (max_value - min_value)


# ---------------------------------------------------------------------
# MASK
# ---------------------------------------------------------------------

def mask_area(mask):

    return int(mask.sum())


def mask_perimeter(mask):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return 0.0

    return cv2.arcLength(max(contours, key=cv2.contourArea), True)