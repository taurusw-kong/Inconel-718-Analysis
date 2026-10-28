"""
ParticleVision V2
=================

SEM Image Preprocessing Engine

Functions
---------
- Manual calibration
- OCR calibration
- Automatic strip detection
- Strip removal
- Image enhancement
- Quality assessment
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

from utils import logger


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Calibration:

    microns_per_pixel: Optional[float]

    magnification: Optional[int]

    source: str


@dataclass
class PreprocessingStatistics:

    strip_height: int = 0

    enhancement_method: str = "none"

    contrast_before: float = 0.0

    contrast_after: float = 0.0

    noise_before: float = 0.0

    noise_after: float = 0.0


# ============================================================
# CALIBRATION
# ============================================================

def load_manual_calibration(
    calibration_csv,
):

    if calibration_csv is None:

        return {}

    if not os.path.exists(calibration_csv):

        return {}

    df = pd.read_csv(calibration_csv)

    calibration = {}

    for _, row in df.iterrows():

        calibration[str(row["image_name"])] = Calibration(

            microns_per_pixel=float(
                row["microns_per_pixel"]
            ),

            magnification=int(
                row["magnification"]
            )
            if "magnification" in row
            and not pd.isna(row["magnification"])
            else None,

            source="manual_csv",

        )

    return calibration


# ============================================================
# OCR
# ============================================================

def read_scale_bar_with_ocr(

    image_bgr,

    strip_fraction=0.08,

):

    try:

        import pytesseract

    except Exception:

        return None

    h = image_bgr.shape[0]

    strip = image_bgr[
        h - int(h * strip_fraction):,
        :
    ]

    gray = cv2.cvtColor(

        strip,

        cv2.COLOR_BGR2GRAY,

    )

    text = pytesseract.image_to_string(

        gray,

        config="--psm 7",

    )

    magnification = None

    scale_um = None

    mag = re.search(

        r"[xX]\s?([\d,]+)",

        text,

    )

    if mag:

        try:

            magnification = int(

                mag.group(1).replace(",", "")

            )

        except Exception:

            pass

    um = re.search(

        r"(\d+(\.\d+)?)\s?(um|µm|μm)",

        text,

        re.IGNORECASE,

    )

    if um:

        try:

            scale_um = float(

                um.group(1)

            )

        except Exception:

            pass

    _, bw = cv2.threshold(

        gray,

        200,

        255,

        cv2.THRESH_BINARY,

    )

    contours, _ = cv2.findContours(

        bw,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE,

    )

    scale_pixels = None

    longest = 0

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        if w > longest and h < 20:

            longest = w

            scale_pixels = w

    if scale_um is None:

        return None

    if scale_pixels is None:

        return None

    return Calibration(

        microns_per_pixel=scale_um / scale_pixels,

        magnification=magnification,

        source="ocr",

    )

# ============================================================
# CALIBRATION RESOLUTION
# ============================================================

def measure_scale_bar_px(image_bgr, strip_search_px=90, min_bar_px=20):
    """
    Measures the burned-in scale bar's length in pixels by finding the
    longest solid horizontal run of white in the microscope info strip
    at the bottom of the frame. No OCR involved — the BAR is a
    geometric feature, and only its LABEL ("10um") is text, which the
    caller supplies as a known constant.

    This exists because the OCR-based calibration path proved
    unreliable on this project's real images: on one it returned
    0.96 um/px when the true value was 0.10 (a ~10x error that silently
    corrupted every size measurement and all three physics formulas),
    and on another it failed outright and returned nothing, leaving
    particles with no size at all. Measuring the bar directly agrees
    with the independent FOV-based calculation to within 1%.
    """

    h, w = image_bgr.shape[:2]

    strip = cv2.cvtColor(image_bgr[max(0, h - strip_search_px):, :], cv2.COLOR_BGR2GRAY)

    white = strip > 240

    longest = 0

    for row in white:

        current = 0

        for value in row:
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

    return longest if longest >= min_bar_px else None


def calibration_from_scale_bar(image_bgr, scale_bar_um=10.0):
    """
    Calibration from the measured scale bar, given the bar's labelled
    value in microns (printed next to it on the image — constant for a
    given magnification, so it's supplied once rather than OCR'd per
    image).
    """

    bar_px = measure_scale_bar_px(image_bgr)

    if not bar_px or not scale_bar_um:
        return None

    return Calibration(

        microns_per_pixel=float(scale_bar_um) / float(bar_px),

        magnification=None,

        source="scale_bar",

    )


def parse_jeol_sidecar_calibration(image_path):
    """
    JEOL's SEM control software writes a `.txt` sidecar next to every
    captured image (same basename — `15.bmp` -> `15.txt`) recording
    the actual acquisition parameters, including the real field-of-
    view width/height in microns ($CM_FIELD_OF_VIEW) and the image's
    pixel dimensions ($CM_IMAGE_SIZE). This is the single most
    authoritative calibration source available — it's not measured or
    estimated from the image at all, it's what the instrument itself
    recorded at capture time.

    Also cross-checks against the sidecar's independent scale-bar spec
    ($SM_MICRON_MARKER label vs. $SM_MICRON_BAR length in pixels) and
    logs a warning if the two disagree by more than 2% — they should
    always agree exactly for a correctly configured microscope, so a
    mismatch would mean something is wrong with either this file or
    the assumption that follows from it.

    Returns None (falls through to the next calibration source) if no
    matching sidecar exists or it can't be parsed — this is an
    optional enhancement, not a required file.
    """

    base, _ = os.path.splitext(str(image_path))
    sidecar_path = base + ".txt"

    if not os.path.exists(sidecar_path):
        return None

    try:
        with open(sidecar_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    fov_match = re.search(r"\$CM_FIELD_OF_VIEW\s+([\d.]+)\s+([\d.]+)", text)
    size_match = re.search(r"\$CM_IMAGE_SIZE\s+(\d+)\s+(\d+)", text)

    if not fov_match or not size_match:
        return None

    fov_width_um = float(fov_match.group(1))
    image_width_px = int(size_match.group(1))

    if fov_width_um <= 0 or image_width_px <= 0:
        return None

    microns_per_pixel = fov_width_um / image_width_px

    # Cross-check against the independent scale-bar spec, if present.
    marker_match = re.search(r"\$SM_MICRON_MARKER\s+([\d.]+)", text)
    bar_px_match = re.search(r"\$SM_MICRON_BAR\s+(\d+)", text)

    if marker_match and bar_px_match:

        scale_bar_um = float(marker_match.group(1))
        scale_bar_px = float(bar_px_match.group(1))

        if scale_bar_px > 0:

            from_bar = scale_bar_um / scale_bar_px
            disagreement = abs(from_bar - microns_per_pixel) / microns_per_pixel

            if disagreement > 0.02:
                logger.warning(
                    f"{os.path.basename(sidecar_path)}: FOV-derived calibration "
                    f"({microns_per_pixel:.4f} um/px) and scale-bar spec "
                    f"({from_bar:.4f} um/px) disagree by {disagreement:.1%} — "
                    "check this sidecar file."
                )

    return Calibration(
        microns_per_pixel=microns_per_pixel,
        magnification=None,
        source="jeol_sidecar",
    )


def calibration_from_fov_width(

    image_bgr,

    fov_width_um,

):
    """
    The simplest, most robust calibration source of all: microscope
    info bars burn in the field-of-view width in real units (e.g.
    "FOV:128x96um"). If you tell us that one number, microns-per-pixel
    is just fov_width_um / image_pixel_width — no OCR, no scale-bar
    detection, no CSV file to author by hand. Assumes every image in
    the batch was captured at the same magnification/FOV, which is
    the normal case for a single imaging session.
    """

    image_width_px = image_bgr.shape[1]

    if not fov_width_um or image_width_px <= 0:
        return None

    return Calibration(

        microns_per_pixel=float(fov_width_um) / float(image_width_px),

        magnification=None,

        source="fov_width",

    )


def get_calibration(

    image_bgr,

    image_name,

    manual_calibration,

    default_microns_per_pixel=None,

    allow_ocr=True,

    fov_width_um=None,

    scale_bar_um=None,

    image_path=None,

):
    """
    Calibration sources, most-trusted first:

      1. manual per-image CSV (you stated it explicitly)
      2. JEOL .txt sidecar (the instrument's own recorded FOV/pixel
         size — not measured or estimated at all, see
         parse_jeol_sidecar_calibration)
      3. field-of-view width (you stated it explicitly, one value for
         the whole batch)
      4. measured scale bar + its known label (geometry, no OCR)
      5. OCR of the info bar  — LAST RESORT, see warning below
      6. the supplied default

    The OCR path is deliberately demoted below direct measurement: on
    this project's real images it once returned 0.96 um/px where the
    true value was 0.10, a ~10x error that silently propagated into
    every particle size and all three physics equations before anyone
    noticed. Anything that states or measures the scale directly is
    preferred over reading text.
    """

    if image_name in manual_calibration:

        return manual_calibration[image_name]

    if image_path is not None:

        calib = parse_jeol_sidecar_calibration(image_path)

        if calib is not None:

            return calib

    if fov_width_um:

        calib = calibration_from_fov_width(

            image_bgr,

            fov_width_um,

        )

        if calib is not None:

            return calib

    if scale_bar_um:

        calib = calibration_from_scale_bar(image_bgr, scale_bar_um=scale_bar_um)

        if calib is not None:

            return calib

    if allow_ocr:

        calib = read_scale_bar_with_ocr(

            image_bgr,

        )

        if calib is not None:

            return calib

    if default_microns_per_pixel is None:

        # Loud, not silent. Previously this returned None quietly, and
        # every downstream size/physics feature became NaN for that
        # image without anything in the log saying so.
        logger.warning(
            f"{image_name}: NO CALIBRATION FOUND — particle sizes and all "
            "physics features will be missing for this image. Supply "
            "--fov_width_um or --scale_bar_um."
        )

    return Calibration(

        microns_per_pixel=default_microns_per_pixel,

        magnification=None,

        source="default",

    )


# ============================================================
# STRIP DETECTION
# ============================================================

def detect_information_strip(

    image_rgb,

    search_fraction=0.20,

    zero_intensity_threshold=3,

    zero_fraction_threshold=0.30,

):
    """
    Finds where the microscope's info bar starts by looking for
    fully-black pixels, not just "low variance" rows.

    Real SEM image content — background or particle surface — never
    has exactly-zero pixels; ordinary sensor noise means every row has
    at least *some* spread. A solid-black info bar background does
    the opposite: even its densest text row is still overwhelmingly
    pure-black pixels (white text covers a small fraction of any given
    row). So the fraction of near-zero pixels in a row cleanly
    separates "real image" (~0%) from "info bar, including text rows"
    (usually 60-100%) — and unlike a simple variance threshold, this
    doesn't get fooled by the dark gap *between* two lines of text
    inside the bar (both the gap and the text rows score high here).

    Empirically validated against real JEOL SED output (two-line info
    bar: readouts + scale bar on one line, FOV/date on the next) —
    exact match against the true, manually-verified boundary.
    """

    gray = cv2.cvtColor(

        image_rgb,

        cv2.COLOR_RGB2GRAY,

    )

    h, w = gray.shape

    search_rows = int(

        h * search_fraction

    )

    zero_fraction_per_row = (gray < zero_intensity_threshold).mean(axis=1)

    strip_start = None

    for r in range(h - 1, max(h - search_rows, 0) - 1, -1):

        if zero_fraction_per_row[r] <= zero_fraction_threshold:

            strip_start = r + 1

            break

    if strip_start is None:

        # The entire search window looks like the bar (or the image
        # has no info bar at all and just happens to have a dark
        # bottom edge) -- fall back to the old conservative default
        # rather than cropping too aggressively.
        return int(

            h * 0.08

        )

    return h - strip_start


def remove_information_strip(

    image_rgb,

    strip_height,

):

    if strip_height <= 0:

        return image_rgb

    return image_rgb[

        :-strip_height

    ].copy()


# ============================================================
# IMAGE QUALITY
# ============================================================

def estimate_noise(

    gray,

):

    lap = cv2.Laplacian(

        gray,

        cv2.CV_64F,

    )

    return float(

        lap.std()

    )


def estimate_contrast(

    gray,

):

    return float(

        gray.std()

    )


# ============================================================
# ENHANCEMENT
# ============================================================

def enhance_clahe(

    image_rgb,

):

    gray = cv2.cvtColor(

        image_rgb,

        cv2.COLOR_RGB2GRAY,

    )

    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8),

    )

    gray = clahe.apply(

        gray

    )

    gray = cv2.fastNlMeansDenoising(

        gray,

        h=7,

    )

    return cv2.cvtColor(

        gray,

        cv2.COLOR_GRAY2RGB,

    )


def enhance_bilateral(

    image_rgb,

):

    gray = cv2.cvtColor(

        image_rgb,

        cv2.COLOR_RGB2GRAY,

    )

    gray = cv2.bilateralFilter(

        gray,

        9,

        75,

        75,

    )

    return cv2.cvtColor(

        gray,

        cv2.COLOR_GRAY2RGB,

    )


def enhance_nlm(

    image_rgb,

):

    gray = cv2.cvtColor(

        image_rgb,

        cv2.COLOR_RGB2GRAY,

    )

    gray = cv2.fastNlMeansDenoising(

        gray,

        h=10,

    )

    return cv2.cvtColor(

        gray,

        cv2.COLOR_GRAY2RGB,

    )


def enhance_gamma(

    image_rgb,

    gamma=1.15,

):

    inv = 1.0 / gamma

    table = np.array([

        ((i / 255.0) ** inv) * 255

        for i in range(256)

    ]).astype(np.uint8)

    return cv2.LUT(

        image_rgb,

        table,

    )


def enhance_none(

    image_rgb,

):

    return image_rgb

# ============================================================
# PREPROCESSOR
# ============================================================

class ImagePreprocessor:

    def __init__(

        self,

        enhancement="clahe",

        remove_strip=True,

        auto_detect_strip=True,

        strip_fraction=0.08,

        strip_height_px=None,

    ):

        self.enhancement = enhancement

        self.remove_strip = remove_strip

        self.auto_detect_strip = auto_detect_strip

        self.strip_fraction = strip_fraction

        # Manual override — if set, always wins over both the
        # heuristic detector and the fallback fraction. Most
        # microscope software (JEOL etc.) burns in an info bar of a
        # *fixed* pixel height for a given resolution/settings, so
        # measuring it once (open any image in an editor, note the
        # y-coordinate where the black bar starts) and setting it
        # here is far more reliable than auto-detection, which can
        # under-crop on info bars that are mostly uniform black with
        # only thin white text (low row-to-row variance almost
        # everywhere, which is exactly what the auto-detector looks
        # for to find the boundary).
        self.strip_height_px = strip_height_px

    def process(

        self,

        image_bgr,

    ):

        stats = PreprocessingStatistics()

        image_rgb = cv2.cvtColor(

            image_bgr,

            cv2.COLOR_BGR2RGB,

        )

        gray_before = cv2.cvtColor(

            image_rgb,

            cv2.COLOR_RGB2GRAY,

        )

        stats.contrast_before = estimate_contrast(

            gray_before,

        )

        stats.noise_before = estimate_noise(

            gray_before,

        )

        if self.remove_strip:

            if self.strip_height_px is not None:

                strip_height = int(self.strip_height_px)

            elif self.auto_detect_strip:

                strip_height = detect_information_strip(

                    image_rgb,

                )

            else:

                strip_height = int(

                    image_rgb.shape[0]

                    * self.strip_fraction

                )

            image_rgb = remove_information_strip(

                image_rgb,

                strip_height,

            )

            stats.strip_height = strip_height

        # Keep a copy of the strip-cropped image BEFORE any contrast
        # enhancement. Enhancement (CLAHE especially) can stretch a
        # genuinely near-black background's brightness/texture enough
        # to fool a brightness-based background-vs-particle check
        # downstream in segmentation — so that check should compare
        # against these raw, physically-meaningful pixel values
        # instead of the enhanced ones.
        raw_rgb = image_rgb.copy()

        if self.enhancement == "clahe":

            image_rgb = enhance_clahe(

                image_rgb,

            )

        elif self.enhancement == "bilateral":

            image_rgb = enhance_bilateral(

                image_rgb,

            )

        elif self.enhancement == "nlm":

            image_rgb = enhance_nlm(

                image_rgb,

            )

        elif self.enhancement == "gamma":

            image_rgb = enhance_gamma(

                image_rgb,

            )

        elif self.enhancement == "none":

            image_rgb = enhance_none(

                image_rgb,

            )

        else:

            raise ValueError(

                f"Unknown enhancement method: {self.enhancement}"

            )

        gray_after = cv2.cvtColor(

            image_rgb,

            cv2.COLOR_RGB2GRAY,

        )

        stats.contrast_after = estimate_contrast(

            gray_after,

        )

        stats.noise_after = estimate_noise(

            gray_after,

        )

        stats.enhancement_method = self.enhancement

        return image_rgb, raw_rgb, stats


# ============================================================
# FACTORY
# ============================================================

def build_preprocessor(

    enhancement="clahe",

    strip_height_px=None,

):

    return ImagePreprocessor(

        enhancement=enhancement,

        remove_strip=True,

        auto_detect_strip=True,

        strip_height_px=strip_height_px,

    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    "Calibration",

    "PreprocessingStatistics",

    "ImagePreprocessor",

    "build_preprocessor",

    "load_manual_calibration",

    "get_calibration",

    "calibration_from_fov_width",

]