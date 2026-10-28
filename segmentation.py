"""
ParticleVision V2
=================

Segmentation Engine

Author: ParticleVision V2

This module performs:

- SAM2 loading
- Automatic mask generation
- Duplicate removal
- Hole filling
- Boundary smoothing
- Watershed splitting
- Background rejection
- Adaptive filtering
- Quality scoring
- Particle object creation

"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import (
    SAM2AutomaticMaskGenerator,
)

from data_models import (
    BoundingBox,
    Centroid,
    Particle,
)


# ============================================================
# STATISTICS
# ============================================================

@dataclass
class SegmentationStatistics:

    raw_masks: int = 0

    kept_masks: int = 0

    rejected_small: int = 0

    rejected_overlap: int = 0

    rejected_background: int = 0

    rejected_quality: int = 0

    rejected_artifact: int = 0

    rejected_contained: int = 0

    split_particles: int = 0


# ============================================================
# MODEL LOADER
# ============================================================

def resolve_sam2_config_name(model_cfg):
    """
    SAM2's build_sam2() resolves `model_cfg` through Hydra's
    `pkg://sam2` search path — i.e. it must be a path *relative to
    the installed sam2 package's own directory*
    (e.g. "configs/sam2.1/sam2.1_hiera_s.yaml"), never an absolute
    filesystem path. Hydra silently mishandles an absolute path
    (it strips the leading slash and searches for the rest inside
    the package, which fails with a confusing
    `MissingConfigException`).

    A file picker (or a user just Browse-ing to the .yaml on disk)
    naturally produces an absolute path, so this normalizes it back
    to the form SAM2/Hydra actually expects whenever the given path
    is a real file that lives inside the installed `sam2` package.
    Anything else (a bare relative name the user already got right,
    e.g. "configs/sam2.1/sam2.1_hiera_s.yaml") is passed through
    unchanged.
    """

    if not os.path.isabs(model_cfg):
        return model_cfg

    if not os.path.exists(model_cfg):
        # Not a real file on disk — leave it alone, let Hydra's own
        # error surface if it's genuinely invalid.
        return model_cfg

    try:
        import sam2 as _sam2_pkg

        package_dir = os.path.dirname(os.path.abspath(_sam2_pkg.__file__))

    except Exception:
        return model_cfg

    abs_cfg = os.path.abspath(model_cfg)

    if abs_cfg.startswith(package_dir + os.sep):

        relative = os.path.relpath(abs_cfg, package_dir)

        # Hydra config names always use forward slashes, regardless
        # of OS.
        relative = relative.replace(os.sep, "/")

        if relative != model_cfg:

            print(
                f"[INFO] --model_cfg was an absolute path; using the "
                f"path relative to the sam2 package instead: {relative}"
            )

        return relative

    # Absolute path that isn't inside the installed sam2 package at
    # all — Hydra can't reach it either way, so leave it as given and
    # let the underlying error explain why.
    return model_cfg


def load_sam2_model(

    checkpoint_path,

    model_cfg,

    device="cuda",

    points_per_side=32,

    pred_iou_thresh=0.86,

    stability_score_thresh=0.92,

    min_mask_region_area=25,

):

    print(f"[INFO] Loading SAM2 model on {device}")

    model_cfg = resolve_sam2_config_name(model_cfg)

    sam_model = build_sam2(

        model_cfg,

        checkpoint_path,

        device=device,

    )

    generator = SAM2AutomaticMaskGenerator(

        model=sam_model,

        points_per_side=points_per_side,

        pred_iou_thresh=pred_iou_thresh,

        stability_score_thresh=stability_score_thresh,

        # crop_n_layers=0 (SAM2's own default) — NOT 1. With it >0, SAM2
        # additionally runs inference on cropped sub-tiles of the image
        # to help find small objects. When a particle straddles one of
        # those internal tile boundaries, SAM2 can propose two separate,
        # complementary masks instead of one whole-particle mask — each
        # half then gets classified and colored independently in the
        # overlay, producing a particle visibly split by a dead-straight
        # rectangular line (not a real physical boundary). This project's
        # own duplicate/containment filters (remove_duplicate_masks)
        # don't catch it either, since the two halves barely overlap —
        # they're not "duplicates," they're a true partition. Full-image-
        # only inference can't produce this artifact at all. Trade-off:
        # slightly lower recall on very small/dense particles, which the
        # crop layer was meant to help with — if satellite-class recall
        # matters more than eliminating this artifact, raise
        # points_per_side instead of re-enabling this.
        crop_n_layers=0,

        crop_n_points_downscale_factor=2,

        min_mask_region_area=min_mask_region_area,

    )

    print("[INFO] SAM2 ready.")

    return generator


# ============================================================
# BASIC UTILITIES
# ============================================================

def mask_iou(

    mask1,

    mask2,

):

    intersection = np.logical_and(

        mask1,

        mask2,

    ).sum()

    union = np.logical_or(

        mask1,

        mask2,

    ).sum()

    if union == 0:

        return 0.0

    return float(intersection / union)


def mask_containment(small_mask, large_mask):
    """
    Fraction of `small_mask`'s own area that overlaps `large_mask` —
    distinct from IoU. A small internal texture patch or surface
    feature sitting inside a larger particle can have LOW IoU with
    that particle (their union is dominated by the big mask's area)
    while still being almost entirely contained within it. IoU-based
    duplicate removal alone misses this case; containment catches it.
    """

    small_area = np.count_nonzero(small_mask)

    if small_area == 0:
        return 0.0

    intersection = np.count_nonzero(
        np.logical_and(small_mask, large_mask)
    )

    return float(intersection / small_area)


def contour_center(contour):

    M = cv2.moments(contour)

    if M["m00"] == 0:

        x, y, w, h = cv2.boundingRect(contour)

        return np.array([

            x + w / 2,

            y + h / 2,

        ])

    return np.array([

        M["m10"] / M["m00"],

        M["m01"] / M["m00"],

    ])


def contour_area(contour):

    return float(

        cv2.contourArea(

            contour

        )

    )


def contour_perimeter(contour):

    return float(

        cv2.arcLength(

            contour,

            True,

        )

    )


# ============================================================
# ARTIFACT REJECTION
# ============================================================

def is_artifact_shape(
    contour,
    image_shape,
    border_margin=5,
    edge_filtering=False,
    extent_threshold=0.94,
):
    """
    Catches two failure modes that otherwise show up as spurious
    "particles" in the overlay — most commonly the microscope's
    info/label bar (scale, kV, magnification) leaking into
    segmentation when the automatic strip-crop under-detects it:

    1. Near-perfect rectangles (area / bounding-box-area above
       `extent_threshold`). A genuine powder particle — round,
       elongated, or irregular — essentially never fills its own
       bounding box this completely; a UI panel, text block, or a
       sliver of the info bar does. This check is always applied
       (it has ~no false-positive risk on real particles).

    2. Optionally (`edge_filtering=True`), anything touching the
       image border within `border_margin` pixels. Off by default,
       since SEM mosaics legitimately have real particles cut off at
       the frame edge — only enable this if your convention is to
       exclude edge-truncated particles.
    """

    h, w = image_shape[:2]

    x, y, cw, ch = cv2.boundingRect(contour)

    area = float(cv2.contourArea(contour))

    bbox_area = float(cw * ch)

    extent = area / bbox_area if bbox_area > 0 else 0.0

    if extent >= extent_threshold:
        return True

    if edge_filtering:

        touches_border = (
            x <= border_margin
            or y <= border_margin
            or (x + cw) >= (w - border_margin)
            or (y + ch) >= (h - border_margin)
        )

        if touches_border:
            return True

    return False


def estimate_border_truncation(mask, image_shape, truncation_threshold=0.15):
    """
    How much of this particle is likely cut off by the image frame,
    not by its own real shape.

    A powder particle is (near-)convex — sphere, cap, elongated blob,
    or fused cluster of the same. When SAM2 sees the true particle
    boundary, the mask's outline is naturally curved/irregular the
    whole way round. When the frame clips the particle, the outline
    instead runs dead straight along whichever image edge cut it —
    that segment isn't the particle's real boundary at all, it's just
    where the photograph stopped. So the fraction of the mask's own
    outline that lies exactly ON the image border is a direct, shape-
    agnostic estimate of how much of the particle was never
    photographed, independent of the particle's actual shape or size.

    Returns (touches_border, border_fraction, is_truncated) where
    is_truncated is border_fraction >= truncation_threshold. The
    default 15% sits in the middle of the "unreliable to identify by
    eye" range this was calibrated against (roughly 10-20% missing).
    """

    h, w = image_shape[:2]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return False, 0.0, False

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    on_border = (
        (contour[:, 0] <= 0) | (contour[:, 0] >= w - 1) |
        (contour[:, 1] <= 0) | (contour[:, 1] >= h - 1)
    )

    border_fraction = float(np.mean(on_border)) if len(contour) > 0 else 0.0
    touches_border = border_fraction > 0.0

    return touches_border, border_fraction, border_fraction >= truncation_threshold


# ============================================================
# MASK CLEANING
# ============================================================

def remove_duplicate_masks(
    masks: List[dict],
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.90,
):

    if len(masks) <= 1:
        return masks, 0, 0

    masks = sorted(
        masks,
        key=lambda x: x["area"],
        reverse=True,
    )

    keep = []

    removed_duplicate = 0

    removed_contained = 0

    for candidate in masks:

        duplicate = False

        contained = False

        for accepted in keep:

            iou = mask_iou(
                candidate["segmentation"],
                accepted["segmentation"],
            )

            if iou >= iou_threshold:

                duplicate = True

                removed_duplicate += 1

                break

            # Not a near-duplicate by IoU, but could still be a small
            # internal fragment (surface texture, a highlight, a
            # sub-region) sitting almost entirely inside an already-
            # accepted larger mask — that's not a second particle.
            containment = mask_containment(
                candidate["segmentation"],
                accepted["segmentation"],
            )

            if containment >= containment_threshold:

                contained = True

                removed_contained += 1

                break

        if not duplicate and not contained:

            keep.append(candidate)

    return keep, removed_duplicate, removed_contained


def remove_contained_final_particles(particles, containment_threshold: float = 0.85):
    """
    Second, LATER duplicate/containment pass — over the fully
    post-processed final particle masks, not the raw SAM2 proposals
    remove_duplicate_masks() already deduplicated.

    Confirmed on real output: a spurious partial/crescent-shaped mask
    proposal can have too little overlap with a genuine particle's RAW
    mask to trip remove_duplicate_masks' containment check at that
    early stage — but fill_mask_holes/smooth_mask/
    split_touching_particles then reshape both masks independently,
    and the crescent can end up almost entirely inside the real
    particle's FINAL boundary even though it wasn't at the raw stage.
    Nothing previously re-checked for this after final shapes settled,
    so both survived as separate particles over the same physical
    region — visible in the overlay as one particle rendered in two
    different colors when they happen to get different class
    predictions. This catches that case: for any pair of final
    particles where the smaller one is mostly contained in the larger
    one, drop the smaller (keep the one that actually represents the
    real, complete particle shape).
    """

    if len(particles) <= 1:
        return particles, 0

    ordered = sorted(particles, key=lambda p: np.count_nonzero(p.mask), reverse=True)

    keep = []
    removed = 0

    for candidate in ordered:

        contained = False

        for accepted in keep:

            if mask_containment(candidate.mask, accepted.mask) >= containment_threshold:
                contained = True
                removed += 1
                break

        if not contained:
            keep.append(candidate)

    keep.sort(key=lambda p: p.particle_id)

    return keep, removed


def _contact_arc_fraction(child_seg, parent_seg, max_gap_px):
    """
    What fraction of the CHILD's own outline sits right at its contact
    with the parent — the physical signal that distinguishes "fused
    onto" from "merely resting against."

    Two solid particles that are just touching (not fused) meet at
    essentially a single tangent point — pixelation/antialiasing gives
    that a few pixels of width, but it's a small, roughly constant
    number of pixels REGARDLESS of the particles' actual size. A
    genuine attachment (satellite embedded in a host, a real fusion
    neck) has a much wider physical interface — a substantial fraction
    of the smaller particle's own diameter.

    So: walk the child's contour, count how many of its own boundary
    points sit within `max_gap_px` of the parent mask, and divide by
    the child's equivalent diameter (not its full perimeter — a wide,
    round particle has a long perimeter, so the raw arc length isn't
    comparable across particle shapes/sizes the way "fraction of own
    diameter" is).
    """

    contours, _ = cv2.findContours(
        child_seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    # Distance from every child boundary point to the nearest parent
    # pixel — cv2.distanceTransform needs the "outside" region as the
    # nonzero input (distance TO the nearest zero pixel).
    outside_parent = (~parent_seg).astype(np.uint8)
    dist = cv2.distanceTransform(outside_parent, cv2.DIST_L2, 5)

    at_contact = dist[contour[:, 1], contour[:, 0]] <= max_gap_px
    contact_arc_px = float(np.count_nonzero(at_contact))

    child_area = float(np.count_nonzero(child_seg))
    child_diameter_px = 2.0 * np.sqrt(child_area / np.pi) if child_area > 0 else 0.0

    return contact_arc_px / child_diameter_px if child_diameter_px > 0 else 0.0


def merge_into_whole_particles(
    masks,
    image_shape,
    bridge_gap_px=2,
    min_area=80,
    max_child_ratio=0.45,
    raw_image_rgb=None,
    min_bridge_intensity=40,
    min_bridge_textured_fraction=0.35,
    min_contact_fraction=0.15,
    max_particle_area_fraction=0.35,
):
    """
    Absorbs SMALL fragment masks into the LARGER particle they sit on,
    so one mask = one whole physical particle INCLUDING whatever is
    attached to it.

    This exists because the defect taxonomy is defined at the level of
    one connected physical unit:

        Molten/Bonded Satellite  = a particle WITH something attached
        Bonded Particle          = two comparable particles fused
        Agglomerate              = many particles fused
        Perfect/Oversized        = a single particle with nothing on it

    SAM2 proposes the host sphere AND its satellite as separate masks,
    and watershed splitting pulls them further apart. The classifier
    then receives a bare sphere and a bare speck — neither of which IS
    the class being named, so the defining evidence is destroyed
    before classification. Absorbing the speck into its host makes
    "has a satellite attached" visible in the mask shape itself.

    IMPORTANT — why this is size-ASYMMETRIC rather than a plain
    connected-component union: in a densely packed powder field almost
    every particle touches a neighbour, so unioning everything
    collapses the frame into a handful of giant blobs (measured: 42
    particles -> 9 on a real image, most of which then tripped the
    max-area artifact filter and were discarded entirely). Merging is
    therefore restricted to masks smaller than `max_child_ratio` of
    the mask they touch — a satellite gets absorbed into its host,
    while two comparable particles resting against each other stay
    separate, which is what SAM2's own proposal already gets right.

    IMPORTANT — why a bridged "touch" must be verified as real material:
    `bridge_gap_px` exists to close a 1-2px anti-aliasing gap at a
    genuine fusion neck, using dilation to test adjacency. Used blindly,
    that dilation also bridges a candidate to a neighbour across a real
    BLACK BACKGROUND gap between two separately-sitting particles — the
    two masks never actually touch, but the dilated probe overlaps
    anyway, and the resulting union paints that background gap in as
    if it were part of the particle (caught on a real image: a
    background wedge between three packed spheres got absorbed into
    one of them this way). So a merge that only exists because of the
    dilation bridge (no direct, undilated overlap) is only allowed
    through if the bridged pixels themselves look like real SEM
    material — bright and textured — using the same intensity/texture
    signature `remove_background_masks` uses to identify particles.
    Direct, undilated overlap (SAM2's own two proposals already
    touching) is always trusted with no extra check.

    IMPORTANT — why touching (even by real material) still isn't
    enough: in a densely packed powder field, particles that are
    genuinely separate constantly rest against each other — they
    touch at a tangent point, not because they're fused. Confirmed on
    a real image: a smaller round particle got absorbed into a larger
    neighbour, which had ALREADY absorbed two others, chaining four
    separate physical particles into one mask spanning nearly the
    whole image height. Area ratio and touching alone can't tell
    "resting against" from "fused onto" — see
    `_contact_arc_fraction`: a real attachment has a wide contact
    interface relative to the smaller particle's own size, while a
    tangent touch is a few pixels wide regardless of particle size.
    `min_contact_fraction` gates every absorption on this.
    """

    ordered = sorted(masks, key=lambda m: np.count_nonzero(m["segmentation"]), reverse=True)

    image_area = image_shape[0] * image_shape[1]
    max_absolute_area = max_particle_area_fraction * image_area

    kept = []            # list of boolean masks, largest first
    kept_sources = []    # the mask_dict each kept mask came from
    kept_original_area = []  # each entry's area at FIRST creation, never updated after merges

    kernel = None
    if bridge_gap_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * bridge_gap_px + 1, 2 * bridge_gap_px + 1),
        )

    gray, texture_map, texture_threshold = None, None, None
    if kernel is not None and raw_image_rgb is not None:
        gray = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2GRAY)
        texture_map = compute_local_texture_map(gray)
        texture_threshold = float(np.percentile(texture_map, 30))

    for mask_dict in ordered:

        seg = mask_dict["segmentation"].astype(bool)
        area = np.count_nonzero(seg)

        if area < min_area:
            continue

        probe = seg
        if kernel is not None:
            probe = cv2.dilate(seg.astype(np.uint8), kernel).astype(bool)

        absorbed = False

        for i, parent in enumerate(kept):

            # Compared against the parent's ORIGINAL starting size, not
            # its current (possibly already-grown) size. Confirmed on a
            # real, densely-packed image: comparing against current
            # size lets a parent keep growing through successive
            # merges, and each growth spurt makes the NEXT absorption
            # easier (its threshold rises with it) — a runaway chain
            # that absorbed 29 separate real particles into one blob
            # covering 97% of the frame, which then got discarded
            # whole by the max-area filter, silently losing every real
            # particle inside it instead of just the bad merge.
            if area > kept_original_area[i] * max_child_ratio:
                continue

            # Hard cap, independent of the ratio check above: even
            # many individually-small, individually-legitimate-looking
            # absorptions must never be allowed to grow one parent past
            # a physically-plausible single-particle size. The ratio
            # check alone doesn't catch this — a large host absorbing
            # several small children ONE AT A TIME, each comfortably
            # under the ratio threshold, can still end up enormous in
            # total. Confirmed on a real densely-packed image: this
            # exact mechanism (not the growing-threshold one the ratio
            # check already guards against) built a blob covering 97%
            # of the frame from otherwise-reasonable pairwise merges.
            parent_current_area = np.count_nonzero(parent)
            if parent_current_area + area > max_absolute_area:
                continue

            direct_touch = np.count_nonzero(seg & parent) > 0
            bridged_only = (not direct_touch) and np.count_nonzero(probe & parent) > 0

            if not (direct_touch or bridged_only):
                continue

            if bridged_only and gray is not None:

                # The pixels that only touch because of dilation — the
                # actual bridge — must look like real particle material,
                # not black background, or this isn't a real fusion neck.
                bridge_pixels = probe & ~seg & ~parent

                if np.count_nonzero(bridge_pixels) > 0:

                    bridge_intensity = float(np.mean(gray[bridge_pixels]))
                    bridge_textured = float(np.mean(
                        texture_map[bridge_pixels] >= texture_threshold
                    ))

                    if (bridge_intensity < min_bridge_intensity or
                            bridge_textured < min_bridge_textured_fraction):
                        continue

            # The core fix: a tangent touch between two separate
            # particles resting against each other must not be treated
            # the same as a real fusion neck. Measured against the
            # child's own diameter so it's scale-invariant.
            contact_fraction = _contact_arc_fraction(seg, parent, bridge_gap_px + 1)

            if contact_fraction < min_contact_fraction:
                continue

            kept[i] = parent | seg
            absorbed = True
            break

        if not absorbed:
            kept.append(seg)
            kept_sources.append(mask_dict)
            kept_original_area.append(area)

    return _split_any_disconnected_masks(kept, min_area)


def _split_any_disconnected_masks(masks, min_area):
    """
    Final safety net, not a first line of defence: guarantees every
    mask this function hands downstream is actually ONE connected
    region.

    A bridged merge (see the `bridged_only` branch above) is approved
    based on a DILATED adjacency test, tolerating a 1-2px anti-
    aliasing gap at a genuine fusion neck — but the actual stored mask
    is a plain union of the UNDILATED masks, so it can still contain
    that small gap. Confirmed on a real image: a merge chain left a
    6px gap (well past anti-aliasing) between two pieces stored under
    one particle ID — downstream contour extraction picked only the
    larger piece's contour, so the smaller piece's outline silently
    never got drawn even though its pixels were genuinely in the mask.
    Ported from v6, where this was found and fixed first.

    Splitting back into real connected components here means a
    mistaken/over-eager merge is visible and correctable, never
    invisible.
    """

    result = []

    for mask in masks:

        n_labels, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

        if n_labels <= 2:  # background (0) + exactly one component
            result.append(mask)
            continue

        for label in range(1, n_labels):

            piece = components == label

            if np.count_nonzero(piece) >= min_area:
                result.append(piece)

    return _resolve_overlaps(result, min_area)


def _resolve_overlaps(masks, min_area):
    """
    Guarantees no two returned masks share a single pixel.

    SAM2's automatic mode routinely proposes several overlapping
    candidates for the same physical particle. Nothing before this
    point compares DIFFERENT assembled particles against each other —
    each raw candidate is only ever tested against one potential
    parent at a time — so two separate final entries could still
    spatially overlap. Confirmed on a real image: this was silently
    inflating particle counts (the same physical particle counted 2-3
    times under different IDs) AND, downstream, writing overlapping
    masks into a shared label-map image meant whichever mask was
    written last carved a chunk out of the other one, producing
    internally-disconnected saved particles even after the merge logic
    itself was fixed. Ported from v6, where this was found and fixed
    first.

    Resolved here, once: process largest-first, and any later
    (necessarily smaller-or-equal) mask has already-claimed pixels
    removed before it's kept. The larger, more established detection
    is never altered.
    """

    ordered = sorted(masks, key=lambda m: np.count_nonzero(m), reverse=True)

    claimed = np.zeros_like(ordered[0]) if ordered else None
    resolved = []

    for mask in ordered:

        exclusive = mask & ~claimed

        n_labels, components = cv2.connectedComponents(exclusive.astype(np.uint8), connectivity=8)

        for label in range(1, n_labels):

            piece = components == label

            if np.count_nonzero(piece) >= min_area:
                resolved.append(piece)

        claimed = claimed | mask

    return resolved


def best_source_mask_for(component, masks):
    """
    The source SAM2 mask overlapping `component` most — used only to
    carry that proposal's quality metadata (predicted_iou,
    stability_score) onto the merged whole particle, since a merged
    component has no single mask_dict of its own.
    """

    best, best_overlap = None, 0

    for mask_dict in masks:

        overlap = np.count_nonzero(component & mask_dict["segmentation"].astype(bool))

        if overlap > best_overlap:
            best_overlap = overlap
            best = mask_dict

    return best if best is not None else {"predicted_iou": np.nan, "stability_score": np.nan, "area": np.nan}


def fill_mask_holes(
    mask,
    max_hole_area=200,
):

    mask = mask.astype(np.uint8)

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )

    flood_inv = cv2.bitwise_not(flood)

    holes = flood_inv & (
        (mask == 0).astype(np.uint8) * 255
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes,
        connectivity=8,
    )

    output = mask.copy()

    for i in range(1, n_labels):

        if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:

            output[labels == i] = 1

    return output.astype(bool)


def compute_raw_hole_stats(mask):
    """
    Hole count/area on the mask AS SAM2 PRODUCED IT, before
    fill_mask_holes() runs. That function intentionally erases any
    hole <= max_hole_area (default 200px) to clean up segmentation
    noise — but morphology.py's own porosity measurement
    (hole_count/hole_area/hole_area_ratio) reads `particle.mask`,
    which by then has already had those same holes filled. Every pore
    at or under that threshold was therefore invisible to porosity
    measurement by construction, not because the particle had none.

    This runs the identical flood-fill hole-detection fill_mask_holes
    uses, but reports the stats instead of erasing them, computed
    once per raw SAM2 mask (i.e. before watershed splitting). Callers
    that split one mask into several touching-particle components
    should treat this as a per-source-mask porosity estimate shared
    across those components, not a per-component exact measurement —
    still far more accurate than silently reading zero.
    """

    mask = mask.astype(np.uint8)

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )

    flood_inv = cv2.bitwise_not(flood)

    holes = flood_inv & (
        (mask == 0).astype(np.uint8) * 255
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes,
        connectivity=8,
    )

    hole_count = 0
    hole_area = 0.0
    largest_hole_area = 0.0

    for i in range(1, n_labels):

        area = float(stats[i, cv2.CC_STAT_AREA])

        hole_count += 1
        hole_area += area
        largest_hole_area = max(largest_hole_area, area)

    mask_area = float(np.count_nonzero(mask))

    hole_area_ratio = hole_area / mask_area if mask_area > 0 else 0.0

    return {

        "raw_hole_count": hole_count,

        "raw_hole_area_px": hole_area,

        "raw_largest_hole_area_px": largest_hole_area,

        "raw_hole_area_ratio": hole_area_ratio,

    }


def smooth_mask(
    mask,
    kernel_size=3,
):

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (kernel_size, kernel_size),

    )

    mask = mask.astype(np.uint8)

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel,

    )

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel,

    )

    return mask.astype(bool)


# ============================================================
# BACKGROUND FILTER
# ============================================================

def compute_local_texture_map(gray, ksize=9):
    """
    Local standard deviation over a small sliding window, computed
    efficiently via the box-filter identity
    std = sqrt(E[x^2] - E[x]^2). A genuine particle surface under SEM
    shows visible grain/texture (high local variance); a background
    gap is comparatively smooth (low local variance) — this holds
    true *regardless of the background's absolute brightness*, unlike
    a fixed intensity threshold, which only works when background
    happens to be near-black. Some microscope/detector setups produce
    a bright-ish, low-contrast gray background rather than black, and
    a pure brightness filter silently does nothing on those.
    """

    gray_f = gray.astype(np.float32)

    mean = cv2.boxFilter(gray_f, -1, (ksize, ksize))
    sq_mean = cv2.boxFilter(gray_f ** 2, -1, (ksize, ksize))

    local_var = np.clip(sq_mean - mean ** 2, 0, None)

    return np.sqrt(local_var)


def remove_background_masks(

    masks,

    image_rgb,

    min_mean_intensity=8,

    max_background_fraction=0.98,

    min_textured_fraction=0.35,

    texture_percentile=30,

):

    gray = cv2.cvtColor(

        image_rgb,

        cv2.COLOR_RGB2GRAY,

    )

    texture_map = compute_local_texture_map(gray)

    texture_threshold = float(np.percentile(texture_map, texture_percentile))

    filtered = []

    rejected = 0

    for m in masks:

        seg = m["segmentation"]

        pixels = gray[seg]

        if len(pixels) == 0:

            rejected += 1

            continue

        mean_intensity = float(

            np.mean(pixels)

        )

        background_fraction = float(

            np.mean(

                pixels < 5

            )

        )

        if mean_intensity < min_mean_intensity:

            rejected += 1

            continue

        if background_fraction > max_background_fraction:

            rejected += 1

            continue

        # Texture check — catches background regions a brightness
        # threshold can't (see compute_local_texture_map docstring).
        # Empirically calibrated against real particle vs. background
        # regions: real particles consistently score 0.92-0.99 here,
        # genuine background scores ~0.14-0.15 — 0.35 leaves generous
        # margin on both sides.
        textured_fraction = float(

            np.mean(

                texture_map[seg] >= texture_threshold

            )

        )

        if textured_fraction < min_textured_fraction:

            rejected += 1

            continue

        filtered.append(m)

    return filtered, rejected


# ============================================================
# ADAPTIVE PARAMETERS
# ============================================================

def estimate_particle_density(

    masks,

    image_shape,

):

    h, w = image_shape[:2]

    image_area = h * w

    particle_area = sum(

        m["area"]

        for m in masks

    )

    return particle_area / image_area


def adaptive_parameters(

    masks,

    image_shape,

):

    coverage = estimate_particle_density(

        masks,

        image_shape,

    )

    params = {

        "hole_area": 200,

        "min_distance": 12,

        "quality_threshold": 0.45,

    }

    if coverage < 0.10:

        params["min_distance"] = 8

    elif coverage > 0.35:

        params["min_distance"] = 18

    if coverage > 0.50:

        params["quality_threshold"] = 0.55

    return params


# ============================================================
# WATERSHED SPLITTING
# ============================================================

def split_touching_particles(
    mask,
    min_distance=12,
    min_component_area=80,
    min_peak_prominence_ratio=0.55,
):
    """
    Split merged/touching particles using watershed — but only when
    there's real evidence of multiple distinct particles, not just
    because a single irregular particle happens to have more than one
    bump in its distance transform.

    A lumpy-but-single particle (e.g. Agglomerate, Cap_Splat,
    Deformed_Particle — exactly the shapes this taxonomy cares about)
    can have several local maxima in its distance transform without
    actually being multiple touching particles. Requiring every
    candidate peak to reach at least `min_peak_prominence_ratio` of
    the mask's own deepest point discards shallow, non-committal
    bumps and only accepts peaks that represent a comparably "major"
    bulge — much closer to what a human would call a separate
    particle center.
    """

    mask = mask.astype(bool)

    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask,
    )

    if len(coords) <= 1:
        return [mask]

    peak_values = distance[coords[:, 0], coords[:, 1]]

    max_distance = float(peak_values.max())

    prominence_threshold = max_distance * min_peak_prominence_ratio

    significant = peak_values >= prominence_threshold

    coords = coords[significant]

    if len(coords) <= 1:
        # Only one "major" bulge survived — this is one irregular
        # particle, not multiple touching ones. Don't split.
        return [mask]

    markers = np.zeros_like(
        distance,
        dtype=np.int32,
    )

    for i, (r, c) in enumerate(coords):

        markers[r, c] = i + 1

    markers = ndi.label(markers)[0]

    labels = watershed(
        -distance,
        markers,
        mask=mask,
    )

    components = []

    for label in np.unique(labels):

        if label == 0:
            continue

        component = labels == label

        if component.sum() < min_component_area:
            continue

        components.append(component)

    if len(components) == 0:
        return [mask]

    return components


# ============================================================
# MASK QUALITY
# ============================================================

def compute_mask_quality(
    mask_dict,
    contour,
):
    """
    Segmentation-acceptance confidence — NOT shape quality. Deliberately
    does not weight circularity: this score gates whether a mask is
    kept at all (quality_threshold), and the defect taxonomy this
    pipeline exists to support is built around irregular shapes
    (Agglomerate, Non_Spherical, Cap_Splat, Deformed_Particle...).
    Penalizing low circularity here would systematically filter out
    exactly the particles most interesting to classify, before they
    ever reach the classifier. Shape belongs in classification, not
    in the segmentation accept/reject gate.
    """

    predicted_iou = mask_dict.get(
        "predicted_iou",
        0.0,
    )

    stability = mask_dict.get(
        "stability_score",
        0.0,
    )

    quality = (

        0.55 * predicted_iou +

        0.45 * stability

    )

    return float(quality)


# ============================================================
# PARTICLE CREATION
# ============================================================

def create_particle(

    particle_id,

    image_name,

    component,

    contour,

    mask_dict,

    raw_hole_stats=None,

):

    x, y, w, h = cv2.boundingRect(contour)

    M = cv2.moments(contour)

    if M["m00"] != 0:

        cx = M["m10"] / M["m00"]

        cy = M["m01"] / M["m00"]

    else:

        cx = x + w / 2

        cy = y + h / 2

    quality = compute_mask_quality(

        mask_dict,

        contour,

    )

    particle = Particle(

        particle_id=particle_id,

        image_name=image_name,

        mask=component,

        contour=contour,

        bbox=BoundingBox(

            x=x,

            y=y,

            width=w,

            height=h,

        ),

        centroid=Centroid(

            x=cx,

            y=cy,

        ),

        metadata={

            "predicted_iou": mask_dict.get(
                "predicted_iou",
                np.nan,
            ),

            "stability_score": mask_dict.get(
                "stability_score",
                np.nan,
            ),

            "sam2_area": mask_dict.get(
                "area",
                np.nan,
            ),

            "quality_score": quality,

            **(raw_hole_stats or {}),

        },

    )

    return particle, quality


# ============================================================
# SEGMENTATION ENGINE
# ============================================================

class SAM2Segmenter:

    def __init__(
        self,
        mask_generator,
        min_area=80,
        adaptive_segmentation=True,
        border_margin=5,
        edge_filtering=False,
        max_particle_area_fraction=0.35,
        whole_particle_mode=True,
        bridge_gap_px=2,
    ):

        self.mask_generator = mask_generator

        self.min_area = min_area

        self.adaptive_segmentation = adaptive_segmentation

        self.border_margin = border_margin

        self.edge_filtering = edge_filtering

        self.max_particle_area_fraction = max_particle_area_fraction

        # whole_particle_mode: one connected physical unit = one
        # particle (see merge_into_whole_particles). ON by default
        # because this taxonomy defines every class at that level —
        # a host sphere separated from its satellite is neither
        # "Perfect_Spherical" nor "Molten_Satellite", it's a broken
        # observation. Set False to restore per-blob watershed
        # splitting (useful for pure size distributions, where
        # touching particles genuinely should be separated).
        self.whole_particle_mode = whole_particle_mode

        self.bridge_gap_px = bridge_gap_px

    def segment(
        self,
        image_rgb,
        image_name="image",
        raw_image_rgb=None,
    ):

        stats = SegmentationStatistics()

        masks = self.mask_generator.generate(
            image_rgb
        )

        stats.raw_masks = len(masks)

        masks, removed_duplicate, removed_contained = remove_duplicate_masks(
            masks
        )

        stats.rejected_overlap = removed_duplicate

        stats.rejected_contained = removed_contained

        # Background-vs-particle brightness rejection is far more
        # reliable on the RAW, pre-enhancement image: CLAHE (and
        # other contrast enhancement) can stretch a genuinely
        # near-black background's brightness/texture just enough to
        # pass a fixed absolute threshold, letting the background
        # itself slip through as a "particle" candidate. Falls back
        # to `image_rgb` (the enhanced image mask generation actually
        # ran on) if no raw version was provided, for backward
        # compatibility.
        masks, rejected = remove_background_masks(
            masks,
            raw_image_rgb if raw_image_rgb is not None else image_rgb,
        )

        stats.rejected_background = rejected

        if self.adaptive_segmentation:

            params = adaptive_parameters(
                masks,
                image_rgb.shape,
            )

        else:

            params = {

                "hole_area": 200,

                "min_distance": 12,

                "quality_threshold": 0.45,

            }

        particles = []

        particle_id = 1

        if self.whole_particle_mode:

            components = merge_into_whole_particles(
                masks,
                image_rgb.shape,
                bridge_gap_px=self.bridge_gap_px,
                min_area=self.min_area,
                raw_image_rgb=raw_image_rgb if raw_image_rgb is not None else image_rgb,
                max_particle_area_fraction=self.max_particle_area_fraction,
            )

            image_area = image_rgb.shape[0] * image_rgb.shape[1]

            for component in components:

                if np.count_nonzero(component) > self.max_particle_area_fraction * image_area:
                    stats.rejected_artifact += 1
                    continue

                source = best_source_mask_for(component, masks)

                raw_hole_stats = compute_raw_hole_stats(component)

                filled = fill_mask_holes(component, max_hole_area=params["hole_area"])
                filled = smooth_mask(filled).astype(np.uint8)

                contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)

                epsilon = max(0.001 * contour_perimeter(contour), 0.5)
                contour = cv2.approxPolyDP(contour, epsilon, True)

                if is_artifact_shape(
                    contour, image_rgb.shape,
                    border_margin=self.border_margin,
                    edge_filtering=self.edge_filtering,
                ):
                    stats.rejected_artifact += 1
                    continue

                particle, quality = create_particle(
                    particle_id, image_name, filled.astype(bool), contour, source,
                    raw_hole_stats=raw_hole_stats,
                )

                if quality < params["quality_threshold"]:
                    stats.rejected_quality += 1
                    continue

                touches_border, border_fraction, is_truncated = estimate_border_truncation(
                    filled.astype(bool), image_rgb.shape,
                )
                particle.metadata["touches_border"] = touches_border
                particle.metadata["border_fraction"] = border_fraction
                particle.metadata["border_truncated"] = is_truncated

                particles.append(particle)
                particle_id += 1

            stats.kept_masks = len(particles)

            return particles, stats

        for mask_dict in sorted(

            masks,

            key=lambda x: x["area"],

            reverse=True,

        ):

            # Must run BEFORE fill_mask_holes — see
            # compute_raw_hole_stats' docstring. Any hole this misses
            # here is gone from every downstream mask permanently.
            raw_hole_stats = compute_raw_hole_stats(
                mask_dict["segmentation"],
            )

            seg = fill_mask_holes(

                mask_dict["segmentation"],

                max_hole_area=params["hole_area"],

            )

            seg = smooth_mask(seg)

            if np.count_nonzero(seg) < self.min_area:

                stats.rejected_small += 1

                continue

            # A real single powder particle should never take up a
            # huge fraction of the whole frame — that's the signature
            # of a mis-segmented BACKGROUND region slipping through
            # (e.g. contrast enhancement pushed the background's
            # brightness/texture just enough to pass
            # remove_background_masks' thresholds). Left unchecked,
            # this large blob then gets fed to the watershed splitter
            # below, which fragments it into a mesh of tiny irregular
            # slivers that all end up classified identically — the
            # solid colored "web" pattern between real particles seen
            # in the overlay. Reject it outright, before splitting,
            # rather than trying to clean up its fragments afterward.
            image_area = image_rgb.shape[0] * image_rgb.shape[1]

            if np.count_nonzero(seg) > self.max_particle_area_fraction * image_area:

                stats.rejected_artifact += 1

                continue

            split_masks = split_touching_particles(

                seg,

                min_distance=params["min_distance"],

            )

            if len(split_masks) > 1:

                stats.split_particles += (

                    len(split_masks) - 1

                )

            for component in split_masks:

                component = component.astype(np.uint8)

                contours, _ = cv2.findContours(

                    component,

                    cv2.RETR_EXTERNAL,

                    cv2.CHAIN_APPROX_NONE,

                )

                if len(contours) == 0:

                    continue

                contour = max(

                    contours,

                    key=cv2.contourArea,

                )

                epsilon = max(

                    0.001 * contour_perimeter(

                        contour

                    ),

                    0.5,

                )

                contour = cv2.approxPolyDP(

                    contour,

                    epsilon,

                    True,

                )

                if is_artifact_shape(

                    contour,

                    image_rgb.shape,

                    border_margin=self.border_margin,

                    edge_filtering=self.edge_filtering,

                ):

                    stats.rejected_artifact += 1

                    continue

                particle, quality = create_particle(

                    particle_id,

                    image_name,

                    component,

                    contour,

                    mask_dict,

                    raw_hole_stats=raw_hole_stats,

                )

                if quality < params["quality_threshold"]:

                    stats.rejected_quality += 1

                    continue

                touches_border, border_fraction, is_truncated = estimate_border_truncation(
                    component, image_rgb.shape,
                )
                particle.metadata["touches_border"] = touches_border
                particle.metadata["border_fraction"] = border_fraction
                particle.metadata["border_truncated"] = is_truncated

                particles.append(

                    particle

                )

                particle_id += 1

        particles, removed_final_dup = remove_contained_final_particles(particles)

        stats.rejected_overlap += removed_final_dup

        stats.kept_masks = len(

            particles

        )

        return particles, stats


    # ============================================================
# PUBLIC API
# ============================================================

    def generate_masks(
        self,
        image_rgb,
    ):
        """
        Generate raw SAM2 masks without post-processing.
        """

        return self.mask_generator.generate(image_rgb)

    def segment_single_mask(
        self,
        mask_dict,
        image_name="image",
    ):
        """
        Convert a single SAM2 mask into one or more Particle objects.
        Useful for debugging and visualization.
        """

        params = {

            "hole_area": 200,

            "min_distance": 12,

            "quality_threshold": 0.45,

        }

        seg = fill_mask_holes(
            mask_dict["segmentation"],
            max_hole_area=params["hole_area"],
        )

        seg = smooth_mask(seg)

        particles = []

        particle_id = 1

        split_masks = split_touching_particles(
            seg,
            min_distance=params["min_distance"],
        )

        for component in split_masks:

            component = component.astype(np.uint8)

            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            if len(contours) == 0:
                continue

            contour = max(
                contours,
                key=cv2.contourArea,
            )

            epsilon = max(
                0.001 * contour_perimeter(contour),
                0.5,
            )

            contour = cv2.approxPolyDP(
                contour,
                epsilon,
                True,
            )

            particle, quality = create_particle(
                particle_id,
                image_name,
                component,
                contour,
                mask_dict,
            )

            if quality < params["quality_threshold"]:
                continue

            touches_border, border_fraction, is_truncated = estimate_border_truncation(
                component, image_rgb.shape,
            )
            particle.metadata["touches_border"] = touches_border
            particle.metadata["border_fraction"] = border_fraction
            particle.metadata["border_truncated"] = is_truncated

            particles.append(
                particle
            )

            particle_id += 1

        return particles


# ============================================================
# FACTORY
# ============================================================

def build_segmenter(
    checkpoint_path,
    model_cfg,
    device="cpu",
    min_area=80,
    border_margin=5,
    edge_filtering=False,
    max_particle_area_fraction=0.35,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
):
    """
    points_per_side/pred_iou_thresh/stability_score_thresh previously
    had NO effect no matter what config.SEGMENTATION said — this
    function never forwarded them to load_sam2_model(), which always
    ran on ITS OWN hardcoded defaults. Found while testing why complex
    particles were going undetected: config.py documents these as
    user-tunable, but they were silently disconnected from the actual
    pipeline. Fixed by threading them through here.
    """

    generator = load_sam2_model(

        checkpoint_path=checkpoint_path,

        model_cfg=model_cfg,

        device=device,

        points_per_side=points_per_side,

        pred_iou_thresh=pred_iou_thresh,

        stability_score_thresh=stability_score_thresh,

    )

    return SAM2Segmenter(

        mask_generator=generator,

        min_area=min_area,

        adaptive_segmentation=True,

        border_margin=border_margin,

        edge_filtering=edge_filtering,

        max_particle_area_fraction=max_particle_area_fraction,

    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    "SegmentationStatistics",

    "SAM2Segmenter",

    "build_segmenter",

    "load_sam2_model",

]