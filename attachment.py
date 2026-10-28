=================
ParticleVision V4
=================

Attachment Detection

Whether particles in the same image are physically attached to each
other (a satellite fused to a host sphere, two comparable particles
bonded together, a multi-particle agglomerate) — computed directly
from mask geometry as a first-class relationship, not inferred
indirectly through a single per-particle shape score the way v3's
`neck_score` did.

This produces CONTINUOUS evidence (contact fraction, neighbor
distance) fed to the unified classifier as features, not a hard
accept/reject gate — v4 deliberately does not hardcode "attached" as a
classification rule for any of the 10 classes; the classifier learns
how much this evidence matters per class.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from data_models import Particle


def _dilate(mask: np.ndarray, gap_px: int) -> np.ndarray:

    if gap_px <= 0:
        return mask.astype(bool)

    kernel_size = 2 * gap_px + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _bboxes_overlap(a: Particle, b: Particle, pad: int) -> bool:

    ax0, ay0 = a.bbox.x - pad, a.bbox.y - pad
    ax1, ay1 = a.bbox.x + a.bbox.width + pad, a.bbox.y + a.bbox.height + pad

    bx0, by0 = b.bbox.x, b.bbox.y
    bx1, by1 = b.bbox.x + b.bbox.width, b.bbox.y + b.bbox.height

    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def compute_attachments(
    particles: List[Particle],
    contact_gap_px: int = 3,
    min_contact_fraction: float = 0.03,
    host_size_ratio: float = 1.8,
) -> List[Particle]:
    """
    Computes attachment relationships between every pair of particles
    in `particles` (expected to all be from the SAME image — attachment
    is only meaningful within one image). Mutates each particle's
    `.attachment` in place and returns the same list.

    For each candidate pair (bounding boxes within `contact_gap_px` of
    each other, cheap pre-filter before any mask math): dilate one
    particle's mask by `contact_gap_px` and intersect with the other's
    mask. Non-zero intersection means the two particles are touching
    or near-touching after accounting for segmentation/smoothing noise
    that can leave a 1-2px real gap at a genuine fusion neck.
    `boundary_contact_fraction` normalizes that intersection by the
    particle's own perimeter, so a large particle grazing a small one
    doesn't score the same as two particles sharing half their
    boundary.
    """

    n = len(particles)

    if n == 0:
        return particles

    # Centroid nearest-neighbor distance — a continuous feature
    # regardless of contact, computed once per particle up front.
    centroids = np.array(
        [[p.centroid.x, p.centroid.y] for p in particles], dtype=float
    )

    for i, particle in enumerate(particles):

        if n > 1:
            others = np.delete(centroids, i, axis=0)
            dists = np.linalg.norm(others - centroids[i], axis=1)
            particle.attachment.nearest_neighbor_distance_px = float(dists.min())
        else:
            particle.attachment.nearest_neighbor_distance_px = float("nan")

    # Pairwise contact detection.
    for i in range(n):

        a = particles[i]
        a_perimeter = a.morphology.get("perimeter")

        if a_perimeter is None or np.isnan(a_perimeter) or a_perimeter <= 0:
            continue

        a_dilated = None  # computed lazily, reused across b's below
        a_equiv = a.morphology.get("equivalent_diameter")

        for j in range(i + 1, n):

            b = particles[j]

            if not _bboxes_overlap(a, b, contact_gap_px):
                continue

            if a_dilated is None:
                a_dilated = _dilate(a.mask, contact_gap_px)

            contact_pixels = int(np.count_nonzero(a_dilated & b.mask.astype(bool)))

            if contact_pixels == 0:
                continue

            b_perimeter = b.morphology.get("perimeter")
            b_equiv = b.morphology.get("equivalent_diameter")

            # contact_pixels is a 2D area (the dilation ring intersected
            # with b's mask); perimeter is a 1D length — dividing area
            # directly by length isn't a bounded "fraction" (confirmed
            # on real segmented particles: it readily exceeds 1, up to
            # ~6, for particles with a wide/long shared contact zone).
            # Normalizing by the ring's own approximate max area
            # (perimeter * gap, i.e. "the whole boundary is touching
            # something") keeps this close to a genuine [0, ~1] fraction.
            ring_area_scale = max(contact_gap_px, 1)

            a_fraction = contact_pixels / (a_perimeter * ring_area_scale)

            b_fraction = (
                contact_pixels / (b_perimeter * ring_area_scale)
                if b_perimeter is not None and not np.isnan(b_perimeter) and b_perimeter > 0
                else 0.0
            )

            if a_fraction < min_contact_fraction and b_fraction < min_contact_fraction:
                continue

            # Both directions get contact evidence recorded; which
            # side (if either) counts as "host" depends on relative
            # size, independent of which fraction is larger.
            a.attachment.boundary_contact_fraction = max(
                a.attachment.boundary_contact_fraction, a_fraction
            )
            b.attachment.boundary_contact_fraction = max(
                b.attachment.boundary_contact_fraction, b_fraction
            )

            a.attachment.is_attached = True
            b.attachment.is_attached = True

            a.attachment.attached_particle_ids.append(b.particle_id)
            b.attachment.attached_particle_ids.append(a.particle_id)

            size_ok = (
                a_equiv is not None and b_equiv is not None
                and not np.isnan(a_equiv) and not np.isnan(b_equiv)
                and a_equiv > 0 and b_equiv > 0
            )

            if size_ok:

                if a_equiv >= b_equiv * host_size_ratio:
                    # a is clearly the larger host; b is the satellite.
                    if b.attachment.host_particle_id is None:
                        b.attachment.host_particle_id = a.particle_id

                elif b_equiv >= a_equiv * host_size_ratio:
                    if a.attachment.host_particle_id is None:
                        a.attachment.host_particle_id = b.particle_id

                # else: comparably-sized fused pair (Bonded_Particle /
                # Agglomerate territory) — no single "host", both just
                # carry mutual attachment evidence via
                # attached_particle_ids above.

    return particles


__all__ = ["compute_attachments"]
