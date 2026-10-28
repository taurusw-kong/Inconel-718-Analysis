"""
Segment a list of images and cache their label maps — nothing else.

Used to extend an existing run with newly-added images without
re-segmenting the ones already done. Writes the same
<base>_labels.npz format main.py does, so reclassify_from_masks.py can
consume the combined set afterwards.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, glob, logging, sys
import cv2

import config
from preprocessing import build_preprocessor, get_calibration
from segmentation import build_segmenter
from main import save_particle_label_map

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--names_file", required=True, help="one image basename (no extension) per line")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--model_cfg", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--scale_bar_um", type=float, default=None)
a = ap.parse_args()

names = [n.strip() for n in open(a.names_file) if n.strip()]
logger.info(f"{len(names)} image(s) to segment")

os.makedirs(os.path.join(a.output_dir, "masks"), exist_ok=True)
preprocessor = build_preprocessor()
# Every parameter here must match main.py's call exactly. The 103
# masks copied from the earlier run were produced with those settings;
# segmenting the new images with different ones would make the two
# halves of the dataset incomparable.
segmenter = build_segmenter(
    checkpoint_path=a.checkpoint,
    model_cfg=a.model_cfg,
    device=a.device,
    min_area=config.SEGMENTATION.min_particle_area_px,
    border_margin=config.SEGMENTATION.border_margin,
    edge_filtering=config.SEGMENTATION.edge_filtering,
    points_per_side=config.SEGMENTATION.points_per_side,
    pred_iou_thresh=config.SEGMENTATION.pred_iou_thresh,
    stability_score_thresh=config.SEGMENTATION.stability_score_thresh,
)

for i, base in enumerate(names, 1):
    matches = glob.glob(os.path.join(a.input_dir, base + ".*"))
    if not matches:
        logger.warning(f"no image for {base}")
        continue
    path = matches[0]
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        logger.warning(f"unreadable: {path}")
        continue

    get_calibration(image_bgr, os.path.basename(path), {},
                    scale_bar_um=a.scale_bar_um, image_path=path)
    image_rgb, raw_rgb, _ = preprocessor.process(image_bgr)
    particles, stats = segmenter.segment(image_rgb, image_name=os.path.basename(path),
                                         raw_image_rgb=raw_rgb)
    if particles:
        save_particle_label_map(particles, image_rgb.shape, a.output_dir,
                                os.path.basename(path), full_image_shape=image_bgr.shape)
    logger.info(f"[{i}/{len(names)}] {os.path.basename(path)}: "
                f"{stats.kept_masks} kept from {stats.raw_masks} raw")

logger.info("SEGMENTONLYDONE")
