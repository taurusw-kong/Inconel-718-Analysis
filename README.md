# Inconel 718 Powder Defect Analysis

Code for automated defect classification and reusability assessment of
Inconel 718 powder for laser powder bed fusion from SEM micrographs.

The pipeline:

1. **Segmentation**: SAM2 automatic mask generation with rules for
   background rejection, bridge splitting and frame-truncated particles
   (`segmentation.py`, `preprocessing.py`).
2. **Features**: morphological descriptors, physics-constrained
   descriptors (Weber number, flaw-sensitivity index, attachment geometry)
   and DINOv2 embeddings, fused into one feature vector
   (`morphology.py`, `physics_formulas*.py`, `attachment.py`,
   `dinov2_features.py`, `feature_fusion.py`).
3. **Classification**: gradient-boosted trees and baselines evaluated
   with grouped cross-validation, plus a fine-tuned DINOv2/ViT head
   (`classification.py`, `ft_transformer.py`, `vit_finetune.py`,
   `vit_inference.py`).
4. **Explainability**: SHAP analysis (`explainability.py`).
5. **Reusability index**: per-particle provisional morphology-based score
   (`reusability.py`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git
```

Download the SAM2.1 Hiera-Large checkpoint (`sam2.1_hiera_large.pt`) from
the [SAM2 repository](https://github.com/facebookresearch/sam2) into
`checkpoints/`.

## Usage

```bash
python main.py \
  --input_dir SEM \
  --output_dir output \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --fov_width_um <field-of-view width in µm> \
  --device cpu
```

Run `python main.py --help` for all options. A desktop interface is
available with `python gui_app.py`.

Settings (class names, severity weights, segmentation thresholds and
reusability weights) are in `config.py`.

## Data

The SEM micrographs, labelled particle crops and trained models are not
included in this repository. They are available from the corresponding
author of the associated article upon reasonable request.

## Licence

MIT. See [LICENSE](LICENSE).
