#!/bin/bash
# run_margin_analysis.sh

set -e

# ─── 1) Paths & config ─────────────────────────────────────────────────────
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/CoOp/vit_b16.yaml"

# ─── 2) Datasets & splits ──────────────────────────────────────────────────
DATASETS=( caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat )
SPLITS=( base new )

# ─── 3) Output dir ──────────────────────────────────────────────────────────
OUTPUT_DIR="margin_analysis_results"
mkdir -p "${OUTPUT_DIR}"

# ─── 4) Run python script ───────────────────────────────────────────────────
python zero_shot_margin.py \
  --root        "${ROOT}" \
  --config-file "${CONFIG}" \
  --datasets    ${DATASETS[@]} \
  --splits      ${SPLITS[@]} \
  --output-dir  "${OUTPUT_DIR}"

echo "✅ Experiments complete. See figures in ${OUTPUT_DIR}/"
