#!/bin/bash
set -e

ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/CoOp/vit_b16.yaml"
DATASETS=( caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat )
SPLITS=( base new )
OUTPUT_BASE="reliability_zsclip"

python zero_shot_exp.py \
  --root        "${ROOT}" \
  --config-file "${CONFIG}" \
  --datasets    ${DATASETS[@]} \
  --splits      ${SPLITS[@]} \
  --output-base "${OUTPUT_BASE}"
