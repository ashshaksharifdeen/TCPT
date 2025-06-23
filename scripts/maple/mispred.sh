#!/bin/bash

# Set required variables
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml"
DATASET_CFG="configs/datasets/eurosat.yaml"
MODEL_DIR="output/base2new/train_base/eurosat/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx"
EPOCH=5
OUTPUT_DIR="misprediction/eurosat_base"
SUBSAMPLE="base"

# Run the Python script
python mispred.py \
  --root ${ROOT} \
  --config-file ${CONFIG} \
  --dataset-config-file ${DATASET_CFG} \
  --model-dir ${MODEL_DIR} \
  --load-epoch ${EPOCH} \
  --output-dir ${OUTPUT_DIR} \
  --subsample-classes ${SUBSAMPLE}
