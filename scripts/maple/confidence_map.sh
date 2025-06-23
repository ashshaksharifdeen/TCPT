#!/bin/bash

ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml"
DATASET_CFG="configs/datasets/food101.yaml"
MODEL_DIR="output/base2new/train_base/food101/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx"
LOAD_EPOCH=5
OUTPUT_DIR="eval_confidence_map_outputs/food101_new"
SUBSAMPLE="new"

python confidence_map.py \
  --root ${ROOT} \
  --config-file ${CONFIG} \
  --dataset-config-file ${DATASET_CFG} \
  --model-dir ${MODEL_DIR} \
  --load-epoch ${LOAD_EPOCH} \
  --output-dir ${OUTPUT_DIR} \
  --subsample-classes ${SUBSAMPLE}
