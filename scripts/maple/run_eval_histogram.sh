#!/bin/bash

# Set required variables
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml"
DATASET_CFG="configs/datasets/dtd.yaml"
MODEL_DIR="output/empty/base2new/train_base/fgvc_aircraft/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx"
LOAD_EPOCH=5
OUTPUT_DIR="histogram/histogram_empty_base"
SUBSAMPLE="base"

# Run the evaluation script with the specified arguments
python eval_and_plot_histogram.py \
  --root ${ROOT} \
  --config-file ${CONFIG} \
  --dataset-config-file ${DATASET_CFG} \
  --model-dir ${MODEL_DIR} \
  --load-epoch ${LOAD_EPOCH} \
  --output-dir ${OUTPUT_DIR} \
  --subsample-classes ${SUBSAMPLE}
