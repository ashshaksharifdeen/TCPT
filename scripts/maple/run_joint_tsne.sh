#!/bin/bash

# Activate conda environment if needed
# source activate maple

# === Required arguments ===
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml"
DATASET_CFG="configs/datasets/eurosat.yaml"
MODEL_DIR="output/base2new/train_base/eurosat/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx"
EPOCH=5
OUTPUT_DIR="eval_tsne_outputs/joint_tsne_eurosat"
SUBSAMPLE="base"  # Use "base", "new", or "all"

# === Run the joint t-SNE visualization ===
python plot_joint_tsne.py \
  --root ${ROOT} \
  --config-file ${CONFIG} \
  --dataset-config-file ${DATASET_CFG} \
  --model-dir ${MODEL_DIR} \
  --load-epoch ${EPOCH} \
  --output-dir ${OUTPUT_DIR} \
  --subsample-classes ${SUBSAMPLE}
