#!/bin/bash

# Activate your conda environment if needed
# source activate maple
#output/base2new/test_new/caltech101/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx
#output/base2new/train_base/caltech101/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx
# Set required variables
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml"
DATASET_CFG="configs/datasets/dtd.yaml"
MODEL_DIR="output/base2new/train_base/dtd/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx"
EPOCH=5
OUTPUT_DIR="eval_tsne_outputs/dtd_novel"
#if it is novel >> new
SUBSAMPLE="new"

# Run the evaluation script
python eval_and_visualize_prompts.py \
  --root ${ROOT} \
  --config-file ${CONFIG} \
  --dataset-config-file ${DATASET_CFG} \
  --model-dir ${MODEL_DIR} \
  --load-epoch ${EPOCH} \
  --output-dir ${OUTPUT_DIR} \
  --subsample-classes ${SUBSAMPLE}
