#!/bin/bash

# Base variables
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml"
EPOCH=5
SUBSAMPLE="new"  # Specify novel classes

# List of datasets and seeds
DATASETS=(caltech101)
SEEDS=(1 2 3)

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    DATASET_CFG="configs/datasets/${DATASET}.yaml"
    MODEL_DIR="output/base2new/train_base/${DATASET}/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed${SEED}"
    OUTPUT_DIR="eval_tsne_outputs_novel/${DATASET}_seed${SEED}"

    echo "========================================="
    echo "Evaluating Novel Classes for: ${DATASET} | Seed: ${SEED}"
    echo "Saving to: ${OUTPUT_DIR}"
    echo "========================================="

    python eval_and_visualize_prompts.py \
      --root ${ROOT} \
      --config-file ${CONFIG} \
      --dataset-config-file ${DATASET_CFG} \
      --model-dir ${MODEL_DIR} \
      --load-epoch ${EPOCH} \
      --output-dir ${OUTPUT_DIR} \
      --subsample-classes ${SUBSAMPLE}
  done
done
