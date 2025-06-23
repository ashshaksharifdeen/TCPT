#!/bin/bash
# run_prompt_calibration.sh
# Usage: ./run_prompt_calibration.sh <dataset> <seed>

# Base data path and trainer information
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16

DATASET=eurosat
SEED=1

# Define the output directory following your naming convention.
OUTPUT_DIR="output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}"

echo "Starting prompt calibration experiment for dataset: ${DATASET}, seed: ${SEED}"
echo "Training output directory: ${OUTPUT_DIR}"

python equvarience_analyse.py \
    --root "${DATA}" \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${OUTPUT_DIR}" \
    --opts DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base

echo "Prompt calibration experiment finished."
