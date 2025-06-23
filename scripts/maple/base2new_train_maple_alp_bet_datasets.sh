#!/usr/bin/env bash
set -euo pipefail

# Base configuration
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER="MaPLe"
CFG="vit_b16_c2_ep5_batch4_2ctx"
SHOTS=16

# What to loop over
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)
SEEDS=(1 2 3)
ALPHAS=(0.1 0.5 1.0 5.0 10.0)
BETAS=(0.01)

for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      for SEED in "${SEEDS[@]}"; do

        # output dir includes the hyperparameters
        DIR="output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/alpha${alpha}_beta${beta}/seed${SEED}"

        if [ -d "$DIR" ]; then
          echo "[SKIP] ${DATASET} seed=${SEED} α=${alpha} β=${beta} → already in ${DIR}"
          continue
        fi

        echo "[RUN ] ${DATASET} seed=${SEED} α=${alpha} β=${beta}"
        mkdir -p "$DIR"

        python train.py \
          --root "${DATA}" \
          --seed "${SEED}" \
          --trainer "${TRAINER}" \
          --dataset-config-file "configs/datasets/${DATASET}.yaml" \
          --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
          --output-dir "${DIR}" \
          DATASET.NUM_SHOTS "${SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES base \
          TRAINER.MAPLE.MARGIN_ALPHA "${alpha}" \
          TRAINER.MAPLE.MARGIN_BETA "${beta}"

      done
    done
  done
done
