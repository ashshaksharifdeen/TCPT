#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION (tweak these)
# -----------------------------------------------------------------------------
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=5             # epoch to load for evaluation
SUB=new              # DATASET.SUBSAMPLE_CLASSES for test

# the hyperparams you swept when training
ALPHAS=(0.1 0.5 1.0 5.0 10.0)
BETAS=(0.01)

# which dataset(s) to evaluate
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)

# which random seeds to evaluate
SEEDS=(1 2 3)
# -----------------------------------------------------------------------------

for ALPHA in "${ALPHAS[@]}"; do
  for BETA in "${BETAS[@]}"; do

    echo
    echo "===== Testing models with alpha=${ALPHA}, beta=${BETA} ====="
    echo

    for DATASET in "${DATASETS[@]}"; do
      for SEED in "${SEEDS[@]}"; do

        # match your training output structure:
        # output/base2new/train_base/<DATASET>/shots_<SHOTS>/<TRAINER>/<CFG>/
        #   alpha<ALPHA>_beta<BETA>/seed<SEED>
        COMMON_DIR="${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/alpha${ALPHA}_beta${BETA}/seed${SEED}"
        MODEL_DIR="output/base2new/train_base/${COMMON_DIR}"
        DIR="output/base2new/test_${SUB}/${COMMON_DIR}"

        echo "---------------------------------------------"
        echo "Evaluating ${DATASET} | seed ${SEED}"
        echo "  model dir: ${MODEL_DIR}"
        echo "  output dir: ${DIR}"
        echo "---------------------------------------------"

        python train.py \
          --root       "${DATA}" \
          --seed       "${SEED}" \
          --trainer    "${TRAINER}" \
          --dataset-config-file "configs/datasets/${DATASET}.yaml" \
          --config-file         "configs/trainers/${TRAINER}/${CFG}.yaml" \
          --output-dir "${DIR}" \
          --model-dir  "${MODEL_DIR}" \
          --load-epoch "${LOADEP}" \
          --eval-only \
          DATASET.NUM_SHOTS         "${SHOTS}" \
          DATASET.SUBSAMPLE_CLASSES "${SUB}"
      done
    done

  done
done
