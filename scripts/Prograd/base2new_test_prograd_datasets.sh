#!/bin/bash
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base config
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=ProGrad
CFG=rn50_ep100
CTP=end 
NCTX=16
SHOTS=4
LOADEP=100
CSC=False
SUB=new
LAMBDA=1.0
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
# List of datasets and seeds
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)
SEEDS=(1 2 3)

# Loop through datasets and seeds
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=/storagepool/Ashshak/output/base2new/train_base/${COMMON_DIR}
        DIR=/storagepool/Ashshak/output/base2new/test_${SUB}/${COMMON_DIR}

        echo "---------------------------------------------"
        echo "Evaluating ${DATASET} | Seed ${SEED}"
        echo "Output dir: ${DIR}"
        echo "Model dir : ${MODEL_DIR}"
        echo "---------------------------------------------"

        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            LOSS.LAMBDA ${LAMBDA} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}
    done
done
