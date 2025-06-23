#!/bin/bash
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base data path and trainer
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=KgCoOp
CFG=vit_b16_ep100_ctxv1
SHOTS=4
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

# List of datasets to loop over
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)

# List of seeds to loop over
SEEDS=(1 2 3)

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    # Loop through each seed
    for SEED in "${SEEDS[@]}"; do
        DIR=/storagepool/Ashshak/output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
        else
            echo "Running training for dataset: ${DATASET}, seed: ${SEED}"
        fi

        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
    done
done
