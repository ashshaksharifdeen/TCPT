#!/bin/bash

#cd ../..

# custom config
DATA=/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset
TRAINER=ZeroshotCLIP
DATASET=caltech101
CFG=vit_b16 # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only