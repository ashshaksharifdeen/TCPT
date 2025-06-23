#!/bin/bash
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
# List of datasets to process
DATASETS=(dtd food101 eurosat)

# Common settings
SHOTS=16
TRAINER=CoOp
CFG=rn50_ep50

# Timestamped logfile name
TIMESTAMP=$(date +%F_%H-%M-%S)
LOGFILE=parse_results_${TIMESTAMP}.txt

# Start logging
echo "Logging all results to $LOGFILE"
echo "========== ALL RESULTS ($TIMESTAMP) ==========" > $LOGFILE
echo "" >> $LOGFILE

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Parsing results for dataset: ${DATASET}" | tee -a $LOGFILE
    echo "--- Base classes ---" | tee -a $LOGFILE

    python parse_test_res.py output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG} \
        | tee -a $LOGFILE

    echo "--- Novel classes ---" | tee -a $LOGFILE

    python parse_test_res.py output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG} --test-log \
        | tee -a $LOGFILE

    echo "-----------------------------" | tee -a $LOGFILE
    echo "" >> $LOGFILE
done
