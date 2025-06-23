#!/bin/bash

# List of datasets to process
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)

# Common settings
SHOTS=16
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx

# Sweep values for your regularizer
ALPHAS=(0.1 0.5 1.0 5.0 10.0)
BETAS=(0.01)

# Timestamped logfile name
TIMESTAMP=$(date +%F_%H-%M-%S)
LOGFILE=parse_results_${TIMESTAMP}.txt

echo "Logging all results to $LOGFILE"
echo "========== ALL RESULTS ($TIMESTAMP) ==========" > "$LOGFILE"
echo "" >> "$LOGFILE"

for DATASET in "${DATASETS[@]}"; do
  echo "===== Dataset: ${DATASET} =====" | tee -a "$LOGFILE"

  for ALPHA in "${ALPHAS[@]}"; do
    for BETA in "${BETAS[@]}"; do

      echo "-- α=${ALPHA}, β=${BETA} (base classes) --" | tee -a "$LOGFILE"
      BASE_DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/alpha${ALPHA}_beta${BETA}

      python parse_test_res.py \
        "${BASE_DIR}" \
        --multi-exp \
        | tee -a "$LOGFILE"

      echo "-- α=${ALPHA}, β=${BETA} (novel classes) --" | tee -a "$LOGFILE"
      NOVEL_DIR=output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/alpha${ALPHA}_beta${BETA}

      python parse_test_res_alp_beta.py \
        "${NOVEL_DIR}" \
        --test-log \
        --multi-exp \
        | tee -a "$LOGFILE"

      echo "" >> "$LOGFILE"

    done
  done

  echo "" >> "$LOGFILE"
done
