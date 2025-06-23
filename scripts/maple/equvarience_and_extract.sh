#!/bin/bash
# train_and_extract.sh
#
# Usage:
#   ./train_and_extract.sh [<DATASET> <SEED> <comma_separated_list_of_prompts>]
#
# If arguments are not provided, default values will be used.
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [ "$#" -lt 3 ]; then
    echo "Insufficient arguments provided. Using default values."
    DATASET="oxford_flowers"
    SEED=1
    PROMPTS_LIST="a photo of a, an image of a, a picture of a, a realistic photo of a, a close-up shot of a, a blurry image of a, a stylized painting of a, a digital illustration of a, a sketch of a, a cartoon of a, a high-resolution photo of a, a vintage photograph of a, a futuristic rendering of a, a low-light shot of a, a scenic image of a, a dramatic portrait of a, a candid snapshot of a, a dynamic angle of a, a serene setting with a, a surreal depiction of a"
else
    DATASET=$1
    SEED=$2
    PROMPTS_LIST=$3
fi

# Fixed settings (update these if necessary)
SHOTS=16
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"

# Split the comma-separated list into an array.
IFS=',' read -ra PROMPTS <<< "$PROMPTS_LIST"

# Loop over each prompt in the list.
for prompt in "${PROMPTS[@]}"; do
  # Trim white space from the prompt.
  prompt_trim=$(echo "$prompt" | xargs)
  
  # Create a safe directory name by replacing spaces and removing non-alphanumeric characters.
  safe_prompt=$(echo "$prompt_trim" | tr ' ' '_' | tr -cd '[:alnum:]_')
  
  # Define the output directory; appending the safe prompt string ensures results for different prompts are stored separately.
  DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_${safe_prompt}/seed${SEED}
  
  # If the directory exists, notify that training will resume.
  if [ -d "$DIR" ]; then
      echo "Results are available in ${DIR}. Resuming training for prompt: \"$prompt_trim\""
  else
      echo "Training with prompt: \"$prompt_trim\""
  fi
  
  # Run the training command.
  python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.MAPLE.CTX_INIT "$prompt_trim"
  
  # After training, extract the learned prompt parameters.
  echo "Extracting trained prompt values for prompt: \"$prompt_trim\""
  # Adjust the path if extract_trained_prompt.py is not in the current directory.
  python prompteval.py --model-dir ${DIR} --prompt-name "prompt_learner.ctx"
  
  echo "-------------------------------------"
done
