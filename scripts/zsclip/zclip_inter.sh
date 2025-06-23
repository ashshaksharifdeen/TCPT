# run_zero_inter_intra.sh
#!/bin/bash
set -e

ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/CoOp/vit_b16.yaml"
DATASETS=( caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat )
SPLITS=( base new )
OUTPUT_DIR="reliability_zsclip_summary"

mkdir -p "${OUTPUT_DIR}"

python zero_inter_intra.py \
  --root        "${ROOT}" \
  --config-file "${CONFIG}" \
  --datasets    ${DATASETS[@]} \
  --splits      ${SPLITS[@]} \
  --output-dir  "${OUTPUT_DIR}"

echo "✅ Done! Check ${OUTPUT_DIR}/ for plots."
