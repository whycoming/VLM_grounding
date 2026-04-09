#!/usr/bin/env bash
set -euo pipefail

source /mnt/VLM_grounding/.venv/bin/activate

MODEL_PATH="${MODEL_PATH:-/mnt/VLM_grounding/models/qwen25vl-3b}"
DATA_ROOT="${DATA_ROOT:-/mnt/VLM_grounding/data/rec_jsons_processed/rec_jsons_processed}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/VLM_grounding/data/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/VLM_grounding/outputs/eval-baseline-3b}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"

python /mnt/VLM_grounding/scripts/eval_rec_baseline_local.py \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --image-root "${IMAGE_ROOT}" \
  --datasets refcoco_val refcocop_val refcocog_val \
  --output-dir "${OUTPUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}"
