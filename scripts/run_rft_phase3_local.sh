#!/usr/bin/env bash
set -euo pipefail

source /mnt/VLM_grounding/.venv/bin/activate

BASE_ADAPTER_PATH="${BASE_ADAPTER_PATH:-/mnt/VLM_grounding/outputs/sft-qwen25vl-7b-local/checkpoint-4800}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-/mnt/VLM_grounding/models/qwen25vl-7b-phase2-merged-4800}"
GENERATION_MODEL_PATH="${GENERATION_MODEL_PATH:-${MERGED_MODEL_PATH}}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/VLM_grounding/data/rec_jsons_processed/rec_jsons_processed/refcoco_train.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/VLM_grounding/data/coco}"
RFT_DATA_DIR="${RFT_DATA_DIR:-/mnt/VLM_grounding/outputs/phase3-rft-data}"
RFT_JSONL="${RFT_JSONL:-${RFT_DATA_DIR}/rft_iter1_refcoco_train.jsonl}"
RFT_STATS="${RFT_STATS:-${RFT_DATA_DIR}/rft_iter1_refcoco_train_stats.json}"
RFT_YAML="${RFT_YAML:-/mnt/VLM_grounding/configs/rft_phase3_local.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/VLM_grounding/outputs/phase3-rft-qwen25vl-7b-local}"
MAX_SOURCE_SAMPLES="${MAX_SOURCE_SAMPLES:-4000}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
MIN_IOU="${MIN_IOU:-0.8}"
TOP_K_PER_PROMPT="${TOP_K_PER_PROMPT:-2}"
MAX_STEPS="${MAX_STEPS:-3000}"

mkdir -p "${RFT_DATA_DIR}"

if [ ! -f "${MERGED_MODEL_PATH}/config.json" ]; then
  python /mnt/VLM_grounding/scripts/export_merged_qwen25vl_lora.py \
    --adapter-path "${BASE_ADAPTER_PATH}" \
    --output-dir "${MERGED_MODEL_PATH}"
fi

python /mnt/VLM_grounding/scripts/generate_rft_dataset.py \
  --model-path "${GENERATION_MODEL_PATH}" \
  --input-jsonl "${TRAIN_JSONL}" \
  --image-root "${IMAGE_ROOT}" \
  --output-jsonl "${RFT_JSONL}" \
  --output-stats "${RFT_STATS}" \
  --max-samples "${MAX_SOURCE_SAMPLES}" \
  --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
  --min-iou "${MIN_IOU}" \
  --top-k-per-prompt "${TOP_K_PER_PROMPT}"

if [ ! -s "${RFT_JSONL}" ]; then
  echo "RFT dataset is empty: ${RFT_JSONL}" >&2
  exit 1
fi

printf 'datasets:\n  - json_path: %s\n' "${RFT_JSONL}" > "${RFT_YAML}"

cd /mnt/VLM_grounding/VLM-R1/src/open-r1-multimodal
export PYTHONPATH="/mnt/VLM_grounding/VLM-R1/src/open-r1-multimodal/src:${PYTHONPATH:-}"

python -m open_r1.sft \
  --model_name_or_path "${MERGED_MODEL_PATH}" \
  --dtype bfloat16 \
  --dataset_name "${RFT_YAML}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --do_train true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --learning_rate 1e-5 \
  --logging_strategy steps \
  --logging_steps 5 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --eval_strategy no \
  --gradient_checkpointing true \
  --bf16 true \
  --max_length 4096 \
  --packing false \
  --report_to none \
  --use_peft true \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_task_type CAUSAL_LM \
  --attn_implementation sdpa \
  --seed 42 \
  --data_seed 42
