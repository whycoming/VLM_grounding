#!/usr/bin/env bash
set -euo pipefail

source /mnt/VLM_grounding/.venv/bin/activate

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-/mnt/VLM_grounding/outputs/sft-qwen25vl-7b-local}"
if [ -z "${BASE_ADAPTER_PATH:-}" ]; then
  LATEST_CKPT=$(ls -d "${SFT_OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
    | awk -F- '{print $NF" "$0}' | sort -n | tail -n1 | awk '{print $2}')
  if [ -z "${LATEST_CKPT}" ]; then
    echo "No checkpoint-* found under ${SFT_OUTPUT_DIR}" >&2
    exit 1
  fi
  BASE_ADAPTER_PATH="${LATEST_CKPT}"
fi

CKPT_TAG="$(basename "${BASE_ADAPTER_PATH}" | sed 's/checkpoint-//')"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-/mnt/VLM_grounding/models/qwen25vl-7b-phase2-merged-${CKPT_TAG}}"
TRAIN_JSON="${TRAIN_JSON:-/mnt/VLM_grounding/data/rec_jsons_processed/rec_jsons_processed/refcoco_train.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/VLM_grounding/data/coco}"
HARD_NEG_DPO_DATA_DIR="${HARD_NEG_DPO_DATA_DIR:-/mnt/VLM_grounding/outputs/phase3-hard-negative-dpo-data}"
HARD_NEG_DPO_JSONL="${HARD_NEG_DPO_JSONL:-${HARD_NEG_DPO_DATA_DIR}/hard_negative_dpo_refcoco_train.jsonl}"
HARD_NEG_DPO_STATS="${HARD_NEG_DPO_STATS:-${HARD_NEG_DPO_DATA_DIR}/hard_negative_dpo_refcoco_train_stats.json}"
HARD_NEG_DPO_YAML="${HARD_NEG_DPO_YAML:-${HARD_NEG_DPO_DATA_DIR}/hard_negative_dpo_refcoco_train.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/VLM_grounding/outputs/phase3-hard-negative-dpo-qwen25vl-7b-local}"

MAX_SOURCE_SAMPLES="${MAX_SOURCE_SAMPLES:-0}"
MAX_NEGATIVES_PER_PROMPT="${MAX_NEGATIVES_PER_PROMPT:-1}"
MIN_NEGATIVE_IOU="${MIN_NEGATIVE_IOU:-0.0}"
MAX_NEGATIVE_IOU="${MAX_NEGATIVE_IOU:-0.3}"
MAX_STEPS="${MAX_STEPS:-1500}"
BETA="${BETA:-0.1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"

mkdir -p "${HARD_NEG_DPO_DATA_DIR}"

if [ ! -f "${MERGED_MODEL_PATH}/config.json" ]; then
  python /mnt/VLM_grounding/scripts/export_merged_qwen25vl_lora.py \
    --adapter-path "${BASE_ADAPTER_PATH}" \
    --output-dir "${MERGED_MODEL_PATH}"
fi

python /mnt/VLM_grounding/scripts/generate_hard_negative_dpo_dataset.py \
  --input-json "${TRAIN_JSON}" \
  --output-jsonl "${HARD_NEG_DPO_JSONL}" \
  --output-stats "${HARD_NEG_DPO_STATS}" \
  --max-samples "${MAX_SOURCE_SAMPLES}" \
  --max-negatives-per-prompt "${MAX_NEGATIVES_PER_PROMPT}" \
  --min-negative-iou "${MIN_NEGATIVE_IOU}" \
  --max-negative-iou "${MAX_NEGATIVE_IOU}"

if [ ! -s "${HARD_NEG_DPO_JSONL}" ]; then
  echo "Hard Negative DPO dataset is empty: ${HARD_NEG_DPO_JSONL}" >&2
  exit 1
fi

printf 'datasets:\n  - json_path: %s\n' "${HARD_NEG_DPO_JSONL}" > "${HARD_NEG_DPO_YAML}"

cd /mnt/VLM_grounding/VLM-R1/src/open-r1-multimodal
export PYTHONPATH="/mnt/VLM_grounding/VLM-R1/src/open-r1-multimodal/src:${PYTHONPATH:-}"

python -m open_r1.online_dpo \
  --model_name_or_path "${MERGED_MODEL_PATH}" \
  --dtype bfloat16 \
  --dataset_name "${HARD_NEG_DPO_YAML}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --do_train true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --learning_rate 5e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --logging_strategy steps \
  --logging_steps 5 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 2 \
  --eval_strategy no \
  --gradient_checkpointing true \
  --bf16 true \
  --max_length 4096 \
  --max_prompt_length 3072 \
  --max_completion_length 512 \
  --packing false \
  --report_to none \
  --beta "${BETA}" \
  --load_in_4bit "${LOAD_IN_4BIT}" \
  --use_peft true \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_task_type CAUSAL_LM \
  --freeze_vision_modules true \
  --attn_implementation sdpa \
  --seed 42 \
  --data_seed 42
