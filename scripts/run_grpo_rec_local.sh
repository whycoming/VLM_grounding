#!/usr/bin/env bash
set -euo pipefail

REPO_HOME="/mnt/VLM_grounding/VLM-R1"
RUN_NAME="${RUN_NAME:-Qwen2.5-VL-7B-GRPO-REC-lora-local}"
MODEL_PATH="${MODEL_PATH:-/mnt/VLM_grounding/models/qwen25vl-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/VLM_grounding/data/coco/train2014}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/VLM_grounding/outputs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-/mnt/VLM_grounding/logs}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${REPO_HOME}/src/open-r1-multimodal"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name /mnt/VLM_grounding/configs/rec_local.yaml \
    --image_root "${IMAGE_ROOT}" \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 2 \
    --run_name "${RUN_NAME}" \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    2>&1 | tee "${LOG_PATH}"
