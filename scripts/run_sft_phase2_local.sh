#!/usr/bin/env bash
set -euo pipefail

source /mnt/VLM_grounding/.venv/bin/activate

cd /mnt/VLM_grounding/VLM-R1/src/open-r1-multimodal

export PYTHONPATH="/mnt/VLM_grounding/VLM-R1/src/open-r1-multimodal/src:${PYTHONPATH:-}"

python -m open_r1.sft \
  --model_name_or_path /mnt/VLM_grounding/models/qwen25vl-7b \
  --dtype bfloat16 \
  --dataset_name /mnt/VLM_grounding/configs/rec_local.yaml \
  --image_root /mnt/VLM_grounding/data/coco \
  --output_dir /mnt/VLM_grounding/outputs/sft-qwen25vl-7b-local \
  --do_train true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --logging_strategy steps \
  --logging_steps 5 \
  --save_strategy steps \
  --save_steps 100 \
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
