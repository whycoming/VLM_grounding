# Local Training Notes

## No-wandb setup

This workspace is configured to log locally instead of using Weights & Biases.

- SFT config: `/mnt/VLM_grounding/configs/qwen25vl_7b_sft_local.yaml`
- REC data config: `/mnt/VLM_grounding/configs/rec_local.yaml`
- GRPO launch script: `/mnt/VLM_grounding/scripts/run_grpo_rec_local.sh`

## Current logging behavior

- Trainer metrics go to stdout and the local output directory.
- GRPO logs are tee'd into `/mnt/VLM_grounding/logs/<run_name>.log`.
- Model checkpoints are written under `/mnt/VLM_grounding/outputs/`.

## Notes

- The GRPO script uses `--report_to none`.
- The GRPO script uses `--attn_implementation sdpa` because `flash-attn` is not installed yet.
- REC annotation paths already point to the local extracted JSONL files.
- Ensure `/mnt/VLM_grounding/data/coco/train2014` exists before starting training.
