# Reproducibility

## Phase 1

1. Run zero-shot baseline:

```bash
cd /mnt/VLM_grounding
source .venv/bin/activate
NUM_SAMPLES=2000 BATCH_SIZE=2 MODEL_PATH=/mnt/VLM_grounding/models/qwen25vl-3b OUTPUT_DIR=/mnt/VLM_grounding/outputs/eval-baseline-3b ./scripts/run_rec_baseline_local.sh
```

2. Summarize and diagnose:

```bash
cd /mnt/VLM_grounding
./scripts/run_phase1_analysis.sh
```

Outputs:

- `/mnt/VLM_grounding/results/baseline_summary.json`
- `/mnt/VLM_grounding/results/baseline_summary.md`
- `/mnt/VLM_grounding/results/diagnostics/diagnostic_report.json`
- `/mnt/VLM_grounding/results/diagnostic_report.md`

## Phase 2

Single-machine local SFT entry:

```bash
cd /mnt/VLM_grounding
./scripts/run_sft_phase2_local.sh
```

Current assumption:

- Phase 2 starts only after Phase 1 baseline and diagnosis are reviewed.
- The local SFT entry uses explicit CLI flags instead of only relying on config parsing.
- The current `VLM-R1` SFT entrypoint does not expose a direct `freeze_vision_modules` switch like the GRPO REC path does, so this deviation must be documented alongside any Phase 2 result.
