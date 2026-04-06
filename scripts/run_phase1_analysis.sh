#!/usr/bin/env bash
set -euo pipefail

source /mnt/VLM_grounding/.venv/bin/activate

INPUT_DIR="${INPUT_DIR:-/mnt/VLM_grounding/outputs/eval-baseline-3b}"

python /mnt/VLM_grounding/scripts/summarize_rec_results.py \
  --input-dir "${INPUT_DIR}" \
  --output-json /mnt/VLM_grounding/results/baseline_summary.json \
  --output-md /mnt/VLM_grounding/results/baseline_summary.md

python /mnt/VLM_grounding/scripts/diagnose_rec_failures.py \
  --input-dir "${INPUT_DIR}" \
  --output-json /mnt/VLM_grounding/results/diagnostics/diagnostic_report.json \
  --output-md /mnt/VLM_grounding/results/diagnostic_report.md
