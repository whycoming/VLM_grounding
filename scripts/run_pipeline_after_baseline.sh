#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="${BASELINE_DIR:-/mnt/VLM_grounding/outputs/eval-baseline-3b}"
RESULTS_DIR="${RESULTS_DIR:-/mnt/VLM_grounding/results}"
LOG_DIR="${LOG_DIR:-/mnt/VLM_grounding/logs}"
PHASE2_LOG="${LOG_DIR}/phase2_sft.log"

mkdir -p "${LOG_DIR}"

while true; do
  count=$(find "${BASELINE_DIR}" -maxdepth 1 -type f -name 'rec_results_*.json' | wc -l)
  if [ "${count}" -ge 3 ]; then
    break
  fi
  sleep 60
done

cd /mnt/VLM_grounding
./scripts/run_phase1_analysis.sh

source /mnt/VLM_grounding/.venv/bin/activate
python /mnt/VLM_grounding/scripts/write_phase1_conclusion.py \
  --summary-json /mnt/VLM_grounding/results/baseline_summary.json \
  --diagnostic-json /mnt/VLM_grounding/results/diagnostics/diagnostic_report.json \
  --output-md /mnt/VLM_grounding/results/phase1_conclusion.md

./scripts/run_sft_phase2_local.sh 2>&1 | tee "${PHASE2_LOG}"
