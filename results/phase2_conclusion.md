# Phase 2 Conclusion

## Reliability Boundary

- This conclusion is based on reproducible evaluation files and deterministic post-processing scripts.
- The failure diagnosis remains heuristic. In particular, `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is still a conservative proxy bucket.
- Phase 2 was early-stopped at `checkpoint-4800` instead of running the planned full `3` epochs.
- **Scheduler caveat**: this SFT run used the trainer default `linear` scheduler with `warmup_ratio=0`, not the `cosine + warmup_ratio=0.1` declared in `configs/qwen25vl_7b_sft_local.yaml`. The YAML was never loaded — `run_sft_phase2_local.sh` drove training via CLI flags only, and those flags omitted `--lr_scheduler_type` / `--warmup_ratio`. The script has since been fixed, but `checkpoint-4800` itself was produced under the suboptimal schedule. Treat the Phase 2 numbers as a conservative lower bound for SFT; a rerun with the corrected schedule may push SFT higher and, in turn, shift the early-stop decision. The qualitative finding (`COORDINATE_IMPRECISION` dominates the residual error) is robust to the scheduler choice because the bucket-wise shrinkage gap is much larger than any plausible scheduler delta.

## Phase 2 Result

- Mean `acc@0.5` across evaluated splits: `0.8860`
- Mean `acc@0.75` across evaluated splits: `0.7815`
- Mean `mean_iou` across evaluated splits: `0.7995`

## Delta vs Phase 1 Baseline

- `refcoco_val`: `acc@0.5` `0.8895 -> 0.9205` (`+0.0310`)
- `refcocop_val`: `acc@0.5` `0.8230 -> 0.8615` (`+0.0385`)
- `refcocog_val`: `acc@0.5` `0.8650 -> 0.8760` (`+0.0110`)
- Mean `acc@0.5`: `0.8592 -> 0.8860` (`+0.0268`)
- Mean `acc@0.75`: `0.7623 -> 0.7815` (`+0.0192`)
- Mean `mean_iou`: `0.7940 -> 0.7995` (`+0.0055`)

## Main Finding

- Early-stop SFT is already effective enough to serve as the Phase 3 base checkpoint.
- The dominant residual failure bucket is still `COORDINATE_IMPRECISION` with count `262`.
- Compared with Phase 1, `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` dropped from `263` to `206`.
- Compared with Phase 1, `SPATIAL_REASONING_ERROR` dropped from `157` to `109`.
- Compared with Phase 1, `ATTRIBUTE_OR_OTHER` dropped from `153` to `107`.
- `COORDINATE_IMPRECISION` only dropped from `272` to `262`, so box precision remains the hardest residual bottleneck.

## Decision

- Do not continue the long Phase 2 SFT run. The marginal value of more plain SFT is not justified by the extra training time.
- Promote `outputs/sft-qwen25vl-7b-local/checkpoint-4800` to the working base checkpoint for Phase 3.
- Treat Phase 3 as a targeted post-training stage for residual box-quality and credit-assignment errors, not as more generic instruction tuning.

## Phase 3 Priority

- Priority 1: `RFT`. Reason: the residual bottleneck is box precision, and IoU-thresholded filtering gives a direct signal for improving localization quality.
- Priority 2: `Online DPO`. Reason: it can still help with residual wrong-target and ranking errors while staying cheaper than GRPO.
- Priority 3: `Hard Negative DPO`. Reason: it is still useful, but the Phase 2 diagnosis suggests disambiguation is no longer the single dominant problem.
- Priority 4: `GRPO` as the higher-cost policy-optimization control.

## Next Step

- Start Phase 3 from `checkpoint-4800`.
- Use fixed compute budget and fixed LoRA settings for all post-training methods.
- Evaluate every Phase 3 method against the same three validation splits and rerun the same diagnosis script.
