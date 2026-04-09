# Phase 3 RFT Conclusion

## Reliability Boundary

- This conclusion is based on reproducible evaluation files and deterministic post-processing scripts.
- The failure diagnosis remains heuristic. In particular, `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is still a conservative proxy bucket.
- The training loop reached `global_step=3000` and produced a usable final checkpoint, but the original `sft.py` had a post-training metrics bug. The checkpoint itself is valid.
- **Scheduler caveat**: this RFT run was launched via `run_rft_phase3_local.sh`, which (like the Phase 2 launcher, now fixed) did not pass `--lr_scheduler_type` / `--warmup_ratio`. Training used the trainer default `linear` scheduler with `warmup_ratio=0`. The scheduler does not explain this negative result — the regression mechanism (self-filtered easy samples + source coverage limited to `refcoco_train`, causing cross-split generalization loss) is orthogonal to the scheduler. The decision to de-prioritize the current RFT configuration stands, but when RFT is revisited it should be rerun under the corrected schedule from a corrected Phase 2 base checkpoint to keep the comparison clean.

## Phase 3 RFT Result

- Mean `acc@0.5` across evaluated splits: `0.8730`
- Mean `acc@0.75` across evaluated splits: `0.7590`
- Mean `mean_iou` across evaluated splits: `0.7742`

## Delta vs Phase 2 Base Checkpoint

- `refcoco_val`: `acc@0.5` `0.9205 -> 0.9060` (`-0.0145`)
- `refcocop_val`: `acc@0.5` `0.8615 -> 0.8485` (`-0.0130`)
- `refcocog_val`: `acc@0.5` `0.8760 -> 0.8645` (`-0.0115`)
- Mean `acc@0.5`: `0.8860 -> 0.8730` (`-0.0130`)
- Mean `acc@0.75`: `0.7815 -> 0.7590` (`-0.0225`)
- Mean `mean_iou`: `0.7995 -> 0.7742` (`-0.0253`)

## Main Finding

- This RFT run did not improve over the Phase 2 early-stop SFT checkpoint.
- The main target bucket `COORDINATE_IMPRECISION` only changed from `262` to `259`, which is too small to justify the method.
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` worsened from `206` to `224`.
- `SPATIAL_REASONING_ERROR` worsened from `109` to `137`.
- `ATTRIBUTE_OR_OTHER` worsened from `107` to `134`.
- New `FORMAT_ERROR` cases appeared (`8`), which were absent in the Phase 2 checkpoint diagnosis.

## Interpretation

- The current RFT pipeline appears to overfit to self-generated high-IoU easy cases rather than improving robust localization.
- The generated training set improved box imitation on accepted samples, but it did not translate into better validation generalization.
- Using only `refcoco_train` for this RFT round likely reduced cross-split robustness, especially for `refcoco+` and `refcocog`.

## Decision

- Do not promote this RFT checkpoint over the Phase 2 early-stop checkpoint.
- Keep `outputs/sft-qwen25vl-7b-local/checkpoint-4800` as the best current model.
- Record this RFT run as a negative result: the current self-filtered RFT configuration is not the right next method for this project.

## Next Step

- De-prioritize RFT in the Phase 3 queue.
- Move the next experiment priority to `Online DPO` or `Hard Negative DPO`.
- If RFT is revisited later, broaden the source data and redesign the filtering strategy rather than simply scaling this configuration.
