# Phase 1 Conclusion

## Reliability Boundary

- This conclusion is based on reproducible result files and deterministic post-processing scripts.
- It is still a heuristic diagnosis, not an oracle. In particular, `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative proxy bucket.

## Baseline

- Mean `acc@0.5` across evaluated splits: `0.8592`
- Mean `mean_iou` across evaluated splits: `0.7940`

## Main Finding

- The dominant residual failure bucket is `COORDINATE_IMPRECISION` with count `272`.
- `FORMAT_ERROR` failure ratio: `0.0000`
- `COORDINATE_IMPRECISION` failure ratio: `0.3219`
- `SPATIAL_REASONING_ERROR` failure ratio: `0.1858`
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` failure ratio: `0.3112`

## Decision

- Proceed to Phase 2 SFT baseline. This is justified regardless of the final Phase 3 method choice, because SFT is the controlled next-stage baseline in the plan.
- Do not over-claim which RL method should win yet. That decision should wait until the Phase 2 residual-error profile is available.

## Next Step

- Start Phase 2 local SFT and preserve logs/checkpoints for a second-round diagnosis.
