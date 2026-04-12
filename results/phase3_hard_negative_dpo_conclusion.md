# Phase 3 Hard Negative DPO Conclusion

## Setup

- Base checkpoint: `outputs/sft-qwen25vl-7b-local/checkpoint-4800`
- Training output: `outputs/phase3-hard-negative-dpo-qwen25vl-7b-local-rerun1`
- Evaluation output: `outputs/eval-phase3-hard-negative-dpo-1500`
- Evaluation splits: `refcoco_val`, `refcocop_val`, `refcocog_val`
- Samples per split: `2000`

## Result

| split | acc@0.5 | acc@0.75 | mean IoU |
|---|---:|---:|---:|
| refcoco_val | 0.9150 | 0.7985 | 0.8176 |
| refcocop_val | 0.8585 | 0.7715 | 0.7790 |
| refcocog_val | 0.8770 | 0.7575 | 0.7895 |

## Comparison To Phase 2 SFT

| split | acc@0.5 delta | acc@0.75 delta | mean IoU delta |
|---|---:|---:|---:|
| refcoco_val | -0.0055 | -0.0135 | -0.0088 |
| refcocop_val | -0.0030 | +0.0000 | -0.0014 |
| refcocog_val | +0.0010 | -0.0035 | -0.0023 |

Hard Negative DPO is very close to the Phase 2 SFT checkpoint, but it does not beat it overall. The best case is `refcocog_val` acc@0.5, which is slightly higher, while the main weakness is lower box quality on `refcoco_val`.

## Comparison To Phase 3 Online DPO

| split | acc@0.5 delta | acc@0.75 delta | mean IoU delta |
|---|---:|---:|---:|
| refcoco_val | +0.0055 | -0.0080 | +0.0016 |
| refcocop_val | +0.0120 | +0.0075 | +0.0084 |
| refcocog_val | +0.0085 | -0.0050 | +0.0028 |

Hard Negative DPO is stronger than Online DPO overall. It recovers most of the acc@0.5 regression, improves mean IoU on all three splits, and avoids the large format-stability regression seen in generic Online DPO.

## Diagnostic Readout

- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET`: `206 -> 194` vs Phase 2 SFT
- `COORDINATE_IMPRECISION`: `262 -> 279` vs Phase 2 SFT
- `FORMAT_ERROR`: `0 -> 1` vs Phase 2 SFT
- `FORMAT_ERROR`: `83 -> 1` vs Online DPO

The method is doing the intended job: it reduces the wrong-target proxy bucket without harming format stability. The tradeoff is that the residual error shifts toward `COORDINATE_IMPRECISION`, which now becomes even more clearly the limiting factor.

## Decision

- Keep Hard Negative DPO above Online DPO as the stronger DPO variant for grounding.
- Do not promote Hard Negative DPO over the Phase 2 SFT checkpoint as the current best model.
- Treat the result as evidence that ambiguity targeting helps, but the dominant remaining bottleneck is still box precision rather than object disambiguation.
- Move the next experiment priority to `GRPO` as the higher-cost control, or another method explicitly optimized for coordinate precision.
