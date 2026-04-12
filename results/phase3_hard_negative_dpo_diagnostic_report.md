# Diagnostic Report

## Reliability Notes

- This diagnosis is heuristic and reproducible, but not omniscient.
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative bucket because the current files do not include all candidate object annotations.

## Overall Failure Counts

| type | count | ratio_over_failures |
|---|---:|---:|
| COORDINATE_IMPRECISION | 279 | 0.3991 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 194 | 0.2775 |
| SPATIAL_REASONING_ERROR | 118 | 0.1688 |
| ATTRIBUTE_OR_OTHER | 107 | 0.1531 |
| FORMAT_ERROR | 1 | 0.0014 |

## Per Dataset Counts

### rec_results_refcoco_val_outputs_phase3-hard-negative-dpo-qwen25vl-7b-local-rerun1.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 18 |
| COORDINATE_IMPRECISION | 84 |
| CORRECT | 1830 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 23 |
| SPATIAL_REASONING_ERROR | 45 |

### rec_results_refcocog_val_outputs_phase3-hard-negative-dpo-qwen25vl-7b-local-rerun1.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 43 |
| COORDINATE_IMPRECISION | 104 |
| CORRECT | 1754 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 58 |
| SPATIAL_REASONING_ERROR | 41 |

### rec_results_refcocop_val_outputs_phase3-hard-negative-dpo-qwen25vl-7b-local-rerun1.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 46 |
| COORDINATE_IMPRECISION | 91 |
| CORRECT | 1717 |
| FORMAT_ERROR | 1 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 113 |
| SPATIAL_REASONING_ERROR | 32 |

