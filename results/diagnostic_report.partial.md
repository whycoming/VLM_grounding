# Diagnostic Report

## Reliability Notes

- This diagnosis is heuristic and reproducible, but not omniscient.
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative bucket because the current files do not include all candidate object annotations.

## Overall Failure Counts

| type | count | ratio_over_failures |
|---|---:|---:|
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 194 | 0.3374 |
| COORDINATE_IMPRECISION | 181 | 0.3148 |
| SPATIAL_REASONING_ERROR | 107 | 0.1861 |
| ATTRIBUTE_OR_OTHER | 93 | 0.1617 |

## Per Dataset Counts

### rec_results_refcoco_val_qwen25vl-3b.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 29 |
| COORDINATE_IMPRECISION | 83 |
| CORRECT | 1779 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 39 |
| SPATIAL_REASONING_ERROR | 70 |

### rec_results_refcocop_val_qwen25vl-3b.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 64 |
| COORDINATE_IMPRECISION | 98 |
| CORRECT | 1646 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 155 |
| SPATIAL_REASONING_ERROR | 37 |

