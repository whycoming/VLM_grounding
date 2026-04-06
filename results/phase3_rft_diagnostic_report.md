# Diagnostic Report

## Reliability Notes

- This diagnosis is heuristic and reproducible, but not omniscient.
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative bucket because the current files do not include all candidate object annotations.

## Overall Failure Counts

| type | count | ratio_over_failures |
|---|---:|---:|
| COORDINATE_IMPRECISION | 259 | 0.3399 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 224 | 0.2940 |
| SPATIAL_REASONING_ERROR | 137 | 0.1798 |
| ATTRIBUTE_OR_OTHER | 134 | 0.1759 |
| FORMAT_ERROR | 8 | 0.0105 |

## Per Dataset Counts

### rec_results_refcoco_val_phase3-rft-qwen25vl-7b-local_checkpoint-3000.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 20 |
| COORDINATE_IMPRECISION | 80 |
| CORRECT | 1812 |
| FORMAT_ERROR | 3 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 24 |
| SPATIAL_REASONING_ERROR | 61 |

### rec_results_refcocog_val_phase3-rft-qwen25vl-7b-local_checkpoint-3000.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 55 |
| COORDINATE_IMPRECISION | 104 |
| CORRECT | 1729 |
| FORMAT_ERROR | 3 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 66 |
| SPATIAL_REASONING_ERROR | 43 |

### rec_results_refcocop_val_phase3-rft-qwen25vl-7b-local_checkpoint-3000.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 59 |
| COORDINATE_IMPRECISION | 75 |
| CORRECT | 1697 |
| FORMAT_ERROR | 2 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 134 |
| SPATIAL_REASONING_ERROR | 33 |

