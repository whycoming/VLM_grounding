# Diagnostic Report

## Reliability Notes

- This diagnosis is heuristic and reproducible, but not omniscient.
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative bucket because the current files do not include all candidate object annotations.

## Overall Failure Counts

| type | count | ratio_over_failures |
|---|---:|---:|
| COORDINATE_IMPRECISION | 262 | 0.3830 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 206 | 0.3012 |
| SPATIAL_REASONING_ERROR | 109 | 0.1594 |
| ATTRIBUTE_OR_OTHER | 107 | 0.1564 |

## Per Dataset Counts

### rec_results_refcoco_val_sft-qwen25vl-7b-local_checkpoint-4800.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 19 |
| COORDINATE_IMPRECISION | 78 |
| CORRECT | 1841 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 20 |
| SPATIAL_REASONING_ERROR | 42 |

### rec_results_refcocog_val_sft-qwen25vl-7b-local_checkpoint-4800.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 39 |
| COORDINATE_IMPRECISION | 106 |
| CORRECT | 1752 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 66 |
| SPATIAL_REASONING_ERROR | 37 |

### rec_results_refcocop_val_sft-qwen25vl-7b-local_checkpoint-4800.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 49 |
| COORDINATE_IMPRECISION | 78 |
| CORRECT | 1723 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 120 |
| SPATIAL_REASONING_ERROR | 30 |

