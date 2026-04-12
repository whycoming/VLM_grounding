# Diagnostic Report

## Reliability Notes

- This diagnosis is heuristic and reproducible, but not omniscient.
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative bucket because the current files do not include all candidate object annotations.

## Overall Failure Counts

| type | count | ratio_over_failures |
|---|---:|---:|
| COORDINATE_IMPRECISION | 224 | 0.2983 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 193 | 0.2570 |
| ATTRIBUTE_OR_OTHER | 127 | 0.1691 |
| SPATIAL_REASONING_ERROR | 124 | 0.1651 |
| FORMAT_ERROR | 83 | 0.1105 |

## Per Dataset Counts

### rec_results_refcoco_val_outputs_phase3-online-dpo-qwen25vl-7b-local.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 22 |
| COORDINATE_IMPRECISION | 64 |
| CORRECT | 1819 |
| FORMAT_ERROR | 33 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 19 |
| SPATIAL_REASONING_ERROR | 43 |

### rec_results_refcocog_val_outputs_phase3-online-dpo-qwen25vl-7b-local.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 46 |
| COORDINATE_IMPRECISION | 87 |
| CORRECT | 1737 |
| FORMAT_ERROR | 26 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 58 |
| SPATIAL_REASONING_ERROR | 46 |

### rec_results_refcocop_val_outputs_phase3-online-dpo-qwen25vl-7b-local.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 59 |
| COORDINATE_IMPRECISION | 73 |
| CORRECT | 1693 |
| FORMAT_ERROR | 24 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 116 |
| SPATIAL_REASONING_ERROR | 35 |

