# Diagnostic Report

## Reliability Notes

- This diagnosis is heuristic and reproducible, but not omniscient.
- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative bucket because the current files do not include all candidate object annotations.

## Overall Failure Counts

| type | count | ratio_over_failures |
|---|---:|---:|
| COORDINATE_IMPRECISION | 272 | 0.3219 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 263 | 0.3112 |
| SPATIAL_REASONING_ERROR | 157 | 0.1858 |
| ATTRIBUTE_OR_OTHER | 153 | 0.1811 |

## Per Dataset Counts

### rec_results_refcoco_val_qwen25vl-3b.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 29 |
| COORDINATE_IMPRECISION | 83 |
| CORRECT | 1779 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 39 |
| SPATIAL_REASONING_ERROR | 70 |

### rec_results_refcocog_val_qwen25vl-3b.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 60 |
| COORDINATE_IMPRECISION | 91 |
| CORRECT | 1730 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 69 |
| SPATIAL_REASONING_ERROR | 50 |

### rec_results_refcocop_val_qwen25vl-3b.json

| type | count |
|---|---:|
| ATTRIBUTE_OR_OTHER | 64 |
| COORDINATE_IMPRECISION | 98 |
| CORRECT | 1646 |
| LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET | 155 |
| SPATIAL_REASONING_ERROR | 37 |

