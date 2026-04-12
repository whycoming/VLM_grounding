# Phase 3 Online DPO Conclusion

## Setup

- Base checkpoint: `outputs/sft-qwen25vl-7b-local/checkpoint-4800`
- Training output: `outputs/phase3-online-dpo-qwen25vl-7b-local`
- Evaluation output: `outputs/eval-phase3-online-dpo-1500`
- Evaluation splits: `refcoco_val`, `refcocop_val`, `refcocog_val`
- Samples per split: `2000`

## Result

| split | acc@0.5 | acc@0.75 | mean IoU |
|---|---:|---:|---:|
| refcoco_val | 0.9095 | 0.8065 | 0.8160 |
| refcocop_val | 0.8465 | 0.7640 | 0.7706 |
| refcocog_val | 0.8685 | 0.7625 | 0.7867 |

## Comparison To Phase 2 SFT

| split | acc@0.5 delta | acc@0.75 delta | mean IoU delta |
|---|---:|---:|---:|
| refcoco_val | -0.0110 | -0.0055 | -0.0104 |
| refcocop_val | -0.0150 | -0.0075 | -0.0098 |
| refcocog_val | -0.0075 | +0.0015 | -0.0051 |

Online DPO does not beat the Phase 2 early-stop SFT checkpoint overall. It slightly improves `refcocog_val` acc@0.75, but this is not enough to offset lower acc@0.5 and mean IoU on all three splits.

## Comparison To Phase 3 RFT

| split | acc@0.5 delta | acc@0.75 delta | mean IoU delta |
|---|---:|---:|---:|
| refcoco_val | +0.0035 | +0.0135 | +0.0162 |
| refcocop_val | -0.0020 | +0.0215 | +0.0175 |
| refcocog_val | +0.0040 | +0.0210 | +0.0172 |

Online DPO is better than RFT for box quality. The strongest signal is the consistent improvement in `acc@0.75` and mean IoU, indicating tighter localization even when acc@0.5 changes are small.

## Decision

- Keep Online DPO as a stronger Phase 3 candidate than RFT.
- Do not promote Online DPO over the Phase 2 SFT checkpoint as the current best model.
- The next Phase 3 priority should be Hard Negative DPO or another targeted method focused on residual wrong-target and ambiguity errors, because generic online DPO improves ranking/box quality but still regresses overall against the SFT baseline.
