# RTX5090 0.8M experiment results

Date: 2026-09-03  
Status: complete, audited, single seed

## Study identity

- Candidate: GraphConditionedPoseMamba W64/D8, 800,083 trainable parameters.
- Baseline: PoseMamba W64/D6/M1, 790,083 trainable parameters.
- Dataset: H36M-SH xy+confidence, T=243, train stride 81.
- Dataset SHA-256:
  `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- Common protocol: seed 0, 120 epochs, batch 4, AdamW, LR 5e-4,
  weight decay 0.012, LR decay 0.99, EMA 0.9998, identical losses,
  augmentation and evaluator.
- Candidate runtime: FP32 `torch.compile(mode="reduce-overhead")` for training;
  FP32 eager evaluation.

The parameter budgets are approximately matched: Graph has 10,000 additional
parameters, or 1.27% more than PoseMamba.

## Accuracy

### Test-monitored best EMA

| Model | Best epoch | P1 MPJPE | paired P2 |
|---|---:|---:|---:|
| PoseMamba W64/D6/M1 | 60 | 40.2260 | 33.5176 |
| GraphConditionedPoseMamba W64/D8 | 53 | **39.8452** | **33.2322** |

At the best EMA checkpoint, Graph improves P1 by 0.3809 mm (0.95%) and P2 by
0.2854 mm (0.85%).  The old PoseMamba checkpoint was strictly loaded and
re-evaluated with the current RTX5090 environment and current evaluator; the
result reproduced exactly.

### Fixed epoch-120 EMA endpoint

| Model | P1 MPJPE | P2 P-MPJPE |
|---|---:|---:|
| PoseMamba W64/D6/M1 | **40.4098** | **33.3151** |
| GraphConditionedPoseMamba W64/D8 | 40.9894 | 33.3666 |

The candidate does not improve the fixed endpoint.  Its best-checkpoint gain
depends on early selection and must not be presented as an epoch-120 gain.

The independently lowest Graph P2 is 33.1231 mm at epoch 89.  It is not paired
with the P1-best checkpoint and therefore is not substituted into the primary
row.

## Training behavior

- The candidate completed 120/120 epochs in 7.6867 wall-clock hours.
- Best P1 occurred at epoch 53; best P2 occurred at epoch 89.
- Training loss continued from 0.010465 at epoch 53 to 0.008504 at epoch 120,
  while P1 regressed by about 1.14 mm.  This indicates late overfitting or an
  objective/evaluation mismatch rather than numerical instability.
- No NaN/Inf, CUDA OOM, traceback or non-finite error occurred.
- Latest/best raw and EMA checkpoint families all load strictly.

## Memory and runtime engineering

The selected runtime changes execution only; model math, FP32 precision, data,
batch, T, losses and EMA are unchanged.

| Mode | Peak reserved | Short throughput |
|---|---:|---:|
| FP32 eager | 2,954 MiB | 9.71 it/s |
| default compile | 2,772 MiB | 11.71 it/s |
| reduce-overhead compile | 2,826 MiB | 13.28 it/s |
| default + activation checkpoint | 1,826 MiB | 9.97 it/s |

The formal process used about 3,734 MiB.  A saved EMA checkpoint performed
FP32 eager batch-1 inference with 610 MiB reserved.  Runtime equivalence checks
reported zero loss delta, output maximum absolute delta `1.79e-6`, and maximum
parameter-gradient delta `5.96e-8`.

## 3DHP protocol port

The project also includes the repaired MPI-INF-3DHP data/training path:
authoritative unclipped files, mandatory hashes, T=81, stride 9, deterministic
tail resampling and one evaluation per valid test centre.  Protocol-only and
CUDA forward/backward smoke tests pass.  No GraphConditionedPoseMamba 3DHP
formal result is claimed here.

## Interpretation limits

- Both accuracy rows are seed 0 only.
- Both best checkpoints use the inherited per-epoch H36M test monitoring.
- Training-speed comparisons against the historical PoseMamba run are invalid
  because concurrent GPU load and runtime implementations differ.
- A strong superiority claim requires paired seeds 1 and 2 for both models and
  preferably validation-based checkpoint selection.

Detailed machine-readable evidence is under
`.experiments/graph_posemamba_5090_memopt_20260902/`.
