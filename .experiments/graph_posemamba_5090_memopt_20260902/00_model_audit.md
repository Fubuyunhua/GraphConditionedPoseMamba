# GraphConditionedPoseMamba 0.8M preflight audit

## Frozen baseline

- Upstream commit: `14216ccc8699839c3a472e32ef95071464732975`.
- Model/config: `GraphConditionedPoseMamba`, W64/D8, 800,083 parameters,
  `configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml`.
- Data: H36M-SH `h36m_sh_conf_cam_source_final.pkl`, SHA-256
  `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- Protocol: FP32, batch 4, T=243, seed 0, AdamW, 120 epochs, EMA 0.9998.
- External comparison: the independently running RTX 5060 Ti seed-0 result;
  its artifacts are not available in this workspace and must not be merged or resumed.

## Tensor and model flow

Input `[B,243,17,3]` is embedded to `[B,243,17,64]`.  Each of eight blocks
uses an H36M bone/symmetry graph context, independent per-frame spatial BiSSM
`[B*243,1,17,64]`, independent per-joint temporal BiSSM
`[B*17,1,243,64]`, and an MLP residual.  Output is `[B,243,17,3]`.
The graph context controls selective-scan Delta/B/C while recurrent content
comes from the unmodified normalized feature.

## Active losses

- MPJPE, coefficient 1.0.
- normalized MPJPE/scale loss, coefficient 0.5.
- 3D velocity loss, coefficient 20.0.
- weighted prediction-difference regularizer, coefficient 0.5.

All other configurable losses are zero and short-circuited.  The memory work
does not change any loss, target, reduction, coefficient, augmentation or
gradient destination.

## Data and evaluation

H36M-SH train/test data are reused through a symlink to the existing audited
dataset.  Root-relative training, confidence input, flip evaluation, clip 243
and stride 81 match the released configuration.  Per-epoch test evaluation is
inherited and is a scientific limitation; the RTX 5090 run is a reproduction
against the RTX 5060 Ti baseline, not a new locked-test claim.

## Runtime findings

- RTX 5090: driver 580.159.03, PyTorch 2.11.0+cu128, compute capability 12.0.
- Existing `selective_scan_cuda_core` imports and forward/backward passes.
- Fourteen repository tests pass on the RTX 5090.
- A concurrent experiment uses about 13.6 GiB, leaving about 18.5 GiB at audit time.
- The upstream `reduce-overhead` mode enables CUDA Graph pools and is the main
  identified high-watermark risk.  Model parameters are only about 3.1 MiB.

## Ranked risks

- P1: the exact peak memory and throughput ranking of eager/default/
  reduce-overhead/checkpointed modes is not yet measured on this RTX 5090.
- P1: activation checkpoint must preserve DropPath RNG and gradients; a paired
  output/gradient test is required before selection.
- P1: the concurrent experiment may change its memory footprint; launch needs a
  fresh headroom check and fail-closed smoke.
- P2: PyTorch 2.11 differs from the 2.10 nightly used on RTX 5060 Ti, so the
  comparison must record environment differences.
- P2: per-epoch H36M test monitoring prevents an unbiased model-selection claim.
- P3: deprecated timm and AMP imports emit warnings but do not block execution.

## Memory A/B and equivalence results

All values are per-process FP32 batch-4 measurements on RTX 5090 while the
existing experiment remained active:

| Candidate | Peak allocated | Peak reserved | Throughput |
|---|---:|---:|---:|
| eager | 2,829 MiB | 2,954 MiB | 9.71 it/s |
| default compile | 2,544 MiB | 2,772 MiB | 11.71 it/s |
| reduce-overhead | 2,601 MiB warmup | 2,826 MiB | 13.28 it/s |
| default + block checkpoint | 1,672 MiB | 1,826 MiB | 9.97 it/s |

Default compile on real H36M data reached 11.61 it/s with 2,836 MiB reserved
and finite loss.  Eager versus default-compiled full-model verification passed:
loss delta `0`, output max absolute delta `1.79e-6`, input-gradient max delta
`1.49e-9`, parameter-gradient max delta `5.96e-8`.

M2 is selected for the 0.8M run.  It has ample concurrent headroom, is faster
than eager, and uses eager evaluation so a separate compiled evaluation graph
cannot accumulate alongside the training graph.  Activation checkpointing is
kept as a verified fallback for larger models because it saves another 1 GiB
but is slower and unnecessary at 0.8M.

## Verdict

`PASS` for M2 (`default compile + eager evaluation + early zero_grad`).  Model
math, FP32 precision, data, batch, T, loss, EMA and evaluation semantics are
unchanged.  The formal run must archive the first epoch checkpoint and confirm
that post-evaluation GPU memory remains stable before being treated as a healthy
long run.
