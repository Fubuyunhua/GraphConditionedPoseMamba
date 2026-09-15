# Objective complexity re-audit: conventions, coverage and corrections

## Outcome

The original mixed-counter values must NOT be used as standardized MACs or
directly compared with the PoseMamba paper table. Actual parameter sums remain
valid. Five architectures were reprofiled on5090; corrected fvcore contraction
counts agree EXACTLY with an independent ATen-dispatch counter for all five.
No architecture or training weights were altered for profiling.

| Model | True parameters | Vanilla THOP parameter subtotal | Vanilla THOP G-MAC report (incomplete) | Audited contraction G-MACs | Contractions + parallel-scan G-MAC-equivalent |
|---|---:|---:|---:|---:|---:|
| Local PoseMamba A0 W64D6/M1 | 790083 | 417091 | 1.747578 | 2.698303 | 4.551106 |
| GCPM W64D8 | 800083 | 560691 | 2.342442 | 3.444395 | 4.602397 |
| GCPM W128D10 | 3435395 | 2768475 | 11.502026 | 15.843740 | 18.738745 |
| GCPM W128D20 | 6836355 | 5535795 | 22.998764 | 31.684307 | 37.474317 |
| Local PoseMamba_L.bin architecture W128D20/C2 | 9450499 | 6713859 | 27.881408 | 35.578155 | 47.930175 |

Input B1,T243,J17; three channels for A0/GCPM, two for the large PoseMamba
checkpoint's actual config. Each result is one eval forward, not flip TTA,
training, latency or a new accuracy result. GPU/torch/fvcore/thop versions,
per-operator results and exact integer counts are in
`evidence/fig2/complexity_reaudit.json`.

## Primary-source evidence

1. [PoseMamba official example](https://github.com/nankingjing/PoseMamba/blob/main/lib/model/PoseMamba.py)
   invokes vanilla `thop.profile`, without a custom selective-scan handler.
2. [MotionAGFormer official example](https://github.com/TaatiTeam/MotionAGFormer/blob/master/model/MotionAGFormer.py)
   invokes `torchprofile.profile_macs` and separately sums model.parameters().
   torchprofile is not installed in the inspected5090 environment: its official
   handlers were inspected, but the package was NOT installed or run.
3. [Mamba author explanation](https://github.com/state-spaces/mamba/issues/110#issuecomment-1919470069)
   explicitly describes9BDLN as forward scalar FLOPs, alongside2*parameters*tokens
   linear FLOPs. It estimates a parallel associative scan, not a hardware trace.
4. [fvcore source](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py)
   counts fused multiply-add as1; its generic
   [matmul/einsum handlers](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/jit_handles.py)
   require additional care for this model's broadcasting and formatted einsum counts.
5. [THOP source](https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/profile.py)
   registers module-specific hooks. Unsupported custom operations and bare
   parameters need not be included in its default subtotals.

## What was wrong or insufficient

- **Inconsistent units:** previous total added FMA=1 contraction counts to9BDLN
  scalar-FLOP scan estimates. Naming it MAC-equivalent and comparing directly
  with literature was insufficiently justified. This affects small models most.
- **Broadcast undercount:**17x17@[243,17,32] needs2,247,264 MACs. Default installed
  fvcore returned9,248, missing the243-way broadcast. Hand count and ATen dispatch
  both confirm2,247,264. The correction increases, rather than decreases, GCPM cost.
- **Einsum rounding:** fvcore's generic numpy.einsum_path formatted text rounded
  A0's contraction subtotal by85,440 MACs. Exact dimension products remove this
  small discrepancy; the independent counters now agree exactly.
- **THOP is not a complete count:** directly adopting its lower totals would omit
  functional projections/scan work and undercount bare SSM parameters. These are
  shown as incomplete method-specific subtotals, not recommended full model sizes.

## Corrected scope and recommendation

Report true parameter sum. For an explicitly declared scan-inclusive analytical
metric, use:

`contraction_MACs + sum(9*B*D*L*N/2 + B*D*L)`.

The second term converts the Mamba-author parallel core FLOPs into consistent
MAC-equivalent units and includes one D*u+y skip MAC per scalar. All12/16/20/40
scan calls (and40 for the large PoseMamba) are included, with directional channels
already present in D; no second multiplication for directions is applied.
Contractions include Linear/Conv/functional einsum and broadcasted matrix products.
Norm, activations, other elementwise operations, exp/softplus, memory movement and
backward are excluded. Therefore this is still a declared analytical estimate,
NOT an exact hardware-instruction count. Do not halve the whole previous total.

Keep vanilla THOP values only as a clearly labeled comparison with the official
example's reporting method. Never choose the smaller counter to make a model
look cheaper. Runtime measurements are a separate metric and cannot calibrate MACs.

The former claim that GCPM0.8M has10.217% less work than A0 DOES NOT survive this
consistent re-audit:4.602397G vs4.551106G is about1.13% more. This says nothing
directly about measured latency or accuracy.

## Correction to the large PoseMamba interpretation

The same9,450,499-parameter architecture reports6,713,859 parameters and27.881G
under vanilla THOP, close to the paper's6.7M/27.9G. Thus the earlier assertion
that the local PoseMamba_L.bin cannot be the paper's L merely because its full
parameter count is9.450M was not justified. The counter discrepancy explains
that apparent mismatch. Checkpoint provenance/evaluation still must be verified
separately; numerical closeness alone is not proof of an exact release identity.

## Verification and publication status

Independent dispatch and JIT contraction totals agree for all5 architectures;
microtest for broadcast and4 arithmetic unit tests pass. No training was started
by the profiler. Spatial-only joined ablation runs independently and unchanged.
Original result files are retained for provenance. The old GitHub table is marked
superseded during the user-authorized result synchronization; corrected eight-model
results are in ALL_MODELS_EFFICIENCY_20260915.md and the public efficiency summary.
Profiling source remains local. No old accuracy record is changed.

