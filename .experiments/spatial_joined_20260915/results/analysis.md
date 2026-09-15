# SJOIN-W64D8-S0 — completed spatial-only recurrence ablation

## Identity and comparability

W64D8,800083 actual parameters,seed0,B4,T243,H36M-SH detector xy+confidence.
Only spatial recurrence scope changes independent->joined. Temporal recurrence
remains independent. Spatial SSM is retained, not removed. Local projections,
convolution, graph-conditioned Delta/B/C, loss,optimizer and80epoch recipe remain
unchanged. This is distinct from the earlier both-axis joined experiment.

The optional factory overrides preserve parameter construction/RNG consumption.
Preflight verified identical current Full/variant initialization fingerprints,
default Full forward/gradient parity with the frozen reference, spatial cross-frame
effects, unchanged temporal independence, finite B1/B2/B4 training and EMA roundtrip.
No baseline checkpoint was overwritten. The original80-row training log was
retrieved; its SHA256 is recorded in metrics.json. Fresh random initialization and
fixed80 EMA save markers are present. Normal exit0 was separately read remotely.

## Results (mm)

| Setting | Best epoch | Best EMA P1 | Paired P2 | Completed budget |
|---|---:|---:|---:|---:|
| Historical Full: spatial independent, temporal independent |53|39.845162|33.232240|80epoch comparison budget|
| Spatial joined, temporal independent |50|40.045326|33.426130|80/80|

The joined variant is0.200164mm higher in best P1. At fixed epoch80 its EMA
P1/P2 is40.572250/33.438188mm. Full per-epoch scalar results are in metrics.json;
secondary metrics are paired with the same checkpoint, not independently selected.
Recent pure training time averaged1.929minutes/epoch; observed late epoch-to-epoch
interval was approximately129seconds including evaluation. This is log timing,
not an exclusive-device speed benchmark or a MAC measurement.

## Interpretation and limits

KEEP as a completed single-factor ablation. This seed supports the direction of
benefit from independent spatial frame boundaries. It does not prove statistical
significance or indispensability. Per-epoch test monitoring and best-checkpoint
selection are disclosed. Independent post-run checkpoint replay and multi-seed
replication are NOT claimed. The new efficiency sweep excluded ablations, so its
latency/MAC results must not be relabeled as this experiment's measurements.

## Archived configuration

Use configs/pose3d/ablation_spatial_joined_temporal_independent_80e.yaml with the
per-axis factory support in lib/utils/learning.py. Older code that ignores these
fields must not be used: it would silently run the wrong experiment.
The config requires a fresh-initialization preflight manifest. The archived
preflight and run helpers are not an instruction to restart training; a new
environment requires its own checks and a new output directory.

Published material contains configuration, implementation support and result
records only; no model weights, raw dataset or credentials.
