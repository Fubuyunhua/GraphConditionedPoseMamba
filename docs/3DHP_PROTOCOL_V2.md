# Reused MPI-INF-3DHP protocol

This project ports the latest audited RTX5090 3DHP data path rather than the
older clipped-label implementation.

## Preserved protocol

- authoritative `data_3dhp`, never `data_3dhp_fixed`;
- mandatory train/test SHA-256 checks;
- T=81, training stride 9;
- deterministic floor-linspace resampling for short tails;
- every valid test centre evaluated exactly once with edge-padded context;
- raw and EMA checkpoints stored separately with RNG and optimizer state;
- MPJPE, P-MPJPE, PCK@150 and AUC[0,150];
- formal test remains locked until the fixed epoch-120 endpoint.

The prior runner intentionally monitored test metrics every epoch.  This port
sets `PER_EPOCH_TEST_MONITORING=False`; it must not select a checkpoint on the
3DHP test set.

## Files

- `lib/data/dataset_mpi3dhp_protocol_v2.py`
- `lib/utils/train_epoch_3dhp.py`
- `train_3dhp.py`
- `configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_protocol_v2_memopt.yaml`

## Preflight

Run protocol and CUDA smoke checks in new, empty output directories before any
formal training.  A formal 3DHP run is not authorized merely because the H36M
run is healthy.
