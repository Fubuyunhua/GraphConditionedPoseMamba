# Spatial-only joined recurrence ablation — COMPLETED

- W64D8,800083 parameters,B4,T243,seed0;80/80epochs,normal exit0.
- Spatial SSM is retained and scans continuously across frame segments.
- Temporal SSM still resets between joint trajectories. This is NOT the earlier
  both-axis joined experiment and NOT deletion of the spatial branch.
- Best EMA: epoch50,P1=40.045326mm,pairedP2=33.426130mm.
- Fixed80 EMA: P1=40.572250mm,P2=33.438188mm.
- Historical Full best P1=39.845162mm; difference+0.200164mm, single seed only.

## Evidence

- [Completed ledger](ledger.json)
- [Original plan and audit](PLAN_AND_AUDIT.md)
- [Archived runtime preflight](PREFLIGHT_PASS.json)
- [All80 epoch metrics and source-log hash](results/metrics.json)
- [Analysis and limitations](results/analysis.md)
- [Exact config](../../configs/pose3d/ablation_spatial_joined_temporal_independent_80e.yaml)

The original raw training log and checkpoints are retained locally/remotely, not
published here. These are original online EMA evaluations; an independent
post-run checkpoint replay has not been performed. No significance claim is made.

## Reproduction support (do not restart an existing run)

The per-axis factory override is in lib/utils/learning.py; effective scopes are
recorded by model.execution_spec(). Omitting overrides preserves old defaults and
parameter construction. The run is fresh-only and refuses checkpoint initialization.
In a new, already provisioned environment and an independent output checkout,
generate the frozen Full fixture from immutable commit9a80bf63623771c8618dda33d7fad008dd1b14c6,
then run the preflight before the launcher:

```bash
mkdir -p verification
python tools/conditioning_reference.py --root /path/to/immutable-baseline-checkout --out verification/old_full_fixture.bin
python tools/preflight_spatial_joined.py
python scripts/run_spatial_joined_80e.py
```

The archived preflight is evidence, not a substitute for verifying a new environment.
This synchronization itself runs no training and uploads no weights or datasets.
