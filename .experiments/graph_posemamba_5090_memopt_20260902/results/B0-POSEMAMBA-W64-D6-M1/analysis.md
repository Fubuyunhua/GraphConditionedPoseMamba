# PoseMamba W64/D6/M1 matched-baseline audit

## Verdict

`KEEP_AS_MATCHED_SEED0_BASELINE`.

The prior PoseMamba W64/D6/M1 run is a valid same-environment, matched-capacity
seed-0 comparison for GraphConditionedPoseMamba.  It does not need to be
retrained for the first seed-0 table.

## Comparability

| Field | PoseMamba | GraphConditionedPoseMamba | Verdict |
|---|---|---|---|
| GPU | RTX 5090 | RTX 5090 | matched |
| data | H36M-SH xy+confidence | same | exact hash match |
| clip/stride | T243/S81 | T243/S81 | matched |
| seed | 0 | 0 | matched |
| epochs | 120 | 120 | matched |
| batch | 4 | 4 | matched |
| optimizer | AdamW | AdamW | matched |
| LR/WD/decay | 5e-4/0.012/0.99 | same | matched |
| EMA | 0.9998 | 0.9998 | matched |
| losses/augmentation/evaluator | frozen matched recipe | same | matched |
| parameters | 790,083 | 800,083 | near-matched; Graph +10,000 (+1.27%) |

PoseMamba uses the frozen official source commit
`df38d599212c058259f473464baec66d6a6487e0`.  The current standalone repository
retains a state-compatible PoseMamba implementation.  Strictly loading and
evaluating the old best EMA checkpoint in the current RTX5090 environment and
current evaluator reproduces exactly `40.226022/33.517626 mm` with 790,083
parameters.  This removes evaluator drift as a confounder.

## Accuracy comparison

### Test-monitored best EMA

| Model | Best epoch | P1 | paired P2 |
|---|---:|---:|---:|
| PoseMamba W64/D6/M1 | 60 | 40.2260 | 33.5176 |
| GraphConditionedPoseMamba W64/D8 | 53 | **39.8452** | **33.2322** |

Graph improves P1 by 0.3809 mm (0.95%) and paired P2 by 0.2854 mm (0.85%).

### Fixed epoch-120 EMA endpoint

| Model | P1 | P2 |
|---|---:|---:|
| PoseMamba W64/D6/M1 | **40.4098** | **33.3151** |
| GraphConditionedPoseMamba W64/D8 | 40.9894 | 33.3666 |

At the fixed endpoint, Graph is worse by 0.5795 mm P1 and 0.0515 mm P2.  The
best-checkpoint gain therefore depends on early selection and must not be
presented as an endpoint gain.

## Limits and recommendation

Both best checkpoints were selected by per-epoch H36M test evaluation and both
models have only seed 0 in this matched comparison.  The comparison is fair for
the existing experimental convention and school accuracy table, but it cannot
support a multi-seed or unbiased-test superiority claim.

Use the existing PoseMamba run in the primary 0.8M seed-0 table.  Report both
best EMA and epoch-120 EMA, call the parameter budget `approximately matched`
rather than exactly equal, and run paired seeds 1/2 for both models before a
strong paper claim.
