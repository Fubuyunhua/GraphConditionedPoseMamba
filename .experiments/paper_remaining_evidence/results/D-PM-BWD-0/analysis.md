# D-PM-BWD-0 final analysis

## Verdict

`KEEP` as a completed implementation diagnostic and `REJECT` as an accuracy
improvement. Correcting the indexed backward path preserves inference but did
not improve the seed-0 monitored-best result.

## Provenance and comparability

- Scientific implementation: `ad4e2fa66492737a3fcd88d9142a88731724f30b`.
- Deployment source: `72c85945a2292efa9cc50c018ffbf08a23a6232d`.
- Config SHA256: `a1a1b03389b302488751f4a9ab0d0222bb2b578c42b623603f6cfe3bf01cbe76`.
- H36M data SHA256: `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- PoseMamba W64/D6/M1, 790,083 parameters, K=4, exact backward, seed0,
  batch4, 80/80 epochs and 354,960 optimizer/EMA updates.
- Released forward, CrossMerge, loss, optimizer, EMA and evaluator are fixed;
  only the indexed-path derivative differs.
- Natural exit code0; all retained checkpoints are finite; best/fixed raw and
  EMA checkpoints were strictly replayed.

## Results

| Checkpoint | Epoch | P1 | Paired P2 |
|---|---:|---:|---:|
| Best EMA, test-monitored | 52 | 40.660688 | 33.532125 |
| Raw at selected best epoch | 52 | 41.930659 | 34.227656 |
| Fixed epoch-80 EMA | 80 | 40.881943 | 33.502404 |
| Fixed epoch-80 raw | 80 | 41.887125 | 33.993313 |

The historical released-backward PoseMamba seed0 best is epoch60
`40.226022/33.517626 mm`. Exact backward is worse by `+0.434666 mm` P1 and
`+0.014498 mm` paired P2. The released run lacks a saved fixed epoch-80
checkpoint, so no fixed-80 comparison is claimed.

## Runtime and interpretation

- Mean time across the mixed standalone/concurrent run: 5.847 minutes/epoch.
- Standalone epochs were about4.38 minutes; MPI concurrency increased later
  epochs to about9.2 minutes. Shared timing is not publication efficiency data.
- Peak trainer-reserved VRAM: 2,854 MiB; final loss: 0.009757.
- No NaN, Inf, OOM, CUDA error or traceback occurred.
- The diagnostic confirms that the released derivative changes training, but
  seed0 does not support treating the correction as an accuracy improvement.
  Released and corrected results remain separate identities.
