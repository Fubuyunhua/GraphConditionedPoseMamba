# E-RWG-0 final analysis

## Verdict

`KEEP` as a valid, completed single-seed topology ablation. Under the frozen
H36M protocol, the degree-preserving rewired graph is worse than the anatomical
graph, so seed 0 supports the claim that anatomical topology is useful. This is
not yet a multi-seed significance claim.

## Provenance and comparability

- Scientific implementation: `ad4e2fa66492737a3fcd88d9142a88731724f30b`.
- Deployment source: `1bb0a15b5308c440fcbd9952e485f59e983a089b`.
- Config: `configs/pose3d/ablation_full_rewired_graph.yaml`, SHA256
  `dbd51395a6530735fdcccd6979fb46e758ad43de7a368b3e3622b9138e2d1a0f`.
- Data: H36M-SH xy+confidence T243/S81, SHA256
  `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- Model: W64/D8, 800,083 parameters, seed 0, rewired graph seed 3407.
- Budget: 80/80 epochs, 354,960 optimizer and EMA updates; natural exit code 0.
- Optimizer: AdamW, LR `5e-4`, WD `0.012`, legacy `0.99` epoch decay;
  linear warmup and gradient clipping disabled.
- All six retained checkpoints are finite. Best and fixed raw/EMA checkpoints
  were strictly replayed with the same evaluator after training.

## Results

| Checkpoint | Epoch | P1 | Paired P2 |
|---|---:|---:|---:|
| Best EMA, test-monitored | 41 | 40.416912 | 33.822350 |
| Raw at the selected best epoch | 41 | 41.049115 | 33.941163 |
| Fixed epoch-80 EMA | 80 | 40.716592 | 33.611194 |
| Fixed epoch-80 raw | 80 | 41.826736 | 33.606595 |

The registered anatomical Full seed-0 best is `39.845162/33.232240 mm` at
epoch 53. Rewired is worse by `+0.571750 mm` P1 and `+0.590110 mm` paired P2
under the same test-monitored-best rule. A fixed epoch-80 comparison is not
available for the historical Full run because that checkpoint was not saved;
it must not be reconstructed or inferred.

## Dynamics and efficiency

- Mean training time: 2.906 minutes/epoch; sustained throughput about 25.46 it/s.
- Trainer peak reserved VRAM: 3,034 MiB.
- Final training loss: 0.009816; final training-time EMA result:
  `40.716592/33.611194 mm`.
- No NaN, Inf, CUDA, OOM, or traceback was found.

## Claim audit

- Supported for seed 0: preserving graph size and node degrees is insufficient
  to match the anatomical graph.
- Unresolved: variance across graph rewiring choices and training seeds; the
  preregistered graph is not changed after observing this result.
- Interpretation boundary: this result supports the value of this anatomical
  topology under the present protocol, not a universal claim about all graphs.
