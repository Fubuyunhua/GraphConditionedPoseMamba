# R3 optimizer stabilization preflight and launch

## Preflight

- Verdict: PASS.
- Scientific source: `0e23c5d01eb5e6c81c66bc17a621d38787e7c46d`.
- Config SHA256: `aa65995971b42892373f8f12d61448dc2d9343b0314d97271532f7b068404fc9`.
- Dataset SHA256: `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- Unit suite: 31 passed.
- Trainable parameters: 20,192,451.
- AdamW decay group: 19,670,211 parameters at WD 0.012.
- Explicit SSM no-decay group: 522,240 parameters at WD 0.
- Effective schedule: 3e-5 start to 3e-4 over eight warmup epochs, then
  per-step cosine decay to 3e-5 by epoch 60.
- GPU free before gate: 32,109 MiB; no competing compute process.

| Stage | Mode | Reserved MiB | Loss | Pre-clip grad | it/s |
|---|---|---:|---:|---:|---:|
| B1 | eager | 6,212 | 1.165424 | 48.1329 | 8.385 |
| B2 | eager | 12,486 | 1.141652 | 43.0647 | 5.075 |
| B4 | compiled | 20,404 | 1.546054 | 46.3108 | 0.963 |

All losses and gradients are finite, clipping telemetry is active, optimizer
groups cover every trainable parameter exactly once, and the batch-4 memory
gate is below 28,672 MiB.

## Launch

Pending immediate launch after the PASS artifact is synchronized.
