# R3 user-stop analysis — 2026-09-06

Status: CANCELLED by explicit user instruction at 2026-09-06 00:49:26 Asia/Shanghai; 54/60 epochs completed, epoch55 interrupted. This is an incomplete scaling exploration, not a completed 60-epoch result or a numerical failure.

Best test-monitored EMA P1: 37.430162 mm at epoch32; paired P2: 31.520564 mm.
Last completed epoch54 EMA: 37.913791/31.753892 mm.
Train loss fell from 0.008266 to 0.005076 (38.59% reduction), while P1 increased by 0.483629 mm over 22 epochs without a new best. Grad norm fell from 0.907351 to 0.396324.

Interpretation: sustained train/test divergence is consistent with overfitting or exhausted generalization gains; the smooth finite gradients and decreasing parameter updates differ from R1/R2 excursions. This does not establish architecture-specific causality. No seed variance is available. The stopping decision also used monitored test results and must be disclosed.

Provenance: source 0e23c5d01eb5e6c81c66bc17a621d38787e7c46d; config SHA256 aa65995971b42892373f8f12d61448dc2d9343b0314d97271532f7b068404fc9; H36M dataset SHA256 73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175. Seed0; W256/D16 20,192,451 parameters; true warmup/cosine optimizer. It is not a controlled architecture-only comparison with legacy W64 runs.

All four best/latest raw/EMA checkpoint files load on CPU, have finite model tensors and retain optimizer entries; hashes are in checkpoint_audit.json. Best files identify epoch32, latest files epoch54. Original remote checkpoints and full logs remain intact. Metrics here are from the training evaluator; no new post-stop evaluator replay was performed. Epoch60 metrics do not exist.

Recommendation: KEEP the partial run as exploratory evidence; prioritize registered W64 structural ablations. Do not resume or label it completed. No model, loss or optimization changes were made to Rewired based on this curve.
