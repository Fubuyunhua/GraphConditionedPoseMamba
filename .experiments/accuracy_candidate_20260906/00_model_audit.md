# Accuracy candidate audit

Verdict: BLOCKED for launch until predecessor experiments finish and deployment/runtime preflight passes. This is a technical dependency, not missing user authorization.

Source baseline: paper-evidence e30ae89, inheriting R3's model and optimizer implementation. Candidate config inherits graph_posemamba_h36m_w256_d16_stable_r3_60e.yaml and changes only maximum DropPath0.20 to0.30 as a scientific training factor. no_eval=true and checkpoint_frequency10 change observation/checkpoint policy, not the optimization objective.

Input[B,243,17,3] -> W256 embedding ->16 anatomical/control/independent K2 blocks -> root-relative output[B,243,17,3]. Position/scale/velocity/difference loss weights1/0.5/20/0.5 unchanged. AdamW explicit SSM no-decay, batch4, seed0, FP32, clipping1.0, true8-epoch linear warmup3e-5 to3e-4 and cosine to3e-5 over60 epochs, EMA0.9998 remain R3 settings. Fresh initialization, no R3 checkpoint resume.

train.py no_eval branch skips evaluation; common latest raw/EMA and periodic checkpoint writes remain active, and best writes are guarded by not no_eval. Final evaluation is an explicit --evaluate invocation after training. Verify this path with a throwaway lifecycle smoke before formal launch; never shorten the formal config for a smoke in place.

Risks: P1 regularization may underfit or fail to improve; P1 historical R3 is a single-seed test-monitored incomplete comparator with no fixed60 checkpoint; P2 DropPath RNG/compile behavior and B4 VRAM must be checked. No statistical superiority claim from one trial. Historical test exposure cannot be undone by locking this new endpoint.

Before PASS: freeze source/config/data hashes, resolve inheritance, verify parameter count20192451 and DropPath schedule, optimizer groups/LR/losses, real-data finite gradients and B1/B2/B4 compiled steps under an exclusive GPU with peak reserved<28672 MiB; verify no-eval checkpoint lifecycle and strict final EMA loading. Reuse unchanged regression tests with recorded source identity. Do not change batch/precision to bypass a failed gate.
