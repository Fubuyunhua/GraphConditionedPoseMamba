# Tested environment

The release snapshot was validated on:

```text
OS: Ubuntu 22.04
Python: 3.10.18
GPU: NVIDIA GeForce RTX 5060 Ti (compute capability 12.0)
Driver-visible CUDA: 13.0
CUDA toolkit / nvcc: 12.8 (V12.8.61)
PyTorch: 2.10.0.dev20251013+cu128
torchvision: 0.25.0.dev20251013+cu128
Triton: 3.5.0
timm: 1.0.20
einops: 0.8.1
fvcore: 0.1.5.post20221221
tensorboardX: 2.6.4
GCC: 11.4.0
```

This is an evidence record, not a universal lock file. In particular, an RTX
5060 Ti requires a PyTorch/CUDA toolchain that supports sm_120. On older GPUs,
install the PyTorch build recommended for that GPU and rebuild the extension.

After any environment change, run:

```bash
bash scripts/build_selective_scan.sh
python scripts/verify_install.py
python -m unittest discover -s tests -v
```
