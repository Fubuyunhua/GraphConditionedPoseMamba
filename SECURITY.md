# Security and private-repository policy

This is a private research repository. Do not commit datasets, checkpoints,
experiment logs, paper drafts, credentials, personal paths, or compiled CUDA
binaries. Run `python tools/audit_release.py` before every push.

If a credential is committed, revoke it first and then coordinate repository
history cleanup. Do not post credentials in an issue.

CUDA extensions are built locally from `kernels/selective_scan/`; binary wheels
and `.so` files are not distributed by this repository.
