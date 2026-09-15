# Reproduction protocol and release boundaries

## Canonical configurations

The six flat YAMLs in `configs/gcs_pose/{detector,gt2d}` are the public entry points.
They preserve the selected historical architecture, loss, augmentation, optimizer
and execution options, while standardizing the stopping budget to 80 epochs and
batch size to 4. They contain no private paths or historical initialization hashes.
The launcher accepts neither resume nor pretrained initialization. The archived
environment-specific fresh-initialization fingerprint guard is not used by these
portable configs; do not describe them as having passed that historical preflight.

For these six recipes the actual optimizer is AdamW, LR 0.0005, epoch decay 0.99,
weight decay 0.012, EMA decay 0.9998. Linear warmup is disabled. W128D20 GT uses
gradient clipping at 1; the other five use no clipping. Check the flat YAML for
all loss coefficients, drop-path, scan ratios and compilation settings.
Do not substitute a width/depth alone in another model's recipe.

80 is a common reproduction budget, not a guarantee of convergence. The observed
best epochs of the selected three detector runs are 53/54/45; GT epochs are
78/79/65. A 79-epoch retrospective coverage bound is not a prospective guarantee.
The historical W64D8 detector ran 120 epochs; W128D20 detector stopped at 65.
Neither is an already-completed matched 80-epoch experiment.

## Data

Obtain Human3.6M under its own license. We distribute neither raw videos nor
preprocessed arrays. Follow `DATA.md` for the expected serialized data layout.
The input is the historical H36M-SH detector preprocessing, not an interchangeable
modern detector. Expected metadata SHA256 for the recorded experiments:
`73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
The recorded material has 17,748 train and 2,228 test clips. A matching hash is a
useful identity check, not proof of legal access or a complete preprocessing audit.
Raw videos alone are insufficient: camera normalization, 2.5D factors, source IDs,
joint mapping and clip boundaries must match the metadata. An end-to-end raw-video
to this exact preprocessed release pipeline is not supplied yet.

## Evaluation

Use the same config as the checkpoint, with `train.py --evaluate PATH` and a new
`--checkpoint` logging prefix. Only load trusted checkpoints: the historical
trainer uses pickle-based `torch.load(..., weights_only=False)`. State loading is
strict. Archive checkpoint SHA256, epoch, raw/EMA type, resolved config and source
commit. No public downloadable checkpoint bundle is included in this release.

Human3.6M uses T=243, 17 joints, root-relative coordinates, flip TTA, and the
historical valid-frame/block-list and action aggregation in `train.py::evaluate`.
P1 is MPJPE in mm; P2 uses rigid alignment and must not be called P1. Detector
inputs include confidence. GT2D sets confidence to one and the historical evaluator
overwrites predicted XY with the GT2D input: these GT results are a known-XY/depth
lifting protocol, not a detector benchmark or an unconstrained XYZ prediction score.

The evaluator denormalizes and scales valid occurrences in place. Its returned
prediction array can have mixed coordinate stages: never denormalize/scale it again
for qualitative plots. Preserve occurrence IDs and the original aggregation.

Historical best checkpoints were selected by monitored test performance. They are
not validation-selected held-out-test estimates. Report both best and fixed-budget
results where available, and use paired seeds for causal ablation claims.

## Scope of this release

The public entry points prioritize the three selected scales. Existing research
configs and `.experiments` are provenance archives, not instructions to resume
queues. A2/Full repeat status was last checked as running; this release does not
claim their completion. Q/H content-only has CPU routing tests but no completed
native CUDA preflight or training result; it is not a recommended release model.
MPI-3DHP GT2D results require an independent protocol audit; do not claim SOTA
from the historical PCK/AUC values or mix them with detector results.

Local syntax/config/counter checks do not establish fresh-clone CUDA reproduction.
No training, package installation, remote process changes or weight uploads are
performed by preparing this release. The installation check intentionally runs a
small forward/backward only when the user explicitly invokes it.
