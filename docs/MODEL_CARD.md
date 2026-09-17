# Model card

Task: video 2D-to-3D pose lifting. Input [B,243,17,3] is xy/confidence; output is
[B,243,17,3]. Use only the intended data conventions; not a medical or safety system.

| Model | Detector best epoch | Historical detector epochs | GT2D best epoch | GT2D P1 / P2 mm |
|---|---:|---:|---:|---:|
| W64D8 |53|120|78|16.816 / 13.979|
| W128D10 |54|80|79|12.653 / 10.812|
| W128D20 |45|65 of planned80|65|11.310 / 10.058|

All are seed0 EMA. W128D10 uses batch4, not the older batch8 recipe. W128D20 detector
is the original checkpoint, not a fine-tuning mixture. GT runs completed80 epochs.
The public80-epoch configs preserve chosen scientific settings but are a common
future budget, not a claim of new completed runs or guaranteed convergence.

Evaluation reuses `train.py::evaluate`: flip TTA, root-relative17-joint P1/MPJPE,
P2/P-MPJPE and original action aggregation and valid-frame/block rules.
GT2D sets confidence to one and overwrites predicted XY with input GT XY during
evaluation: this is known-XY/depth lifting, not unconstrained XYZ or detector input.
Best checkpoints were selected by monitored test performance, not an independent
validation split. Report this limitation; one seed does not establish significance.

Returned evaluator arrays can mix coordinate-processing stages because valid
occurrences are scaled in place. Do not denormalize/scale them again for plots.
Use the exact root-relative values used by the metric when exporting predictions.

Public weights contain model tensors and minimal identity metadata only, not
optimizer states or private logs. They support evaluation, not exact optimizer
resume. Do not silently load mismatched keys or protocols.
