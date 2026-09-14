# Detection-noise adaptation toward active37.5mm goal

Previous three8e layerwise regularization/loss trials completed without P1 gains.
Train-only affine head calibration improved training validation but failed standard
test(P1=37.683932),so it is not adopted. A new bounded8e trial targets input
robustness rather than repeating the same clean-input optimization.

Original W128D20 EMA45 source retained; all parameters,layerwiseLR1e-6/3e-6/1e-5,
batch4,8epochs,warmup1+cosine,EMA.9998,DP.20,clip1,original losses unchanged.
Only new delta: on50% of training clips,add zero-mean temporally interpolated
27-keyframe Gaussian xy noise,std.002 normalized(~1pixel at1000width before
interpolation). Confidence and3D target unchanged. No test corruption or metric
change. This is mild augmentation,not calibrated confidence modeling. It can still
hurt precision; target is hypothesis,not promised outcome.

DIAG tests defaultidentity,no mutation,confidence preservation,determinism and
smooth temporal perturbations; actual B1/B2/B4 finite gradients,all-layer updates,
source/reset/roundtrip/initial full replay gates before launch. Review correct
source(Raw diagnostic may inform later choice; this registered run stays originalEMA).
Primary bestEMA P1 vs original37.659,pairedP2/fixed8/velocity losses secondary.
Report single-seed repeated-test tuning as exploratory. No automatic extension,
extra noise strengths or seed sweep. Source checkpoint and previous runs immutable.
