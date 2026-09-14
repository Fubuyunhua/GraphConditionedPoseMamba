# Bounded training-only scale fine-tuning diagnostic

Active user goal: original W128D20 best detector model near37.5mm. Previous two
regularization/loss arms failed; third remains live and is not interrupted.
Test a post-hoc hypothesis of global output-scale bias with one parameter rather
than another full-network sweep. Exact same6.84M model architecture is retained:
learned positive scalar folds into existing head weight/bias,not an extra inference
module. This is trained head calibration on source EMA,not ordinary EMA updates.

Fit on256 preselected training clips,validate on128 other training clips,seed20260914,
stride9 frames to reduce redundancy,flip-averaged source predictions. No S9/S11
labels used for fitting or selecting scale. These clips were seen in source training,
so validation is not an independent validation of the pretrained source.
One scalar in[.95,1.05],50LBFGS iterations maximum; normalized MPJPE objective.
If validation does not improve,export nothing. If it improves,one full standard
detector test evaluation,then independently repeat if within target. No test-scale
sweep,postprocessing-only score changes or modified metric. Original source untouched.
Need strict source hash,finite scale andfolded-head equivalence. Document any result
as calibrated EMA-source,not pretend it arose from standard model EMA training.
Aim near37.5mm,requires verified full evaluator result,not rounding37.659 down.
