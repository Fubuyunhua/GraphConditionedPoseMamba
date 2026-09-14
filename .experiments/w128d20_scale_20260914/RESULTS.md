# Goal evidence:37.5mm not yet reached

2026-09-14 23:32 China time: source epoch45 RAW fully evaluated at
38.06752302741678/32.14069474392708mm. Original EMA remains preferred.
Entire3-arm sequence confirmed COMPLETED; velocity10 best37.723871,final38.086749,
no improvement. Next bounded trial: train-only mild detector jitter,not further calibration.

Original W128D20 detector EMA45:37.659287/31.871699. Target remains near37.5,
not redefined as successful code execution or improved P2.

Training-only scalar calibration (256fit/128validation clips,seed20260914):
scale.9993329164,validation normalizedP1 .0049709367 -> .0049645358.
Affine calibration with three scales and Y/Z bias, Xbias0 for flip consistency:
scales[1.0047337414,.9973937868,1.0050803391],bias[0,-.0002822034,.0001340838].
Validation normalizedP1 .0049709367 -> .0049271243. Selected affine before test,
folded into existing head,strict prediction-equivalence check passed.

Full unchanged H36M detector evaluator for selected affine checkpoint:
P1=37.68393222021504mm,P2=31.791669662833705mm. P1 is worse than source,so REJECT
as improvement. Do not claim goal met using alignedP2 improvement. Calibrated
checkpoint SHA63908b0d2d00bda626bfdd99de8969343bc6d7da1904c12a76021dc54a8f84ba;
source SHA6715a014823ac3b7365b8e8d64625486330f706f87191ddcdef2a313f599cbed.
Remote diagnostic root /scratch/home/caiwei/GraphConditionedPoseMamba_SCALE_DIAGNOSTIC_20260914;
calibration report verification/affine_diagnostic/report.json; evaluation log
launch_logs/affine_eval.log. Original weights untouched. Raw epoch45 diagnostic
was attempted but successful execution not confirmed after connection failures;
check process/file before retrying,never assume it ran or completed.

Series: noDropPath completed8,best37.715134;noDiff completed8,best37.720397.
Velocity10 confirmed live PID2455560 and controller2396003 at2026-09-14T13:50:26Z.
Subsequent observations suffered SSH/authentication and approval timeouts; these
are not evidence of training failure/completion. No process restarted/stopped.

This goal turn made progress: independent calibration implemented,train-validation
selection verified and full test refuted improvement. Goal remains active.
Next: revalidate same live/terminal handles,collect third outcome,verify raw-source
performance,then choose another evidence-backed adaptation if still below target.
