# Logic

Retain the anatomy/control/reset mechanism supported by existing small-model ablations. W128D20 has direct evidence of strong accuracy with one-third of W256D16 parameters, so shrinking it to an untested depth solely for novelty is not justified. W256D16 has the best detected-input observation to date and enough capacity; pushing width/depth further has no supporting evidence and risks more overfit.

R3 optimizer avoids the earlier excursions; modest extra DropPath tests generalization. W128's best appeared at45 and leaves room for an80e run with gentler scheduling;256 uses the agreed60e budget. Detector and GT pairs share their training recipe. More parameters/regularization cannot guarantee improvement;36.x is an objective, not a forecast.
