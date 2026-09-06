# Accuracy candidate logic

R3 train loss fell38.59% between epochs32 and54 while monitored-test P1 rose0.483629 mm. Finite gradients and shrinking updates make overfitting/exhausted generalization gains a plausible diagnosis, not proof of a specific cause.

Hypothesis: a modest increase in stochastic depth reduces co-adaptation in the large16-block model and improves fixed-endpoint generalization. Preserve anatomical graph and independent recurrence because current W64 ablations support them. DropPath is conventional regularization, not a new paper contribution. W64 structural effects are not automatically established for D16.

Prediction: training loss may be higher while final test error improves. Falsification: fixed60 EMA fails the registered accuracy target or training becomes invalid. A negative result is retained; it does not justify automatically trying a ladder of probabilities or combining new losses/augmentations.
