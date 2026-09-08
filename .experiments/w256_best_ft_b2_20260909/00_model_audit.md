# Audit: R3 best-EMA fine-tuning

Runtime verdict PASS on2026-09-09 02:32 Asia/Shanghai. B1/B2 loss and gradients
finite; B2 compiled peak11740MiB. Source/export equality, optimizer/EMA resets
and strict save/load/prediction roundtrip passed. Initial full evaluator returns
37.43025800596525/31.520638074032693 mm, within0.0001mm of historical result.
Formal PID1885417, actual8874 batches/epoch and15epoch budget printed in log.

Static verdict CONDITIONAL: execution is gated on source/checkpoint hashes, exact
strict loading, fresh optimizer/scheduler/EMA, real-data B1/B2 finite gradients,
EMA save/load/prediction roundtrip and initial full evaluation within0.02mm of
37.430162/31.520564. Script emits PASS only after numerical gates; launcher also
requires initial full evaluation before formal training.

Unchanged architecture W256/D16,20,192,451 parameters,T243,J17,three-channel
detector input,anatomical graph,independent factorized bidirectional recurrence.
Source is EMA epoch32 from R3; exact SHA c2175c027329e2d323bba5ceb3e613d2391f8dbf48a9e5cf6477eaf83bce7c7f.
Model tensors must match a weights-only export exactly. No optimizer,RNG,
scheduler state or old EMA count is carried forward. train.py:851-890 strict-loads
the export and initializes EMA from it; :953-986 restores optimizer only with
--resume, which is prohibited here. All layers trainable. Fresh seed0.

Active losses remain position1,scale-aligned.5,GT velocity20,predicted temporal
difference.5. Existing target-free smoothing is an acknowledged hypothesis,
not changed in this run. Detector dataset/split/root/flip and eval batch4 retained.
Original checkpoint immutable; new run directory and independent initializer.

Runtime clarification: train.py:790-796 uses args.batch_size for BOTH loaders;
the inherited test_batch_size field is not consumed. Effective evaluation batch
is2 (1114 batches), not the nominal4. Initial full replay gates metric parity.
This changes batching rather than dataset or averaging protocol. No trainer
change was introduced during deployment.

P2 risks: exploratory test-monitored selection, single seed, a changed batch and
fine-tuning schedule cannot isolate batch causality; extra compute must be reported.
No prediction of guaranteed accuracy gain. No periodic monitoring recreated.
