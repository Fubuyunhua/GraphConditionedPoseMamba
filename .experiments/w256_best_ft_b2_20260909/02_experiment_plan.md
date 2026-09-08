# Registered plan

ID R3-BEST-FT-B2-S0, IMPROVEMENT, user authorized2026-09-09.
One15-epoch run,batch2,FP32,all layers trainable,seed0,AdamW peak1.5e-5,
warmup1e from1.5e-6,cosine floor1.5e-6,WD.012 with SSM exemptions,clip1,
DropPath.20,EMA.9998999949995,unchanged data/augmentation/losses/eval batch4.
No resume; CLI --pretrained initializers --selection r3_best_ema.bin.

Primary: best EMA P1 across fine-tuning epochs1-15, always compared with untouched
epoch0 source37.430162; paired P2 and fixed15 secondary. Any lower P1 is an observed
gain; >=0.2mm is a provisional meaningful single-run threshold, not statistical
significance. Falsification: no improvement after15e; keep source as preferred.
Stop only at fixed15, technical failure or explicit user instruction. No adaptive
budget extension, automatic rerun or new seed. Store config/source/data hashes,
initial replay,preflight,logs,best/latest weights and full paired metrics.

Budget disclosure: original R3 consumed54epochs to identify epoch32 weights;
this adds15epochs at batch2, approximately twice the steps/epoch. Not comparable
to a fresh15e or fixed60e total budget. Estimated runtime awaits measured batch2
throughput; do not promise batch2 is faster. Other seed/GT plans are not launched
by this script.
