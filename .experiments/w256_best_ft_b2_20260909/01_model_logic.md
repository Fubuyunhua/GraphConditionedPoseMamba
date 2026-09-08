# Fine-tuning logic

Observed failure: FINAL DP.30 did not beat R3's best and plateaued near38mm.
User explicitly chooses to stop FINAL and fine-tune the earlier best with batch2.
Hypothesis: small updates around the37.430mm weights may improve generalization
without relearning from scratch. This is conventional fine-tuning, not a new model
innovation. Smaller batches do not guarantee better accuracy.

Keep R3 DropPath.20 and original losses. Peak1.5e-5 is a conservative engineering
choice, not an AdamW scaling law. One warmup epoch starts1.5e-6; cosine reaches
1.5e-6 at15epochs. EMA decay sqrt(.9998)=.9998999949995 approximately preserves
the sample-domain averaging window when batch halves. Epoch0 weights are retained
as the fallback comparator and must not be silently replaced by a worse FT result.
