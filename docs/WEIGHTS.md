# Core checkpoints

Six EMA checkpoints are prepared for the three core models and separate detector/
GT2D protocols. They retain model tensors exactly while omitting optimizer states,
training logs and private filesystem metadata. Original checkpoints are archived
privately. The public manifest records SHA256, architecture, protocol and epoch.

Download from [GCS-Pose core weights](https://github.com/Fubuyunhua/GraphConditionedPoseMamba/releases/tag/gcs-pose-v1.0).
The [manifest](../checkpoints/manifest.json) records sizes, SHA256 hashes, epochs,
and input protocols. The downloader verifies both file size and SHA256:

```bash
python scripts/download_weights.py --model w128d20 --protocol detector
```

Use `--protocol gt2d` only with the matching GT configuration. All six exports
passed exact tensor roundtrip and strict loading against the public model code.
Only model state and minimal identity metadata are included; there is no optimizer
state for resuming the original training. Read the model card before comparing scores.
