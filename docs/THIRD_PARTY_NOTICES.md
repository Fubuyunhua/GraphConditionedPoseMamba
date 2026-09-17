# Third-party acknowledgments

This project builds on [PoseMamba](https://github.com/nankingjing/PoseMamba),
the existing ReliPose/pose-lifting code lineage, and selective-scan implementations
from [Mamba](https://github.com/state-spaces/mamba). Preserve the root `LICENSE`,
`NOTICE` and individual source copyright/license headers when redistributing.
PoseMamba and Mamba upstream license files were checked as Apache-2.0 for this
release. Existing copyright notices are not replaced by the GCS-Pose project name.
Other dependencies retain their respective licenses. Dataset licenses are separate
and are not granted by this repository. No author list, DOI, publication acceptance,
or permission to redistribute third-party weights is implied.

## Relationship and contribution boundaries

GCS-Pose is a separate research extension of a PoseMamba-derived implementation,
not an official PoseMamba distribution or a claim of upstream endorsement.

- Inherited foundation: the PoseMamba pose-lifting implementation lineage,
  Mamba/selective-scan machinery, bidirectional scan concepts, and associated
  data/training utilities. These are not claimed as new GCS-Pose contributions.
- GCS-Pose implementation changes: separate factorized spatial/temporal
  recurrence, skeleton graph mixing, graph-enhanced context for Δ/B/C while
  pose features provide recurrent content U and output gate Z, and a
  compile-compatible scan wrapper. Describing these changes does not claim
  invention of state-space models or establish priority over all related work.
- Scientific comparisons must identify the actual PoseMamba version, input
  protocol and evaluation settings. Local resized baselines are not automatically
  equivalent to an upstream released model.

## Citation sources

PoseMamba by Yunlong Huang, Junshuo Liu, Ke Xian and Robert Caiming Qiu was published
at AAAI 2025. Use the [publisher page](https://ojs.aaai.org/index.php/AAAI/article/view/32401)
and [official citation entry](https://github.com/nankingjing/PoseMamba#citation)
for bibliographic metadata. For Mamba, use its
[official citation guidance](https://github.com/state-spaces/mamba#citation).
Please cite relevant underlying work alongside GCS-Pose; this request does not
replace or add conditions to the applicable software licenses.

This is an attribution summary, not an exhaustive legal audit of every dependency.
Existing source-file notices and the original root LICENSE remain authoritative;
they are not removed or relicensed by this documentation update.
