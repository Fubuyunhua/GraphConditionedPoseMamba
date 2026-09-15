# GCS-Pose source/recipe release

Scope: existing public repository main, branded as GCS-Pose; historical URL and
archives retained. New public entry points are six flat recipes, a scratch launcher,
English/Chinese README and reproduction/attribution documentation. No model,
training objective or evaluator changes are included in this release upload.
Local uncommitted Q/H and scheduler work is deliberately not mixed into the stable
public entry point. Native core remains the previously published implementation.

Validation: six recipe identities/protocols checked; launcher help checked without
ML imports; four counter arithmetic tests checked. No new CUDA installation,
full training, independent accuracy replay or clean-clone end-to-end run performed.
Workflow is limited to CPU configuration/counter checks and does not train.

Release exclusions: credentials, host addresses, user paths, datasets, videos,
checkpoints, private logs, raw evidence and local before-snapshots. Existing git
history is retained, not rewritten or represented as a freshly sanitized archive.

Outstanding: distributable checkpoint bundle and hashes, exact dependency lock,
complete preprocessing pipeline and clean-environment validation. Paper author list
and DOI are unknown and not invented. See docs/GCS_POSE_REPRODUCTION.md.
