#!/usr/bin/env bash
set -euo pipefail

repo=$(readlink -f "${1:-.}")
ablation_repo=$(readlink -f "${2:-/scratch/home/caiwei/GraphConditionedPoseMamba_5090_20260902}")
py=${PYTHON_BIN:-/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python}
study=.experiments/model_scaling_80e_20260903
ablation_study=.experiments/minimal_ablation_80e_20260903
threshold_mib=24576

cd "$repo"
mkdir -p "$study/preflight" "$study/runtime"
exec 9>"$study/preflight.lock"
flock -n 9 || { echo "scale preflight lock is held" >&2; exit 3; }

[[ -f "$study/USER_AUTHORIZED.txt" ]] || {
  echo "missing user authorization record" >&2
  exit 4
}
[[ -f "$ablation_repo/$ablation_study/SEQUENCE_COMPLETE.txt" ]] || {
  echo "A1/A2 sequence is not complete" >&2
  exit 5
}
for id in A1 A2; do
  [[ -f "$ablation_repo/$ablation_study/runtime/${id}_COMPLETE.txt" ]] || {
    echo "$id completion marker is missing" >&2
    exit 6
  }
done

if ! git diff --quiet -- lib/model/PoseMamba.py lib/model/mambablocks.py \
  lib/utils/learning.py train.py tools/benchmark_training.py \
  tools/preflight_scale_models.py \
  configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml \
  configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml; then
  echo "tracked scale source is dirty" >&2
  exit 7
fi

active=$(
  for pid in $(pgrep -f '[p]ython.*train\.py' || true); do
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    if [[ "$cwd" == "$repo" ]]; then
      ps -p "$pid" -o pid=,args=
    fi
  done
)
[[ -z "$active" ]] || { echo "active target-repo training: $active" >&2; exit 8; }

dataset=data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
expected=73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
actual=$(sha256sum "$dataset" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "dataset hash mismatch" >&2; exit 9; }

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "invalid free-memory reading" >&2; exit 10; }
(( free_mib >= threshold_mib )) || {
  echo "only ${free_mib} MiB GPU memory is free; require ${threshold_mib}" >&2
  exit 11
}

"$py" -m unittest tests.test_graph_conditioned_posemamba -v \
  > "$study/preflight/unit_tests.stdout" \
  2> "$study/preflight/unit_tests.stderr"
"$py" tools/preflight_scale_models.py \
  --output "$study/static_preflight.json" \
  > "$study/preflight/static_preflight.stdout" \
  2> "$study/preflight/static_preflight.stderr"

for item in \
  "S1 configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml" \
  "S2 configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml"; do
  read -r id config <<<"$item"
  for batch in 1 2; do
    CUDA_VISIBLE_DEVICES=0 "$py" tools/benchmark_training.py \
      --config "$config" \
      --warmup-steps 1 \
      --steps 1 \
      --real-data \
      --batch-size "$batch" \
      --no-compile \
      > "$study/preflight/${id}_B${batch}_real_smoke.json" \
      2> "$study/preflight/${id}_B${batch}_real_smoke.stderr"
  done
  CUDA_VISIBLE_DEVICES=0 "$py" tools/benchmark_training.py \
    --config "$config" \
    --warmup-steps 1 \
    --steps 1 \
    --real-data \
    --batch-size 4 \
    > "$study/preflight/${id}_B4_real_smoke.json" \
    2> "$study/preflight/${id}_B4_real_smoke.stderr"
done

"$py" - "$study" "$actual" "$threshold_mib" <<'PY'
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

study = Path(sys.argv[1])
dataset_hash = sys.argv[2]
threshold = float(sys.argv[3])
static = json.loads((study / "static_preflight.json").read_text())
smokes = {}
gates = {"static_preflight": static.get("status") == "PASS"}
for key in ("S1", "S2"):
    stages = {}
    for batch in (1, 2, 4):
        smoke = json.loads(
            (study / "preflight" / f"{key}_B{batch}_real_smoke.json").read_text()
        )
        stages[f"B{batch}"] = smoke
        gates[f"{key}_B{batch}_compile_mode"] = (
            smoke.get("compiled") is (batch == 4)
        )
        gates[f"{key}_B{batch}_real_data"] = smoke.get("real_data") is True
        gates[f"{key}_B{batch}_batch"] = smoke.get("batch_size") == batch
        gates[f"{key}_B{batch}_finite_loss"] = math.isfinite(
            float(smoke.get("loss", float("nan")))
        )
        gates[f"{key}_B{batch}_memory_below_threshold"] = float(
            smoke.get("measured_peak_reserved_mib", float("inf"))
        ) < threshold
        gates[f"{key}_B{batch}_positive_throughput"] = float(
            smoke.get("iterations_per_second", 0)
        ) > 0
    smokes[key] = stages
configs = [
    Path("configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml"),
    Path("configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml"),
]
report = {
    "status": "PASS" if all(gates.values()) else "FAIL",
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "dataset_sha256": dataset_hash,
    "memory_threshold_mib": threshold,
    "config_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in configs
    },
    "static": static,
    "smokes": smokes,
    "gates": gates,
}
target = study / "PRELAUNCH_PASS.json"
with tempfile.NamedTemporaryFile("w", dir=study, delete=False, encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")
    temporary = handle.name
if report["status"] != "PASS":
    os.unlink(temporary)
    raise SystemExit(json.dumps(report, indent=2))
os.replace(temporary, target)
print(json.dumps(report, indent=2))
PY
