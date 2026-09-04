#!/usr/bin/env bash
set -euo pipefail

repo=$(readlink -f "${1:-.}")
py=${PYTHON_BIN:-/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python}
study=.experiments/w256_d16_60e_20260904
config=configs/pose3d/graph_posemamba_h36m_w256_d16_stable_r3_60e.yaml
out=$study/PRELAUNCH_PASS_R3.json
work=$study/preflight_r3

cd "$repo"
mkdir -p "$work"
exec 9>"$study/preflight_r3.lock"
flock -n 9 || { echo "R3 preflight lock is held" >&2; exit 3; }

if ! git diff --quiet -- lib/model/PoseMamba.py lib/model/mambablocks.py lib/utils/learning.py train.py tools/benchmark_training.py "$config"; then
  echo "tracked R3 source is dirty" >&2
  exit 4
fi

dataset=data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
expected_dataset=73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
actual_dataset=$(sha256sum "$dataset" | awk '{print $1}')
[[ "$actual_dataset" == "$expected_dataset" ]] || { echo "dataset hash mismatch" >&2; exit 5; }

source_commit=$(git rev-parse HEAD)
config_sha256=$(sha256sum "$config" | awk '{print $1}')
gpu_free_before=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

"$py" -m unittest tests.test_graph_conditioned_posemamba \
  > "$work/unit_tests.stdout" 2> "$work/unit_tests.stderr"

"$py" - "$config" > "$work/static.json" <<'PY'
import json
import sys

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from train import build_adamw_parameter_groups

config = get_config(sys.argv[1])
model = load_backbone(config)
groups = build_adamw_parameter_groups(
    model,
    weight_decay=config.weight_decay,
    honor_no_weight_decay=config.honor_no_weight_decay,
)
no_decay_names = [
    name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad and getattr(parameter, "_no_weight_decay", False)
]
print(json.dumps({
    "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "no_decay_parameter_count": sum(
        p.numel() for p in model.parameters()
        if p.requires_grad and getattr(p, "_no_weight_decay", False)
    ),
    "no_decay_parameter_tensors": no_decay_names,
    "optimizer_groups": [
        {
            "name": group["group_name"],
            "parameter_count": sum(p.numel() for p in group["params"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in groups
    ],
    "settings": {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "warmup_epochs": config.warmup_epochs,
        "warmup_start_factor": config.warmup_start_factor,
        "lr_schedule_mode": config.lr_schedule_mode,
        "min_lr_ratio": config.min_lr_ratio,
        "honor_no_weight_decay": config.honor_no_weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "track_parameter_update_norm": config.track_parameter_update_norm,
    },
}, indent=2))
PY

for batch in 1 2; do
  CUDA_VISIBLE_DEVICES=0 "$py" tools/benchmark_training.py \
    --config "$config" --batch-size "$batch" --real-data \
    --warmup-steps 1 --steps 1 --no-compile \
    > "$work/B${batch}.json" 2> "$work/B${batch}.stderr"
done
CUDA_VISIBLE_DEVICES=0 "$py" tools/benchmark_training.py \
  --config "$config" --batch-size 4 --real-data \
  --warmup-steps 1 --steps 1 \
  > "$work/B4.json" 2> "$work/B4.stderr"

"$py" - "$work" "$out" "$source_commit" "$config_sha256" "$actual_dataset" "$gpu_free_before" <<'PY'
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

work = Path(sys.argv[1])
output = Path(sys.argv[2])
static = json.loads((work / "static.json").read_text())
stages = {name: json.loads((work / f"{name}.json").read_text()) for name in ("B1", "B2", "B4")}
settings = static["settings"]
gates = {
    "parameter_count": static["parameters"] == 20_192_451,
    "optimizer_two_groups": [g["name"] for g in static["optimizer_groups"]] == ["decay", "no_decay"],
    "optimizer_no_decay_nonempty": static["no_decay_parameter_count"] > 0,
    "optimizer_no_decay_zero": static["optimizer_groups"][-1]["weight_decay"] == 0.0,
    "optimizer_decay_preserved": static["optimizer_groups"][0]["weight_decay"] == 0.012,
    "optimizer_group_coverage": sum(g["parameter_count"] for g in static["optimizer_groups"]) == static["parameters"],
    "r3_settings": settings == {
        "epochs": 60,
        "learning_rate": 0.0003,
        "warmup_epochs": 8,
        "warmup_start_factor": 0.1,
        "lr_schedule_mode": "cosine",
        "min_lr_ratio": 0.1,
        "honor_no_weight_decay": True,
        "max_grad_norm": 1.0,
        "track_parameter_update_norm": True,
    },
}
for name, stage in stages.items():
    expected_batch = int(name[1:])
    gates[f"{name}_batch"] = stage["batch_size"] == expected_batch
    gates[f"{name}_compile"] = stage["compiled"] == (name == "B4")
    gates[f"{name}_cosine"] = stage["lr_schedule_mode"] == "cosine"
    gates[f"{name}_finite"] = all(math.isfinite(stage[key]) for key in (
        "loss", "grad_norm_preclip", "grad_norm_max", "grad_clip_fraction"
    ))
    gates[f"{name}_clip_fraction"] = 0.0 <= stage["grad_clip_fraction"] <= 1.0
    gates[f"{name}_warmup_lr"] = 3e-5 <= stage["optimizer_lr_after_steps"] < 3.01e-5
    gates[f"{name}_memory"] = stage["measured_peak_reserved_mib"] < 28672
    gates[f"{name}_groups"] = stage["optimizer_groups"] == static["optimizer_groups"]
payload = {
    "status": "PASS" if all(gates.values()) else "BLOCKED",
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "source_commit": sys.argv[3],
    "config_sha256": sys.argv[4],
    "dataset_sha256": sys.argv[5],
    "gpu_free_before_mib": int(sys.argv[6].strip()),
    "parameters": static["parameters"],
    "optimizer_groups": static["optimizer_groups"],
    "no_decay_parameter_count": static["no_decay_parameter_count"],
    "settings": settings,
    "stages": stages,
    "gates": gates,
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(1)
PY

