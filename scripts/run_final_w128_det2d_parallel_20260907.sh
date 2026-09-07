#!/usr/bin/env bash
set -euo pipefail

root=/scratch/home/caiwei/GraphConditionedPoseMamba_FINAL_W128_DET2D_20260907
cd "$root"
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
config=configs/pose3d/graph_posemamba_h36m_final_w128_d20_det2d.yaml
run_prefix=runs/final_two_sizes_20260906/w128_det2d_seed0

mkdir -p verification launch_logs runs/final_two_sizes_20260906
exec 9>verification/final_w128.lock
flock -n 9
test ! -e verification/FORMAL_STARTED.json
if compgen -G "${run_prefix}_*" >/dev/null; then
  echo "refusing existing FINAL-W128 run" >&2
  exit 8
fi

active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u)
test "$active" = "1136724"
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
(( free_mib >= 16000 ))

"$py" - <<'PY'
import datetime
import hashlib
import json
import math
import pathlib
import torch

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from train import build_adamw_parameter_groups

root = pathlib.Path('.')
report = json.loads(pathlib.Path('verification/PREFLIGHT_PASS.json').read_text())
assert report['status'] == 'PASS'
assert report['parallel_with_pid'] == 1136724
for name in ('B1', 'B2', 'B4'):
    stage = report['stages'][name]
    assert math.isfinite(stage['loss'])
    assert math.isfinite(stage['grad_norm_preclip'])
    assert stage['measured_peak_reserved_mib'] < 20000

config_path = pathlib.Path(
    'configs/pose3d/graph_posemamba_h36m_final_w128_d20_det2d.yaml'
)
config = get_config(str(config_path))
assert config.dim_feat == 128 and config.depth == 20
assert config.epochs == 80 and config.drop_path_rate == 0.25
assert config.batch_size == 4 and config.seed == 0 and not config.gt_2d
assert config.enable_linear_warmup and config.warmup_epochs == 8
assert config.learning_rate == 0.0003
assert config.lr_schedule_mode == 'cosine' and config.min_lr_ratio == 0.1
assert config.honor_no_weight_decay and config.max_grad_norm == 1.0
assert config.use_ema and config.ema_decay == 0.9998
assert not config.no_eval
model = load_backbone(config)
assert sum(parameter.numel() for parameter in model.parameters()) == 6836355
groups = build_adamw_parameter_groups(
    model,
    weight_decay=config.weight_decay,
    honor_no_weight_decay=config.honor_no_weight_decay,
)
assert [group['group_name'] for group in groups] == ['decay', 'no_decay']
assert groups[1]['weight_decay'] == 0.0

manifest = json.loads(pathlib.Path('verification/source_manifest.json').read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    assert actual == expected, name

dataset = pathlib.Path('data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl')
dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
assert dataset_hash == '73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175'
deployment = pathlib.Path('verification/deployment_source.txt').read_text().strip()
identity = {
    'status': 'PASS',
    'user_parallel_authorization': True,
    'parallel_with_pid': 1136724,
    'deployment_source': f'git:{deployment}',
    'dataset_sha256': dataset_hash,
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'parameters': 6836355,
    'dim_feat': 128,
    'depth': 20,
    'drop_path_rate': 0.25,
    'preflight': report,
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'command': (
        'python train.py --config '
        'configs/pose3d/graph_posemamba_h36m_final_w128_d20_det2d.yaml '
        '--checkpoint runs/final_two_sizes_20260906/w128_det2d_seed0 --seed 0'
    ),
}
pathlib.Path('verification/effective_config.json').write_text(
    json.dumps(dict(config), indent=2, default=str)
)
pathlib.Path('verification/FORMAL_STARTED.json').write_text(
    json.dumps(identity, indent=2)
)
print(json.dumps(identity), flush=True)
PY

set +e
CUDA_VISIBLE_DEVICES=0 "$py" -u train.py \
  --config "$config" \
  --checkpoint "$run_prefix" \
  --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
