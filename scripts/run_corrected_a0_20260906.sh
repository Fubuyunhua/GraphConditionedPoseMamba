#!/usr/bin/env bash
set -euo pipefail

root=/scratch/home/caiwei/GraphConditionedPoseMamba_CORRECTED_A0_20260906
cd "$root"
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
run_prefix=runs/paper_remaining_evidence/posemamba_corrected_backward_seed0

mkdir -p verification launch_logs runs/paper_remaining_evidence
exec 9>verification/corrected_a0.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test ! -e verification/FORMAL_STARTED.json
if compgen -G "${run_prefix}_*" >/dev/null; then
  echo "refusing existing corrected-A0 run" >&2
  exit 8
fi

"$py" - <<'PY'
import datetime
import hashlib
import json
import math
import pathlib
import torch

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config

root = pathlib.Path('.')
benchmark_log = (root / 'verification/corrected_a0_b4.log').read_text()
benchmark = json.loads(benchmark_log[benchmark_log.rfind('\n{') + 1:])
assert benchmark['real_data'] and benchmark['batch_size'] == 4
assert not benchmark['compiled'] and math.isfinite(benchmark['loss'])
assert benchmark['iterations_per_second'] > 0
assert max(
    benchmark['warmup_peak_reserved_mib'],
    benchmark['measured_peak_reserved_mib'],
) < 28672
assert not benchmark['linear_warmup_enabled']
assert benchmark['optimizer_lr_after_steps'] == 0.0005

config_path = pathlib.Path(
    'configs/pose3d/diagnostic_posemamba_corrected_backward.yaml'
)
config = get_config(str(config_path))
assert config.epochs == 80 and config.seed == 0
assert config.backbone == 'PoseMamba'
assert config.posemamba_backward_mode == 'exact'
assert not config.compile_model and not config.enable_linear_warmup
assert not config.honor_no_weight_decay and config.max_grad_norm == 0
model = load_backbone(config)
assert sum(parameter.numel() for parameter in model.parameters()) == 790083

manifest = json.loads(pathlib.Path('verification/source_manifest.json').read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    assert actual == expected, name

dataset = pathlib.Path('data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl')
dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
assert dataset_hash == '73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175'

deployment_source = pathlib.Path('verification/deployment_source.txt').read_text().strip()
identity = {
    'status': 'PASS',
    'deployment_source': f'git:{deployment_source}',
    'scientific_implementation': 'ad4e2fa66492737a3fcd88d9142a88731724f30b',
    'dataset_sha256': dataset_hash,
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'parameters': 790083,
    'backward_mode': 'exact',
    'benchmark': benchmark,
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'command': (
        'python train.py --config '
        'configs/pose3d/diagnostic_posemamba_corrected_backward.yaml '
        '--checkpoint '
        'runs/paper_remaining_evidence/posemamba_corrected_backward_seed0 '
        '--seed 0'
    ),
}
pathlib.Path('verification/effective_config.json').write_text(
    json.dumps(dict(config), indent=2, default=str)
)
pathlib.Path('verification/PREFLIGHT_PASS.json').write_text(
    json.dumps(identity, indent=2)
)
pathlib.Path('verification/FORMAL_STARTED.json').write_text(
    json.dumps(identity, indent=2)
)
print(json.dumps(identity), flush=True)
PY

set +e
CUDA_VISIBLE_DEVICES=0 "$py" -u train.py \
  --config configs/pose3d/diagnostic_posemamba_corrected_backward.yaml \
  --checkpoint "$run_prefix" \
  --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
