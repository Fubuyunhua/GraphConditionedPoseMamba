#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_MPI_FULL_20260906
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
exec 9>verification/testbest.lock
flock -n 9
test -f verification/policy_change_20260906/EVAL_PASS
test ! -e verification/TESTBEST_STARTED.json
"$py" - <<'PY'
import torch,json,pathlib,hashlib,datetime,math
from lib.utils.tools import get_config
from train_3dhp import _should_evaluate_epoch
p=pathlib.Path('verification/policy_change_20260906')
assert _should_evaluate_epoch('legacy_test_best',False)
a=get_config('configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_protocol_v2_memopt.yaml')
b=get_config('configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_testbest.yaml')
changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
assert changed=={'config','name','checkpoint','pretrained','mpi3dhp_checkpoint_selection'},changed
best={}
payloads={}
for kind in ['raw','ema']:
 x=torch.load(p/f'original_{kind}_epoch1.bin',map_location='cpu',weights_only=False)
 assert x['epoch']==1 and x['checkpoint_type']==kind
 assert all(torch.isfinite(v).all() for v in x['model_pos'].values() if torch.is_floating_point(v))
 assert x['optimizer'] and x['rng_state'] and x['ema_shadow']
 e=json.loads((p/f'eval_{kind}/evaluation.json').read_text())
 assert all(math.isfinite(e[k]) for k in ['mpjpe_mm','p_mpjpe_mm','pck_150_percent','auc_0_150_percent'])
 best.update({f'{kind}_epoch':1,f'{kind}_p1_mm':e['mpjpe_mm'],f'{kind}_p2_mm':e['p_mpjpe_mm'],f'{kind}_pck_150_percent':e['pck_150_percent'],f'{kind}_auc_0_150_percent':e['auc_0_150_percent']})
 payloads[kind]=x
out=pathlib.Path('runs/mpi_full_testbest_seed0')
assert not out.exists(),str(out)
out.mkdir(parents=True)
for kind,x in payloads.items():
 x['best_metrics']=best.copy();x['min_loss']=best[f'{kind}_p1_mm']
 torch.save(x,out/('best_epoch.bin' if kind=='raw' else 'best_ema_epoch.bin'))
torch.save(payloads['raw'],p/'resume_raw_epoch1_with_best.bin')
identity={'status':'PASS','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'change':'User-selected per-epoch test-monitored best EMA MPJPE; fixed120 secondary','resumed_epoch':1,'best_metrics_initialized_from_epoch1':best,'config_sha256':hashlib.sha256(pathlib.Path('configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_testbest.yaml').read_bytes()).hexdigest(),'trainer_sha256':hashlib.sha256(pathlib.Path('train_3dhp.py').read_bytes()).hexdigest(),'resume_sha256':hashlib.sha256((p/'resume_raw_epoch1_with_best.bin').read_bytes()).hexdigest(),'preserved_state':'model, optimizer, EMA, LR and RNG copied without alteration; only best metric metadata populated','prior_run':'runs/mpi_full_seed0','first_epoch_evaluation':'verification/policy_change_20260906/eval_ema/evaluation.json','discarded_work':'partial epoch2 before policy transition; re-executed from saved epoch1'}
pathlib.Path('verification/TESTBEST_STARTED.json').write_text(json.dumps(identity,indent=2))
print(json.dumps(identity),flush=True)
PY
# This new directory contains only the two intentionally seeded best checkpoints.
set +e
CUDA_VISIBLE_DEVICES=0 "$py" -u train_3dhp.py --config configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_testbest.yaml --checkpoint runs/mpi_full_testbest_seed0 --resume verification/policy_change_20260906/resume_raw_epoch1_with_best.bin --allow-overwrite --seed 0
code=$?
printf '%s\n' "$code" > verification/testbest_exit_code.txt
exit "$code"
