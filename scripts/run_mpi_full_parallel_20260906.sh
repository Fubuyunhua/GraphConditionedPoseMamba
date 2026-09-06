#!/usr/bin/env bash
set -euo pipefail
cd /scratch/home/caiwei/GraphConditionedPoseMamba_MPI_FULL_20260906
py=/scratch/home/caiwei/miniforge3/envs/relipose_torch211/bin/python
exec 9>verification/formal.lock
flock -n 9
test ! -e verification/FORMAL_STARTED.json
test ! -e runs/mpi_full_seed0
"$py" - <<'PY'
import pathlib,json,subprocess,datetime,hashlib
root=pathlib.Path('.')
s=json.loads((root/'verification/mpi_full_b4/cuda_smoke.json').read_text())
m=json.loads((root/'verification/mpi_full_b4/run_manifest.json').read_text())
assert s['status']=='passed' and s['finite_gradients']
assert s['input_shape']==[4,81,17,3] and s['output_shape']==[4,81,17,3]
assert s['max_memory_allocated_mib']<12000
assert m['trainable_parameter_count']==789715
config=pathlib.Path('configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_protocol_v2_memopt.yaml')
assert hashlib.sha256(config.read_bytes()).hexdigest()==m['hashes']['config']
assert hashlib.sha256(pathlib.Path('train_3dhp.py').read_bytes()).hexdigest()==m['hashes']['trainer']
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
assert set(active)<= {'1102590'},active
if active:
 assert pathlib.Path('/proc/1102590/cwd').resolve()==pathlib.Path('/scratch/home/caiwei/GraphConditionedPoseMamba_CORRECTED_A0_20260906')
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=16000,free
p={'status':'PASS','source_commit':'3920d956ba46deb8d02498e4ad6373c7bbba8a39','source_archive_sha256':hashlib.sha256(pathlib.Path('source.tar').read_bytes()).hexdigest(),'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'user_authorized_parallel':True,'concurrent_pid':1102590 if active else None,'gpu_free_mib_before_launch':free,'smoke':s,'protocol_manifest':m,'command':'python -u train_3dhp.py --config configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_protocol_v2_memopt.yaml --checkpoint runs/mpi_full_seed0 --seed 0','throughput_use':'shared GPU engineering telemetry only'}
pathlib.Path('verification/PREFLIGHT_PASS.json').write_text(json.dumps(p,indent=2))
pathlib.Path('verification/FORMAL_STARTED.json').write_text(json.dumps(p,indent=2))
print(json.dumps(p),flush=True)
PY
set +e
CUDA_VISIBLE_DEVICES=0 "$py" -u train_3dhp.py --config configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_protocol_v2_memopt.yaml --checkpoint runs/mpi_full_seed0 --seed 0
code=$?
printf '%s\n' "$code" > verification/formal_exit_code.txt
exit "$code"
