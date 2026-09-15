"""Single authorized 80e run. Fail closed and never restart/stop on accuracy."""
import json,subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'verification/spatial_joined'
assert json.loads((out/'PREFLIGHT_PASS.json').read_text())['status']=='PASS'
with (out/'formal_started.json').open('x') as f:json.dump({'status':'STARTING'},f)
assert not (root/'runs/spatial_joined_20260915/seed0').exists()
cmd=[sys.executable,'-u','train.py','--config','configs/pose3d/ablation_spatial_joined_temporal_independent_80e.yaml','--checkpoint','runs/spatial_joined_20260915/seed0','--seed','0']
print('COMMAND',cmd,flush=True)
p=subprocess.Popen(cmd)
(out/'formal_started.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'status':'RUNNING'}))
code=p.wait();(out/'formal_exit_code.txt').write_text(str(code));sys.exit(code)
