import json,subprocess,sys
from pathlib import Path
from preflight_conditioning_job import CONFIGS
job=sys.argv[1]
config=CONFIGS[job]
out=Path('verification')/job
assert json.loads((out/'PREFLIGHT_PASS.json').read_text())['status']=='PASS'
marker=out/'formal_started.json'
with marker.open('x') as f:json.dump({'job':job,'status':'STARTING'},f)
command=[sys.executable,'-u','train.py','--config','configs/pose3d/'+config,
         '--checkpoint','runs/conditioning_20260911/'+job+'_seed0','--seed','0']
print('COMMAND',command,flush=True)
process=subprocess.Popen(command)
marker.write_text(json.dumps({'job':job,'pid':process.pid,'command':command}))
code=process.wait();(out/'formal_exit_code.txt').write_text(str(code))
sys.exit(code)
