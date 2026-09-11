"""User-authorized parallel ablations + conditional matched batch4 restart."""
import fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
def call(*args):subprocess.run([sys.executable,*args],check=True)
def launch(job):
    log=open('launch_logs/'+job+'.log','x')
    p=subprocess.Popen([sys.executable,'-u','tools/run_conditioning_job.py',job],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();return p.pid
def main():
    Path('verification').mkdir(exist_ok=True);Path('launch_logs').mkdir(exist_ok=True)
    lock=open('verification/controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>20000,'Insufficient space for bounded parallel preflight'
    call('tools/conditioning_reference.py','--root','/scratch/home/caiwei/GraphConditionedPoseMamba_W128_D10_BEST_FT_20260911','--out','verification/legacy_full.pt')
    call('-m','unittest','tests.test_conditioning_routes','tests.test_graph_conditioned_posemamba')
    call('tools/verify_conditioning_equivalence.py')
    for job in ('delta','bc','d10_b4'):call('tools/preflight_conditioning_job.py',job)
    status={'delta_launcher':launch('delta'),'bc_launcher':launch('bc'),'d10_b4':'WAITING_FOR_FT'}
    path=Path('verification/launch_state.json');path.write_text(json.dumps(status,indent=2))
    old=Path('/scratch/home/caiwei/GraphConditionedPoseMamba_W128_D10_BEST_FT_20260911')
    deadline=time.monotonic()+7200
    while not (old/'verification/formal_exit_code.txt').exists():
        if time.monotonic()>deadline:raise RuntimeError('FT completion timeout; no batch4 restart')
        time.sleep(20)
    assert (old/'verification/formal_exit_code.txt').read_text().strip()=='0','FT runtime failure requires diagnosis'
    text=(old/'launch_logs/finetune.log').read_text(errors='replace')
    rows=[(int(e),float(p)) for e,p in re.findall(r'\[(\d+)\] time .*? e1 ([\d.]+)',text)]
    assert len(rows)==8 and rows[-1][0]==8
    best=min(p for _,p in rows);status['ft_best']=best
    status['meaningful_threshold_mm']=39.38356650289372-.2
    if best>=status['meaningful_threshold_mm']:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        assert free>12000,'Insufficient space for matched batch4 restart'
        status['d10_b4_launcher']=launch('d10_b4');status['d10_b4']='STARTED_FROM_SCRATCH'
    else:status['d10_b4']='SKIPPED_FT_MEANINGFUL_GAIN'
    path.write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)

if __name__=='__main__':main()
