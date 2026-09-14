"""Three preregistered independent strategies, serial execution, no auto sweep."""
import fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
ORDER=('no_droppath','no_diff','velocity10')
state={'order':ORDER,'status':'STARTING','completed':[]}
def save():
    Path('sequence_state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state),flush=True)
def run(args,log):
    with open(log,'x') as f:
        p=subprocess.Popen([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT)
        state['active_pid']=p.pid;save()
        code=p.wait()
    if code:raise RuntimeError(f'{args[0]} exited{code}; inspect{log}')
def main():
    import torch
    Path('launch_logs').mkdir(exist_ok=True);Path('verification').mkdir(exist_ok=True)
    lock=open('verification/sequence.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if Path('sequence_state.json').exists():raise RuntimeError('Refuse rerunning existing sequence')
    run(['-m','unittest','tests.test_layerwise_finetune'],'launch_logs/unit.log')
    for name in ORDER:
        state.update(status='PREFLIGHT',strategy=name);save()
        deadline=time.monotonic()+21600
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<20000:
            state['status']='WAITING_MEMORY';save()
            if time.monotonic()>deadline:raise RuntimeError('Memory wait timeout; sequence paused')
            time.sleep(30)
        run(['tools/preflight_ft_sequence.py',name],f'launch_logs/{name}_preflight.log')
        cfg=f'configs/pose3d/ft_sequence_{name}_8e.yaml'
        initializer=f'initializers/{name}/source_ema.bin'
        run(['train.py','--config',cfg,'--checkpoint',f'runs/initial_{name}',
             '--evaluate',initializer,'--seed','0'],f'launch_logs/{name}_initial.log')
        text=Path(f'launch_logs/{name}_initial.log').read_text(errors='replace')
        p1=float(re.findall(r'Protocol #1 Error \(MPJPE\):\s*([\d.]+)',text)[-1])
        p2=float(re.findall(r'Protocol #2 Error \(P-MPJPE\):\s*([\d.]+)',text)[-1])
        assert abs(p1-37.659301)<.02 and abs(p2-31.871747)<.02,(p1,p2)
        state.update(status='RUNNING',initial_p1=p1,initial_p2=p2);save()
        run(['train.py','--config',cfg,'--checkpoint',f'runs/ft_sequence_20260914/{name}_seed0',
             '--pretrained',f'initializers/{name}','--selection','source_ema.bin','--seed','0'],f'launch_logs/{name}.log')
        text=Path(f'launch_logs/{name}.log').read_text(errors='replace')
        rows=[(int(e),float(a),float(b)) for e,a,b in re.findall(r'\[(\d+)\] time .*? e1 ([\d.]+) e2 ([\d.]+)',text)]
        assert len(rows)==8 and rows[-1][0]==8,rows
        folders=list(Path('runs/ft_sequence_20260914').glob(name+'_seed0_*'))
        assert len(folders)==1
        for file in ('best_ema_epoch.bin','latest_ema_epoch.bin'):
            checkpoint=torch.load(folders[0]/file,map_location='cpu',weights_only=False)
            assert all(torch.isfinite(v).all() for v in checkpoint['model_pos'].values() if v.is_floating_point())
        best=min(rows,key=lambda row:row[1])
        result={'strategy':name,'status':'COMPLETED','initial_p1':p1,'initial_p2':p2,
                'best_epoch':best[0],'best_p1':best[1],'paired_p2':best[2],
                'fixed8_p1':rows[-1][1],'fixed8_p2':rows[-1][2],'gain_mm':p1-best[1],
                'meaningful_single_seed_gain':p1-best[1]>=.2,'rows':rows,'run_directory':str(folders[0])}
        Path(f'verification/{name}/result.json').write_text(json.dumps(result,indent=2))
        state['completed'].append(result);state['active_pid']=None;save()
    state.update(status='COMPLETED',strategy=None);save()
if __name__=='__main__':
    try:main()
    except Exception as error:
        state.update(status='FAILED',error=repr(error));save();raise
