"""Fresh W128 GT2D; explicitly no checkpoint used for initialization."""
import gc
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from train import EMAModel,build_adamw_parameter_groups,build_lr_schedule,train_epoch,set_random_seed
from tools.benchmark_training import make_meters

def main():
    cfg='configs/pose3d/graph_posemamba_h36m_w128_gt2d_scratch_80e.yaml'
    c=get_config(cfg)
    assert c.dim_feat==128 and c.depth==20 and c.epochs==80 and c.batch_size==4
    assert c.gt_2d and not c.finetune and not c.pretrained and not c.resume and not c.evaluate
    assert c.learning_rate==.0005 and not c.enable_linear_warmup and c.drop_path_rate==.2
    assert not c.selective_last_block_head and c.max_grad_norm==1
    out=Path('verification');out.mkdir(exist_ok=True)
    set_random_seed(0)
    initial=load_backbone(c)
    assert sum(p.numel() for p in initial.parameters())==6836355
    assert all(p.requires_grad for p in initial.parameters())
    def fingerprint(model):
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    initial_hash=fingerprint(initial)
    set_random_seed(0);repeat=load_backbone(c)
    assert fingerprint(repeat)==initial_hash
    del initial,repeat
    datahash=hashlib.sha256(Path('data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl').read_bytes()).hexdigest()
    assert datahash=='73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175'
    dataset=MotionDataset3D(c,c.subset_list,'train')
    for split in ('train','test'):
        probe=MotionDataset3D(c,c.subset_list,split)
        x,y=probe[0]
        assert x.shape==y.shape==(243,17,3)
        assert torch.equal(x[...,:2],y[...,:2]) and torch.all(x[...,2]==1)
    c.mask=False
    report={'status':'PENDING','initialization':'random seed0; no checkpoint loading',
            'initial_model_sha256':initial_hash,'dataset_sha256':datahash,
            'config_sha256':hashlib.sha256(Path(cfg).read_bytes()).hexdigest(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'parameters':6836355,'stages':{}}
    (out/'effective_config.json').write_text(json.dumps(dict(c),indent=2,default=str))
    for batch in (1,2,4):
        set_random_seed(0)
        base=load_backbone(c).cuda()
        assert fingerprint(base)==initial_hash
        groups=build_adamw_parameter_groups(base,c.weight_decay,honor_no_weight_decay=False)
        assert len(groups)==1 and sum(p.numel() for p in groups[0]['params'])==6836355
        optimizer=torch.optim.AdamW(groups,lr=c.learning_rate)
        assert len(optimizer.state)==0
        ema=EMAModel(base,c.ema_decay)
        assert ema.num_updates==0
        loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0)
        assert build_lr_schedule(c,optimizer,len(loader)) is None
        if batch==4:
            torch._dynamo.config.recompile_limit=64
            model=torch.compile(base,mode=c.compile_mode)
        else:model=base
        torch.cuda.reset_peak_memory_stats()
        meters=make_meters()
        train_epoch(c,model,itertools.islice(loader,2),meters,optimizer,has_3d=True,has_gt=True,ema_helper=ema)
        assert ema.num_updates==2
        assert all(torch.isfinite(p).all() for p in base.parameters())
        assert all(torch.isfinite(torch.tensor(meters[k].avg)) for k in ('total','grad_norm'))
        assert optimizer.param_groups[0]['lr']==.0005
        base.eval();sample=next(iter(loader))[0].cuda()
        with torch.no_grad(),ema.average_parameters(base):
            before=base(sample)
            state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
        path=out/f'B{batch}_roundtrip.bin';torch.save({'model_pos':state},path)
        reloaded=torch.load(path,map_location='cpu',weights_only=True)['model_pos']
        base.load_state_dict(reloaded,strict=True)
        with torch.no_grad():torch.testing.assert_close(base(sample),before,atol=1e-5,rtol=1e-5)
        peak=torch.cuda.max_memory_reserved()/1024**2
        assert peak<19000
        report['stages'][f'B{batch}']={'loss':meters['total'].avg,'grad_norm':meters['grad_norm'].avg,
            'peak_reserved_mib':peak,'roundtrip':True,'random_initialization_verified':True}
        del model,base,optimizer,groups,ema,state,reloaded,sample,before
        gc.collect();torch.cuda.empty_cache()
    report['status']='PASS'
    (out/'PREFLIGHT_PASS.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
