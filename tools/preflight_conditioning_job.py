import argparse,gc,hashlib,itertools,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from train import EMAModel,build_adamw_parameter_groups,train_epoch,set_random_seed
from tools.benchmark_training import make_meters

CONFIGS={'delta':'ablation_condition_delta_only_80e.yaml','bc':'ablation_condition_bc_only_80e.yaml','d10_b4':'graph_posemamba_h36m_w128_d10_b4_matched_80e.yaml'}
def fingerprint(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=CONFIGS);a=p.parse_args()
    config='configs/pose3d/'+CONFIGS[a.job];c=get_config(config)
    expected=3435395 if a.job=='d10_b4' else 800083
    assert c.batch_size==4 and c.epochs==80 and c.ema_decay==.9998
    assert c.learning_rate==.0005 and c.lr_decay==.99 and c.weight_decay==.012
    assert c.max_grad_norm==0 and not c.enable_linear_warmup and not c.honor_no_weight_decay
    assert not c.gt_2d and not c.finetune and not c.pretrained and not c.resume
    assert c.graph_conditioning_targets==('all' if a.job=='d10_b4' else a.job)
    set_random_seed(0);initial=load_backbone(c)
    assert sum(p.numel() for p in initial.parameters())==expected
    initial_hash=fingerprint(initial);del initial
    dataset_hash=hashlib.sha256(Path('data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl').read_bytes()).hexdigest()
    assert dataset_hash=='73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175'
    out=Path('verification')/a.job;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(c),indent=2,default=str))
    report={'status':'PENDING','job':a.job,'parameters':expected,'initial_model_sha256':initial_hash,
            'config_sha256':hashlib.sha256(Path(config).read_bytes()).hexdigest(),'dataset_sha256':dataset_hash,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'stages':{}}
    c.mask=False;dataset=MotionDataset3D(c,c.subset_list,'train')
    for batch in (1,2,4):
        set_random_seed(0);base=load_backbone(c).cuda()
        assert fingerprint(base)==initial_hash
        groups=build_adamw_parameter_groups(base,c.weight_decay,honor_no_weight_decay=False)
        assert len(groups)==1 and sum(p.numel() for p in groups[0]['params'])==expected
        opt=torch.optim.AdamW(groups,lr=c.learning_rate);assert not opt.state
        ema=EMAModel(base,c.ema_decay);assert ema.num_updates==0
        loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0)
        if batch==4:
            torch._dynamo.config.recompile_limit=64
            model=torch.compile(base,mode=c.compile_mode)
        else:model=base
        torch.cuda.reset_peak_memory_stats();meters=make_meters()
        train_epoch(c,model,itertools.islice(loader,2),meters,opt,has_3d=True,has_gt=True,ema_helper=ema)
        gradients=[p.grad for p in base.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert all(torch.isfinite(p).all() for p in base.parameters()) and ema.num_updates==2
        assert torch.isfinite(torch.tensor(meters['total'].avg))
        graph_grad=sum(float(p.grad.abs().sum()) for n,p in base.named_parameters() if 'graph_mixer' in n and p.grad is not None)
        assert graph_grad>0
        sample=next(iter(loader))[0].cuda();base.eval()
        with torch.no_grad(),ema.average_parameters(base):
            before=base(sample);state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
        checkpoint=out/f'B{batch}.bin';torch.save(state,checkpoint)
        base.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        with torch.no_grad():torch.testing.assert_close(before,base(sample),atol=2e-5,rtol=2e-5)
        peak=torch.cuda.max_memory_reserved()/1024**2
        assert peak<(11000 if a.job=='d10_b4' else 6000)
        report['stages'][f'B{batch}']={'loss':meters['total'].avg,'graph_gradient_l1':graph_grad,'peak_reserved_mib':peak,'roundtrip':True}
        del gradients,base,model,opt,groups,ema,loader,sample,before,state
        gc.collect();torch.cuda.empty_cache()
    report['status']='PASS';(out/'PREFLIGHT_PASS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
