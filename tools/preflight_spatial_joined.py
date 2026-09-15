"""Fail-closed runtime gates for spatial-only joined recurrence."""
import gc,hashlib,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_motion_3d import MotionDataset3D
from train import EMAModel,build_adamw_parameter_groups,train_epoch,set_random_seed
from tools.benchmark_training import make_meters


def fingerprint(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    cfg='configs/pose3d/ablation_spatial_joined_temporal_independent_80e.yaml'
    c=get_config(cfg);ref=get_config('configs/pose3d/conditioning_full_reference_80e.yaml')
    assert c.epochs==80 and c.batch_size==4 and not(c.pretrained or c.resume or c.finetune)
    for k in ['learning_rate','lr_decay','weight_decay','ema_decay','drop_path_rate','max_grad_norm','enable_linear_warmup','honor_no_weight_decay','lambda_3d','lambda_scale','lambda_3d_velocity','lambda_diff','flip','no_conf','gt_2d']:
        assert c[k]==ref[k],k
    set_random_seed(0);full=load_backbone(ref).cuda().eval()
    set_random_seed(0);joined=load_backbone(c).cuda().eval()
    assert sum(p.numel() for p in joined.parameters())==800083
    initial=fingerprint(joined);assert initial==fingerprint(full)
    assert all(b.spatial_ssm.recurrence_scope=='joined' and b.temporal_ssm.recurrence_scope=='independent' for b in joined.blocks)
    fixture=torch.load('verification/old_full_fixture.bin',map_location='cpu',weights_only=True)
    full.load_state_dict(fixture['model'],strict=True)
    x=fixture['x'].cuda();y=full(x);torch.testing.assert_close(y,fixture['y'].cuda(),atol=2e-5,rtol=2e-5)
    y.square().mean().backward()
    for name,p in full.named_parameters():
        if name in fixture['grads']:torch.testing.assert_close(p.grad,fixture['grads'][name].cuda(),atol=2e-5,rtol=2e-5)
    joined.load_state_dict(full.state_dict(),strict=True)
    inp=torch.randn(3,1,17,64,device='cuda');ctx=torch.randn_like(inp);changed=inp.clone();changed[0]+=1
    with torch.no_grad():
        independent=full.blocks[0].spatial_ssm
        a=independent(inp,context=ctx,segments_per_sample=3);b=independent(changed,context=ctx,segments_per_sample=3)
        torch.testing.assert_close(a[1:],b[1:],atol=0,rtol=0)
        spatial=joined.blocks[0].spatial_ssm
        a=spatial(inp,context=ctx,segments_per_sample=3);b=spatial(changed,context=ctx,segments_per_sample=3)
        cross_effect=float((a[1:]-b[1:]).abs().max());assert cross_effect>1e-7
        temporal=joined.blocks[0].temporal_ssm
        a=temporal(inp,context=ctx,segments_per_sample=3);b=temporal(changed,context=ctx,segments_per_sample=3)
        torch.testing.assert_close(a[1:],b[1:],atol=0,rtol=0)
    del full,joined,fixture,x,y,a,b,inp,ctx,changed,independent,spatial,temporal
    gc.collect();torch.cuda.empty_cache()
    meta=Path(c.data_root)/c.dt_file
    with meta.open('rb') as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    assert h.hexdigest()=='73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175'
    report=dict(status='PENDING',parameters=800083,initial_model_sha256=initial,dataset_sha256=h.hexdigest(),
                default_parity='forward_and_gradients_PASS',spatial_cross_frame_effect=cross_effect,
                temporal_independence='PASS',effective_config=dict(c),stages={})
    c.mask=False;data=MotionDataset3D(c,c.subset_list,'train')
    out=Path('verification/spatial_joined');out.mkdir(parents=True,exist_ok=True)
    for bs in (1,2,4):
        set_random_seed(0);base=load_backbone(c).cuda();assert fingerprint(base)==initial
        groups=build_adamw_parameter_groups(base,c.weight_decay,honor_no_weight_decay=False)
        assert len(groups)==1
        opt=torch.optim.AdamW(groups,lr=c.learning_rate);ema=EMAModel(base,c.ema_decay)
        loader=DataLoader(data,batch_size=bs,shuffle=False,num_workers=0)
        torch._dynamo.config.recompile_limit=64
        model=torch.compile(base,mode=c.compile_mode) if bs==4 else base
        torch.cuda.reset_peak_memory_stats();meters=make_meters()
        train_epoch(c,model,itertools.islice(loader,2),meters,opt,has_3d=True,has_gt=True,ema_helper=ema)
        grads=[p.grad for p in base.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert all(torch.isfinite(p).all() for p in base.parameters()) and ema.num_updates==2
        graph_grad=sum(float(p.grad.abs().sum()) for n,p in base.named_parameters() if 'graph_mixer' in n and p.grad is not None)
        assert graph_grad>0
        base.eval();sample=next(iter(loader))[0].cuda()
        with torch.no_grad(),ema.average_parameters(base):
            expected=base(sample);state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
        base.load_state_dict(state,strict=True)
        with torch.no_grad():torch.testing.assert_close(expected,base(sample),atol=2e-5,rtol=2e-5)
        report['stages'][f'B{bs}']=dict(loss=meters['total'].avg,graph_gradient_l1=graph_grad,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,ema_roundtrip='PASS')
        assert report['stages'][f'B{bs}']['peak_reserved_mib']<12000
        del grads,base,model,opt,groups,ema,loader,sample,expected,state
        gc.collect();torch.cuda.empty_cache()
    report['status']='PASS'
    report['source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['lib/model/PoseMamba.py','lib/utils/learning.py','lib/model/mambablocks.py','train.py']}
    (out/'PREFLIGHT_PASS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
