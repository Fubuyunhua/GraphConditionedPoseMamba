"""Fit one output-scale parameter using training clips only, fold into same head."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import flip_data

SOURCE=Path('/scratch/home/caiwei/GraphConditionedPoseMamba_SCALE_80e_20260903/runs/model_scaling_80e/S1_w128_d20_seed0_2026_09_03_T_21_41_37/best_ema_epoch.bin')
SHA='6715a014823ac3b7365b8e8d64625486330f706f87191ddcdef2a313f599cbed'
def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    out=Path('verification/scale_diagnostic');out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    checkpoint=torch.load(SOURCE,map_location='cpu',weights_only=False)
    c=get_config('configs/pose3d/graph_posemamba_h36m_w128_d20_layerwise_ft_8e.yaml')
    c.flip=False
    dataset=MotionDataset3D(c,c.subset_list,'train')
    indices=np.random.default_rng(20260914).permutation(len(dataset))[:384]
    (out/'training_clip_indices.json').write_text(json.dumps(indices.tolist()))
    model=load_backbone(c).cuda().eval()
    model.load_state_dict(checkpoint['model_pos'],strict=True)
    model.requires_grad_(False)
    predictions=[];targets=[]
    with torch.no_grad():
        for x,y in DataLoader(Subset(dataset,indices.tolist()),batch_size=4,num_workers=0):
            x=x.cuda()
            p=(model(x)+flip_data(model(flip_data(x))))/2
            p[:,:,0]=0
            y=y-y[:,:,:1]
            predictions.append(p[:,::9].cpu());targets.append(y[:,::9])
    pred=torch.cat(predictions).double();target=torch.cat(targets).double()
    raw=torch.nn.Parameter(torch.zeros((),dtype=torch.double))
    opt=torch.optim.LBFGS([raw],lr=1,max_iter=50,tolerance_grad=1e-12,tolerance_change=1e-14,line_search_fn='strong_wolfe')
    def objective(p,y,s):return torch.linalg.vector_norm(p*s-y,dim=-1).mean()
    def closure():
        opt.zero_grad();s=1+.05*raw.tanh();loss=objective(pred[:256],target[:256],s);loss.backward();return loss
    opt.step(closure);scale=float((1+.05*raw.tanh()).detach())
    report={'source_sha256':SHA,'scale':scale,'fit_clips':256,'validation_clips':128,
            'selection_data':'training clips only; previously seen by original source model, not an independent held-out test',
            'fit_before':float(objective(pred[:256],target[:256],1)),
            'fit_after':float(objective(pred[:256],target[:256],scale)),
            'validation_before':float(objective(pred[256:],target[256:],1)),
            'validation_after':float(objective(pred[256:],target[256:],scale))}
    report['candidate_valid']=report['validation_after']<report['validation_before']
    if report['candidate_valid']:
        state={k:v.detach().clone() for k,v in checkpoint['model_pos'].items()}
        state['head.1.weight']*=scale;state['head.1.bias']*=scale
        model.load_state_dict(state,strict=True)
        with torch.no_grad():
            actual=(model(x)+flip_data(model(flip_data(x))))/2;actual[:,:,0]=0
        torch.testing.assert_close(actual,p*scale,atol=1e-5,rtol=1e-5)
        torch.save({'model_pos':state,'checkpoint_type':'train_calibrated_source_ema','epoch':0,
                    'calibration':report},out/'calibrated.bin')
        report['checkpoint_sha256']=hashlib.sha256((out/'calibrated.bin').read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
