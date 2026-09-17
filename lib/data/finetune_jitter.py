"""Small train-only temporally correlated detector jitter; default path unchanged."""
import torch
from torch.nn import functional as F

def jitter_detector_input(x, std=0.0, probability=0.5):
    if std == 0 or probability == 0:
        return x
    if not 0 < std <= .005 or not 0 < probability <= 1:
        raise ValueError('Unregistered jitter magnitude/probability')
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError('Expected B,T,J,xy+confidence')
    b,t,j,_=x.shape
    keyframes=min(t,27)
    noise=torch.randn(b,j*2,keyframes,device=x.device,dtype=x.dtype)*std
    noise=F.interpolate(noise,size=t,mode='linear',align_corners=True)
    noise=noise.transpose(1,2).reshape(b,t,j,2)
    active=(torch.rand(b,1,1,1,device=x.device)<probability).to(x.dtype)
    result=x.clone()
    result[...,:2]=result[...,:2]+noise*active
    return result
