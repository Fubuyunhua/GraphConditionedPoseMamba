"""Create default-Full fixture using an immutable prior checkout, read-only."""
import argparse,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
a=p.parse_args();sys.path.insert(0,a.root)
import torch
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
torch.manual_seed(0)
c=get_config(str(Path(a.root)/'configs/pose3d/repro_full_seed1_80e.yaml'))
m=load_backbone(c).cuda().eval()
x=torch.randn(1,9,17,3,device='cuda')
y=m(x);y.square().mean().backward()
torch.save({'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'x':x.cpu(),
            'y':y.detach().cpu(),'grads':{k:p.grad.detach().cpu() for k,p in m.named_parameters() if p.grad is not None}},a.out)
print('LEGACY FULL FIXTURE SAVED',flush=True)
