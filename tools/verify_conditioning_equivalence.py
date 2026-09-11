import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
c=get_config('configs/pose3d/conditioning_full_reference_80e.yaml')
m=load_backbone(c).cuda().eval()
fixture=torch.load('verification/legacy_full.pt',map_location='cpu',weights_only=True)
m.load_state_dict(fixture['model'],strict=True)
x=fixture['x'].cuda();y=m(x);y.square().mean().backward()
torch.testing.assert_close(y.cpu(),fixture['y'],atol=2e-5,rtol=2e-5)
max_gradient_error=0.
for k,p in m.named_parameters():
    if k in fixture['grads']:
        torch.testing.assert_close(p.grad.cpu(),fixture['grads'][k],atol=5e-5,rtol=1e-4)
        max_gradient_error=max(max_gradient_error,float((p.grad.cpu()-fixture['grads'][k]).abs().max()))
states=m.state_dict()
c.graph_conditioning_targets='none'
none=load_backbone(c).cuda().eval();none.load_state_dict(states,strict=True)
a=get_config('configs/pose3d/ablation_factorized_only.yaml')
a1=load_backbone(a).cuda().eval()
a1.load_state_dict({k:states[k] for k in a1.state_dict()},strict=True)
with torch.no_grad():torch.testing.assert_close(none(x),a1(x),atol=2e-5,rtol=2e-5)
report={'status':'PASS','legacy_full_output_max_abs':float((y.cpu()-fixture['y']).abs().max()),
        'legacy_full_gradient_max_abs':max_gradient_error,'none_equals_a1_on_shared_weights':True,
        'full_parameters':sum(p.numel() for p in m.parameters()),'a1_parameters':sum(p.numel() for p in a1.parameters()),
        'note':'A1 reuses functional baseline but initialization/allocated graph parameters differ; not strict matched factorial interaction evidence'}
Path('verification/equivalence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
