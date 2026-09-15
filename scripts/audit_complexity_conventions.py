"""Independent inference-only complexity audit. Never trains or changes models.

Separates vanilla THOP (official PoseMamba example), contraction MACs checked
by ATen dispatch, and analytical parallel-scan FLOPs converted consistently.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import warnings


def matrix_macs(left,right,output):
    k=left[-1]
    assert k == (right[-2] if len(right)>1 else right[-1])
    return math.prod(output)*k


def scan_cost(b,d,l,n):
    # Albert Gu: 9BDLN counts scalar FLOPs for parallel associative scan.
    # Skip D*u+y adds one multiply-add per scalar, i.e. BDL MACs.
    return {'parallel_core_flops':9*b*d*l*n,'skip_flops':2*b*d*l,
            'mac_equivalent':4.5*b*d*l*n+b*d*l,
            'old_mixed_count':9*b*d*l*n+b*d*l}


def model_specs(all_models=False):
    specs=[('PoseMamba-A0','configs/pose3d/posemamba_h36m_w64_d6_m1_current_eval.yaml',790083),
           ('GCPM-W64D8','configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m_memopt_speed.yaml',800083),
           ('GCPM-W128D10','configs/pose3d/graph_posemamba_h36m_w128_d10_b4_matched_80e.yaml',3435395),
           ('GCPM-W128D20','configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml',6836355),
           ('PoseMamba-L-file-architecture','configs/gcs_pose/baselines/posemamba_l_architecture.yaml',9450499)]
    if all_models:
        specs += [('GCPM-W160D32','configs/pose3d/graph_posemamba_h36m_w160_d32_max_det2d_80e.yaml',16534531),
                  ('GCPM-W256D10','configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml',12646107),
                  ('GCPM-W256D16','configs/pose3d/graph_posemamba_h36m_w256_d16_stable_r3_60e.yaml',20192451)]
    return specs


def main(root,out,all_models=False):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape, matmul_flop_jit
    from thop import profile
    sys.path.insert(0,str(root))
    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone

    class CountContractions(TorchDispatchMode):
        def __init__(self):
            super().__init__();self.macs=Counter();self.seen=Counter()
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            y=func(*args,**(kwargs or {}));name=func._schema.name;self.seen[name]+=1
            cost=0
            if name in ('aten::mm','aten::bmm','aten::matmul'):
                cost=matrix_macs(args[0].shape,args[1].shape,y.shape)
            elif name in ('aten::addmm','aten::baddbmm'):
                cost=matrix_macs(args[1].shape,args[2].shape,y.shape)
            elif name=='aten::linear': cost=y.numel()*args[0].shape[-1]
            elif name=='aten::einsum':
                equation=args[0].replace(' ','');operands=args[1]
                subs=equation.split('->')[0].split(',')
                assert len(subs)==2 and '...' not in equation
                dims={}
                for sub,tensor in zip(subs,operands):
                    for key,size in zip(sub,tensor.shape):
                        assert key not in dims or dims[key]==size
                        dims[key]=size
                cost=math.prod(dims.values())
            elif name in ('aten::convolution','aten::_convolution','aten::conv1d','aten::conv2d','aten::conv3d'):
                if name in ('aten::convolution','aten::_convolution'):
                    assert not args[6], 'Transposed conv not audited'
                cost=y.numel()*math.prod(args[1].shape[1:])
            if cost:self.macs[name]+=int(cost)
            return y

    class BroadcastProbe(torch.nn.Module):
        def forward(self,a,b):return a@b
    a=torch.ones(17,17,device='cuda');b=torch.ones(243,17,32,device='cuda')
    count=CountContractions()
    with torch.inference_mode(),count:y=BroadcastProbe()(a,b)
    expected=243*17*17*32
    assert sum(count.macs.values())==expected
    default_probe=int(FlopCountAnalysis(BroadcastProbe(),(a,b)).total())
    rows=[]
    configs=model_specs(all_models)
    for name,path,expected_params in configs:
        args=get_config(str(root/path));model=load_backbone(args).cuda().eval()
        torch.manual_seed(20260915)
        channels=2 if args.no_conf else 3
        x=torch.randn(1,243,17,channels,device='cuda')
        if channels==3:x[...,2]=1
        params=sum(p.numel() for p in model.parameters());assert params==expected_params
        counter=CountContractions()
        with torch.inference_mode(),counter:
            y=model(x)
        assert torch.isfinite(y).all() and y.shape==(1,243,17,3)
        dispatch_total=sum(counter.macs.values())
        scans=[];broadcasts=[]
        def scan_handler(inputs,outputs):
            b,d,l=get_shape(inputs[0]);n=get_shape(inputs[2])[1]
            scans.append(dict(B=b,D=d,L=l,N=n,**scan_cost(b,d,l,n)))
            return Counter() # Keep scan separate from contraction MACs.
        def matmul_handler(inputs,outputs):
            left,right=get_shape(inputs[0]),get_shape(inputs[1]);output=get_shape(outputs[0])
            correct=matrix_macs(left,right,output)
            broadcasts.append(dict(left=left,right=right,output=output,correct=correct,
                                   fvcore_default=int(matmul_flop_jit(inputs,outputs))))
            return Counter(matmul=correct)
        def einsum_handler(inputs,outputs):
            # fvcore's generic np.einsum_path text rounds large FLOP counts.
            # These model equations are two-operand contractions; count indices
            # exactly rather than parsing scientific notation from that report.
            equation=inputs[0].toIValue().replace(' ','')
            subs=equation.split('->')[0].split(',')
            shapes=[get_shape(v) for v in inputs[1].node().inputs()]
            assert len(subs)==len(shapes)==2 and '...' not in equation
            dims={}
            for sub,shape in zip(subs,shapes):
                for key,size in zip(sub,shape):
                    assert key not in dims or dims[key]==size
                    dims[key]=size
            return Counter(einsum=math.prod(dims.values()))
        analysis=FlopCountAnalysis(model,(x,))
        for op in ('posemamba::selective_scan_core_fwd','prim::PythonOp.SelectiveScanCore','prim::PythonOp.SelectiveScanMamba','prim::PythonOp.SelectiveScanOflex'):
            analysis.set_op_handle(op,scan_handler)
        for op in ('prim::PythonOp.CrossScan1DBidirectional','prim::PythonOp.CrossMerge1DBidirectional','prim::PythonOp.CrossScan_plus_poselimbs','prim::PythonOp.CrossMerge_plus_poselimbs'):
            analysis.set_op_handle(op,lambda i,o:Counter())
        analysis.set_op_handle('aten::matmul',matmul_handler)
        analysis.set_op_handle('aten::einsum',einsum_handler)
        analysis.set_op_handle('aten::layer_norm',lambda i,o:Counter())
        with torch.inference_mode():dense=int(analysis.total())
        assert dense==dispatch_total,(name,'Independent counters disagree',dense,dispatch_total)
        assert len(scans)==args.depth*2,(name,len(scans))
        unsupported=dict(analysis.unsupported_ops())
        allowed={'aten::add','aten::add_','aten::sub','aten::mul','aten::mul_','aten::div','aten::neg','aten::exp','aten::gelu','aten::silu','aten::flip','aten::softplus','aten::sum'}
        assert not(set(unsupported)-allowed),(name,unsupported)
        # THOP is run on a fresh instance: do not let its profiling buffers
        # contaminate independent dispatch/JIT measurements or saved state.
        other=load_backbone(args).cuda().eval()
        with warnings.catch_warnings(record=True) as ws:
            with torch.inference_mode():thop_ops,thop_params=profile(other,inputs=(x,),verbose=False)
        assert sum(p.numel() for p in other.parameters())==params
        scan_macs=sum(s['mac_equivalent'] for s in scans)
        row=dict(model=name,true_parameters=params,vanilla_thop_parameters=int(thop_params),
                 vanilla_thop_macs=int(thop_ops),input_shape=list(x.shape),
                 contraction_macs=dense,dispatch_macs=dispatch_total,counter_agreement=True,
                 parallel_scan_mac_equivalent=scan_macs,
                 corrected_total_mac_equivalent=dense+scan_macs,
                 corrected_per_frame_mac_equivalent=(dense+scan_macs)/243,
                 counted_scope_flops=2*dense+sum(s['parallel_core_flops']+s['skip_flops'] for s in scans),
                 scan_calls=scans,broadcast_matmuls=broadcasts,by_operator=dict(analysis.by_operator()),
                 dispatch_by_operator=dict(counter.macs),excluded_pointwise=unsupported,
                 thop_warnings=[str(w.message) for w in ws],
                 config_sha256=hashlib.sha256((root/path).read_bytes()).hexdigest())
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k in ['model','true_parameters','vanilla_thop_parameters','vanilla_thop_macs','contraction_macs','corrected_total_mac_equivalent']}),flush=True)
        del model,other,x,y,analysis,counter
    result=dict(status='COUNTERS_AGREE',versions={p:importlib.metadata.version(p) for p in ['torch','thop','fvcore']},
                torchprofile='not installed; official handlers inspected, no package installed',
                broadcast_probe=dict(default_fvcore=default_probe,manual=expected,dispatch=expected),models=rows,
                contract='Contraction MACs exact under dense convolution convention; scan is analytic parallel-algorithm FLOPs/2 plus skip MAC. Norm/activation/other pointwise ops, exp, softplus, memory and backward excluded. Not exact hardware instructions.',
                sources=['https://github.com/nankingjing/PoseMamba/blob/main/lib/model/PoseMamba.py',
                         'https://github.com/TaatiTeam/MotionAGFormer/blob/master/model/MotionAGFormer.py',
                         'https://github.com/state-spaces/mamba/issues/110#issuecomment-1919470069',
                         'https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/jit_handles.py'])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);p.add_argument('--all-nonablation',action='store_true');a=p.parse_args();main(a.root.resolve(),a.output,a.all_nonablation)
