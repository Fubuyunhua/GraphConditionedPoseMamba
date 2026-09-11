import unittest
import torch
from lib.model.conditioning_routes import route_projection
from lib.model.mambablocks import FactorizedBiSSM

class RouteTests(unittest.TestCase):
    def test_slice_gradients(self):
        for target in ('all','none','delta','bc'):
            content=torch.randn(1,2,11,7,requires_grad=True)
            context=torch.randn_like(content,requires_grad=True)
            out=route_projection(content,context,3,4,target)
            out.sum().backward()
            if target=='all':self.assertIsNone(content.grad)
            elif target=='none':self.assertIsNone(context.grad)
            elif target=='delta':
                self.assertEqual(content.grad[...,:3,:].abs().sum(),0)
                self.assertEqual(context.grad[...,3:,:].abs().sum(),0)
                self.assertGreater(context.grad[...,:3,:].abs().sum(),0)
            else:
                self.assertEqual(context.grad[...,:3,:].abs().sum(),0)
                self.assertEqual(content.grad[...,3:,:].abs().sum(),0)
                self.assertGreater(context.grad[...,3:,:].abs().sum(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA kernel required')
    def test_actual_scan_parameters_and_content(self):
        torch.manual_seed(7)
        m=FactorizedBiSSM(d_model=16,d_state=4,ssm_ratio=2,d_conv=3,axis='temporal',compile_compatible_scan=True).cuda()
        m.__DEBUG__=True
        x=torch.randn(2,m.d_inner,1,7,device='cuda')
        ctx=torch.randn_like(x)
        for no_einsum in (False,True):
            observed={}
            for mode in ('all','none','delta','bc'):
                m.graph_conditioning_targets=mode
                m.forward_core(x,context=ctx,no_einsum=no_einsum)
                observed[mode]={k:m.__data__[k].detach().clone() for k in ('us','dts','Bs','Cs')}
            for mode in observed:
                torch.testing.assert_close(observed[mode]['us'],observed['all']['us'],rtol=0,atol=0)
            for field in ('Bs','Cs'):
                torch.testing.assert_close(observed['delta'][field],observed['none'][field])
                torch.testing.assert_close(observed['bc'][field],observed['all'][field])
            torch.testing.assert_close(observed['delta']['dts'],observed['all']['dts'])
            torch.testing.assert_close(observed['bc']['dts'],observed['none']['dts'])
            self.assertGreater((observed['all']['dts']-observed['none']['dts']).abs().sum(),0)

if __name__=='__main__':unittest.main()
