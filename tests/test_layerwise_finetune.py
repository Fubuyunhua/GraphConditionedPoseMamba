import unittest
from types import SimpleNamespace
import torch
from lib.utils.layerwise_finetune import layerwise_lr_groups
from train import build_adamw_parameter_groups,build_lr_schedule

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed_dim=128
        self.input=torch.nn.Linear(2,2)
        self.blocks=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(20)])
        self.head=torch.nn.Linear(2,1)

class Tests(unittest.TestCase):
    def test_default_identity(self):
        m=Tiny();g=build_adamw_parameter_groups(m,.012)
        self.assertIs(layerwise_lr_groups(m,g,SimpleNamespace(layerwise_finetune=False)),g)
    def test_groups_schedule_and_all_updates(self):
        m=Tiny();a=SimpleNamespace(layerwise_finetune=True,finetune=True,gt_2d=False,
          selective_last_block_head=False,learning_rate=1e-6,late_learning_rate=3e-6,
          head_learning_rate=1e-5,enable_linear_warmup=True,warmup_epochs=1,
          warmup_start_factor=.1,lr_schedule_mode='cosine',min_lr_ratio=.1,lr_decay=.99,epochs=8)
        groups=layerwise_lr_groups(m,build_adamw_parameter_groups(m,.012),a)
        self.assertEqual([g['group_name'] for g in groups],['input_decay','early_decay','late_decay','head_decay'])
        opt=torch.optim.AdamW(groups,lr=1e-6);sched=build_lr_schedule(a,opt,4)
        self.assertEqual(sched.base_lrs,[1e-6,1e-6,3e-6,1e-5])
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        sched.prepare_step();sum(p.sum() for p in m.parameters()).backward();opt.step()
        for n,p in m.named_parameters():self.assertFalse(torch.equal(p,before[n]),n)
        self.assertAlmostEqual(opt.param_groups[2]['lr']/opt.param_groups[0]['lr'],3)
        self.assertAlmostEqual(opt.param_groups[3]['lr']/opt.param_groups[0]['lr'],10)

if __name__=='__main__':unittest.main()
