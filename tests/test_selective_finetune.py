import unittest
from types import SimpleNamespace
import torch
from lib.utils.selective_finetune import configure_selective, selective_train_mode, selective_lr_groups
from train import EMAModel, build_adamw_parameter_groups, build_lr_schedule

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.2)) for _ in range(16)])
        self.head = torch.nn.Linear(2, 1)
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)

class SelectiveTests(unittest.TestCase):
    def test_two_blocks(self):
        m = Tiny()
        a = SimpleNamespace(selective_last_block_head=True, selective_train_blocks=2,
            finetune=True, partial_train=None, learning_rate=5e-7, head_learning_rate=2e-6)
        configure_selective(m,a)
        m.train()
        selective_train_mode(m)
        self.assertFalse(m.blocks[13].training)
        self.assertTrue(m.blocks[14].training and m.blocks[15].training)
        self.assertTrue(all(p.requires_grad == n.startswith(('blocks.14.','blocks.15.','head.')) for n,p in m.named_parameters()))
        groups = selective_lr_groups(m,build_adamw_parameter_groups(m,.012),a)
        self.assertEqual([g['lr'] for g in groups],[5e-7,2e-6])

    def test_default_unchanged(self):
        m = Tiny()
        a = SimpleNamespace(selective_last_block_head=False)
        configure_selective(m, a)
        m.train()
        selective_train_mode(m)
        self.assertTrue(all(x.training for x in m.modules()))
        g = build_adamw_parameter_groups(m, .012)
        self.assertIs(selective_lr_groups(m, g, a), g)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_freeze_lr_and_ema(self):
        torch.manual_seed(0)
        m = Tiny()
        a = SimpleNamespace(selective_last_block_head=True, finetune=True, partial_train=None,
            learning_rate=1e-6, head_learning_rate=5e-6, enable_linear_warmup=True,
            warmup_epochs=1,warmup_start_factor=.1,lr_decay=.99,lr_schedule_mode='cosine',
            min_lr_ratio=.1,epochs=5)
        configure_selective(m, a)
        before = {n:p.detach().clone() for n,p in m.named_parameters()}
        ema = EMAModel(m, .9998999949995)
        groups = selective_lr_groups(m, build_adamw_parameter_groups(m,.012), a)
        opt = torch.optim.AdamW(groups,lr=a.learning_rate)
        sched = build_lr_schedule(a,opt,steps_per_epoch=4)
        self.assertEqual(sched.base_lrs,[1e-6,5e-6])
        for _ in range(3):
            m.train()
            selective_train_mode(m)
            self.assertFalse(m.blocks[0][1].training)
            self.assertTrue(m.blocks[-1][1].training)
            opt.zero_grad(set_to_none=True)
            sched.prepare_step()
            m(torch.ones(2,2)).square().mean().backward()
            opt.step()
            ema.update(m)
            sched.complete_step()
        for n,p in m.named_parameters():
            if n in m._selective_frozen_names:
                self.assertIsNone(p.grad)
                self.assertTrue(torch.equal(before[n],p))
                self.assertTrue(torch.equal(before[n],ema.shadow[n]))
        for prefix in ('blocks.15.', 'head.'):
            self.assertTrue(any(not torch.equal(before[n],p) for n,p in m.named_parameters() if n.startswith(prefix)))
        self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'],5)

if __name__ == '__main__':
    unittest.main()
