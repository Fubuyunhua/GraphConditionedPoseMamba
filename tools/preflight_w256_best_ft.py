"""User-authorized weights-only R3 fine-tune: identity, reset and real-data gates."""
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from train import EMAModel, build_adamw_parameter_groups, build_lr_schedule, train_epoch, set_random_seed
from tools.benchmark_training import make_meters

CONFIG = 'configs/pose3d/graph_posemamba_h36m_w256_r3_best_ft_b2_15e.yaml'
SOURCE = Path('/scratch/home/caiwei/GraphConditionedPoseMamba_W256_D16_R3_60e_20260905/runs/w256_d16_r3_60e/D16_w256_d16_stable_r3_seed0_2026_09_05_T_00_48_32/best_ema_epoch.bin')
EXPECTED = 'c2175c027329e2d323bba5ceb3e613d2391f8dbf48a9e5cf6477eaf83bce7c7f'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    set_random_seed(0)
    out = Path('verification')
    out.mkdir(exist_ok=True)
    init = Path('initializers/r3_best_ema.bin')
    init.parent.mkdir(exist_ok=True)
    assert not init.exists(), 'Never overwrite initializer'
    assert sha(SOURCE) == EXPECTED
    payload = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert payload['epoch'] == 32 and payload['checkpoint_type'] == 'ema'
    weights = payload['model_pos']
    assert all(torch.isfinite(v).all() for v in weights.values() if v.is_floating_point())
    c = get_config(CONFIG)
    assert c.dim_feat == 256 and c.depth == 16 and c.epochs == 15
    assert c.batch_size == 2 and c.drop_path_rate == .2 and c.finetune and not c.gt_2d
    assert c.learning_rate == 1.5e-5 and c.warmup_epochs == 1 and c.max_grad_norm == 1
    assert math.isclose(c.ema_decay ** 2, .9998, abs_tol=1e-12)
    model = load_backbone(c)
    model.load_state_dict(weights, strict=True)
    assert sum(p.numel() for p in model.parameters()) == 20192451
    assert all(p.requires_grad for p in model.parameters())
    torch.save({'model_pos': weights, 'checkpoint_type': 'ema', 'ema_updates': 0, 'epoch': 0}, init)
    exported = torch.load(init, map_location='cpu', weights_only=False)
    assert all(torch.equal(weights[k], exported['model_pos'][k]) for k in weights)
    assert 'optimizer' not in exported and 'lr_schedule_state' not in exported
    dataset_hash = sha('data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl')
    assert dataset_hash == '73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175'
    (out/'effective_config.json').write_text(json.dumps(dict(c), indent=2, default=str))
    report = {'status': 'PENDING', 'source_checkpoint': str(SOURCE), 'source_sha256': EXPECTED,
              'initializer_sha256': sha(init), 'config_sha256': sha(CONFIG), 'dataset_sha256': dataset_hash,
              'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'stages': {}}
    del model, payload, exported
    dataset = MotionDataset3D(c, c.subset_list, 'train')
    c.mask = False
    for batch in (1, 2):
        set_random_seed(0)
        base = load_backbone(c).cuda()
        base.load_state_dict(weights, strict=True)
        groups = build_adamw_parameter_groups(base, c.weight_decay, honor_no_weight_decay=True)
        assert sum(p.numel() for g in groups for p in g['params']) == 20192451
        assert groups[-1]['weight_decay'] == 0
        optimizer = torch.optim.AdamW(groups, lr=c.learning_rate)
        assert len(optimizer.state) == 0
        ema = EMAModel(base, c.ema_decay)
        assert ema.num_updates == 0
        assert all(torch.equal(ema.shadow[k], base.state_dict()[k]) for k in ema.shadow)
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
        schedule = build_lr_schedule(c, optimizer, len(loader), start_step=0)
        assert schedule.global_step == 0
        assert math.isclose(schedule.scale_at(0) * c.learning_rate, 1.5e-6)
        torch.cuda.reset_peak_memory_stats()
        if batch == 2:
            torch._dynamo.config.recompile_limit = 64
            net = torch.compile(base, mode=c.compile_mode)
        else:
            net = base
        meters = make_meters()
        train_epoch(c, net, itertools.islice(loader, 2), meters, optimizer,
                    has_3d=True, has_gt=True, ema_helper=ema, lr_schedule=schedule)
        torch.cuda.synchronize()
        assert ema.num_updates == 2 and schedule.global_step == 2
        assert all(math.isfinite(meters[k].avg) for k in ('total','3d_pos','grad_norm'))
        assert all(torch.isfinite(p).all() for p in base.parameters())
        sample = next(iter(loader))[0].cuda()
        base.eval()
        with torch.no_grad(), ema.average_parameters(base):
            before = base(sample)
            assert torch.isfinite(before).all()
            state = {k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
        smoke = out/f'B{batch}_ema_roundtrip.bin'
        torch.save({'model_pos':state}, smoke)
        restored = torch.load(smoke, map_location='cpu', weights_only=False)['model_pos']
        assert all(torch.equal(state[k], restored[k]) for k in state)
        base.load_state_dict(restored, strict=True)
        with torch.no_grad():
            after = base(sample)
        assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
        peak = torch.cuda.max_memory_reserved()/1024**2
        assert peak < 28000
        report['stages'][f'B{batch}'] = {'loss':meters['total'].avg,
            'grad_norm':meters['grad_norm'].avg, 'peak_reserved_mib':peak, 'strict_roundtrip':True,
            'optimizer_reset':True, 'ema_reset':True, 'compiled': batch == 2}
        del net, base, optimizer, groups, ema, schedule, sample, before, after, state, restored
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'PASS'
    (out/'PREFLIGHT_PASS.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
