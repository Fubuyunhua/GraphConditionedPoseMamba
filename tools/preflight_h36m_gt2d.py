"""Protocol and real-gradient gate for the explicitly requested small GT2D run."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.datareader_h36m import DataReaderH36M
from lib.utils.utils_data import flip_data

c = get_config('configs/pose3d/graph_posemamba_h36m_w64_d8_gt2d_80e.yaml')
assert c.gt_2d and not c.train_2d and not c.no_conf and not c.no_eval
assert c.scale_range_pretrain is None and not c.noise and not c.synthetic
reader = DataReaderH36M(243, 1, 81, 243, dt_root=c.data_root, dt_file=c.dt_file)
split_ids = reader.get_split_id()
checks = {}
for split, ids in zip(['train', 'test'], split_ids):
    ds = MotionDataset3D(c, c.subset_list, split)
    with patch('lib.data.augmentation.random.random', return_value=0.0):
        x, y = ds[0]
    assert x.shape == y.shape == (243,17,3)
    assert torch.equal(x[...,:2], y[...,:2]) and torch.all(x[...,2] == 1)
    idx = np.asarray(ids[0])
    source = reader.dt_dataset[split]
    expected = source['joint3d_image'][idx,:,:3].astype(np.float32).copy()
    for k, camera in enumerate(source['camera_name'][idx]):
        h = 1002 if camera in ['54138969','60457274'] else 1000
        expected[k,:,:2] = expected[k,:,:2] / 1000 * 2 - [1,h/1000]
        expected[k,:,2:] = expected[k,:,2:] / 1000 * 2
    error = float(np.max(np.abs(expected-y.numpy())))
    assert error < 2e-6, (split,error)
    assert np.allclose(flip_data(flip_data(x.numpy())),x.numpy())
    if split == 'train':
        train_x, train_y = x.clone(), y.clone()
        with patch('lib.data.augmentation.random.random', return_value=1.0):
            xf,yf=ds[0]
        assert np.allclose(xf.numpy(),flip_data(x.numpy()))
        assert np.allclose(yf.numpy(),flip_data(y.numpy()))
    else:
        pixel_xy=np.empty_like(expected[...,:2])
        for k,camera in enumerate(source['camera_name'][idx]):
            h=1002 if camera in ['54138969','60457274'] else 1000
            pixel_xy[k]=(x.numpy()[k,:,:2]+[1,h/1000])*500
        lifted_xy=pixel_xy*source['2.5d_factor'][idx,None,None]
        gt_xy=source['joints_2.5d_image'][idx,:,:2]
        root_error=float(np.max(np.abs((lifted_xy-lifted_xy[:,0:1])-(gt_xy-gt_xy[:,0:1]))))
        assert root_error<0.02,root_error
        checks['test_known_xy_root_error_mm']=root_error
    checks[split]={'samples':len(ds),'cached_label_vs_source_max_abs':error,'xy_is_label':True,'confidence_one':True}
del reader
m=load_backbone(c).cuda();assert sum(p.numel() for p in m.parameters())==800083
m.train();pred=m(train_x.unsqueeze(0).cuda());target=(train_y-train_y[:,0:1]).unsqueeze(0).cuda()
loss=(pred-target).square().mean();assert torch.isfinite(loss)
loss.backward();grads=[p.grad for p in m.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert any(g.abs().sum()>0 for g in grads)
checks.update(status='PASS',parameters=800083,finite_gradients=True,gradient_tensors=len(grads),gradient_smoke_loss=float(loss),evaluation_convention='known input xy passthrough then denormalize/factor/root-center')
Path('verification/gt2d_protocol_gate.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks))
