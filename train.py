import os
import numpy as np
import argparse
import errno
import math
import pickle
import datetime
import tensorboardX
import torch.distributed
from tqdm import tqdm
import time
import copy
import random
import prettytable
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import contextmanager

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M  
from lib.model.loss import *
import logger
from logger import colorlogger


def _unwrap_compiled_model(model):
    """Return the checkpoint-compatible module behind torch.compile wrappers."""
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


class CudaGraphTrainModel(nn.Module):
    """Replay a fixed-shape training forward/backward with CUDA Graphs.

    Evaluation always calls the original eager forward, so flip evaluation,
    variable batch sizes and checkpoint behavior stay unchanged.  The wrapped
    module is exposed as ``_orig_mod`` for the existing checkpoint/EMA helper.
    """

    def __init__(self, module, warmup_iters=3):
        super().__init__()
        self._orig_mod = module
        self.warmup_iters = int(warmup_iters)
        self._eager_forward = module.forward
        self._graph_forward = None
        self._input_signature = None

    @staticmethod
    def _signature(x):
        return (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)

    def _capture(self, x):
        torch.cuda.make_graphed_callables(
            self._orig_mod,
            (x,),
            num_warmup_iters=self.warmup_iters,
            allow_unused_input=True,
        )
        self._graph_forward = self._orig_mod.forward
        self._orig_mod.forward = self._eager_forward
        self._input_signature = self._signature(x)

    def forward(self, x, *args, **kwargs):
        use_graph = (
            self.training
            and torch.is_grad_enabled()
            and x.is_cuda
            and not args
            and not kwargs
        )
        if not use_graph:
            return self._eager_forward(x, *args, **kwargs)
        signature = self._signature(x)
        if self._graph_forward is None:
            self._capture(x)
        elif signature != self._input_signature:
            return self._eager_forward(x)
        return self._graph_forward(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LinearWarmupEpochDecay:
    """Per-step linear warmup followed by epoch-wise exponential decay."""

    def __init__(
        self,
        optimizer,
        steps_per_epoch,
        warmup_epochs,
        start_factor,
        lr_decay,
        start_step=0,
    ):
        self.optimizer = optimizer
        self.steps_per_epoch = int(steps_per_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_steps = self.steps_per_epoch * self.warmup_epochs
        self.start_factor = float(start_factor)
        self.lr_decay = float(lr_decay)
        self.global_step = int(start_step)
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if self.warmup_epochs <= 0:
            raise ValueError("warmup_epochs must be positive when warmup is enabled")
        if not 0.0 < self.start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0, 1]")
        self.base_lrs = [
            float(group.get("initial_lr", group["lr"]))
            for group in optimizer.param_groups
        ]
        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            group["initial_lr"] = base_lr

    def scale_at(self, step):
        step = int(step)
        if step < self.warmup_steps:
            denominator = max(self.warmup_steps - 1, 1)
            progress = step / denominator
            return self.start_factor + (1.0 - self.start_factor) * progress
        post_warmup_epoch = (step - self.warmup_steps) // self.steps_per_epoch
        return self.lr_decay ** post_warmup_epoch

    def prepare_step(self):
        scale = self.scale_at(self.global_step)
        lrs = []
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = base_lr * scale
            group["lr"] = lr
            lrs.append(lr)
        return lrs

    def complete_step(self):
        self.global_step += 1

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "steps_per_epoch": self.steps_per_epoch,
            "warmup_epochs": self.warmup_epochs,
            "start_factor": self.start_factor,
            "lr_decay": self.lr_decay,
            "base_lrs": list(self.base_lrs),
        }


def build_lr_schedule(args, optimizer, steps_per_epoch, start_step=0):
    if not bool(getattr(args, "enable_linear_warmup", False)):
        return None
    return LinearWarmupEpochDecay(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=int(args.warmup_epochs),
        start_factor=float(getattr(args, "warmup_start_factor", 0.1)),
        lr_decay=float(args.lr_decay),
        start_step=start_step,
    )

def _clone_state_dict_for_save(state_dict):
    cloned = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            cloned[k] = v.detach().cpu().clone()
        else:
            cloned[k] = copy.deepcopy(v)
    return cloned

def _move_state_dict_to_model_devices(state_dict, model):
    ref_state = model.state_dict()
    moved = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v) and k in ref_state and torch.is_tensor(ref_state[k]):
            moved[k] = v.to(device=ref_state[k].device, dtype=ref_state[k].dtype)
        else:
            moved[k] = v
    return moved

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss, model_state_dict=None, extra_state=None):
    checkpoint_model = _unwrap_compiled_model(model_pos)
    log.info(f'Saving checkpoint to{chk_path}')
    payload = {
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'optimizer_lrs': [pg.get('lr') for pg in optimizer.param_groups] if optimizer is not None else None,
        'model_pos': _clone_state_dict_for_save(checkpoint_model.state_dict() if model_state_dict is None else model_state_dict),
        'min_loss': min_loss,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, chk_path)

class EMAModel:
    """Exponential Moving Average wrapper for model weights."""
    def __init__(self, model, decay):
        model = _unwrap_compiled_model(model)
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        self.backup = {}

    def update(self, model):
        model = _unwrap_compiled_model(model)
        with torch.no_grad():
            grouped = {}
            for k, v in model.state_dict().items():
                if not v.is_floating_point():
                    continue
                if k not in self.shadow:
                    self.shadow[k] = v.detach().clone()
                else:
                    shadow = self.shadow[k]
                    key = (shadow.device, shadow.dtype)
                    shadows, sources = grouped.setdefault(key, ([], []))
                    shadows.append(shadow)
                    sources.append(v.detach())
            for shadows, sources in grouped.values():
                torch._foreach_mul_(shadows, self.decay)
                torch._foreach_add_(shadows, sources, alpha=1 - self.decay)

    @contextmanager
    def average_parameters(self, model):
        checkpoint_model = _unwrap_compiled_model(model)
        if not self.shadow:
            yield model
            return
        self.backup = {k: v.detach().clone() for k, v in checkpoint_model.state_dict().items()}
        checkpoint_model.load_state_dict(self.shadow, strict=False)
        try:
            yield model
        finally:
            checkpoint_model.load_state_dict(self.backup, strict=False)
            self.backup = {}

def save_ema_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss, ema_helper):
    if ema_helper is None:
        return
    with ema_helper.average_parameters(model_pos):
        save_checkpoint(
            chk_path,
            epoch,
            lr,
            optimizer,
            model_pos,
            min_loss,
            extra_state={
                'checkpoint_type': 'ema',
                'ema_decay': ema_helper.decay,
            },
        )

def evaluate(args, model_pos, test_loader, datareader):
    log.info('INFO: Testing')
    results_all = []
    model_pos.eval()            
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0

            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    log.info(len(results_all))# 2228
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)# [n_clips, -1, 17, 3]
    log.info(results_all.shape)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])#['s_09_act_02_subact_01_ca_01',...]
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])#[4.656527,...,2.9163694]
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    log.info(f"num_test_frames:{num_test_frames}")# num_test_frames:566920
    frames = np.array(range(num_test_frames))
    # print(split_id_test)
    """ 
    [range(0, 243), range(243, 486), range(486, 729),...,range(566532, 566775)]
    """
    log.info(len(split_id_test))
    action_clips = actions[split_id_test]# ndarray (2228,243)
    factor_clips = factors[split_id_test]# ndarray (2228,243)
    source_clips = sources[split_id_test]# ndarray (2228,243)
    frame_clips = frames[split_id_test]# ndarray (2228,243) [[0,...,242],...,[566532,...,566774]]
    gt_clips = gts[split_id_test]# ndarray (2228,243,17,3)
    assert len(results_all)==len(action_clips)
    
    e1_all = np.zeros(num_test_frames)# ndarray (566920,)
    e2_all = np.zeros(num_test_frames)# ndarray (566920,)
    oc = np.zeros(num_test_frames)# ndarray (566920,)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    #['Direction', 'Discuss', 'Eating', 'Greet', 'Phone', 'Photo', 'Pose', 'Purchase', 'Sitting', 'SittingDown', 'Smoke', 'Wait', 'Walk', 'WalkDog', 'WalkTwo']
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02', 
                  's_09_act_10_subact_02', 
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6] # s_09_act_05_subact_02
        if source in block_list:
            continue
        frame_list = frame_clips[idx] # [0,...,242]
        action = action_clips[idx][0] # Direction
        factor = factor_clips[idx][:,None,None] # ndarray (243,1,1)
        gt = gt_clips[idx] # ndarray (243,17,3)
        pred = results_all[idx] # ndarray (243,17,3)
        pred *= factor# (243,17,3)
        
        # Root-relative Errors
        pred = pred - pred[:,0:1,:] # (243,17,3) 减去Pelvis骨盆的
        gt = gt - gt[:,0:1,:]# (243, 17, 3) 减去Pelvis骨盆的
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    log.info(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    log.info(f'Protocol #1 Error (MPJPE):{e1}mm')
    log.info(f'Protocol #2 Error (P-MPJPE):{e2}mm')
    log.info('----------')
    return e1, e2, results_all
        
def train_epoch(
    args,
    model_pos,
    train_loader,
    losses,
    optimizer,
    has_3d,
    has_gt,
    ema_helper=None,
    lr_schedule=None,
):
    model_pos.train()
    metric_sums = None
    metric_count = 0
    metric_keys = None
    grad_norm_sum = None
    grad_norm_count = 0
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    
        if lr_schedule is not None:
            lr_schedule.prepare_step()
        if (
            bool(getattr(args, "compile_model", False))
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()
        batch_size = len(batch_input)        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]# (N, T, 17, 2) 
            if not has_3d:
                # 得到2D骨架训练所需要的confidence
                conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
            if args.rootrel: # 相对于根
                batch_gt = batch_gt - batch_gt[:,:,0:1,:]
            else:
                batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Release the previous step's gradients before allocating the next
        # forward activations.  This is numerically equivalent to clearing
        # them after forward because gradients are only consumed by step().
        optimizer.zero_grad(set_to_none=True)
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input)    # (N, T, 17, 3)

        if has_3d:
            """ 
            lambda_3d_velocity: 20.0
            lambda_scale: 0.5
            lambda_lv: 0.0
            lambda_lg: 0.0
            lambda_a: 0.0
            lambda_av: 0.0
            """
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt) # 3D
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)# weighted 2D re-projection loss
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt) #LO
            zero_loss = predicted_3d_pos.new_zeros(())
            loss_lv = (
                loss_limb_var(predicted_3d_pos)
                if float(args.lambda_lv) != 0.0
                else zero_loss
            )
            loss_lg = (
                loss_limb_gt(predicted_3d_pos, batch_gt)
                if float(args.lambda_lg) != 0.0
                else zero_loss
            )
            loss_a = (
                loss_angle(predicted_3d_pos, batch_gt)
                if float(args.lambda_a) != 0.0
                else zero_loss
            )
            loss_av = (
                loss_angle_velocity(predicted_3d_pos, batch_gt)
                if float(args.lambda_av) != 0.0
                else zero_loss
            )

            need_joint_weights = (
                float(args.lambda_3dw) != 0.0 or float(args.lambda_diff) != 0.0
            )
            if need_joint_weights:
                w_mpjpe = predicted_3d_pos.new_tensor(
                    [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]
                )
            loss_3d_w = (
                weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe)
                if float(args.lambda_3dw) != 0.0
                else zero_loss
            )
            
            # Temporal Consistency Loss
            dif_seq = predicted_3d_pos[:,1:,:,:] - predicted_3d_pos[:,:-1,:,:]
            # weights_diff = 0.5
            # index = [1,1,1,1,2,2,2,2,1]
            # dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)), dim=-1)
            loss_diff = (
                torch.mean(
                    torch.square(dif_seq) * w_mpjpe.view(1, 1, -1, 1)
                )
                if float(args.lambda_diff) != 0.0
                else zero_loss
            )

            # Attention regularization（时间注意力）
            attn_reg = zero_loss
            lam_diag_value = float(args.lambda_attn_diag)
            lam_ent_value = float(args.lambda_attn_entropy)
            if lam_diag_value != 0.0 or lam_ent_value != 0.0:
                if hasattr(model_pos, "module") and hasattr(model_pos.module, "collect_temporal_attn_tensors"):
                    attn_maps = model_pos.module.collect_temporal_attn_tensors()
                elif hasattr(model_pos, "collect_temporal_attn_tensors"):
                    attn_maps = model_pos.collect_temporal_attn_tensors()
                else:
                    attn_maps = []
            else:
                attn_maps = []
            if attn_maps:
                diag_loss = torch.tensor(0.0, device=predicted_3d_pos.device, dtype=predicted_3d_pos.dtype)
                ent_loss = torch.tensor(0.0, device=predicted_3d_pos.device, dtype=predicted_3d_pos.dtype)
                count = 0
                for attn in attn_maps:
                    # attn: [BJ,H,T,T]
                    diag_loss = diag_loss + attn.diagonal(dim1=-2, dim2=-1).mean()
                    ent_loss = ent_loss - (attn * torch.log(attn + 1e-8)).mean()
                    count += 1
                if count > 0:
                    diag_loss = diag_loss / count
                    ent_loss = ent_loss / count
                    lam_diag = predicted_3d_pos.new_tensor(lam_diag_value)
                    lam_ent = predicted_3d_pos.new_tensor(lam_ent_value)
                    attn_reg = lam_diag * diag_loss + lam_ent * ent_loss
            
            # 辅助监督（如尾部辅助头），可选
            aux_loss = zero_loss
            lam_aux = float(getattr(args, 'lambda_tail_aux', 0.0))
            if lam_aux > 0:
                aux_pred = None
                if hasattr(model_pos, "module") and hasattr(model_pos.module, "_tail_aux"):
                    aux_pred = model_pos.module._tail_aux
                elif hasattr(model_pos, "_tail_aux"):
                    aux_pred = model_pos._tail_aux
                if aux_pred is not None:
                    aux_loss = loss_mpjpe(aux_pred, batch_gt)

            gate_loss = zero_loss
            lam_gate = float(getattr(args, 'lambda_gate_sparsity', 0.0))
            if lam_gate > 0:
                gate_mean = None
                if hasattr(model_pos, "module") and hasattr(model_pos.module, "_last_gate_mean"):
                    gate_mean = model_pos.module._last_gate_mean
                elif hasattr(model_pos, "_last_gate_mean"):
                    gate_mean = model_pos._last_gate_mean
                if gate_mean is not None:
                    gate_loss = gate_mean.to(predicted_3d_pos.device, dtype=predicted_3d_pos.dtype)

            loss_total = args.lambda_3d * loss_3d_pos + \
                         args.lambda_scale       * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv          * loss_lv + \
                         args.lambda_lg          * loss_lg + \
                         args.lambda_a           * loss_a  + \
                         args.lambda_av          * loss_av + \
                         args.lambda_3dw          * loss_3d_w + \
                         args.lambda_diff          * loss_diff + \
                         lam_aux * aux_loss + \
                         attn_reg + \
                         lam_gate * gate_loss

            metric_keys = (
                '3d_pos', '3d_scale', '3d_velocity', 'lv', 'lg',
                'angle', 'angle_velocity', 'total',
            )
            metric_values = torch.stack(
                (
                    loss_3d_pos,
                    loss_3d_scale,
                    loss_3d_velocity,
                    loss_lv if float(args.lambda_lv) != 0.0 else zero_loss,
                    loss_lg if float(args.lambda_lg) != 0.0 else zero_loss,
                    loss_a if float(args.lambda_a) != 0.0 else zero_loss,
                    loss_av if float(args.lambda_av) != 0.0 else zero_loss,
                    loss_total,
                )
            ).detach()
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            metric_keys = ('2d_proj', 'total')
            metric_values = torch.stack((loss_2d_proj, loss_total)).detach()
        weighted_metrics = metric_values * batch_size
        metric_sums = (
            weighted_metrics
            if metric_sums is None
            else metric_sums + weighted_metrics
        )
        metric_count += batch_size
        loss_total.backward()
        max_grad_norm = float(getattr(args, "max_grad_norm", 0.0))
        if max_grad_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_pos.parameters(),
                max_norm=max_grad_norm,
                error_if_nonfinite=bool(
                    getattr(args, "grad_clip_error_if_nonfinite", True)
                ),
            )
            grad_norm_sum = (
                grad_norm.detach()
                if grad_norm_sum is None
                else grad_norm_sum + grad_norm.detach()
            )
            grad_norm_count += 1
        optimizer.step()
        if lr_schedule is not None:
            lr_schedule.complete_step()
        if ema_helper is not None:
            ema_helper.update(model_pos)
    if metric_sums is not None:
        metric_averages = (metric_sums / metric_count).cpu().tolist()
        for key, value in zip(metric_keys, metric_averages):
            losses[key].update(value, metric_count)
    if grad_norm_sum is not None and grad_norm_count > 0:
        losses["grad_norm"].update(
            float((grad_norm_sum / grad_norm_count).cpu()),
            grad_norm_count,
        )
def get_beijing_timestamp():
    local_offset = time.localtime().tm_gmtoff   # 当前机器utc时间偏移量
    beijing_offset = int(8 * 60*60)
    offset = local_offset - beijing_offset
    timestamp = int(datetime.datetime.now().timestamp())
    beijing_timestamp = timestamp - offset

    return beijing_timestamp

def train_with_config(args, opts):

    opts.checkpoint = opts.checkpoint +'_'+ datetime.datetime.fromtimestamp(get_beijing_timestamp()).strftime('%Y_%m_%d_T_%H_%M_%S')
    # global log
    # log = logger.set_save_path(opts.checkpoint)
    # log(args)
    global log
    log = colorlogger(opts.checkpoint, log_name='log.txt')
    log.info(args)
    with open(os.path.join(opts.checkpoint, 'config.yaml'), 'w') as f:
        yaml.dump(args, f, sort_keys=False)
    log.info(f"Number of GPUs found:{torch.cuda.device_count()}")
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = None
    try:
        train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    except PermissionError as e:
        log.warning(f'Create SummaryWriter failed ({e}); TensorBoard logging disabled.')

    log.info('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 0,
          'pin_memory': False,
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 0,
          'pin_memory': False,
    }

    def _build_loader(dataset, params, name):
        try:
            return DataLoader(dataset, **params)
        except PermissionError as e:
            log.warning(f'Create DataLoader({name}) failed ({e}); fallback to single-process loader.')
            safe_params = dict(params)
            safe_params['num_workers'] = 0
            safe_params['pin_memory'] = False
            safe_params.pop('prefetch_factor', None)
            safe_params.pop('persistent_workers', None)
            return DataLoader(dataset, **safe_params)

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = _build_loader(train_dataset, trainloader_params, "train")
    test_loader = _build_loader(test_dataset, testloader_params, "test")
    
    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = _build_loader(posetrack, trainloader_params, "posetrack_2d")
        instav = InstaVDataset2D()
        instav_loader_2d = _build_loader(instav, trainloader_params, "instav_2d")
        
    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = args.data_root, dt_file=args.dt_file)
    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    log.info(f'INFO: Trainable parameter count:{model_params}')

    compile_model = bool(getattr(args, "compile_model", False))
    cuda_graph_model = bool(getattr(args, "cuda_graph_model", False))
    if compile_model and cuda_graph_model:
        raise ValueError("compile_model and cuda_graph_model cannot both be enabled")
    if torch.cuda.is_available():
        # torch.distributed.init_process_group('nccl', init_method='tcp://localhost:23456', world_size=2, rank=0)
        # model_backbone = nn.parallel.DistributedDataParallel(model_backbone)
        if torch.cuda.device_count() > 1 or not (compile_model or cuda_graph_model):
            model_backbone = nn.DataParallel(model_backbone)
        # k = torch.cuda.device_count()
        # model_backbone = nn.DataParallel(model_backbone, device_ids=list(range(k)))
        model_backbone = model_backbone.cuda()

    ema_helper = None
    if getattr(args, "use_ema", False):
        ema_helper = EMAModel(model_backbone, decay=float(getattr(args, "ema_decay", 0.9998)))

    checkpoint = None
    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint{chk_filename}')
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage, weights_only=False
            )
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            log.info(f'Loading checkpoint{chk_filename}')
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage, weights_only=False
            )
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone            
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint{chk_filename}')
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage, weights_only=False
            )
            if args.backbone == 'MotionAGFormer':
                model_backbone.load_state_dict(checkpoint['model'], strict=True)
            else:
                model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone
        
    if ema_helper is not None and checkpoint is not None:
        if checkpoint.get('ema_shadow') is not None:
            ema_helper.shadow = _move_state_dict_to_model_devices(checkpoint['ema_shadow'], model_backbone)
        elif checkpoint.get('checkpoint_type') == 'ema':
            ema_helper.shadow = {k: v.detach().clone() for k, v in model_backbone.state_dict().items() if v.is_floating_point()}

    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model=True requires torch.compile support")
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.recompile_limit = max(
                int(torch._dynamo.config.recompile_limit),
                64,
            )
        compile_mode = str(getattr(args, "compile_mode", "reduce-overhead"))
        allowed_compile_modes = {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }
        if compile_mode not in allowed_compile_modes:
            raise ValueError(
                f"Unsupported compile_mode={compile_mode!r}; "
                f"expected one of {sorted(allowed_compile_modes)}"
            )
        log.info(f"INFO: Enabling torch.compile(mode={compile_mode!r})")
        log.info(
            "INFO: The first forward/backward is compiling and may remain at "
            "0it for about 10-60 seconds; do not interrupt unless a final "
            "RuntimeError/CUDA error is printed."
        )
        model_pos = torch.compile(model_pos, mode=compile_mode, fullgraph=False)
    elif cuda_graph_model:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_graph_model=True requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("cuda_graph_model currently requires exactly one visible GPU")
        log.info("INFO: Enabling fixed-shape CUDA Graph training replay")
        model_pos = CudaGraphTrainModel(model_pos)

    if not opts.evaluate:        
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            log.info(f'INFO: Training on {len(train_loader_3d)}(3D)+{len(instav_loader_2d) + len(posetrack_loader_2d)}(2D) batches')
        else:
            log.info(f'INFO: Training on {len(train_loader_3d)}(3D) batches')
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                log.info('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = float(checkpoint.get('lr', lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = lr
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
            log.info(f'INFO: Resumed from epoch {st} with lr={lr:.10f}, min_loss={min_loss}')

        steps_per_epoch = len(train_loader_3d)
        if args.train_2d:
            steps_per_epoch += len(instav_loader_2d) + len(posetrack_loader_2d)
        schedule_start_step = st * steps_per_epoch
        if checkpoint is not None and checkpoint.get("lr_schedule_state") is not None:
            schedule_start_step = int(
                checkpoint["lr_schedule_state"].get(
                    "global_step", schedule_start_step
                )
            )
        lr_schedule = build_lr_schedule(
            args,
            optimizer,
            steps_per_epoch=steps_per_epoch,
            start_step=schedule_start_step,
        )
        if lr_schedule is not None:
            log.info(
                "INFO: Enabling per-step linear warmup: "
                f"start_factor={lr_schedule.start_factor}, "
                f"warmup_epochs={lr_schedule.warmup_epochs}, "
                f"warmup_steps={lr_schedule.warmup_steps}, "
                f"post_warmup_lr_decay={lr_schedule.lr_decay}"
            )
                
        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)
        
        # Training
        for epoch in range(st, args.epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            log.info(f'Training epoch {epoch}.')
            start_time = time.time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            losses['grad_norm'] = AverageMeter()
            N = 0
                        
            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True, ema_helper=ema_helper, lr_schedule=lr_schedule)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False, ema_helper=ema_helper, lr_schedule=lr_schedule)
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True, ema_helper=ema_helper, lr_schedule=lr_schedule)
            elapsed = (time.time() - start_time) / 60
            lr = float(optimizer.param_groups[0]['lr'])

            if args.no_eval:
                log.info('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                def _run_eval():
                    eval_model = (
                        _unwrap_compiled_model(model_pos)
                        if compile_model
                        and bool(getattr(args, "eager_eval_when_compiled", False))
                        else model_pos
                    )
                    if ema_helper is not None:
                        with ema_helper.average_parameters(model_pos):
                            return evaluate(args, eval_model, test_loader, datareader)
                    return evaluate(args, eval_model, test_loader, datareader)

                e1, e2, results_all = _run_eval()
                if float(getattr(args, "max_grad_norm", 0.0)) > 0.0:
                    log.info(
                        '[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f grad_norm %f' % (
                            epoch + 1,
                            elapsed,
                            lr,
                            losses['3d_pos'].avg,
                            e1,
                            e2,
                            losses['grad_norm'].avg,
                        )
                    )
                else:
                    log.info('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                        epoch + 1,
                        elapsed,
                        lr,
                        losses['3d_pos'].avg,
                        e1, e2))
                log.info(f'Remaining training time: {datetime.timedelta(seconds=time.time() - start_time) * (args.epochs - epoch)}')
                if train_writer is not None:
                    train_writer.add_scalar('Error P1', e1, epoch + 1)
                    train_writer.add_scalar('Error P2', e2, epoch + 1)
                    train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                    train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                    train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                    train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                    train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                    train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                    train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                    train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                    train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)

            train_batches = len(train_loader_3d)
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_batches += len(instav_loader_2d) + len(posetrack_loader_2d)
            train_it_per_sec = train_batches / max(elapsed * 60.0, 1e-12)
            if torch.cuda.is_available():
                peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
                peak_reserved_mib = torch.cuda.max_memory_reserved() / 2**20
            else:
                peak_allocated_mib = 0.0
                peak_reserved_mib = 0.0
            log.info(
                f'RUNTIME epoch={epoch + 1} '
                f'train_it_per_sec={train_it_per_sec:.6f} '
                f'peak_allocated_mib={peak_allocated_mib:.3f} '
                f'peak_reserved_mib={peak_reserved_mib:.3f} '
                f'grad_norm_preclip={losses["grad_norm"].avg:.6f}'
            )
                
            # Decay learning rate exponentially
            if lr_schedule is None:
                lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin')
            chk_path_latest_ema = os.path.join(opts.checkpoint, 'latest_ema_epoch.bin')
            chk_path_best_ema = os.path.join(opts.checkpoint, 'best_ema_epoch.bin')
            chk_path_ema = os.path.join(opts.checkpoint, 'epoch_{}_ema.bin'.format(epoch))

            raw_extra_state = {'checkpoint_type': 'raw'}
            if ema_helper is not None:
                raw_extra_state['ema_shadow'] = _clone_state_dict_for_save(ema_helper.shadow)
                raw_extra_state['ema_decay'] = ema_helper.decay
            if lr_schedule is not None:
                raw_extra_state['lr_schedule_state'] = lr_schedule.state_dict()

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss, extra_state=raw_extra_state)
            if ema_helper is not None:
                save_ema_checkpoint(chk_path_latest_ema, epoch, lr, optimizer, model_pos, min_loss, ema_helper)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss, extra_state=raw_extra_state)
                if ema_helper is not None:
                    save_ema_checkpoint(chk_path_ema, epoch, lr, optimizer, model_pos, min_loss, ema_helper)
            if not args.no_eval and e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss, extra_state=raw_extra_state)
                if ema_helper is not None:
                    save_ema_checkpoint(chk_path_best_ema, epoch, lr, optimizer, model_pos, min_loss, ema_helper)
                
    if opts.evaluate:
        eval_model = (
            _unwrap_compiled_model(model_pos)
            if compile_model
            and bool(getattr(args, "eager_eval_when_compiled", False))
            else model_pos
        )
        e1, e2, results_all = evaluate(args, eval_model, test_loader, datareader)

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
