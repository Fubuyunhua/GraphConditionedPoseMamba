"""Protocol-v2 training epoch for GraphConditionedPoseMamba.

The released 3DHP NPZ has no paired clean GT2D target or detector confidence.
Consequently this runner never fabricates reliability labels: ReliPoseMU gets
``gt_2d=None`` and its direct gain is trained only by the downstream 3D loss.
The ordinary PoseMamba path is byte-for-byte equivalent at the loss level to
the frozen protocol-v2 trainer.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from lib.model.loss import (
    loss_angle,
    loss_angle_velocity,
    loss_limb_gt,
    loss_limb_var,
    loss_mpjpe,
    loss_velocity,
    n_mpjpe,
    weighted_mpjpe,
)


def train_epoch_3dhp(
    args,
    model_pos,
    train_loader,
    losses,
    optimizer,
    *,
    epoch: int,
    ema_helper=None,
):
    model_pos.train()
    for batch_input, batch_gt in tqdm(train_loader):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda(non_blocking=True)
            batch_gt = batch_gt.cuda(non_blocking=True)

        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[..., :2]
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            else:
                batch_gt[:, 0, 0, 2] = 0

        optimizer.zero_grad(set_to_none=True)
        predicted_3d_pos = model_pos(batch_input)

        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        zero_loss = predicted_3d_pos.new_zeros(())
        loss_lv = loss_limb_var(predicted_3d_pos) if float(args.lambda_lv) else zero_loss
        loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt) if float(args.lambda_lg) else zero_loss
        loss_a = loss_angle(predicted_3d_pos, batch_gt) if float(args.lambda_a) else zero_loss
        loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt) if float(args.lambda_av) else zero_loss

        device = predicted_3d_pos.device
        dtype = predicted_3d_pos.dtype
        w_mpjpe = torch.tensor(
            [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4],
            device=device,
            dtype=dtype,
        )
        loss_3d_w = (
            weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe)
            if float(args.lambda_3dw)
            else zero_loss
        )
        dif_seq = predicted_3d_pos[:, 1:] - predicted_3d_pos[:, :-1]
        loss_diff = (
            torch.mean(torch.square(dif_seq) * w_mpjpe.view(1, 1, -1, 1))
            if float(args.lambda_diff)
            else zero_loss
        )

        attn_reg = torch.zeros((), device=device, dtype=dtype)
        owner = model_pos.module if hasattr(model_pos, "module") else model_pos
        attn_maps = (
            owner.collect_temporal_attn_tensors()
            if hasattr(owner, "collect_temporal_attn_tensors")
            else []
        )
        if attn_maps:
            diag_loss = torch.stack(
                [attn.diagonal(dim1=-2, dim2=-1).mean() for attn in attn_maps]
            ).mean()
            ent_loss = torch.stack(
                [-(attn * torch.log(attn + 1e-8)).mean() for attn in attn_maps]
            ).mean()
            attn_reg = (
                float(getattr(args, "lambda_attn_diag", 0.0)) * diag_loss
                + float(getattr(args, "lambda_attn_entropy", 0.0)) * ent_loss
            )

        aux_loss = torch.zeros((), device=device, dtype=dtype)
        lam_aux = float(getattr(args, "lambda_tail_aux", 0.0))
        if lam_aux > 0 and getattr(owner, "_tail_aux", None) is not None:
            aux_loss = loss_mpjpe(owner._tail_aux, batch_gt)

        gate_loss = torch.zeros((), device=device, dtype=dtype)
        lam_gate = float(getattr(args, "lambda_gate_sparsity", 0.0))
        if lam_gate > 0 and getattr(owner, "_last_gate_mean", None) is not None:
            gate_loss = owner._last_gate_mean.to(device=device, dtype=dtype)

        loss_total = (
            float(args.lambda_3d) * loss_3d_pos
            + float(args.lambda_scale) * loss_3d_scale
            + float(args.lambda_3d_velocity) * loss_3d_velocity
            + float(args.lambda_lv) * loss_lv
            + float(args.lambda_lg) * loss_lg
            + float(args.lambda_a) * loss_a
            + float(args.lambda_av) * loss_av
            + float(args.lambda_3dw) * loss_3d_w
            + float(args.lambda_diff) * loss_diff
            + lam_aux * aux_loss
            + attn_reg
            + lam_gate * gate_loss
        )

        losses["3d_pos"].update(loss_3d_pos.item(), batch_size)
        losses["3d_scale"].update(loss_3d_scale.item(), batch_size)
        losses["3d_velocity"].update(loss_3d_velocity.item(), batch_size)
        losses["lv"].update(loss_lv.item(), batch_size)
        losses["lg"].update(loss_lg.item(), batch_size)
        losses["angle"].update(loss_a.item(), batch_size)
        losses["angle_velocity"].update(loss_av.item(), batch_size)
        losses["total"].update(loss_total.item(), batch_size)

        if not bool(torch.isfinite(loss_total)):
            raise RuntimeError(f"non-finite 3DHP loss at epoch {epoch}")
        loss_total.backward()
        optimizer.step()
        if ema_helper is not None:
            ema_helper.update(model_pos)


__all__ = ["train_epoch_3dhp"]
