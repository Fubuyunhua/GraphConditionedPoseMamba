"""torch.compile-compatible wrapper around the existing selective-scan kernel.

The numerical implementation remains ``selective_scan_cuda_core``.  Registering
its forward and backward entry points as opaque PyTorch custom operators gives
Dynamo/AOTAutograd the shape and autograd metadata it needs without tracing into
the pybind extension.
"""

from __future__ import annotations

import torch
from torch import Tensor


try:
    import selective_scan_cuda_core as _selective_scan_cuda_core
except ImportError:
    _selective_scan_cuda_core = None


COMPILE_SCAN_AVAILABLE = _selective_scan_cuda_core is not None


if COMPILE_SCAN_AVAILABLE:

    @torch.library.custom_op("posemamba::selective_scan_core_fwd", mutates_args=())
    def _scan_fwd(
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        delta_bias: Tensor,
        delta_softplus: bool,
    ) -> tuple[Tensor, Tensor]:
        output, state, *_ = _selective_scan_cuda_core.fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            delta_softplus,
            1,
        )
        return output, state

    @_scan_fwd.register_fake
    def _scan_fwd_fake(u, delta, A, B, C, D, delta_bias, delta_softplus):
        chunks = (u.shape[-1] + 2047) // 2048
        state = u.new_empty(
            (u.shape[0], u.shape[1], chunks, A.shape[-1] * 2)
        )
        return torch.empty_like(u), state

    @torch.library.custom_op("posemamba::selective_scan_core_bwd", mutates_args=())
    def _scan_bwd(
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        delta_bias: Tensor,
        grad_output: Tensor,
        state: Tensor,
        delta_softplus: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if grad_output.stride(-1) != 1:
            grad_output = grad_output.contiguous()
        gradients = _selective_scan_cuda_core.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            grad_output,
            state,
            delta_softplus,
            1,
        )
        return tuple(gradients[:7])

    @_scan_bwd.register_fake
    def _scan_bwd_fake(
        u,
        delta,
        A,
        B,
        C,
        D,
        delta_bias,
        grad_output,
        state,
        delta_softplus,
    ):
        return tuple(
            torch.empty_like(value)
            for value in (u, delta, A, B, C, D, delta_bias)
        )

    def _setup_scan_context(ctx, inputs, output):
        u, delta, A, B, C, D, delta_bias, delta_softplus = inputs
        _, state = output
        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, state)

    def _scan_autograd_backward(ctx, grad_output, grad_state):
        u, delta, A, B, C, D, delta_bias, state = ctx.saved_tensors
        gradients = _scan_bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            grad_output,
            state,
            ctx.delta_softplus,
        )
        return (*gradients, None)

    torch.library.register_autograd(
        "posemamba::selective_scan_core_fwd",
        _scan_autograd_backward,
        setup_context=_setup_scan_context,
    )


class SelectiveScanCoreCompile:
    """``SelectiveScanCore.apply``-compatible entry point for factorized SSMs."""

    @staticmethod
    def apply(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        delta_bias=None,
        delta_softplus=False,
        nrows=1,
        backnrows=1,
        oflex=True,
    ):
        if not COMPILE_SCAN_AVAILABLE:
            raise RuntimeError("selective_scan_cuda_core is unavailable")
        if D is None or delta_bias is None:
            raise ValueError("compile-compatible factorized scan requires D and delta_bias")
        return _scan_fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            bool(delta_softplus),
        )[0]


__all__ = ["COMPILE_SCAN_AVAILABLE", "SelectiveScanCoreCompile"]
