"""Fused Triton RMSNorm forward and backward kernels for Qwen2RMSNorm.

Replaces Qwen2RMSNorm's multi-step eager sequence (fp32 cast -> pow(2) -> mean ->
rsqrt -> multiply -> cast -> multiply by fp32 weight, which in PEFT k-bit training
accidentally promotes the output back to fp32) with a single-pass register-fused
Triton kernel that accumulates variance in fp32 and writes the normalized result in
the input activation's native dtype (fp16).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    Rstd_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    tl.store(Rstd_ptr + row, rstd)

    y = (x * rstd) * w
    y_ptrs = Y_ptr + row * stride_y_row + cols
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    DY_ptr,
    X_ptr,
    W_ptr,
    Rstd_ptr,
    DX_ptr,
    stride_dy_row,
    stride_x_row,
    stride_dx_row,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dy_w = dy * w
    mean_dyw_xhat = tl.sum(dy_w * x_hat, axis=0) / N
    dx = rstd * (dy_w - x_hat * mean_dyw_xhat)

    tl.store(DX_ptr + row * stride_dx_row + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)


def _pick_rmsnorm_launch_config(n_cols: int) -> tuple[int, int]:
    block_n = triton.next_power_of_2(n_cols)
    if block_n <= 1024:
        num_warps = 4
    elif block_n <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    return block_n, num_warps


class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x_contig = x if x.is_contiguous() else x.contiguous()
        orig_shape = x_contig.shape
        n_cols = orig_shape[-1]
        n_rows = x_contig.numel() // n_cols

        x_2d = x_contig.view(n_rows, n_cols)
        y_2d = torch.empty_like(x_2d)
        rstd = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

        block_n, num_warps = _pick_rmsnorm_launch_config(n_cols)
        if n_rows > 0:
            rmsnorm_fwd_kernel[(n_rows,)](
                x_2d,
                weight,
                y_2d,
                rstd,
                x_2d.stride(0),
                y_2d.stride(0),
                n_cols,
                float(eps),
                BLOCK_N=block_n,
                num_warps=num_warps,
            )

        ctx.save_for_backward(x_contig, weight, rstd)
        ctx.block_n = block_n
        ctx.num_warps = num_warps
        ctx.n_cols = n_cols
        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_contig, weight, rstd = ctx.saved_tensors
        orig_shape = x_contig.shape
        n_cols = ctx.n_cols
        n_rows = x_contig.numel() // n_cols

        dy_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        dy_2d = dy_contig.view(n_rows, n_cols)
        x_2d = x_contig.view(n_rows, n_cols)

        dx = None
        if ctx.needs_input_grad[0]:
            dx_2d = torch.empty_like(x_2d)
            if n_rows > 0:
                rmsnorm_bwd_kernel[(n_rows,)](
                    dy_2d,
                    x_2d,
                    weight,
                    rstd,
                    dx_2d,
                    dy_2d.stride(0),
                    x_2d.stride(0),
                    dx_2d.stride(0),
                    n_cols,
                    BLOCK_N=ctx.block_n,
                    num_warps=ctx.num_warps,
                )
            dx = dx_2d.view(orig_shape)

        dw = None
        if ctx.needs_input_grad[1]:
            x_hat = x_2d.float() * rstd.unsqueeze(1)
            dw = (dy_2d.float() * x_hat).sum(dim=0).to(weight.dtype)

        return dx, dw, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonRMSNormFunction.apply(x, weight, eps)


def patch_qwen2_rmsnorm() -> None:
    from transformers.models.qwen2 import modeling_qwen2

    orig_forward = modeling_qwen2.Qwen2RMSNorm.forward
    if getattr(orig_forward, "_soup_triton_rmsnorm", False):
        return

    def rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_cuda:
            return orig_forward(self, hidden_states)
        return TritonRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon)

    rmsnorm_forward._soup_triton_rmsnorm = True
    modeling_qwen2.Qwen2RMSNorm.forward = rmsnorm_forward
