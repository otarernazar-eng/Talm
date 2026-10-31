"""EXAMPLE: fused SwiGLU, silu(gate) * up, for the Qwen2 MLP, forward and backward in Triton.

It shows the whole path: a Triton kernel, a torch.autograd.Function around it, and a patch that
makes Soup's model call it. These kernels do not count as your own kernel, under any name: add your
own kernels next to them (keeping the example running is fine) or replace this file.

(Ours, around the example: graceful degradation, see state.py. An unexpected error in a launch
turns "swiglu" off; the forward then computes transformers' own `act_fn(gate) * up` and the
backward the same gradient formula with PyTorch ops.)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .engaged import FELL_BACK
from .state import DISABLED, fallback

BLOCK = 1024


@triton.jit
def example_swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def example_swiglu_bwd_kernel(grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements,
                              BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(dgate_ptr + offsets, dgate.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offsets, dup.to(dup_ptr.dtype.element_ty), mask=mask)


def reference_grads(grad, gate, up):
    """The backward kernel's formulas in PyTorch ops (fp32), for a launch that failed."""
    FELL_BACK["swiglu_inplace"] += 1  # engaged.py's MLP counter: this call degraded
    g, a, b = grad.float(), gate.float(), up.float()
    sig = torch.sigmoid(a)
    dgate = g * b * sig * (1.0 + a * (1.0 - sig))
    return dgate.to(gate.dtype), (g * a * sig).to(up.dtype)


class SwiGLU(torch.autograd.Function):
    """silu(gate) * up in one launch each way instead of three PyTorch ops."""

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        grad = grad.to(gate.dtype).contiguous()
        if "swiglu" not in DISABLED:
            try:
                dgate, dup = torch.empty_like(gate), torch.empty_like(up)
                n = gate.numel()
                example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate, up, dgate, dup, n,
                                                                   BLOCK=BLOCK)
                return dgate, dup
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("swiglu", exc):
                    raise
        return reference_grads(grad, gate, up)


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


def activate(mlp, gate, up, fused):
    """`fused(gate, up)`; on an unexpected error (state.py) the original activation of the
    MLP, `act_fn(gate) * up`, on the same projections."""
    try:
        return fused(gate, up)
    except Exception as exc:  # noqa: BLE001 - see state.py
        if not fallback("swiglu", exc):
            raise
    FELL_BACK["swiglu_inplace"] += 1  # engaged.py's MLP counter: this call degraded
    return getattr(mlp, "act_fn", F.silu)(gate) * up


def patch_qwen2_mlp():
    """Send Qwen2MLP.forward through the fused kernel. A class patch works before or after Soup
    builds the model: Python looks the method up on every call."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda or "swiglu" in DISABLED:
            return original(self, x)
        return self.down_proj(activate(self, self.gate_proj(x), self.up_proj(x), swiglu))

    modeling_qwen2.Qwen2MLP.forward = forward
