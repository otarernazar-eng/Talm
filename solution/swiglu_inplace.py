"""The organizers' example SwiGLU with its backward written in place (our wrapper, their kernels).

The example's autograd function keeps `gate` and `up` ([N, 8960] fp16 each) for its backward and
allocates two new tensors of that size for `dgate` and `dup`. Nothing else reads `gate` and `up`
after this backward, so here the example's backward kernel writes `dgate` over `gate` and `dup`
over `up` (each program loads its elements before storing the results at the same offsets, so
the in-place update is safe). Same kernels, same arithmetic, same results; two [N, 8960] buffers
fewer at the start of the backward pass, where every activation of the step is still alive.
The kernels stay the example's (`example_*`): they do not count as ours.
SOLUTION_DISABLE_SWIGLU_INPLACE=1 (read by solution.apply()) keeps the example's own backward.
"""

import torch
import triton

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback
from .swiglu import (BLOCK, activate, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel,
                     reference_grads, swiglu)


class SwiGLUInPlaceGrad(torch.autograd.Function):
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
                n = gate.numel()
                example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate, up, gate, up, n,
                                                                   BLOCK=BLOCK)
                return gate, up  # now dgate and dup
            except Exception as exc:  # noqa: BLE001 - see state.py
                # a launch that raised wrote nothing: gate and up still hold the activations
                if not fallback("swiglu", exc):
                    raise
        return reference_grads(grad, gate, up)


def patch_qwen2_mlp_inplace():
    """Qwen2MLP.forward through the in-place-backward wrapper when gate and up are separate
    fresh tensors of the same dtype (the training path); otherwise the previous forward."""
    from transformers.models.qwen2 import modeling_qwen2

    from . import _in_chain

    cls = modeling_qwen2.Qwen2MLP
    previous = cls.forward
    if _in_chain(previous, "_solution_inplace"):
        return

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda or "swiglu" in DISABLED:
            FELL_BACK["swiglu_inplace"] += 1
            return previous(self, x)
        gate, up = self.gate_proj(x), self.up_proj(x)
        if gate.dtype != up.dtype or gate.data_ptr() == up.data_ptr():
            FELL_BACK["swiglu_inplace"] += 1
            return self.down_proj(activate(self, gate, up, swiglu))
        RAN["swiglu_inplace"] += 1
        return self.down_proj(activate(self, gate, up, SwiGLUInPlaceGrad.apply))

    forward._solution_inplace = True
    forward._previous = previous
    cls.forward = forward
    ON.add("swiglu_inplace")
