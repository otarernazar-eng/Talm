"""Qwen2 MLP with recomputation: the biggest activation of the step is not stored at all.

With the example SwiGLU the MLP keeps `gate` and `up` ([N, 8960] fp16 each, two per layer) for
its backward; at the peak of a step that is about 1 GiB, more than any other activation. Here
the MLP forward runs without a graph and saves only its input `x` ([N, 1536] fp16, which the
norm produced anyway); the backward recomputes `gate` and `up` with the same 4-bit projections
(bit-identical, the GEMMs are deterministic), takes the gradient of `down_proj` from the
dequantized weight directly, and backpropagates through the activation and the two
projections. The extra work is two projection GEMMs per layer in the backward pass; the
down_proj GEMM is not recomputed. The recomputation lives in our own code and switches no
Soup mode: Soup's gradient_checkpointing stays off (check 5 compares the trainer settings).
Opt-in and OFF by default (SOLUTION_MLP_RECOMPUTE=1): it costs about 25% of the training-loop
time for about 1.2 GiB of peak memory, and speed is scored as a log ratio too.

The MLP projections carry no LoRA in this task (q_proj and v_proj only), so nothing trainable
sits inside the recomputed region and no RNG is involved.
"""

import torch

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback
from .swiglu import swiglu                                                                      


def _weight16(linear):
    """The fp16 dense weight of a 4-bit (bitsandbytes) or plain linear layer."""
    weight = linear.weight
    state = getattr(weight, "quant_state", None)
    if state is not None:
        import bitsandbytes.functional as bnbf

        return bnbf.dequantize_4bit(weight.data, state).to(torch.float16)
    return weight.detach().to(torch.float16)


class MLPRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mlp):
        with torch.no_grad():
            out = mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x)))
        ctx.mlp = mlp
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        mlp = ctx.mlp
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            act = swiglu(mlp.gate_proj(x_), mlp.up_proj(x_))
                                                                                       
        grad_act = torch.mm(grad_out.reshape(-1, grad_out.shape[-1]).to(torch.float16),
                            _weight16(mlp.down_proj)).view(act.shape).to(act.dtype)
        (grad_x,) = torch.autograd.grad(act, x_, grad_act)
        return grad_x, None


def _recompute_applies(mlp, x):
    return (x.is_cuda and torch.is_grad_enabled() and x.requires_grad
            and getattr(mlp, "training", False)
            and getattr(getattr(mlp, "config", None), "hidden_act", "silu") == "silu"
            and all(not p.requires_grad for p in mlp.parameters())
            and getattr(mlp.down_proj, "bias", None) is None)


def patch_qwen2_mlp_recompute():
    """Send `Qwen2MLP.forward` through the recomputing function in training; otherwise through
    whatever forward was installed before (the example's fused SwiGLU)."""
    from transformers.models.qwen2 import modeling_qwen2

    from . import _in_chain

    cls = modeling_qwen2.Qwen2MLP
    previous = cls.forward
    if _in_chain(previous, "_solution_recompute"):
        return

    def forward(self, x):
                                                                                              
                                                                                    
        if "mlp_recompute" not in DISABLED and _recompute_applies(self, x):
            try:
                y = MLPRecompute.apply(x, self)
            except Exception as exc:                               
                if not fallback("mlp_recompute", exc):
                    raise
            else:
                RAN["mlp_recompute"] += 1
                return y
        FELL_BACK["mlp_recompute"] += 1
        return previous(self, x)

    forward._solution_recompute = True
    forward._previous = previous
    cls.forward = forward
    ON.add("mlp_recompute")
