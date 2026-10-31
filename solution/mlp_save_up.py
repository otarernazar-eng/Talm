"""Qwen2 MLP that keeps `up` and the MLP input for its backward and recomputes `gate` there.

With the example SwiGLU (and our in-place wrapper, swiglu_inplace.py) every decoder layer keeps
`gate` and `up` ([N, 8960] fp16 each) for the backward: at the peak of the step (the longest
batch of the public rows, N = 1427 real tokens) 2 x 28 x 24.4 = 1 366 MiB, the largest item of
the step. Here the two projections and the activation are one autograd node (`SwiGLUSaveUp`)
that keeps only `up` and the MLP input x ([N, 1536] fp16, the post-attention norm's output):
24.4 - 4.2 = 20.2 MiB less per layer at N = 1427. The activation silu(gate) * up is written over
`gate` (nothing reads gate after it any more), so the forward of every layer also allocates one
[N, 8960] buffer less, and that buffer was alive at the peak of the step (the down projection of
the last layer).

The backward recomputes gate = x W_gate^T with the fp16 weight that the backward of the layer
dequantizes anyway (qgroup.py: one dequantization per layer and pass; the same weight serves
the recomputation and the gradient GEMM, so the layer group sees the same number of weight
uses as before): one extra GEMM [N, 1536] x [1536, 8960] per layer and step, no extra
dequantization and no extra kernel of ours. The recomputed gate is bit-identical to the
forward's (the same GEMM on the same fp16 operands), and everything after it is the existing
path op for op: the example's backward kernel writing dgate over gate and dup over up
(swiglu_inplace.py), dx = dgate W_gate + dup W_up as two fp16 GEMMs and one fp16 add (autograd
added the two contributions the same way; fp16 addition is commutative), so outputs and
gradients are bit-identical to the default path (tested with torch.equal).

Measured (the T4's fp16 path on an RTX 4080, the allocator's block sizes as on the T4: main's
max_allocated 3 659.2 MiB matches the T4's 3 659), at the peak step: max_allocated 3 659.2 ->
3 101.0 MiB, reserved 3 776 -> 3 216 MiB, so about -560 MiB of nvidia-smi peak; the peak stays the
down projection of the last layer in the forward. Time: one GEMM of 2 x N x 1536 x 8960 FLOP per
layer, 0.64 TFLOP on an average step (N = 826); at the ~26 TFLOPS cuBLAS reaches on the T4
(40% of the base's 884 ms step for its ~9.3 TFLOP of matmuls) about 24 ms per step and 1.6 s per
run (+6% of the training loop, +3.6% of the wall time) estimated; measured on the T4 (one
session, runs alternating ABABAB): peak 3 921 -> 3 361 MiB (-560), wall time +1.2 to +2.0 s
(+4%), timed loop +1.5 to +2.5 s.

Opt-in (SOLUTION_MLP_SAVE_UP=1), off by default: speed is the first priority. The MLP
projections carry no LoRA in this task (q_proj and v_proj only): the node applies only when
gate_proj and up_proj are packed 4-bit projections of the same layer group (qgroup) with fp16
input that requires grad, and takes the previous MLP forward otherwise (eval, no grad, adapters
on the MLP, another activation).
"""

import torch
import torch.nn.functional as F
import triton

from .state import DISABLED, fallback
from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

GATE, UP = 4, 5                                                                    


class SwiGLUSaveUp(torch.autograd.Function):
    """silu(x W_gate^T) * (x W_up^T) on a packed 4-bit layer group; saves x and up only."""

    @staticmethod
    def forward(ctx, x, group):
        weight = group.weight("fwd", GATE)
        gate = F.linear(x, weight, group.biases[GATE])                                      
        del weight                                                                       
        up = F.linear(x, group.weight("fwd", UP), group.biases[UP])
        n = gate.numel()
                                                                                              
                                                       
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, gate, n, BLOCK=BLOCK)
        ctx.group = group
        ctx.save_for_backward(x, up)
        return gate

    @staticmethod
    def backward(ctx, grad):
        x, up = ctx.saved_tensors
        group = ctx.group
        weight = group.weight("bwd", GATE)
        gate = F.linear(x, weight, group.biases[GATE])                                   
        grad = grad.to(gate.dtype).contiguous()
        n = gate.numel()
                                                         
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate, up, gate, up, n,
                                                           BLOCK=BLOCK)
        del grad
        dx = torch.matmul(gate, weight)                                     
        del gate, weight
        dx += torch.matmul(up, group.weight("bwd", UP))                                         
        return dx, None


def save_up_group(mlp, x):
    """The layer group when `SwiGLUSaveUp` can take this MLP call, else None: training (grad
    on, x requires grad), fp16 CUDA input, SiLU, gate_proj and up_proj packed by qgroup in the
    same group at their usual positions, no hooks on them (their forward is not called)."""
    if not (x.is_cuda and x.dtype == torch.float16 and torch.is_grad_enabled()
            and x.requires_grad):
        return None
    if getattr(getattr(mlp, "config", None), "hidden_act", "silu") != "silu":
        return None
    gate, up = getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None)
    tag_gate = getattr(gate, "_solution_group", None)
    tag_up = getattr(up, "_solution_group", None)
    if tag_gate is None or tag_up is None or tag_gate[0] is not tag_up[0] \
            or tag_gate[1] != GATE or tag_up[1] != UP:
        return None
    if any(m._forward_hooks or m._forward_pre_hooks for m in (gate, up)):
        return None
    return tag_gate[0]


def patch_qwen2_mlp_save_up():
    """`Qwen2MLP.forward` through `SwiGLUSaveUp` when it applies (see `save_up_group`), else
    through the forward installed before (the example / in-place wrapper)."""
    from transformers.models.qwen2 import modeling_qwen2

    from .engaged import FELL_BACK, ON, RAN

    cls = modeling_qwen2.Qwen2MLP
    previous = cls.forward
    ON.add("mlp_save_up")
    from . import _in_chain

    if _in_chain(previous, "_solution_save_up"):
        return

    def forward(self, x):
                                                                                        
                                                                
        group = None if "mlp_save_up" in DISABLED else save_up_group(self, x)
        if group is not None:
            try:
                act = SwiGLUSaveUp.apply(x, group)
            except Exception as exc:                               
                if not fallback("mlp_save_up", exc):
                    raise
            else:
                group.expected_bwd += 2                                                   
                RAN["mlp_save_up"] += 1
                return self.down_proj(act)
        FELL_BACK["mlp_save_up"] += 1
        return previous(self, x)

    forward._solution_save_up = True
    forward._previous = previous
    cls.forward = forward
