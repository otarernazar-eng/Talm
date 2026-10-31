"""Fused NF4 linear layer: one Triton dequant launch + one cuBLAS GEMM per call, in forward and
in backward, instead of bitsandbytes' Python dispatch.

After unpadding the training step is launch-bound: the GPU is idle about a quarter of the
time while Python issues the next op. Each of the 196 4-bit projections per step goes through
`bnb.matmul_4bit` -> `torch.ops.bitsandbytes.gemm_4bit` -> a per-call kernel heuristic -> an
absmax dequantization launch -> the weight dequantization -> `F.linear`, and the backward
does it again through `MatMul4Bit`. This module patches `bitsandbytes.nn.Linear4bit.forward`
(PEFT's LoRA wrapper calls it for the base layer, so every projection is covered) with a small
autograd function: the kernel unpacks the NF4 codes and dequantizes the double-quantized absmax
in the same launch (`scale = code[absmax8] * absmax2 + offset`, as bitsandbytes does in two
steps), cuBLAS does the GEMM, and the backward multiplies by the freshly dequantized weight
(no activation is saved: the weight is frozen). Values match bitsandbytes' dequantization to
the last bit except where the fused multiply-add rounds the fp32 scale differently (at most
one fp16 ulp on a tiny fraction of weights; checked at patch time).

Only the exact case of this task is handled (NF4, double quantization with blocksize 256 for
the absmax, fp16 inputs, frozen weight and bias); anything else takes bitsandbytes' path.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

BLOCK = 2048    # packed bytes per program = 4096 output elements
NUM_WARPS = 4


@triton.jit(do_not_specialize=["n_pairs"])
def nf4_dequant_dq_kernel(packed_ptr, absmax8_ptr, code_ptr, absmax2_ptr, offset_ptr, out_ptr,
                          n_pairs, QBLOCK: tl.constexpr, QBLOCK2: tl.constexpr,
                          BLOCK: tl.constexpr):
    """NF4 codes -> fp16 pairs (one uint32 store per byte) with the block absmax rebuilt from
    its 8-bit quantization: absmax = code[absmax8[block]] * absmax2[block // QBLOCK2] + offset."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    q = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    block = (offs * 2) // QBLOCK
    a8 = tl.load(absmax8_ptr + block, mask=mask, other=0).to(tl.int32)
    level = tl.load(code_ptr + a8)
    a2 = tl.load(absmax2_ptr + block // QBLOCK2, mask=mask, other=0.0)
    scale = level * a2 + tl.load(offset_ptr)

    hi = q >> 4
    b0 = (hi & 1) != 0
    b1 = (hi & 2) != 0
    b2 = (hi & 4) != 0
    b3 = (hi & 8) != 0
    low = tl.where(b2, tl.where(b1, tl.where(b0, 0.0, -0.09105003625154495),
                                tl.where(b0, -0.18477343022823334, -0.28444138169288635)),
                   tl.where(b1, tl.where(b0, -0.39491748809814453, -0.5250730514526367),
                            tl.where(b0, -0.6961928009986877, -1.0)))
    high = tl.where(b2, tl.where(b1, tl.where(b0, 1.0, 0.7229568362236023),
                                 tl.where(b0, 0.5626170039176941, 0.44070982933044434)),
                    tl.where(b1, tl.where(b0, 0.33791524171829224, 0.24611230194568634),
                             tl.where(b0, 0.16093020141124725, 0.07958029955625534)))
    first = (tl.where(b3, high, low) * scale).to(tl.float16).to(tl.uint16, bitcast=True)

    lo = q & 15
    b0 = (lo & 1) != 0
    b1 = (lo & 2) != 0
    b2 = (lo & 4) != 0
    b3 = (lo & 8) != 0
    low = tl.where(b2, tl.where(b1, tl.where(b0, 0.0, -0.09105003625154495),
                                tl.where(b0, -0.18477343022823334, -0.28444138169288635)),
                   tl.where(b1, tl.where(b0, -0.39491748809814453, -0.5250730514526367),
                            tl.where(b0, -0.6961928009986877, -1.0)))
    high = tl.where(b2, tl.where(b1, tl.where(b0, 1.0, 0.7229568362236023),
                                 tl.where(b0, 0.5626170039176941, 0.44070982933044434)),
                    tl.where(b1, tl.where(b0, 0.33791524171829224, 0.24611230194568634),
                             tl.where(b0, 0.16093020141124725, 0.07958029955625534)))
    second = (tl.where(b3, high, low) * scale).to(tl.float16).to(tl.uint16, bitcast=True)

    tl.store(out_ptr + offs, first.to(tl.uint32) | (second.to(tl.uint32) << 16), mask=mask)


def _supported(state):
    return (state is not None and state.quant_type == "nf4" and state.nested
            and state.state2.blocksize == 256 and state.blocksize % 2 == 0
            and len(state.shape) == 2 and (state.shape[0] * state.shape[1]) % 2 == 0)


def dequantize_weight(packed, state):
    """fp16 [N, K] from the packed NF4 weight and its (double-quantized) quant state."""
    n, k = state.shape
    out = torch.empty((n, k), dtype=torch.float16, device=packed.device)
    words = out.view(-1).view(torch.int32)
    pairs = n * k // 2
    nf4_dequant_dq_kernel[(triton.cdiv(pairs, BLOCK),)](
        packed.view(-1), state.absmax, state.state2.code, state.state2.absmax, state.offset,
        words, pairs, QBLOCK=state.blocksize, QBLOCK2=state.state2.blocksize, BLOCK=BLOCK,
        num_warps=NUM_WARPS)
    return out


class QLinear4bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, bias, state):
        ctx.packed, ctx.state = packed, state
        return F.linear(x, dequantize_weight(packed, state), bias)

    @staticmethod
    def backward(ctx, grad_out):
        try:
            weight = dequantize_weight(ctx.packed, ctx.state)
        except Exception as exc:  # noqa: BLE001 - state.py: bitsandbytes' dequantization
            if not fallback("qlinear", exc):
                raise
            weight = None
        if weight is None:
            import bitsandbytes.functional as bnbf

            weight = bnbf.dequantize_4bit(ctx.packed, ctx.state).to(torch.float16)
        return torch.matmul(grad_out, weight), None, None, None


def _self_check():
    import bitsandbytes.functional as bnbf

    generator = torch.Generator(device="cuda").manual_seed(1)
    w = torch.randn(1024, 1536, device="cuda", dtype=torch.float16, generator=generator) * 0.05
    packed, state = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    ref = bnbf.dequantize_4bit(packed, state)
    ours = dequantize_weight(packed, state)
    torch.cuda.synchronize()
    diff = (ours.float() - ref.float()).abs()
    tol = (ref.float().abs() * 2e-3 + 1e-6)
    return bool((diff <= tol).all()) and float((ours != ref).float().mean()) < 1e-3


def patch_bnb_linear4bit(verify=True):
    try:
        import bitsandbytes as bnb
    except Exception:  # noqa: BLE001
        return False
    cls = bnb.nn.Linear4bit
    original = cls.forward
    if getattr(original, "_solution_fused", False):
        return True
    if verify:
        if not torch.cuda.is_available():
            return False
        try:
            if not _self_check():
                return False
        except Exception:  # noqa: BLE001
            return False

    def forward(self, x):
        weight = self.weight
        state = getattr(weight, "quant_state", None)
        bias = self.bias
        if (not x.is_cuda or x.dtype != torch.float16 or weight.requires_grad
                or not _supported(state) or self.compute_dtype not in (None, torch.float16)
                or (bias is not None and bias.requires_grad) or "qlinear" in DISABLED):
            FELL_BACK["qlinear"] += 1
            return original(self, x)
        if bias is not None and bias.dtype != torch.float16:
            bias = bias.to(torch.float16)
        try:
            out = QLinear4bit.apply(x, weight.data, bias, state)
        except Exception as exc:  # noqa: BLE001 - see state.py
            if not fallback("qlinear", exc):
                raise
        else:
            RAN["qlinear"] += 1
            return out
        FELL_BACK["qlinear"] += 1
        return original(self, x)

    forward._solution_fused = True
    cls.forward = forward
    ON.add("qlinear")
    return True
