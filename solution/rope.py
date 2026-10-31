"""Fused rotary position embedding (RoPE) for q and k, forward and backward in one Triton kernel.

transformers' `apply_rotary_pos_emb` is `q * cos + rotate_half(q) * sin` on q and k: two slices,
a negation, a `cat`, two multiplies and an add per tensor, in fp32 because `cos`/`sin` are fp32
(they follow the fp32 residual stream), followed by the autocast cast of the fp32 result back to
fp16 for SDPA; the backward mirrors every op. That is about 25 small launches per layer, around
900 per step, and roughly 25 ms of GPU time plus the launch overhead that keeps the T4 idle.

Here one launch rotates both q (12 heads) and k (2 heads): a program takes a block of tokens of
one head, loads the two halves of the head dimension, applies
    y1 = x1 * cos1 - x2 * sin1,   y2 = x2 * cos2 + x1 * sin2
in fp32 and writes fp16 directly (the same single rounding the autocast cast did). The backward
is the same kernel with `backward=1`:
    dx1 = dy1 * cos1 + dy2 * sin2,   dx2 = dy2 * cos2 - dy1 * sin1.
Only `cos` and `sin` are kept for the backward, as in the base.

Output layouts: the forward writes q and k head-major ([1, H, N, D], what the attention reads),
the backward writes dq and dk token-major ([1, N, H, D] memory, seen as [1, H, N, D]). q and k
are `proj(x).view(1, N, H, D).transpose(1, 2)`, so autograd turns their gradients back into the
[1, N, H * D] projection-output layout with a transpose and a view: from a head-major gradient
that view is a copy (one strided [N, H * D] copy kernel for q and one for k per layer, 56 per
step); from a token-major one it is free. The same values, only where they are stored
(bit-identical, tested); SOLUTION_DISABLE_ROPE_TOKEN_MAJOR=1 writes the backward head-major as
before. Measured on the T4 together with the attention backward of unpad.py (same session,
per-kernel profile): strided copy kernels 111 -> 28 per step, 2.39 -> 1.01 ms.

Addressing: every stride is passed in whole rows of the head dimension and multiplied by a
constexpr in the kernel (`UNIT`), so the compiler can prove each row of q, k, cos and sin is
aligned and vectorizes the loads and stores (with element strides, runtime integers that are not
specialized, it cannot, and every lane moved one fp16). The q/k views always have such strides;
anything else is copied first (`_row_strided`). Same values (bit-identical, tested:
test_rope_copies_inputs_whose_strides_are_not_whole_rows). Measured on the T4 (same
session, per-kernel profile): rope_kernel 3.75 -> 1.65 ms per step.
SOLUTION_DISABLE_ROPE_ROWS=1 passes element strides as before.

Graceful degradation (state.py): an unexpected error in the forward launch turns "rope" off and
the call runs through transformers' `apply_rotary_pos_emb`; an error in the backward launch
computes the same rotation with PyTorch ops (`_reference`).

Any even head size up to 256: a program loads BLOCK_D = next_power_of_2(D / 2) columns of each
half, the ones past D / 2 masked (tl.arange takes powers of two only). Head sizes whose half is
a power of two (Qwen2.5's 128) compile exactly as before (BLOCK_D = HALF, the mask all true);
the others (96, 80, 6, ...) failed to compile before this, although `_applies` sent them here
(found by solution/test_fuzz.py).
"""

import os

import torch
import triton
import triton.language as tl

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

BLOCK_T = 16   # tokens per program
NUM_WARPS = 4
# The backward writes dq/dk token-major ([1, N, H, D] memory): see the module docstring.
TOKEN_MAJOR_GRAD = os.environ.get("SOLUTION_DISABLE_ROPE_TOKEN_MAJOR") != "1"
# Input strides passed in whole rows of the head dimension (UNIT = D): see the kernel.
ROW_STRIDES = os.environ.get("SOLUTION_DISABLE_ROPE_ROWS") != "1"


@triton.jit(do_not_specialize=["n_tokens", "n_q_heads", "stride_qh", "stride_qn", "stride_kh",
                               "stride_kn", "stride_cn", "rows_qh", "rows_qn", "rows_kh",
                               "rows_kn", "backward"])
def rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, out_q_ptr, out_k_ptr, n_tokens, n_q_heads,
                stride_qh, stride_qn, stride_kh, stride_kn, stride_cn, rows_qh, rows_qn, rows_kh,
                rows_kn, backward, HALF: tl.constexpr, BLOCK_T: tl.constexpr,
                UNIT: tl.constexpr, BLOCK_D: tl.constexpr):
    """Inputs: element (head, token, d) of q at (head * stride_qh + token * stride_qn) * UNIT + d
    (k and cos/sin alike); outputs at (head * rows_h + token * rows_n) * 2 * HALF + d, strides
    in whole rows of the head dimension (head-major: rows_h = N, rows_n = 1; token-major:
    rows_h = 1, rows_n = heads). With UNIT = 2 * HALF (the input strides in rows too) every
    access is provably row-aligned and vectorizes; UNIT = 1 takes element strides. BLOCK_D: the
    power of two >= HALF that tl.arange needs, columns d >= HALF masked."""
    pid_t = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    tok = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, BLOCK_D)
    tmask = tok < n_tokens
    mask = tmask[:, None] & (d[None, :] < HALF)
    tok64 = tok.to(tl.int64)[:, None]

    c_off = tok64 * stride_cn * UNIT + d[None, :]
    c1 = tl.load(cos_ptr + c_off, mask=mask, other=0.0)
    c2 = tl.load(cos_ptr + c_off + HALF, mask=mask, other=0.0)
    s1 = tl.load(sin_ptr + c_off, mask=mask, other=0.0)
    s2 = tl.load(sin_ptr + c_off + HALF, mask=mask, other=0.0)

    if head < n_q_heads:
        in_base = q_ptr + (head * stride_qh + tok64 * stride_qn) * UNIT + d[None, :]
        out_base = out_q_ptr + (head * rows_qh + tok64 * rows_qn) * (2 * HALF) + d[None, :]
    else:
        kh = head - n_q_heads
        in_base = k_ptr + (kh * stride_kh + tok64 * stride_kn) * UNIT + d[None, :]
        out_base = out_k_ptr + (kh * rows_kh + tok64 * rows_kn) * (2 * HALF) + d[None, :]

    x1 = tl.load(in_base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in_base + HALF, mask=mask, other=0.0).to(tl.float32)
    if backward == 0:
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    else:
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    tl.store(out_base, y1.to(out_q_ptr.dtype.element_ty), mask=mask)
    tl.store(out_base + HALF, y2.to(out_q_ptr.dtype.element_ty), mask=mask)


def _launch(q, k, cos, sin, backward):
    """q [1, Hq, N, D], k [1, Hk, N, D] (last dim contiguous), cos/sin [1, N, D] fp32 contiguous.
    Returns new fp16 tensors of the same shapes: head-major (contiguous), or, for the backward
    when TOKEN_MAJOR_GRAD is on, [1, N, H, D] memory seen as [1, H, N, D]."""
    _, hq, n, dim = q.shape
    hk = k.shape[1]
    if backward and TOKEN_MAJOR_GRAD:
        out_q = torch.empty((1, n, hq, dim), dtype=torch.float16, device=q.device).transpose(1, 2)
        out_k = torch.empty((1, n, hk, dim), dtype=torch.float16, device=q.device).transpose(1, 2)
        rows = (1, hq, 1, hk)
    else:
        out_q = torch.empty((1, hq, n, dim), dtype=torch.float16, device=q.device)
        out_k = torch.empty((1, hk, n, dim), dtype=torch.float16, device=q.device)
        rows = (n, 1, n, 1)
    unit = dim if ROW_STRIDES else 1
    if unit > 1:  # input strides in whole rows (a copy only for a stride that is not one)
        q, k, cos = _row_strided(q, dim), _row_strided(k, dim), _row_strided(cos, dim)
    grid = (triton.cdiv(n, BLOCK_T), hq + hk)
    rope_kernel[grid](q, k, cos, sin, out_q, out_k, n, hq, q.stride(1) // unit,
                      q.stride(2) // unit, k.stride(1) // unit, k.stride(2) // unit,
                      cos.stride(1) // unit, *rows, int(backward), HALF=dim // 2,
                      BLOCK_T=BLOCK_T, UNIT=unit, BLOCK_D=triton.next_power_of_2(dim // 2),
                      num_warps=NUM_WARPS)
    return out_q, out_k


def _row_strided(t, dim):
    """`t` ([1, H, N, D] or [1, N, D]) with every stride above the last a multiple of `dim`
    (dimensions of size 1 do not count), copied to a contiguous tensor only if it is not."""
    if all(t.stride(i) % dim == 0 or t.shape[i] == 1 for i in range(t.dim() - 1)):
        return t
    return t.contiguous()


def _reference(x, cos, sin, backward):
    """The kernel's rotation of x [1, H, N, D] in PyTorch ops (fp32, written as fp16), for a
    launch that failed: forward y1 = x1 c1 - x2 s1, y2 = x2 c2 + x1 s2; backward
    dx1 = dy1 c1 + dy2 s2, dx2 = dy2 c2 - dy1 s1."""
    half = x.shape[-1] // 2
    x, c, s = x.float(), cos[:, None].float(), sin[:, None].float()
    x1, x2 = x[..., :half], x[..., half:]
    c1, c2, s1, s2 = c[..., :half], c[..., half:], s[..., :half], s[..., half:]
    if backward:
        y1, y2 = x1 * c1 + x2 * s2, x2 * c2 - x1 * s1
    else:
        y1, y2 = x1 * c1 - x2 * s1, x2 * c2 + x1 * s2
    return torch.cat((y1, y2), -1).to(torch.float16)


class RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        cos = cos.contiguous()
        sin = sin.contiguous()
        ctx.save_for_backward(cos, sin)
        RAN["rope"] += 1
        return _launch(q, k, cos, sin, backward=0)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        if dq.stride(-1) != 1:
            dq = dq.contiguous()
        if dk.stride(-1) != 1:
            dk = dk.contiguous()
        if dq.dtype != torch.float16:
            dq = dq.to(torch.float16)
        if dk.dtype != torch.float16:
            dk = dk.to(torch.float16)
        if "rope" not in DISABLED:
            try:
                gq, gk = _launch(dq, dk, cos, sin, backward=1)
                RAN["rope"] += 1
                return gq, gk, None, None
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("rope", exc):
                    raise
        FELL_BACK["rope"] += 1
        return _reference(dq, cos, sin, True), _reference(dk, cos, sin, True), None, None


def _applies(q, k, cos, sin, unsqueeze_dim):
    return (unsqueeze_dim == 1 and q.is_cuda and q.dtype == torch.float16
            and k.dtype == torch.float16 and q.dim() == 4 and k.dim() == 4 and q.shape[0] == 1
            and k.shape[0] == 1 and cos.dim() == 3 and cos.shape[0] == 1
            and cos.dtype == torch.float32 and sin.dtype == torch.float32
            and cos.shape == sin.shape and cos.shape[1] == q.shape[2] == k.shape[2]
            and cos.shape[2] == q.shape[3] == k.shape[3] and q.shape[3] % 2 == 0
            and q.shape[3] <= 256 and q.stride(-1) == 1 and k.stride(-1) == 1)


def patch_qwen2_rope():
    """Route `modeling_qwen2.apply_rotary_pos_emb` (looked up by name in Qwen2Attention.forward)
    through the fused kernel for the fp16 flat-batch case; anything else takes the original."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    ON.add("rope")
    if getattr(original, "_solution_rope", False):
        return

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, **kwargs):
        if not kwargs and "rope" not in DISABLED and _applies(q, k, cos, sin, unsqueeze_dim):
            try:
                return RoPE.apply(q, k, cos, sin)
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("rope", exc):
                    raise
        FELL_BACK["rope"] += 1
        return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim, **kwargs)

    apply_rotary_pos_emb._solution_rope = True
    apply_rotary_pos_emb._original = original
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
