"""Fused Triton Rotary Position Embedding (RoPE) forward and backward kernels.

Replaces transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb, which in
eager PyTorch creates 6 temporary tensors per attention layer via rotate_half
(slice + negate + torch.cat) and elementwise broadcasts, and saves q and k in
autograd. Our Triton kernels load cos/sin once per (batch, seq) token position into
registers, rotate all Query and Key heads in a single launch, and compute exact
analytical gradients in the backward pass without saving q or k.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rope_qk_fwd_kernel(
    Q_ptr,
    K_ptr,
    COS_ptr,
    SIN_ptr,
    QO_ptr,
    KO_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_cos_b,
    stride_cos_s,
    stride_sin_b,
    stride_sin_s,
    stride_qob,
    stride_qoh,
    stride_qos,
    stride_kob,
    stride_koh,
    stride_kos,
    S,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF_D

    cos_base = COS_ptr + b * stride_cos_b + s * stride_cos_s
    sin_base = SIN_ptr + b * stride_sin_b + s * stride_sin_s

    c1 = tl.load(cos_base + offs, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(cos_base + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sin_base + offs, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_base + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)

    q_row_in = Q_ptr + b * stride_qb + s * stride_qs
    q_row_out = QO_ptr + b * stride_qob + s * stride_qos
    for h in range(H_Q):
        q_in = q_row_in + h * stride_qh
        q_out = q_row_out + h * stride_qoh
        q1 = tl.load(q_in + offs, mask=mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_in + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)
        oq1 = q1 * c1 - q2 * s1
        oq2 = q2 * c2 + q1 * s2
        tl.store(q_out + offs, oq1.to(QO_ptr.dtype.element_ty), mask=mask)
        tl.store(q_out + HALF_D + offs, oq2.to(QO_ptr.dtype.element_ty), mask=mask)

    k_row_in = K_ptr + b * stride_kb + s * stride_ks
    k_row_out = KO_ptr + b * stride_kob + s * stride_kos
    for h in range(H_K):
        k_in = k_row_in + h * stride_kh
        k_out = k_row_out + h * stride_koh
        k1 = tl.load(k_in + offs, mask=mask, other=0.0).to(tl.float32)
        k2 = tl.load(k_in + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)
        ok1 = k1 * c1 - k2 * s1
        ok2 = k2 * c2 + k1 * s2
        tl.store(k_out + offs, ok1.to(KO_ptr.dtype.element_ty), mask=mask)
        tl.store(k_out + HALF_D + offs, ok2.to(KO_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rope_qk_bwd_kernel(
    DQO_ptr,
    DKO_ptr,
    COS_ptr,
    SIN_ptr,
    DQ_ptr,
    DK_ptr,
    stride_dqob,
    stride_dqoh,
    stride_dqos,
    stride_dkob,
    stride_dkoh,
    stride_dkos,
    stride_cos_b,
    stride_cos_s,
    stride_sin_b,
    stride_sin_s,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dkb,
    stride_dkh,
    stride_dks,
    S,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF_D

    cos_base = COS_ptr + b * stride_cos_b + s * stride_cos_s
    sin_base = SIN_ptr + b * stride_sin_b + s * stride_sin_s

    c1 = tl.load(cos_base + offs, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(cos_base + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sin_base + offs, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_base + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)

    dqo_row = DQO_ptr + b * stride_dqob + s * stride_dqos
    dq_row = DQ_ptr + b * stride_dqb + s * stride_dqs
    for h in range(H_Q):
        dqo_in = dqo_row + h * stride_dqoh
        dq_out = dq_row + h * stride_dqh
        dq1 = tl.load(dqo_in + offs, mask=mask, other=0.0).to(tl.float32)
        dq2 = tl.load(dqo_in + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)
        gq1 = dq1 * c1 + dq2 * s2
        gq2 = dq2 * c2 - dq1 * s1
        tl.store(dq_out + offs, gq1.to(DQ_ptr.dtype.element_ty), mask=mask)
        tl.store(dq_out + HALF_D + offs, gq2.to(DQ_ptr.dtype.element_ty), mask=mask)

    dko_row = DKO_ptr + b * stride_dkob + s * stride_dkos
    dk_row = DK_ptr + b * stride_dkb + s * stride_dks
    for h in range(H_K):
        dko_in = dko_row + h * stride_dkoh
        dk_out = dk_row + h * stride_dkh
        dk1 = tl.load(dko_in + offs, mask=mask, other=0.0).to(tl.float32)
        dk2 = tl.load(dko_in + HALF_D + offs, mask=mask, other=0.0).to(tl.float32)
        gk1 = dk1 * c1 + dk2 * s2
        gk2 = dk2 * c2 - dk1 * s1
        tl.store(dk_out + offs, gk1.to(DK_ptr.dtype.element_ty), mask=mask)
        tl.store(dk_out + HALF_D + offs, gk2.to(DK_ptr.dtype.element_ty), mask=mask)


class TritonRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()
        if cos.stride(-1) != 1:
            cos = cos.contiguous()
        if sin.stride(-1) != 1:
            sin = sin.contiguous()

        if unsqueeze_dim == 1:
            B, H_q, S, D = q.shape
            _, H_k, _, _ = k.shape
            stride_qb, stride_qh, stride_qs = q.stride(0), q.stride(1), q.stride(2)
            stride_kb, stride_kh, stride_ks = k.stride(0), k.stride(1), k.stride(2)
        else:
            B, S, H_q, D = q.shape
            _, _, H_k, _ = k.shape
            stride_qb, stride_qs, stride_qh = q.stride(0), q.stride(1), q.stride(2)
            stride_kb, stride_ks, stride_kh = k.stride(0), k.stride(1), k.stride(2)

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        if unsqueeze_dim == 1:
            stride_qob, stride_qoh, stride_qos = q_out.stride(0), q_out.stride(1), q_out.stride(2)
            stride_kob, stride_koh, stride_kos = k_out.stride(0), k_out.stride(1), k_out.stride(2)
        else:
            stride_qob, stride_qos, stride_qoh = q_out.stride(0), q_out.stride(1), q_out.stride(2)
            stride_kob, stride_kos, stride_koh = k_out.stride(0), k_out.stride(1), k_out.stride(2)

        stride_cos_b = 0 if cos.shape[0] == 1 else cos.stride(0)
        stride_cos_s = cos.stride(1)
        stride_sin_b = 0 if sin.shape[0] == 1 else sin.stride(0)
        stride_sin_s = sin.stride(1)

        half_d = D // 2
        block_half = triton.next_power_of_2(half_d)
        grid = (B * S,)

        if B * S > 0:
            rope_qk_fwd_kernel[grid](
                q,
                k,
                cos,
                sin,
                q_out,
                k_out,
                stride_qb,
                stride_qh,
                stride_qs,
                stride_kb,
                stride_kh,
                stride_ks,
                stride_cos_b,
                stride_cos_s,
                stride_sin_b,
                stride_sin_s,
                stride_qob,
                stride_qoh,
                stride_qos,
                stride_kob,
                stride_koh,
                stride_kos,
                S,
                H_Q=H_q,
                H_K=H_k,
                HALF_D=half_d,
                BLOCK_HALF=block_half,
                num_warps=2,
            )

        ctx.save_for_backward(cos, sin)
        ctx.unsqueeze_dim = unsqueeze_dim
        ctx.half_d = half_d
        ctx.block_half = block_half
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        cos, sin = ctx.saved_tensors
        unsqueeze_dim = ctx.unsqueeze_dim
        half_d = ctx.half_d
        block_half = ctx.block_half

        if grad_q.stride(-1) != 1:
            grad_q = grad_q.contiguous()
        if grad_k.stride(-1) != 1:
            grad_k = grad_k.contiguous()

        if unsqueeze_dim == 1:
            B, H_q, S, _ = grad_q.shape
            _, H_k, _, _ = grad_k.shape
            stride_dqob, stride_dqoh, stride_dqos = grad_q.stride(0), grad_q.stride(1), grad_q.stride(2)
            stride_dkob, stride_dkoh, stride_dkos = grad_k.stride(0), grad_k.stride(1), grad_k.stride(2)
        else:
            B, S, H_q, _ = grad_q.shape
            _, _, H_k, _ = grad_k.shape
            stride_dqob, stride_dqos, stride_dqoh = grad_q.stride(0), grad_q.stride(1), grad_q.stride(2)
            stride_dkob, stride_dkos, stride_dkoh = grad_k.stride(0), grad_k.stride(1), grad_k.stride(2)

        dq = torch.empty_like(grad_q)
        dk = torch.empty_like(grad_k)

        if unsqueeze_dim == 1:
            stride_dqb, stride_dqh, stride_dqs = dq.stride(0), dq.stride(1), dq.stride(2)
            stride_dkb, stride_dkh, stride_dks = dk.stride(0), dk.stride(1), dk.stride(2)
        else:
            stride_dqb, stride_dqs, stride_dqh = dq.stride(0), dq.stride(1), dq.stride(2)
            stride_dkb, stride_dks, stride_dkh = dk.stride(0), dk.stride(1), dk.stride(2)

        stride_cos_b = 0 if cos.shape[0] == 1 else cos.stride(0)
        stride_cos_s = cos.stride(1)
        stride_sin_b = 0 if sin.shape[0] == 1 else sin.stride(0)
        stride_sin_s = sin.stride(1)

        if B * S > 0:
            rope_qk_bwd_kernel[(B * S,)](
                grad_q,
                grad_k,
                cos,
                sin,
                dq,
                dk,
                stride_dqob,
                stride_dqoh,
                stride_dqos,
                stride_dkob,
                stride_dkoh,
                stride_dkos,
                stride_cos_b,
                stride_cos_s,
                stride_sin_b,
                stride_sin_s,
                stride_dqb,
                stride_dqh,
                stride_dqs,
                stride_dkb,
                stride_dkh,
                stride_dks,
                S,
                H_Q=H_q,
                H_K=H_k,
                HALF_D=half_d,
                BLOCK_HALF=block_half,
                num_warps=2,
            )

        return dq, dk, None, None, None


def triton_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not q.is_cuda or q.ndim != 4 or cos.ndim != 3 or (q.shape[-1] % 2 != 0):
        from transformers.models.qwen2.modeling_qwen2 import rotate_half

        cos_u = cos.unsqueeze(unsqueeze_dim)
        sin_u = sin.unsqueeze(unsqueeze_dim)
        return (q * cos_u) + (rotate_half(q) * sin_u), (k * cos_u) + (rotate_half(k) * sin_u)
    return TritonRoPEFunction.apply(q, k, cos, sin, unsqueeze_dim)


def patch_qwen2_rope() -> None:
    from transformers.models.qwen2 import modeling_qwen2

    if getattr(modeling_qwen2.apply_rotary_pos_emb, "_soup_triton_rope", False):
        return

    triton_apply_rotary_pos_emb._soup_triton_rope = True
    modeling_qwen2.apply_rotary_pos_emb = triton_apply_rotary_pos_emb
    if hasattr(modeling_qwen2.Qwen2Attention, "apply_rotary_pos_emb"):
        setattr(
            modeling_qwen2.Qwen2Attention,
            "apply_rotary_pos_emb",
            staticmethod(triton_apply_rotary_pos_emb),
        )
