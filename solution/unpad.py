"""Unpadded Decoder Engine for Qwen2.5 (Triton + SDPA).

Executes the Qwen2.5 transformer backbone exclusively on real (non-pad) tokens:
1. `_build_ctx`: builds batch-level metadata once per step. Gathers non-pad indices
   into a contiguous 1D stream of shape `[T, H]`, where `T << B * S` (for Soup Kazakh
   dataset, `T` is ~850 instead of 2048, eliminating ~58% of all forward/backward FLOPs).
2. All 28 decoder layers (RMSNorm, fused LoRA/NF4 Q/K/V/O projections, fused SwiGLU MLP)
   execute directly on packed `[T, H]` token representations.
3. `_attn_forward`: rotates Q and K via analytical Rotary Position Embeddings on real
   positions, scatters `[T, n, hd]` into `[B, n, S_aligned, hd]` via `index_copy` for PyTorch's
   hardware-accelerated SDPA kernel, and immediately gathers the result back to `[T, H]`
   via `index_select`.
4. `fused_linear_cross_entropy`: chunked cross-entropy with in-place gradient computation
   directly on unpadded `[T, H]` hidden states and unpadded labels, reconstructing exact
   per-token loss, Shannon entropy, and token accuracy back into `[B, S]` tensors for
   loss metrics and `rewind.jsonl`.
"""

from __future__ import annotations

import types
from typing import Any, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint

from solution.rmsnorm import fused_rmsnorm
from solution.swiglu import fused_swiglu

IGNORE_INDEX = -100
_COMPUTE_DTYPE = torch.float16


# =====================================================================================
# 1. Chunked Cross-Entropy + Shannon Entropy + Token Accuracy Kernel on Unpadded Tokens
# =====================================================================================
@triton.jit
def unpad_ce_chunk_kernel(
    logits_ptr,      # [C, V] fp16/fp32, overwritten IN PLACE with (softmax - onehot)
    labels_ptr,      # [C] int64
    loss_ptr,        # [C] fp32 per-token unscaled loss
    entropy_ptr,     # [C] fp32 exact Shannon entropy
    correct_ptr,     # [C] int32 argmax correctness
    stride_logits,
    V,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_logits
    label = tl.load(labels_ptr + row).to(tl.int32)
    valid = (label != ignore_index) & (label >= 0) & (label < V)

    # Pass 1: online logsumexp, sum(exp(x - m) * x), and argmax over vocabulary
    m_curr = -float("inf")
    d_curr = 0.0
    s_curr = 0.0
    best_val = -float("inf")
    best_idx = 0

    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)

        tile_max = tl.max(x, 0)
        tile_arg = (off + tl.argmax(x, 0)).to(tl.int32)
        better = tile_max > best_val
        best_idx = tl.where(better, tile_arg, best_idx)
        best_val = tl.where(better, tile_max, best_val)

        m_next = tl.maximum(m_curr, tile_max)
        rescale = tl.exp(m_curr - m_next)
        exp_x = tl.exp(x - m_next)
        x_zero = tl.where(mask, x, 0.0)
        d_curr = d_curr * rescale + tl.sum(exp_x, 0)
        s_curr = s_curr * rescale + tl.sum(exp_x * x_zero, 0)
        m_curr = m_next

    lse = m_curr + tl.log(d_curr)
    entropy = lse - s_curr / d_curr
    tl.store(entropy_ptr + row, entropy)

    safe_label = tl.where(valid, label, 0)
    x_lab = tl.load(base + safe_label).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(valid, lse - x_lab, 0.0))
    hit = tl.where(valid & (best_idx == label), 1, 0)
    tl.store(correct_ptr + row, hit.to(tl.int32))

    # Pass 2: in-place unnormalized gradient wrt logits (softmax - onehot)
    for off in range(0, V, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        g = tl.exp(x - lse) - tl.where(cols == label, 1.0, 0.0)
        g = tl.where(valid, g, 0.0)
        tl.store(base + cols, g.to(logits_ptr.dtype.element_ty), mask=mask)


class UnpadFusedLinearCE(torch.autograd.Function):
    """Computes CE loss + gradient on packed [T, H] hidden states chunk-by-chunk."""

    @staticmethod
    def forward(ctx, h, W, labels, denom, chunk_size, ignore_index):
        with torch.autocast("cuda", enabled=False):
            T, H = h.shape
            V = W.shape[0]
            need_gh = h.requires_grad
            h_contig = h.contiguous()
            grad_h = torch.empty_like(h_contig) if need_gh else None

            loss_rows = torch.empty(T, dtype=torch.float32, device=h.device)
            entropy_rows = torch.empty(T, dtype=torch.float32, device=h.device)
            correct_rows = torch.empty(T, dtype=torch.int32, device=h.device)

            Wt = W.t()
            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                cur_chunk = e - s
                logits = torch.mm(h_contig[s:e], Wt)  # [cur_chunk, V] fp16 tensor cores
                unpad_ce_chunk_kernel[(cur_chunk,)](
                    logits,
                    labels[s:e],
                    loss_rows[s:e],
                    entropy_rows[s:e],
                    correct_rows[s:e],
                    logits.stride(0),
                    V,
                    ignore_index,
                    BLOCK_V=4096,
                    num_warps=8,
                )
                if need_gh:
                    torch.mm(logits, W, out=grad_h[s:e])
                del logits

            safe_denom = denom.clamp_min(1.0)
            loss = loss_rows.sum() / safe_denom

            ctx.save_for_backward(grad_h, safe_denom)
            ctx.need_gh = need_gh
            ctx.mark_non_differentiable(loss_rows, entropy_rows, correct_rows)

        return loss, loss_rows, entropy_rows, correct_rows

    @staticmethod
    def backward(ctx, grad_loss, _g_loss_rows, _g_ent_rows, _g_corr_rows):
        if not ctx.need_gh:
            return None, None, None, None, None, None
        grad_h, denom = ctx.saved_tensors
        scale = grad_loss.float() / denom
        return (grad_h.float() * scale).to(grad_h.dtype), None, None, None, None, None


# =====================================================================================
# 2. Batch Unpadding Context & Scatter/Gather Attention
# =====================================================================================
class _BatchCtx:
    __slots__ = ("B", "S0", "S", "T", "flat", "idx", "cos", "sin", "mask", "nh", "nkv", "hd", "rep")


def _build_batch_ctx(attention_mask: torch.Tensor, cfg: Any, device: torch.device, dtype: torch.dtype, inv_freq: torch.Tensor) -> _BatchCtx:
    B, S0 = attention_mask.shape
    S = (S0 + 7) // 8 * 8  # aligned sequence length for memory-efficient SDPA
    am = attention_mask.bool()
    flat = am.reshape(-1).nonzero(as_tuple=False).squeeze(1)

    c = _BatchCtx()
    c.B, c.S0, c.S, c.T = B, S0, S, flat.numel()
    c.flat = flat
    c.idx = (flat // S0) * S + (flat % S0)

    pos = (am.long().cumsum(1) - 1).clamp_min(0).reshape(-1)[flat]
    freqs = pos[:, None].float() * inv_freq[None, :]
    c.cos = freqs.cos().to(dtype)[:, None, :]
    c.sin = freqs.sin().to(dtype)[:, None, :]

    key_ok = torch.zeros(B, S, dtype=torch.bool, device=device)
    key_ok[:, :S0] = am
    causal = torch.ones(S, S, dtype=torch.bool, device=device).tril_()
    eye = torch.eye(S, dtype=torch.bool, device=device)
    allowed = (causal[None] & key_ok[:, None, :]) | eye[None]
    c.mask = torch.zeros(B, 1, S, S, dtype=dtype, device=device).masked_fill_(
        ~allowed[:, None], torch.finfo(dtype).min
    )

    c.nh = cfg.num_attention_heads
    c.nkv = cfg.num_key_value_heads
    c.hd = cfg.hidden_size // cfg.num_attention_heads
    c.rep = c.nh // c.nkv
    return c


def _apply_unpad_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _to_padded_tensor(x: torch.Tensor, c: _BatchCtx) -> torch.Tensor:
    T, n, hd = x.shape
    buf = x.new_zeros(c.B * c.S, n * hd).index_copy(0, c.idx, x.reshape(T, n * hd))
    return buf.view(c.B, c.S, n, hd).permute(0, 2, 1, 3)


def _attn_unpad_forward(attn: Any, x: torch.Tensor, c: _BatchCtx) -> torch.Tensor:
    T = x.shape[0]
    q = attn.q_proj(x).view(T, c.nh, c.hd)
    k = attn.k_proj(x).view(T, c.nkv, c.hd)
    v = attn.v_proj(x).view(T, c.nkv, c.hd)

    q = _apply_unpad_rope(q, c.cos, c.sin)
    k = _apply_unpad_rope(k, c.cos, c.sin)

    qp = _to_padded_tensor(q, c)
    kp = _to_padded_tensor(k, c)
    vp = _to_padded_tensor(v, c)

    if c.rep > 1:
        kp = kp[:, :, None].expand(c.B, c.nkv, c.rep, c.S, c.hd).reshape(c.B, c.nh, c.S, c.hd)
        vp = vp[:, :, None].expand(c.B, c.nkv, c.rep, c.S, c.hd).reshape(c.B, c.nh, c.S, c.hd)

    o = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=c.mask)
    o = o.permute(0, 2, 1, 3).reshape(c.B * c.S, c.nh * c.hd).index_select(0, c.idx)
    return attn.o_proj(o)


def _layer_unpad_forward(layer: Any, h: torch.Tensor, c: _BatchCtx) -> torch.Tensor:
    h = h + _attn_unpad_forward(layer.self_attn, layer.input_layernorm(h), c)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h


# =====================================================================================
# 3. Model Forward Replacement with TRL Metrics Preservation
# =====================================================================================
def unpad_qwen2_causal_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Any] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    head_weight = getattr(self.lm_head, "weight", None)
    fallback = (
        labels is None
        or input_ids is None
        or head_weight is None
        or head_weight.requires_grad
        or past_key_values is not None
        or inputs_embeds is not None
        or position_ids is not None
        or not input_ids.is_cuda
    )

    if fallback:
        return self._orig_soup_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    from transformers.modeling_outputs import CausalLMOutputWithPast

    base = self.model
    device = input_ids.device
    dtype = _COMPUTE_DTYPE

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    inv_freq = getattr(self, "_soup_inv_freq", None)
    if inv_freq is None:
        cfg = self.config
        hd = cfg.hidden_size // cfg.num_attention_heads
        theta = getattr(cfg, "rope_theta", None)
        if theta is None:
            theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 1_000_000.0)
        inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=device) / hd))
        self._soup_inv_freq = inv_freq

    c = _build_batch_ctx(attention_mask, self.config, device, dtype, inv_freq)

    # 1. Embeddings on real tokens only: [T, H]
    h = base.embed_tokens(input_ids.reshape(-1)[c.flat])

    # 2. 28 Decoder layers on unpadded tokens
    use_ckpt = getattr(base, "gradient_checkpointing", False) and self.training
    for layer in base.layers:
        if use_ckpt:
            h = checkpoint(_layer_unpad_forward, layer, h, c, use_reentrant=False)
        else:
            h = _layer_unpad_forward(layer, h, c)

    h = base.norm(h)  # fused RMSNorm + cast -> fp16 [T, H]

    # 3. Next-token targets aligned for real tokens
    am = attention_mask.bool()
    nxt_lab = torch.cat([labels[:, 1:], labels.new_full((labels.shape[0], 1), IGNORE_INDEX)], dim=1)
    nxt_ok = torch.cat([am[:, 1:], torch.zeros_like(am[:, :1])], dim=1)
    nxt_lab = torch.where(nxt_ok, nxt_lab, torch.full_like(nxt_lab, IGNORE_INDEX))
    lab_unpad = nxt_lab.reshape(-1)[c.flat]

    # 4. Cached fp16 lm_head weight
    if head_weight.dtype != dtype:
        cache = getattr(self, "_soup_w16", None)
        if cache is None or cache[0] != head_weight._version:
            cache = (head_weight._version, head_weight.detach().to(dtype))
            self._soup_w16 = cache
        W16 = cache[1]
    else:
        W16 = head_weight.detach()

    num_items = kwargs.get("num_items_in_batch")
    if num_items is None:
        denom = (lab_unpad != IGNORE_INDEX).sum().to(torch.float32)
    elif torch.is_tensor(num_items):
        denom = num_items.to(device=device, dtype=torch.float32)
    else:
        denom = torch.as_tensor(num_items, device=device, dtype=torch.float32)

    chunk_size = max(64, ((96 << 20) // (W16.shape[0] * 2)) // 64 * 64)
    loss, unpad_losses, unpad_entropies, unpad_corrects = UnpadFusedLinearCE.apply(
        h.to(dtype), W16, lab_unpad, denom, chunk_size, IGNORE_INDEX
    )

    # 5. Reconstruct 2D metrics for Soup / TRL / rewind.jsonl
    B, S0 = c.B, c.S0
    loss_full = torch.zeros(B * S0, dtype=torch.float32, device=device)
    loss_full.index_copy_(0, c.flat, unpad_losses)
    loss_full = loss_full.view(B, S0)

    entropy_full = torch.zeros(B * S0, dtype=torch.float32, device=device)
    entropy_full.index_copy_(0, c.flat, unpad_entropies)
    entropy_full = entropy_full.view(B, S0)

    correct_full = torch.zeros(B * S0, dtype=torch.int32, device=device)
    correct_full.index_copy_(0, c.flat, unpad_corrects)
    correct_full = correct_full.view(B, S0)

    try:
        from soup_cli.utils.stream_loss_chunk import ChunkedStep
        labelled = (nxt_lab != IGNORE_INDEX).sum()
        correct_total = correct_full.sum(dtype=torch.long)
        self._last_triton_chunked_step = ChunkedStep(
            loss=loss.detach(),
            token_entropy=entropy_full.detach(),
            token_loss=loss_full.detach(),
            correct=correct_total.detach(),
            labelled=labelled.detach(),
        )
    except Exception:
        pass

    return CausalLMOutputWithPast(
        loss=loss,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    )


def patch_unpadded_engine() -> None:
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception:
        return

    orig_forward = modeling_qwen2.Qwen2ForCausalLM.forward
    if not getattr(orig_forward, "_soup_unpad_patched", False):
        modeling_qwen2.Qwen2ForCausalLM._orig_soup_forward = orig_forward
        modeling_qwen2.Qwen2ForCausalLM.forward = unpad_qwen2_causal_forward
        modeling_qwen2.Qwen2ForCausalLM.forward._soup_unpad_patched = True
