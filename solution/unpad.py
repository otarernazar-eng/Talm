"""Padding removal inside a batch (allowed by the rules; mixing rows across batches is not).

TRL right-pads the 4 rows of a batch to the longest of them; on this data about 45% of the
positions are padding, and every position-wise op (projections, LoRA, norms, MLP, the output
head) does that work for nothing. The causal-LM patch in `patches.py` gathers the real tokens
into one flat sequence [1, N] with per-row position ids, runs the decoder on it, and scatters
the per-token metrics back to [B, T]. Attention must not cross rows: while the flat forward
runs, the registered "sdpa" attention interface is replaced by `rows_sdpa`, which runs causal
attention per row with the memory-efficient kernel SDPA itself uses on a T4. By default all
rows go in one launch per KV head (the kernel's variable-length mode, row offsets in
`cu_seqlens`; `VarlenRowsAttention`), else one launch per row (`RowsAttention`); both give the
output and the key/value gradients of `scaled_dot_product_attention(..., is_causal=True)` per
row on that backend bit for bit. The query gradient, which that backward accumulates over key
blocks in an order that depends on the launch's layout (one row, the K/V heads as the batch, all
rows in one launch), may differ in the last bit of a few elements: one ulp of the 16-bit dtype
(near zero, the fp32 rounding of the summed terms: 2^-18 of the largest dQ at most)
(solution/test_fuzz.py, 200 random shapes: dQ off in about 10% of them, mostly bf16; bit for
bit on the task's fp16 shapes in test_fusions.py).
Nothing TRL or Soup reads changes: `inputs` keep the padded `attention_mask` and `labels`, so
`num_tokens`, entropy weighting, accuracy and the rewind rows are computed exactly as before.

Graceful degradation (state.py): an unexpected error in the variable-length attention turns
"varlen_attention" off and the call takes the per-row launches, an error there turns
"rows_attention" off and the call takes PyTorch's own `scaled_dot_product_attention` per row;
an error in either backward recomputes the rows' attention with PyTorch's SDPA and takes its
gradient (`_reference_grads`). The padding removal itself is guarded in patches.py ("unpad").
"""

import contextlib
import os

import torch
import torch.nn.functional as F

from .engaged import FELL_BACK, RAN
from .state import DISABLED, fallback

_rows = []  # the [(start, end)] row bounds of the flat sequence while a flat forward runs
_cu = []    # the matching (cu_seqlens int32 on the GPU, longest row, total tokens) or None
_ROWS_ATTENTION = os.environ.get("SOLUTION_DISABLE_ROWS_ATTN") != "1"
_VARLEN = os.environ.get("SOLUTION_DISABLE_VARLEN") != "1"
# VarlenRowsAttention.backward without its per-KV-head gradient copies and concatenations
_NOCAT = os.environ.get("SOLUTION_DISABLE_ATTN_NOCAT") != "1"


@contextlib.contextmanager
def rows_context(bounds, device=None):
    """`device` given: the row offsets are also put on the GPU once here (one small copy before
    the decoder runs) for the variable-length attention."""
    cu = None
    if device is not None and torch.device(device).type == "cuda":
        starts = [s for s, e in bounds if e > s]
        ends = [e for s, e in bounds if e > s]
        if starts and starts[0] == 0 and starts[1:] == ends[:-1]:
            cu = (torch.tensor([0] + ends, dtype=torch.int32, device=device),
                  max(e - s for s, e in zip(starts, ends)), ends[-1])
    _rows.append(bounds)
    _cu.append(cu)
    try:
        yield
    finally:
        _rows.pop()
        _cu.pop()


def unpad_batch(attention_mask, targets=None):
    """From a right-padded [B, T] mask: flat indices of the real tokens, their positions inside
    their rows, the row bounds in the flat sequence, and the lengths. One host sync.
    With `targets` (the [B * T] shifted labels) the same host copy also brings the number of
    targets >= 0 among the gathered positions (the first `length` of each row) and among all
    positions, as `counts` = [gathered, all], for the head (fused_head `n_labelled`)."""
    batch, seq = attention_mask.shape
    device = attention_mask.device
    lengths_gpu = attention_mask.to(torch.int64).sum(1)
    if targets is None:
        lengths = lengths_gpu.tolist()
    else:
        lab = targets.reshape(batch, seq) >= 0
        prefix = torch.arange(seq, device=device)[None] < lengths_gpu[:, None]
        counts = torch.stack(((lab & prefix).sum(), lab.sum()))
        values = torch.cat((lengths_gpu, counts)).tolist()
        lengths, counts = values[:batch], values[batch:]
    idx, positions, bounds, start = [], [], [], 0
    for b, length in enumerate(lengths):
        ar = torch.arange(length, device=device)
        idx.append(ar + b * seq)
        positions.append(ar)
        bounds.append((start, start + length))
        start += length
    if targets is None:
        return torch.cat(idx), torch.cat(positions)[None], bounds, lengths
    return torch.cat(idx), torch.cat(positions)[None], bounds, lengths, counts


def is_right_padded(attention_mask, lengths):
    seq = attention_mask.shape[1]
    prefix = torch.arange(seq, device=attention_mask.device)[None] < \
        torch.tensor(lengths, device=attention_mask.device)[:, None]
    return bool((attention_mask.bool() == prefix).all().item())


def _repeat_kv(x, n_rep):
    batch, kv_heads, seq, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(batch, kv_heads, n_rep, seq, head_dim).reshape(
        batch, kv_heads * n_rep, seq, head_dim)


def _grouped(query, key, value):
    """[1, Hq, N, D] query and [1, Hkv, N, D] key/value as [Hkv, Hq/Hkv, N, D]: the query heads
    of one KV head form one batch entry, and K/V are broadcast over them with stride 0 (views,
    no copy). Head h uses KV head h // (Hq/Hkv), exactly as transformers' repeat_kv."""
    _, hq, n, d = query.shape
    hkv = key.shape[1]
    rep = hq // hkv
    return (query.reshape(hkv, rep, n, d), key.reshape(hkv, 1, n, d).expand(hkv, rep, n, d),
            value.reshape(hkv, 1, n, d).expand(hkv, rep, n, d))


def _reference_grads(query, key, value, bounds, scale, grad_out):
    """Gradients of q [1, Hq, N, D], k and v [1, Hkv, N, D] for the [1, N, Hq, D] output
    gradient of the rows' causal attention, by autograd through PyTorch's own SDPA per row with
    the K/V heads repeated as transformers' repeat_kv does (for a backward launch that failed:
    the forward is recomputed here)."""
    with torch.enable_grad():
        q, k, v = (t.detach().requires_grad_(True) for t in (query, key, value))
        rep = q.shape[1] // k.shape[1]
        kr, vr = _repeat_kv(k, rep), _repeat_kv(v, rep)
        outs = [F.scaled_dot_product_attention(q[:, :, s:e], kr[:, :, s:e], vr[:, :, s:e],
                                               is_causal=True, scale=scale)
                for s, e in bounds if e > s]
        out = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)).transpose(1, 2)
        return torch.autograd.grad(out, (q, k, v), grad_out.reshape(out.shape))


class RowsAttention(torch.autograd.Function):
    """Causal attention of each row of the flat sequence with PyTorch's memory-efficient SDPA
    kernel (the one F.scaled_dot_product_attention runs on a T4), without materializing the
    KV heads repeated for every query head: they were [1, Hq, N, D] copies of K and of V that the
    SDPA graph kept for backward, 6x the size of K/V for Qwen2.5-1.5B (about 6 MiB per layer per
    1 000 tokens). Saved here: q, the un-repeated k and v, the output and the per-row
    log-sum-exp. The backward writes each row's gradients once (the slice-of-input graph of the
    plain version materializes full-size zero gradients per row and sums them) and reduces the
    key/value gradients over the query heads that shared them, as repeat_kv's backward does."""

    @staticmethod
    def forward(ctx, query, key, value, bounds, scale):
        _, hq, n, d = query.shape
        q, k, v = _grouped(query, key, value)
        saved = []
        for s, e in bounds:
            out, lse, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention(
                q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], None, True, 0.0, True, scale=scale)
            # the backward kernel reads `out` in exactly the layout its forward wrote
            saved += [out, lse, seed, offset]
        rows = [saved[i].permute(2, 0, 1, 3) for i in range(0, len(saved), 4)]  # [len, Hkv, rep, D]
        attn = torch.cat(rows, 0) if len(rows) > 1 else rows[0].contiguous()
        ctx.bounds, ctx.scale = bounds, scale
        ctx.save_for_backward(query, key, value, *saved)
        return attn.view(1, n, hq, d)

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, *saved = ctx.saved_tensors
        if "rows_attention" not in DISABLED:
            try:
                return RowsAttention._backward(ctx, grad_out, query, key, value, saved)
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("rows_attention", exc):
                    raise
        FELL_BACK["rows_attn"] += 1
        return (*_reference_grads(query, key, value, ctx.bounds, ctx.scale, grad_out), None,
                None)

    @staticmethod
    def _backward(ctx, grad_out, query, key, value, saved):
        _, hq, n, d = query.shape
        hkv = key.shape[1]
        q, k, v = _grouped(query, key, value)
        grad = grad_out.reshape(n, hkv, hq // hkv, d)
        dqs, dks, dvs = [], [], []
        for i, (s, e) in enumerate(ctx.bounds):
            out, lse, seed, offset = saved[4 * i:4 * i + 4]
            dq, dk, dv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                grad[s:e].permute(1, 2, 0, 3), q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], None,
                out, lse, seed, offset, 0.0, [True, True, True, False], True, scale=ctx.scale)
            dqs.append(dq)
            dks.append(dk)
            dvs.append(dv)

        def rows(parts):  # [Hkv, rep, len_i, D] per row -> [Hkv, rep, N, D]
            return parts[0] if len(parts) == 1 else torch.cat(parts, 2)

        return (rows(dqs).reshape(1, hq, n, d), rows(dks).sum(1).reshape(1, hkv, n, d),
                rows(dvs).sum(1).reshape(1, hkv, n, d), None, None)


class VarlenRowsAttention(torch.autograd.Function):
    """The same attention with all rows in ONE launch per KV head: the memory-efficient kernel's
    variable-length mode (`aten._efficient_attention_forward/backward` with `cu_seqlens`, the
    row offsets), on the [1, N, H, D] layout the kernel works in. For KV head g the query heads
    g*rep .. g*rep+rep-1 are one call and K/V head g is broadcast over them with stride 0, as in
    `RowsAttention`: no repeated K/V, no per-row launches, no per-row slices or concatenation
    of the rows (the kernel walks the rows itself; a short row no longer runs as its own small
    grid). Measured on the T4 with the rows in one launch (K/V repeated, before the broadcast):
    attention kernels 20.9 instead of 28.6 ms per step. Saved: q, the un-repeated k and v, the
    outputs and log-sum-exps of the calls.

    The backward (SOLUTION_DISABLE_ATTN_NOCAT=1: as it was) copies the incoming [1, N, Hq, D]
    gradient once into KV-head-major order, so each call gets a contiguous slice (the kernel
    copies a strided slice by itself: one copy per KV head before), and reduces the key/value
    gradients over their query heads straight into their slice of the [1, N, Hkv, D] result
    (`torch.sum(..., out=slice)`) instead of concatenating per-head sums: per layer 1 copy
    instead of 2 and 1 concatenation instead of 3, the same values (bit-identical, tested).
    Measured on the T4 (same session, per-kernel profile): concatenations 112 -> 56 per step,
    1.34 -> 1.00 ms; the copies, together with RoPE's token-major gradients, 111 -> 28 per
    step, 2.39 -> 1.01 ms."""

    @staticmethod
    def forward(ctx, query, key, value, cu, max_len, scale):
        _, hq, n, d = query.shape
        hkv = key.shape[1]
        rep = hq // hkv
        qt, kt, vt = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        outs, saved = [], []
        for g in range(hkv):
            out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
                qt[:, :, g * rep:(g + 1) * rep], kt[:, :, g:g + 1].expand(1, n, rep, d),
                vt[:, :, g:g + 1].expand(1, n, rep, d), None, cu, cu, max_len, max_len, 0.0, 1,
                True, scale=scale)
            outs.append(out)
            saved += [out, lse, seed, offset]
        RAN["varlen"] += hkv
        ctx.max_len, ctx.scale = max_len, scale
        ctx.save_for_backward(query, key, value, cu, *saved)
        return outs[0] if hkv == 1 else torch.cat(outs, 2)  # [1, N, Hq, D]

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, cu, *saved = ctx.saved_tensors
        if "varlen_attention" not in DISABLED:
            try:
                return VarlenRowsAttention._backward(ctx, grad_out, query, key, value, cu, saved)
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("varlen_attention", exc):
                    raise
        FELL_BACK["varlen"] += 1
        ends = cu.tolist()  # the row offsets (one host sync, on this fallback only)
        bounds = list(zip(ends[:-1], ends[1:]))
        return (*_reference_grads(query, key, value, bounds, ctx.scale, grad_out), None, None,
                None)

    @staticmethod
    def _backward(ctx, grad_out, query, key, value, cu, saved):
        _, hq, n, d = query.shape
        hkv = key.shape[1]
        rep = hq // hkv
        qt, kt, vt = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        if _NOCAT:
            # The kernel copies a non-contiguous gradient by itself, here once per KV head (the
            # head slices of [1, N, Hq, D]): one copy into KV-head-major order instead, after
            # which every group's slice is contiguous. The key/value gradients are reduced over
            # their query heads straight into their slice of the result: no concatenation.
            grads = grad_out.reshape(1, n, hkv, rep, d).permute(2, 0, 1, 3, 4).contiguous()
            dk_all = torch.empty(1, n, hkv, d, dtype=key.dtype, device=key.device)
            dv_all = torch.empty(1, n, hkv, d, dtype=value.dtype, device=value.device)
        elif grad_out.stride(-1) != 1:
            grad_out = grad_out.contiguous()
        dqs, dks, dvs = [], [], []
        for g in range(hkv):
            out, lse, seed, offset = saved[4 * g:4 * g + 4]
            dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
                grads[g] if _NOCAT else grad_out[:, :, g * rep:(g + 1) * rep],
                qt[:, :, g * rep:(g + 1) * rep],
                kt[:, :, g:g + 1].expand(1, n, rep, d), vt[:, :, g:g + 1].expand(1, n, rep, d),
                None, out, cu, cu, ctx.max_len, ctx.max_len, lse, 0.0, seed, offset, 1, False,
                scale=ctx.scale)
            dqs.append(dq)
            if _NOCAT:  # over the query heads that shared KV head g, into its slice
                torch.sum(dk, 2, keepdim=True, out=dk_all[:, :, g:g + 1])
                torch.sum(dv, 2, keepdim=True, out=dv_all[:, :, g:g + 1])
            else:
                dks.append(dk.sum(2, keepdim=True))  # over the query heads that shared head g
                dvs.append(dv.sum(2, keepdim=True))

        def heads(parts):  # [1, N, h_i, D] per KV head -> [1, h, N, D]
            return (parts[0] if len(parts) == 1 else torch.cat(parts, 2)).transpose(1, 2)

        RAN["varlen"] += hkv
        if _NOCAT:
            RAN["nocat"] += 1
            return heads(dqs), dk_all.transpose(1, 2), dv_all.transpose(1, 2), None, None, None
        return heads(dqs), heads(dks), heads(dvs), None, None, None


def _rows_attention_applies(query, key, value, dropout):
    return (_ROWS_ATTENTION and dropout == 0.0 and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)
            and key.dtype == query.dtype and value.dtype == query.dtype
            and query.dim() == 4 and key.dim() == 4 and value.dim() == 4
            and query.shape[0] == key.shape[0] == value.shape[0] == 1
            and key.shape == value.shape and query.shape[1] % key.shape[1] == 0
            and query.shape[2] == key.shape[2] and query.shape[3] == key.shape[3]
            and query.shape[3] % 8 == 0 and query.stride(-1) == 1 and key.stride(-1) == 1
            and value.stride(-1) == 1)


def _varlen_applies(query, key, value, dropout, cu):
    """`RowsAttention`'s conditions, the offsets on the GPU for this sequence, a head size the
    kernel's fast path takes, and a GPU where SDPA itself would take the memory-efficient kernel
    (no flash attention below compute capability 8, so on a T4 this path is always taken;
    SOLUTION_VARLEN=1 forces it on newer GPUs too, which our tests use on an RTX 4080)."""
    return (_VARLEN and cu is not None and cu[2] == query.shape[2] and query.shape[3] <= 128
            and _rows_attention_applies(query, key, value, dropout)
            and torch.backends.cuda.mem_efficient_sdp_enabled()
            # SOLUTION_VARLEN, a test knob, is read at each call: tests set it with setenv
            and (os.environ.get("SOLUTION_VARLEN") == "1"
                 or torch.cuda.get_device_capability(query.device)[0] < 8))


def _autocast_half(query):
    """The autocast dtype when `query` is an fp32 CUDA tensor under fp16/bf16 autocast (the
    dtype autocast would cast SDPA's inputs to), else None."""
    if query.dtype != torch.float32 or not query.is_cuda:
        return None
    try:
        if not torch.is_autocast_enabled("cuda"):
            return None
        dtype = torch.get_autocast_dtype("cuda")
    except TypeError:  # older signature
        if not torch.is_autocast_enabled():
            return None
        dtype = torch.get_autocast_gpu_dtype()
    return dtype if dtype in (torch.float16, torch.bfloat16) else None


def patch_sdpa_rows():
    """Wrap transformers' registered "sdpa" attention: per-row causal attention when a flat
    forward is running, the original interface otherwise."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(original, "_solution_rows", False):
        return

    def rows_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                  sliding_window=None, **kwargs):
        if not _rows or query.shape[0] != 1 or sliding_window is not None:
            if _rows and _ROWS_ATTENTION:  # inside a flat forward: engaged.py's fallback count
                FELL_BACK["varlen" if _VARLEN else "rows_attn"] += 1
            return original(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, sliding_window=sliding_window, **kwargs)
        scale = scaling if scaling is not None else query.shape[-1] ** -0.5
        cu = _cu[-1] if _cu else None
        half = _autocast_half(query)
        if half is not None:
            # fp32 q/k/v under fp16/bf16 autocast (transformers' RoPE with fp32 cos/sin, i.e.
            # SOLUTION_DISABLE_ROPE=1): cast as autocast casts SDPA's inputs, so the per-row
            # paths below take them instead of the plain fallback (the same values: every path
            # computes on the cast inputs, as the fallback's autocast SDPA does)
            query, key, value = query.to(half), key.to(half), value.to(half)
        if "varlen_attention" not in DISABLED and _varlen_applies(query, key, value, dropout, cu):
            try:
                return VarlenRowsAttention.apply(query, key, value, cu[0], cu[1], scale), None
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("varlen_attention", exc):
                    raise
        if _VARLEN and _ROWS_ATTENTION:
            FELL_BACK["varlen"] += 1
        if "rows_attention" not in DISABLED and _rows_attention_applies(query, key, value,
                                                                         dropout):
            try:
                bounds = [(s, e) for s, e in _rows[-1] if e > s]
                out = RowsAttention.apply(query, key, value, bounds, scale)
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("rows_attention", exc):
                    raise
            else:
                RAN["rows_attn"] += 1
                return out, None
        if _ROWS_ATTENTION and not _VARLEN:
            FELL_BACK["rows_attn"] += 1
        n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
        key = _repeat_kv(key, n_rep)
        value = _repeat_kv(value, n_rep)
        outs = [F.scaled_dot_product_attention(query[:, :, s:e], key[:, :, s:e],
                                               value[:, :, s:e], dropout_p=dropout,
                                               is_causal=True, scale=scaling)
                for s, e in _rows[-1] if e > s]
        attn = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        return attn.transpose(1, 2).contiguous(), None

    rows_sdpa._solution_rows = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = rows_sdpa
