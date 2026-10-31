"""Fused output head + causal-LM loss without a [B, T, V] logits tensor.

The base run materializes fp16 logits [4, 512, 151936], upcasts them to fp32 for the loss and
keeps three fp32 copies of that shape alive through backward (about 4 GiB on this task). Here
the head never holds more than one slice of the logits: one Triton kernel per row computes
everything that is read from the logits (log-sum-exp, cross-entropy of the target, Shannon
entropy, argmax) and overwrites the slice in place with softmax weights, and cuBLAS turns those
into the gradient of the hidden states. Only the [N, H] fp16 hidden gradient survives the
forward; no graph over the logits exists.

Default path (`FusedHeadLossVocab`): the head walks the VOCABULARY in slices of VOCAB_CHUNK
columns, all tokens at once, so it never needs an fp16 copy of the whole frozen fp32 weight
(445 MiB, alive on top of every saved activation at the peak of the step): only one 24 MiB
slice of it (VOCAB_CHUNK = 8192 columns) is cast at a time. With the weight stored in fp16
(`fp16_embed.py`, the default on the T4) the slices are views of it: no cast and no buffer, the
same fp16 GEMM operands. Per slice: cuBLAS writes the logits [N, VOCAB_CHUNK]; the kernel
continues each row's online log-sum-exp / entropy / argmax from the previous slices, rescales the
row's fp32 gradient accumulator when its running max moved, and rewrites the row in place as
exp(z - running max); cuBLAS adds that times the weight slice to the accumulator (fp32 output),
as in the online-softmax accumulation of FlashAttention. A last launch turns the accumulator into
dH = acc / sum - W16[target] (0 on unlabelled rows). With VOCAB_CHUNK a multiple of BLOCK_V the
row statistics (and so loss, entropy, argmax) are computed in exactly the same order as by the
token-chunked kernel below.

The accumulator covers the labelled rows only (SOLUTION_DISABLE_HEAD_COMPACT=1: all rows, as
before). The hidden gradient of an unlabelled (prompt) row is 0, and 40-55% of the rows of this
task are prompt. With the gradient on, the rows are put labelled-first (`labelled_first`: the
order from a cumsum on the GPU, the number of labelled rows from the host sync the padding
removal makes anyway, `patches.py`; one gather of the fp16 hidden states and the targets), and
each slice's accumulator GEMM takes only the n_labelled leading rows of the logits slice. Every
slice's logits GEMM and kernel still run over all rows: loss, the entropy of every position and
the accuracy are bit-identical (the same rows through the same GEMM, in another order; tested),
and the per-row outputs and the gradient are put back in the input order at the end. The
accumulator GEMM has fewer rows, which cuBLAS may tile differently: the gradient is NOT bit for
bit the all-rows one; the tests bound it by one fp16 ulp of the largest element (atol =
max|dH| * 2^-10, rtol 1e-3), which holds on the RTX 4080 and on the T4. Counted in ulps of each
element, the difference depends on the kernel cuBLAS picks for the smaller GEMM: 1-3 ulp on at
most 0.2% of the task-shape elements on the RTX 4080 (kernel tests; none in test_parity.py), up
to 48 ulp on small elements on the T4 (test_parity.py), still inside that bound (checks 5-7
compare at 3% / 0.5%). Measured on the Kaggle T4 (per-kernel profiles of
adjacent runs, two sessions): accumulator GEMM 21.1 -> 14.8 and 18.8 -> 13.6 ms per step (it
does not halve: fewer rows tile less evenly over the 40 SMs), peak memory and eval loss
unchanged.

Token-chunked path (`FusedHeadLoss`, SOLUTION_DISABLE_HEAD_VCHUNK=1): `chunk` tokens at a time
against the full fp16 weight; the row kernel writes `p - onehot(target)` in place.

Ideas: chunked head + loss as in Soup's own `stream_loss_chunk`, the fused linear-cross-entropy of
Liger-Kernel (Hsu et al., 2024), Cut Cross-Entropy (Wijmans et al., 2024), online softmax
(Milakov & Gimelshein, 2018) and FlashAttention's rescaled accumulator (Dao et al., 2022). The
code is ours.

Formulas match the base exactly in semantics:
- loss  = sum_{t: y_t != -100} (lse_t - z_{t, y_t}) / num_items_in_batch   (transformers
  `ForCausalLMLoss`; `/ #labelled` when num_items_in_batch is None);
- entropy_t = lse_t - sum_v p_v z_v for EVERY position (TRL averages it with attention_mask);
- correct_t = [argmax_v z_{t,v} == y_t] for labelled positions (TRL's mean_token_accuracy);
- per-token loss for Soup's rewind log, 0 where y_t == -100.
"""

import os

import torch
import triton
import triton.language as tl

from .engaged import FELL_BACK, RAN

BLOCK_V = 4096                                           
NUM_WARPS = 8
DEFAULT_CHUNK = 256                                                                             
                                                                                            
                                        
VOCAB_CHUNK = int(os.environ.get("SOLUTION_HEAD_VCHUNK", "8192"))
BLOCK_H = 512                                                  
VOCAB_PATH = os.environ.get("SOLUTION_DISABLE_HEAD_VCHUNK") != "1"
                                                                              
COMPACT = os.environ.get("SOLUTION_DISABLE_HEAD_COMPACT") != "1"
                                                                                                
                                                                                                
                                                                                               
                                                                                              
                                                                                          
                                                                               
ROW_UNIT = 1 if os.environ.get("SOLUTION_DISABLE_HEAD_ROWS") == "1" else 8


@triton.jit(do_not_specialize=["n_cols", "col0", "stride_row", "n_hidden", "stride_acc",
                               "stride_w", "write_grad", "first", "last", "finalize"])
def fused_head_vocab_kernel(logits_ptr, targets_ptr, stats_ptr, best_idx_ptr, acc_ptr, w_ptr,
                            grad_ptr, loss_ptr, entropy_ptr, correct_ptr, n_cols, col0,
                            stride_row, n_hidden, stride_acc, stride_w, write_grad, first, last,
                            finalize, BLOCK: tl.constexpr, BLOCK_H: tl.constexpr,
                            UNIT: tl.constexpr):
    """One program per token row. finalize == 0: one vocabulary slice [col0, col0 + n_cols) of
    the row (statistics, then, for labelled rows when write_grad != 0, the accumulator rescale
    and exp(z - m) in place). finalize != 0: dH = acc / s - W16[target] (0 when unlabelled).
    n_cols, stride_row, n_hidden, stride_acc and stride_w come in units of UNIT elements: with
    UNIT = 8 every row and every bound is provably a multiple of 8, so the row loads and stores
    vectorize (16 bytes of fp16 per access instead of 2)."""
    n_cols = n_cols * UNIT
    stride_row = stride_row * UNIT
    n_hidden = n_hidden * UNIT
    stride_acc = stride_acc * UNIT
    stride_w = stride_w * UNIT
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(targets_ptr + row)
    labelled = target >= 0
    sp = stats_ptr + row * 5                                                                   
    acc_row = acc_ptr + row * stride_acc
    if finalize != 0:
        s = tl.load(sp + 1)
        w_row = w_ptr + tl.maximum(target, 0) * stride_w
        g_row = grad_ptr + row * n_hidden
        for h in range(0, n_hidden, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            hmask = offs < n_hidden
                                                                                                 
                                                 
            lmask = hmask & labelled
            a = tl.load(acc_row + offs, mask=lmask, other=0.0)
                                                                                      
            w16 = tl.load(w_row + offs, mask=lmask, other=0.0).to(tl.float16).to(tl.float32)
            g = tl.where(labelled, a / s - w16, 0.0)
            tl.store(g_row + offs, g.to(grad_ptr.dtype.element_ty), mask=hmask)
    else:
        base = logits_ptr + row * stride_row
        not_first = first == 0
        m = tl.load(sp + 0, mask=not_first, other=-float("inf"))
        s = tl.load(sp + 1, mask=not_first, other=0.0)
        t = tl.load(sp + 2, mask=not_first, other=0.0)
        best = tl.load(sp + 3, mask=not_first, other=-float("inf"))
        zt = tl.load(sp + 4, mask=not_first, other=0.0)
        best_idx = tl.load(best_idx_ptr + row, mask=not_first, other=0)
        m_old = m
        for start in range(0, n_cols, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n_cols
            z = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            block_max = tl.max(z, axis=0)
            m_new = tl.maximum(m, block_max)
            alpha = tl.exp(m - m_new)
            e = tl.exp(z - m_new)
            s = s * alpha + tl.sum(e, axis=0)
            t = t * alpha + tl.sum(e * tl.where(mask, z, 0.0), axis=0)
            m = m_new
            block_idx = tl.argmax(z, axis=0, tie_break_left=True)
            better = block_max > best
            best_idx = tl.where(better, col0 + start + block_idx, best_idx)
            best = tl.maximum(best, block_max)
        rel = target - col0
        here = labelled & (rel >= 0) & (rel < n_cols)
        zt = tl.where(here, tl.load(base + rel, mask=here, other=0.0).to(tl.float32), zt)
        tl.store(sp + 0, m)
        tl.store(sp + 1, s)
        tl.store(sp + 2, t)
        tl.store(sp + 3, best)
        tl.store(sp + 4, zt)
        tl.store(best_idx_ptr + row, best_idx)
        if last != 0:
            ls = tl.log(s)
            lse = m + ls
            tl.store(entropy_ptr + row, lse - t / s)
                                                                                                
                                                                                   
                                                                                          
            tl.store(loss_ptr + row, tl.where(labelled, ls + (m - zt), 0.0))
            hit = best_idx.to(tl.int64) == target
            tl.store(correct_ptr + row, (labelled & hit).to(tl.int32))
        if labelled & (write_grad != 0):
            if not_first & (m > m_old):                                                  
                scale = tl.exp(m_old - m)
                for h in range(0, n_hidden, BLOCK_H):
                    offs = h + tl.arange(0, BLOCK_H)
                    hmask = offs < n_hidden
                    a = tl.load(acc_row + offs, mask=hmask, other=0.0)
                    tl.store(acc_row + offs, a * scale, mask=hmask)
            for start in range(0, n_cols, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                mask = offs < n_cols
                z = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
                tl.store(base + offs, tl.exp(z - m).to(logits_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n_rows", "n_cols", "stride_row", "write_grad"])
def fused_head_rows_kernel(logits_ptr, targets_ptr, loss_ptr, entropy_ptr, correct_ptr,
                           n_rows, n_cols, stride_row, write_grad, BLOCK: tl.constexpr):
    """Token-chunked path. One program per row of the logits chunk. Pass 1 streams the row once
    for the online log-sum-exp, the entropy numerator and the first argmax; pass 2 (labelled
    rows only, when write_grad != 0) rewrites the row in place as softmax(z) - onehot(target)."""
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride_row
    target = tl.load(targets_ptr + row)
    labelled = target >= 0

    m = -float("inf")
    s = 0.0
    t = 0.0
    best = -float("inf")
    best_idx = 0
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        z = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(z, axis=0)
        m_new = tl.maximum(m, block_max)
        alpha = tl.exp(m - m_new)
        e = tl.exp(z - m_new)
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(e * tl.where(mask, z, 0.0), axis=0)
        m = m_new
        block_idx = tl.argmax(z, axis=0, tie_break_left=True)
        better = block_max > best
        best_idx = tl.where(better, start + block_idx, best_idx)
        best = tl.maximum(best, block_max)

    ls = tl.log(s)
    lse = m + ls
    tl.store(entropy_ptr + row, lse - t / s)
    z_target = tl.load(base + target, mask=labelled, other=0.0).to(tl.float32)
                                                                                                
                                                                    
    tl.store(loss_ptr + row, tl.where(labelled, ls + (m - z_target), 0.0))
    hit = best_idx.to(tl.int64) == target
    tl.store(correct_ptr + row, (labelled & hit).to(tl.int32))

    if labelled & (write_grad != 0):
        for start in range(0, n_cols, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n_cols
            z = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            grad = tl.exp(z - lse) - (offs == target).to(tl.float32)
            tl.store(base + offs, grad.to(logits_ptr.dtype.element_ty), mask=mask)


def launch_vocab(n_rows, logits, targets, stats, best_idx, acc, weight, grad, token_loss,
                 token_entropy, correct, n_cols, col0, stride_row, n_hidden, write_grad, first,
                 last, finalize):
    """The one launch site of the vocabulary-sliced kernel (training and the warm-up thread).
    Sizes and strides go in units of ROW_UNIT elements when they all are multiples of it (every
    launch of this task: slices of 8192 and 4480 columns, hidden size 1536, so still one
    specialization), else of 1. Returns the unit."""
    sizes = (n_cols, stride_row, n_hidden, acc.stride(0), weight.stride(0))
    unit = ROW_UNIT if all(size % ROW_UNIT == 0 for size in sizes) else 1
    n_cols, stride_row, n_hidden, stride_acc, stride_w = (size // unit for size in sizes)
    fused_head_vocab_kernel[(n_rows,)](
        logits, targets, stats, best_idx, acc, weight, grad, token_loss, token_entropy, correct,
        n_cols, col0, stride_row, n_hidden, stride_acc, stride_w, int(write_grad), int(first),
        int(last), int(finalize), BLOCK=BLOCK_V, BLOCK_H=BLOCK_H, UNIT=unit,
        num_warps=NUM_WARPS)
    return unit


def _count_rows8(unit):
    """engaged.py's `rows8` counter: a training launch of the vocabulary kernel in units of 8,
    or with element units although they are on (a size that is not a multiple of 8)."""
    if unit == 8:
        RAN["rows8"] += 1
    elif ROW_UNIT == 8:
        FELL_BACK["rows8"] += 1


def launch_rows(logits, targets, token_loss, token_entropy, correct, write_grad):
    """The one launch site of the kernel (training and the warm-up thread): logits [rows, V]
    with contiguous rows; the four per-row tensors are [rows] slices."""
    rows, vocab = logits.shape
    fused_head_rows_kernel[(rows,)](
        logits, targets, token_loss, token_entropy, correct, rows, vocab, logits.stride(0),
        int(write_grad), BLOCK=BLOCK_V, num_warps=NUM_WARPS)


def _denominator(num_items, labelled_count):
    if num_items is None:
        return labelled_count.clamp(min=1).to(torch.float32)
    if torch.is_tensor(num_items):
        return num_items.to(torch.float32)
    return torch.tensor(float(num_items), device=labelled_count.device)


def _head_backward(ctx, grad_loss):
    (grad_hidden,) = ctx.saved_tensors
    scale = grad_loss.to(torch.float32) / ctx.denom
                                                                                              
                                                                                         
                                                                                              
    scale = torch.where(ctx.denom != 0, scale, 0.0)
    return (grad_hidden.to(torch.float32) * scale).to(ctx.out_dtype)


def labelled_first(labelled, n_labelled):
    """The order that puts the labelled rows first (each group in its original order), without
    a host sync: (dest, src) with row r of the input at position dest[r] of the reordered rows
    and src the inverse (reordered row i is input row src[i])."""
    n = labelled.shape[0]
    pos = torch.arange(n, device=labelled.device)
    seen = torch.cumsum(labelled, 0)                                               
    dest = torch.where(labelled, seen - 1, pos - seen + n_labelled)
    src = torch.empty_like(dest).index_copy_(0, dest, pos)
    return dest, src


class FusedHeadLossVocab(torch.autograd.Function):
    """loss, per-token loss, per-token entropy, per-token correctness from hidden states and the
    (frozen) output weight, walking the vocabulary in slices. Saves only the fp16 hidden
    gradient; holds one fp16 weight slice and one [N, vocab_chunk] logits slice at a time.

    The hidden gradient of an unlabelled row is 0: with `n_labelled` (the number of rows with
    target >= 0, a host int; counted here with one host sync when it is not given) the rows are
    reordered labelled-first (`labelled_first`), every slice's logits GEMM and kernel still run
    over all rows (the statistics of every row, unchanged), but the accumulator GEMM and the
    accumulator itself cover only the n_labelled leading rows. The per-row outputs and the
    gradient are put back in the input order at the end (SOLUTION_DISABLE_HEAD_COMPACT=1: the
    accumulator over all rows, as before)."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, num_items, vocab_chunk, need_grad, n_labelled):
        n_tokens, hidden_size = hidden.shape
        vocab = weight.shape[0]
        device = hidden.device
        h16 = hidden.detach().to(torch.float16).contiguous()
        weight = weight.detach()
        targets = targets.contiguous()
        vchunk = max(1, min(int(vocab_chunk), vocab))
        labelled = targets >= 0

                                                                                   
        n_grad, dest, rows_targets = n_tokens, None, targets
        if need_grad and COMPACT and n_tokens > 0:
            if n_labelled is None:
                n_labelled = int(labelled.sum())
            n_grad = int(n_labelled)
            if 0 < n_grad < n_tokens:
                dest, src = labelled_first(labelled, n_grad)
                h16 = h16.index_select(0, src)
                rows_targets = targets.index_select(0, src)
                RAN["compact"] += 1

        token_loss = torch.empty(n_tokens, dtype=torch.float32, device=device)
        token_entropy = torch.empty_like(token_loss)
        correct = torch.empty(n_tokens, dtype=torch.int32, device=device)
        stats = torch.empty(n_tokens, 5, dtype=torch.float32, device=device)
        best_idx = torch.empty(n_tokens, dtype=torch.int32, device=device)
        grad_hidden = torch.empty_like(h16) if need_grad else None
                                                                                   
        acc = torch.empty(n_grad if need_grad and n_grad > 0 else 1, hidden_size,
                          dtype=torch.float32, device=device)
        grad_arg = grad_hidden if need_grad else h16
        if n_tokens > 0:
            logits_buf = torch.empty(n_tokens * vchunk, dtype=torch.float16, device=device)
                                                                                           
                                                                                            
                                                                                             
            sliced = weight.dtype == torch.float16 and weight.is_contiguous()
            w16_buf = None if sliced else torch.empty(vchunk * hidden_size, dtype=torch.float16,
                                                      device=device)
            for start in range(0, vocab, vchunk):
                cols = min(vchunk, vocab - start)
                if sliced:
                    w16 = weight[start:start + cols]
                else:
                    w16 = w16_buf[:cols * hidden_size].view(cols, hidden_size)
                    w16.copy_(weight[start:start + cols])
                logits = logits_buf[:n_tokens * cols].view(n_tokens, cols)
                torch.mm(h16, w16.t(), out=logits)
                _count_rows8(launch_vocab(
                    n_tokens, logits, rows_targets, stats, best_idx, acc, weight, grad_arg,
                    token_loss, token_entropy, correct, cols, start, logits.stride(0),
                    hidden_size, need_grad, start == 0, start + cols >= vocab, False))
                if need_grad and n_grad > 0:
                    torch.addmm(acc, logits[:n_grad], w16, beta=0 if start == 0 else 1,
                                out_dtype=torch.float32, out=acc)
            del logits_buf, w16_buf, logits, w16
            if need_grad:
                _count_rows8(launch_vocab(
                    n_tokens, h16, rows_targets, stats, best_idx, acc, weight, grad_hidden,
                    token_loss, token_entropy, correct, 0, 0, 0, hidden_size, True, False, False,
                    True))
        del acc, stats, best_idx, h16
        if dest is not None:                           
            token_loss = token_loss.index_select(0, dest)
            token_entropy = token_entropy.index_select(0, dest)
            correct = correct.index_select(0, dest)
            grad_hidden = grad_hidden.index_select(0, dest)

        denom = _denominator(num_items, labelled.sum())
        loss = token_loss.sum() / denom
        correct = correct.to(torch.bool)
        ctx.denom = denom
        ctx.out_dtype = hidden.dtype
        if need_grad:
            ctx.save_for_backward(grad_hidden)
        ctx.mark_non_differentiable(token_loss, token_entropy, correct)
        return loss, token_loss, token_entropy, correct

    @staticmethod
    def backward(ctx, grad_loss, _grad_token_loss, _grad_entropy, _grad_correct):
        return _head_backward(ctx, grad_loss), None, None, None, None, None, None


class FusedHeadLoss(torch.autograd.Function):
    """Token-chunked path: the same outputs from `chunk` tokens at a time against a full fp16
    copy of the weight (no copy when the weight is stored in fp16)."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, num_items, chunk, need_grad):
        n_tokens, hidden_size = hidden.shape
        vocab = weight.shape[0]
        h16 = hidden.detach().to(torch.float16).contiguous()
        w16 = weight.detach().to(torch.float16)                                               
        targets = targets.contiguous()

        token_loss = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
        token_entropy = torch.empty_like(token_loss)
        correct = torch.empty(n_tokens, dtype=torch.int32, device=hidden.device)
        grad_hidden = torch.empty_like(h16) if need_grad else None
        buf = torch.empty(min(chunk, n_tokens), vocab, dtype=torch.float16, device=hidden.device)
        labelled = targets >= 0

        for start in range(0, n_tokens, chunk):
            end = min(start + chunk, n_tokens)
            rows = end - start
            logits = buf[:rows]
            torch.mm(h16[start:end], w16.t(), out=logits)
            launch_rows(logits, targets[start:end], token_loss[start:end],
                        token_entropy[start:end], correct[start:end], need_grad)
            if need_grad:
                torch.mm(logits, w16, out=grad_hidden[start:end])
                                                                                            
                                                                                   
                grad_hidden[start:end].masked_fill_(~labelled[start:end, None], 0)

        denom = _denominator(num_items, labelled.sum())
        loss = token_loss.sum() / denom
        correct = correct.to(torch.bool)
        ctx.denom = denom
        ctx.out_dtype = hidden.dtype
        if need_grad:
            ctx.save_for_backward(grad_hidden)
        ctx.mark_non_differentiable(token_loss, token_entropy, correct)
        return loss, token_loss, token_entropy, correct

    @staticmethod
    def backward(ctx, grad_loss, _grad_token_loss, _grad_entropy, _grad_correct):
        return _head_backward(ctx, grad_loss), None, None, None, None, None


def fused_head_loss(hidden, weight, targets, num_items=None, vocab_chunk=None, *,
                    token_chunk=None, n_labelled=None):
    """hidden [N, H] (fp16/fp32, may require grad), weight [V, H] frozen, targets [N] int64
    already shifted (-100 = ignore). Returns (loss, token_loss[N], token_entropy[N], correct[N]).
    Vocabulary-sliced by default; `token_chunk=` (or SOLUTION_DISABLE_HEAD_VCHUNK=1) selects the
    token-chunked path. `n_labelled`: the host count of targets >= 0, when the caller has it
    (saves the vocabulary-sliced path a host sync); it must be exact."""
    need_grad = torch.is_grad_enabled() and hidden.requires_grad
    if token_chunk is None and VOCAB_PATH and weight.dim() == 2 and weight.stride(1) == 1:
        RAN["vchunk"] += 1
        return FusedHeadLossVocab.apply(hidden, weight, targets, num_items,
                                        vocab_chunk or VOCAB_CHUNK, need_grad, n_labelled)
    if VOCAB_PATH:
        FELL_BACK["vchunk"] += 1                                                             
    return FusedHeadLoss.apply(hidden, weight, targets, num_items, token_chunk or DEFAULT_CHUNK,
                               need_grad)
