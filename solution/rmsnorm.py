"""Fused RMSNorm for Qwen2, forward and backward in one Triton kernel, emitting the autocast dtype.

The base `Qwen2RMSNorm.forward` is a chain of fp32 ops (`pow`, `mean`, `rsqrt`, two `mul`, a
cast) over the fp32 residual stream, each one a separate kernel reading and writing the
[B, T, 1536] tensor, and every consumer of its fp32 output (the 4-bit projections, LoRA, the
output head) casts it to fp16 again, and bitsandbytes casts the projection outputs back to
fp32 because the input was fp32. One kernel here does the whole norm per row and writes fp16
directly when autocast is on: the same values (one rounding to fp16, as the autocast cast did),
no cast kernels on either side of the projections, and the activations downstream (gate/up,
q/k/v, LoRA inputs) are stored in fp16 instead of fp32.

The norm weight is frozen in this task (prepare_model_for_kbit_training), so the backward
computes only the input gradient, which needs no reduction over rows:
    dx = rstd * (g - xhat * mean(g * xhat)),  g = dy * w,  xhat = x * rstd.
Saved for backward: `xhat` in the output dtype (fp16: half the size of the fp32 residual the
base keeps for its `pow` backward) and rstd [N]. Nothing else of ours holds the fp32 residual,
so each of the 57 norms costs 2 instead of 4 bytes per element of saved activation.

Forward and backward are one kernel with a runtime `backward` flag (like the RoPE kernel): every
run starts with an empty Triton cache, so each kernel is compiled (and gets a gcc-built C
launcher) inside the measured time, and this way the norm is compiled once instead of twice.

The residual adds of the decoder (`AddRMSNorm`, `patch_qwen2_add_norm`): every
`residual + branch` of a Qwen2 decoder layer is read only by the norm that follows it (the
post-attention norm, the next layer's input norm, the final norm) and by the next residual add.
The same kernel, with its runtime `add` flag, takes the add: the forward reads x + branch and
writes the fp32 sum (the residual stream) along with the norm; the backward adds the sum's
gradient from the next residual add to dx and writes the branch's gradient in the branch's
dtype. That removes, per add, the separate fp32 add of the forward, the fp32 gradient
accumulation of the backward and the cast of the gradient to the 16-bit branch, with the same
values (the same fp32 add, the same norm arithmetic; bit-identical, tested).

Graceful degradation (state.py): an unexpected error in a forward launch turns the component
off ("rmsnorm", "addnorm") and the call runs through transformers' norm (or the plain add and
the norm); an error in a backward launch computes the same formula with PyTorch ops from the
saved xhat and rstd (`_reference_dx`).

Under gradient checkpointing (off in this task: Soup sets it off; TRL's default is on) a decoder
layer's forward runs again inside the backward. The add inside a layer (the attention's add into
the post-attention norm) stays fused: the replay recomputes it identically. The hand-off across a
layer boundary (the MLP's add computed with the next layer's input norm, its output riding on the
layer output) is disabled for a checkpointed layer, on both sides: the replay of the receiving
layer would not find the handed-over output (a different set of saved tensors than its first
run), and with use_reentrant the first run of the handing layer is under no_grad.
"""

import torch
import triton
import triton.language as tl

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

BLOCK = 2048                                                               
NUM_WARPS = 4


@triton.jit(do_not_specialize=["n_cols", "stride_row", "backward", "add"])
def rmsnorm_kernel(a_ptr, w_ptr, h_ptr, xhat_ptr, rstd_ptr, p_ptr, q_ptr, n_cols, stride_row,
                   backward, add, eps, BLOCK: tl.constexpr):
    """One row per program; every [N, H] operand has row stride `stride_row`.
    backward == 0: reads x = a, writes y = h, xhat and rstd.
    backward != 0: reads dy = h, xhat and rstd, writes dx = a.
    add != 0 (the residual add that produced x): the forward reads x = a + p and also writes
    the sum to q; the backward adds q (the sum's gradient from its other reader) to dx and also
    writes the total to p, in p's dtype (the branch's gradient)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    at = row * stride_row + offs
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if backward == 0:
        x = tl.load(a_ptr + at, mask=mask, other=0.0).to(tl.float32)
        if add != 0:
            x += tl.load(p_ptr + at, mask=mask, other=0.0).to(tl.float32)
            tl.store(q_ptr + at, x.to(q_ptr.dtype.element_ty), mask=mask)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)
        xhat = x * rstd
        y = w * xhat
        tl.store(h_ptr + at, y.to(h_ptr.dtype.element_ty), mask=mask)
        tl.store(xhat_ptr + at, xhat.to(xhat_ptr.dtype.element_ty), mask=mask)
        tl.store(rstd_ptr + row, rstd)
    else:
        dy = tl.load(h_ptr + at, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + at, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        g = dy * w
        c = tl.sum(g * xhat, axis=0) / n_cols
        dx = rstd * (g - xhat * c)
        if add != 0:
            dx += tl.load(q_ptr + at, mask=mask, other=0.0).to(tl.float32)
            tl.store(p_ptr + at, dx.to(p_ptr.dtype.element_ty), mask=mask)
        tl.store(a_ptr + at, dx.to(a_ptr.dtype.element_ty), mask=mask)


def _rows(t):
    """[N, H] view of `t` with contiguous rows (row stride H), copying only if it is not."""
    t2 = t.reshape(-1, t.shape[-1])
    return t2 if t2.is_contiguous() else t2.contiguous()


def launch(a, weight, h, xhat, rstd, eps, backward, p=None, q=None):
    """The one launch site of the kernel (training and the warm-up thread): a, h, xhat (and
    p, q) are contiguous [N, H]; forward reads a and writes h, xhat, rstd; backward reads h,
    xhat, rstd and writes a. With p and q, the residual add (see the kernel): p has h's dtype
    (the 16-bit branch / its gradient) and q has a's (the fp32 sum / its gradient), and without
    them h and a stand in, so both cases are one kernel specialization. One program holds one
    whole row: rows wider than BLOCK raise (the kernel would normalize only their first BLOCK
    columns and leave the rest unwritten; the Qwen2 patch never sends them, `_norm_applies`;
    found by solution/test_fuzz.py)."""
    n_rows, n_cols = xhat.shape
    if n_cols > BLOCK:
        raise ValueError(f"RMSNorm kernel: rows of {n_cols} > BLOCK = {BLOCK} columns")
    add = p is not None
    rmsnorm_kernel[(n_rows,)](a, weight, h, xhat, rstd, p if add else h, q if add else a, n_cols,
                              n_cols, int(backward), int(add), float(eps), BLOCK=BLOCK,
                              num_warps=NUM_WARPS)


def _reference_dx(dy, xhat, weight, rstd):
    """The kernel's backward in PyTorch ops, fp32: dx = rstd * (g - xhat * mean(g * xhat)),
    g = dy * w (for a backward launch that failed)."""
    g = dy.float() * weight.float()
    xh = xhat.float()
    c = (g * xh).sum(-1, keepdim=True) / xh.shape[-1]
    return rstd[:, None] * (g - xh * c)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype):
        x2 = _rows(x)
        n_rows, n_cols = x2.shape
        y = torch.empty((n_rows, n_cols), dtype=out_dtype, device=x.device)
        xhat = torch.empty_like(y)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        launch(x2, weight, y, xhat, rstd, eps, backward=0)
        RAN["rmsnorm"] += 1
        ctx.save_for_backward(xhat, weight, rstd)
        ctx.x_dtype = x.dtype
        ctx.eps = eps
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        xhat, weight, rstd = ctx.saved_tensors
        dy2 = _rows(dy)                                                     
        if "rmsnorm" not in DISABLED:
            try:
                dx = torch.empty(xhat.shape, dtype=ctx.x_dtype, device=dy.device)
                launch(dx, weight, dy2, xhat, rstd, ctx.eps, backward=1)
                RAN["rmsnorm"] += 1
                return dx.view(dy.shape), None, None, None
            except Exception as exc:                               
                if not fallback("rmsnorm", exc):
                    raise
        FELL_BACK["rmsnorm"] += 1
        dx = _reference_dx(dy2, xhat, weight, rstd).to(ctx.x_dtype)
        return dx.view(dy.shape), None, None, None


class AddRMSNorm(torch.autograd.Function):
    """(s, y) = (x + branch, rmsnorm(x + branch)), the residual add folded into the norm that
    reads it: one launch forward, one backward (see the module docstring). The branch must have
    the norm output's dtype (the kernel's single specialization); the caller checks."""

    @staticmethod
    def forward(ctx, x, branch, weight, eps, out_dtype):
        x2, b2 = _rows(x), _rows(branch)
        n_rows, n_cols = x2.shape
        total = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
        y = torch.empty((n_rows, n_cols), dtype=out_dtype, device=x.device)
        xhat = torch.empty_like(y)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        launch(x2, weight, y, xhat, rstd, eps, backward=0, p=b2, q=total)
        RAN["addnorm"] += 1
        ctx.save_for_backward(xhat, weight, rstd)
        ctx.x_dtype, ctx.branch_dtype, ctx.eps = x.dtype, branch.dtype, eps
        return total.view(x.shape), y.view(x.shape)

    @staticmethod
    def backward(ctx, ds, dy):
        xhat, weight, rstd = ctx.saved_tensors
        if dy is None:                                                                   
            return ds, ds.to(ctx.branch_dtype), None, None, None
        shape = dy.shape
        if ds is None:
            ds = torch.zeros(shape, dtype=ctx.x_dtype, device=dy.device)
        if "addnorm" not in DISABLED:
            try:
                dx = torch.empty(xhat.shape, dtype=ctx.x_dtype, device=dy.device)
                db = torch.empty(xhat.shape, dtype=ctx.branch_dtype, device=dy.device)
                launch(dx, weight, _rows(dy), xhat, rstd, ctx.eps, backward=1, p=db, q=_rows(ds))
                RAN["addnorm"] += 1
                return dx.view(shape), db.view(shape), None, None, None
            except Exception as exc:                               
                if not fallback("addnorm", exc):
                    raise
        FELL_BACK["addnorm"] += 1
        total = _reference_dx(_rows(dy), xhat, weight, rstd) + _rows(ds).float()
        return (total.to(ctx.x_dtype).view(shape), total.to(ctx.branch_dtype).view(shape), None,
                None, None)


def _autocast_dtype():
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:                   
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return None


def rmsnorm(x, weight, eps, out_dtype=None):
    """RMSNorm of `x` with the frozen `weight`; output in `out_dtype` (default: x's dtype)."""
    return RMSNorm.apply(x, weight, eps, out_dtype or x.dtype)


def _norm_applies(norm, hidden_states):
    weight = norm.weight
    return not (not hidden_states.is_cuda or weight.requires_grad or hidden_states.dim() < 2
                or hidden_states.shape[-1] != weight.numel() or hidden_states.shape[-1] > BLOCK
                or hidden_states.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or (hidden_states.dtype != torch.float32 and weight.dtype != hidden_states.dtype))


def _out_dtype(hidden_states):
    if hidden_states.dtype == torch.float32:
        return _autocast_dtype() or torch.float32
    return hidden_states.dtype


def patch_qwen2_rmsnorm():
    """Send `Qwen2RMSNorm.forward` through the fused kernel on CUDA when the weight is frozen;
    under autocast the output is written in the autocast dtype directly."""
    from transformers.models.qwen2 import modeling_qwen2

    from . import _in_chain

    cls = modeling_qwen2.Qwen2RMSNorm
    original = cls.forward
    if _in_chain(original, "_solution_rmsnorm"):
        return

    def forward(self, hidden_states):
        normed = _take_normed(hidden_states, self)
        if normed is not None:
            return normed                                                              
                                                                                                
        if "rmsnorm" not in DISABLED and _norm_applies(self, hidden_states):
            try:
                return RMSNorm.apply(hidden_states, self.weight, self.variance_epsilon,
                                     _out_dtype(hidden_states))
            except Exception as exc:                               
                if not fallback("rmsnorm", exc):
                    raise
        FELL_BACK["rmsnorm"] += 1
        return original(self, hidden_states)

    forward._solution_rmsnorm = True
    forward._original = original
    cls.forward = forward
    ON.add("rmsnorm")


def _take_normed(hidden_states, norm):
    """The output of `norm` on `hidden_states` if the residual add that produced this tensor
    already computed it (AddRMSNorm at the end of the previous decoder layer), else None."""
    pending = getattr(hidden_states, "_solution_normed", None)
    if pending is None or pending[0] is not norm:
        return None
    del hidden_states._solution_normed
    return pending[1]


def link_next_norms(model):
    """Tell every decoder layer of a Qwen2Model which norm reads its output: the next layer's
    `input_layernorm`, and the final `norm` for the last layer. The layer then folds its MLP
    residual add into that norm's kernel (`patch_qwen2_add_norm`)."""
    if not LAYER_FORWARDS:                                                                  
        return
    layers = list(getattr(model, "layers", []))
    final = getattr(model, "norm", None)
    if not layers or getattr(model, "_solution_norms_linked", None) == len(layers):
        return
    for layer, nxt in zip(layers, layers[1:] + [None]):
        layer._solution_next_norm = nxt.input_layernorm if nxt is not None else final
    model._solution_norms_linked = len(layers)


def _link_norms_in_model_forward(model_cls):
    """`Qwen2Model.forward` calls `link_next_norms` on its model first (a no-op once linked).
    The fused head's causal-LM forward links them too; with it switched off
    (SOLUTION_DISABLE_FUSED_HEAD=1) nothing else would, and every MLP residual add of the
    decoder would take the plain add (engaged.py counted them as fallbacks)."""
    import functools

    forward = model_cls.forward
    if getattr(forward, "_solution_link_norms", False):
        return

    @functools.wraps(forward)
    def forward_linked(self, *args, **kwargs):
        link_next_norms(self)
        return forward(self, *args, **kwargs)

    forward_linked._solution_link_norms = True
    model_cls.forward = forward_linked


def _add_ok(branch, residual, norm):
    """The residual add can go into `norm`'s kernel: an fp32 residual stream, a branch of the
    norm output's dtype (the kernel's one specialization), the norm taking our kernel."""
    return (residual.dtype == torch.float32 and branch.shape == residual.shape
            and branch.is_cuda and branch.dtype == _out_dtype(residual)
            and _norm_applies(norm, residual))


def _add_norm(residual, branch, norm):
    """(residual + branch, norm(residual + branch)) in one AddRMSNorm launch, or None when that
    is not applicable or failed (the caller then does the plain add and the norm)."""
    if "addnorm" in DISABLED or not _add_ok(branch, residual, norm):
        return None
    try:
        return AddRMSNorm.apply(residual, branch, norm.weight, norm.variance_epsilon,
                                _out_dtype(residual))
    except Exception as exc:                               
        if not fallback("addnorm", exc):
            raise
    return None


def patch_qwen2_add_norm():
    """`Qwen2DecoderLayer.forward` with the residual adds folded into the norms that read them
    (AddRMSNorm): the attention's add into `post_attention_layernorm`, and, once
    `link_next_norms` ran on the model, the MLP's add into the next layer's `input_layernorm`
    (or the final norm). The sum itself is still returned, so whatever else reads the layer
    output gets the usual value; the norm's output rides along on it and is taken by that norm
    only. The same operations in the same order otherwise; bit-identical (fp32 add, the same
    norm kernel). Anything else (a norm off our kernel, another dtype) takes the plain add.
    `Qwen2Model.forward` links the norms itself (`_link_norms_in_model_forward`), so the MLP's
    add folds in on every path, not only under the fused head's causal-LM forward."""
    from transformers.models.qwen2 import modeling_qwen2

    _link_norms_in_model_forward(modeling_qwen2.Qwen2Model)
    cls = modeling_qwen2.Qwen2DecoderLayer
    original = cls.forward
    from . import _in_chain

    if _in_chain(original, "_solution_add_norm"):
                                                                                               
                                                         
        return

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
                                                                                                
                                                                                             
                                                                                             
        checkpointed = self.training and getattr(self, "gradient_checkpointing", False)
        if checkpointed:
            _take_normed(hidden_states, self.input_layernorm)                               
        post = self.post_attention_layernorm
        if "addnorm" in DISABLED or not (hidden_states.dtype == torch.float32
                                         and _norm_applies(post, hidden_states)):
            FELL_BACK["addnorm"] += 1
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings,
                            **kwargs)
        if getattr(original, "_solution_group", False):
            from .qgroup import enter_layer                                                      

            enter_layer(self)                          
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs)
        fused = _add_norm(residual, hidden_states, post)
        if fused is not None:
            residual, hidden_states = fused
        else:
            FELL_BACK["addnorm"] += 1
            residual = residual + hidden_states
            hidden_states = post(residual)
        hidden_states = self.mlp(hidden_states)
        nxt = None if checkpointed else getattr(self, "_solution_next_norm", None)
        fused = None if nxt is None else _add_norm(residual, hidden_states, nxt)
        if fused is not None:
            total, normed = fused
            total._solution_normed = (nxt, normed)                                        
            return total
        FELL_BACK["addnorm"] += 1                                                    
        return residual + hidden_states

    forward._solution_add_norm = True
    forward._original = original
    LAYER_FORWARDS.update(original=original, add_norm=forward)
    cls.forward = forward
    ON.add("addnorm")


LAYER_FORWARDS = {}                                                                           
