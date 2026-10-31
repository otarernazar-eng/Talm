"""Fewer, bigger GEMMs per decoder layer on the packed 4-bit weights of qgroup.py.

qgroup.py dequantizes the seven projections of a layer into ONE contiguous fp16 buffer per pass,
in the order q, k, v, o, gate, up, down. So q|k|v ([1536 + 256 + 256, 1536] for Qwen2.5-1.5B)
and gate|up ([8960 + 8960, 1536]) are each one row-major matrix inside that buffer, and each
group reads one input tensor (the attention's norm output, the MLP's norm output). Here each
group is one cuBLAS GEMM forward and one backward instead of three and two:

* `FusedProjections` (the attention, `patch_qwen2_attention`): [q|k|v] = x @ [Wq; Wk; Wv]^T +
  [bq|bk|bv] in one GEMM; q and k are column views of its [N, 2048] result (RoPE reads them in
  place and keeps nothing), v is a fresh tensor (the attention keeps it for its backward, a view
  would keep the whole [N, 2048] result alive). The LoRA adapters of q_proj and v_proj are
  computed with PEFT's ops in PEFT's order, as `lora.LoraProjectionTorch` does (the same dropout
  draws from the same generator states, replayed in the backward, the same two fp16 GEMMs, the
  same `* scaling` and the same fp16 add onto the base output), so the adapter part is unchanged.
  Backward: the input gradient of the three base projections is one GEMM over the concatenated
  output gradient, dX = [dq|dk|dv] @ [Wq; Wk; Wv] (K = 2048), instead of three GEMMs and their
  adds; the adapters' input gradients (PEFT's ops) are then added to it, and the adapter weight
  gradients are PEFT's two GEMMs each.
* `FusedMLP` (`patch_qwen2_mlp`): gate|up in one GEMM, computed transposed: [gate^T; up^T] =
  [Wg; Wu] @ x^T, [17920, N]. Both halves of that result are contiguous, so the organizers'
  example SwiGLU kernels, which take flat contiguous operands, run on them directly (a row-major
  [N, 17920] result would give row-strided halves, which their launcher copies with
  .contiguous()). down_proj reads the activation transposed (one cuBLAS GEMM with transposed
  operands, no copy), and in the backward d(act)^T comes out of one GEMM in the same transposed
  layout, the example's backward kernel writes [dgate^T; dup^T] over the saved [gate^T; up^T]
  (as swiglu_inplace.py does), and dX is one GEMM over it (K = 17920) instead of two GEMMs and
  an add. The node keeps [gate^T; up^T] for its backward, as the SwiGLU node kept gate and up.
  OPT-IN (SOLUTION_FUSED_MLP=1), because it did not pay on the T4: in the transposed layout the
  token count is the contiguous extent of the GEMMs, and for a count that is not a multiple of 8
  cuBLAS takes CUTLASS kernels for 1-/2-element alignment (profile, session fp1-49dc: +72 ms of
  GEMM time per step, timed loop +22%); padding only the leading dimension changed nothing
  (session fp2-934f: still +14%). This version pads the extent itself (`_pad_rows`: x and dy
  copied into 8-row-aligned buffers, two small copies per layer), where the micro-benchmark
  (tools/bench_fused_proj.py, T4) has the fused MLP GEMMs 2-6% faster than the separate ones at
  aligned token counts (1104, 1600, 2048); not measured end to end yet.

Numerics. The base projections compute the same dot products (fp16 operands, fp32 accumulation
in cuBLAS) as the separate GEMMs; what can change is the order in which cuBLAS's kernel, picked
per problem shape, accumulates them, which moves a result by one fp16 rounding now and then.
In the backward the separate path rounds dq @ Wq, dk @ Wk and dv @ Wv to fp16 and adds them in
fp16 (two more roundings); the fused GEMM sums them in fp32 and rounds once (closer to the exact
value, not bit-identical). The forward adapter part, the adapter weight gradients, the generator
state and the SwiGLU arithmetic are bit-identical (tests: test_fused_proj.py).

Fallbacks: anything that is not exactly this situation takes the previous code path, per call:
a layer whose projections are not all packed in one group, an adapter that lora.lora_plan does
not take (or the LoRA node switched off), projections with hooks, a KV cache, other dtypes, and
the forward passes that qgroup dequantizes projection by projection (the last layer, at the
memory peak, `qgroup.SPLIT_LAYERS`). A fused node whose backward runs in such a split pass (the
layer before the last) takes the weights one at a time there and accumulates the separate GEMMs
with cuBLAS (beta = 1). SOLUTION_DISABLE_FUSED_PROJ=1 (or SOLUTION_DISABLE_FUSED_QKV=1)
switches the q|k|v fusion off; the MLP fusion is opt-in, SOLUTION_FUSED_MLP=1.

Graceful degradation (state.py): an unexpected error in a fused node's forward turns
"fused_proj" (q|k|v) or "fused_mlp" off and the call runs through the previous code path; the
node first gives back the uses of the layer's dequantized buffer it took and puts the CUDA
generator back to its state before its dropout draws, so the previous path (the LoRA node /
PEFT) draws the same masks. The plans step aside when "qgroup" is off (and the q|k|v plan when
the LoRA node is off), as for any layer qgroup does not pack. In the backward the weights come
from the layer group, which falls back by itself (`span` gives None, `weight` bitsandbytes'
dequantization); the rest is PEFT's arithmetic in PyTorch ops, as in lora.py.

Measured on the T4 (same-session A/B against SOLUTION_DISABLE_FUSED_PROJ=1, sessions fp1-49dc
and fp2-934f): q|k|v fused, timed loop -0.9% and -1.3%, peak memory -22 MiB, about 230 fewer
kernel launches per step at the same GPU time (the GEMM merge saves what the gradient
concatenation and the strided LoRA adds cost); loss curve within 0.015% (first step 0.005%).
"""

import atexit
import os

import torch
import torch.nn.functional as F
import triton

from .engaged import FELL_BACK, ON, RAN
from .lora import _power_of_two, _to_half, lora_plan
from .state import DISABLED, fallback
from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

Q, K, V, O, GATE, UP, DOWN = range(7)                                          
                                                                                   
INPLACE = os.environ.get("SOLUTION_DISABLE_SWIGLU_INPLACE") != "1"
COUNTS = {"attn": 0, "attn_fallback": 0, "mlp": 0, "mlp_fallback": 0}                      


class FusedProjections(torch.autograd.Function):
    """y_i = x @ W_i^T + b_i (+ scaling_i * B_i(A_i(dropout_i(x)))) for the adjacent packed
    projections first .. last - 1 of a qgroup.LayerGroup, the base part as one GEMM."""

    @staticmethod
    def forward(ctx, x, group, first, last, adapters, compact, *lora_weights):
        """`adapters`: ((segment, scaling, p), ...) in ascending segment order, their fp32
        (lora_A, lora_B) weights in `lora_weights` in the same order; `compact`: segments
        returned as fresh tensors (the others are column views of one [N, sum of out] result).
        Returns one [..., out_i] tensor per projection."""
        ctx.fetched, states = False, []
        try:
            return FusedProjections._forward(ctx, x, group, first, last, adapters, compact,
                                             lora_weights, states)
        except Exception:
                                                                                             
            if ctx.fetched:
                group.refund("fwd", last - first)
            drawn = [state for state in states if state is not None]
            if drawn:
                torch.cuda.set_rng_state(drawn[0], x.device)
            raise

    @staticmethod
    def _forward(ctx, x, group, first, last, adapters, compact, lora_weights, states):
        x2 = x.reshape(-1, x.shape[-1])
        sizes = [group.shapes[i][0] for i in range(first, last)]
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        weight = group.span("fwd", first, last)
        ctx.fetched = weight is not None
        out = F.linear(x2, weight, group.span_bias(first, last))
        del weight
        results = [None] * len(sizes)
        saved = [x2]
        for k, (seg, scaling, p) in enumerate(adapters):
                                                                                               
                                                                                           
                                                                         
            a16, b16 = _to_half(lora_weights[2 * k], lora_weights[2 * k + 1])
            state = None
            if p > 0:
                state = torch.cuda.get_rng_state(x.device)
                xd = torch.native_dropout(x2, p, True)[0]
            else:
                xd = x2
            t = torch.mm(xd, a16.t())
            del xd                                         
            branch = torch.mm(t, b16.t())
            base = out[:, bounds[seg]:bounds[seg + 1]]
            alpha = scaling
            if not _power_of_two(scaling):
                branch.mul_(scaling)
                alpha = 1
                                                                                             
                                                                                 
            results[seg] = (torch.add(base, branch, alpha=alpha) if seg in compact
                            else base.add_(branch, alpha=alpha))
            states.append(state)
            saved += [t, a16, b16]
        for seg in range(len(sizes)):
            if results[seg] is None:
                base = out[:, bounds[seg]:bounds[seg + 1]]
                results[seg] = base.contiguous() if seg in compact else base
        ctx.save_for_backward(*saved)
        ctx.group, ctx.first, ctx.last = group, first, last
        ctx.adapters, ctx.states, ctx.sizes = adapters, states, sizes
        ctx.x_shape = x.shape
        lead = x.shape[:-1]
        return tuple(r.view(*lead, r.shape[1]) for r in results)

    @staticmethod
    def backward(ctx, *grads):
        x2, *rest = ctx.saved_tensors
        n_tok = x2.shape[0]
        group = ctx.group
        dys = []
        for grad, size in zip(grads, ctx.sizes):
            if grad is None:                                         
                dys.append(x2.new_zeros(n_tok, size))
                continue
            dy = grad.reshape(n_tok, size)
            dys.append(dy if dy.dtype == torch.float16 else dy.to(torch.float16))
        dx = None
        if ctx.needs_input_grad[0]:
            weight = group.span("bwd", ctx.first, ctx.last)
            if weight is not None:                                                     
                dx = torch.mm(torch.cat(dys, 1), weight)
                del weight
            else:                                                                       
                for i, dy in enumerate(dys):
                    weight = group.weight("bwd", ctx.first + i)
                    if dx is None:
                        dx = torch.mm(dy, weight)
                    else:
                        dx.addmm_(dy, weight)
                    del weight
        lora_grads = []
        for k, (seg, scaling, p) in enumerate(ctx.adapters):
            t, a16, b16 = rest[3 * k:3 * k + 3]
            dy = dys[seg]
            state = ctx.states[k]
            mask = None
            xd = x2
            if state is not None:                                                             
                device = x2.device
                current = torch.cuda.get_rng_state(device)
                torch.cuda.set_rng_state(state, device)
                xd, mask = torch.native_dropout(x2, p, True)
                torch.cuda.set_rng_state(current, device)
            g = dy * scaling                                                                   
            u = torch.mm(g, b16)
            if dx is not None:
                                                                                             
                                                                                        
                                                                                      
                dx_lora = torch.mm(u, a16)
                if mask is not None:
                    dx_lora = torch.ops.aten.native_dropout_backward(dx_lora, mask,
                                                                     1.0 / (1.0 - p))
                dx.add_(dx_lora)
                del dx_lora
            index = 6 + 2 * k                                            
            lora_grads.append(torch.mm(u.t(), xd).float() if ctx.needs_input_grad[index]
                              else None)
            lora_grads.append(torch.mm(g.t(), t).float() if ctx.needs_input_grad[index + 1]
                              else None)
            del xd, mask
        dx = None if dx is None else dx.view(ctx.x_shape)
        return (dx, None, None, None, None, None, *lora_grads)


PAD = 8                                                                                


def _pad_rows(t2):
    """[N8, C] with the rows of `t2` [N, C] first (N8 = N rounded up to a multiple of PAD): the
    token count is the contiguous extent of the transposed GEMMs, and cuBLAS on the T4 takes its
    fast tensor-core kernels only when that extent is a multiple of 8 (otherwise CUTLASS kernels
    for 1- or 2-element alignment: measured, +72 ms of GEMM time per step; padding only the
    leading dimension did not help, measured too). The padding rows are left uninitialized: a
    token is never a contraction index here, so they only produce output rows/columns that are
    dropped (and SwiGLU values nothing reads)."""
    n_tok = t2.shape[0]
    n8 = -(-n_tok // PAD) * PAD
    if n8 == n_tok:
        return t2
    out = t2.new_empty((n8, t2.shape[1]))
    out[:n_tok].copy_(t2)
    return out


class FusedMLP(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) on a packed layer: gate|up as one transposed GEMM, the
    example's SwiGLU kernels on its two contiguous halves, down_proj on the transposed
    activation (see the module docstring)."""

    @staticmethod
    def forward(ctx, x, group, inplace):
        x2 = x.reshape(-1, x.shape[-1])
        n_tok = x2.shape[0]
        inter = group.shapes[GATE][0]
        out_t = torch.mm(group.span("fwd", GATE, DOWN), _pad_rows(x2).t())                  
        act_t = torch.empty((inter, out_t.shape[1]), dtype=out_t.dtype, device=out_t.device)
        n = act_t.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](out_t[:inter], out_t[inter:], act_t,
                                                           n, BLOCK=BLOCK)
        y = torch.mm(act_t.t(), group.span("fwd", DOWN, DOWN + 1).t())[:n_tok]          
        del act_t
        ctx.save_for_backward(out_t)
        ctx.group, ctx.inplace, ctx.x_shape, ctx.n_tok = group, inplace, x.shape, n_tok
        return y.view(*x.shape[:-1], y.shape[1])

    @staticmethod
    def backward(ctx, grad_out):
        (out_t,) = ctx.saved_tensors
        group, n_tok = ctx.group, ctx.n_tok
        inter = out_t.shape[0] // 2
        dy = grad_out.reshape(n_tok, -1)
        if dy.dtype != torch.float16:
            dy = dy.to(torch.float16)
        w_down = group.span("bwd", DOWN, DOWN + 1)
        if w_down is None:
            w_down = group.weight("bwd", DOWN)
        dact_t = torch.mm(w_down.t(), _pad_rows(dy).t())                                  
        del w_down
        d_t = out_t if ctx.inplace else torch.empty_like(out_t)                    
        n = dact_t.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](dact_t, out_t[:inter], out_t[inter:],
                                                           d_t[:inter], d_t[inter:], n,
                                                           BLOCK=BLOCK)
        del dact_t
        weight = group.span("bwd", GATE, DOWN)
        if weight is not None:                                        
            dx = torch.mm(d_t.t(), weight)
        else:                                      
            dx = torch.mm(d_t[:inter].t(), group.weight("bwd", GATE))
            dx.addmm_(d_t[inter:].t(), group.weight("bwd", UP))
        return dx[:n_tok].view(ctx.x_shape), None, None


_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")


def _no_hooks(modules):
    """No module hook would be skipped by calling the fused node instead of the modules."""
    from torch.nn.modules import module as nn_module

    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                 "_global_backward_pre_hooks"):
        if getattr(nn_module, name, None):
            return False
    return all(not getattr(m, attr, None) for m in modules for attr in _HOOK_DICTS)


def _packed(module):
    """(group, idx) of a plain packed bitsandbytes projection (qgroup tags it), else None."""
    return getattr(module, "_solution_group", None)


def _counted_fallback(module):
    """Whether a layer call that did not take a fused node counts in engaged.py's `fallbacks=`:
    `module` (q_proj / gate_proj, or the base layer under its PEFT adapter) is packed by qgroup,
    the layers these nodes are for, and the layer's forward pass is not a split one (the last
    layer's, qgroup.SPLIT_LAYERS, or a checkpointed layer's: the previous path by design)."""
    tag = _packed(getattr(module, "base_layer", module))
    return tag is not None and not tag[0].split_now("fwd")


def qkv_plan(attn, x):
    """(group, adapters, lora weights, modules) when q_proj, k_proj and v_proj of this attention
    are the projections 0, 1, 2 of one packed layer group, each either plain or a PEFT LoRA
    layer that lora.lora_plan takes (with the LoRA node patch installed), and this forward pass
    of the layer dequantizes the whole layer; else None."""
    if not (x.is_cuda and x.dtype == torch.float16 and x.dim() >= 2) \
            or "fused_proj" in DISABLED or "qgroup" in DISABLED:
        return None
    group, adapters, weights, modules = None, [], [], []
    for seg, name in enumerate(("q_proj", "k_proj", "v_proj")):
        module = getattr(attn, name, None)
        if module is None:
            return None
        tag = _packed(module)
        modules.append(module)
        if tag is None:
            if not getattr(type(module).forward, "_solution_lora", False) or "lora" in DISABLED:
                return None                                                                     
            plan = lora_plan(module, x)
            if plan is None:
                return None
            tag = plan[:2]
            name_ = module.active_adapters[0]
            modules += [module.base_layer, module.lora_A[name_], module.lora_B[name_],
                        module.lora_dropout[name_]]
            adapters.append((seg, plan[4], plan[5]))
            weights += [plan[2], plan[3]]
        if group is None:
            group = tag[0]
        if tag[0] is not group or tag[1] != seg:
            return None
    if any(group.shapes[i][1] != x.shape[-1] for i in (Q, K, V)):
        return None
    if not _no_hooks(modules) or group.split_now("fwd"):
        return None
    return group, tuple(adapters), weights


def mlp_plan(mlp, x):
    """The layer group when gate_proj, up_proj and down_proj are the plain packed projections
    4, 5, 6 of one group (no adapter, no bias), SiLU, and this forward pass dequantizes the
    whole layer; else None."""
    if not (x.is_cuda and x.dtype == torch.float16 and x.dim() >= 2) \
            or "fused_mlp" in DISABLED or "qgroup" in DISABLED:
        return None
    if getattr(getattr(mlp, "config", None), "hidden_act", "silu") != "silu":
        return None
    modules = [getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj")]
    tags = [_packed(m) for m in modules]
    if any(t is None for t in tags):
        return None
    group = tags[0][0]
    if any(t[0] is not group for t in tags) or [t[1] for t in tags] != [GATE, UP, DOWN]:
        return None
    hidden, inter = x.shape[-1], group.shapes[GATE][0]
    if group.shapes[GATE] != (inter, hidden) or group.shapes[UP] != (inter, hidden) \
            or group.shapes[DOWN] != (hidden, inter) \
            or any(group.biases[i] is not None for i in (GATE, UP, DOWN)):
        return None
    if not _no_hooks(modules) or group.split_now("fwd"):
        return None
    return group


def patch_qwen2_attention():
    """`Qwen2Attention.forward` with q|k|v as one FusedProjections node when `qkv_plan` takes
    the layer; transformers' own forward otherwise. The rest is transformers' forward as is
    (RoPE, the registered attention interface, o_proj)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Attention
    original = cls.forward
    ON.add("fused_proj")
    if getattr(original, "_solution_fused", False):
        return

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        plan = None
        if position_embeddings is not None and past_key_values is None:
            plan = qkv_plan(self, hidden_states)
        if plan is None:
            COUNTS["attn_fallback"] += 1
            if _counted_fallback(getattr(self, "q_proj", None)):
                FELL_BACK["fused_proj"] += 1                                         
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        group, adapters, weights = plan
        try:
            q, k, v = FusedProjections.apply(hidden_states, group, Q, V + 1, adapters, (V - Q,),
                                             *weights)
        except Exception as exc:                               
            if not fallback("fused_proj", exc):
                raise
            q = None
        if q is None:
            COUNTS["attn_fallback"] += 1
            FELL_BACK["fused_proj"] += 1                                                 
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        COUNTS["attn"] += 1
        RAN["fused_proj"] += 1
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            group.expected_bwd += 3                                                      
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)
        del q, k, v
        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states,
                                                                       cos, sin)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, modeling_qwen2.eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    forward._solution_fused = True
    forward._original = original
    cls.forward = forward
    _report_at_exit()


def patch_qwen2_mlp():
    """`Qwen2MLP.forward` as one FusedMLP node when `mlp_plan` takes the layer; the forward
    installed before (the example SwiGLU through swiglu_inplace.py) otherwise."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    previous = cls.forward
    ON.add("fused_mlp")
    if getattr(previous, "_solution_fused", False):
        return

    def forward(self, x):
        group = mlp_plan(self, x)
        if group is None:
            COUNTS["mlp_fallback"] += 1
            if _counted_fallback(getattr(self, "gate_proj", None)):
                FELL_BACK["fused_mlp"] += 1                                         
            return previous(self, x)
        uses = group.cache["fwd"][1]
        try:
            y = FusedMLP.apply(x, group, INPLACE)
        except Exception as exc:                                             
            if not fallback("fused_mlp", exc):
                raise
            y = None
        if y is None:
            entry = group.cache["fwd"]
            if entry[0] is not None and entry[1] < uses:                                 
                entry[1] = uses
            COUNTS["mlp_fallback"] += 1
            FELL_BACK["fused_mlp"] += 1                                                 
            return previous(self, x)
        COUNTS["mlp"] += 1
        RAN["fused_mlp"] += 1
        if torch.is_grad_enabled() and x.requires_grad:
            group.expected_bwd += 3                                                     
        return y

    forward._solution_fused = True
    forward._original = previous
    cls.forward = forward
    _report_at_exit()


_REPORTED = []


def _report_at_exit():
    """One `[solution]` line at exit: how many layer calls took the fused nodes."""
    if _REPORTED:
        return
    _REPORTED.append(True)

    def report():
        if any(COUNTS.values()):
            from . import _mark

            _mark(f"fused projections: attention {COUNTS['attn']} calls fused, "
                  f"{COUNTS['attn_fallback']} not; MLP {COUNTS['mlp']} fused, "
                  f"{COUNTS['mlp_fallback']} not")

    atexit.register(report)
