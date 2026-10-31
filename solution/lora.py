"""The 4-bit projections that carry an adapter (q_proj and v_proj here) as one autograd node.

PEFT's `Linear4bit.forward` runs the adapter as a chain of small ops on the [N, 1536] input:
`result.clone()`, dropout (mask generation + masked copy), the fp32 -> fp16 casts of A and B,
two skinny cuBLAS GEMMs ([N, 1536] x [1536, 16] and [N, 16] x [16, out]), `* scaling`,
`result + output`; the backward mirrors each of them. It keeps the dropout output (fp16) and the
mask (bool) of every adapted projection for the backward: 3 bytes per element of the layer input
per projection, 6 per layer.

`LoraProjectionTorch` runs PEFT's ops one for one (the same GEMMs, dtypes, roundings and
gradient accumulation order: bit-identical results, tested with torch.equal) but keeps only the
layer input x for the backward, shared by q_proj and v_proj of a layer, and replays the dropout
there from the CUDA generator state saved before it. No kernel of ours, nothing to compile.

A fused Triton kernel for the same branch was measured and dropped: on the T4, Triton lowers
`tl.dot` to FMA (no tensor cores on sm75), see tools/lora_branch_kernel.py.

The fp16 copies of A and B (`HalfCopies`; SOLUTION_DISABLE_LORA_CAST=1: one `.to(float16)` per
projection and forward, as before): PEFT's autocast casts each fp32 adapter weight to fp16 at
every call, 112 casts of 25-100 KB per step here, each its own launch. The weights change only
when the optimizer steps, so the copies are made for all adapters at once, in one
`torch._foreach_copy_` into one new fp16 buffer (two launches), at the first forward that finds
any weight changed since its copy (keyed on the weight's storage and version counter, which
every in-place update bumps: the optimizer's, `load_state_dict`'s, ...). The conversion is the
same fp32 -> fp16 round-to-nearest-even as `.to(float16)` (bit-identical copies, tested); a
refresh writes a new buffer, so the copies an earlier forward saved for its backward stay as
they were. Measured on the T4 (session t3a, per-kernel profiles of adjacent runs): fp16 copy
kernels 132 -> 20 per step, 5.93 -> 5.56 ms, plus 2 foreach launches (+0.06 ms): -0.31 ms of
GPU time and -110 launches per step.

Graceful degradation (state.py): an unexpected error in the node's forward turns "lora" off and
the call runs through PEFT's own forward; the CUDA generator is first put back to its state
before the node's dropout draw, so PEFT draws the very mask the node drew, and since the node is
bit-identical to PEFT the step gets the same values. The backward is PEFT's arithmetic in PyTorch
ops plus the base weight from the layer group, which falls back by itself (qgroup.py): nothing
else in it to replace (PEFT's own path would run the same ops). The node needs the packed base,
so it is off whenever "qgroup" is.
"""

import math
import os
import weakref

import torch
import torch.nn.functional as F

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

CAST_CACHE = os.environ.get("SOLUTION_DISABLE_LORA_CAST") != "1"


def _key(param):
    return param.data_ptr(), param._version


class HalfCopies:
    """fp16 copies of fp32 tensors (the adapters' weights), all refreshed in one foreach copy at
    the first `get` after any of them changed: another storage or version counter, or any
    optimizer step since the copy (`invalidate`, a step post-hook of torch.optim, also for
    optimizers that write the weights without bumping their version counter).

    Each entry also holds the storage it copied from: a data pointer names a storage only while
    that storage is alive. Without the reference, `param.data = a; param.data = b` (no version
    bump) could give `b` the freed block of the copied storage, with an equal key, and the stale
    copy would be served (found by solution/test_fuzz.py; the optimizer's in-place updates were
    never affected). The held storage is the parameter's own until such an assignment, and is
    released at the next refresh."""

    def __init__(self):
                                                                                              
        self.entries = []
        self.slots = {}                                    

    def get(self, param):
        slot = self.slots.get(id(param))
        if slot is None or self.entries[slot][0]() is not param:
            slot = len(self.entries)
            self.entries.append([weakref.ref(param), None, None, None])
            self.slots[id(param)] = slot
        entry = self.entries[slot]
        if entry[1] != _key(param):
            self.refresh()
        return entry[2]

    def invalidate(self, *_args, **_kwargs):
        """After an optimizer step every copy is stale: drop them too, so no CUDA memory is
        held from the step to the next forward (or after the last step); `get` refreshes them
        all at once anyway. A copy a pending backward saved stays alive through that backward."""
        for entry in self.entries:
            entry[1] = entry[2] = entry[3] = None

    @torch.no_grad()
    def refresh(self):
        live = [entry for entry in self.entries if entry[0]() is not None]
        if len(live) != len(self.entries):                                      
            self.entries = live
            self.slots = {id(entry[0]()): slot for slot, entry in enumerate(live)}
        stale = {}
        for entry in live:
            param = entry[0]()
            if entry[1] != _key(param):
                stale.setdefault(param.device, []).append((entry, param.detach()))
        if stale:
            RAN["lora_cast"] += 1
        for device, items in stale.items():
                                                                                              
            sizes = [param.numel() for _, param in items]
            flat = torch.empty(sum(sizes), dtype=torch.float16, device=device)
            copies = [part.view(param.shape)
                      for part, (_, param) in zip(flat.split(sizes), items)]
            torch._foreach_copy_(copies, [param for _, param in items])
            for (entry, param), copy in zip(items, copies):
                entry[1], entry[2], entry[3] = _key(param), copy, param.untyped_storage()


_half = HalfCopies()


_hooks = []


def _to_half(lora_a, lora_b):
    """The fp16 A and B of one adapter (fp32 CUDA tensors, as `lora_plan` checked)."""
    if not (CAST_CACHE and _hooks):
        if CAST_CACHE:
            FELL_BACK["lora_cast"] += 1                                                        
        return lora_a.to(torch.float16), lora_b.to(torch.float16)
    return _half.get(lora_a), _half.get(lora_b)


def _install_cast_cache():
    """The optimizer step post-hook that invalidates every copy (once per process)."""
    if CAST_CACHE and not _hooks:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        _hooks.append(register_optimizer_step_post_hook(_half.invalidate))


def _power_of_two(value):
    """A power of two >= 1: then branch * value is exact in fp16 (no bit lost below the normal
    range), so one rounding of out + value * branch (the add's alpha) equals PEFT's rounded
    product followed by the rounded add; the one exception is a product that overflows fp16,
    where PEFT's own output is already inf. Powers below 1 are not exact on subnormal products:
    with scaling 0.5 an output differed from PEFT's in the last bit (found by
    solution/test_fuzz.py), so they take PEFT's separate multiply."""
    mantissa, _ = math.frexp(value)
    return value >= 1 and mantissa == 0.5


class LoraProjectionTorch(torch.autograd.Function):
    """y = x @ W^T + bias + scaling * B(A(dropout(x))) on a packed 4-bit base (`qgroup`).

    PEFT's arithmetic op for op: the base GEMM on the layer group's dequantized weight, PEFT's
    dropout (`native_dropout`: the same RNG stream, the same masks), the same two fp16 GEMMs of
    the adapter, and the input gradients of the adapter and of the base accumulated by autograd
    as two contributions, adapter first, as in PEFT's graph (x enters twice for that).

    What changes is what is kept for the backward: only x (q_proj and v_proj of a layer keep the
    same tensor) and the CUDA generator state from just before the dropout. The backward restores
    that state and draws the dropout again, which reproduces the forward's mask exactly (same
    Philox seed and offset), then puts the generator back where it was, so every later draw is
    the one of PEFT's path. Also gone: PEFT's defensive `result.clone()`, and the forward's
    separate `* scaling` pass when scaling is a power of two >= 1 (2 here: it is the add's
    `alpha` then, exact and therefore the same values; see `_power_of_two`)."""

    @staticmethod
    def forward(ctx, x, x_base, lora_a, lora_b, group, idx, scaling, p):
                                                                                    
        ctx.rng_state = None
        fetched = False
        try:
            x2 = x.reshape(-1, x.shape[-1])
            weight = group.weight("fwd", idx)
            fetched = True
            out = F.linear(x2, weight, group.biases[idx])
            a16, b16 = _to_half(lora_a, lora_b)
            if p > 0:
                ctx.rng_state = torch.cuda.get_rng_state(x.device)
                xd = torch.native_dropout(x2, p, True)[0]
            else:
                xd = x2
            t = torch.mm(xd, a16.t())
            del xd                                         
            branch = torch.mm(t, b16.t())
            if _power_of_two(scaling):
                out.add_(branch, alpha=scaling)
            else:
                out.add_(branch.mul_(scaling))
        except Exception:
                                                                                 
            if fetched:
                group.refund("fwd")
            if ctx.rng_state is not None:
                torch.cuda.set_rng_state(ctx.rng_state, x.device)
            raise
        ctx.save_for_backward(x2, t, a16, b16)
        ctx.group, ctx.idx, ctx.scaling, ctx.p = group, idx, scaling, p
        ctx.x_shape = x.shape
        return out.view(*x.shape[:-1], out.shape[1])

    @staticmethod
    def backward(ctx, grad_out):
        x2, t, a16, b16 = ctx.saved_tensors
        dy = grad_out.reshape(-1, grad_out.shape[-1])
        if dy.dtype != torch.float16:
            dy = dy.to(torch.float16)
        mask = None
        xd = x2
        if ctx.rng_state is not None:
            device = x2.device
            current = torch.cuda.get_rng_state(device)
            torch.cuda.set_rng_state(ctx.rng_state, device)
            try:
                xd, mask = torch.native_dropout(x2, ctx.p, True)                                
            finally:                                                                         
                torch.cuda.set_rng_state(current, device)
        g = dy * ctx.scaling                                                                   
        u = torch.mm(g, b16)
        dx_lora = dx_base = None
        if ctx.needs_input_grad[0]:
            dx_lora = torch.mm(u, a16)
            if mask is not None:
                dx_lora = torch.ops.aten.native_dropout_backward(dx_lora, mask,
                                                                 1.0 / (1.0 - ctx.p))
            dx_lora = dx_lora.view(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            dx_base = torch.mm(dy, ctx.group.weight("bwd", ctx.idx)).view(ctx.x_shape)
        da = torch.mm(u.t(), xd).float() if ctx.needs_input_grad[2] else None
        db = torch.mm(g.t(), t).float() if ctx.needs_input_grad[3] else None
        return dx_lora, dx_base, da, db, None, None, None, None


def _autocast_fp16():
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    except TypeError:                   
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


def lora_plan(module, x):
    """(group, idx, A, B, scaling, p) when this PEFT LoRA Linear4bit can take the node, else
    None: one plain adapter (no DoRA/variants, no bias, not merged or disabled), a base layer
    packed by qgroup, fp16 input under fp16 autocast, fp32 A and B (as get_peft_model leaves
    them), dropout an nn.Dropout or nn.Identity."""
    if not (x.is_cuda and x.dtype == torch.float16 and x.dim() >= 2):
        return None
    if getattr(module, "disable_adapters", True) or getattr(module, "merged", True):
        return None
    adapters = module.active_adapters
    if len(adapters) != 1:
        return None
    name = adapters[0]
    if name not in module.lora_A or name in getattr(module, "lora_variant", {}) \
            or getattr(module, "lora_bias", {}).get(name, False):
        return None
    tag = getattr(module.base_layer, "_solution_group", None)
    if tag is None or "qgroup" in DISABLED or not _autocast_fp16():
        return None
    lin_a, lin_b = module.lora_A[name], module.lora_B[name]
    if not isinstance(lin_a, torch.nn.Linear) or not isinstance(lin_b, torch.nn.Linear) \
            or lin_a.bias is not None or lin_b.bias is not None:
        return None
    lora_a, lora_b = lin_a.weight, lin_b.weight
    rank, n_in = lora_a.shape
    if lora_b.shape[1] != rank or n_in != x.shape[-1] or lora_a.dtype != torch.float32 \
            or lora_b.dtype != torch.float32 or not lora_a.is_cuda:
        return None
    group, idx = tag
    if tuple(group.shapes[idx]) != (lora_b.shape[0], n_in):
        return None
    dropout = module.lora_dropout[name]
    if isinstance(dropout, torch.nn.Dropout):
        p = float(dropout.p) if dropout.training else 0.0
    elif isinstance(dropout, torch.nn.Identity):
        p = 0.0
    else:
        return None
    if not 0.0 <= p < 1.0:
        return None
    return group, idx, lora_a, lora_b, float(module.scaling[name]), p


def patch_peft_lora():
    """Route PEFT's LoRA `Linear4bit.forward` through `LoraProjectionTorch` in the plain case;
    anything else (extra arguments, adapters mixed per batch, no packed base, no autocast)
    takes PEFT's own path."""
    try:
        from peft.tuners.lora import bnb as lora_bnb
    except ImportError:
        return False
    cls = getattr(lora_bnb, "Linear4bit", None)
    if cls is None:
        return False
    _install_cast_cache()
    ON.add("lora")
    if getattr(cls.forward, "_solution_lora", False):
        return True
    original = cls.forward

    def forward(self, x, *args, **kwargs):
        plan = None if (args or kwargs or "lora" in DISABLED) else lora_plan(self, x)
        if plan is not None:
            group, idx, lora_a, lora_b, scaling, p = plan
            try:
                out = LoraProjectionTorch.apply(x, x, lora_a, lora_b, group, idx, scaling, p)
            except Exception as exc:                               
                if not fallback("lora", exc):
                    raise
            else:
                if torch.is_grad_enabled() and x.requires_grad:
                    group.expected_bwd += 1                                                    
                RAN["lora"] += 1
                return out
        FELL_BACK["lora"] += 1
        return original(self, x, *args, **kwargs)

    forward._solution_lora = True
    forward._original = original
    cls.forward = forward
    return True
