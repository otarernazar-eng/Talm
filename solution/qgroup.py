"""One NF4 dequantization launch per decoder layer instead of one per projection.

After unpadding the step is launch-bound: every Triton launch costs about 70 us of host time in
Triton's launcher plus the harness's timing wrapper, and the dequantization of the 4-bit weights
is launched 389 times per step (7 projections x 28 layers, forward and backward). This module
packs the seven NF4 weights of a layer into one contiguous buffer (the parameters become views
of it, so no memory is duplicated) and dequantizes them with ONE kernel launch when the layer's
first projection runs, forward or backward; the other six projections of that layer slice the
cached fp16 buffer, and the buffer is dropped after the seventh use. The GEMMs stay cuBLAS.

The kernel works on a tile of ROWS absmax blocks x QBLOCK/2 packed bytes (one row = the 64
weights that share one absmax). Per row it rebuilds the fp32 scale exactly as bitsandbytes does
for double-quantized absmax: `code[absmax8] * absmax2`, rounded, plus the offset of the weight
the row belongs to (`offset2` holds it per absmax2 block, so no per-element segment search).
Per byte, the two NF4 levels come from a 16-entry fp32 table (L1-resident), are multiplied by
the row's scale and rounded to fp16, and both halves are written as one 32-bit word. The grid
tiles the layer exactly, so there is no mask: the byte loads and the word stores are 16-byte
vector accesses (a mask against a runtime length would make every access scalar), and all index
math is int32. Fused multiply-add is disabled at compile time, so the scale is rounded twice as
bitsandbytes rounds it, and the output matches bitsandbytes' dequantization bit for bit
(measured: 0 of the 46.8 million weights differ; the test `test_group_dequant_is_bit_exact` asserts
it with torch.equal). The check at set-up is a tolerance check (relative 2e-3, at most 0.1% of
the values differing) after which a layer falls back to the per-projection path.

Graceful degradation (state.py): `LayerGroup.weight` is where every packed projection gets its
fp16 weight, forward and backward (GroupLinear here, the LoRA node in lora.py). An unexpected
error there turns "qgroup" off and the weight comes from bitsandbytes' own `dequantize_4bit` of
that projection, which our kernel matches bit for bit (tested), so the step's values do not
change; from then on every packed projection takes bitsandbytes' Linear4bit.forward.

The per-pass state (the cached buffers and their use counters) starts afresh at the beginning of
every forward of the layer (`LayerGroup.start_forward`, pure Python): a step cut short (an
exception between the forward and the backward or inside the backward, a loss whose backward
never runs) can leave a pass half-counted, and the next step must not reuse that buffer, hold it
(89 MiB per layer for Qwen2.5-1.5B) or count its uses on top of its own. Under gradient
checkpointing (off in this task: Soup sets it off; TRL's default is on) a layer's forward runs a
second time inside the backward, so its pass counts would not match its uses: a checkpointed layer
keeps no per-pass state and dequantizes each projection alone at each use (robust, not disabled:
the same kernel, the same values, more launches; there the training is slower anyway).
"""

import os
import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

ROWS = 64                                                                             
NUM_WARPS = 4
NSEG = 7                              
                                                                                               
                                                                                               
WARMUP_QBLOCK, WARMUP_QBLOCK2 = 64, 256
                                                                                              
NF4_LEVELS = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
              0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
              0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
                                                                                         
                                                                                                
                                                                                                
                                                                     
_SPLIT_OFF = os.environ.get("SOLUTION_DISABLE_SPLIT_DEQUANT") == "1"
SPLIT_LAYERS = {"fwd": 0 if _SPLIT_OFF else 1, "bwd": 0 if _SPLIT_OFF else 2}


@triton.jit
def nf4_group_dequant_kernel(packed_ptr, absmax8_ptr, code_ptr, absmax2_ptr, offset2_ptr,
                             levels_ptr, out_ptr, QBLOCK: tl.constexpr, QBLOCK2: tl.constexpr,
                             ROWS: tl.constexpr):
    """out (fp16 pairs as uint32) of rows [pid * ROWS, pid * ROWS + ROWS) of QBLOCK weights."""
    HALF: tl.constexpr = QBLOCK // 2                                 
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
                                                                                          
    code_index = tl.load(absmax8_ptr + rows).to(tl.int32)
    block2 = rows // QBLOCK2
    scale = tl.load(code_ptr + code_index) * tl.load(absmax2_ptr + block2) \
        + tl.load(offset2_ptr + block2)
                                                                                            
    words = rows[:, None] * HALF + tl.arange(0, HALF)[None, :]
    packed = tl.load(packed_ptr + words).to(tl.int32)
    level_first = tl.load(levels_ptr + (packed >> 4))
    level_second = tl.load(levels_ptr + (packed & 15))
    half_first = (level_first * scale[:, None]).to(tl.float16).to(tl.uint16, bitcast=True)
    half_second = (level_second * scale[:, None]).to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(out_ptr + words, half_first.to(tl.uint32) | (half_second.to(tl.uint32) << 16))


def launch(packed, absmax8, code, absmax2, offset2, levels, out, n_rows, qblock, qblock2):
    """The one launch site of the kernel (training and the warm-up thread): `out` is the
    contiguous fp16 output of `n_rows` absmax blocks (a multiple of ROWS)."""
    nf4_group_dequant_kernel[(n_rows // ROWS,)](
        packed, absmax8, code, absmax2, offset2, levels, out.view(torch.int32),
        QBLOCK=qblock, QBLOCK2=qblock2, ROWS=ROWS, num_warps=NUM_WARPS, enable_fp_fusion=False)


_LEVELS = {}


def _levels(device):
    """The NF4 level table on `device`, shared by all layers."""
    if device not in _LEVELS:
        _LEVELS[device] = torch.tensor(NF4_LEVELS, dtype=torch.float32, device=device)
    return _LEVELS[device]


def _state_ok(state, first):
    return (state is not None and state.quant_type == "nf4" and state.nested
            and state.blocksize == first.blocksize
            and state.state2.blocksize == first.state2.blocksize
            and len(state.shape) == 2 and state.absmax.dtype == torch.uint8
            and state.state2.absmax.dtype == torch.float32
            and torch.equal(state.state2.code, first.state2.code))


class LayerGroup:
    """The seven 4-bit projections of one decoder layer, packed, with per-pass caches."""

    def __init__(self, modules):
        assert len(modules) == NSEG
        states = [m.weight.quant_state for m in modules]
        first = states[0]
        if not all(_state_ok(s, first) for s in states):
            raise ValueError("unsupported quant state")
        if first.blocksize % 2 or (first.blocksize * first.state2.blocksize) % 2:
            raise ValueError("odd block size")
        for s in states:
            n = s.shape[0] * s.shape[1]
            if n % (first.blocksize * first.state2.blocksize):
                raise ValueError("weight size not a multiple of the absmax double-block")
        device = modules[0].weight.device
        self.qblock, self.qblock2 = first.blocksize, first.state2.blocksize
        self.shapes = [tuple(s.shape) for s in states]
        self.numels = [a * b for a, b in self.shapes]
        self.pair_bounds = []                                         
        total = 0
        for n in self.numels:
            self.pair_bounds.append(total)
            total += n // 2
        self.pair_bounds.append(total)
        self.elem_bounds = [2 * p for p in self.pair_bounds]
                                                                                                 
        packed = torch.cat([m.weight.data.view(-1) for m in modules])
        absmax8 = torch.cat([s.absmax.view(-1) for s in states])
        absmax2 = torch.cat([s.state2.absmax.view(-1) for s in states])
        p = a = c = 0
        for m, s in zip(modules, states):
            n = s.shape[0] * s.shape[1]
            m.weight.data = packed[p:p + n // 2].view(-1, 1)
            s.absmax = absmax8[a:a + n // self.qblock]
            s.state2.absmax = absmax2[c:c + n // (self.qblock * self.qblock2)]
            p += n // 2
            a += n // self.qblock
            c += n // (self.qblock * self.qblock2)
        self.packed, self.absmax8, self.absmax2 = packed, absmax8, absmax2
        self.code = first.state2.code
        self.offsets = torch.stack([s.offset.reshape(()).to(torch.float32)
                                    for s in states]).to(device)
        self.n_pairs = total
                                                                                 
        half = self.qblock // 2
        if half & (half - 1) or total % (ROWS * half) or total >= 2 ** 31:
            raise ValueError("layer size does not fit the kernel's tiling")
        self.n_rows = total // half
                                                                                         
        self.offset2 = torch.cat([self.offsets[i:i + 1].expand(n // (self.qblock * self.qblock2))
                                  for i, n in enumerate(self.numels)])
        self.levels = _levels(device)
        self.biases = [None if m.bias is None else m.bias.detach().to(torch.float16)
                       for m in modules]
        self.params = [m.weight for m in modules]                                               
        self._span_biases = {}                                                    
        self.cache = {"fwd": [None, 0], "bwd": [None, 0]}
        self.expected_bwd = 0                                                           
        self.split_left = 0                                               
        self.checkpointed = False                                                               
        self.depth = None                                                           
                                                                                             
                                                                                            
                                                                                             
                                                                                                 
                                                                                              
                                   
        self.row_bounds = [e // self.qblock for e in self.elem_bounds]
        self.splittable = all(
            (r1 - r0) % ROWS == 0 and r0 % (4 * self.qblock2) == 0
            for r0, r1 in zip(self.row_bounds[:-1], self.row_bounds[1:]))

    def dequantize_all(self):
        out = torch.empty(self.n_pairs * 2, dtype=torch.float16, device=self.packed.device)
        launch(self.packed, self.absmax8, self.code, self.absmax2, self.offset2, self.levels, out,
               self.n_rows, self.qblock, self.qblock2)
        return out

    def dequantize_one(self, idx):
        """fp16 [out, in] of projection `idx` alone: the same kernel launched on that weight's
        slice of the packed buffers (it starts on an absmax2 block, so the block indices inside
        the kernel stay right; every slice pointer keeps the 16-byte alignment of the full
        launch, so the compiled kernel is the same)."""
        r0, r1 = self.row_bounds[idx], self.row_bounds[idx + 1]
        b0 = r0 // self.qblock2
        out = torch.empty(self.numels[idx], dtype=torch.float16, device=self.packed.device)
        launch(self.packed[r0 * (self.qblock // 2):], self.absmax8[r0:], self.code,
               self.absmax2[b0:], self.offset2[b0:], self.levels, out, r1 - r0, self.qblock,
               self.qblock2)
        return out.view(self.shapes[idx])

    def weight(self, which, idx):
        """fp16 [out, in] of projection `idx`; one dequant launch per pass per layer (one per
        projection in the passes of the last layers listed in SPLIT_LAYERS, and per projection
        and use in a gradient-checkpointed layer). Once "qgroup" is off (state.py),
        bitsandbytes' dequantization of that projection, the same values."""
        if "qgroup" not in DISABLED:
            try:
                return self._weight(which, idx)
            except Exception as exc:                               
                if not fallback("qgroup", exc):
                    raise
        return self.reference_weight(idx)

    def reference_weight(self, idx):
        """bitsandbytes' own dequantization of projection `idx` (bit-identical to the kernel's),
        from the parameter's views of the packed buffers. The pass caches are dropped: nothing
        reads them any more."""
        import bitsandbytes.functional as bnbf

        if self.cache["fwd"][0] is not None or self.cache["bwd"][0] is not None:
            self.reset()
        FELL_BACK["group"] += 1
        param = self.params[idx]
        return bnbf.dequantize_4bit(param.data, param.quant_state).to(torch.float16)

    def reset(self):
        """Forget the pass state: cached weights and backward counts."""
        self.cache = {"fwd": [None, 0], "bwd": [None, 0]}
        self.expected_bwd = 0
        self.split_left = 0

    def refund(self, which, uses=1):
        """Give back the uses of the pass cache taken by a call whose result was dropped (a LoRA
        or fused q|k|v node failing after it fetched its base weights: the fallback path fetches
        them again)."""
        entry = self.cache[which]
        if entry[0] is not None:
            entry[1] += uses

    def start_forward(self, checkpointed=False):
        """The layer's forward begins: the step boundary of its passes. Whatever a step cut short
        left (a buffer still cached, uses counted for a backward that never ran) is dropped, so
        no step reuses or holds a stale buffer. On the normal path everything is at rest already
        and nothing changes (pure Python, no launch). `checkpointed`: the layer runs under
        gradient checkpointing, where its forward runs again inside the backward (cut short by
        PyTorch's early stop, after the last saved tensor) and pass counts cannot match the uses:
        `weight` then keeps no state."""
        fwd, bwd = self.cache["fwd"], self.cache["bwd"]
        fwd[0] = bwd[0] = None
        fwd[1] = bwd[1] = 0
        self.expected_bwd = self.split_left = 0
        self.checkpointed = checkpointed

    def split_now(self, which):
        """True when the next use in pass `which` dequantizes one projection alone (the passes of
        the last layers listed in SPLIT_LAYERS, and every pass of a gradient-checkpointed layer,
        which keeps no per-pass state: start_forward)."""
        return (self.cache[which][0] is None and self.depth is not None
                and self.depth < SPLIT_LAYERS[which] and self.splittable) or self.checkpointed

    def _take(self, which, uses):
        """The layer's dequantized buffer for pass `which` (dequantized at the first use of the
        pass), counted as `uses` projection uses; dropped after the last expected use."""
        entry = self.cache[which]
        if entry[0] is None:
            entry[0] = self.dequantize_all()
            RAN["group"] += 1
            if which == "fwd":
                entry[1] = NSEG
            else:                                                                        
                entry[1] = max(self.expected_bwd, 1)
                self.expected_bwd = 0
        buf = entry[0]
        entry[1] -= uses
        if entry[1] <= 0:
            entry[0] = None                                                                 
        return buf

    def _weight(self, which, idx):
        if self.split_now(which):
            if self.checkpointed:                                     
                self.expected_bwd = 0                                                         
                if not self.splittable:
                    RAN["group"] += 1
                    full = self.dequantize_all()
                    return full[self.elem_bounds[idx]:self.elem_bounds[idx + 1]].view(
                        self.shapes[idx])
            elif which == "bwd":
                if self.split_left == 0:                                   
                    self.split_left = max(self.expected_bwd, 1)
                    self.expected_bwd = 0
                self.split_left -= 1
            RAN["split"] += 1
            return self.dequantize_one(idx)
        buf = self._take(which, 1)
        return buf[self.elem_bounds[idx]:self.elem_bounds[idx + 1]].view(self.shapes[idx])

    def span(self, which, first, last):
        """fp16 [sum of out, in] of the adjacent projections first .. last - 1 (q|k|v, gate|up)
        as ONE view of the layer's dequantized buffer, counted as last - first uses of the pass
        (fused_proj.py); None in a split pass, where the caller takes them one at a time with
        `weight`. The projections must share their input size (checked by the caller). Also
        None once "qgroup" is off or when this fails (state.py): `weight` then gives each
        projection bitsandbytes' dequantization, the same values."""
        if "qgroup" in DISABLED:
            return None
        try:
            if self.split_now(which):
                return None
            buf = self._take(which, last - first)
        except Exception as exc:                               
            if not fallback("qgroup", exc):
                raise
            return None
        rows = sum(self.shapes[i][0] for i in range(first, last))
        return buf[self.elem_bounds[first]:self.elem_bounds[last]].view(rows, self.shapes[first][1])

    def span_bias(self, first, last):
        """The fp16 biases of projections first .. last - 1 concatenated (zeros for a projection
        without one: adding 0 is exact), or None when none has a bias; built once."""
        key = (first, last)
        if key not in self._span_biases:
            parts = self.biases[first:last]
            if all(b is None for b in parts):
                self._span_biases[key] = None
            else:
                ref = next(b for b in parts if b is not None)
                self._span_biases[key] = torch.cat([
                    b if b is not None else ref.new_zeros(self.shapes[first + i][0])
                    for i, b in enumerate(parts)])
        return self._span_biases[key]


class GroupLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, idx):
        ctx.group, ctx.idx = group, idx
        return F.linear(x, group.weight("fwd", idx), group.biases[idx])

    @staticmethod
    def backward(ctx, grad_out):
        weight = ctx.group.weight("bwd", ctx.idx)
        return torch.matmul(grad_out, weight), None, None


_CHECKED = False                                                                      
_GROUPS = []                                                             


def reset_passes():
    """Reset every layer group's pass state (state.py: before a forward that failed part-way is
    run again from the start). The failed attempt counted backward uses that no backward will
    make, and may have left a pass cache half used: kept, those would hold a dequantized layer
    (89 MiB on the task) alive between passes, for every layer it ran."""
    alive = []
    for ref in _GROUPS:
        group = ref()
        if group is not None:
            group.reset()
            alive.append(ref)
    _GROUPS[:] = alive

PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _bnb_linear(module):
    """The bitsandbytes Linear4bit under a PEFT wrapper (or the module itself)."""
    import bitsandbytes as bnb

    base = getattr(module, "base_layer", module)
    return base if isinstance(base, bnb.nn.Linear4bit) else None


def _self_check(group, modules):
    import bitsandbytes.functional as bnbf

    full = group.dequantize_all()
    for idx, m in enumerate(modules):
        ref = bnbf.dequantize_4bit(m.weight.data, m.weight.quant_state).to(torch.float16)
        ours = full[group.elem_bounds[idx]:group.elem_bounds[idx + 1]].view(group.shapes[idx])
        diff = (ours.float() - ref.float()).abs()
        if not bool((diff <= ref.float().abs() * 2e-3 + 1e-6).all()):
            return False
        if float((ours != ref).float().mean()) > 1e-3:
            return False
    return True


def _pack(layer):
    """The layer's seven projections as one checked LayerGroup, or None."""
    modules = []
    for path in PROJECTIONS:
        obj = layer
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        base = _bnb_linear(obj)
        if base is None or base.weight.requires_grad or not base.weight.is_cuda:
            return None
        if base.bias is not None and base.bias.requires_grad:
            return None
        if getattr(base, "compute_dtype", None) not in (None, torch.float16):
            return None
        modules.append(base)
    try:
        group = LayerGroup(modules)
        if not _CHECKED and not _self_check(group, modules):
            return None
    except Exception:                                                              
        return None
    globals()["_CHECKED"] = True
    attn = getattr(layer, "self_attn", None)
    layer_idx = getattr(attn, "layer_idx", None)
    n_layers = getattr(getattr(attn, "config", None), "num_hidden_layers", None)
    if isinstance(layer_idx, int) and isinstance(n_layers, int) and 0 <= layer_idx < n_layers:
        group.depth = n_layers - 1 - layer_idx
    for idx, base in enumerate(modules):
        base._solution_group = (group, idx)
    _GROUPS.append(weakref.ref(group))
    return group


                                                                                              
                                                                                               
                                                                                             
                                                                                 
FALLBACK_DEQUANT = False


def _has_4bit(layer):
    modules = getattr(layer, "modules", None)
    return callable(modules) and any(_bnb_linear(m) is not None for m in modules())


def setup_layer(layer):
    """Pack the layer's seven projections once; tag each bitsandbytes module with its group."""
    if getattr(layer, "_solution_group_done", False):
        return
    layer._solution_group_done = True
    group = _pack(layer)
    layer._solution_layer_group = group                                                  
    if group is None and FALLBACK_DEQUANT and _has_4bit(layer):
        from .nf4_dequant import patch_bnb_dequant

        patch_bnb_dequant()                                                                  


def _wrapped(layer):
    """An instance-level `forward` on the decoder layer or on its attention: a wrapper this
    package cannot see through. Soup's `gradient_checkpointing: selective` tier puts one on every
    layer's attention (torch.utils.checkpoint inside, HF's flag left off), so the attention runs
    again inside the backward, outside the layer's forward: its weight uses there belong to no
    counted pass, and a whole-layer buffer they dequantized would stay cached until the layer's
    next forward."""
    attention = getattr(layer, "_modules", {}).get("self_attn")
    return "forward" in getattr(layer, "__dict__", {}) or (
        attention is not None and "forward" in attention.__dict__)


def enter_layer(layer):
    """The decoder layer's forward begins: pack the layer on first use, then start its passes
    afresh (`LayerGroup.start_forward`), checkpointed when the layer runs under gradient
    checkpointing (transformers' GradientCheckpointingLayer: the flag and training mode; or a
    wrapper on the layer or its attention, `_wrapped`).
    Never fatal (state.py): an unexpected error turns "qgroup" off and the layer runs unpacked
    (bitsandbytes' path)."""
    if "qgroup" in DISABLED:
        return
    try:
        if not getattr(layer, "_solution_group_done", False):
            setup_layer(layer)
        group = getattr(layer, "_solution_layer_group", None)
        if group is not None:
            group.start_forward(bool(getattr(layer, "training", False) and (
                getattr(layer, "gradient_checkpointing", False) or _wrapped(layer))))
    except Exception as exc:                               
        if not fallback("qgroup", exc):
            raise


prepare_layer = enter_layer                                                        


def patch_group_dequant():
    """`Qwen2DecoderLayer.forward` packs its layer on first use and starts its passes afresh
    (enter_layer); `bitsandbytes.nn.Linear4bit.forward` routes packed modules through
    GroupLinear for fp16 inputs."""
    import bitsandbytes as bnb
    from transformers.models.qwen2 import modeling_qwen2

    from . import _in_chain

    layer_cls = modeling_qwen2.Qwen2DecoderLayer
    layer_forward = layer_cls.forward
    ON.add("group")
    if not _in_chain(layer_forward, "_solution_group"):
        def forward(self, *args, **kwargs):
            enter_layer(self)
            return layer_forward(self, *args, **kwargs)

        forward._solution_group = True
        forward._original = layer_forward
        layer_cls.forward = forward

    lin_cls = bnb.nn.Linear4bit
    lin_forward = lin_cls.forward
    if _in_chain(lin_forward, "_solution_group"):
        return True

    def linear_forward(self, x):
        tag = getattr(self, "_solution_group", None)
        if tag is not None and "qgroup" not in DISABLED and x.is_cuda \
                and x.dtype == torch.float16:
            group, idx = tag
            try:
                out = GroupLinear.apply(x, group, idx)
            except Exception as exc:                               
                if not fallback("qgroup", exc):
                    raise
            else:
                if torch.is_grad_enabled() and x.requires_grad:
                    group.expected_bwd += 1                                                      
                return out
                                                                                    
        FELL_BACK["group"] += 1
        return lin_forward(self, x)

    linear_forward._solution_group = True
    linear_forward._original = lin_forward
    lin_cls.forward = linear_forward
    return True
