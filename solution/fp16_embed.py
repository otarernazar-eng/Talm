"""The frozen tied embedding / output weight kept in fp16, the dtype Soup loaded it in.

On a T4 Soup loads Qwen2.5-1.5B in fp16. PEFT's `prepare_model_for_kbit_training` then upcasts
every parameter that is not 4-bit to fp32 "for stability", the tied input embedding / `lm_head`
weight [151936, 1536] included: 890 MiB, frozen, alive at the memory peak of every step (the
largest tensor of the run). An fp16 value is exact in fp32, so after that upcast an fp16 tensor
still holds the same numbers in 445 MiB. This module undoes that one upcast:

- `patch_prepare_model_for_kbit_training` wraps PEFT's function. When the model's input
  embedding was fp16 before the call and PEFT upcast it to fp32, and it is a plain, frozen
  `nn.Embedding` on CUDA whose weight IS the output head's weight (tied: one Parameter),
  `store_fp16` puts that weight back in fp16 (`param.data = fp16 copy`), after checking on the
  GPU that the copy converts back to the fp32 tensor bit for bit (otherwise nothing changes).
  The Parameter object stays the same, so `lm_head.weight is embed_tokens.weight` still holds.
- A forward hook on that embedding returns the looked-up rows as fp32: a gather from the fp16
  weight, then one exact cast (one small kernel per step), so the decoder gets the same fp32
  residual-stream input as from the fp32 weight, bit for bit.
- The output head reads fp16 values anyway. Our fused head (`fused_head.py`) cast every fp32
  vocabulary slice to fp16 before its GEMMs; from an fp16 weight it takes the slice as it is
  (the same fp16 operands without the copy, so the same logits, loss, entropy, accuracy and
  gradient, bit for bit; tested). Under autocast transformers' `lm_head` (the fallback paths)
  likewise casts the weight to fp16 first: the same values.

For the jury:
- Same values: every number of the weight is unchanged (fp16 -> fp32 is exact; checked
  bitwise on the GPU before the switch).
- Same computation, bit for bit: the embedding output is torch.equal to the fp32 lookup, the
  fused head's outputs and gradient are torch.equal with either storage, and a LoRA training
  step of a small Qwen2 through `solution.apply()` gives the same loss and LoRA gradients
  (test_fp16_embed.py, also run on the T4).
- Nothing trainable changes: the weight is frozen before and after (requires_grad False); LoRA
  weights (fp32), optimizer, adapter tensors and every file Soup writes are untouched.
- No mode is switched: Soup's config, TRL and PEFT run as before (PEFT's function itself runs
  unchanged); only this one frozen tensor is stored back in the dtype it was loaded in.
- The one observable difference: transformers' `model.dtype` (the dtype of the FIRST floating
  parameter, which is this weight) reads float16 instead of float32. Nothing in the training
  run reads it: the code of Soup's SFT trainer, TRL's SFTTrainer, transformers' Trainer and
  Qwen2 model, PEFT's LoRA set-up and saving, and accelerate was checked; the attention mask and
  rotary dtypes come from the embedding output, which stays fp32; the trainer's FLOP count and
  PEFT's parameter counts read numbers of elements, not dtypes.
- Memory at load: the fp16 copy exists next to the fp32 weight for a moment, the same transient
  as PEFT's own upcast a moment earlier (peak allocated during `prepare` 1,981 -> 2,005 MiB on
  the real model), far below the training peak; afterwards 1,536 -> 1,091 MiB are allocated.
- Measured on the real Qwen2.5-1.5B (4-bit, fp16, RTX 4080 with deterministic algorithms, so
  that two runs of the same code agree): three AdamW steps through `solution.apply()` give the
  same losses, LoRA gradients and LoRA weights bit for bit with either storage.
The frozen RMSNorm weights and q/k/v biases (also upcast, 0.6 MiB together) stay fp32: nothing
to gain there.

SOLUTION_DISABLE_FP16_EMBED=1 switches it off (the weight stays fp32, as PEFT leaves it).
"""

import functools
import os
import weakref

import torch

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

# (weak reference to the embedding module, hook handle, MiB saved) for every embedding this module
# stored in fp16. Weak: a strong reference would keep the module and its fp16 weight (445 MiB
# here) alive after the model is gone (the handle itself holds only weak references).
_applied = []


def enabled():
    return os.environ.get("SOLUTION_DISABLE_FP16_EMBED") != "1"


def head_weight_dtype(half):
    """The dtype of the output weight the fused head gets in training, for the warm-up's
    compile: fp16 when this module stores it (the fp16 path of a T4, where Soup loads the model
    in fp16 and trains under fp16 autocast, with the switch on), fp32 (PEFT's upcast) otherwise."""
    return torch.float16 if half == torch.float16 and enabled() else torch.float32


def tied_embedding(model):
    """(embedding module, weight) when `model`'s input embedding is a plain `nn.Embedding` on
    CUDA, without max_norm (which would write the weight), whose weight is the same Parameter as
    the output head's weight; None for anything else."""
    try:
        emb = model.get_input_embeddings()
        head = model.get_output_embeddings()
    except Exception:  # noqa: BLE001 - not a transformers model, or no embeddings: not ours
        return None
    if type(emb) is not torch.nn.Embedding or head is None:
        return None
    weight = emb.weight
    if (not isinstance(weight, torch.nn.Parameter) or getattr(head, "weight", None) is not weight
            or weight.dim() != 2 or not weight.is_cuda or emb.max_norm is not None):
        return None
    return emb, weight


def bitwise_equal(w16, w32, rows=2048):
    """Whether the fp16 tensor converts back to the fp32 one bit for bit (fp32 bit patterns
    compared, so even the sign of a zero counts), in blocks of rows (temporaries of 12 MiB at
    this task's width)."""
    differs = torch.zeros((), dtype=torch.bool, device=w32.device)
    for start in range(0, w32.shape[0], rows):
        back = w16[start:start + rows].float().view(torch.int32)
        differs |= (back != w32[start:start + rows].view(torch.int32)).any()
    return not bool(differs)


def _fp32_output(module, args, output):
    """Forward hook of an embedding stored in fp16: the looked-up rows as fp32, exactly the
    values the fp32 weight gives (fp16 -> fp32 is exact)."""
    if isinstance(output, torch.Tensor) and output.dtype == torch.float16:
        return output.float()
    return output


def store_fp16(model, expected=None, clear_cache=True):
    """Store `model`'s tied, frozen fp32 embedding / output weight in fp16 when that loses
    nothing (bitwise check); its lookups then return fp32 as before. `expected`: the weight must
    be this Parameter. Returns the MiB saved (0: nothing changed)."""
    found = tied_embedding(model)
    if found is None:
        return 0
    emb, weight = found
    if (expected is not None and weight is not expected) or weight.dtype != torch.float32 \
            or weight.requires_grad or not weight.is_contiguous():
        return 0
    w16 = weight.data.to(torch.float16)
    if not bitwise_equal(w16, weight.data):
        return 0
    saved = weight.numel() * 2 / 2**20
    # the hook first (a no-op while the weight is fp32): an error between the two steps cannot
    # leave an fp16 weight whose lookups return fp16
    handle = emb.register_forward_hook(_fp32_output)
    weight.data = w16  # the same Parameter object: still tied to the output head
    _applied.append((weakref.ref(emb), handle, saved))
    if clear_cache:
        torch.cuda.empty_cache()  # the freed fp32 weight, as PEFT does after its own casts
    return saved


def patch_prepare_model_for_kbit_training():
    """Wrap PEFT's `prepare_model_for_kbit_training` (in each PEFT namespace that exports it):
    after it ran, the tied embedding it upcast from fp16 goes back to fp16 (`store_fp16`)."""
    try:
        import peft
        import peft.utils as peft_utils
        from peft.utils import other
    except ImportError:
        return False
    original = getattr(other, "prepare_model_for_kbit_training", None)
    if original is None:
        return False
    if getattr(original, "_solution_fp16_embed", False):
        return True

    @functools.wraps(original)
    def prepare_model_for_kbit_training(model, *args, **kwargs):
        # (graceful degradation, state.py: an unexpected error in our steps before and after
        # PEFT's own function leaves the weight as PEFT leaves it; PEFT's call is not ours)
        loaded = None
        try:
            found = tied_embedding(model)
            loaded = found[1] if found is not None and found[1].dtype == torch.float16 else None
        except Exception as exc:  # noqa: BLE001 - see state.py
            if not fallback("fp16_embed", exc):
                raise
        result = original(model, *args, **kwargs)
        saved = 0
        if loaded is not None and "fp16_embed" not in DISABLED:
            # PEFT's upcast of a weight loaded in fp16: back to fp16
            try:
                clear = kwargs.get("auto_clear_cache", args[2] if len(args) > 2 else True)
                saved = store_fp16(result, expected=loaded, clear_cache=bool(clear))
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("fp16_embed", exc):
                    raise
            from . import _mark

            _mark("tied embedding/lm_head kept in fp16 after PEFT's upcast: "
                  + (f"-{saved:.0f} MiB" if saved else "not applicable"))
        if saved:  # engaged.py's counters (not stored: no fp16 tied weight, or not exact)
            RAN["fp16_embed"] += 1
        else:
            FELL_BACK["fp16_embed"] += 1
        return result

    prepare_model_for_kbit_training._solution_fp16_embed = True
    prepare_model_for_kbit_training._original = original
    for namespace in (peft, peft_utils, other):
        if getattr(namespace, "prepare_model_for_kbit_training", None) is original:
            namespace.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    ON.add("fp16_embed")
    return True
