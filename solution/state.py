"""Graceful degradation: an unexpected error in one of our fast paths turns that component off
for the rest of the process, and the call that failed runs again through the original code.

The jury runs `soup train` on rows we have never seen. Every patch checks the types and shapes
it was written for and takes the original path otherwise; this module covers what those checks
cannot foresee (a Triton launch failing on an odd shape, a bitsandbytes or transformers state we
did not meet, an error inside one of our autograd functions). A crash ends the run; a fallback
costs at most the speed or memory that component saves.

Rules every guarded call site follows:
- Forward: the fast path runs inside `try`. On an `Exception`, `fallback(component, exc)`
  records the component in `DISABLED` and prints one line, the first time only:
      [solution] fallback: <component>: <exception type>: <message>
  and the same call then runs through the original code path, on the same inputs, so the step
  completes with the original path's values. The original call is made after the `except`
  block, when the failed attempt's traceback (and whatever it kept alive) is gone. Every later
  call checks `DISABLED` first and goes to the original path directly: a failing kernel is not
  retried (a failed compilation alone would cost seconds per attempt).
- Backward (autograd functions): a step cannot be re-run from inside its backward, so each
  backward that launches a kernel of ours computes the same gradient with PyTorch ops from what
  its forward saved, or for the 4-bit weights with bitsandbytes' own dequantization, which is
  bit-identical to ours. Backward passes made of PyTorch ops only (the fused head's, LoRA's
  apart from the weight it asks the layer group for) are left alone: the original path would
  run the same ops.
- CUDA out-of-memory is never absorbed (`fallback` returns False and the caller re-raises): our
  paths need less memory than the original ones, so the original would run out too. Neither is
  PyTorch's own control flow (the early stop of a checkpoint's recomputation, `is_control_flow`).
  KeyboardInterrupt and SystemExit are not `Exception`s and are never caught.
- The self-checks that decide at set-up whether a patch is installed keep their own handling.
- The happy path is unchanged: a `try` costs nothing until something raises and the `DISABLED`
  test is a dict lookup; no extra kernel, no host sync.

`DISABLED` (component -> "<exception type>: <message>" of its first failure) is what the run's
summary reads; the names are those of `COMPONENTS`.
"""

import contextlib

import torch

DISABLED = {}

                                                              
COMPONENTS = {
    "fused_head": "fused causal-LM forward and TRL compute_loss (patches.py, fused_head.py) -> "
                  "transformers' forward and TRL's compute_loss (full logits)",
    "unpad": "padding removal (unpad.py, patches.py) -> the fused head on the padded batch",
    "varlen_attention": "all rows in one attention launch per KV head -> rows_attention",
    "rows_attention": "one attention launch per row -> F.scaled_dot_product_attention per row",
    "rmsnorm": "RMSNorm kernel -> transformers' Qwen2RMSNorm.forward",
    "addnorm": "residual add folded into the norm -> the plain add, then the norm",
    "rope": "RoPE kernel -> transformers' apply_rotary_pos_emb",
    "swiglu": "the example's SwiGLU kernels and our in-place backward -> act_fn(gate) * up",
    "qgroup": "grouped NF4 dequantization -> bitsandbytes' Linear4bit.forward / dequantize_4bit",
    "lora": "the LoRA projection node -> PEFT's Linear4bit.forward",
    "fused_proj": "q|k|v as one GEMM node (fused_proj.py) -> transformers' attention forward",
    "fused_mlp": "opt-in gate|up as one GEMM node -> the MLP forward installed before it",
    "nf4_dequant": "per-projection NF4 dequantization -> bitsandbytes' kernel",
    "qlinear": "opt-in fused 4-bit linear -> bitsandbytes' Linear4bit.forward",
    "mlp_recompute": "opt-in MLP recomputation -> the MLP forward installed before it",
    "mlp_save_up": "opt-in MLP keeping only up and its input -> the MLP forward before it",
    "fp16_embed": "fp16 storage of the tied embedding / lm_head weight -> PEFT's fp32 copy",
    "bookkeeping": "trainer bookkeeping without module-tree walks -> transformers' methods",
    "defer": "metrics fetched when logged -> fetched right after each forward",
    "triton_build": "Triton C helpers built at -O0 -> Triton's own build",
    "warmup": "kernels compiled in a thread during start-up -> compiled at first launch",
    "startup.<name>": "one start-up patch of startup.py -> left out",
}


def _text(exc):
    try:
        return str(exc)
    except Exception:                                                             
        return "<message unavailable>"


def _one_line(exc, limit=500):
    text = " ".join(_text(exc).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def is_out_of_memory(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = _text(exc).lower()
    return "out of memory" in text or "out_of_memory" in text


def is_control_flow(exc):
    """PyTorch's own control flow passing through a fast path, not a failure: the early stop of a
    non-reentrant checkpoint's recomputation (`torch.utils.checkpoint._StopRecomputationError`,
    raised from a saved-tensor hook inside the recomputed forward once every tensor it needs is
    back). Absorbing it would break the checkpoint ("early_stop is set")."""
    return type(exc).__name__ == "_StopRecomputationError"


def fallback(component, exc):
    """For `except Exception as exc:` around a fast path. Returns False when `exc` must
    propagate (out of memory, or PyTorch's own control flow: re-raise it). Otherwise turns
    `component` off for the rest of the process, printing one `[solution] fallback:` line the
    first time, and returns True: the caller then runs the original code path."""
    if not isinstance(exc, Exception) or is_out_of_memory(exc) or is_control_flow(exc):
        return False
    if component not in DISABLED:
        reason = f"{type(exc).__name__}: {_one_line(exc)}"
        DISABLED[component] = reason
        try:
            print(f"[solution] fallback: {component}: {reason}", flush=True)
        except Exception:                                                              
            pass
    return True


@contextlib.contextmanager
def installing(component):
    """Around one patch installation in `solution.apply()`: an unexpected error leaves that
    component off (one fallback line) instead of failing `soup train`."""
    try:
        yield
    except Exception as exc:                                 
        if not fallback(component, exc):
            raise
