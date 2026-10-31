"""Our solution: Triton kernels and patches that make `soup train` lighter and faster on a T4.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every patch is class-level, looked up at call
time, and falls back to the original code outside the task's exact situation (see
`patches.py`), so Soup's regression tests pass with it applied.

Kernels of ours:
- `fused_head.py`: fused output head + cross-entropy + entropy + accuracy + per-row losses in
  vocabulary slices with an online-softmax gradient accumulator, with no [B, T, V] logits tensor
  and no full fp16 copy of the frozen output weight (about -4.4 GiB of peak memory);
- `rmsnorm.py`: fused RMSNorm forward/backward (one kernel) writing the autocast dtype, which
  removes the fp32 elementwise chain and the casts around every 4-bit projection and halves the
  stored activations downstream; the same kernel also takes the residual add that produces its
  input (`AddRMSNorm`, every residual add of the decoder; SOLUTION_DISABLE_ADDNORM=1);
- `rope.py`: fused rotary embedding of q and k, forward and backward (one kernel);
- `qgroup.py`: one NF4 dequantization launch per decoder layer and pass for its seven 4-bit
  projections (one per projection in the passes of the last layers, at the memory peak); on
  that packed buffer `fused_proj.py` runs q|k|v as one cuBLAS GEMM forward and one backward
  (SOLUTION_DISABLE_FUSED_PROJ=1; gate|up likewise, transposed, is opt-in with
  SOLUTION_FUSED_MLP=1: measured slower on the T4);
- `qlinear.py` (opt-in): the 4-bit linear layer as one dequant launch plus one cuBLAS GEMM;
- `nf4_dequant.py`: NF4 -> fp16 unpacking as a drop-in for bitsandbytes' dequant kernel, for
  projections the per-layer group does not pack (none in this task: installed only on need).
Plus `unpad.py`: padding removed inside each batch (flat forward, causal attention of all rows
in one variable-length launch of PyTorch's memory-efficient kernel per KV head, the KV heads
broadcast, not repeated, for the query heads that share them; SOLUTION_DISABLE_VARLEN=1 for one
launch per row); `lora.py`: the adapted q/v projections as one autograd node, bit-identical to
PEFT, that keeps only the layer input (dropout replayed in the backward;
SOLUTION_DISABLE_LORA=1); and `host.py`: host-side cost of the step cut with the same results
(trainer bookkeeping without module-tree walks, SOLUTION_DISABLE_BOOKKEEPING=1 switches it off;
per-step metrics fetched when they are logged instead of synced between forward and backward,
SOLUTION_DISABLE_DEFER=1).

`fp16_embed.py`: the frozen tied embedding / lm_head weight, which PEFT's
`prepare_model_for_kbit_training` upcasts from fp16 to fp32, stored back in fp16 (the same
values bit for bit; lookups still return fp32; the fused head takes its fp16 slices without a
copy): -445 MiB at the peak, SOLUTION_DISABLE_FP16_EMBED=1 switches it off.

Kept from the starter kit: `swiglu.py`, the example fused SwiGLU (does not count as ours);
`swiglu_inplace.py` is our wrapper around its kernels that writes the backward over gate/up.

Outside the training loop, `startup.py` removes work the run pays for and never uses (imports
of unused optional dependencies, a tokenizer pickled only to name a cache entry, a dtype
conversion done on the CPU, garbage-collector rescans of the import-time objects, a repeated
nvidia-smi query, a per-weight NF4 table upload, repeated chat-template renders of a row);
each patch has its own SOLUTION_DISABLE_* switch (gc.freeze: SOLUTION_DISABLE_GC_FREEZE), and
SOLUTION_DISABLE_STARTUP=1 turns them all off. (The wall-clock timeline these numbers come from
is a development tool outside the package, tools/timeline.py.)

Switches, for ablations (environment variables; with none set, the submitted solution runs). A
part that is on by default goes off with SOLUTION_DISABLE_<NAME>=1, an opt-in experiment goes on
with SOLUTION_<NAME>=1. Each is read in one place: by apply() or a helper it calls when it
removes a patch, as a constant of the module that has the code path when it picks a path
inside a patch (read when apply() imports that module); SOLUTION_VARLEN, a test knob, at each
attention call, and SOLUTION_DISABLE_SWIGLU_INPLACE also by fused_proj.py, whose opt-in MLP
node has its own copy of that backward.
- Parts: SOLUTION_DISABLE_FUSED_HEAD (the padding removal, the per-row attention and host.py go
  with it) / _UNPAD / _RMSNORM (the add + norm fusion goes with it) / _ADDNORM / _ROPE / _GROUP
  (the per-layer dequant; the LoRA node and fused_proj, which need its packed base, go with
  it) / _FUSED_PROJ / _FUSED_QKV (the q|k|v GEMM of fused_proj alone) / _LORA /
  _FP16_EMBED / _SWIGLU_INPLACE / _DEFER / _BOOKKEEPING / _DEQUANT (bitsandbytes' own kernel for
  a layer the group does not pack) = 1, and _STARTUP with every start-up patch: _GC_FREEZE and
  the rows of the `STARTUP` table below.
- Memory refinements: SOLUTION_DISABLE_HEAD_VCHUNK (token-chunked head with a full fp16 weight
  copy) / _ROWS_ATTN (repeat_kv copies + plain SDPA per row) / _SPLIT_DEQUANT (whole-layer
  dequant also on the boundary layers) = 1.
- GPU-time refinements: SOLUTION_DISABLE_HEAD_COMPACT (the head's gradient GEMM over all rows,
  not only the labelled ones) / _HEAD_ROWS (the head kernel's rows in element units, not
  vectorized) / _VARLEN (one attention launch per row) / _ATTN_NOCAT (per-KV-head gradient
  copies and concatenations in the attention backward) / _ROPE_TOKEN_MAJOR (RoPE gradients
  head-major, copied back by autograd) / _ROPE_ROWS (RoPE with element strides) / _LORA_CAST
  (the adapters' fp16 weights cast per projection and forward, not once per optimizer step) = 1.
- Opt-in: SOLUTION_MLP_RECOMPUTE=1 (`mlp_recompute.py`: about 1.2 GiB less peak memory for
  about +25% training-loop time; the per-layer dequant and the LoRA node go off with it);
  SOLUTION_MLP_SAVE_UP=1 (`mlp_save_up.py`: the MLP keeps up and its input and recomputes gate
  in the backward: -560 MiB of peak for +4% wall time on the T4); SOLUTION_FUSED_MLP=1
  (fused_proj.py's transposed gate|up GEMM: measured slower on the T4); SOLUTION_QLINEAR=1
  (`qlinear.py`; acts only with SOLUTION_DISABLE_GROUP=1, since the group takes every packed
  layer); SOLUTION_VARLEN=1 (the variable-length attention also on GPUs of compute capability
  8 or more, where SDPA would take flash attention: for the tests).
- Tuning: SOLUTION_HEAD_VCHUNK=N, vocabulary columns per slice of the head (default 8192).
- Start-up: SOLUTION_WARMUP_THREAD and SOLUTION_FAST_TRITON_BUILD, below.

Compilation inside the measured time: every run starts with an empty Triton cache, so each
kernel is compiled, gets a gcc-built C launcher and is loaded inside the measured wall time.
Each kernel compiles once (one specialization; RMSNorm and RoPE forward and backward are one
kernel each). The step's kernels compile in a thread of this very process, started by apply(),
while the imports run and Soup starts up and loads the model (`warmup.py`: same process, no
cache, nothing prebuilt, the compilation stays inside the measured wall time;
SOLUTION_WARMUP_THREAD=0 switches it off, `late` starts it after the imports of apply() for
comparison), and Triton's C helpers are built with gcc -O0 instead of -O3 (`triton_build.py`;
SOLUTION_FAST_TRITON_BUILD=0 switches it off).
`[solution]` lines in the training log give the wall time of the patches (a breakdown of
apply()) and of the first fused step, for the time budget outside the training loop, and one
`[solution] engaged: ...` line, after the first training step and again at exit, states for
every component whether its fast path ran in this process and how many times, and counts every
call in which an installed patch took the original code path (`engaged.py`; host counters only:
no sync, no kernel).
"""

import atexit
import os
import time

_t0 = time.perf_counter()


def _mark(label):
    # One write per line (print() writes the text and the newline separately): the warm-up
    # thread prints too, and two threads' lines must not interleave.
    print(f"[solution] {label} at {time.perf_counter() - _t0:.2f} s\n", end="", flush=True)


_lap_t = [_t0]
_exit_mark = []  # the "process exit" mark is registered (once per process)


def _in_chain(fn, marker):
    """Whether a wrapper carrying `marker` is in the patch chain of `fn`. Each of our wrappers
    keeps the function it wraps in `_original` or `_previous`, so a patch already installed
    under a later wrapper is found: apply() and each patch function install nothing twice."""
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        if getattr(fn, marker, False):
            return True
        fn = getattr(fn, "_previous", None) or getattr(fn, "_original", None)
    return False


def _lap(label):
    """Time since the previous lap: the breakdown of apply() in the training log."""
    now = time.perf_counter()
    print(f"[solution] apply: {label} {now - _lap_t[0]:.2f} s\n", end="", flush=True)
    _lap_t[0] = now


def _warmup_mode():
    """SOLUTION_WARMUP_THREAD: "0" off, "late" started after the imports of apply(), anything
    else (the default) started at the beginning of apply()."""
    return os.environ.get("SOLUTION_WARMUP_THREAD", "1")


# Outside the training loop (startup.py): work the run pays for and never uses. One row per
# patch: the switch that turns it off, the function in startup.py, a label for the log. The
# order matters: each import here would otherwise happen inside a later one.
STARTUP = (
    ("SOLUTION_DISABLE_LAZY_BOTO3", "import_accelerate_without_boto3", "accelerate w/o boto3"),
    ("SOLUTION_DISABLE_LAZY_COMPILE", "import_bitsandbytes", "bnb, deferred torch.compile"),
    ("SOLUTION_DISABLE_LAZY_SKLEARN", "import_without_sklearn", "deferred scikit-learn"),
    ("SOLUTION_DISABLE_LAZY_SCIPY", "import_losses_without_scipy", "deferred SciPy (losses)"),
    ("SOLUTION_DISABLE_SHARED_MAPPING", "import_trl", "trl, shared package mapping"),
    ("SOLUTION_DISABLE_GPU_CONVERT", "patch_load_conversion", "load conversion on the GPU"),
    ("SOLUTION_DISABLE_FAST_FINGERPRINT", "patch_tokenizer_fingerprint", "tokenizer digest"),
    ("SOLUTION_DISABLE_GPU_QUERY_ONCE", "share_gpu_query", "one nvidia-smi query"),
    ("SOLUTION_DISABLE_NF4_CODE_ONCE", "share_4bit_code", "one NF4 code table"),
    ("SOLUTION_DISABLE_RENDER_ONCE", "share_chat_renders", "one render per conversation"),
)


def _startup():
    from . import startup
    from .engaged import TEXT
    from .state import installing

    applied = attempted = 0
    for switch, name, label in STARTUP:
        if os.environ.get(switch) != "1":
            attempted += 1
            with installing(f"startup.{name}"):
                result = getattr(startup, name)()
                applied += bool(result)
                _mark(f"{label}: {result}")
    TEXT["startup"] = f"{applied}/{attempted}"


def _freeze_gc(when):
    """startup.py 8: the objects alive now leave the garbage collector's scans (gc.freeze), at
    the start of apply() (the harness's imports) and at its end (apply's imports)."""
    from . import startup

    _mark(f"gc freeze ({when}): {startup.freeze_gc()} objects")


def _patch_host():
    """Host-side cost of the step (host.py): metrics fetched when they are logged instead of
    synced between forward and backward; trainer bookkeeping without module-tree walks."""
    from . import host, patches
    from .state import installing

    if os.environ.get("SOLUTION_DISABLE_DEFER") != "1":
        with installing("defer"):
            host.defer_metrics()
            patches._host["defer"] = True
    if os.environ.get("SOLUTION_DISABLE_BOOKKEEPING") != "1":
        with installing("bookkeeping"):
            host.trainer_bookkeeping()
            patches._host["bookkeeping"] = True


def _compiler_startup(warm, kernels):
    """Triton's start-up (every run starts with an empty Triton cache): see
    triton_build.py and warmup.py. The warm-up thread starts here, before the imports
    of apply() and startup.py, to overlap them; it compiles the kernel groups `kernels` that
    apply() is about to install."""
    from .state import installing

    if os.environ.get("SOLUTION_FAST_TRITON_BUILD") != "0":
        # Triton's C launcher / cuda_utils builds at -O0 instead of -O3 (triton_build.py).
        with installing("triton_build"):
            from . import triton_build
            from .engaged import TEXT

            installed = triton_build.install()
            TEXT["build"] = triton_build.OPT_LEVEL if installed else "unavailable"
            _mark(f"Triton C helper builds at {triton_build.OPT_LEVEL}: "
                  f"{'on' if installed else 'OFF'}")
    if warm not in ("0", "late"):
        # The step's kernels compile in a thread of this process while Soup starts up and loads
        # the model (warmup.py; SOLUTION_WARMUP_THREAD=0 switches it off). Started first, so it
        # also overlaps the imports below.
        with installing("warmup"):
            from . import warmup

            warmup.start(kernels)


def apply():
    """Install everything. A later call installs what an earlier call's switches left out and
    nothing twice: the class-level patches wrap whatever method is in place, so each one is
    skipped when it is in the chain already (`_in_chain`), the warm-up thread starts once per
    process and the exit mark is registered once. Each installation step runs under
    `state.installing`: an unexpected error there leaves that component off (one
    `[solution] fallback:` line) instead of failing `soup train`; at run time every fast path
    degrades the same way (state.py)."""
    from .state import installing

    _lap_t[0] = time.perf_counter()
    startup_on = os.environ.get("SOLUTION_DISABLE_STARTUP") != "1"
    freeze = startup_on and os.environ.get("SOLUTION_DISABLE_GC_FREEZE") != "1"
    if freeze:
        _freeze_gc("start")
    warm = _warmup_mode()
    recompute = os.environ.get("SOLUTION_MLP_RECOMPUTE") == "1"
    group = os.environ.get("SOLUTION_DISABLE_GROUP") != "1" and not recompute
    # Our kernel groups the training step launches, by the switches read here once: the patches
    # below install them, and the warm-up thread compiles the same ones.
    kernels = {"rmsnorm": os.environ.get("SOLUTION_DISABLE_RMSNORM") != "1",
               "rope": os.environ.get("SOLUTION_DISABLE_ROPE") != "1",
               "group_dequant": group,
               "fused_head": os.environ.get("SOLUTION_DISABLE_FUSED_HEAD") != "1"}
    _compiler_startup(warm, kernels)
    if startup_on:
        with installing("startup"):
            _startup()
        _lap("startup.py")

    with installing("swiglu"):
        from transformers.models.qwen2 import modeling_qwen2

        from .swiglu import patch_qwen2_mlp

        if not _in_chain(modeling_qwen2.Qwen2MLP.forward, "_solution_example"):
            patch_qwen2_mlp()  # the example: works out of the box, not our own kernel
            modeling_qwen2.Qwen2MLP.forward._solution_example = True  # its wrapper, `_in_chain`
        _lap("swiglu patch (imports triton, transformers' qwen2)")

    if os.environ.get("SOLUTION_DISABLE_SWIGLU_INPLACE") != "1":
        with installing("swiglu"):
            from .swiglu_inplace import patch_qwen2_mlp_inplace

            patch_qwen2_mlp_inplace()  # the example's kernels, its backward written over gate/up

    if os.environ.get("SOLUTION_MLP_SAVE_UP") == "1":
        # Opt-in: the MLP keeps up and its input, gate is recomputed in the backward from the
        # layer's dequantized weight (one extra GEMM per layer; mlp_save_up.py).
        with installing("mlp_save_up"):
            from .mlp_save_up import patch_qwen2_mlp_save_up

            patch_qwen2_mlp_save_up()

    if recompute:
        # Opt-in: about -1.2 GiB at the peak for about +25% training-loop time (measured);
        # the scoring is a log ratio to the best team on each axis, so speed is the default.
        with installing("mlp_recompute"):
            from .mlp_recompute import patch_qwen2_mlp_recompute

            patch_qwen2_mlp_recompute()  # on top of the example: saves only the MLP input

    if kernels["rmsnorm"]:
        with installing("rmsnorm"):
            from .rmsnorm import patch_qwen2_rmsnorm

            patch_qwen2_rmsnorm()
        if os.environ.get("SOLUTION_DISABLE_ADDNORM") != "1":
            # every residual add inside the kernel of the norm that reads it (installed before
            # the per-layer dequant wrapper below, which wraps this layer forward)
            with installing("addnorm"):
                from .rmsnorm import patch_qwen2_add_norm

                patch_qwen2_add_norm()

    if kernels["rope"]:
        with installing("rope"):
            from .rope import patch_qwen2_rope

            patch_qwen2_rope()
    _lap("rmsnorm + rope patches")

    fused_linear = False
    if os.environ.get("SOLUTION_QLINEAR") == "1":
        # Opt-in experiment: measured neutral on the T4 (the step is not bound by bitsandbytes'
        # Python dispatch); the bit-exact dequant drop-in below is the default.
        with installing("qlinear"):
            from .qlinear import patch_bnb_linear4bit

            fused_linear = patch_bnb_linear4bit()
            _mark(f"fused 4-bit linear patch {'on' if fused_linear else 'OFF'}")

    dequant = not fused_linear and os.environ.get("SOLUTION_DISABLE_DEQUANT") != "1"
    if dequant and not group:
        # bitsandbytes' own dispatch with our dequant kernel for every 4-bit projection.
        with installing("nf4_dequant"):
            from .nf4_dequant import patch_bnb_dequant

            patched = patch_bnb_dequant()
            _lap(f"nf4 dequant patch + self-check ({'on' if patched else 'OFF'})")
            _mark(f"nf4 dequant patch {'on' if patched else 'OFF'}")

    if group:
        # One dequant launch per layer and pass (7 projections at once) instead of 389 per step.
        # Our per-projection dequant (nf4_dequant.py) is installed only if a layer cannot be
        # packed (qgroup.setup_layer); in this task none is, so its kernel is never compiled.
        with installing("qgroup"):
            from . import qgroup

            qgroup.FALLBACK_DEQUANT = dequant
            qgroup.patch_group_dequant()
            _lap("group dequant patch (imports bitsandbytes)")

        if os.environ.get("SOLUTION_DISABLE_LORA") != "1":
            # The packed q/v projections and their adapter as one autograd node: PEFT's ops,
            # bit-identical, but only the layer input kept for the backward (the dropout is
            # replayed there); PEFT's path everywhere else.
            with installing("lora"):
                from .lora import patch_peft_lora

                patch_peft_lora()

        if os.environ.get("SOLUTION_DISABLE_FUSED_PROJ") != "1":
            # q|k|v and gate|up, adjacent in the packed buffer, as one GEMM each forward and
            # backward (fused_proj.py); per call, the previous path outside that situation.
            if os.environ.get("SOLUTION_DISABLE_FUSED_QKV") != "1":
                with installing("fused_proj"):
                    from .fused_proj import patch_qwen2_attention

                    patch_qwen2_attention()
            if os.environ.get("SOLUTION_FUSED_MLP") == "1":
                # opt-in: the transposed gate|up GEMM measured slower on the T4 (fused_proj.py)
                with installing("fused_mlp"):
                    from .fused_proj import patch_qwen2_mlp

                    patch_qwen2_mlp()

    with installing("fp16_embed"):
        from . import fp16_embed

        if fp16_embed.enabled():  # SOLUTION_DISABLE_FP16_EMBED, also read there by the warm-up
            # The frozen tied embedding / lm_head weight back in fp16 right after PEFT's fp32
            # upcast (the same values, bitwise; lookups still return fp32): -445 MiB at the peak.
            _mark("fp16 tied embedding patch "
                  f"{'on' if fp16_embed.patch_prepare_model_for_kbit_training() else 'OFF'}")

    if kernels["fused_head"]:
        # (a compute_loss patch that fails to install leaves the fused forward off too: TRL's
        # own compute_loss needs the logits)
        with installing("fused_head"):
            from .patches import patch_qwen2_causal_lm, patch_sft_compute_loss
            from .unpad import patch_sdpa_rows

            patch_sdpa_rows()
            patch_qwen2_causal_lm()
            _lap("causal LM + sdpa patches")
            try:
                patch_sft_compute_loss()
            except ImportError:  # no trl in this process (some regression tests): nothing to do
                pass
            else:
                _patch_host()
            _lap("SFTTrainer.compute_loss + host patches (imports trl)")
    if warm == "late":
        # For comparison: the same warm-up thread started after the imports above.
        with installing("warmup"):
            from . import warmup

            warmup.start(kernels)
    if freeze:
        _freeze_gc("end")
    _mark("apply done")
    if not _exit_mark:  # once per process, however often apply() runs (the tests call it a lot)
        _exit_mark.append(True)
        atexit.register(_mark, "process exit")
    from . import engaged

    # `[solution] engaged: ...`: which fast paths ran and how often (engaged.py), after the
    # first training step and at exit (registered after the mark above, so printed before it).
    engaged.install()
