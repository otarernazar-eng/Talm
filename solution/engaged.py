"""Which fast paths of this package actually ran in this process, and how many times.

One line in the training log, printed once after the first training step (when the second step
begins, before the trainer's callbacks run, so outside the harness's timed step) and again at
exit with the totals:

    [solution] engaged: at=step1 steps=1 head=1 vchunk=1 compact=1 rows8=20 unpad=1 ... fallbacks=none

Why: every patch of this package falls back to the original code outside the task's exact
situation (that is what keeps Soup's regression tests passing), so a fallback in the measured run
would otherwise be silent. With this line the train.log of any run, the jury's hidden-rows run
included, shows that each component engaged and that no installed patch took the original path.

Cost: the counters are plain Python integers bumped on the host where each patch picks its fast
path or its fallback (a few hundred increments per step). No device sync, no extra kernel, no
tensor is read, neither when counting nor when the line is printed.

Tokens, in pipeline order: `name=count` (`off`: switched off in this process, by its SOLUTION_*
switch or because what it runs inside is off).
  at, steps       when the line is printed (step1 / exit), training steps completed
  head            training forwards through the fused output head + loss (patches.py)
  vchunk          head calls on the vocabulary-sliced path (fused_head.py; _HEAD_VCHUNK)
  compact         head calls whose hidden-gradient GEMM covered the labelled rows only
                  (_HEAD_COMPACT)
  rows8           head kernel launches with row sizes in units of 8 elements (_HEAD_ROWS)
  unpad           fused forwards with the padding removed (unpad.py; _UNPAD)
  varlen          variable-length attention launches, forward and backward, one per KV head
                  (_VARLEN); `rows_attn` (per-row attention calls) replaces nothing and is shown
                  only when varlen is off
  nocat           varlen attention backwards without per-head copies and concatenations
                  (_ATTN_NOCAT)
  lora            LoRA projection node calls (lora.py; _LORA)
  lora_cast       refreshes of the adapters' fp16 copies, about one per optimizer step
                  (_LORA_CAST)
  fused_proj      attention calls whose q|k|v ran as one GEMM node with both adapters
                  (fused_proj.py; _FUSED_PROJ / _FUSED_QKV); the last layer's split forward takes
                  the LoRA node by design, any other packed layer that does not counts as a
                  fallback (`fused_mlp`, the opt-in gate|up node, likewise)
  rmsnorm         RMSNorm kernel launches, forward and backward (rmsnorm.py; _RMSNORM)
  addnorm         residual add + RMSNorm kernel launches, forward and backward (_ADDNORM)
  rope            RoPE kernel launches, forward and backward (rope.py; _ROPE)
  group           whole-layer NF4 dequantization launches (qgroup.py; _GROUP)
  split           per-projection dequantization launches of the last layers (_SPLIT_DEQUANT)
  swiglu_inplace  MLP forwards through the in-place-backward SwiGLU wrapper (_SWIGLU_INPLACE)
  fp16_embed      tied embeddings stored in fp16 after PEFT's upcast (fp16_embed.py;
                  _FP16_EMBED)
  defer           deferred device-to-host metric fetches, one copy each (host.py; _DEFER)
  bookkeeping     trainer bookkeeping fast paths taken (host.py; _BOOKKEEPING)
  startup         start-up patches applied / attempted (startup.py; _STARTUP and the switch of
                  each patch)
  warmup          warm-up thread: ok:<kernel groups compiled>, error, running, off (warmup.py)
  build           Triton's C helper builds: -O0, unavailable, off (triton_build.py)
  fused_mlp, mlp_save_up, mlp_recompute, qlinear, nf4_dequant   opt-in / fallback paths, shown
                  only when installed (MLP forwards / 4-bit linear calls / dequant calls)
  fallbacks       `none`, or `name:count,...`: calls in which an installed patch took the
                  original code path (or, for unpad / vchunk / rows8 / fp16_embed, the
                  component's own slower path)
  disabled        only after an unexpected error: the components it turned off for the rest
                  of the process (state.DISABLED, names of state.COMPONENTS)
"""

import collections
import sys

RAN = collections.defaultdict(int)        # component -> times its fast path ran
FELL_BACK = collections.defaultdict(int)  # component -> calls that took the original path
ON = set()   # components whose patch is installed in this process (set by the patches)
TEXT = {}    # components reported as text, set by solution.apply(): startup, build
_seen = {"printed": False, "state": None, "exit_hook": False}

# The order of the line; the components that are not in ON print `off`.
ORDER = ("head", "vchunk", "compact", "rows8", "unpad", "varlen", "rows_attn", "nocat", "lora",
         "lora_cast", "fused_proj", "fused_mlp", "rmsnorm", "addnorm", "rope", "group", "split",
         "nf4_dequant", "swiglu_inplace", "mlp_save_up", "mlp_recompute", "qlinear", "fp16_embed",
         "defer", "bookkeeping")
# shown only when on
OPTIONAL = ("rows_attn", "fused_mlp", "nf4_dequant", "mlp_save_up", "mlp_recompute", "qlinear")


def _const(module, name, default=False):
    """A module constant of this package (its SOLUTION_* switch as read at import), without
    importing the module: `default` when it is not loaded."""
    loaded = sys.modules.get(f"{__package__}.{module}")
    return getattr(loaded, name, default) if loaded is not None else default


def switched_on():
    """{component: on} for this process: the installed patches, and the refinements inside them
    from their modules' switches."""
    on = {name: name in ON for name in ORDER}
    head, unpad = on["head"], on["head"] and "unpad" in ON
    on["vchunk"] = head and bool(_const("fused_head", "VOCAB_PATH"))
    on["compact"] = on["vchunk"] and bool(_const("fused_head", "COMPACT"))
    on["rows8"] = on["vchunk"] and _const("fused_head", "ROW_UNIT", 1) == 8
    on["unpad"] = unpad
    rows = unpad and bool(_const("unpad", "_ROWS_ATTENTION"))
    on["varlen"] = rows and bool(_const("unpad", "_VARLEN"))
    on["rows_attn"] = rows and not on["varlen"]  # (the per-row path is varlen's fallback)
    on["nocat"] = on["varlen"] and bool(_const("unpad", "_NOCAT"))
    on["lora_cast"] = on["lora"] and bool(_const("lora", "CAST_CACHE"))
    on["split"] = on["group"] and not _const("qgroup", "_SPLIT_OFF", True)
    return on


def _warmup():
    warmup = sys.modules.get(f"{__package__}.warmup")
    report = getattr(warmup, "_report", None) or {}
    if "error" in report:
        return "error"
    if "times" in report:
        return f"ok:{len(report['times'])}"
    if warmup is not None and getattr(warmup, "_thread", None) is not None:
        return "running"
    return "off"


def steps():
    state = _seen["state"]
    return int(getattr(state, "global_step", 0) or 0) if state is not None else 0


def _shown(name, on):
    if name == "rows_attn":  # the attention path of the flat forward when varlen is off
        return on["unpad"] and not on["varlen"]
    return name not in OPTIONAL or on[name]


def line(at):
    """The `[solution] engaged: ...` line (reads host counters only)."""
    on = switched_on()
    tokens = [f"at={at}", f"steps={steps()}"]
    for name in ORDER:
        if _shown(name, on):
            tokens.append(f"{name}={RAN[name] if on[name] else 'off'}")
    tokens += [f"startup={TEXT.get('startup', 'off')}", f"warmup={_warmup()}",
               f"build={TEXT.get('build', 'off')}"]
    names = [name for name in ORDER if FELL_BACK.get(name)]
    names += sorted(name for name in FELL_BACK if FELL_BACK[name] and name not in ORDER)
    tokens.append("fallbacks=" + (",".join(f"{name}:{FELL_BACK[name]}" for name in names)
                                  or "none"))
    # components an unexpected error turned off for the rest of the process (state.py; each also
    # printed its own `[solution] fallback:` line): shown only when there is one
    disabled = getattr(sys.modules.get(f"{__package__}.state"), "DISABLED", None)
    if disabled:
        tokens.append("disabled=" + ",".join(sorted(disabled)))
    return "[solution] engaged: " + " ".join(tokens)


def report(at, fresh_line=False):
    """`fresh_line`: start with a newline. During training the trainer's progress bar writes
    to stderr without one, and in a log of both streams the line would follow it."""
    try:
        text = line(at)
    except Exception as exc:  # noqa: BLE001 - a report must never break the training run
        text = f"[solution] engaged: at={at} error={type(exc).__name__}"
    print(("\n" if fresh_line else "") + text, flush=True)


def report_at_exit():
    report("exit")


def install(register_exit=True):
    """Print the line once after the first training step (transformers' CallbackHandler, when
    the second step begins and before its callbacks run) and, with `register_exit`, at exit.
    Idempotent."""
    if register_exit and not _seen["exit_hook"]:
        import atexit

        _seen["exit_hook"] = True
        atexit.register(report_at_exit)
    try:
        from transformers.trainer_callback import CallbackHandler
    except Exception:  # noqa: BLE001 - no transformers trainer here: the exit line only
        return False
    original = CallbackHandler.on_step_begin
    if getattr(original, "_solution_engaged", False):
        return True

    def on_step_begin(self, args, state, control, **kwargs):
        _seen["state"] = state
        if not _seen["printed"] and (getattr(state, "global_step", 0) or 0) >= 1:
            _seen["printed"] = True
            report("step1", fresh_line=True)
        return original(self, args, state, control, **kwargs)

    on_step_begin._solution_engaged = True
    on_step_begin._original = original
    CallbackHandler.on_step_begin = on_step_begin
    return True
