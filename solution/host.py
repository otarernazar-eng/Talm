"""Host-side cost of the training step, cut without changing any value the run produces.

After the fused kernels, the grouped dequantization and unpadding, part of every step is bound
by the host: the T4 waits for Python. tools/gpu_idle.py on a Kaggle T4 (green-5 code) puts the
GPU idle time at about 36 ms of a 440 ms step. The fixes here are exact; measured in one
Kaggle session (profiled steps 4-7): GPU idle 47.7 -> 24.8 ms per step (between the step's
first and last kernel 38.0 -> 19.2, in the logging and data loading around it 9.7 -> 5.6),
host syncs 22 -> 12 per loop iteration (16 -> 4 between forward and optimizer step). All of
it is part of the wall time of `soup train`, which is what is scored.

1. Trainer bookkeeping that walks the whole module tree (about 1 000 modules and 700
   parameters) after the backward, while the GPU has nothing queued (`trainer_bookkeeping`;
   measured: floating_point_ops 5.5 ms, clip_grad_norm_ 4.1, model.train() 3.5, zero_grad
   2.6 ms per step):
   - `Trainer.floating_point_ops` recounts the parameters for `total_flos`: the count cannot
     change during training, it is taken once per model (same `total_flos`);
   - `training_step` calls `model.train()` on a model that is already in training mode: the
     recursion is skipped when every module already has `training == True` (checked on a flat
     list of the modules, so a submodule switched to eval still gets switched back);
   - `model.zero_grad()` and `clip_grad_norm_(model.parameters())` iterate `parameters()`, a
     recursive generator: they get the same parameters, in the same order, from a flat list
     built once (so the gradient norm is summed in the same order).
   The model-level parts are installed on the trainer's model at its first fused training
   step, i.e. only in the situation of this task; anything else keeps the original methods.

2. Per-step metrics without host syncs between the forward and the backward
   (`defer_metrics`). TRL's `entropy`, `num_tokens` and `mean_token_accuracy` (computed by
   `patches._log_trl_metrics` from the fused step) and Soup's rewind row losses
   (`soup_cli.trainer.rewind_hf.row_losses_from_token_losses`) ended in 12 `.item()` /
   `float()` / `int()` calls right after the forward (of the 16 host syncs in a step), each
   draining the GPU, so the backward started from an idle GPU. The values now stay on the GPU
   until they are first read, in `SFTTrainer.log` (every logging step, outside the timed
   step, and at the end of training), where all pending ones are fetched with one copy. They
   are the same tensors' values; TRL's metric dict gets the same keys in the same order;
   Soup's rewind recorder pops its row ids and numbers its micro-batches when it did before
   and writes the same records, in the same order, to the same file.

Graceful degradation (state.py): an unexpected error in the cached FLOP count turns
"bookkeeping" off (transformers' own methods from then on; clipping too); an error while the
pending metrics are fetched, or in the lazy rewind rows, turns "defer" off (metrics fetched
right after each forward, Soup's own rewind rows) instead of failing the run. Not guarded:
the clipping call itself (the same accelerate call on a list instead of a generator; a retry
after a partial unscale would raise in the GradScaler) and the deferred `on_micro_batch` (it
pops Soup's row ids, which a retry would pop twice; Soup's recorder catches its errors).
"""

import collections

import torch

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

_metric_queue = []  # [(trainer, mode, entropy, tokens or None, accuracy, labelled)] GPU scalars
_rewind_queue = collections.deque()  # [(state, step, micro, rows, LazyRows, LazyRows)]
_fused_token_loss = [None]  # token_loss of the last fused step (the only one made lazy)


# --- metrics fetched when they are first read (2) -------------------------------------------

def queue_trl_metrics(trainer, mode, entropy, tokens, accuracy, labelled):
    """`entropy` (fp32), `tokens` (int64, train only, else None), `accuracy` = correct/labelled
    (fp32) and `labelled` (int64): 0-d GPU tensors, as TRL computes them before `.item()`."""
    _metric_queue.append((trainer, mode, entropy, tokens, accuracy, labelled))


def _fetch(tensors):
    """Python numbers of 0-d / 1-d tensors, with one device-to-host copy for all of them.
    float64 holds every fp32 and every int below 2**53 exactly."""
    if not tensors:
        return []
    RAN["defer"] += 1
    flat = torch.cat([t.reshape(-1).to(torch.float64) for t in tensors])
    return flat.tolist()


def resolve_metrics():
    """Apply the pending metrics to their trainers exactly as TRL's compute_loss would have."""
    if not _metric_queue:
        return
    entries = list(_metric_queue)
    _metric_queue.clear()
    tensors = []
    for _, _, entropy, tokens, accuracy, labelled in entries:
        tensors += [entropy, accuracy, labelled] + ([tokens] if tokens is not None else [])
    values = iter(_fetch(tensors))
    for trainer, mode, _, tokens, _, _ in entries:
        entropy, accuracy, labelled = next(values), next(values), next(values)
        metrics = trainer._metrics
        metrics[mode]["entropy"].append(entropy)
        if tokens is not None:
            trainer._total_train_tokens += int(next(values))
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
        metrics[mode]["mean_token_accuracy"].append(accuracy if labelled > 0 else 0.0)


class LazyRows:
    """Per-row numbers still on the GPU (Soup's rewind `row_loss` / `row_tokens`). `len()` is
    free; iterating fetches the values (flush_rewind fetches all pending ones in one copy)."""

    __slots__ = ("tensor", "cast", "values")

    def __init__(self, tensor, cast):
        self.tensor, self.cast, self.values = tensor, cast, None

    def __len__(self):
        return int(self.tensor.shape[0])

    def __iter__(self):
        if self.values is None:
            _materialize([self])
        return iter(self.values)


def _materialize(lazies):
    todo = [lazy for lazy in lazies if lazy.values is None]
    values = _fetch([lazy.tensor for lazy in todo])
    start = 0
    for lazy in todo:
        n = len(lazy)
        lazy.values = [lazy.cast(v) for v in values[start:start + n]]
        start += n


def mark_fused_token_loss(token_loss):
    _fused_token_loss[0] = token_loss


def _patch_rewind():
    try:
        from soup_cli.trainer import rewind_hf
    except ImportError:
        return
    original_rows = rewind_hf.row_losses_from_token_losses
    if getattr(original_rows, "_solution_lazy", False):
        return
    state_cls = rewind_hf.RewindState
    original_micro = state_cls.on_micro_batch

    def row_losses_from_token_losses(token_loss, labels):
        """Soup's per-row mean token loss and token count (same ops), left on the GPU for the
        fused step's token losses."""
        if token_loss is _fused_token_loss[0] and "defer" not in DISABLED:
            try:
                with torch.no_grad():
                    tokens = (labels[:, 1:] != -100).sum(dim=1)
                    totals = token_loss.float().sum(dim=1).to(tokens.device)
                    means = totals / tokens.clamp(min=1).to(totals.dtype)
                return LazyRows(means, float), LazyRows(tokens, int)
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("defer", exc):
                    raise
                FELL_BACK["defer"] += 1
        return original_rows(token_loss, labels)

    def on_micro_batch(self, *, step, row_loss, row_tokens):
        """RewindState.on_micro_batch (soup-cli 0.77.0) with the sink write deferred: the row
        ids are popped and the micro index advanced now, the record is written by
        `flush_rewind` with the same content."""
        lazy = isinstance(row_loss, LazyRows) and isinstance(row_tokens, LazyRows)
        if not lazy or not all(hasattr(self, name) for name in
                               ("_pending", "_last_step", "_micro", "_sink", "dropped")):
            flush_rewind()  # keep the records in order
            return original_micro(self, step=step, row_loss=row_loss, row_tokens=row_tokens)
        n = len(row_loss)
        if len(self._pending) < n:
            self.dropped += 1
            return
        rows = [self._pending.popleft() for _ in range(n)]
        if step != self._last_step:
            self._micro = 0
            self._last_step = step
        else:
            self._micro += 1
        _rewind_queue.append((self, int(step), self._micro, rows, row_loss, row_tokens))

    row_losses_from_token_losses._solution_lazy = True
    row_losses_from_token_losses._original = original_rows
    on_micro_batch._original = original_micro
    rewind_hf.row_losses_from_token_losses = row_losses_from_token_losses
    state_cls.on_micro_batch = on_micro_batch


def flush_rewind():
    """Write the deferred rewind records (one device-to-host copy for all of them)."""
    if not _rewind_queue:
        return
    entries = list(_rewind_queue)
    _rewind_queue.clear()
    _materialize([lazy for entry in entries for lazy in entry[4:]])
    for state, step, micro, rows, row_loss, row_tokens in entries:
        if getattr(state, "failed", False):
            continue
        try:
            state._sink.record_batch(step=step, micro=micro, rows=rows,
                                     row_loss=[float(x) for x in row_loss],
                                     row_tokens=[int(t) for t in row_tokens])
        except Exception as exc:  # noqa: BLE001 - as Soup's recorder: never raise into training
            state.fail(exc)


def flush_all():
    resolve_metrics()
    flush_rewind()
    # Soup's rewind recorder read this step's token losses before the log: the reference is not
    # needed any more and would otherwise keep the last step's tensor alive after the run.
    _fused_token_loss[0] = None


def _flush_guarded():
    """flush_all where a failure must not end the run (state.py): "defer" goes off."""
    try:
        flush_all()
    except Exception as exc:  # noqa: BLE001 - see state.py
        if not fallback("defer", exc):
            raise
        FELL_BACK["defer"] += 1


def defer_metrics():
    """Resolve pending metrics and rewind records in `SFTTrainer.log` (where TRL first reads
    them) and when `Trainer.train` returns or raises."""
    import transformers
    import trl

    ON.add("defer")
    _patch_rewind()
    log = trl.SFTTrainer.log
    if not getattr(log, "_solution_flush", False):
        def sft_log(self, logs, start_time=None):
            _flush_guarded()
            return log(self, logs, start_time)

        sft_log._solution_flush = True
        trl.SFTTrainer.log = sft_log
    train = transformers.Trainer.train
    if not getattr(train, "_solution_flush", False):
        def trainer_train(self, *args, **kwargs):
            try:
                result = train(self, *args, **kwargs)
            except BaseException:
                try:  # what was recorded before the failure still reaches its files
                    flush_all()
                except Exception:  # noqa: BLE001 - the training error is the one to report
                    pass
                raise
            _flush_guarded()  # normally empty: the final log of the run has fetched everything
            return result

        trainer_train._solution_flush = True
        transformers.Trainer.train = trainer_train


# --- trainer bookkeeping (1) ------------------------------------------------------------------

def install_model_fastpaths(model):
    """Instance-level `train` and `zero_grad` on the trainer's model, and the flat parameter
    list `Trainer._clip_grad_norm` uses (see the module docstring)."""
    if "_solution_params" in model.__dict__ or not isinstance(model, torch.nn.Module):
        return
    modules = list(model.modules())
    params = list(model.parameters())
    cls = type(model)

    def train(self, mode=True):
        if mode is True and all(m.training for m in modules):
            RAN["bookkeeping"] += 1
            return self  # every module is in training mode already: nothing would change
        if mode is True:
            FELL_BACK["bookkeeping"] += 1  # a module was in eval mode: the full recursion
        return cls.train(self, mode)

    def zero_grad(self, set_to_none=True):
        if set_to_none is not True or getattr(self, "_is_replica", False):
            FELL_BACK["bookkeeping"] += 1
            return cls.zero_grad(self, set_to_none)
        RAN["bookkeeping"] += 1
        for p in params:
            if p.grad is not None:
                p.grad = None

    import types

    model.__dict__["_solution_params"] = params
    model.train = types.MethodType(train, model)
    model.zero_grad = types.MethodType(zero_grad, model)


def trainer_bookkeeping():
    import transformers

    cls = transformers.Trainer
    flops = cls.floating_point_ops
    ON.add("bookkeeping")
    if getattr(flops, "_solution_cached", False):
        return

    def floating_point_ops(self, inputs):
        if "bookkeeping" not in DISABLED:
            try:
                model = self.model
                main_input = getattr(model, "main_input_name", "input_ids")
                if main_input in inputs and hasattr(model, "num_parameters"):
                    cache = self.__dict__.get("_solution_nparams")
                    if cache is None or cache[0] is not model:
                        cache = (model, model.num_parameters(exclude_embeddings=True))
                        self.__dict__["_solution_nparams"] = cache
                    RAN["bookkeeping"] += 1
                    return 6 * inputs[main_input].numel() * cache[1]
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("bookkeeping", exc):
                    raise
        FELL_BACK["bookkeeping"] += 1
        return flops(self, inputs)

    floating_point_ops._solution_cached = True
    floating_point_ops._original = flops
    cls.floating_point_ops = floating_point_ops

    clip = cls._clip_grad_norm

    def _clip_grad_norm(self, model):
        params = getattr(model, "__dict__", {}).get("_solution_params")
        if params is None or "bookkeeping" in DISABLED \
                or getattr(self.args, "fp16", False) and _sagemaker_mp():
            FELL_BACK["bookkeeping"] += 1
            return clip(self, model)
        RAN["bookkeeping"] += 1
        return self.accelerator.clip_grad_norm_(params, self.args.max_grad_norm)

    _clip_grad_norm._original = clip
    cls._clip_grad_norm = _clip_grad_norm


def _sagemaker_mp():
    try:
        from transformers.utils import is_sagemaker_mp_enabled

        return is_sagemaker_mp_enabled()
    except ImportError:
        return False
