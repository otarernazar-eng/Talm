"""Patches that route the training step through the fused head (solution/fused_head.py).

Two class-level patches, both looked up at call time and both falling back to the original
code whenever the situation is not the one of this task (no labels, eval mode, a model that is
not a Qwen2 causal LM on CUDA, a trainable or hooked head, gradient checkpointing on). The
conditions are type and shape checks only; the fake modules of Soup's regression tests simply
fail them and take the original path.

1. `Qwen2ForCausalLM.forward`: with labels, runs the decoder, computes the loss with
   `fused_head_loss` and returns `logits=None` plus what the trainer's metrics need, in the same
   `ChunkedStep` shape Soup's own chunked-loss mode uses, so Soup's rewind recorder
   (`soup_cli/trainer/rewind_hf.py`) reads the per-row losses from `_soup_chunked_step`.
2. `trl.SFTTrainer.compute_loss`: TRL reads `outputs.logits` for its `entropy` and
   `mean_token_accuracy` columns and would fail on `None`; this computes the same three
   columns (`entropy`, `num_tokens`, `mean_token_accuracy`) by TRL's formulas from the
   fused step instead.

Graceful degradation (state.py), for what those checks cannot foresee:
- an error while the batch is unpadded turns "unpad" off and the step continues on the padded
  batch; an error in the decoder forward over the unpadded batch does the same and runs that
  forward again, padded, still through the fused head;
- any other error in the fused forward turns "fused_head" off and the step runs again through
  transformers' own forward, and through TRL's own compute_loss when the trainer called it
  (full logits: TRL and Soup compute their metrics and rewind rows from them, as in the base).
Before a forward runs again, the CUDA generator goes back to its state before the failed
attempt (the LoRA dropout draws the same masks) and the 4-bit layer groups forget the
bookkeeping of the failed attempt (`_restart`).
"""

import os
import sys

import torch

from . import fused_head
from .fused_head import fused_head_loss
from .engaged import FELL_BACK, ON, RAN
from .rmsnorm import link_next_norms
from .state import DISABLED, fallback
from .unpad import is_right_padded, rows_context, unpad_batch

STEP_ATTR = "_soup_chunked_step"  # soup_cli.utils.stream_loss_chunk.STEP_ATTR

_pending_step = []  # the fused step of the last labelled forward, consumed by compute_loss
_first_forward = [True]  # cleared by the first fused forward (joins the warm-up thread)
_unpad = {"enabled": os.environ.get("SOLUTION_DISABLE_UNPAD") != "1", "checked": 0}
# Host-side options (solution/host.py), switched on by solution.apply().
_host = {"defer": False, "bookkeeping": False}
ORIGINAL_FORWARD = {}  # class -> original forward, for tests
# Non-empty while the patched SFTTrainer.compute_loss runs the fused forward: a "fused_head"
# failure in the forward is raised on to it, which runs the step again through TRL's own path.
_in_compute_loss = []


def _restart(device, rng_state):
    """Before a forward that failed part-way runs again: the CUDA generator back to `rng_state`
    (the same dropout masks as the failed attempt drew), and the 4-bit layer groups' pass
    bookkeeping reset (qgroup.reset_passes: the failed attempt counted backward uses that will
    never come, which would keep dequantized layers alive)."""
    torch.cuda.set_rng_state(rng_state, device)
    qgroup = sys.modules.get(f"{__package__}.qgroup")
    if qgroup is not None:
        qgroup.reset_passes()


def _chunked_step(token_entropy, token_loss, correct, labelled):
    """Soup's `ChunkedStep` when soup_cli is importable, a plain namespace otherwise."""
    try:
        from soup_cli.utils.stream_loss_chunk import ChunkedStep

        return ChunkedStep(token_entropy=token_entropy, token_loss=token_loss, correct=correct,
                           labelled=labelled)
    except ImportError:
        from types import SimpleNamespace

        return SimpleNamespace(token_entropy=token_entropy, token_loss=token_loss,
                               correct=correct, labelled=labelled)


def _head_is_fusable(lm):
    head = getattr(lm, "lm_head", None)
    weight = getattr(head, "weight", None)
    return (isinstance(weight, torch.Tensor) and weight.is_cuda and not weight.requires_grad
            and getattr(head, "bias", None) is None and weight.dim() == 2
            and "forward" not in lm.__dict__)  # an instance-level forward (accelerate hook)


def _gradient_checkpointing(lm):
    """Gradient checkpointing is on (TRL's default; Soup sets it off in this task): the decoder
    layers run again inside the backward, outside this forward, so the padding removal's row
    bounds (`rows_context`) would be gone in that replay. The flag `gradient_checkpointing_enable`
    sets on the Qwen2Model (with the layers' own flags; `is_gradient_checkpointing` walks every
    module, too slow for each step). Soup's `gradient_checkpointing: selective` tier leaves that
    flag off and gives every layer's attention an instance-level `forward` that runs it under
    torch.utils.checkpoint (soup_cli/utils/gradient_ckpt.py): an instance-level forward on a
    decoder layer or its attention (a wrapper we cannot see through) counts as well."""
    model = getattr(lm, "model", None)
    if getattr(model, "gradient_checkpointing", False):
        return True
    layers = getattr(model, "layers", None)
    for layer in layers._modules.values() if isinstance(layers, torch.nn.ModuleList) else ():
        attention = layer._modules.get("self_attn")
        if "forward" in layer.__dict__ or (attention is not None
                                           and "forward" in attention.__dict__):
            return True
    return False


def _fused_forward_applies(lm, labels, logits_to_keep, kwargs):
    return (labels is not None and isinstance(labels, torch.Tensor) and labels.dim() == 2
            and getattr(lm, "training", False) and isinstance(logits_to_keep, int)
            and logits_to_keep == 0 and "shift_labels" not in kwargs and _head_is_fusable(lm)
            and not _gradient_checkpointing(lm))


def patch_qwen2_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    from . import _in_chain

    cls = modeling_qwen2.Qwen2ForCausalLM
    if _in_chain(cls.forward, "_solution_fused_lm"):
        return
    original = ORIGINAL_FORWARD.setdefault(cls, cls.forward)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if "fused_head" not in DISABLED \
                and _fused_forward_applies(self, labels, logits_to_keep, kwargs):
            device = self.lm_head.weight.device
            rng_state = torch.cuda.get_rng_state(device)
            stage = []
            for _ in range(2):  # a failure of the unpadded decoder is retried once, padded
                try:
                    return _fused_forward(self, input_ids, attention_mask, position_ids,
                                          past_key_values, inputs_embeds, labels, use_cache,
                                          dict(kwargs), CausalLMOutputWithPast, stage)
                except Exception as exc:  # noqa: BLE001 - see state.py
                    component = "unpad" if stage[-1:] == ["unpadded"] else "fused_head"
                    if not fallback(component, exc) or (component == "fused_head"
                                                        and _in_compute_loss):
                        raise  # out of memory; or compute_loss runs the step again (TRL's)
                _restart(device, rng_state)
                if "fused_head" in DISABLED:
                    break
        FELL_BACK["head"] += 1
        return original(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                        logits_to_keep=logits_to_keep, **kwargs)

    forward._solution_fused_lm = True
    forward._original = original
    cls.forward = forward
    ON.add("head")
    if _unpad["enabled"]:
        ON.add("unpad")


def _join_warmup():
    """The compile thread of warmup.py, if it was started (never fatal: state.py)."""
    try:
        from . import warmup

        warmup.join()
    except Exception as exc:  # noqa: BLE001 - see state.py
        if not fallback("warmup", exc):
            raise


def _fused_forward(self, input_ids, attention_mask, position_ids, past_key_values,
                   inputs_embeds, labels, use_cache, kwargs, output_cls, stage):
    """The fused training forward of `Qwen2ForCausalLM` (see the module docstring). `stage`
    records how far it got: "unpadded" / "padded" while the decoder runs, then "head"."""
    stage.clear()
    if _first_forward:
        _join_warmup()
        _first_forward.clear()  # after the join: an interrupted join is retried next time
    kwargs.pop("return_dict", None)
    num_items = kwargs.get("num_items_in_batch")
    batch, seq = labels.shape
    # Causal shift as ForCausalLMLoss does: position t is scored against label t + 1; the
    # last position (and every pad) gets -100, so no shift crosses a row.
    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:].reshape(-1)

    unpad = ("unpad" not in DISABLED and _unpad["enabled"] and input_ids is not None
             and inputs_embeds is None and position_ids is None and past_key_values is None
             and isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2
             and tuple(attention_mask.shape) == (batch, seq)
             and tuple(input_ids.shape) == (batch, seq)
             and not getattr(self.model, "has_sliding_layers", True))
    counts = None  # [labelled among the gathered tokens, labelled overall], host ints
    if unpad:
        try:
            if fused_head.COMPACT:  # the head's labelled count, in the unpad sync (no extra one)
                idx, positions, bounds, lengths, counts = unpad_batch(
                    attention_mask, shift.to(attention_mask.device))
            else:
                idx, positions, bounds, lengths = unpad_batch(attention_mask)
            if _unpad["checked"] < 3:
                _unpad["checked"] += 1
                if not is_right_padded(attention_mask, lengths):
                    _unpad["enabled"] = False
                    unpad = False
        except Exception as exc:  # noqa: BLE001 - see state.py: the padded batch instead
            if not fallback("unpad", exc):
                raise
            unpad, counts = False, None
    RAN["head"] += 1
    if unpad:
        RAN["unpad"] += 1
    elif "unpad" in ON:
        FELL_BACK["unpad"] += 1
    link_next_norms(self.model)  # no-op unless the residual add + norm fusion is on
    n_labelled = None if counts is None else counts[0 if unpad else 1]
    if unpad:
        stage.append("unpadded")
        shift = shift.to(idx.device)[idx]
        # A prepared mask mapping skips `create_causal_mask` (the per-row attention below
        # needs no mask at all); only full-attention layers exist here (guard above).
        with rows_context(bounds, device=input_ids.device):
            outputs = self.model(input_ids=input_ids.reshape(-1)[idx][None],
                                 attention_mask={"full_attention": None},
                                 position_ids=positions, past_key_values=None,
                                 inputs_embeds=None, use_cache=False, **kwargs)
    else:
        stage.append("padded")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
    stage.append("head")
    hidden = outputs.last_hidden_state
    size = hidden.shape[-1]
    shift = shift.to(hidden.device)
    loss, token_loss, token_entropy, correct = fused_head_loss(
        hidden.reshape(-1, size), self.lm_head.weight, shift, num_items,
        n_labelled=n_labelled)
    if _unpad["checked"] <= 1:
        from . import _mark

        _mark(f"first fused forward done (unpad={unpad}, tokens={hidden.shape[1]})")
    if unpad:  # back to the padded layout TRL and Soup's rewind read
        full_loss = token_loss.new_zeros(batch * seq)
        full_entropy = token_entropy.new_zeros(batch * seq)
        full_loss[idx] = token_loss
        full_entropy[idx] = token_entropy
        token_loss, token_entropy = full_loss, full_entropy
    _pending_step.clear()
    step = _chunked_step(
        token_entropy=token_entropy.view(batch, seq), token_loss=token_loss.view(batch, seq),
        correct=correct.sum(), labelled=(shift >= 0).sum())
    if _host["defer"] and "defer" not in DISABLED:
        from .host import mark_fused_token_loss

        mark_fused_token_loss(step.token_loss)  # Soup's rewind rows of it stay on the GPU
    _pending_step.append(step)
    return output_cls(loss=loss, logits=None, past_key_values=outputs.past_key_values,
                      hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def _causal_lm_of(model):
    """The Qwen2ForCausalLM under PEFT / accelerate wrappers, or None."""
    from transformers.models.qwen2 import modeling_qwen2

    seen = 0
    while model is not None and seen < 8:
        if isinstance(model, modeling_qwen2.Qwen2ForCausalLM):
            return model
        get_base = getattr(model, "get_base_model", None)
        nxt = get_base() if callable(get_base) else None
        if nxt is None or nxt is model:
            nxt = getattr(model, "module", None) or getattr(model, "base_model", None) \
                or getattr(model, "model", None)
        if nxt is model:
            return None
        model, seen = nxt, seen + 1
    return None


def _log_trl_metrics(trainer, mode, inputs, step):
    """`entropy`, `num_tokens`, `mean_token_accuracy` exactly as trl.SFTTrainer.compute_loss
    (trl 0.29.1, sft_trainer.py:1238-1320) logs them from the full logits."""
    metrics = getattr(trainer, "_metrics", None)
    accelerator = getattr(trainer, "accelerator", None)
    if metrics is None or accelerator is None:
        return
    gather = accelerator.gather_for_metrics
    attention_mask = inputs.get("attention_mask")
    with torch.no_grad():
        if attention_mask is not None:
            entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
            local_tokens = attention_mask.sum()
        elif "position_ids" in inputs:
            entropy = torch.mean(step.token_entropy)
            local_tokens = torch.tensor(inputs["position_ids"].size(1),
                                        device=inputs["position_ids"].device)
        else:
            raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        if _host["defer"] and "defer" not in DISABLED:
            # The same tensors, fetched when TRL's log reads them (host.resolve_metrics): no
            # host sync between this forward and the backward.
            from .host import queue_trl_metrics

            entropy = gather(entropy).mean()
            tokens = gather(local_tokens).sum() if mode == "train" else None
            correct = gather(step.correct)
            total_sum = gather(step.labelled).sum()
            queue_trl_metrics(trainer, mode, entropy, tokens, correct.sum() / total_sum, total_sum)
            return
        metrics[mode]["entropy"].append(gather(entropy).mean().item())
        if mode == "train":
            trainer._total_train_tokens += gather(local_tokens).sum().item()
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
        correct = gather(step.correct)
        total = gather(step.labelled)
        total_sum = total.sum()
        accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
        metrics[mode]["mean_token_accuracy"].append(accuracy)


def _fused_model(trainer, model, inputs):
    """The Qwen2ForCausalLM whose fused forward this compute_loss call takes, or None (TRL's
    own path)."""
    lm = _causal_lm_of(model)
    labels = inputs.get("labels") if isinstance(inputs, dict) else None
    # A label smoother or a custom loss function makes the Trainer pop the labels before
    # the forward: the fused path would not run, so take TRL's original path instead.
    if lm is None or getattr(trainer.args, "use_liger_kernel", False) \
            or getattr(trainer, "label_smoother", None) is not None \
            or getattr(trainer, "compute_loss_func", None) is not None \
            or not _fused_forward_applies(lm, labels, 0, inputs):
        return None
    return lm


def _install_fastpaths(model):
    """host.install_model_fastpaths, never fatal (state.py: "bookkeeping" off instead)."""
    try:
        from .host import install_model_fastpaths

        install_model_fastpaths(model)  # flat module/parameter lists for train/zero_grad/clip
    except Exception as exc:  # noqa: BLE001 - see state.py
        if not fallback("bookkeeping", exc):
            raise


def patch_sft_compute_loss():
    import transformers
    import trl

    from . import _in_chain

    original = trl.SFTTrainer.compute_loss
    if _in_chain(original, "_solution_fused_loss"):
        return

    def fused_compute_loss(self, model, inputs, return_outputs, num_items_in_batch):
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        if _host["bookkeeping"] and "bookkeeping" not in DISABLED \
                and "_solution_params" not in getattr(model, "__dict__", {}):
            _install_fastpaths(model)
        _pending_step.clear()
        _in_compute_loss.append(True)
        try:
            loss, outputs = transformers.Trainer.compute_loss(
                self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        finally:
            _in_compute_loss.pop()
        step = _pending_step.pop() if _pending_step else None
        if step is None:
            # The fused path did not run after all: nothing was logged, and the logits exist.
            raise RuntimeError("fused head did not run although it was expected to")
        setattr(self, STEP_ATTR, step)
        _log_trl_metrics(self, mode, inputs, step)
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        lm = None
        if "fused_head" not in DISABLED:
            try:
                lm = _fused_model(self, model, inputs)
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("fused_head", exc):
                    raise
        if lm is not None:
            device = lm.lm_head.weight.device
            rng_state = torch.cuda.get_rng_state(device)
            try:
                return fused_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
            except Exception as exc:  # noqa: BLE001 - see state.py
                if not fallback("fused_head", exc):
                    raise
            # TRL's own compute_loss runs the step again: its forward returns the logits, from
            # which TRL logs its metrics and Soup's rewind recorder takes the row losses.
            _restart(device, rng_state)
            _pending_step.clear()
            if getattr(self, STEP_ATTR, None) is not None:
                setattr(self, STEP_ATTR, None)
        return original(self, model, inputs, return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch)

    compute_loss._solution_fused_loss = True
    compute_loss._original = original
    trl.SFTTrainer.compute_loss = compute_loss
