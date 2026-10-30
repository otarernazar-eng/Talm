"""Fused Triton Chunked Cross-Entropy, Shannon Entropy, and Token Accuracy Kernel.

Eliminates the `[batch, seq, 151936]` full-vocabulary logits and their fp32 upcast
copies at the output head during Qwen2.5-1.5B fine-tuning. The hidden states are
processed in chunks of `DEFAULT_CHUNK_SIZE` sequence positions (`B * 64 = 256` token
rows per chunk) reusing a single fp16 buffer (`[256, 151936]`, 74 MiB):

1. Chunk logits `hidden_chunk @ lm_head.weight.T` are written into the buffer.
2. `chunked_ce_fwd_bwd_kernel` (one program per token row) runs Pass 1: online
   log-sum-exp (`m`, `d`), an online weighted sum `s = sum(exp(x - m) * x)` for the
   exact Shannon entropy `H = lse - s / d`, the argmax for token accuracy and the
   next-token cross-entropy. Pass 2 overwrites the buffer in place with the
   UNNORMALIZED logit gradient `softmax(x) - onehot(y)` (values in [-1, 1], safe in
   fp16; rows without a label get zeros).
3. `d_logits @ lm_head.weight` gives the unnormalized hidden-state gradient of the
   chunk. The division by the token count and the multiplication by the GradScaler's
   loss scale happen together in fp32 in `backward`, so no fp16 underflow occurs.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
DEFAULT_CHUNK_SIZE = 64

_CE_TLS = threading.local()


def _is_ce_context_active() -> bool:
    return bool(getattr(_CE_TLS, "active", False))


@triton.jit
def chunked_ce_fwd_bwd_kernel(
    Logits_ptr,
    Target_ptr,
    Entropy_ptr,
    Loss_ptr,
    Correct_ptr,
    stride_logits_row,
    V,
    seq_len,
    chunk_start,
    chunk_len,
    ignore_index,
    write_grad_flag,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    # Position of this chunk row inside the full [B, S] outputs / targets.
    out_idx = (row // chunk_len) * seq_len + chunk_start + (row % chunk_len)
    row_ptr = Logits_ptr + row.to(tl.int64) * stride_logits_row
    label = tl.load(Target_ptr + out_idx).to(tl.int32)

    m_curr = tl.full((), -float("inf"), tl.float32)
    d_curr = tl.zeros((), tl.float32)
    s_curr = tl.zeros((), tl.float32)
    best_val = tl.full((), -float("inf"), tl.float32)
    best_idx = tl.zeros((), tl.int32)

    for start_v in range(0, V, BLOCK_V):
        offs_v = start_v + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        x = tl.load(row_ptr + offs_v, mask=mask_v, other=-float("inf")).to(tl.float32)

        tile_max = tl.max(x, axis=0)
        tile_arg = (start_v + tl.argmax(x, axis=0)).to(tl.int32)
        better = tile_max > best_val
        best_idx = tl.where(better, tile_arg, best_idx)
        best_val = tl.where(better, tile_max, best_val)

        m_next = tl.maximum(m_curr, tile_max)
        rescale = tl.exp(m_curr - m_next)
        exp_x = tl.exp(x - m_next)
        x_zero = tl.where(mask_v, x, 0.0)
        d_curr = d_curr * rescale + tl.sum(exp_x, axis=0)
        s_curr = s_curr * rescale + tl.sum(exp_x * x_zero, axis=0)
        m_curr = m_next

    lse = m_curr + tl.log(d_curr)
    entropy = lse - s_curr / d_curr
    tl.store(Entropy_ptr + out_idx, entropy.to(Entropy_ptr.dtype.element_ty))

    has_label = (label != ignore_index) & (label >= 0) & (label < V)
    safe_label = tl.where(has_label, label, 0)
    x_label = tl.load(row_ptr + safe_label).to(tl.float32)
    loss_val = tl.where(has_label, lse - x_label, 0.0)
    hit = tl.where(has_label & (best_idx == label), 1, 0)
    tl.store(Loss_ptr + out_idx, loss_val)
    tl.store(Correct_ptr + out_idx, hit.to(tl.int32))

    if write_grad_flag != 0:
        # 0/1 multiplier: rows without a label write an all-zero gradient.
        active = tl.where(has_label, 1.0, 0.0)
        for start_v in range(0, V, BLOCK_V):
            offs_v = start_v + tl.arange(0, BLOCK_V)
            mask_v = offs_v < V
            x = tl.load(row_ptr + offs_v, mask=mask_v, other=-float("inf")).to(tl.float32)
            prob = tl.exp(x - lse)
            onehot = tl.where(offs_v == label, 1.0, 0.0)
            grad = (prob - onehot) * active
            tl.store(row_ptr + offs_v, grad.to(Logits_ptr.dtype.element_ty), mask=mask_v)


class TritonChunkedCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        target: torch.Tensor,
        chunk_size: int,
        denominator: Any,
        ignore_index: int,
        with_grad: bool,
    ):
        B, S, H = hidden.shape
        V = head_weight.shape[0]
        device = hidden.device

        if torch.is_autocast_enabled():
            compute_dtype = torch.get_autocast_gpu_dtype()
        elif head_weight.dtype in (torch.float16, torch.bfloat16):
            compute_dtype = head_weight.dtype
        else:
            compute_dtype = hidden.dtype
        w_comp = head_weight if head_weight.dtype == compute_dtype else head_weight.to(compute_dtype)
        b_comp = None
        if head_bias is not None:
            b_comp = head_bias if head_bias.dtype == compute_dtype else head_bias.to(compute_dtype)

        if torch.is_tensor(denominator):
            denom_f32 = denominator.to(device=device, dtype=torch.float32).reshape(())
        else:
            denom_f32 = torch.tensor(float(denominator), dtype=torch.float32, device=device)

        target_c = target.contiguous()
        grad_hidden = torch.empty((B, S, H), dtype=compute_dtype, device=device) if with_grad else None
        token_entropy = torch.empty((B, S), dtype=compute_dtype, device=device)
        token_loss = torch.empty((B, S), dtype=torch.float32, device=device)
        correct_buf = torch.empty((B, S), dtype=torch.int32, device=device)

        max_rows = B * min(S, chunk_size)
        logits_buf = torch.empty((max_rows, V), dtype=compute_dtype, device=device)
        block_v = 4096 if V >= 4096 else triton.next_power_of_2(max(V, 16))

        for start in range(0, S, chunk_size):
            stop = min(start + chunk_size, S)
            cur_len = stop - start
            m_rows = B * cur_len
            if m_rows == 0:
                continue

            piece = hidden[:, start:stop].detach()
            if piece.dtype != compute_dtype:
                piece = piece.to(compute_dtype)
            piece_2d = piece.reshape(m_rows, H)

            logits_2d = logits_buf[:m_rows]
            if b_comp is None:
                torch.matmul(piece_2d, w_comp.t(), out=logits_2d)
            else:
                torch.addmm(b_comp, piece_2d, w_comp.t(), out=logits_2d)

            chunked_ce_fwd_bwd_kernel[(m_rows,)](
                logits_2d,
                target_c,
                token_entropy,
                token_loss,
                correct_buf,
                logits_2d.stride(0),
                V,
                S,
                start,
                cur_len,
                int(ignore_index),
                1 if with_grad else 0,
                BLOCK_V=block_v,
                num_warps=8,
            )

            if with_grad:
                grad_hidden[:, start:stop] = torch.matmul(logits_2d, w_comp).view(B, cur_len, H)

        del logits_buf

        total_loss = token_loss.sum()
        safe_denom = torch.where(denom_f32 > 0, denom_f32, torch.ones_like(denom_f32))
        loss = torch.where(denom_f32 > 0, total_loss / safe_denom, torch.zeros_like(total_loss))
        correct = correct_buf.sum(dtype=torch.long)

        ctx.with_grad = with_grad
        ctx.hidden_dtype = hidden.dtype
        if with_grad:
            inv_denom = torch.where(denom_f32 > 0, 1.0 / safe_denom, torch.zeros_like(denom_f32))
            ctx.save_for_backward(grad_hidden, inv_denom)
        ctx.mark_non_differentiable(token_entropy, token_loss, correct)
        return loss, token_entropy, token_loss, correct

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
        if not ctx.with_grad:
            return (None,) * 8
        grad_hidden, inv_denom = ctx.saved_tensors
        # (1 / denom) * loss_scale applied in fp32: the unnormalized fp16 gradient never
        # has to represent values as small as 1 / num_tokens.
        factor = grad_loss.float() * inv_denom
        scaled = (grad_hidden.float() * factor).to(ctx.hidden_dtype)
        return (scaled, None, None, None, None, None, None, None)


def triton_chunked_causal_lm_loss(
    hidden: torch.Tensor,
    head_module: Any,
    labels: torch.Tensor,
    chunk: int = DEFAULT_CHUNK_SIZE,
    *,
    num_items_in_batch: Any = None,
    shift_labels: Any = None,
    ignore_index: int = IGNORE_INDEX,
):
    from soup_cli.utils.stream_loss_chunk import ChunkedStep, shift_for_next_token

    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk!r}")

    target = shift_labels if shift_labels is not None else shift_for_next_token(labels, ignore_index)
    target = target.to(hidden.device)
    if target.shape != hidden.shape[:2]:
        raise ValueError(
            f"labels of shape {tuple(target.shape)} do not match hidden states of "
            f"shape {tuple(hidden.shape)}"
        )

    labelled = (target != ignore_index).sum()
    if num_items_in_batch is None:
        denominator = labelled
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch

    head_weight = getattr(head_module, "weight", None)
    head_bias = getattr(head_module, "bias", None)
    if head_weight is None and hasattr(head_module, "__self__"):
        owner = head_module.__self__
        head_weight = getattr(owner, "weight", None)
        head_bias = getattr(owner, "bias", None)

    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    loss, token_entropy, token_loss, correct = TritonChunkedCEFunction.apply(
        hidden,
        head_weight,
        head_bias,
        target,
        chunk,
        denominator,
        ignore_index,
        with_grad,
    )
    step = ChunkedStep(
        token_entropy=token_entropy,
        token_loss=token_loss,
        correct=correct,
        labelled=labelled,
    )
    return loss, step


def patch_fused_cross_entropy() -> None:
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    orig_qwen_forward = modeling_qwen2.Qwen2ForCausalLM.forward
    if not getattr(orig_qwen_forward, "_soup_triton_ce", False):

        def qwen2_causal_lm_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            logits_to_keep=0,
            **kwargs,
        ):
            head_weight = getattr(self.lm_head, "weight", None)
            if (
                labels is None
                or not _is_ce_context_active()
                or head_weight is None
                or not head_weight.is_cuda
                or head_weight.requires_grad
                or getattr(self.config, "final_logit_softcapping", None) is not None
            ):
                return orig_qwen_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            orig_seq_len = input_ids.shape[1] if (input_ids is not None and input_ids.ndim == 2) else None
            trimmed = False
            trimmed_len = orig_seq_len

            # Dynamic batch tail trimming: in right-padded SFT batches, tail tokens
            # past the maximum active label index across all batch rows never attend to
            # future positions (causal mask) and have label == IGNORE_INDEX. Trimming
            # tail padding saves up to 50% FLOPs and activation VRAM while computing
            # mathematically identical losses, gradients, and metrics.
            if (
                self.training
                and input_ids is not None
                and labels is not None
                and torch.is_tensor(labels)
                and labels.is_cuda
                and labels.ndim == 2
                and orig_seq_len is not None
                and orig_seq_len > 32
            ):
                active_mask = labels != IGNORE_INDEX
                if active_mask.any():
                    max_active_col = active_mask.nonzero()[:, 1].max().item()
                    needed_len = min(orig_seq_len, max_active_col + 1)
                    cand_len = min(orig_seq_len, ((needed_len + 31) // 32) * 32)
                    if cand_len < orig_seq_len:
                        trimmed = True
                        trimmed_len = cand_len
                        input_ids = input_ids[:, :trimmed_len]
                        if attention_mask is not None and attention_mask.ndim == 2:
                            attention_mask = attention_mask[:, :trimmed_len]
                        if position_ids is not None and position_ids.ndim == 2:
                            position_ids = position_ids[:, :trimmed_len]
                        labels = labels[:, :trimmed_len]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
            loss, step = triton_chunked_causal_lm_loss(
                outputs.last_hidden_state,
                self.lm_head,
                labels,
                chunk=DEFAULT_CHUNK_SIZE,
                num_items_in_batch=kwargs.get("num_items_in_batch"),
                shift_labels=kwargs.get("shift_labels"),
            )
            if trimmed and orig_seq_len is not None and trimmed_len < orig_seq_len:
                pad_s = orig_seq_len - trimmed_len
                token_entropy = torch.nn.functional.pad(step.token_entropy, (0, pad_s), value=0.0)
                token_loss = torch.nn.functional.pad(step.token_loss, (0, pad_s), value=0.0)
                correct = torch.nn.functional.pad(step.correct, (0, pad_s), value=0.0)
                step = ChunkedStep(
                    token_entropy=token_entropy,
                    token_loss=token_loss,
                    correct=correct,
                    labelled=step.labelled,
                )
            self._last_triton_chunked_step = step
            return CausalLMOutputWithPast(
                loss=loss,
                logits=None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        qwen2_causal_lm_forward._soup_triton_ce = True
        modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_causal_lm_forward

    try:
        from soup_cli.utils import stream_loss_chunk
        from soup_cli.utils.stream_loss_chunk import (
            HOOK_ATTR,
            STEP_ATTR,
            _log_trl_metrics,
            causal_lm_of,
        )
        from trl.trainer.sft_trainer import SFTTrainer
    except Exception:
        return

    orig_chunked_loss_fn = stream_loss_chunk.chunked_causal_lm_loss
    if not getattr(orig_chunked_loss_fn, "_soup_triton_ce", False):

        def patched_chunked_causal_lm_loss(
            hidden: Any,
            head: Any,
            labels: Any,
            chunk: int,
            *,
            num_items_in_batch: Any = None,
            shift_labels: Any = None,
            ignore_index: int = IGNORE_INDEX,
        ):
            if torch.is_tensor(hidden) and hidden.is_cuda:
                head_w = getattr(head, "weight", None)
                if head_w is None and hasattr(head, "__self__"):
                    head_w = getattr(head.__self__, "weight", None)
                if torch.is_tensor(head_w) and head_w.is_cuda:
                    return triton_chunked_causal_lm_loss(
                        hidden,
                        head,
                        labels,
                        chunk=chunk,
                        num_items_in_batch=num_items_in_batch,
                        shift_labels=shift_labels,
                        ignore_index=ignore_index,
                    )
            return orig_chunked_loss_fn(
                hidden,
                head,
                labels,
                chunk,
                num_items_in_batch=num_items_in_batch,
                shift_labels=shift_labels,
                ignore_index=ignore_index,
            )

        patched_chunked_causal_lm_loss._soup_triton_ce = True
        stream_loss_chunk.chunked_causal_lm_loss = patched_chunked_causal_lm_loss

    orig_sft_compute_loss = SFTTrainer.compute_loss
    if not getattr(orig_sft_compute_loss, "_soup_triton_ce", False):

        def sft_compute_loss(
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ):
            labels = inputs.get("labels") if isinstance(inputs, dict) else None
            if (
                getattr(self, HOOK_ATTR, None) is not None
                or getattr(self, "label_smoother", None) is not None
                or getattr(self, "compute_loss_func", None) is not None
                or getattr(getattr(self, "args", None), "use_liger_kernel", False)
                or not torch.is_tensor(labels)
                or not labels.is_cuda
                or "shift_labels" in inputs
                or not isinstance(causal_lm_of(model), modeling_qwen2.Qwen2ForCausalLM)
            ):
                return orig_sft_compute_loss(
                    self,
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            base_causal = causal_lm_of(model)
            mode = "train" if self.model.training else "eval"
            # As TRL's `compute_loss` does before it delegates (and Soup's own
            # ChunkedLossTrainer in soup_cli/utils/stream_loss_chunk.py).
            inputs.pop("_prediction_loss_only", None)
            inputs["use_cache"] = False

            prev_active = getattr(_CE_TLS, "active", False)
            _CE_TLS.active = True
            try:
                loss, outputs = Trainer.compute_loss(
                    self,
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                _CE_TLS.active = prev_active

            step = getattr(base_causal, "_last_triton_chunked_step", None)
            base_causal._last_triton_chunked_step = None
            setattr(self, STEP_ATTR, step)
            if step is not None:
                _log_trl_metrics(self, mode, inputs, step)
            return (loss, outputs) if return_outputs else loss

        sft_compute_loss._soup_triton_ce = True
        SFTTrainer.compute_loss = sft_compute_loss
