"""Entry point of our Soup Kernel Challenge solution.

`harness/_train.py` imports `solution` and calls `solution.apply()` once before
`soup train` starts. Every optimization and custom `@triton.jit` kernel is
wired in through `apply()`:
1. `solution/fused_cross_entropy.py` (`chunked_ce_fwd_bwd_kernel`):
   Single-buffer chunked cross-entropy + online Shannon entropy + token accuracy +
   in-place d_logits gradient over the 151,936-token Qwen2.5 vocabulary.
2. `solution/nf4_linear.py` (`nf4_dequant_triton_kernel`):
   Fused double-quant NF4 dequantization + memory-free frozen 4-bit linear autograd
   (eliminating saved `x` on 140 frozen projections) + fused LoRA 4-bit autograd.
3. `solution/swiglu.py` (`tiled_swiglu_fwd_kernel`, `tiled_swiglu_bwd_kernel`):
   2D row-tiled SwiGLU kernels with in-place activation/gradient buffers and
   activation-checkpointed frozen MLP (`TritonFusedMLPFunction`).
4. `solution/rmsnorm.py` (`rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel`):
   Fused fp32-accumulated RMSNorm kernels that preserve native fp16 activation dtype.
5. `solution/rope.py` (`rope_qk_fwd_kernel`, `rope_qk_bwd_kernel`):
   Single-launch fused Q/K Rotary Position Embedding kernels without saving Q/K.
6. k-bit preparation:
   Keeps frozen 2D embedding/head weights (`[151936, 1536]`) in native fp16 instead of
   upcasting to fp32 (saving 890 MiB VRAM and avoiding fp32 residual promotion).
"""

from __future__ import annotations

import torch

from solution.fused_cross_entropy import patch_fused_cross_entropy
from solution.nf4_linear import patch_bnb_and_lora_linear4bit
from solution.rmsnorm import patch_qwen2_rmsnorm
from solution.rope import patch_qwen2_rope
from solution.swiglu import patch_qwen2_mlp


def _patch_kbit_training_prep() -> None:
    try:
        import peft
        import peft.utils.other as peft_other
    except Exception:
        return

    orig_prep = peft_other.prepare_model_for_kbit_training
    if getattr(orig_prep, "_soup_triton_kbit_prep", False):
        return

    def fast_prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing: bool = True,
        gradient_checkpointing_kwargs: dict | None = None,
        auto_clear_cache: bool = True,
    ):
        # Preserve native fp16 dtype on large 2D frozen embedding/lm_head tables
        # ([151936, 1536] = 445 MiB in fp16 vs 890 MiB in fp32) by temporarily
        # marking 2D floating-point parameters before calling PEFT's prep, then
        # restoring them without allocating a duplicate 890 MiB fp32 copy.
        saved_2d_fp16: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        for param in model.parameters():
            if (
                param.ndim >= 2
                and param.dtype in (torch.float16, torch.bfloat16)
                and param.__class__.__name__ != "Params4bit"
            ):
                saved_2d_fp16.append((param, param.data))
                # Dummy 1-element fp32 view so PEFT's loop skips reallocating 445 MiB
                param.data = torch.empty((0,), dtype=torch.float32, device=param.device)

        try:
            res = orig_prep(
                model,
                use_gradient_checkpointing=use_gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                auto_clear_cache=False,
            )
        finally:
            for param, orig_data in saved_2d_fp16:
                param.data = orig_data
                param.requires_grad = False

        return res

    fast_prepare_model_for_kbit_training._soup_triton_kbit_prep = True
    peft_other.prepare_model_for_kbit_training = fast_prepare_model_for_kbit_training
    peft.prepare_model_for_kbit_training = fast_prepare_model_for_kbit_training


def apply() -> None:
    _patch_kbit_training_prep()
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_bnb_and_lora_linear4bit()
    patch_qwen2_mlp()
    patch_fused_cross_entropy()
