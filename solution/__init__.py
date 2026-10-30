"""Entry point of our Soup Kernel Challenge solution.

`harness/_train.py` imports `solution` and calls `solution.apply()` once before
`soup train` starts. Every optimization and custom `@triton.jit` kernel is
wired in through `apply()`:
1. `solution/fused_cross_entropy.py` (`chunked_ce_fwd_bwd_kernel`):
   Single-buffer chunked cross-entropy + online Shannon entropy + token accuracy +
   dynamic batch sequence padding trimming + in-place d_logits gradient over the
   151,936-token Qwen2.5 vocabulary.
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
"""

from __future__ import annotations

import torch

from solution.fused_cross_entropy import patch_fused_cross_entropy
from solution.nf4_linear import patch_bnb_and_lora_linear4bit
from solution.rmsnorm import patch_qwen2_rmsnorm
from solution.rope import patch_qwen2_rope
from solution.swiglu import patch_qwen2_mlp


def apply() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_bnb_and_lora_linear4bit()
    patch_qwen2_mlp()
    patch_fused_cross_entropy()
