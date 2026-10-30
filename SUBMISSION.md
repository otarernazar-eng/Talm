# Submission: Team Talm

## Links
- Repository, final commit: `https://github.com/<you>/<repo>/commit/<sha>`
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  `https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>`
- Demo video (up to 3 minutes): `<link>`
- Presentation: `<link>`

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| `<Name>` | ML / GPU Kernel Engineer | `<github>` | `<kaggle>` | `<contact>` |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `chunked_ce_fwd_bwd_kernel` | `solution/fused_cross_entropy.py` | Full-vocab `[B, S, 151936]` `ForCausalLMLoss` + `entropy_from_logits` + argmax accuracy + `rewind` token losses + in-place `d_logits` | `<fill>%` |
| `nf4_dequant_triton_kernel` | `solution/nf4_linear.py` | `bitsandbytes` two-stage nested double-quant `state2` + 4-bit NF4 dequantization across all 196 projections (forward & backward) | `<fill>%` |
| `tiled_swiglu_fwd_kernel` | `solution/swiglu.py` | `Qwen2MLP` `F.silu(gate) * up` forward activation (2D row-tiled, in-place output into `gate`) | `<fill>%` |
| `tiled_swiglu_bwd_kernel` | `solution/swiglu.py` | `Qwen2MLP` SwiGLU backward gradient computation (2D row-tiled, in-place into recomputed `gate` and `up`) | `<fill>%` |
| `rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm.forward` eager fp32 cast, variance, rsqrt, and weight multiply (preserving native fp16 output) | `<fill>%` |
| `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` backward pass over 57 normalization layers per step | `<fill>%` |
| `rope_qk_fwd_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` forward pass across all 12 Q heads and 2 K heads in one launch without `rotate_half` copies | `<fill>%` |
| `rope_qk_bwd_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` backward pass across all 12 Q heads and 2 K heads without saving `q` or `k` | `<fill>%` |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Online softmax normalizer calculation: Milakov & Gimelshein (2018) (https://arxiv.org/abs/1805.02867) and FlashAttention (https://arxiv.org/abs/2205.14135), extended with an online weighted sum for single-pass exact Shannon entropy.
- Chunked output-head cross-entropy and in-place logit gradients: Apple Cut Cross-Entropy (https://arxiv.org/abs/2411.09009), Liger-Kernel (https://github.com/linkedin/Liger-Kernel), and Soup v0.77.0's `soup_cli/utils/stream_loss_chunk.py`.
- 4-bit NF4 packed nibble and double-quantization (`state2`) layout: QLoRA / bitsandbytes (https://arxiv.org/abs/2305.14314, https://github.com/bitsandbytes-foundation/bitsandbytes).
- Custom autograd without saving frozen-projection inputs and deterministic MLP activation recomputation: Unsloth (https://github.com/unslothai/unsloth).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | `<base_wall_s>` | `<sol_wall_s>` | speedup `<speedup>x` |
| Peak GPU memory, MiB | `<base_peak_mib>` | `<sol_peak_mib>` | base/ours `<memory_ratio>x` |
| Eval loss (public rows) | `<base_eval_loss>` | `<sol_eval_loss>` | `<quality_delta_pct>%` vs base |

- [x] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why, how our 8 `@triton.jit` kernels and custom autograd functions work, results against the base (speed, memory, quality), limits and next steps.

## Limitations
- Cold Triton JIT compilation (~3–4 s) is paid on step 0 because compiler caches start empty on each measured run; we mitigated this by avoiding `@triton.autotune` and minimizing `tl.constexpr` specializations.
- 4-bit linear projections use our fused `nf4_dequant_triton_kernel` into a transient `fp16` buffer followed by cuBLAS GEMM rather than a fused W4A16 GEMM kernel, as cuBLAS Tensor Core kernels on Turing (`sm_75`) are faster for `batch * seq = 2048` than un-autotuned Triton GEMMs.
- Custom MLP recomputation (`TritonFusedMLPFunction`) applies when MLP weights are frozen (`target_modules: [q_proj, v_proj]`); when MLP projections carry trainable LoRA adapters, `Qwen2MLP` automatically falls back to `triton_swiglu` without discarding `gate`/`up`.
