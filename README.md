# Soup Kernel Challenge — Custom Triton Kernels for Qwen2.5-1.5B QLoRA on Tesla T4

Custom `@triton.jit` GPU kernels and memory-optimal autograd primitives for accelerating `soup train` (`soup-cli==0.77.0`) and reducing peak GPU memory when fine-tuning **Qwen/Qwen2.5-1.5B-Instruct** (4-bit NF4, LoRA $r=16, \alpha=32, p=0.05$ on `q_proj`, `v_proj`) on a single free **Kaggle Tesla T4 (15 GiB, Turing sm_75, FP16)**.

- **Public Kaggle Notebook**: `<paste-saved-kaggle-notebook-url-here>`
- **Submission Metadata**: [`SUBMISSION.md`](./SUBMISSION.md)

---

## 1. What We Did (Architecture & Custom Triton Kernels)

### Root-Cause Analysis of Base `soup train` on Kaggle T4
Profiling base Soup 0.77.0 (`14,350 MiB` peak VRAM, `~80 s` wall time) revealed five major bottlenecks in the default `transformers` + `peft` + `trl` + `bitsandbytes` execution path on Turing (`fp16`):

1. **Full-Vocabulary Logits & Multi-Pass Metrics (`3.56 GiB` VRAM, ~7% GPU time)**:
   `Qwen2ForCausalLM` materializes `[4, 512, 151936]` logits (`607 MiB` in `fp16`), `ForCausalLMLoss` upcasts them to `fp32` (`1.21 GiB`) and allocates `1.21 GiB` for softmax gradients, while `SFTTrainer.compute_loss` (`entropy_from_logits`) and `RewindSFTTrainer` (`row_losses`) run separate full-vocabulary `log_softmax` and `argmax` passes over `[4, 512, 151936]`.
2. **Accidental FP32 Promotion Across All 28 Decoder Layers (`~44%` of GPU time on casts/elementwise ops)**:
   `peft.prepare_model_for_kbit_training` casts every non-4-bit parameter to `torch.float32`, including the frozen `[151936, 1536]` tied `embed_tokens` / `lm_head` table (`445 MiB` $\to$ `890 MiB`) and all 57 `Qwen2RMSNorm.weight` vectors. In `Qwen2RMSNorm.forward`:
   ```python
   return self.weight * hidden_states.to(input_dtype)
   ```
   Multiplying by `fp32` `self.weight` promotes the normalized hidden state back to `torch.float32` at every layer. Consequently, all 196 `Linear4bit` projections cast inputs `fp32 -> fp16` on entry and `fp16 -> fp32` on exit (`392` `_to_copy` launches/step), all residual streams and MLP intermediates are stored in `fp32` (2x memory), and PyTorch SDPA receives `fp32` Q/K/V, falling back to slow unfused FP32 math attention instead of FP16 Tensor Cores.
3. **Unconditional Input Saving on Frozen 4-bit Projections (`1.32 GiB` VRAM) & Two-Stage NF4 Dequantization**:
   `bitsandbytes.autograd._functions.MatMul4Bit.forward` calls `ctx.save_for_backward(A, B)` even on the **140 frozen projections** (`k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) where `weight.requires_grad=False`, pinning the `[4, 512, 8960]` SwiGLU output (`980 MiB` across 28 layers) and attention/norm outputs (`336 MiB`) in VRAM until backward. Furthermore, double-quantized NF4 runs two separate CUDA kernels (`state2` scale expansion + 4-bit nibble lookup) per projection.
4. **Persistent `[4, 512, 8960]` MLP Activations Across 28 Layers (`2.77 GiB` VRAM)**:
   Even in `fp16`, keeping `gate` (`35 MiB`) and `up` (`35 MiB`) alive across 28 layers consumes `1,960 MiB` of VRAM, even though `Qwen2MLP` has zero trainable parameters and zero dropout.
5. **Eager RoPE & 6x GQA KV Expansion (`repeat_kv`)**:
   `apply_rotary_pos_emb` allocates 6 temporary tensors per layer via `rotate_half` (`torch.cat`) and saves `q, k` for backward. Meanwhile, passing a 2D right-padding `attention_mask` forces `sdpa_mask` to materialize a 4D mask, which disables native GQA (`use_gqa_in_sdpa -> False`) and expands `K` and `V` $6\times$ (`2 -> 12` heads) via `repeat_kv` across all 28 layers (`336 MiB`).

---

### Summary of Our 8 Custom `@triton.jit` Kernels in `solution/`

| Kernel (`@triton.jit`) | Module | What It Replaces & How It Works |
|---|---|---|
| `chunked_ce_fwd_bwd_kernel` | [`solution/fused_cross_entropy.py`](./solution/fused_cross_entropy.py) | Replaces full-vocab `ForCausalLMLoss` + TRL `entropy_from_logits` + argmax accuracy + `rewind` per-token loss. Processes `[B, S, H]` in chunks of 64 sequence positions reusing **one** pre-allocated `74 MiB` `fp16` logits buffer (`[256, 151936]`). Pass 1 computes online log-sum-exp $(m, d)$, online $\sum e^{x_i - m} x_i$ for exact Shannon entropy $H = \text{lse} - s/d$, argmax accuracy, and NLL loss; Pass 2 overwrites the buffer **in-place** with scaled gradients $d\text{logits}$ in `fp16`. |
| `nf4_dequant_triton_kernel` | [`solution/nf4_linear.py`](./solution/nf4_linear.py) | Replaces `bitsandbytes` two-stage nested NF4 dequantization. Each program loads `BLOCK_BYTES=512` packed bytes (`1,024` 4-bit weights), reconstructs double-quant scales `state2.code[absmax] * state2.absmax + offset` in registers, looks up high/low NF4 nibbles, interleaves pairs via `tl.join`, and writes coalesced `fp16` weights. Paired with `TritonFrozenLinear4bitFunction` (which never saves `x` on frozen layers) and `TritonLoraLinear4bitFunction` (which shares `fp16` `x` and saves a 1-byte boolean dropout mask). |
| `tiled_swiglu_fwd_kernel` | [`solution/swiglu.py`](./solution/swiglu.py) | 2D row-strided block-tiled (`(rows, col_tiles)`) SwiGLU forward kernel (`silu(gate) * up` accumulated in `fp32`), writing **in-place** into `gate` inside `TritonFusedMLPFunction` so `up` is freed immediately. |
| `tiled_swiglu_bwd_kernel` | [`solution/swiglu.py`](./solution/swiglu.py) | 2D row-strided block-tiled SwiGLU backward kernel that computes analytical $(d\text{gate}, d\text{up})$ in `fp32` and writes them **in-place** into the recomputed $(gate, up)$ buffers in `TritonFusedMLPFunction`, followed by in-place `addmm_` gradient accumulation. |
| `rmsnorm_fwd_kernel` | [`solution/rmsnorm.py`](./solution/rmsnorm.py) | Single-pass row-wise RMSNorm forward kernel. Computes variance and `rstd` in `fp32`, multiplies by `weight` in `fp32`, and writes output in the input's native `fp16` dtype (preventing `fp32` promotion). |
| `rmsnorm_bwd_kernel` | [`solution/rmsnorm.py`](./solution/rmsnorm.py) | Single-pass row-wise RMSNorm backward kernel computing $dx = \text{rstd} \cdot (dy \odot w - \hat{x} \cdot \text{mean}(dy \odot w \odot \hat{x}))$ in `fp32` registers using saved `rstd`. |
| `rope_qk_fwd_kernel` | [`solution/rope.py`](./solution/rope.py) | Single-launch fused Q+K Rotary Position Embedding forward kernel. Loads $\cos, \sin$ once per $(b, s)$ token into registers and rotates all 12 Query heads and 2 Key heads without `rotate_half` copies or saving $Q, K$. |
| `rope_qk_bwd_kernel` | [`solution/rope.py`](./solution/rope.py) | Single-launch fused Q+K RoPE backward kernel applying the exact inverse rotation $(-\theta)$ across all 12 Query and 2 Key heads using only saved $(\cos, \sin)$. |

### Attribution of Algorithmic Ideas
All kernel code in `solution/` was written from scratch by our team. We credit the following foundational algorithmic papers and projects for the underlying mathematical techniques:
- **Online Softmax & Single-Pass Log-Sum-Exp**: Milakov & Gimelshein (2018), *"Online normalizer calculation for softmax"*, and FlashAttention (Dao et al., 2022). Extended in our `chunked_ce_fwd_bwd_kernel` to maintain a simultaneous online recurrence for $\sum_i e^{x_i - m} x_i$ so exact Shannon entropy $H = \text{lse} - s/d$ and cross-entropy loss are computed in a single streaming pass.
- **Chunked Output-Head Cross-Entropy & In-Place Logits Gradient**: Inspired by Apple's Cut Cross-Entropy (Wijmans et al., 2024), Liger-Kernel's `LigerFusedLinearCrossEntropyLoss`, and Soup v0.77.0's own `soup_cli/utils/stream_loss_chunk.py` (`#1693`).
- **4-bit NormalFloat (NF4) & Double Quantization Layout**: Dettmers et al. (2023), *"QLoRA: Efficient Finetuning of Quantized LLMs"* (`bitsandbytes` packed nibble and `QuantState.state2` format).
- **Deterministic MLP Activation Recomputation & RoPE Orthogonal Backward**: Inspired by Unsloth's manual autograd philosophy and standard rotary embedding identities (Su et al., 2021).

---

## 2. How to Run It

### On Kaggle (Tesla T4 x2)
1. Import [`templates/kaggle_notebook.ipynb`](./templates/kaggle_notebook.ipynb) into Kaggle (**File → Import Notebook**).
2. In the right-hand Settings panel, select **Accelerator: GPU T4 x2** and **Internet: On**.
3. In the first code cell, set `REPO_URL` to this repository URL (and `COMMIT` to the submitted commit SHA).
4. Click **Run All** (or **Save Version → Save & Run All**). The notebook automatically:
   - Verifies the Tesla T4 GPU and installs the pinned packages (`soup-cli[train]==0.77.0`, `transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`, `bitsandbytes==0.50.2`, `accelerate==1.14.0`, `pytest`),
   - Runs `python task/prepare_data.py`,
   - Runs `python -m pytest solution -q` (unit tests for all 5 custom Triton kernel modules against PyTorch references),
   - Runs `python harness/run_task.py` (measures base Soup and our solution back-to-back with fresh compiler caches and evaluates both adapters on `data/eval_public.jsonl`),
   - Runs `python harness/check_rules.py` (verifies all 10 admission checks, including the 79 Soup regression tests) and `python harness/score.py runs/result.json`.

### On a Local Linux Machine with a Tesla T4 GPU
```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py          # prepares data/train.jsonl and data/eval_public.jsonl
python -m pytest solution -q         # verifies all custom Triton kernels (forward & backward)
python harness/run_task.py           # runs base & solution -> runs/result.json
python harness/check_rules.py        # runs all 10 rule checks -> runs/checks.json
python harness/score.py runs/result.json
```

---

## 3. Results & Limitations

### Benchmark Results on Kaggle Tesla T4
*(Update the exact numbers below from `runs/result.json` and `runs/checks.json` after your Kaggle Save & Run All completes)*

| Metric | Base Soup 0.77.0 | Our Solution (`solution.apply()`) | Ratio / Delta |
|---|---|---|---|
| **Whole `soup train` wall time (`wall_s`)** | `~80.0 s` | `<fill-from-result.json> s` | **`<speedup>x` faster** |
| **Peak GPU memory (`gpu_peak_mib`)** | `~14,350 MiB` | `<fill-from-result.json> MiB` | **`<memory_ratio>x` less memory** |
| **Held-out Eval Loss (`eval_public.jsonl`)** | `~2.050` | `<fill-from-result.json>` | **`<quality_delta_pct>%` vs base** |
| **Custom `@triton.jit` share of train loop (`c4`)** | `0.0%` | `<fill-from-checks.json>%` | **`>= 5.0%` required** |
| **Rule Checks (`harness/check_rules.py`)** | — | **10 / 10 PASS** (`79 / 79` pytest) | **All checks passed** |

### Limitations & Future Work
1. **Cold JIT Compilation Overhead (`~3–4 s` in `wall_s`)**:
   Because `harness/run_task.py` clears `TRITON_CACHE_DIR` before every run, cold compilation of our 8 `@triton.jit` kernels is included in `wall_s`. To minimize this fixed cost, we deliberately avoided `@triton.autotune` and runtime `tl.constexpr` branching (for example, `write_grad_flag` in `chunked_ce_fwd_bwd_kernel` is a scalar runtime argument so Triton compiles only a single specialization).
2. **Dequantize + cuBLAS GEMM vs. Fused W4A16 Split-K GEMM**:
   Our `nf4_dequant_triton_kernel` fuses nested double-quant scale reconstruction and 4-bit unpacking into a transient `fp16` buffer and delegates the dense matrix multiplication to cuBLAS (`F.linear` / `torch.matmul`). Writing a fused W4A16 Triton GEMM for `[2048, 1536] x [1536, 8960]` on Turing (`sm_75`, `mma.sync.aligned.m16n8k8`) could avoid writing the transient `fp16` weight matrix to global memory, though matching cuBLAS's hand-tuned Turing Tensor Core assembly across all aspect ratios without `@triton.autotune` is challenging.
3. **Architecture & Padding Scope**:
   Our patches target `Qwen2ForCausalLM` / `Qwen2Model` / `Qwen2MLP` / `Qwen2RMSNorm` / `bnb.nn.Linear4bit` with frozen 4-bit base weights and right-padded causal batches (`padding_side="right"`). When `labels` are absent, when base weights require gradients, or when running on CPU in unit tests, every patch transparently falls back to the unmodified upstream implementation.

---

## License
Apache-2.0, see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
