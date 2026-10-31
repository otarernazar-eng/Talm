"""End-to-end parity: a small QLoRA training run through `solution.apply()` against the code it
replaces, component by component, with the relation each docstring and README section 4 claim.

A child process trains the same small model on the same data many times, in process: first
through transformers' and PEFT's own code (nothing of the solution installed), then through
`solution.apply()` with every component on, then once per component with that component
switched off by its own SOLUTION_DISABLE_* switch (everything `apply()` replaced is put back
first and `apply()` runs again under the new switches), and at the end the default solution and
the unpatched baseline once more, to show that the switching itself changes nothing. A second
child does the same for the remaining switches at the same time (with its own default run as
their reference: every switch is compared with a run of the same process). Two more children
call `apply()` first thing as the harness does (with and without the start-up patches) and train
the default solution once, for the start-up patches, which act at import and load time. The
children write their results to files; the tests here compare them.

The LoRA dropout masks are a different realization of the same RNG stream whenever the dropout
input differs in shape (unpad: [1, N, H] instead of [B, T, H]) or in dtype (the fp16 norm output
instead of transformers' fp32 one: `native_dropout` maps the Philox numbers to elements in
dtype-dependent vector widths). Runs that differ in either are therefore compared once more with
dropout 0 (the "_p0" runs) for their numerics, and with dropout on only at the harness's checks.

The run (as in the task, scaled down): a tied Qwen2 (vocabulary 1000, hidden 512, 3 layers,
4 query / 2 KV heads of 128, MLP 1024; random weights, saved in bf16 like the real checkpoint),
loaded through `from_pretrained` in 4-bit NF4 with double quantization and fp16 compute,
`peft.prepare_model_for_kbit_training`, LoRA r=16 / alpha 32 / dropout 0.05 on q_proj and
v_proj, TRL's SFTTrainer with fp16 autocast and GradScaler (fp16=True, set explicitly), AdamW,
lr 2e-4 cosine, weight decay 0.01, clipping at 1.0, batch 4 of right-padded rows of 64-512
tokens with a prompt part (labels -100) and a completion part, 4 optimizer steps, dropout on,
`torch.use_deterministic_algorithms(True)` (and a deterministic cuBLAS workspace), so that two
runs of the same code agree bit for bit. The hidden size is 512 rather than 64-128 because the
per-layer dequantization packs a layer only when every projection is a whole number of
second-level absmax blocks (and splits the last layers only when they start on 1024-row
boundaries); smaller layers would silently take the fallback paths. Two settings depart from
the task on purpose: lora_B starts non-zero (PEFT starts it at 0, which makes the step-0
gradient of lora_A zero and the dropout mask irrelevant to the step-0 forward), and the head
walks the vocabulary in slices of 384 columns (SOLUTION_HEAD_VCHUNK; with 1000 tokens the
default 8192 would be one slice, and the task has 19). SOLUTION_VARLEN=1 makes the attention take
the T4's variable-length path on newer GPUs.

Compared per run: the logged loss, grad_norm, learning rate, TRL's entropy / num_tokens /
mean_token_accuracy (the same TRL formulas, from the logits in the baseline and from the fused
step with the solution), total_flos, the GradScaler's skipped steps, the LoRA gradients of step
0 (unscaled and clipped, as the optimizer gets them), the adapter state dict after training, and
with the solution the hidden-state gradient at the fused head in step 0. Every comparison of a
switch also checks that the component ran in one run and not in the other (call counts), so no
check passes because a fallback path ran in both.

Run on a GPU: `python -m pytest solution/test_parity.py -q` (about 30 s on an idle RTX 4080 with
a warm Triton cache; other jobs on the same GPU make it a few times slower).
"""

import collections
import contextlib
import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

REPO = Path(__file__).resolve().parents[1]
STEPS = 4
BATCH = 4
SEED = 1
VOCAB = 1000
WORKER_TIMEOUT = 1500  # about 40 s on an idle RTX 4080; other jobs on the GPU slow it down

BASE_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # deterministic cuBLAS (use_deterministic_algorithms)
    "SOLUTION_VARLEN": "1",                # the T4's single-launch attention on this GPU too
    "SOLUTION_HEAD_VCHUNK": "384",         # several vocabulary slices, the last one partial
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
}
# The in-process runs: start-up patches (tested by the other two processes) and the warm-up
# thread act outside the training step and are left out here.
MAIN_ENV = {"SOLUTION_DISABLE_STARTUP": "1", "SOLUTION_WARMUP_THREAD": "0"}
WORKERS = {
    "main": MAIN_ENV,                               # the baseline and the core switches
    "more": MAIN_ENV,                               # the other switches (in parallel)
    "startup": {},                                  # apply() first, every default (the harness)
    "control": {"SOLUTION_DISABLE_STARTUP": "1"},   # the same without the start-up patches
}

# Runs with the solution: worker -> {name: switches on top of the defaults}. Each switching
# worker trains the default solution ("full") first, the reference of its other runs. Names
# ending in "_p0" train with LoRA dropout 0 (see test_fp16_norm_output_path).
CONFIGS = {
    "main": {
        "full": {},
        "no_fused_qkv": {"SOLUTION_DISABLE_FUSED_PROJ": "1"},
        "no_lora_node": {"SOLUTION_DISABLE_LORA": "1"},
        "no_lora_cast": {"SOLUTION_DISABLE_LORA_CAST": "1"},
        "no_add_norm": {"SOLUTION_DISABLE_ADDNORM": "1"},
        "rows_attention": {"SOLUTION_DISABLE_VARLEN": "1"},
        "bnb_dequant": {"SOLUTION_DISABLE_GROUP": "1", "SOLUTION_DISABLE_DEQUANT": "1"},
        "no_fp16_embed": {"SOLUTION_DISABLE_FP16_EMBED": "1"},
        "no_defer": {"SOLUTION_DISABLE_DEFER": "1"},
        "no_bookkeeping": {"SOLUTION_DISABLE_BOOKKEEPING": "1"},
        "full_p0": {},
        "no_rmsnorm_p0": {"SOLUTION_DISABLE_RMSNORM": "1"},
        "no_unpad_p0": {"SOLUTION_DISABLE_UNPAD": "1"},
        "full_again": {},
    },
    "more": {
        "full": {},
        "no_fused_qkv": {"SOLUTION_DISABLE_FUSED_PROJ": "1"},
        "no_head_compact": {"SOLUTION_DISABLE_HEAD_COMPACT": "1"},
        "head_rows_x1": {"SOLUTION_DISABLE_HEAD_ROWS": "1"},
        "no_rmsnorm": {"SOLUTION_DISABLE_RMSNORM": "1"},
        "attention_cat": {"SOLUTION_DISABLE_ATTN_NOCAT": "1"},
        "nf4_dequant": {"SOLUTION_DISABLE_GROUP": "1"},
        "no_split_dequant": {"SOLUTION_DISABLE_SPLIT_DEQUANT": "1",
                             "SOLUTION_DISABLE_FUSED_PROJ": "1"},
        "rope_head_major": {"SOLUTION_DISABLE_ROPE_TOKEN_MAJOR": "1"},
        "rope_element_strides": {"SOLUTION_DISABLE_ROPE_ROWS": "1"},
        "no_swiglu_inplace": {"SOLUTION_DISABLE_SWIGLU_INPLACE": "1"},
        "mlp_save_up": {"SOLUTION_MLP_SAVE_UP": "1"},
        "mlp_recompute": {"SOLUTION_MLP_RECOMPUTE": "1"},
    },
}
SHARED = ("full", "no_fused_qkv")  # trained by every switching worker: references
WORKER_OF = {name: worker for worker, runs in CONFIGS.items() for name in runs
             if name not in SHARED}


def _key(name, worker=None):
    """The results key of a run: the main worker's names as they are, the others prefixed."""
    worker = worker or WORKER_OF.get(name, "main")
    return name if worker == "main" else f"{worker}:{name}"


def _pair(runs, name, reference="full"):
    """(`reference` as trained by the worker that ran `name`, the run `name`): both from the same
    process."""
    worker = WORKER_OF.get(name, "main")
    return runs[_key(reference, worker)], runs[_key(name, worker)]


# Claimed bit-identical: (component, run with it switched off, the run it is compared with,
# counter that must be > 0 in the reference and 0 in that run, where the claim is made; a counter
# "-name": an opt-in that must run in that run only). Everything compared must be equal. The
# fused q|k|v projection (on by default, not bit-identical: fused_proj.py) needs the packed group
# and the LoRA node and leaves the split passes alone, so the switches that take those away are
# compared with the default solution without it ("no_fused_qkv").
BIT_IDENTICAL = [
    ("LoRA node vs PEFT's LoRA forward", "no_lora_node", "no_fused_qkv", "lora_node", "lora.py"),
    ("LoRA fp16 copies once per step vs a cast per forward", "no_lora_cast", "full",
     "lora_cast_refresh", "lora.py"),
    ("residual add inside the norm kernel vs separate add + norm", "no_add_norm", "full",
     "add_norm", "rmsnorm.py"),
    ("varlen attention vs one launch per row (forward and backward)", "rows_attention", "full",
     "varlen_attention", "unpad.py"),
    ("attention backward without copies/cat vs with", "attention_cat", "full", None, "unpad.py"),
    ("group dequant + LoRA node vs bitsandbytes' dequant + PEFT", "bnb_dequant", "no_fused_qkv",
     "group_linear", "qgroup.py, lora.py"),
    ("group dequant vs our per-projection dequant", "nf4_dequant", "no_fused_qkv",
     "group_linear", "qgroup.py, nf4_dequant.py"),
    ("split dequant of the last layers vs whole-layer dequant", "no_split_dequant",
     "no_fused_qkv", "split_dequant", "qgroup.py"),
    ("fp16 tied embedding vs PEFT's fp32 upcast", "no_fp16_embed", "full", "fp16_embed",
     "fp16_embed.py"),
    ("metrics read at logging time vs synced after the forward", "no_defer", "full", "deferred",
     "host.py"),
    ("trainer bookkeeping without module walks vs Trainer's", "no_bookkeeping", "full",
     "bookkeeping", "host.py"),
    ("RoPE gradients token-major vs head-major", "rope_head_major", "full", None, "rope.py"),
    ("RoPE strides in rows vs elements", "rope_element_strides", "full", None, "rope.py"),
    ("SwiGLU backward in place vs the example's", "no_swiglu_inplace", "full", "swiglu_inplace",
     "swiglu_inplace.py"),
    ("opt-in: MLP keeps up, recomputes gate (SOLUTION_MLP_SAVE_UP=1)", "mlp_save_up", "full",
     "-mlp_save_up", "mlp_save_up.py"),
    ("opt-in: MLP recomputes gate and up (SOLUTION_MLP_RECOMPUTE=1)", "mlp_recompute",
     "no_fused_qkv", "-mlp_recompute", "mlp_recompute.py"),
]


# --------------------------------------------------------------------------------------------
# The worker: runs in a child process (python -m solution.test_parity <mode> <out_dir>).
# Nothing of transformers / PEFT / TRL is imported at module level: the "startup" worker calls
# solution.apply() before them, as the harness does.
# --------------------------------------------------------------------------------------------

_MISSING = object()
_T0 = time.perf_counter()


def _log(what):
    print(f"[parity] {time.perf_counter() - _T0:7.2f} s  {what}", flush=True)


# Everything solution.apply() replaces: (module, class or None, attribute).
PATCHED = (
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP", "forward"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2RMSNorm", "forward"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2DecoderLayer", "forward"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2Attention", "forward"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2Model", "forward"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM", "forward"),
    ("transformers.models.qwen2.modeling_qwen2", None, "apply_rotary_pos_emb"),
    ("bitsandbytes.nn", "Linear4bit", "forward"),
    ("bitsandbytes.backends.cuda.ops", None, "_dequantize_4bit_impl"),
    ("peft.tuners.lora.bnb", "Linear4bit", "forward"),
    ("peft", None, "prepare_model_for_kbit_training"),
    ("peft.utils", None, "prepare_model_for_kbit_training"),
    ("peft.utils.other", None, "prepare_model_for_kbit_training"),
    ("trl", "SFTTrainer", "compute_loss"),
    ("trl", "SFTTrainer", "log"),
    ("transformers", "Trainer", "train"),
    ("transformers", "Trainer", "floating_point_ops"),
    ("transformers", "Trainer", "_clip_grad_norm"),
    ("soup_cli.trainer.rewind_hf", None, "row_losses_from_token_losses"),
    ("soup_cli.trainer.rewind_hf", "RewindState", "on_micro_batch"),
)


def _owner(module_name, cls_name):
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return module if cls_name is None else getattr(module, cls_name, None)


def _snapshot():
    """The original objects behind every attribute apply() replaces (before any apply())."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    saved = {}
    for module_name, cls_name, attr in PATCHED:
        owner = _owner(module_name, cls_name)
        if owner is not None:
            saved[(module_name, cls_name, attr)] = (owner, vars(owner).get(attr, _MISSING))
    saved["sdpa"] = ALL_ATTENTION_FUNCTIONS["sdpa"]
    return saved


def _restore(saved):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for key, value in saved.items():
        if key == "sdpa":
            ALL_ATTENTION_FUNCTIONS["sdpa"] = value
            continue
        owner, original = value
        attr = key[2]
        if original is _MISSING:
            if attr in vars(owner):
                delattr(owner, attr)
        else:
            setattr(owner, attr, original)


def _flags():
    """The module-level switches of the solution, as each module computes them from the
    environment at import (re-read here for every configuration of the main worker)."""
    env = os.environ.get
    split_off = env("SOLUTION_DISABLE_SPLIT_DEQUANT") == "1"
    return {
        ("fused_head", "VOCAB_CHUNK"): int(env("SOLUTION_HEAD_VCHUNK", "8192")),
        ("fused_head", "VOCAB_PATH"): env("SOLUTION_DISABLE_HEAD_VCHUNK") != "1",
        ("fused_head", "COMPACT"): env("SOLUTION_DISABLE_HEAD_COMPACT") != "1",
        ("fused_head", "ROW_UNIT"): 1 if env("SOLUTION_DISABLE_HEAD_ROWS") == "1" else 8,
        ("lora", "CAST_CACHE"): env("SOLUTION_DISABLE_LORA_CAST") != "1",
        ("unpad", "_ROWS_ATTENTION"): env("SOLUTION_DISABLE_ROWS_ATTN") != "1",
        ("unpad", "_VARLEN"): env("SOLUTION_DISABLE_VARLEN") != "1",
        ("unpad", "_NOCAT"): env("SOLUTION_DISABLE_ATTN_NOCAT") != "1",
        ("rope", "TOKEN_MAJOR_GRAD"): env("SOLUTION_DISABLE_ROPE_TOKEN_MAJOR") != "1",
        ("rope", "ROW_STRIDES"): env("SOLUTION_DISABLE_ROPE_ROWS") != "1",
        ("fused_proj", "INPLACE"): env("SOLUTION_DISABLE_SWIGLU_INPLACE") != "1",
        ("qgroup", "_SPLIT_OFF"): split_off,
        ("qgroup", "SPLIT_LAYERS"): {"fwd": 0 if split_off else 1, "bwd": 0 if split_off else 2},
    }


def _set_flags():
    import importlib

    from solution import patches, rmsnorm

    for (module, name), value in _flags().items():
        setattr(importlib.import_module(f"solution.{module}"), name, value)
    patches._unpad.update(enabled=os.environ.get("SOLUTION_DISABLE_UNPAD") != "1", checked=0)
    patches._host.update(defer=False, bookkeeping=False)  # apply() switches them on
    patches._pending_step.clear()
    patches._first_forward[:] = [True]
    rmsnorm.LAYER_FORWARDS.clear()  # filled again by patch_qwen2_add_norm when it is on


def _check_flags_match_modules():
    """The switches as the modules computed them at import equal `_flags()` under the same
    environment: the re-reading above follows the modules' own definitions."""
    import importlib

    for (module, name), value in _flags().items():
        current = getattr(importlib.import_module(f"solution.{module}"), name)
        assert current == value, f"solution.{module}.{name} = {current!r}, expected {value!r}"


def _tokenizer():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {f"<t{i}>": i for i in range(VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<t1>"))
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<t0>", unk_token="<t1>",
                                   eos_token="<t2>")


def _rows():
    """STEPS * BATCH tokenized prompt/completion rows of 64..512 tokens (the task's max_length):
    runs of repeated tokens, so that the random model's argmax is right on part of them
    (accuracy about 0.4, entropy about 6.4 nats, loss about 5.3)."""
    gen = torch.Generator().manual_seed(SEED)
    rows = []
    for _ in range(STEPS * BATCH):
        length = int(torch.randint(64, 513, (1,), generator=gen))
        prompt = int(length * (0.25 + 0.5 * float(torch.rand(1, generator=gen))))
        ids = []
        while len(ids) < length:
            token = int(torch.randint(3, VOCAB, (1,), generator=gen))
            ids += [token] * int(torch.randint(1, 5, (1,), generator=gen))
        rows.append({"input_ids": ids[:length],
                     "completion_mask": [0] * prompt + [1] * (length - prompt)})
    return rows


def _make_checkpoint(path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=VOCAB, hidden_size=512, intermediate_size=1024,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=512, tie_word_embeddings=True,
                         rms_norm_eps=1e-6, attn_implementation="sdpa")
    torch.manual_seed(SEED)
    model = Qwen2ForCausalLM(config)
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(0.8 + 0.4 * torch.rand(param.shape, generator=gen))
            elif name.endswith("embed_tokens.weight"):  # step 0: loss 5.3, entropy 6.5, acc 0.45
                param.copy_(torch.randn(param.shape, generator=gen) * 0.04)
            else:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.02)
    model.to(torch.bfloat16).save_pretrained(path)
    return path


class _Counters:
    """Calls of the solution's autograd functions and launches of every Triton kernel during a
    run, by name (the components that ran)."""

    FUNCTIONS = {
        "lora_node": ("lora", "LoraProjectionTorch"),
        "fused_qkv": ("fused_proj", "FusedProjections"),
        "group_linear": ("qgroup", "GroupLinear"),
        "varlen_attention": ("unpad", "VarlenRowsAttention"),
        "rows_attention": ("unpad", "RowsAttention"),
        "add_norm": ("rmsnorm", "AddRMSNorm"),
        "norm": ("rmsnorm", "RMSNorm"),
        "rope": ("rope", "RoPE"),
        "head": ("fused_head", "FusedHeadLossVocab"),
        "swiglu_inplace": ("swiglu_inplace", "SwiGLUInPlaceGrad"),
        "mlp_save_up": ("mlp_save_up", "SwiGLUSaveUp"),
        "mlp_recompute": ("mlp_recompute", "MLPRecompute"),
    }

    def __init__(self):
        self.counts = collections.Counter()
        self.undo = []

    def _wrap(self, owner, attr, key):
        original = vars(owner).get(attr, _MISSING)
        bound = getattr(owner, attr)
        counts = self.counts

        def counted(*args, **kwargs):
            counts[key] += 1
            return bound(*args, **kwargs)

        setattr(owner, attr, counted)
        self.undo.append((owner, attr, original))

    def __enter__(self):
        import importlib

        from triton.runtime.jit import JITFunction

        for key, (module, cls) in self.FUNCTIONS.items():
            self._wrap(getattr(importlib.import_module(f"solution.{module}"), cls), "apply", key)
        from solution import host, lora, qgroup

        self._wrap(lora._half, "refresh", "lora_cast_refresh")
        self._wrap(qgroup.LayerGroup, "dequantize_one", "split_dequant")
        self._wrap(host, "queue_trl_metrics", "deferred")
        self._wrap(host, "install_model_fastpaths", "bookkeeping")
        run = JITFunction.run
        counts = self.counts

        def counted_run(jit, *args, **kwargs):
            counts["triton:" + getattr(getattr(jit, "fn", None), "__name__", "?")] += 1
            return run(jit, *args, **kwargs)

        JITFunction.run = counted_run
        self.undo.append((JITFunction, "run", vars(JITFunction).get("run", _MISSING)))
        return self.counts

    def __exit__(self, *exc):
        for owner, attr, original in reversed(self.undo):
            if original is _MISSING:
                delattr(owner, attr)
            else:
                setattr(owner, attr, original)
        self.undo.clear()


@contextlib.contextmanager
def _head_grad(store):
    """The gradient of the hidden states the fused head gets in the first forward (step 0)."""
    from solution import patches

    original = patches.fused_head_loss

    def fused_head_loss(hidden, *args, **kwargs):
        if "dh0" not in store and hidden.requires_grad:
            store["dh0"] = None

            def hook(grad):
                store["dh0"] = grad.detach().cpu().clone()

            hidden.register_hook(hook)
        return original(hidden, *args, **kwargs)

    patches.fused_head_loss = fused_head_loss
    try:
        yield
    finally:
        patches.fused_head_loss = original


def _group_dequant_exact(model):
    """Every packed layer's one-launch dequantization against bitsandbytes' dequantize_4bit of
    each projection: the number of fp16 values that differ (None: no layer was packed)."""
    import bitsandbytes.functional as bnbf

    groups = {}
    for module in model.modules():
        tag = getattr(module, "_solution_group", None)
        if tag is not None:
            groups.setdefault(id(tag[0]), (tag[0], []))[1].append((tag[1], module))
    if not groups:
        return None
    differing = 0
    with torch.no_grad():
        for group, members in groups.values():
            full = group.dequantize_all()
            for idx, module in members:
                ref = bnbf.dequantize_4bit(module.weight.data, module.weight.quant_state)
                ours = full[group.elem_bounds[idx]:group.elem_bounds[idx + 1]].view(ref.shape)
                differing += int((ours != ref.to(torch.float16)).sum())
                one = group.dequantize_one(idx)  # the split path's launch
                differing += int((one != ref.to(torch.float16)).sum())
    return differing


def _sft_config(SFTConfig, **kwargs):
    """SFTConfig with the arguments this TRL version has (`max_seq_length` before `max_length`;
    options it does not know are left at their behaviour)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(SFTConfig)}
    if "max_length" not in names and "max_seq_length" in names:
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in names})


def _train(ckpt, rows, solution_on, dropout=0.05):
    import inspect

    import peft
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    class Recorder(TrainerCallback):
        def __init__(self):
            self.grads0, self.skipped, self.scales = None, [], []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if state.global_step == 0:
                self.grads0 = {n: p.grad.detach().cpu().clone()
                               for n, p in model.named_parameters() if p.requires_grad}

        def on_optimizer_step(self, args, state, control, **kwargs):
            accelerator = recorder_trainer[0].accelerator
            self.skipped.append(bool(accelerator.optimizer_step_was_skipped))
            scaler = getattr(accelerator, "scaler", None)
            self.scales.append(float(scaler.get_scale()) if scaler is not None else None)

    started = time.perf_counter()
    set_seed(SEED)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_arg = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), quantization_config=quant,
                                                 device_map={"": 0}, attn_implementation="sdpa",
                                                 **{dtype_arg: torch.float16})
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    embed_dtype = str(model.get_input_embeddings().weight.dtype)
    _log("model loaded and prepared")
    set_seed(SEED)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    with tempfile.TemporaryDirectory() as out:
        args = _sft_config(
            SFTConfig, output_dir=out, max_steps=STEPS, per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=1, learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_steps=0, weight_decay=0.01, max_grad_norm=1.0, optim="adamw_torch",
            fp16=True, bf16=False, logging_steps=1, save_strategy="no", report_to="none",
            seed=SEED, data_seed=SEED, dataloader_num_workers=0, gradient_checkpointing=False,
            max_length=512, packing=False, padding_free=False, completion_only_loss=True,
            disable_tqdm=True, dataset_num_proc=None)
        tok_arg = ("processing_class" if "processing_class"
                   in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer")
        trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                             **{tok_arg: _tokenizer()})
        _log("trainer built")
        # TRL casts a quantized model's trainable weights to bf16; for fp16 training Soup casts
        # the LoRA weights back to fp32 before the optimizer exists
        # (soup_cli.utils.mixed_precision.align_trainable_dtype_for_fp16): the same here.
        gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and "lora_" in name and param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float32)
                if "lora_B" in name:  # non-zero from the start: step 0 runs the whole adapter
                    param.copy_(torch.randn(param.shape, generator=gen, device=param.device)
                                * 0.02)
        recorder_trainer = [trainer]
        recorder = Recorder()
        trainer.add_callback(recorder)
        # the exact fp32 loss of every step (older transformers log it rounded to 4 digits)
        losses, training_step = [], trainer.training_step

        def recorded_step(*args, **kwargs):
            loss = training_step(*args, **kwargs)
            losses.append(loss.detach().float().clone())
            return loss

        trainer.training_step = recorded_step
        lora_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
        adapter0 = {k: v.detach().cpu().clone()
                    for k, v in get_peft_model_state_dict(model).items()}
        store = {}
        with contextlib.ExitStack() as stack:
            counts = stack.enter_context(_Counters()) if solution_on else collections.Counter()
            if solution_on:
                stack.enter_context(_head_grad(store))
            trainer.train()
        _log("trained")
        history = [h for h in trainer.state.log_history if "loss" in h]
        result = {
            "loss": [float(loss) for loss in losses],
            "grad_norm": [h.get("grad_norm") for h in history],
            "learning_rate": [h.get("learning_rate") for h in history],
            "entropy": [h.get("entropy") for h in history],
            "num_tokens": [h.get("num_tokens") for h in history],
            "mean_token_accuracy": [h.get("mean_token_accuracy") for h in history],
            "total_flos": float(trainer.state.total_flos),
            "skipped": recorder.skipped,
            "scales": recorder.scales,
            "grads0": recorder.grads0,
            "adapter0": adapter0,
            "adapter": {k: v.detach().cpu().clone()
                        for k, v in get_peft_model_state_dict(model).items()},
            "dh0": store.get("dh0"),
            "counts": dict(counts),
            "embed_dtype": embed_dtype,
            "lora_dtypes": lora_dtypes,
            "group_dequant_differing": _group_dequant_exact(model) if solution_on else None,
            "seconds": time.perf_counter() - started,
        }
        if solution_on:
            from solution import fp16_embed

            result["counts"]["fp16_embed"] = int(embed_dtype == "torch.float16")
            result["counts"]["deferred"] = result["counts"].get("deferred", 0)
            del fp16_embed._applied[:]
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _worker(mode, out_dir):
    out = Path(out_dir)
    # one GPU: with several visible the Trainer would wrap the model in DataParallel
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.init()
    _log(f"worker {mode}: CUDA up")
    import solution

    if mode not in CONFIGS:
        solution.apply()  # first, before transformers / PEFT / TRL are imported (the harness)
    rows = _rows()
    ckpt = _make_checkpoint(out / f"checkpoint_{mode}")
    _log("checkpoint written")
    results = {}
    if mode in CONFIGS:
        _check_flags_match_modules()
        pristine = _snapshot()
        _log("originals saved")
        if mode == "main":
            results["baseline"] = _train(ckpt, rows, solution_on=False)
            results["baseline_p0"] = _train(ckpt, rows, solution_on=False, dropout=0.0)
        base_env = dict(os.environ)
        for name, switches in CONFIGS[mode].items():
            _restore(pristine)
            os.environ.clear()
            os.environ.update(base_env, **switches)
            _set_flags()
            solution.apply()
            dropout = 0.0 if name.endswith("_p0") else 0.05
            results[name] = _train(ckpt, rows, solution_on=True, dropout=dropout)
            print(f"[parity] {name}: {results[name]['seconds']:.1f} s", flush=True)
        if mode == "main":
            _restore(pristine)
            os.environ.clear()
            os.environ.update(base_env)
            _set_flags()
            results["baseline_again"] = _train(ckpt, rows, solution_on=False)
    else:
        results["full"] = _train(ckpt, rows, solution_on=True)
    torch.save(results, out / f"{mode}.pt")


# --------------------------------------------------------------------------------------------
# The tests (in the pytest process): start the workers, compare their results.
# --------------------------------------------------------------------------------------------

def _worker_slots():
    """How many workers run at once: one per CPU and per 3 GiB of available memory (each holds
    torch, transformers, PEFT, TRL and a CUDA context, about 2.5 GiB), at most all of them
    (Kaggle's T4 machines: 2 vCPUs, 13 GiB)."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    slots = min(len(WORKERS), cpus or 1)
    try:
        with open("/proc/meminfo") as meminfo:
            kib = next(int(line.split()[1]) for line in meminfo
                       if line.startswith("MemAvailable:"))
        slots = min(slots, kib // (3 * 2**20))
    except (OSError, StopIteration, ValueError):
        pass
    return max(1, slots)


def _failure(mode, code, log_path):
    """One line naming the worker and its last error (notebooks cut long lines), then the log."""
    text = log_path.read_text(errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    errors = [line for line in lines if "Error" in line or "Exception" in line]
    cause = (errors or lines or ["no output"])[-1].strip()
    return f"worker {mode} exited with {code}: {cause[:300]}\n--- its log:\n{text[-6000:]}"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("parity")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLUTION_")}
    env.update(BASE_ENV)
    # the parent's GPU only: on a machine with several (Kaggle's have two T4s) the Trainer of a
    # worker would otherwise wrap the model in DataParallel
    env.setdefault("CUDA_VISIBLE_DEVICES", str(torch.cuda.current_device()))
    gc.collect()
    torch.cuda.empty_cache()  # what earlier tests left cached in this process, for the workers
    pending, running, failed = list(WORKERS), {}, []
    slots = _worker_slots()
    while pending or running:
        while pending and len(running) < slots:
            mode = pending.pop(0)
            log = open(out / f"{mode}.log", "w")
            proc = subprocess.Popen(
                [sys.executable, "-m", "solution.test_parity", mode, str(out)], cwd=REPO,
                env={**env, **WORKERS[mode]}, stdout=log, stderr=subprocess.STDOUT)
            running[mode] = (proc, log, time.monotonic())
        time.sleep(0.5)
        for mode, (proc, log, started) in list(running.items()):
            code = proc.poll()
            if code is None and time.monotonic() - started > WORKER_TIMEOUT:
                proc.kill()
                proc.wait()
                code = "timeout"
            if code is None:
                continue
            log.close()
            del running[mode]
            if code != 0:
                failed.append(_failure(mode, code, out / f"{mode}.log"))
                print(failed[-1])
    assert not failed, "\n".join(failed)
    results = {}
    for mode in WORKERS:
        loaded = torch.load(out / f"{mode}.pt", weights_only=True)
        for name, run in loaded.items():
            results[name if mode == "main" else f"{mode}:{name}"] = run
    return results


SCALARS = ("loss", "grad_norm", "learning_rate", "entropy", "num_tokens", "mean_token_accuracy",
           "total_flos", "skipped", "scales")
TENSORS = ("grads0", "adapter0", "adapter")


def _differences(a, b, with_head_grad=True):
    """Every compared quantity of two runs that is not exactly equal."""
    diffs = []
    for key in SCALARS:
        if a[key] != b[key]:
            diffs.append(f"{key}: {a[key]} != {b[key]}")
    for key in TENSORS:
        if a[key].keys() != b[key].keys():
            diffs.append(f"{key}: different names")
            continue
        for name in a[key]:
            x, y = a[key][name], b[key][name]
            if x.dtype != y.dtype or not torch.equal(x, y):
                err = float((x.double() - y.double()).abs().max())
                diffs.append(f"{key}[{name}]: max |diff| {err:.3g}")
    if with_head_grad and a["dh0"] is not None and b["dh0"] is not None:
        if a["dh0"].dtype != b["dh0"].dtype or not torch.equal(a["dh0"], b["dh0"]):
            diffs.append("dh0 (hidden gradient at the head, step 0)")
    return diffs


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def _rel_l2(x, y):
    x, y = x.double(), y.double()
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def _updates(run):
    return {name: run["adapter"][name].double() - run["adapter0"][name].double()
            for name in run["adapter"]}


def _ulps16(x, y):
    """Distance in fp16 ulps between two fp16 tensors (on the ordered integer line)."""
    def ordered(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(x) - ordered(y)).abs()


def test_harness_reproduces_itself(runs):
    """The same code twice gives the same run bit for bit (deterministic algorithms): the
    unpatched baseline again after apply() was undone, the solution again after every switch
    was flipped and apply() re-run; every run starts from the same adapter, no GradScaler step
    is skipped, and the LoRA weights are fp32 when training starts (as Soup leaves them). The
    comparisons below measure the code, not noise."""
    assert not _differences(runs["baseline"], runs["baseline_again"])
    assert not _differences(runs["full"], runs["full_again"])
    assert not _differences(runs["full"], runs["more:full"])  # another process: the same too
    start = runs["baseline"]["adapter0"]
    for name, run in runs.items():
        assert run["lora_dtypes"] == ["torch.float32"], name
        assert not any(run["skipped"]) and len(run["loss"]) == STEPS, name
        assert all(torch.equal(run["adapter0"][k], start[k]) for k in start), name


def test_every_component_runs_in_the_default_solution(runs):
    counts = runs["full"]["counts"]
    for key in ("fused_qkv", "lora_node", "group_linear", "varlen_attention", "add_norm",
                "norm", "rope", "head", "swiglu_inplace", "split_dequant", "fp16_embed",
                "deferred", "bookkeeping", "lora_cast_refresh"):
        assert counts.get(key, 0) > 0, key
    assert counts.get("rows_attention", 0) == 0
    for kernel in ("rmsnorm_kernel", "rope_kernel", "nf4_group_dequant_kernel",
                   "fused_head_vocab_kernel", "example_swiglu_fwd_kernel",
                   "example_swiglu_bwd_kernel"):
        assert counts.get(f"triton:{kernel}", 0) > 0, kernel
    assert runs["full"]["embed_dtype"] == "torch.float16"
    assert runs["baseline"]["embed_dtype"] == "torch.float32"


@pytest.mark.parametrize("component,config,reference,counter,where", BIT_IDENTICAL,
                         ids=[c for _, c, _, _, _ in BIT_IDENTICAL])
def test_bit_identical_component(runs, component, config, reference, counter, where):
    """Claimed bit-identical (see `where`): with the component switched off every logged value,
    the step-0 LoRA gradients, the trained adapter and the head's hidden gradient are equal."""
    full, run = _pair(runs, config, reference)
    if counter is not None and counter.startswith("-"):  # an opt-in: runs in `config` only
        assert full["counts"].get(counter[1:], 0) == 0, f"{counter[1:]} ran by default"
        assert run["counts"].get(counter[1:], 0) > 0, f"{counter[1:]} did not run"
    elif counter is not None:
        assert full["counts"].get(counter, 0) > 0, f"{counter} did not run"
        assert run["counts"].get(counter, 0) == 0, f"{counter} still ran"
    diffs = _differences(full, run)
    assert not diffs, f"{component} ({where}) is not bit-identical: " + "; ".join(diffs[:8])


def test_group_dequant_is_bitsandbytes_dequant(runs):
    """qgroup.py: the one-launch dequantization of every packed layer (and the per-projection
    launch of the split path) equals bitsandbytes' dequantize_4bit, value for value."""
    for name in ("full", "no_lora_node", "no_split_dequant"):
        assert runs[_key(name)]["group_dequant_differing"] == 0, name


def test_varlen_attention_vs_rows_attention(runs):
    """unpad.py: all rows in one launch per KV head vs one launch per row. Claimed: the forward
    bit for bit, the gradients bit for bit where the kernel's backward is deterministic."""
    full, rows = _pair(runs, "rows_attention")
    assert full["counts"].get("varlen_attention", 0) > 0
    assert rows["counts"].get("varlen_attention", 0) == 0
    assert rows["counts"].get("rows_attention", 0) > 0
    # the step-0 forward: loss and TRL's metrics
    for key in ("loss", "entropy", "mean_token_accuracy", "num_tokens"):
        assert full[key][0] == rows[key][0], key
    # with deterministic algorithms the backward kernel is deterministic: everything equal
    assert not _differences(full, rows)


def test_fused_qkv_projection(runs):
    """fused_proj.py: q|k|v as one GEMM on the packed layer buffer vs three. Claimed: not
    bit-identical (cuBLAS may order the base projections' accumulation differently, a fp16
    rounding now and then; the backward sums dq Wq + dk Wk + dv Wv in fp32 and rounds once), the
    adapter part and the generator state bit-identical (so the same dropout masks: compared
    directly, dropout on); loss curve within 0.015% on the T4 (first step 0.005%). Measured
    here on the RTX 4080: loss 7e-6, LoRA gradients 7e-4; asserted: first step 5e-5, curve
    1.5e-4 (the T4 figures above), gradients 1e-2."""
    full, ref = _pair(runs, "no_fused_qkv")
    assert full["counts"].get("fused_qkv", 0) > 0 and ref["counts"].get("fused_qkv", 0) == 0
    assert ref["counts"].get("lora_node", 0) > 0
    assert full["num_tokens"] == ref["num_tokens"]
    for key in ("loss", "entropy"):
        assert _rel(full[key][0], ref[key][0]) <= 5e-5, key
        for a, b in zip(full[key], ref[key]):
            assert _rel(a, b) <= 1.5e-4, key
    for a, b in zip(full["mean_token_accuracy"], ref["mean_token_accuracy"]):
        assert abs(a - b) <= 0.01
    for name, grad in full["grads0"].items():
        assert _rel_l2(grad, ref["grads0"][name]) <= 1e-2, name
    upd, upd_ref = _updates(full), _updates(ref)
    for name in upd:
        assert _rel_l2(upd[name], upd_ref[name]) <= 1e-1, name


def test_head_compaction(runs):
    """fused_head.py: the hidden-gradient GEMM over the labelled rows only vs over all rows.
    Claimed: loss, entropy and accuracy bit-identical; the hidden gradient within one fp16 ulp
    of its largest element plus rtol 1e-3 (atol = max|dH| * 2^-10). That bound is asserted and
    holds on both GPUs. Counted in ulps of each element, the difference depends on the kernel
    cuBLAS picks for the smaller GEMM: none here on the RTX 4080 (the kernel tests saw 1-3 ulp on
    up to 0.2% of the task's elements), up to 48 ulp on the T4 (small elements, still inside
    the bound above); so no per-element ulp count is asserted, only reported. Also asserted: the
    step-0 forward equal, the LoRA gradients within 1e-3 and the curve within 1e-4 (all values
    are reported together when one is over)."""
    full, ref = _pair(runs, "no_head_compact")
    for key in ("loss", "entropy", "mean_token_accuracy", "num_tokens"):
        assert full[key][0] == ref[key][0], key  # step 0: the same forward
    dh, dh_ref = full["dh0"], ref["dh0"]
    assert dh.dtype == dh_ref.dtype == torch.float16 and dh.shape == dh_ref.shape
    ulps = _ulps16(dh, dh_ref)
    measured = {
        "dH max ulp": int(ulps.max()),
        "dH elements differing": float((ulps > 0).double().mean()),
        "dH max |diff| / max|dH|": float((dh.float() - dh_ref.float()).abs().max()
                                         / dh_ref.float().abs().max()),
        "curve": max(_rel(a, b) for key in ("loss", "entropy")
                     for a, b in zip(full[key], ref[key])),
        "LoRA gradients": max(_rel_l2(g, ref["grads0"][n]) for n, g in full["grads0"].items()),
    }
    report = ", ".join(f"{k} {v:.3g}" for k, v in measured.items())
    torch.testing.assert_close(dh.float(), dh_ref.float(), rtol=1e-3,
                               atol=float(dh_ref.float().abs().max()) * 2**-10,
                               msg=lambda m: f"{m}\nmeasured: {report}")
    assert measured["curve"] <= 1e-4 and measured["LoRA gradients"] <= 1e-3, report


def test_head_rows_sum_order(runs):
    """fused_head.py: the head kernel's rows in units of 8 elements (the default) vs 1
    (SOLUTION_DISABLE_HEAD_ROWS=1). Claimed: not bit-identical, only the order of the in-row
    fp32 reductions changes: the last bits of the fp32 statistics, a few hidden-gradient
    elements by an fp16 ulp or two. Measured here on the RTX 4080: 3 of 739,328 elements, by up
    to 2 ulp; asserted: at most 2 ulp on at most 0.01% of them, the curve within 1e-4."""
    full, ref = _pair(runs, "head_rows_x1")
    assert full["counts"].get("triton:fused_head_vocab_kernel", 0) > 0
    for key in ("loss", "entropy"):
        assert _rel(full[key][0], ref[key][0]) <= 1e-6, key  # a few fp32 ulps
    assert full["mean_token_accuracy"][0] == ref["mean_token_accuracy"][0]
    assert full["dh0"].shape == ref["dh0"].shape
    ulps = _ulps16(full["dh0"], ref["dh0"])
    assert int(ulps.max()) <= 2 and float((ulps > 0).double().mean()) <= 1e-4
    for key in ("loss", "entropy"):
        for a, b in zip(full[key], ref[key]):
            assert _rel(a, b) <= 1e-4, key
    for name, grad in full["grads0"].items():
        assert _rel_l2(grad, ref["grads0"][name]) <= 1e-2, name


def test_fp16_norm_output_path(runs):
    """rmsnorm.py: the fused RMSNorm writing fp16 (projections, RoPE, the attention input and
    silu * up then computed from fp16; the backward from the fp16 x-hat) vs transformers' fp32
    RMSNorm. Claimed: not bit-identical, a divergence of order 1e-3 (loss curve 0.03% at the
    first step and 0.05% on average on the real run).

    Compared with LoRA dropout 0: with dropout on, the two runs draw different masks, because
    the dropout input (the norm output) is an fp16 tensor in one and an fp32 tensor in the other
    and `native_dropout` maps the Philox stream onto the elements by dtype-dependent vector
    widths (the same p and RNG state, an independent realization). That is asserted here too:
    it is a second source of the mask difference README section 4 attributes to unpad alone."""
    full, ref = runs["full_p0"], runs["no_rmsnorm_p0"]
    full_masked, ref_masked = _pair(runs, "no_rmsnorm")
    for run in (full, full_masked):
        assert run["counts"].get("norm", 0) > 0 and run["counts"].get("add_norm", 0) > 0
    for run in (ref, ref_masked):
        assert run["counts"].get("norm", 0) == 0 and run["counts"].get("add_norm", 0) == 0
    assert full["num_tokens"] == ref["num_tokens"]
    for key in ("loss", "entropy"):
        for a, b in zip(full[key], ref[key]):
            assert _rel(a, b) <= 1e-3, key
    for a, b in zip(full["mean_token_accuracy"], ref["mean_token_accuracy"]):
        assert abs(a - b) <= 0.01
    worst = max(_rel_l2(g, ref["grads0"][n]) for n, g in full["grads0"].items())
    assert worst <= 1e-2  # of order 1e-3
    # dropout on: the masks differ (fp16 vs fp32 dropout input), so the gradients differ by far
    # more than the numerics above
    masked = max(_rel_l2(g, ref_masked["grads0"][n]) for n, g in full_masked["grads0"].items())
    assert masked > 10 * worst


def test_solution_vs_unpatched_baseline(runs):
    """The whole solution against transformers' and PEFT's own code. With LoRA dropout 0 (the
    same computation, no mask realizations involved): loss and entropy within 1e-3, accuracy
    within 1%, num_tokens equal, the step-0 LoRA gradients within 1e-2 (relative L2), the
    adapter updates within 1e-1 after 4 AdamW steps (measured on the RTX 4080: 1e-5, 1.2e-3 and
    3e-2). With dropout on, the solution draws other
    masks (unpad: a [1, N, H] tensor; the fp16 norm output: an fp16 tensor), so the default
    runs are held to the harness's checks 5-6: first-step loss within 1%, mean relative gap of
    the loss, entropy and accuracy curves within 3%."""
    base, full = runs["baseline_p0"], runs["full_p0"]
    assert runs["baseline"]["counts"] == {} and base["counts"] == {}  # no solution code ran
    for run in (full, runs["no_unpad_p0"]):
        assert run["num_tokens"] == base["num_tokens"]
        for key in ("loss", "entropy"):
            for a, b in zip(run[key], base[key]):
                assert _rel(a, b) <= 1e-3, key
        for a, b in zip(run["mean_token_accuracy"], base["mean_token_accuracy"]):
            assert abs(a - b) <= 0.01
        for name, grad in run["grads0"].items():
            assert _rel_l2(grad, base["grads0"][name]) <= 1e-2, name
        upd, upd_base = _updates(run), _updates(base)
        for name in upd:  # AdamW's first steps turn sign flips of tiny gradients into lr steps
            assert _rel_l2(upd[name], upd_base[name]) <= 1e-1, name
    base, full = runs["baseline"], runs["full"]
    assert full["num_tokens"] == base["num_tokens"]
    assert _rel(full["loss"][0], base["loss"][0]) <= 0.01
    for key in ("loss", "entropy", "mean_token_accuracy"):
        gap = sum(_rel(a, b) for a, b in zip(full[key], base[key])) / STEPS
        assert gap <= 0.03, key


def test_startup_patches(runs):
    """startup.py (with the warm-up thread and the -O0 Triton helpers, as the harness runs it):
    apply() before transformers, PEFT and TRL are imported, in another process, gives the same
    run bit for bit as the in-process solution without them; the control process (the same
    without the start-up patches) separates them from the process boundary."""
    assert not _differences(runs["control:full"], runs["full"])
    assert not _differences(runs["startup:full"], runs["full"])


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2])
