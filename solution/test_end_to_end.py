"""End to end against an unpatched model: transformers, PEFT and bitsandbytes as they are.

Every other end-to-end reference in this suite is `patches.ORIGINAL_FORWARD`, which still runs
the patched decoder (RMSNorm, add+norm, RoPE, SwiGLU, the per-layer dequantization, the LoRA
node). Here one script runs twice, each time in a fresh process: without `solution.apply()`
(the reference) and with it, from an empty TRITON_CACHE_DIR as every jury run starts (every
kernel and Triton's C helpers are built inside the run). The model is the task's set-up in
small: a tied Qwen2 (hidden 512, 8 query heads on 2 KV heads, so the per-layer dequantization
packs its layers and splits the last one) with trained-looking weights (N(0, 0.08), logits with
a standard deviation above 2, so the losses depend on the context), loaded in 4-bit NF4 with
double quantization and fp16 compute, PEFT's prepare_model_for_kbit_training, LoRA r=16 /
alpha=32 on q_proj and v_proj (dropout 0: a padded [B, T, H] batch and the flat [1, N, H] one
cannot draw the same masks), padding removal with the variable-length attention
(SOLUTION_VARLEN=1, the T4 path). Compared, with the batches of test_robustness.py:

- every right-padded edge batch at the initial weights: loss, Soup's per-row losses and every
  LoRA gradient (relative L2 error); the masks that are not right-padded take the padded
  fallback of the fused head and give the same values;
- 3 training steps as the Trainer takes them (fp16 autocast, GradScaler, gradient clipping at
  1.0, AdamW): the same values, the GradScaler scale after each step (the same skipped steps)
  and the LoRA updates;
- the eval loss and per-row losses (eval mode, no_grad, labels: transformers' own head after
  every decoder patch) before and after training.

Tolerances: 4x the largest error measured on an RTX 4080 (see TOL), where the patched run is
as close to the reference as the reference with eager attention instead of SDPA is. Negative
control: the patched run with the row bounds merged (attention across the rows of the batch)
misses them by more than 10x.

Run on a GPU: `python -m pytest solution/test_end_to_end.py -q`.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import torch

from solution.test_robustness import CASES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

REPO = pathlib.Path(__file__).resolve().parents[1]
FALLBACK = ["left_padded", "mask_with_hole"]  # not right-padded: the padding removal turns off
EDGE = [case for case in CASES if case not in FALLBACK]
TRAIN = ["row_without_labels", "full_and_short_rows", "odd_token_counts"]
# batches whose rows would see each other with merged row bounds (two rows or more, labels)
CONTROL = [case for case in EDGE if len(CASES[case][0]) > 1 and case != "no_label_in_batch"]
GRAD_SCALE = 4096.0  # the loss scale of the backward passes outside the training steps

# 4x the largest error measured on an RTX 4080 (sm89, torch 2.11; the runs are deterministic,
# the same errors to the last digit run after run). Losses: absolute errors (the losses are
# 6-10); gradients and LoRA updates: relative L2 errors per tensor, the largest of the 8.
# For scale, the reference's own rounding: the same reference with eager instead of SDPA
# attention is as far from it (in brackets) as the patched run is.
TOL = {
    # at the initial weights (edge and fallback batches, training step 0); measured
    # loss 1.53e-3 [1.81e-3], per-row losses 4.69e-3 [5.93e-3], gradients 6.02e-3 [8.53e-3]
    "loss": 6.2e-3, "rows": 1.9e-2, "grad": 2.5e-2,
    # training steps 1-2 (the weights already differ by the previous updates): per-row losses
    # 1.46e-2 [7.97e-3], gradients 1.07e-2 [1.23e-2]; the LoRA updates of the 3 steps 3.92e-2
    # [4.06e-2] (AdamW's first steps are about lr * sign(g): near-zero components can flip)
    "train_rows": 6e-2, "train_grad": 4.3e-2, "delta": 1.6e-1,
    # eval loss and per-row losses before / after training: 1.54e-3 / 2.39e-3 [2.07e-3]
    "eval": 9.6e-3,
}


def _checkpoint(path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=600, hidden_size=512, intermediate_size=1024,
                         num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                         max_position_embeddings=1024, tie_word_embeddings=True,
                         attn_implementation="sdpa")
    model = Qwen2ForCausalLM(config)
    gen = torch.Generator().manual_seed(41)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith("norm.weight"):  # trained-looking scales, not the init's
                param.copy_(torch.randn(param.shape, generator=gen) * 0.08)
        model.model.norm.weight.fill_(1.3)  # logits with a standard deviation above 2
    model.to(torch.bfloat16).save_pretrained(path)  # bf16, as the real checkpoint


def _qlora_model(ckpt, attn="sdpa"):
    """The task's set-up on the checkpoint: 4-bit NF4 load with double quantization and fp16
    compute (the T4 path), PEFT's prepare_model_for_kbit_training (ours once apply() ran),
    LoRA r=16 / alpha=32 on q_proj and v_proj, dropout 0."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), quantization_config=quant,
                                                 device_map={"": 0}, dtype=torch.float16,
                                                 attn_implementation=attn)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():  # the same adapter in every process, whatever else drew numbers
        for n, p in model.named_parameters():  # (PEFT starts B at 0: A would get no gradient)
            if p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=gen) * (0.04 if "lora_A" in n else 0.05))
    return model


def _child(mode, ckpt, out_path, attn="sdpa"):
    """One run of the comparison: mode "reference" (nothing of ours imported) or "patched"
    (`solution.apply()` first). Writes the measured values to `out_path`. (`attn`: the
    reference's attention implementation; "eager" measures the reference's own rounding.)"""
    import collections

    import torch.nn.functional as F

    counts = collections.Counter()
    if mode == "reference":
        if not os.path.exists(os.path.join(ckpt, "config.json")):
            _checkpoint(ckpt)
    else:
        from solution import triton_build

        build = triton_build._build  # installed as Triton's build hook by apply()
        triton_build._build = lambda *a, **k: counts.update(["builds"]) or build(*a, **k)
        import solution

        solution.apply()
        from solution import fused_proj, lora, patches, unpad

        for cls, name in ((unpad.VarlenRowsAttention, "varlen"), (lora.LoraProjectionTorch, "lora"),
                          (fused_proj.FusedProjections, "qkv")):  # which of our nodes ran
            cls.apply = (lambda fn, name: lambda *a: counts.update([name]) or fn(*a))(cls.apply,
                                                                                      name)

    from solution.test_robustness import _batch

    model = _qlora_model(ckpt, attn)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    initial = {n: p.detach().clone() for n, p in named}
    res = {"edge": {}, "fallback": {}, "train": [], "eval": {}, "control": {}, "fused": {},
           "nodes": {}, "unpad_off": {}, "gpu": torch.cuda.get_device_name()}

    def rows_of(token_loss, shift):
        return token_loss.sum(1) / (shift >= 0).sum(1).clamp(min=1)

    def forward(case, key):
        lengths, kwargs = CASES[case]
        input_ids, mask, labels = _batch(lengths, **kwargs)
        num_items = (labels[:, 1:] != -100).sum()
        before = dict(counts)
        with torch.autocast("cuda", dtype=torch.float16):  # use_cache=False: as TRL calls it
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels,
                        num_items_in_batch=num_items, use_cache=False)
        res["fused"][key] = out.logits is None
        res["nodes"][key] = {k: counts[k] - before.get(k, 0) for k in ("varlen", "lora", "qkv")}
        shift = F.pad(labels, (0, 1), value=-100)[:, 1:]
        if out.logits is None:  # the fused head: Soup's rewind reads these token losses
            token_loss = patches._pending_step[-1].token_loss.detach().float()
        else:
            token_loss = F.cross_entropy(out.logits.float().flatten(0, 1), shift.flatten(),
                                         ignore_index=-100, reduction="none").view(shift.shape)
        return out.loss, rows_of(token_loss, shift).detach()

    def grads(scale=1.0):
        return {n: (p.grad.float() / scale) if p.grad is not None else torch.zeros_like(p)
                for n, p in named}

    def values(case, key):
        for p in params:
            p.grad = None
        loss, rows = forward(case, key)
        (loss * GRAD_SCALE).backward()
        return loss.detach().float(), rows, grads(GRAD_SCALE)

    def evaluate(key):
        input_ids, mask, labels = _batch([30, 12, 25, 7], seed=3)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        model.train()
        shift = F.pad(labels, (0, 1), value=-100)[:, 1:]
        logits = out.logits.float()
        token_loss = F.cross_entropy(logits.flatten(0, 1), shift.flatten(), ignore_index=-100,
                                     reduction="none").view(shift.shape)
        res["eval"][key] = (out.loss.float(), rows_of(token_loss, shift),
                            float(logits[mask.bool()].std()))

    model.train()
    evaluate("before")
    for case in EDGE:
        res["edge"][case] = values(case, f"edge/{case}")
    for case in FALLBACK:
        if mode != "reference":
            patches._unpad["checked"] = 0  # the right-padding check of the first batches runs
        res["fallback"][case] = values(case, f"fallback/{case}")
        if mode != "reference":
            res["unpad_off"][case] = not patches._unpad["enabled"]
            patches._unpad.update(enabled=True, checked=0)  # a fresh run for the steps below

    if mode != "reference":  # the engaged-line counters (engaged.py) around the training steps
        from solution import engaged, state

        fell_back, ran = dict(engaged.FELL_BACK), dict(engaged.RAN)
    opt = torch.optim.AdamW(params, lr=2e-3)
    scaler = torch.amp.GradScaler("cuda")
    for i, case in enumerate(TRAIN):
        opt.zero_grad(set_to_none=True)
        loss, rows = forward(case, f"train/{i}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        step_grads = grads()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        res["train"].append((loss.detach().float(), rows, step_grads, float(scaler.get_scale())))
    res["delta"] = {n: (p.detach() - initial[n]).float() for n, p in named}
    if mode != "reference":
        res["training"] = {
            "fell_back": {k: v - fell_back.get(k, 0) for k, v in engaged.FELL_BACK.items()
                          if v != fell_back.get(k, 0)},
            "ran": {k: v - ran.get(k, 0) for k, v in engaged.RAN.items()},
            "disabled": dict(state.DISABLED), "engaged": engaged.line("training")}
    evaluate("after")

    if mode != "reference":
        # Negative control: the initial adapter again, the rows of each batch merged into one
        # (attention across rows; RoPE positions still restart at each row).
        with torch.no_grad():
            for n, p in named:
                p.copy_(initial[n])
        rows_context = patches.rows_context
        patches.rows_context = lambda bounds, device=None: rows_context(
            [(0, bounds[-1][1])], device=device)
        for case in CONTROL:
            res["control"][case] = values(case, f"control/{case}")
        res["builds"] = counts["builds"]
        res["disabled"] = dict(state.DISABLED)
    torch.save(res, out_path)


def _run(mode, ckpt, out, cache, attn="sdpa"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLUTION_")}
    env.update(SOLUTION_VARLEN="1", SOLUTION_WARMUP_THREAD="0", TRITON_CACHE_DIR=str(cache))
    code = (f"import sys; sys.path.insert(0, {str(REPO)!r})\n"
            "from solution.test_end_to_end import _child\n"
            f"_child({mode!r}, {str(ckpt)!r}, {str(out)!r}, {attn!r})\n")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=1200, cwd=str(REPO))
    assert proc.returncode == 0, f"{mode} run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-6000:]}"
    return torch.load(out, weights_only=False)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("end_to_end")
    ckpt = tmp / "checkpoint"
    results = {}
    for mode in ("reference", "patched"):  # the reference also writes the checkpoint
        cache = tmp / f"triton_cache_{mode}"
        cache.mkdir()
        results[mode] = _run(mode, ckpt, tmp / f"{mode}.pt", cache)
    return results["reference"], results["patched"]


def _rel(a, b):
    """Relative L2 error of `a` against `b` (the norm of `a` where `b` is 0)."""
    a, b = a.double(), b.double()
    norm = float(b.norm())
    return float((a - b).norm()) / norm if norm > 0 else float(a.norm())


def _errors(ours, ref):
    """(loss, per-row losses, gradients) errors of one batch: absolute, largest absolute,
    largest relative L2 over the LoRA tensors."""
    loss, rows, grads = ours[:3]
    ref_loss, ref_rows, ref_grads = ref[:3]
    if torch.isnan(ref_loss):  # no labelled token: 0 / 0, as the base gives
        assert torch.isnan(loss)
        e_loss = 0.0
    else:
        e_loss = abs(float(loss) - float(ref_loss))
    assert grads.keys() == ref_grads.keys() and len(grads) == 8  # q/v A/B in two layers
    return (e_loss, float((rows - ref_rows).abs().max()),
            max(_rel(grads[n], ref_grads[n]) for n in grads))


def _check(errors, keys):
    """errors: {batch: (error per key)}; asserts the largest error of each key against TOL."""
    report = "\n".join(f"{name}: " + ", ".join(f"{k} {e:.2e}" for k, e in zip(keys, errs))
                       for name, errs in errors.items())
    print("\n" + report)
    for i, k in enumerate(keys):
        worst = max(errs[i] for errs in errors.values())
        assert worst <= TOL[k], f"{k}: {worst:.3e} > {TOL[k]:.1e}\n{report}"


def test_edge_batches_match_the_unpatched_model(runs):
    ref, ours = runs
    assert ours["gpu"] == ref["gpu"]
    errors = {}
    for case in EDGE + FALLBACK:
        right_padded = case in EDGE
        key = f"{'edge' if right_padded else 'fallback'}/{case}"
        assert ours["fused"][key] and not ref["fused"][key]
        nodes = ours["nodes"][key]  # the variable-length attention in both layers (padding
        assert nodes["varlen"] == (2 if right_padded else 0)  # removal off: none), q and v
        assert nodes["qkv"] >= 1 and nodes["lora"] + 2 * nodes["qkv"] == 4  # adapted by ours
        if not right_padded:
            assert ours["unpad_off"][case]
        errors[case] = _errors(ours["edge" if right_padded else "fallback"][case],
                               ref["edge" if right_padded else "fallback"][case])
    _check(errors, ("loss", "rows", "grad"))


def test_training_steps_match_the_unpatched_model(runs):
    ref, ours = runs
    errors = {}
    for i, case in enumerate(TRAIN):
        assert ours["fused"][f"train/{i}"] and not ref["fused"][f"train/{i}"]
        assert ours["train"][i][3] == ref["train"][i][3], "GradScaler scales differ"
        finite = all(torch.isfinite(g).all() for g in ref["train"][i][2].values())
        assert finite == all(torch.isfinite(g).all() for g in ours["train"][i][2].values())
        if finite:  # (a step the GradScaler skips has inf gradients in both runs)
            errors[f"step {i} ({case})"] = _errors(ours["train"][i], ref["train"][i])
    assert errors, "every step skipped by the GradScaler"
    # a healthy run: the fast paths ran in every step, none fell back or was disabled (state.py)
    training = ours["training"]
    print("\n" + training["engaged"])
    assert training["fell_back"] == {} and training["disabled"] == {}, training
    assert "disabled=" not in training["engaged"] and not ours["disabled"]
    assert training["ran"]["head"] == training["ran"]["unpad"] == len(TRAIN)
    for name in ("varlen", "nocat", "lora", "rmsnorm", "addnorm", "rope", "group", "split",
                 "swiglu_inplace", "lora_cast"):
        assert training["ran"].get(name, 0) > 0, (name, training["engaged"])
    first = {name: errs for name, errs in errors.items() if name.startswith("step 0")}
    _check(first, ("loss", "rows", "grad"))
    _check(errors, ("loss", "train_rows", "train_grad"))
    assert all(float(d.norm()) > 0 for d in ref["delta"].values())  # the adapter was trained
    deltas = max(_rel(ours["delta"][n], ref["delta"][n]) for n in ref["delta"])
    _check({"LoRA updates of the 3 steps": (deltas,)}, ("delta",))


def test_eval_loss_matches_the_unpatched_model(runs):
    ref, ours = runs
    assert ref["eval"]["before"][2] >= 2.0, "logits too flat to tell contexts apart"
    errors = {}
    for when in ("before", "after"):
        (loss, rows, _), (ref_loss, ref_rows, _) = ours["eval"][when], ref["eval"][when]
        errors[f"eval {when}"] = (max(abs(float(loss) - float(ref_loss)),
                                      float((rows - ref_rows).abs().max())),)
    _check(errors, ("eval",))
    # the training moved the eval values far more than the tolerance (0.11 measured), so the
    # comparison after training sees the trained adapter (no stale fp16 copy of it)
    change = max(abs(float(ref["eval"]["after"][0]) - float(ref["eval"]["before"][0])),
                 float((ref["eval"]["after"][1] - ref["eval"]["before"][1]).abs().max()))
    assert change > 5 * TOL["eval"], change


def test_merged_row_bounds_miss_the_tolerances(runs):
    """Negative control: the same patched run with attention across the rows of each batch
    (measured: per-row losses 0.57-3.7 off, gradients 0.98-1.48 relative L2) fails them."""
    ref, ours = runs
    for case in CONTROL:
        assert ours["fused"][f"control/{case}"]
        e_loss, e_rows, e_grad = _errors(ours["control"][case], ref["edge"][case])
        print(f"control {case}: loss {e_loss:.2e}, rows {e_rows:.2e}, grad {e_grad:.2e}")
        assert e_rows > 10 * TOL["rows"] and e_grad > 10 * TOL["grad"], case


def test_triton_helpers_were_built_in_the_run(runs):
    """The patched run started from an empty Triton cache: Triton's C helpers (cuda_utils and
    the kernel launchers) went through our -O0 build hook (triton_build._build) and worked."""
    assert runs[1]["builds"] >= 2


def test_triton_build_hook_in_an_empty_cache(tmp_path):
    """triton_build._build never runs with the warm Triton cache of a development machine but in
    every jury run (empty cache): in a fresh process with an empty TRITON_CACHE_DIR it builds
    cuda_utils and the first kernel's launcher, and that kernel then runs and is right."""
    code = (f"import sys; sys.path.insert(0, {str(REPO)!r})\n"
            "import torch\n"
            "from solution import triton_build\n"
            "names, build = [], triton_build._build\n"
            "triton_build._build = lambda *a, **k: names.append(a[0]) or build(*a, **k)\n"
            "assert triton_build.install()\n"
            "from solution.rmsnorm import rmsnorm\n"
            "x = torch.randn(3, 77, 512, device='cuda') * 3\n"
            "w = 1 + 0.1 * torch.randn(512, device='cuda')\n"
            "out = rmsnorm(x, w, 1e-6, torch.float32)\n"
            "ref = w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))\n"
            "torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)\n"
            "assert len(names) >= 2, names\n"
            "print('built', names)\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOLUTION_")}
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton_cache")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=600, cwd=str(REPO))
    assert proc.returncode == 0 and "built" in proc.stdout, proc.stdout + proc.stderr[-4000:]


# --- a real trl.SFTTrainer run: the fused path against TRL's own ------------------------------

def _sft_rows(model):
    """4 rows of different lengths, mostly the model's own greedy continuation (every third
    token random instead): about 60% of the targets are the argmax, so the accuracy can be
    wrong both ways (a head reporting every token wrong, or every token right, fails)."""
    gen = torch.Generator().manual_seed(5)
    rows = []
    model.eval()
    for length in (24, 13, 31, 9):
        ids = torch.randint(0, 598, (4,), generator=gen).tolist()
        while len(ids) < length:
            if len(ids) % 3 == 2:
                ids.append(int(torch.randint(0, 598, (1,), generator=gen)))
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=torch.tensor([ids], device="cuda")).logits
            ids.append(int(logits[0, -1].argmax()))
        rows.append(ids)
    model.train()
    return rows


def _sft_trainer(model, rows, out_dir, callbacks=None):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer

    vocab = {f"t{i}": i for i in range(600)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="t0")), unk_token="t0",
        eos_token="t598", pad_token="t599")
    args = SFTConfig(output_dir=str(out_dir), fp16=True, bf16=False, max_steps=2,
                     logging_steps=1, report_to=[], per_device_train_batch_size=4,
                     learning_rate=2e-3, save_strategy="no", seed=0, max_length=512,
                     gradient_checkpointing=False, dataloader_num_workers=0, disable_tqdm=True)
    trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                         train_dataset=Dataset.from_dict({"input_ids": rows}), callbacks=callbacks)
    for name, param in trainer.model.named_parameters():
        # TRL 0.29 stores QLoRA adapters in bf16; with fp16 Soup casts them back to fp32 before
        # the optimizer exists (soup_cli.utils.mixed_precision.align_trainable_dtype_for_fp16)
        if param.requires_grad and "lora_" in name and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
    return trainer


@pytest.fixture
def sft_setup(monkeypatch, tmp_path):
    """solution.apply() in this process (no warm-up thread, no gc.freeze: nothing outlives the
    test), the checkpoint, and the rows generated from the reference model."""
    import solution
    from solution import patches, unpad

    monkeypatch.setenv("SOLUTION_WARMUP_THREAD", "0")
    monkeypatch.setenv("SOLUTION_DISABLE_GC_FREEZE", "1")
    monkeypatch.setenv("SOLUTION_VARLEN", "1")
    solution.apply()
    monkeypatch.setattr(unpad, "_VARLEN", True)
    monkeypatch.setitem(patches._unpad, "enabled", True)
    monkeypatch.setitem(patches._unpad, "checked", 0)
    _checkpoint(tmp_path / "checkpoint")
    rows = _sft_rows(_qlora_model(tmp_path / "checkpoint"))
    return tmp_path, rows


def test_sft_trainer_logs_trl_values_from_the_fused_step(monkeypatch, sft_setup):
    """2 steps of trl.SFTTrainer (fp16, GradScaler, AdamW, logging every step) through the fused
    path (patch_sft_compute_loss, deferred metrics, trainer bookkeeping) against the same run
    with the fused path switched off (TRL computes loss, entropy and accuracy from real
    logits): logged loss within 2e-3, entropy within 1e-3, num_tokens exact, accuracy within
    one token of the labelled ones (measured on an RTX 4080: loss 1.3e-4, entropy 1.2e-4,
    accuracy equal, 40 of 73)."""
    from solution import patches

    tmp_path, rows = sft_setup
    n_labelled = sum(len(row) - 1 for row in rows)
    logs = {}
    for name in ("trl", "fused"):
        with monkeypatch.context() as m:
            if name == "trl":
                m.setattr(patches, "_fused_forward_applies", lambda *args: False)
            trainer = _sft_trainer(_qlora_model(tmp_path / "checkpoint"), rows, tmp_path / name)
            trainer.train()
        logs[name] = [e for e in trainer.state.log_history if "loss" in e]
        step = getattr(trainer, patches.STEP_ATTR, None)
        if name == "trl":
            assert step is None
        else:
            assert step is not None and int(step.labelled) == n_labelled
            wrapped = trainer.model_wrapped if trainer.model_wrapped is not None else trainer.model
            assert "_solution_params" in wrapped.__dict__  # the bookkeeping fast paths
    print(logs)
    assert len(logs["trl"]) == len(logs["fused"]) == 2
    first_accuracy = logs["trl"][0]["mean_token_accuracy"]
    assert 0.3 < first_accuracy < 0.9  # planted hits and misses: the check can fail both ways
    for ours, ref in zip(logs["fused"], logs["trl"]):
        assert abs(ours["loss"] - ref["loss"]) <= 2e-3
        assert abs(ours["entropy"] - ref["entropy"]) <= 1e-3
        assert ours["num_tokens"] == ref["num_tokens"]
        assert abs(ours["mean_token_accuracy"] - ref["mean_token_accuracy"]) <= 1 / n_labelled


def test_sft_compute_loss_falls_back_to_trl(monkeypatch, sft_setup):
    """A label smoother, a custom loss function, the Liger kernel or eval mode take TRL's own
    compute_loss (the logits exist); a fused path that does not run although expected turns
    the fused head off and TRL's compute_loss runs the step (nothing is left unlogged); a
    failing Trainer.train still flushes the deferred metrics and re-raises the training
    error."""
    import warnings

    import transformers

    from solution import host, patches, state

    tmp_path, rows = sft_setup
    trainer = _sft_trainer(_qlora_model(tmp_path / "checkpoint"), rows, tmp_path / "out")
    model = trainer.model
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(4)])

    def call():
        inputs = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            _, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
        return outputs, inputs

    model.train()
    assert call()[0].logits is None  # the fused path by default
    with monkeypatch.context() as m:
        m.setattr(trainer, "label_smoother", transformers.trainer_pt_utils.LabelSmoother(0.1))
        assert call()[0].logits is not None
    with monkeypatch.context() as m:
        m.setattr(trainer, "compute_loss_func",
                  lambda outputs, labels, num_items_in_batch=None: outputs.logits.float().mean())
        assert call()[0].logits is not None
    with monkeypatch.context() as m, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TRL: Liger "did not return token_accuracy"
        m.setattr(trainer.args, "use_liger_kernel", True)
        assert call()[1].get("return_token_accuracy") is True  # TRL's own branch ran
    model.eval()
    assert call()[0].logits is not None
    model.train()

    # the wrapper expects the fused path, the forward then does not take it: the guard's error
    # turns the fused head off (state.py) and TRL's own compute_loss runs the step instead
    expected, saved = [], dict(state.DISABLED)
    try:
        state.DISABLED.clear()
        with monkeypatch.context() as m:
            m.setattr(patches, "_fused_forward_applies",
                      lambda *args: expected.append(1) or len(expected) == 1)
            assert call()[0].logits is not None
        assert state.DISABLED == {
            "fused_head": "RuntimeError: fused head did not run although it was expected to"}
    finally:
        state.DISABLED.clear()
        state.DISABLED.update(saved)

    class Fail(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            raise KeyboardInterrupt  # after the step's metrics were queued, before its log

    assert patches._host["defer"]
    trainer = _sft_trainer(_qlora_model(tmp_path / "checkpoint"), rows, tmp_path / "fail",
                           callbacks=[Fail()])
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert not host._metric_queue  # flushed on the way out
    assert len(trainer._metrics["train"]["entropy"]) == 1
    assert len(trainer._metrics["train"]["mean_token_accuracy"]) == 1
