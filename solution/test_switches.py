"""Every SOLUTION_* switch does what solution/__init__.py says: with SOLUTION_DISABLE_X=1 the
part X is not installed / its code path does not run, with an opt-in SOLUTION_X=1 it does.

Each case is a fresh process (module constants are read at import, patches are class-level):
`solution.apply()` with the switch set, then one training step of the task's pipeline on a
tiny tied Qwen2 (4-bit NF4 load in fp16, PEFT's prepare, LoRA r=16 on q/v, a right-padded
batch under fp16 autocast), with a counter on every component's entry point. The checks compare
against the same process without the switch, so each one is shown to be live there first.
SOLUTION_VARLEN=1 is set in every case (the T4's variable-length attention on any GPU).

Run on a GPU: `python -m pytest solution -q`.
"""

import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch

import solution

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

REPO = pathlib.Path(__file__).resolve().parents[1]
_CAPABILITY = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0

                                                                                     
STARTUP_LABELS = tuple(label for _, _, label in solution.STARTUP)
GC_FREEZE_LABELS = ("gc freeze (start)", "gc freeze (end)")

                                                                                             
                                                                                          
                                                                                             
                                               
CASES = {
    "DISABLE_FUSED_HEAD": ({"SOLUTION_DISABLE_FUSED_HEAD": "1"},
                           ["count:head_vocab", "flag:logits_none", "count:unpad", "count:varlen",
                            "installed:compute_loss", "installed:defer", "installed:bookkeeping"],
                           [], []),
    "DISABLE_UNPAD": ({"SOLUTION_DISABLE_UNPAD": "1"}, ["count:unpad", "count:varlen"], [],
                      ["count:head_vocab"]),
    "DISABLE_ROWS_ATTN": ({"SOLUTION_DISABLE_ROWS_ATTN": "1"}, ["count:varlen"], [],
                          ["count:unpad"]),
    "DISABLE_VARLEN": ({"SOLUTION_DISABLE_VARLEN": "1"}, ["count:varlen"], ["count:rows_attn"], []),
    "DISABLE_ATTN_NOCAT": ({"SOLUTION_DISABLE_ATTN_NOCAT": "1"}, ["const:nocat"], [],
                           ["count:varlen"]),
    "DISABLE_HEAD_VCHUNK": ({"SOLUTION_DISABLE_HEAD_VCHUNK": "1"},
                            ["count:head_vocab", "const:vocab_path"], ["count:head_tokens"], []),
    "DISABLE_HEAD_COMPACT": ({"SOLUTION_DISABLE_HEAD_COMPACT": "1"},
                             ["count:head_compact", "const:head_compact"], [], ["count:head_vocab"]),
    "DISABLE_HEAD_ROWS": ({"SOLUTION_DISABLE_HEAD_ROWS": "1"}, ["const:head_rows"], [],
                          ["count:head_vocab"]),
    "DISABLE_RMSNORM": ({"SOLUTION_DISABLE_RMSNORM": "1"}, ["count:rmsnorm", "count:addnorm"], [],
                        []),
    "DISABLE_ADDNORM": ({"SOLUTION_DISABLE_ADDNORM": "1"}, ["count:addnorm"], [],
                        ["count:rmsnorm"]),
    "DISABLE_ROPE": ({"SOLUTION_DISABLE_ROPE": "1"}, ["count:rope"], [], []),
    "DISABLE_ROPE_TOKEN_MAJOR": ({"SOLUTION_DISABLE_ROPE_TOKEN_MAJOR": "1"},
                                 ["const:rope_token_major"], [], ["count:rope"]),
    "DISABLE_ROPE_ROWS": ({"SOLUTION_DISABLE_ROPE_ROWS": "1"}, ["const:rope_rows"], [],
                          ["count:rope"]),
    "DISABLE_GROUP": ({"SOLUTION_DISABLE_GROUP": "1"},
                      ["count:group", "count:split_dequant", "count:lora", "count:fused_qkv"],
                      ["installed:nf4_dequant"], []),
                                                                                         
    "DISABLE_FUSED_PROJ": ({"SOLUTION_DISABLE_FUSED_PROJ": "1"}, ["count:fused_qkv"], [],
                           ["count:lora", "count:group"]),
    "DISABLE_FUSED_QKV": ({"SOLUTION_DISABLE_FUSED_QKV": "1"}, ["count:fused_qkv"], [],
                          ["count:lora"]),
    "FUSED_MLP": ({"SOLUTION_FUSED_MLP": "1"}, [], ["count:fused_mlp"], ["count:fused_qkv"]),
    "DISABLE_SPLIT_DEQUANT": ({"SOLUTION_DISABLE_SPLIT_DEQUANT": "1"},
                              ["count:split_dequant", "const:split"], [], ["count:group"]),
    "DISABLE_DEQUANT": ({"SOLUTION_DISABLE_DEQUANT": "1"}, ["const:fallback_dequant"], [],
                        ["count:group"]),
    "DISABLE_GROUP+DEQUANT": ({"SOLUTION_DISABLE_GROUP": "1", "SOLUTION_DISABLE_DEQUANT": "1"},
                              ["count:group"], [], []),
    "DISABLE_LORA": ({"SOLUTION_DISABLE_LORA": "1"}, ["count:lora"], [], ["count:group"]),
    "DISABLE_LORA_CAST": ({"SOLUTION_DISABLE_LORA_CAST": "1"},
                          ["count:lora_cast", "const:lora_cast"], [], ["count:lora"]),
    "DISABLE_FP16_EMBED": ({"SOLUTION_DISABLE_FP16_EMBED": "1"}, ["flag:fp16_embed"], [],
                           ["count:head_vocab"]),
    "DISABLE_SWIGLU_INPLACE": ({"SOLUTION_DISABLE_SWIGLU_INPLACE": "1"},
                               ["count:swiglu_inplace"], [], []),
    "DISABLE_DEFER": ({"SOLUTION_DISABLE_DEFER": "1"}, ["installed:defer"], [],
                      ["installed:bookkeeping"]),
    "DISABLE_BOOKKEEPING": ({"SOLUTION_DISABLE_BOOKKEEPING": "1"}, ["installed:bookkeeping"], [],
                            ["installed:defer"]),
    "DISABLE_STARTUP": ({"SOLUTION_DISABLE_STARTUP": "1"},
                        [f"log:{label}" for label in STARTUP_LABELS + GC_FREEZE_LABELS], [], []),
    "DISABLE_GC_FREEZE": ({"SOLUTION_DISABLE_GC_FREEZE": "1"},
                          [f"log:{label}" for label in GC_FREEZE_LABELS], [],
                          [f"log:{label}" for label in STARTUP_LABELS]),
    **{switch[len("SOLUTION_"):]: ({switch: "1"}, [f"log:{label}"], [],
                                   [f"log:{other}" for other in STARTUP_LABELS + GC_FREEZE_LABELS
                                    if other != label])
       for switch, _, label in solution.STARTUP},
    "WARMUP_THREAD=0": ({"SOLUTION_WARMUP_THREAD": "0"}, ["flag:warmup_thread"], [], []),
    "WARMUP_THREAD=late": ({"SOLUTION_WARMUP_THREAD": "late"}, [], [], ["flag:warmup_thread"]),
    "FAST_TRITON_BUILD=0": ({"SOLUTION_FAST_TRITON_BUILD": "0"}, ["flag:fast_triton_build"], [],
                            []),
    "MLP_RECOMPUTE": ({"SOLUTION_MLP_RECOMPUTE": "1"}, ["count:group", "count:lora"],
                      ["count:mlp_recompute"], []),
    "MLP_SAVE_UP": ({"SOLUTION_MLP_SAVE_UP": "1"}, ["count:swiglu_inplace"],
                    ["count:mlp_save_up"], ["count:group", "count:lora"]),
                                                                                        
    "QLINEAR": ({"SOLUTION_QLINEAR": "1"}, [], ["log:fused 4-bit linear patch on"],
                ["count:group"]),
    "QLINEAR+DISABLE_GROUP": ({"SOLUTION_QLINEAR": "1", "SOLUTION_DISABLE_GROUP": "1"},
                              ["count:group"], ["count:qlinear"], []),
                                                                                      
    "VARLEN unset": ({"SOLUTION_VARLEN": None}, ["count:varlen"], ["count:rows_attn"], [])
    if _CAPABILITY >= 8 else ({"SOLUTION_VARLEN": None}, [], [], ["count:varlen"]),
    "HEAD_VCHUNK=512": ({"SOLUTION_HEAD_VCHUNK": "512"}, [], ["const:head_vchunk_512"],
                        ["count:head_vocab"]),
}


def _tiny_checkpoint(path):
    """A small tied Qwen2 in bf16, as the real checkpoint is stored, with sizes every kernel of
    ours takes: head size 128 (RoPE, attention), each layer packed by the per-layer dequant and
    splittable into per-projection launches, vocabulary and hidden size multiples of 8."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=1000, hidden_size=512, intermediate_size=1024,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=True,
                         attn_implementation="sdpa")
    torch.manual_seed(51)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.08)
    model.to(torch.bfloat16).save_pretrained(path)


def probe(checkpoint):
    """In a fresh process: apply(), one training step, the report as a `PROBE {json}` line."""
    import transformers                                       

    import solution

    solution.apply()
    from solution import (fused_head, fused_proj, lora, mlp_recompute, mlp_save_up, patches,
                          qgroup, qlinear, rmsnorm, rope, swiglu_inplace, triton_build, unpad,
                          warmup)

    report = {"warmup_thread": warmup._thread is not None}
    warmup.join()
    import triton

    report["fast_triton_build"] = triton.knobs.build.impl is triton_build._build

    import peft

    quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_compute_dtype=torch.float16)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint, quantization_config=quant, device_map={"": 0}, dtype=torch.float16)
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    torch.manual_seed(52)
    model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                       target_modules=["q_proj", "v_proj"],
                                                       task_type="CAUSAL_LM"))
    weight = model.get_base_model().get_input_embeddings().weight
    report["fp16_embed"] = weight.dtype == torch.float16

    counts = {}

    def count(owner, name, key):
        fn = getattr(owner, name)

        def wrapped(*args, **kwargs):
            counts[key] = counts.get(key, 0) + 1
            return fn(*args, **kwargs)

        setattr(owner, name, wrapped)

    for owner, name, key in (
            (rmsnorm.RMSNorm, "apply", "rmsnorm"), (rmsnorm.AddRMSNorm, "apply", "addnorm"),
            (rope.RoPE, "apply", "rope"), (patches, "unpad_batch", "unpad"),
            (unpad.VarlenRowsAttention, "apply", "varlen"),
            (unpad.RowsAttention, "apply", "rows_attn"), (qgroup.GroupLinear, "apply", "group"),
            (qgroup.LayerGroup, "dequantize_one", "split_dequant"),
            (lora.LoraProjectionTorch, "apply", "lora"), (lora.HalfCopies, "refresh", "lora_cast"),
            (fused_head.FusedHeadLossVocab, "apply", "head_vocab"),
            (fused_head.FusedHeadLoss, "apply", "head_tokens"),
            (fused_head, "labelled_first", "head_compact"),
            (swiglu_inplace.SwiGLUInPlaceGrad, "apply", "swiglu_inplace"),
            (mlp_recompute.MLPRecompute, "apply", "mlp_recompute"),
            (mlp_save_up.SwiGLUSaveUp, "apply", "mlp_save_up"),
            (fused_proj.FusedProjections, "apply", "fused_qkv"),
            (fused_proj.FusedMLP, "apply", "fused_mlp"),
            (qlinear.QLinear4bit, "apply", "qlinear")):
        count(owner, name, key)

    batch, seq = 3, 40
    gen = torch.Generator(device="cuda").manual_seed(53)
    input_ids = torch.randint(0, 1000, (batch, seq), device="cuda", generator=gen)
    lengths = torch.tensor([40, 17, 29], device="cuda")
    attention_mask = (torch.arange(seq, device="cuda")[None] < lengths[:, None]).long()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :7] = -100                                                       
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    torch.cuda.synchronize()
    report["logits_none"] = out.logits is None
    report["counts"] = counts
    report["const"] = {
        "vocab_path": fused_head.VOCAB_PATH, "head_compact": fused_head.COMPACT,
        "head_rows": fused_head.ROW_UNIT == 8, "head_vchunk_512": fused_head.VOCAB_CHUNK == 512,
        "nocat": unpad._NOCAT,
        "rope_token_major": rope.TOKEN_MAJOR_GRAD, "rope_rows": rope.ROW_STRIDES,
        "split": qgroup.SPLIT_LAYERS["bwd"] > 0, "fallback_dequant": qgroup.FALLBACK_DEQUANT,
        "lora_cast": lora.CAST_CACHE}

    import bitsandbytes.backends.cuda.ops as bnb_ops
    import trl

    report["installed"] = {
        "compute_loss": trl.SFTTrainer.compute_loss.__module__ == "solution.patches",
        "defer": getattr(trl.SFTTrainer.log, "_solution_flush", False),
        "bookkeeping": getattr(transformers.Trainer.floating_point_ops, "_solution_cached", False),
        "nf4_dequant": getattr(bnb_ops._dequantize_4bit_impl, "_solution_patched", False)}
    print("PROBE " + json.dumps(report), flush=True)


def _run_probe(env_changes, checkpoint):
    env = dict(os.environ, SOLUTION_VARLEN="1")
    for name in list(env):
        if name.startswith("SOLUTION_") and name != "SOLUTION_VARLEN":
            del env[name]                                                               
    for name, value in env_changes.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    code = (f"import sys; sys.path.insert(0, {str(REPO)!r}); "
            f"from solution.test_switches import probe; probe({str(checkpoint)!r})")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=900)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("PROBE ")]
    if proc.returncode != 0 or not lines:
        return {"error": f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"}
    report = json.loads(lines[-1][len("PROBE "):])
    report["log"] = proc.stdout
    return report


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """The default probe and one probe per case, a few processes at a time."""
    checkpoint = tmp_path_factory.mktemp("tiny_qwen2")
    _tiny_checkpoint(checkpoint)
    jobs = {"default": {}, **{name: case[0] for name, case in CASES.items()}}
    workers = max(1, min(4, (os.cpu_count() or 2) // 2))
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {name: pool.submit(_run_probe, env, checkpoint) for name, env in jobs.items()}
        return {name: future.result() for name, future in futures.items()}


def _live(report, key):
    kind, name = key.split(":", 1)
    if kind == "count":
        return report["counts"].get(name, 0) > 0
    if kind == "log":
        return f"[solution] {name}" in report["log"]
    if kind == "flag":
        return bool(report[name])
    return bool(report[kind if kind == "const" else "installed"][name])


@pytest.mark.parametrize("case", list(CASES))
def test_switch(reports, case):
    _env, gone, comes, stays = CASES[case]
    default, report = reports["default"], reports[case]
    assert "error" not in default, default.get("error")
    assert "error" not in report, report.get("error")
    for key in gone:
        assert _live(default, key), f"{key} is not live without the switch"
        assert not _live(report, key), f"{key} is still live with {case}"
    for key in comes:
        assert not _live(default, key), f"{key} is live without the switch"
        assert _live(report, key), f"{key} is not live with {case}"
    for key in stays:
        assert _live(report, key), f"{key} went away with {case}"


def test_every_switch_read_in_solution_has_a_case_and_is_documented():
    """Every SOLUTION_* name read in solution/ has a case above and is in solution/__init__.py's
    docstring (in full, as `_NAME` in a SOLUTION_DISABLE_ list, or in the STARTUP table)."""
    import re

    read = set()
    for path in pathlib.Path(__file__).parent.glob("*.py"):
        if not path.name.startswith("test_"):
            read |= set(re.findall(r"\"(SOLUTION_[A-Z0-9_]+)\"", path.read_text(encoding="utf-8")))
    covered = {name for env, *_ in CASES.values() for name in env}
    assert read - covered == set(), read - covered
    doc = solution.__doc__
    documented = set(re.findall(r"SOLUTION_[A-Z0-9_]+", doc))
    documented |= {"SOLUTION_DISABLE" + name for name in re.findall(r"(?<![A-Z0-9_])_[A-Z][A-Z0-9_]*", doc)}
    documented |= {switch for switch, _, _ in solution.STARTUP}
    assert read - documented == set(), read - documented
