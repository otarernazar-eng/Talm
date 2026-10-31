"""The `[solution] engaged: ...` line (`solution/engaged.py`): its format, when it is printed, and
that the counters behind it follow the fast paths of a training step.

Run on a GPU: `python -m pytest solution -q`. (Named to run after test_fp16_embed.py, whose
memory_allocated delta depends on which blocks earlier GPU tests left in PyTorch's cache.)
"""

import collections
import re
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LINE = re.compile(r"\[solution\] engaged: at=(step1|exit) steps=\d+"
                  r"( [a-z0-9_]+=[\w.:/-]+)+"
                  r" fallbacks=(none|[a-z0-9_]+:\d+(,[a-z0-9_]+:\d+)*)")
# Always on the line, in this order (the optional ones appear only when installed).
ALWAYS = ("head", "vchunk", "compact", "rows8", "unpad", "varlen", "nocat", "lora", "lora_cast",
          "fused_proj", "rmsnorm", "addnorm", "rope", "group", "split", "swiglu_inplace",
          "fp16_embed", "defer", "bookkeeping", "startup", "warmup", "build")


@pytest.fixture(autouse=True)
def _release_cached_blocks():
    """Leave no cached free blocks behind: later tests measure memory_allocated deltas, which a
    reused (unsplit) cached block would inflate."""
    yield
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def _tokens(text):
    assert LINE.fullmatch(text), text
    return dict(token.split("=", 1) for token in text.split()[2:])


def test_engaged_line_format(monkeypatch):
    from solution import engaged

    ran, fell = collections.defaultdict(int), collections.defaultdict(int)
    monkeypatch.setattr(engaged, "RAN", ran)
    monkeypatch.setattr(engaged, "FELL_BACK", fell)
    monkeypatch.setattr(engaged, "ON", {"head", "unpad", "rmsnorm", "lora", "group"})
    monkeypatch.setattr(engaged, "TEXT", {"startup": "7/7", "build": "-O0"})
    monkeypatch.setitem(engaged._seen, "state", SimpleNamespace(global_step=3))
    ran.update(head=3, unpad=3, rmsnorm=6, lora=12, group=9)
    text = engaged.line("step1")
    tokens = _tokens(text)
    names = [name for name in tokens if name not in ("at", "steps", "fallbacks")]
    assert [name for name in names if name in ALWAYS] == list(ALWAYS)
    assert tokens["at"] == "step1" and tokens["steps"] == "3"
    assert (tokens["head"], tokens["unpad"], tokens["rmsnorm"], tokens["lora"]) == \
        ("3", "3", "6", "12")
    assert tokens["rope"] == tokens["addnorm"] == tokens["defer"] == "off"  # not installed
    assert tokens["startup"] == "7/7" and tokens["build"] == "-O0"
    assert tokens["fallbacks"] == "none"
    fell.update(rope=2, unpad=1)
    tokens = _tokens(engaged.line("exit"))
    # in the line's order, whatever the order of the counts
    assert tokens["at"] == "exit" and tokens["fallbacks"] == "unpad:1,rope:2"
    monkeypatch.setattr(engaged, "ON", set())
    monkeypatch.setattr(engaged, "TEXT", {})
    tokens = _tokens(engaged.line("exit"))
    assert all(tokens[name] == "off" for name in ALWAYS if name != "warmup")


def test_engaged_line_printed_once_after_the_first_step_before_the_callbacks(
        monkeypatch, capsys):
    """Through transformers' CallbackHandler: nothing at the first step's begin, the line when
    the second step begins (after the first step), before any callback of that step runs (the
    harness's step timer is one), then never again; at exit the totals."""
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.trainer_callback import CallbackHandler

    from solution import engaged

    assert engaged.install(register_exit=False)
    assert engaged.install(register_exit=False)  # idempotent: wrapped once
    assert not getattr(CallbackHandler.on_step_begin._original, "_solution_engaged", False)
    monkeypatch.setitem(engaged._seen, "printed", False)
    monkeypatch.setitem(engaged._seen, "state", None)
    seen = []

    class Spy(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            seen.append(capsys.readouterr().out)  # printed before this callback ran

    handler = CallbackHandler([Spy()], None, None, None, None)
    state, control = TrainerState(), TrainerControl()
    capsys.readouterr()
    for step in range(3):
        state.global_step = step
        handler.on_step_begin(None, state, control)
    lines = [[line for line in out.splitlines() if "engaged:" in line] for out in seen]
    assert lines[0] == [] and lines[2] == [] and len(lines[1]) == 1
    tokens = _tokens(lines[1][0])
    assert tokens["at"] == "step1" and tokens["steps"] == "1"
    state.global_step = 64
    engaged.report_at_exit()
    tokens = _tokens(capsys.readouterr().out.strip())
    assert tokens["at"] == "exit" and tokens["steps"] == "64"


def _delta(before, after):
    return {name: after[name] - before.get(name, 0) for name in after
            if after[name] != before.get(name, 0)}


def test_engaged_counts_the_fast_paths_of_a_training_step(monkeypatch):
    """apply(), then one forward and backward of a small Qwen2 causal LM through the patched
    model (padding removal, fused head, our norms, RoPE, the single-launch attention, SwiGLU):
    each counter moves by exactly the launches / calls of that step and nothing falls back."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.trainer_callback import CallbackHandler

    import solution
    from solution import engaged

    monkeypatch.setenv("SOLUTION_VARLEN", "1")  # the single-launch attention on any GPU
    solution.apply()
    assert getattr(CallbackHandler.on_step_begin, "_solution_engaged", False)
    for name in ("head", "unpad", "rmsnorm", "addnorm", "rope", "group", "lora",
                 "swiglu_inplace", "fp16_embed", "defer", "bookkeeping"):
        assert name in engaged.ON, name  # every default component installed by apply()
    assert re.fullmatch(r"\d+/\d+", engaged.TEXT["startup"])
    config = Qwen2Config(vocab_size=600, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=False,
                         attn_implementation="sdpa")
    torch.manual_seed(5)
    model = Qwen2ForCausalLM(config).cuda().train()
    model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)  # a gradient through every layer
    batch, seq = 3, 24
    input_ids = torch.randint(0, 600, (batch, seq), device="cuda")
    lengths = [24, 9, 17]
    attention_mask = (torch.arange(seq, device="cuda")[None]
                      < torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :6] = -100
    ran, fell = dict(engaged.RAN), dict(engaged.FELL_BACK)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    ran, fell = _delta(ran, engaged.RAN), _delta(fell, engaged.FELL_BACK)
    layers, kv_heads = 2, 2
    expected = {"head": 1, "vchunk": 1, "compact": 1, "unpad": 1,
                "rows8": 2,  # one vocabulary slice (600 columns) and the gradient's finalize
                "varlen": layers * kv_heads * 2, "nocat": layers,  # forward + backward
                "rmsnorm": 2,  # layer 0's input norm alone, forward + backward
                "addnorm": layers * 2 * 2,  # both residual adds of a layer, forward + backward
                "rope": layers * 2, "swiglu_inplace": layers}
    assert {name: ran.get(name, 0) for name in expected} == expected
    assert fell == {}, fell
    tokens = _tokens(engaged.line("step1"))
    for name in expected:
        assert tokens[name] == str(engaged.RAN[name])


def test_mlp_residual_add_folds_without_the_fused_head():
    """A plain Qwen2Model forward (the path with SOLUTION_DISABLE_FUSED_HEAD=1, whose causal-LM
    forward never links the norms) links them itself: every residual add of the decoder goes
    into a norm kernel and none takes the plain add."""
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution import engaged, rmsnorm

    rmsnorm.patch_qwen2_rmsnorm()
    rmsnorm.patch_qwen2_add_norm()
    config = Qwen2Config(hidden_size=256, intermediate_size=512, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=3, vocab_size=128,
                         max_position_embeddings=128, attn_implementation="sdpa")
    torch.manual_seed(36)
    model = modeling_qwen2.Qwen2Model(config).cuda().train()
    model.requires_grad_(False)
    ran, fell = dict(engaged.RAN), dict(engaged.FELL_BACK)
    with torch.autocast("cuda", dtype=torch.float16):
        model(inputs_embeds=torch.randn(1, 40, 256, device="cuda"))
    ran, fell = _delta(ran, engaged.RAN), _delta(fell, engaged.FELL_BACK)
    assert model.layers[-1]._solution_next_norm is model.norm
    assert ran.get("addnorm") == 3 * 2 and ran.get("rmsnorm") == 1  # forward only
    assert "addnorm" not in fell and "rmsnorm" not in fell, fell


def test_flat_attention_takes_fp32_inputs_like_autocast(monkeypatch):
    """q/k/v in fp32 under fp16 autocast (transformers' RoPE: SOLUTION_DISABLE_ROPE=1) inside a
    flat forward: cast as autocast casts SDPA's inputs, then the single-launch attention runs
    (before: the plain per-row fallback with repeated K/V) with the fallback's result."""
    import torch.nn.functional as F
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from solution import engaged, unpad

    monkeypatch.setenv("SOLUTION_VARLEN", "1")  # the single-launch attention on any GPU
    unpad.patch_sdpa_rows()
    interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
    torch.manual_seed(37)
    lengths = [37, 5, 128, 64]
    bounds, start = [], 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length
    q = torch.randn(1, 12, start, 128, device="cuda", requires_grad=True)
    k = torch.randn(1, 2, start, 128, device="cuda", requires_grad=True)
    v = torch.randn(1, 2, start, 128, device="cuda", requires_grad=True)

    class Module:
        num_key_value_groups = 6
        training = True

    ran, fell = dict(engaged.RAN), dict(engaged.FELL_BACK)
    with unpad.rows_context(bounds, device="cuda"), torch.autocast("cuda", dtype=torch.float16):
        out, _ = interface(Module(), q, k, v, None, dropout=0.0, scaling=128 ** -0.5,
                           sliding_window=None)
    out.float().sum().backward()
    ran, fell = _delta(ran, engaged.RAN), _delta(fell, engaged.FELL_BACK)
    assert ran.get("varlen") == 2 * 2 and "varlen" not in fell  # forward + backward, 2 KV heads
    assert out.dtype == torch.float16 and q.grad.dtype == torch.float32
    rows = [F.scaled_dot_product_attention(
        q[:, :, s:e].half(), k[:, :, s:e].half().repeat_interleave(6, 1),
        v[:, :, s:e].half().repeat_interleave(6, 1), is_causal=True, scale=128 ** -0.5)
        for s, e in bounds]
    ref = torch.cat(rows, 2).transpose(1, 2)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)


def test_engaged_counts_the_fused_qkv_node(tmp_path):
    """fused_proj.py's q|k|v node (on by default) is on the line: `fused_proj=` counts the layer
    calls it took, and a packed layer whose attention took the previous path anyway counts in
    `fallbacks=` (here a forward hook on layer 0's k_proj: the node stands aside); the last
    layer's split forward pass (qgroup.SPLIT_LAYERS: the previous path by design) does not."""
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    import solution
    from solution import engaged, fused_proj
    from solution.test_steady_state import VOCAB, _load, _save_tiny_model

    solution.apply()
    _save_tiny_model(tmp_path / "tiny")  # 3 packed layers; the last one's forward is split
    model = _load(tmp_path / "tiny").train()
    gen = torch.Generator().manual_seed(9)
    input_ids = torch.randint(3, VOCAB, (3, 30), generator=gen).cuda()
    attention_mask = (torch.arange(30)[None] < torch.tensor([30, 12, 21])[:, None]).long().cuda()
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    k_proj = model.get_base_model().model.layers[0].self_attn.k_proj
    deltas = []
    for hooked in (False, True):
        handle = k_proj.register_forward_hook(lambda *args: None) if hooked else None
        ran, fell = dict(engaged.RAN), dict(engaged.FELL_BACK)
        calls = dict(fused_proj.COUNTS)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        if handle is not None:
            handle.remove()
        deltas.append((_delta(ran, engaged.RAN).get("fused_proj", 0),
                       _delta(fell, engaged.FELL_BACK).get("fused_proj", 0),
                       fused_proj.COUNTS["attn"] - calls["attn"]))
    assert deltas == [(2, 0, 2), (1, 1, 1)], deltas
    tokens = _tokens(engaged.line("exit"))
    assert tokens["fused_proj"] == str(engaged.RAN["fused_proj"])
    assert "fused_proj:" in tokens["fallbacks"]


def test_engaged_counts_the_lora_node_and_the_group_dequant():
    """q_proj and v_proj of a packed 4-bit layer through PEFT's patched LoRA forward: two node
    calls, one whole-layer dequant launch in the forward and one in the backward, the adapters'
    fp16 copies made, and no PEFT fallback."""
    from solution import engaged, lora
    from solution.qgroup import setup_layer
    from solution.test_lora import _lora_layer

    assert lora.patch_peft_lora() is True
    torch.manual_seed(24)
    layer = _lora_layer(hidden=256, kv=64)
    setup_layer(layer)
    attn = layer.self_attn
    assert hasattr(attn.q_proj.base_layer, "_solution_group"), "layer was not packed"
    ran, fell = dict(engaged.RAN), dict(engaged.FELL_BACK)
    x = torch.randn(1, 50, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        h = x * 1.0
        outs = [attn.q_proj(h), attn.v_proj(h)]
    sum(o.float().sum() for o in outs).backward()
    ran, fell = _delta(ran, engaged.RAN), _delta(fell, engaged.FELL_BACK)
    assert ran.get("lora") == 2 and ran.get("group") == 2 and ran.get("lora_cast", 0) >= 1
    assert "lora" not in fell and "lora_cast" not in fell and "group" not in fell
