"""Graceful degradation (solution/state.py): a failure injected into a fast path (a kernel
launcher made to raise once) must leave the step with the original path's values, print one
`[solution] fallback:` line, keep the component off, and let later calls run the original path;
KeyboardInterrupt and CUDA out-of-memory must propagate.

Run on a GPU: `python -m pytest solution -q`.
"""

import gc
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from solution.test_robustness import _batch, _tiny_lm, tiny_qlora  # noqa: F401 - a fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

PREFIX = "[solution] fallback: "


class Faults:
    """Injects failures and reads the fallback lines; `solution.state.DISABLED` is restored
    after the test."""

    def __init__(self, monkeypatch, capsys):
        from solution import state

        self.monkeypatch, self.capsys, self.state = monkeypatch, capsys, state
        self.out = []

    def inject(self, owner, name, exc=None, when=None, times=1):
        """`owner.name` raises `exc` (RuntimeError("injected failure")) at its first `times`
        calls for which `when(*args, **kwargs)` holds; calls are counted in the result."""
        real = getattr(owner, name)
        calls = {"n": 0, "raised": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["raised"] < times and (when is None or when(*args, **kwargs)):
                calls["raised"] += 1
                raise exc if exc is not None else RuntimeError("injected failure")
            return real(*args, **kwargs)

        self.monkeypatch.setattr(owner, name, flaky)
        return calls

    def lines(self):
        self.out += [line for line in self.capsys.readouterr().out.splitlines()
                     if line.startswith(PREFIX)]
        return list(self.out)

    def assert_off(self, *components):
        """Exactly these components are off, each with one line naming the injected error."""
        lines = self.lines()
        assert sorted(self.state.DISABLED) == sorted(components), self.state.DISABLED
        assert len(lines) == len(components), lines
        for component in components:
            assert self.state.DISABLED[component] == "RuntimeError: injected failure"
            assert f"{PREFIX}{component}: RuntimeError: injected failure" in lines


@pytest.fixture
def faults(monkeypatch, capsys):
    from solution import state

    saved = dict(state.DISABLED)
    state.DISABLED.clear()
    yield Faults(monkeypatch, capsys)
    state.DISABLED.clear()
    state.DISABLED.update(saved)
    # the injected exceptions' tracebacks can form reference cycles holding CUDA tensors: free
    # them here, not inside a later test's memory measurement
    gc.collect()


def _nth(n):
    """A `when` that holds at the n-th call (1-based)."""
    seen = [0]

    def when(*args, **kwargs):
        seen[0] += 1
        return seen[0] == n

    return when


def _transformers_rmsnorm(norm, x):
    """transformers' Qwen2RMSNorm.forward, op for op."""
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * h.to(x.dtype)


def _norm(hidden=256):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution import rmsnorm

    rmsnorm.patch_qwen2_rmsnorm()
    norm = Qwen2RMSNorm(hidden, eps=1e-6).cuda().requires_grad_(False)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
    return norm


# --- forward failures: the call runs again through the original code ------------------------

def test_rmsnorm_forward_failure_runs_transformers_norm(faults):
    from solution import rmsnorm

    norm = _norm()
    x = torch.randn(3, 7, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        ref = _transformers_rmsnorm(norm, x)
        calls = faults.inject(rmsnorm, "launch")
        first, later = norm(x), norm(x)
    assert torch.equal(first, ref) and torch.equal(later, ref)
    assert calls["n"] == 1  # the kernel is not tried again
    faults.assert_off("rmsnorm")


def test_rope_forward_failure_runs_transformers_rotation(faults):
    from transformers.models.qwen2 import modeling_qwen2

    from solution import rope

    rope.patch_qwen2_rope()
    rotate = modeling_qwen2.apply_rotary_pos_emb
    torch.manual_seed(1)
    q = torch.randn(1, 50, 12, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(1, 50, 2, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    cos, sin = torch.randn(2, 1, 50, 128, device="cuda").unbind(0)
    ref = rotate._original(q, k, cos, sin)
    calls = faults.inject(rope, "_launch")
    for _ in range(2):
        out = rotate(q, k, cos, sin)
        assert all(torch.equal(a, b) for a, b in zip(out, ref))
    assert calls["n"] == 1
    faults.assert_off("rope")


class _FlakyKernel:
    """A @triton.jit kernel whose launch raises the first time."""

    def __init__(self, kernel):
        self.kernel, self.launches = kernel, 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches += 1
            if self.launches == 1:
                raise RuntimeError("injected failure")
            return self.kernel[grid](*args, **kwargs)

        return launch


def test_swiglu_forward_failure_runs_transformers_activation(faults, monkeypatch):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution import swiglu, swiglu_inplace

    swiglu.patch_qwen2_mlp()
    swiglu_inplace.patch_qwen2_mlp_inplace()
    torch.manual_seed(2)
    mlp = Qwen2MLP(Qwen2Config(hidden_size=64, intermediate_size=128)).cuda().half()
    x = torch.randn(5, 64, device="cuda", dtype=torch.float16)
    ref = mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))  # transformers' forward
    kernel = _FlakyKernel(swiglu_inplace.example_swiglu_fwd_kernel)
    monkeypatch.setattr(swiglu_inplace, "example_swiglu_fwd_kernel", kernel)
    assert torch.equal(mlp(x), ref) and torch.equal(mlp(x), ref)
    assert kernel.launches == 1
    faults.assert_off("swiglu")


def test_attention_failures_fall_back_to_pytorch_sdpa_per_row(faults, monkeypatch):
    """Variable-length attention fails in its forward: the per-row launches take the call;
    those fail in their backward: the gradient comes from PyTorch's SDPA per row."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from solution import unpad
    from solution.test_fusions import _row_bounds

    monkeypatch.setenv("SOLUTION_VARLEN", "1")
    monkeypatch.setattr(unpad, "_VARLEN", True)
    unpad.patch_sdpa_rows()
    interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
    module = SimpleNamespace(num_key_value_groups=6)
    torch.manual_seed(3)
    bounds, n = _row_bounds([37, 5, 64])
    q = torch.randn(1, 12, n, 128, device="cuda", dtype=torch.float16)
    k, v = (torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16) for _ in range(2))
    grad = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16)

    def run():
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
        with unpad.rows_context(bounds, device="cuda"):
            out, _ = interface(module, q1, k1, v1, None, scaling=128 ** -0.5)
        out.backward(grad)
        return out.detach(), [t.grad for t in (q1, k1, v1)]

    ref, ref_grads = run()  # the variable-length path, no failure
    varlen = faults.inject(unpad.VarlenRowsAttention, "apply")
    rows_bwd = faults.inject(unpad.RowsAttention, "_backward")
    out, grads = run()
    assert varlen["n"] == 1 and rows_bwd["n"] == 1
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    for ours, theirs in zip(grads, ref_grads):
        torch.testing.assert_close(ours, theirs, rtol=2e-2, atol=2e-2 * float(theirs.abs().max()))
    faults.assert_off("varlen_attention", "rows_attention")
    run()  # later calls: PyTorch's SDPA per row, no further failure or line
    assert varlen["n"] == 1 and rows_bwd["n"] == 1 and len(faults.lines()) == 2


def _packed_layer(monkeypatch):
    import bitsandbytes.functional as bnbf

    from solution import qgroup
    from solution.qgroup import PROJECTIONS, setup_layer
    from solution.test_kernels import _fake_layer

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 0, "bwd": 0})
    torch.manual_seed(4)
    layer = _fake_layer()
    modules = []
    for path in PROJECTIONS:
        obj = layer
        for part in path.split("."):
            obj = getattr(obj, part)
        modules.append(obj)
    refs = [bnbf.dequantize_4bit(m.weight.data, m.weight.quant_state).half() for m in modules]
    setup_layer(layer)
    return modules, refs, modules[0]._solution_group[0]


@pytest.mark.parametrize("where", ["forward", "backward"])
def test_group_dequant_failure_takes_bitsandbytes_weights(faults, monkeypatch, where):
    """The weight comes from bitsandbytes' dequantization, which our kernel matches bit for
    bit: the projection's output and input gradient are unchanged, in either pass."""
    from solution import qgroup
    from solution.qgroup import GroupLinear

    modules, refs, group = _packed_layer(monkeypatch)
    x = torch.randn(3, 40, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    g = torch.randn(3, 40, 256, device="cuda", dtype=torch.float16)
    bias = modules[0].bias.half()
    calls = faults.inject(qgroup, "launch") if where == "forward" else None
    out = GroupLinear.apply(x, group, 0)
    if where == "backward":
        calls = faults.inject(qgroup, "launch")
    (out * g).sum().backward()
    assert torch.equal(out, F.linear(x, refs[0], bias))
    assert torch.equal(x.grad, torch.matmul(g, refs[0]))
    assert calls["n"] == 1
    faults.assert_off("qgroup")
    assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
    # later projections: bitsandbytes' Linear4bit.forward through the patched entry point
    with torch.no_grad():
        later = modules[1](x)
    torch.testing.assert_close(later, F.linear(x, refs[1], modules[1].bias.half()),
                               rtol=2e-3, atol=2e-3)
    assert calls["n"] == 1
    group.expected_bwd = 3
    qgroup.reset_passes()  # what a forward run again from the start does first
    assert group.expected_bwd == 0 and group.split_left == 0


def test_lora_node_failure_gives_pefts_values_and_rng_stream(faults, monkeypatch):
    """The node fails after its base weight and its dropout draw: PEFT's forward takes the
    call from the same generator state (the same mask), so output and RNG stream equal PEFT's,
    and the layer group's pass cache counts the base weight once."""
    import bitsandbytes as bnb

    from solution import lora
    from solution.qgroup import setup_layer
    from solution.test_lora import _group_forward, _lora_layer

    assert lora.patch_peft_lora() is True
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", _group_forward)
    torch.manual_seed(5)
    layer = _lora_layer(hidden=1536, kv=256)
    setup_layer(layer)
    q = layer.self_attn.q_proj
    q.train()
    group = q.base_layer._solution_group[0]
    x = torch.randn(1, 64, 1536, device="cuda", dtype=torch.float16)

    def run(forward):
        torch.cuda.manual_seed(11)
        with torch.autocast("cuda", dtype=torch.float16):
            out = forward(q, x)
        return out, torch.cuda.get_rng_state()

    ref, ref_state = run(type(q).forward._original)  # PEFT: one use of the layer's weights
    calls = faults.inject(lora, "_power_of_two")  # after the base GEMM and the dropout
    out, state = run(lambda module, inp: module(inp))
    assert calls["n"] == 1
    assert torch.equal(out, ref) and torch.equal(state, ref_state)
    assert group.cache["fwd"][1] == 7 - 2  # two uses (PEFT's twice), not three
    faults.assert_off("lora")


def test_fused_head_failure_runs_transformers_forward(faults):
    from transformers import Qwen2ForCausalLM

    from solution import fused_head, patches

    model = _tiny_lm()
    input_ids, mask, labels = _batch([24, 9, 17, 20])
    batch = dict(input_ids=input_ids, attention_mask=mask, labels=labels,
                 num_items_in_batch=(labels[:, 1:] != -100).sum())
    with torch.autocast("cuda", dtype=torch.float16):
        assert model(**batch).logits is None  # the fused path, no fallback
        assert not faults.lines()
        ref = patches.ORIGINAL_FORWARD[Qwen2ForCausalLM](model, **batch)
        calls = faults.inject(fused_head, "launch_vocab")
        from solution import engaged

        fell_back = engaged.FELL_BACK["head"]
        out, later = model(**batch), model(**batch)
    assert engaged.FELL_BACK["head"] - fell_back == 2  # both calls counted as fallbacks
    assert calls["n"] == 1
    for result in (out, later):
        assert torch.equal(result.loss, ref.loss) and torch.equal(result.logits, ref.logits)
    out.loss.backward()  # the step completes
    faults.assert_off("fused_head")
    line = engaged.line("test")  # the run's summary shows it: a fallback count and the switch-off
    assert "head:" in line.split(" fallbacks=")[1] and line.endswith(" disabled=fused_head")


def test_unpadded_decoder_failure_reruns_it_padded_through_the_fused_head(faults, monkeypatch):
    from solution import patches

    model = _tiny_lm()
    monkeypatch.setitem(patches._unpad, "enabled", True)
    input_ids, mask, labels = _batch([24, 9, 17, 20])
    batch = dict(input_ids=input_ids, attention_mask=mask, labels=labels)
    with torch.autocast("cuda", dtype=torch.float16):
        ref = model(**batch)  # unpadded
        calls = faults.inject(patches, "rows_context")
        out = model(**batch)
    assert calls["n"] == 1 and out.logits is None  # the fused head ran, on the padded batch
    torch.testing.assert_close(out.loss, ref.loss, rtol=1e-2, atol=1e-2)
    out.loss.backward()
    faults.assert_off("unpad")


def _fake_trainer(model):
    """An SFTTrainer with the attributes TRL's and transformers' compute_loss read."""
    import collections

    import trl

    trainer = object.__new__(trl.SFTTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(use_liger_kernel=False, average_tokens_across_devices=False,
                                   n_gpu=1)
    trainer.accelerator = SimpleNamespace(gather_for_metrics=lambda t: t, num_processes=1,
                                          parallelism_config=None)
    trainer.label_smoother = trainer.compute_loss_func = None
    trainer.model_accepts_loss_kwargs = True
    trainer._metrics = {"train": collections.defaultdict(list),
                        "eval": collections.defaultdict(list)}
    trainer._total_train_tokens = 0
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    return trainer


def test_compute_loss_failure_runs_trls_own_compute_loss(faults, monkeypatch):
    """The fused forward fails inside SFTTrainer.compute_loss: the step runs again through
    TRL's own compute_loss (full logits), with TRL's loss and metrics."""
    pytest.importorskip("trl")
    from solution import fused_head, patches

    model = _tiny_lm()  # applies the solution, compute_loss patch included
    monkeypatch.setitem(patches._host, "defer", False)
    monkeypatch.setitem(patches._host, "bookkeeping", False)
    input_ids, mask, labels = _batch([24, 9, 17, 20])
    num_items = (labels[:, 1:] != -100).sum()

    def step(trainer):
        inputs = dict(input_ids=input_ids, attention_mask=mask, labels=labels)
        with torch.autocast("cuda", dtype=torch.float16):
            return trainer.compute_loss(model, inputs, num_items_in_batch=num_items)

    calls = faults.inject(fused_head, "launch_vocab")
    ours = _fake_trainer(model)
    loss = step(ours)
    assert calls["n"] == 1 and getattr(ours, patches.STEP_ATTR, None) is None
    faults.assert_off("fused_head")
    reference = _fake_trainer(model)  # "fused_head" is off now: TRL's own path
    assert torch.equal(loss, step(reference))
    assert ours._metrics == reference._metrics and ours._total_train_tokens == int(mask.sum())
    loss.backward()


def test_qlora_failures_mid_model_keep_the_values_and_the_dequant_bookkeeping(
        faults, monkeypatch, tiny_qlora):
    """The task's module stack, three failures in one step: the fused q|k|v node fails in the
    first layer at v_proj's adapter (transformers' attention takes over; the last layer's forward
    is a split pass, never fused), the LoRA node that then runs q_proj fails (PEFT takes over),
    then the fused head fails (the whole forward runs again through
    transformers'). The loss and LoRA gradients equal those of the original path with separate
    projections (the LoRA node is bit-identical to PEFT), and the layer groups end the step at
    rest (the failed nodes gave back their weight uses, the failed attempt's backward counts are
    forgotten: no dequantized layer kept between passes)."""
    from transformers import Qwen2ForCausalLM

    from solution import fused_head, fused_proj, lora, patches, unpad

    model = tiny_qlora
    inner = model.get_base_model()
    monkeypatch.setenv("SOLUTION_VARLEN", "1")
    monkeypatch.setattr(unpad, "_VARLEN", True)
    monkeypatch.setitem(patches._unpad, "enabled", True)
    params = [p for p in model.parameters() if p.requires_grad]
    input_ids, mask, labels = _batch([24, 9, 17, 20])
    batch = dict(input_ids=input_ids, attention_mask=mask, labels=labels,
                 num_items_in_batch=(labels[:, 1:] != -100).sum())

    def run(forward):
        for p in params:
            p.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            out = forward()
        out.loss.backward()
        groups = {id(m._solution_group[0]): m._solution_group[0] for m in inner.modules()
                  if getattr(m, "_solution_group", None) is not None}
        assert len(groups) == 2  # both layers packed (on their first forward)
        for group in groups.values():
            assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
            assert group.expected_bwd == 0 and group.split_left == 0
        return out, [p.grad.clone() for p in params]

    with monkeypatch.context() as patch:  # the reference: separate q, k, v projections
        patch.setattr(fused_proj, "qkv_plan", lambda attn, x: None)
        ref, ref_grads = run(lambda: patches.ORIGINAL_FORWARD[Qwen2ForCausalLM](inner, **batch))
    fused = faults.inject(fused_proj, "_power_of_two", when=_nth(2))  # layer 0's v adapter
    node = faults.inject(lora, "_power_of_two")  # the LoRA node, which then takes layer 1's q
    head = faults.inject(fused_head, "launch_vocab")
    out, grads = run(lambda: model(**batch))
    assert fused["raised"] == node["raised"] == head["raised"] == 1
    assert out.logits is not None and torch.equal(out.loss, ref.loss)
    assert all(torch.equal(a, b) for a, b in zip(grads, ref_grads))
    faults.assert_off("fused_proj", "lora", "fused_head")


# --- backward failures: the same gradient from PyTorch ops ----------------------------------

def test_rmsnorm_and_addnorm_backward_failures_compute_the_same_gradient(faults):
    from solution import rmsnorm

    torch.manual_seed(6)
    x = torch.randn(64, 1536, device="cuda", requires_grad=True)
    b = torch.randn(64, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.rand(1536, device="cuda") + 0.5
    dy = torch.randn(64, 1536, device="cuda", dtype=torch.float16)
    ds = torch.randn(64, 1536, device="cuda")

    def grads():
        x.grad = b.grad = None
        y = rmsnorm.RMSNorm.apply(x, w, 1e-6, torch.float16)
        s, y2 = rmsnorm.AddRMSNorm.apply(x, b, w, 1e-6, torch.float16)
        ((y.float() * dy).sum() + (y2.float() * dy).sum() + (s * ds).sum()).backward()
        return x.grad.clone(), b.grad.clone()

    ref = grads()
    calls = faults.inject(rmsnorm, "launch", when=lambda *a, **k: k.get("backward") == 1,
                          times=2)
    from solution.engaged import FELL_BACK

    before = (FELL_BACK["rmsnorm"], FELL_BACK["addnorm"])
    got = grads()
    assert calls["raised"] == 2
    assert (FELL_BACK["rmsnorm"] - before[0], FELL_BACK["addnorm"] - before[1]) == (1, 1)
    for ours, theirs in zip(got, ref):
        torch.testing.assert_close(ours, theirs, rtol=1e-3, atol=1e-3)
    faults.assert_off("rmsnorm", "addnorm")


def test_rope_backward_failure_computes_the_same_gradient(faults):
    from solution import rope

    torch.manual_seed(7)
    q = torch.randn(1, 40, 12, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(1, 40, 2, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    cos, sin = torch.randn(2, 1, 40, 128, device="cuda").unbind(0)
    gq = torch.randn(1, 12, 40, 128, device="cuda", dtype=torch.float16)
    gk = torch.randn(1, 2, 40, 128, device="cuda", dtype=torch.float16)

    def grads():
        q1, k1 = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        oq, ok = rope.RoPE.apply(q1, k1, cos, sin)
        torch.autograd.backward((oq, ok), (gq, gk))
        return q1.grad, k1.grad

    ref = grads()
    calls = faults.inject(rope, "_launch", when=lambda *a, **k: k.get("backward") == 1)
    got = grads()
    assert calls["raised"] == 1
    for ours, theirs in zip(got, ref):
        torch.testing.assert_close(ours, theirs, rtol=1e-3, atol=2e-3)
    faults.assert_off("rope")


def test_swiglu_backward_failure_computes_the_same_gradient(faults, monkeypatch):
    from solution import swiglu_inplace

    torch.manual_seed(8)
    gate = torch.randn(64, 512, device="cuda", dtype=torch.float16)
    up = torch.randn(64, 512, device="cuda", dtype=torch.float16)
    grad = torch.randn(64, 512, device="cuda", dtype=torch.float16)

    def grads():
        g1, u1 = gate.clone().requires_grad_(True), up.clone().requires_grad_(True)
        swiglu_inplace.SwiGLUInPlaceGrad.apply(g1, u1).backward(grad)
        return g1.grad, u1.grad

    ref = grads()
    kernel = _FlakyKernel(swiglu_inplace.example_swiglu_bwd_kernel)
    monkeypatch.setattr(swiglu_inplace, "example_swiglu_bwd_kernel", kernel)
    got = grads()
    assert kernel.launches == 1
    for ours, theirs in zip(got, ref):
        torch.testing.assert_close(ours, theirs, rtol=2e-3, atol=2e-3)
    faults.assert_off("swiglu")


# --- set-up and build steps ---------------------------------------------------------------

def test_apply_leaves_a_component_off_when_its_installation_fails(faults, monkeypatch):
    import solution
    from solution import rope

    monkeypatch.setenv("SOLUTION_WARMUP_THREAD", "0")
    monkeypatch.setenv("SOLUTION_DISABLE_STARTUP", "1")
    calls = faults.inject(rope, "patch_qwen2_rope")
    solution.apply()  # completes
    assert calls["n"] == 1
    faults.assert_off("rope")


def test_triton_helper_build_falls_back_to_tritons_own(faults, monkeypatch, tmp_path):
    from triton import knobs

    from solution import triton_build

    monkeypatch.setattr(knobs.build, "impl", triton_build._build)
    calls = faults.inject(triton_build, "_build_o0")
    src = tmp_path / "solution_probe.c"
    src.write_text("int solution_probe(void) { return 1; }\n")
    so = triton_build._build("solution_probe", str(src), str(tmp_path), [], [], [])
    assert calls["n"] == 1 and os.path.exists(so)  # built by Triton's own command
    assert knobs.build.impl is None  # Triton's own builds from now on
    faults.assert_off("triton_build")


def test_fp16_embed_failure_leaves_the_weight_as_peft_left_it(faults, monkeypatch):
    import peft.utils.other as other

    from solution import fp16_embed
    from solution.test_fp16_embed import _fp16_values, _TiedToy

    def upcast(model, *args, **kwargs):  # PEFT's fp32 upcast of the tied weight
        weight = model.get_input_embeddings().weight
        weight.data = weight.data.float()
        return model

    monkeypatch.setattr(other, "prepare_model_for_kbit_training", upcast)
    assert fp16_embed.patch_prepare_model_for_kbit_training() is True
    prepare = other.prepare_model_for_kbit_training
    toy = _TiedToy(_fp16_values(300, 64, seed=3))
    calls = faults.inject(fp16_embed, "bitwise_equal")
    assert prepare(toy) is toy and calls["n"] == 1
    weight = toy.embed.weight
    assert weight.dtype == torch.float32 and toy.head.weight is weight
    assert not toy.embed._forward_hooks  # no fp32 hook on an fp32 weight
    faults.assert_off("fp16_embed")


# --- what must propagate ------------------------------------------------------------------

@pytest.mark.parametrize("exc", [KeyboardInterrupt("injected"),
                                 torch.cuda.OutOfMemoryError("CUDA out of memory (injected)"),
                                 torch.utils.checkpoint._StopRecomputationError()])
def test_interrupt_out_of_memory_and_checkpoint_early_stop_propagate(faults, exc):
    """Ctrl-C, CUDA out of memory and the early stop of a non-reentrant checkpoint's
    recomputation (PyTorch's control flow through our forwards) pass through untouched."""
    from solution import rmsnorm

    norm = _norm()
    x = torch.randn(3, 7, 256, device="cuda")
    faults.inject(rmsnorm, "launch", exc=exc)
    with pytest.raises(type(exc)):
        norm(x)
    assert not faults.state.DISABLED and not faults.lines()
    # still our kernel (fp32 here: the last bits of rsqrt differ from transformers')
    torch.testing.assert_close(norm(x), _transformers_rmsnorm(norm, x), rtol=1e-4, atol=1e-5)
