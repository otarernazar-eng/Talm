"""fused_proj.py (q|k|v and gate|up as one GEMM each on the packed layer) against the separate
projections: outputs and input gradients within fp16 GEMM rounding, the LoRA adapter gradients
and the CUDA generator state bit-identical to PEFT's, one dequantization per pass, the backward
of the layer before the last (a split pass), and the patched Qwen2 attention and MLP against the
forwards they replace.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _group_of(module):
    base = getattr(module, "base_layer", module)
    return base._solution_group[0]


def _reset(group):
    """A clean pass bookkeeping (the tests below run partial layers)."""
    group.cache = {"fwd": [None, 0], "bwd": [None, 0]}
    group.expected_bwd = group.split_left = 0


def _close(ours, ref, scale=2e-3):
    """Within a few fp16 roundings: absolute tolerance relative to the largest magnitude of the
    reference (an fp16 rounding is relative to the magnitude of the operands of the last sum,
    not of the element, so near-cancelling elements may differ by more than rtol)."""
    torch.testing.assert_close(ours, ref, rtol=2e-3, atol=scale * float(ref.abs().max()))


@pytest.mark.parametrize("n_tokens", [1, 150, 1427])
@pytest.mark.parametrize("train", [True, False])
def test_fused_qkv_matches_peft(n_tokens, train):
    """q, k, v from one norm output with the dropout on (train) or off, PEFT's own forward as
    the reference: the adapter weight gradients and the generator state afterwards are PEFT's
    bit for bit (the adapter ops are PEFT's, the same draws); the outputs differ at most by the
    base GEMM's rounding, the input gradient by the base gradients' fp32 sum."""
    import bitsandbytes as bnb

    from solution import fused_proj, lora
    from solution.qgroup import setup_layer
    from solution.test_lora import _group_forward, _lora_layer

    assert lora.patch_peft_lora() is True
    torch.manual_seed(23)
    layer = _lora_layer(hidden=1536, kv=256)
    setup_layer(layer)
    attn = layer.self_attn
    q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
    group = _group_of(q)
    for module in (q, v):
        module.train(train)
    original = type(q).forward._original
    params = [m.lora_A["default"].weight for m in (q, v)] + \
             [m.lora_B["default"].weight for m in (q, v)]
    leaf = torch.randn(1, n_tokens, 1536, device="cuda", dtype=torch.float16)
    grads = [torch.randn(1, n_tokens, n, device="cuda", dtype=torch.float16)
             for n in (1536, 256, 256)]

    def run(project):
        _reset(group)
        for prm in params:
            prm.grad = None
        x = leaf.clone().requires_grad_(True)
        torch.cuda.manual_seed(11)
        with torch.autocast("cuda", dtype=torch.float16):
            h = x * 1.0
            outs = project(h)
        sum((o * g).float().sum() for o, g in zip(outs, grads)).backward()
        return ([o.detach() for o in outs], x.grad, [prm.grad.clone() for prm in params],
                torch.cuda.get_rng_state())

    def peft(h):
        return [original(q, h), k(h), original(v, h)]

    def fused(h):
        plan = fused_proj.qkv_plan(attn, h)
        assert plan is not None
        plan_group, adapters, weights = plan
        assert plan_group is group and [a[0] for a in adapters] == [0, 2]
        group.expected_bwd += 3
        return fused_proj.FusedProjections.apply(h, group, 0, 3, adapters, (2,), *weights)

    saved = bnb.nn.Linear4bit.forward
    bnb.nn.Linear4bit.forward = _group_forward
    try:
        ref_outs, ref_dx, ref_grads, ref_state = run(peft)
        outs, dx, ours, state = run(fused)
    finally:
        bnb.nn.Linear4bit.forward = saved
    for g, r in zip(ours, ref_grads):
        assert g.dtype == torch.float32 and torch.equal(g, r)
    assert torch.equal(state, ref_state)
    for o, r in zip(outs, ref_outs):
        assert o.dtype == torch.float16 and o.shape == r.shape
        _close(o, r)
    assert outs[2].is_contiguous()  # v is kept by the attention: not a view of the [N, 2048]
    # PEFT rounds each of the five contributions to fp16 and adds them in fp16; here the three
    # base ones are summed in fp32: a difference of an fp16 rounding of dx's magnitude
    _close(dx, ref_dx)


def test_fused_qkv_follows_the_optimizer(monkeypatch):
    """The node takes the adapters' fp16 A and B from lora.py's cast cache: after an AdamW step
    its outputs equal, bit for bit, those computed with a fresh `.to(float16)` of the new
    weights."""
    from solution import fused_proj, lora
    from solution.qgroup import setup_layer
    from solution.test_lora import _lora_layer

    assert lora.patch_peft_lora() is True
    torch.manual_seed(28)
    layer = _lora_layer(hidden=1536, kv=256)
    setup_layer(layer)
    attn = layer.self_attn
    group = _group_of(attn.q_proj)
    for module in (attn.q_proj, attn.v_proj):
        module.eval()
    x = torch.randn(1, 200, 1536, device="cuda", dtype=torch.float16)
    params = [m.lora_A["default"].weight for m in (attn.q_proj, attn.v_proj)] + \
             [m.lora_B["default"].weight for m in (attn.q_proj, attn.v_proj)]

    def fused():
        _reset(group)
        with torch.autocast("cuda", dtype=torch.float16):
            _, adapters, weights = fused_proj.qkv_plan(attn, x)
            outs = fused_proj.FusedProjections.apply(x, group, 0, 3, adapters, (2,), *weights)
        return [o.detach().clone() for o in outs]

    before = fused()
    for prm in params:
        prm.grad = torch.randn_like(prm)
    torch.optim.AdamW(params, lr=0.05).step()
    after = fused()
    monkeypatch.setattr(lora, "CAST_CACHE", False)
    fresh = fused()
    assert not torch.equal(before[0], after[0]) and not torch.equal(before[2], after[2])
    for a, b in zip(after, fresh):
        assert torch.equal(a, b)


@pytest.mark.parametrize("n_tokens", [1, 64, 77, 1427])
def test_fused_mlp_matches_separate_projections(n_tokens):
    """gate|up as one transposed GEMM, the example's SwiGLU on its halves, down_proj on the
    transposed activation == the separate projections through swiglu_inplace.py, within the
    GEMMs' rounding (token counts with and without padding of the transposed buffers); the
    in-place and the out-of-place backward agree bit for bit."""
    from solution import fused_proj
    from solution.qgroup import GroupLinear, setup_layer
    from solution.swiglu_inplace import SwiGLUInPlaceGrad
    from solution.test_kernels import _fake_layer

    torch.manual_seed(41)
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    setup_layer(layer)
    group = _group_of(layer.mlp.gate_proj)
    leaf = torch.randn(1, n_tokens, 512, device="cuda", dtype=torch.float16)
    grad = torch.randn(1, n_tokens, 512, device="cuda", dtype=torch.float16)

    def separate(x):
        group.expected_bwd += 3
        act = SwiGLUInPlaceGrad.apply(GroupLinear.apply(x, group, 4),
                                      GroupLinear.apply(x, group, 5))
        return GroupLinear.apply(act, group, 6)

    def fused(inplace):
        def run(x):
            assert fused_proj.mlp_plan(layer.mlp, x) is group
            group.expected_bwd += 3
            return fused_proj.FusedMLP.apply(x, group, inplace)
        return run

    results = []
    for fn in (separate, fused(True), fused(False)):
        _reset(group)
        x = leaf.clone().requires_grad_(True)
        y = fn(x)
        (y.float() * grad.float()).sum().backward()
        assert group.expected_bwd == 0
        results.append((y.detach(), x.grad))
    (ref_y, ref_dx), (y, dx), (y2, dx2) = results
    assert torch.equal(y, y2) and torch.equal(dx, dx2)
    _close(y, ref_y)
    _close(dx, ref_dx)


def test_fused_nodes_keep_what_the_separate_path_kept():
    """Memory: the MLP node keeps [gate^T; up^T] (what the SwiGLU node kept, gate and up) and
    nothing else of activation size; the q|k|v node keeps only its input (as the LoRA node),
    and the v it returns, which the attention keeps, is its own tensor, not a view that would
    keep the whole [N, sum of out] GEMM result alive."""
    from solution.fused_proj import FusedMLP, FusedProjections
    from solution.qgroup import setup_layer
    from solution.test_kernels import _fake_layer

    torch.manual_seed(44)
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    setup_layer(layer)
    group = _group_of(layer.mlp.gate_proj)
    _reset(group)
    n_tok = 300
    x = torch.randn(1, n_tok, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    saved = []

    def pack(t):
        saved.append((t.untyped_storage().data_ptr(), t.numel()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        q, k, v = FusedProjections.apply(x, group, 0, 3, (), (2,))
        y = FusedMLP.apply(x, group, True)
    big = [s for s in saved if s[1] >= x.numel()]
    n_pad = -(-n_tok // 8) * 8  # the transposed buffer's token columns, padded to 8
    assert big == [(x.untyped_storage().data_ptr(), x.numel()), (big[1][0], 2 * 1024 * n_pad)]
    assert v.untyped_storage().data_ptr() != q.untyped_storage().data_ptr()
    assert q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    del y


def _layer_pass(group, x, adapters=(), weights=()):
    """A whole layer's projections as the patched forwards run them: q|k|v fused, o_proj alone,
    the MLP fused; returns what reaches the loss."""
    from solution.fused_proj import FusedMLP, FusedProjections
    from solution.qgroup import GroupLinear

    group.expected_bwd += 3
    q, k, v = FusedProjections.apply(x, group, 0, 3, adapters, (2,), *weights)
    group.expected_bwd += 1
    o = GroupLinear.apply(q, group, 3)
    group.expected_bwd += 3
    y = FusedMLP.apply(o, group, True)
    return [q, k, v, y]


def _counting(monkeypatch):
    from solution.qgroup import LayerGroup

    calls = {"all": 0, "one": 0}
    orig_all, orig_one = LayerGroup.dequantize_all, LayerGroup.dequantize_one

    def count_all(self):
        calls["all"] += 1
        return orig_all(self)

    def count_one(self, idx):
        calls["one"] += 1
        return orig_one(self, idx)

    monkeypatch.setattr(LayerGroup, "dequantize_all", count_all)
    monkeypatch.setattr(LayerGroup, "dequantize_one", count_one)
    return calls


def test_fused_layer_dequantizes_once_per_pass(monkeypatch):
    """The fused nodes take their weights as spans of the one buffer per pass: one launch in
    the forward, one in the backward, both buffers dropped after the layer's seventh use."""
    from solution import qgroup
    from solution.qgroup import setup_layer
    from solution.test_kernels import _fake_layer

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 0, "bwd": 0})
    torch.manual_seed(42)
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    setup_layer(layer)
    group = _group_of(layer.self_attn.q_proj)
    _reset(group)
    calls = _counting(monkeypatch)
    x = torch.randn(1, 40, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    outs = _layer_pass(group, x)
    assert calls == {"all": 1, "one": 0} and group.cache["fwd"][0] is None
    sum((o.float() ** 2).mean() for o in outs).backward()
    assert calls == {"all": 2, "one": 0} and group.cache["bwd"][0] is None
    assert group.expected_bwd == 0


def test_fused_backward_in_a_split_pass(monkeypatch):
    """The layer before the last: its forward dequantizes the whole layer (fused), its backward
    each projection alone (qgroup.SPLIT_LAYERS). The fused nodes then take the weights one at a
    time and accumulate the separate GEMMs; same outputs bit for bit, input gradient within
    rounding of the unsplit backward, and the bookkeeping is reset after each step."""
    from solution import qgroup
    from solution.qgroup import setup_layer
    from solution.test_kernels import _fake_layer

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    torch.manual_seed(43)
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    setup_layer(layer)
    group = _group_of(layer.self_attn.q_proj)
    assert group.splittable
    calls = _counting(monkeypatch)
    leaf = torch.randn(1, 40, 512, device="cuda", dtype=torch.float16)
    results = []
    for depth in (1, None):  # None: never split, the reference
        group.depth = depth
        _reset(group)
        for _ in range(2):  # two steps: the bookkeeping must be reset after each
            calls.update(all=0, one=0)
            x = leaf.clone().requires_grad_(True)
            outs = _layer_pass(group, x)
            assert calls == {"all": 1, "one": 0}  # the forward is never split here
            sum((o.float() ** 2).mean() for o in outs).backward()
            assert group.split_left == 0 and group.expected_bwd == 0
            assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
        results.append((dict(calls), [o.detach() for o in outs], x.grad))
    assert results[0][0] == {"all": 1, "one": 7} and results[1][0] == {"all": 2, "one": 0}
    for o, r in zip(results[0][1], results[1][1]):
        assert torch.equal(o, r)
    _close(results[0][2], results[1][2])


def _qwen2_4bit_layer(n_layers, layer_idx, rank=16):
    """A Qwen2 decoder layer (hidden 512, 4 query heads and 2 KV heads of 128) whose seven
    projections are bitsandbytes NF4 layers, q_proj and v_proj wrapped by PEFT LoRA."""
    import bitsandbytes as bnb
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    config = Qwen2Config(hidden_size=512, intermediate_size=1024, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=n_layers,
                         max_position_embeddings=256, attn_implementation="sdpa")
    layer = modeling_qwen2.Qwen2DecoderLayer(config, layer_idx=layer_idx)

    def quantized(linear):
        bias = linear.bias is not None
        q4 = bnb.nn.Linear4bit(linear.in_features, linear.out_features, bias=bias,
                               quant_type="nf4", compress_statistics=True,
                               compute_dtype=torch.float16)
        q4.weight = bnb.nn.Params4bit(
            (torch.randn(linear.out_features, linear.in_features) * 0.05).half(),
            requires_grad=False, quant_type="nf4", compress_statistics=True)
        if bias:
            q4.bias = torch.nn.Parameter(torch.randn(linear.out_features) * 0.1,
                                         requires_grad=False)
        return q4

    for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                          (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
        for name in names:
            setattr(parent, name, quantized(getattr(parent, name)))
    layer = layer.cuda()
    layer.requires_grad_(False)
    lora_config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05)
    for name in ("q_proj", "v_proj"):
        wrapped = LoraLinear4bit(getattr(layer.self_attn, name), "default", config=lora_config,
                                 r=rank, lora_alpha=2 * rank)
        wrapped.lora_A.float()
        wrapped.lora_B.float()
        torch.nn.init.normal_(wrapped.lora_B["default"].weight, std=0.05)
        setattr(layer.self_attn, name, wrapped.cuda())
    layer.train()
    return config, layer


@pytest.mark.parametrize("n_layers,fused_attn", [(4, True), (2, True), (1, False)])
def test_patched_qwen2_attention_and_mlp_match_the_separate_path(monkeypatch, n_layers,
                                                                 fused_attn):
    """Qwen2Attention.forward and Qwen2MLP.forward patched by fused_proj on a 4-bit layer with
    LoRA on q and v, against transformers' attention forward and the swiglu_inplace MLP (the
    previous path): the same outputs and gradients within the GEMMs' rounding. Layer depth 3:
    fused, unsplit; depth 1 (the layer before the last): fused forward, split backward; depth 0
    (the last layer): the forward is split, so both patches take the previous path."""
    import bitsandbytes as bnb
    from transformers.models.qwen2 import modeling_qwen2

    from solution import fused_proj, lora
    from solution.qgroup import setup_layer
    from solution.swiglu_inplace import SwiGLUInPlaceGrad
    from solution.test_lora import _group_forward

    def mlp_previous(self, x):
        return self.down_proj(SwiGLUInPlaceGrad.apply(self.gate_proj(x), self.up_proj(x)))

    attn_cls, mlp_cls = modeling_qwen2.Qwen2Attention, modeling_qwen2.Qwen2MLP
    attn_original = getattr(attn_cls.forward, "_original", attn_cls.forward)
    monkeypatch.setattr(bnb.nn.Linear4bit, "forward", _group_forward)
    monkeypatch.setattr(attn_cls, "forward", attn_original)
    monkeypatch.setattr(mlp_cls, "forward", mlp_previous)
    assert lora.patch_peft_lora() is True
    torch.manual_seed(52)
    config, layer = _qwen2_4bit_layer(n_layers, layer_idx=0)
    setup_layer(layer)
    group = _group_of(layer.self_attn.q_proj)
    assert group.depth == n_layers - 1
    rotary = modeling_qwen2.Qwen2RotaryEmbedding(config).cuda()
    n_tok = 97
    pos = torch.arange(n_tok, device="cuda")[None]
    leaf_a = torch.randn(1, n_tok, 512, device="cuda", dtype=torch.float16)
    leaf_m = torch.randn(1, n_tok, 512, device="cuda", dtype=torch.float16)
    grad_a = torch.randn(1, n_tok, 512, device="cuda", dtype=torch.float16)
    grad_m = torch.randn(1, n_tok, 512, device="cuda", dtype=torch.float16)
    params = [p for p in layer.parameters() if p.requires_grad]
    assert len(params) == 4

    def run():
        _reset(group)
        for prm in params:
            prm.grad = None
        h_a = leaf_a.clone().requires_grad_(True)
        h_m = leaf_m.clone().requires_grad_(True)
        torch.cuda.manual_seed(13)
        with torch.autocast("cuda", dtype=torch.float16):
            cos, sin = rotary(leaf_a.float(), pos)  # fp32, as from the fp32 residual stream
            out_a, _ = layer.self_attn(hidden_states=h_a, position_embeddings=(cos, sin),
                                       attention_mask=None)
            out_m = layer.mlp(h_m)
        ((out_a.float() * grad_a.float()).sum() + (out_m.float() * grad_m.float()).sum()) \
            .backward()
        assert group.expected_bwd == 0 and group.split_left == 0
        assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
        return [out_a.detach(), out_m.detach(), h_a.grad, h_m.grad] + \
               [prm.grad.clone() for prm in params]

    ref = run()
    fused_proj.patch_qwen2_attention()
    fused_proj.patch_qwen2_mlp()
    before = dict(fused_proj.COUNTS)
    ours = run()
    took = {key: fused_proj.COUNTS[key] - before[key] for key in before}
    if fused_attn:
        assert took == {"attn": 1, "attn_fallback": 0, "mlp": 1, "mlp_fallback": 0}
    else:  # the previous path itself (only the attention backward may vary run to run)
        assert took == {"attn": 0, "attn_fallback": 1, "mlp": 0, "mlp_fallback": 1}
        assert torch.equal(ours[0], ref[0]) and torch.equal(ours[1], ref[1])
    for a, b in zip(ours[:4], ref[:4]):
        _close(a, b)
    for a, b in zip(ours[4:], ref[4:]):  # the adapters see slightly different upstream grads
        _close(a, b, scale=1e-2)
