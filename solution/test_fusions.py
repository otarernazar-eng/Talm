"""The residual add folded into RMSNorm (`rmsnorm.AddRMSNorm`, `patch_qwen2_add_norm`) and the
variable-length attention of all rows in one launch per KV head (`unpad.VarlenRowsAttention`)
against plain PyTorch and the per-row `RowsAttention`, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("n_rows", [1, 77, 1427])
def test_add_rmsnorm_matches_pytorch(n_rows):
    from solution.rmsnorm import AddRMSNorm

    torch.manual_seed(30)
    hidden, eps = 1536, 1e-6
    x = torch.randn(1, n_rows, hidden, device="cuda", requires_grad=True)
    branch = torch.randn(1, n_rows, hidden, device="cuda", dtype=torch.float16,
                         requires_grad=True)
    weight = torch.rand(hidden, device="cuda") + 0.5
    g_sum = torch.randn(1, n_rows, hidden, device="cuda")
    g_y = torch.randn(1, n_rows, hidden, device="cuda", dtype=torch.float16)

    ref_sum = x + branch                                                    
    var = ref_sum.pow(2).mean(-1, keepdim=True)
    ref_y = (weight * (ref_sum * torch.rsqrt(var + eps))).to(torch.float16)
    ((ref_sum * g_sum).sum() + (ref_y.float() * g_y.float()).sum()).backward()
    ref_dx, ref_db = x.grad.clone(), branch.grad.clone()
    x.grad = branch.grad = None

    total, y = AddRMSNorm.apply(x, branch, weight, eps, torch.float16)
    assert total.dtype == torch.float32 and y.dtype == torch.float16
    torch.testing.assert_close(total, ref_sum.detach(), rtol=0, atol=0)                     
    torch.testing.assert_close(y.float(), ref_y.float(), rtol=2e-3, atol=2e-3)
    ((total * g_sum).sum() + (y.float() * g_y.float()).sum()).backward()
    assert branch.grad.dtype == torch.float16
    torch.testing.assert_close(x.grad, ref_dx, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(branch.grad.float(), ref_db.float(), rtol=2e-3, atol=2e-3)


def test_add_norm_layer_matches_original_layer():
    """The patched Qwen2DecoderLayer (residual add inside post_attention_layernorm) gives the
    original layer's output and input gradient under fp16 autocast."""
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution.rmsnorm import LAYER_FORWARDS, patch_qwen2_add_norm, patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    patch_qwen2_add_norm()
    cls = modeling_qwen2.Qwen2DecoderLayer
    original, fused = LAYER_FORWARDS["original"], LAYER_FORWARDS["add_norm"]
    config = Qwen2Config(hidden_size=256, intermediate_size=512, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         attn_implementation="sdpa")
    torch.manual_seed(31)
    layer = cls(config, layer_idx=0).cuda().train()
    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        norm.weight.data.uniform_(0.5, 1.5)
    layer.requires_grad_(False)
    rotary = modeling_qwen2.Qwen2RotaryEmbedding(config).cuda()
    x = torch.randn(1, 40, 256, device="cuda")
    pos = torch.arange(40, device="cuda")[None]
    grad = torch.randn(1, 40, 256, device="cuda")

    def run(fn):
        inp = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            cos, sin = rotary(inp, pos)
            out = fn(layer, inp, attention_mask=None, position_ids=pos,
                     position_embeddings=(cos, sin))
        (out.float() * grad).sum().backward()
        return out, inp.grad

    ref_out, ref_dx = run(original)
    out, dx = run(fused)
    assert out.dtype == ref_out.dtype == torch.float32
    torch.testing.assert_close(out, ref_out, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(dx, ref_dx, rtol=2e-2, atol=2e-2)


def test_add_norm_across_layers_is_bit_identical():
    """A 3-layer Qwen2Model with every residual add folded into the norm that reads it (the
    attention's into post_attention_layernorm, the MLP's into the next input_layernorm and, for
    the last layer, the final norm) gives the same output and input gradient, bit for bit, as
    transformers' layer forward with separate adds (both with our RMSNorm kernel)."""
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution import rmsnorm

    rmsnorm.patch_qwen2_rmsnorm()
    rmsnorm.patch_qwen2_add_norm()
    cls = modeling_qwen2.Qwen2DecoderLayer
    config = Qwen2Config(hidden_size=256, intermediate_size=512, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=3, vocab_size=128,
                         max_position_embeddings=128, attn_implementation="sdpa")
    torch.manual_seed(35)
    model = modeling_qwen2.Qwen2Model(config).cuda().train()
    for module in model.modules():
        if isinstance(module, modeling_qwen2.Qwen2RMSNorm):
            module.weight.data.uniform_(0.5, 1.5)
    model.requires_grad_(False)
    embeds = torch.randn(1, 40, 256, device="cuda")
    grad = torch.randn(1, 40, 256, device="cuda", dtype=torch.float16)

    def run(layer_forward, linked):
        for layer in model.layers:
            layer.__dict__.pop("_solution_next_norm", None)
        model.__dict__.pop("_solution_norms_linked", None)
        if linked:
            rmsnorm.link_next_norms(model)
        saved = cls.forward
        cls.forward = layer_forward
        try:
            inp = embeds.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(inputs_embeds=inp).last_hidden_state
            (out.float() * grad.float()).sum().backward()
        finally:
            cls.forward = saved
        return out, inp.grad

    ref_out, ref_dx = run(rmsnorm.LAYER_FORWARDS["original"], linked=False)
    _, again_dx = run(rmsnorm.LAYER_FORWARDS["original"], linked=False)
    calls = {"norm": 0, "add_norm": 0}
    norm_apply, add_apply = rmsnorm.RMSNorm.apply, rmsnorm.AddRMSNorm.apply

    def count(name, fn):
        def wrapped(*args):
            calls[name] += 1
            return fn(*args)
        return wrapped

    rmsnorm.RMSNorm.apply = count("norm", norm_apply)
    rmsnorm.AddRMSNorm.apply = count("add_norm", add_apply)
    try:
        out, dx = run(rmsnorm.LAYER_FORWARDS["add_norm"], linked=True)
    finally:
        rmsnorm.RMSNorm.apply, rmsnorm.AddRMSNorm.apply = norm_apply, add_apply
                                                                                    
    assert calls == {"norm": 1, "add_norm": 6}
    assert model.layers[-1]._solution_next_norm is model.norm
    assert out.dtype == torch.float16 and torch.equal(out, ref_out)
                                                                                                
                                                                                            
    spread = float((ref_dx - again_dx).abs().max())
    if spread == 0.0:
        assert torch.equal(dx, ref_dx)
    else:
        torch.testing.assert_close(dx, ref_dx, rtol=2e-3,
                                   atol=2e-3 * float(ref_dx.abs().max()) + 2 * spread)


def _row_bounds(lengths):
    bounds, start = [], 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length
    return bounds, start


@pytest.mark.parametrize("lengths", [[37, 5, 128, 64], [512], [1, 300, 0, 211]])
def test_varlen_attention_matches_rows_attention(lengths):
    """All rows in one launch per KV head == one memory-efficient SDPA call per row
    (`RowsAttention`, the kernel SDPA takes on a T4), with K/V broadcast over the 6 query heads
    that share them (no repeat_kv copy): the forward bit for bit; the gradients bit for bit where
    that backward is deterministic (on a T4 it may split the keys and accumulate dQ/dK in a
    varying order, so there the reference is only as exact as it is with itself)."""
    from solution.unpad import RowsAttention, VarlenRowsAttention, rows_context, _cu

    torch.manual_seed(33)
    bounds, n = _row_bounds(lengths)
    q = torch.randn(1, 12, n, 128, device="cuda", dtype=torch.float16)
    k, v = (torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16) for _ in range(2))
    grad = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16)
    scale = 128 ** -0.5
    rows = [(s, e) for s, e in bounds if e > s]

    def run(fn):
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = fn(q1, k1, v1)
        out.backward(grad)
        return out.detach(), [t.grad for t in (q1, k1, v1)]

    def varlen(q1, k1, v1):
        with rows_context(bounds, device="cuda"):
            cu, max_len, total = _cu[-1]
            assert total == n and max_len == max(lengths)
            return VarlenRowsAttention.apply(q1, k1, v1, cu, max_len, scale)

    ref, ref_grads = run(lambda q1, k1, v1: RowsAttention.apply(q1, k1, v1, rows, scale))
    _, again = run(lambda q1, k1, v1: RowsAttention.apply(q1, k1, v1, rows, scale))
    out, grads = run(varlen)
    assert out.shape == (1, n, 12, 128) and out.is_contiguous()
    assert torch.equal(out, ref)                                            
    for ours, theirs, theirs_again in zip(grads, ref_grads, again):
        assert ours.shape == theirs.shape
        spread = float((theirs.float() - theirs_again.float()).abs().max())
        if spread == 0.0:
            assert torch.equal(ours, theirs)
        else:
            torch.testing.assert_close(ours, theirs, rtol=2e-3,
                                       atol=2e-3 * float(theirs.abs().max()) + 2 * spread)


@pytest.mark.parametrize("lengths", [[37, 5, 128, 64], [1, 300, 0, 211]])
def test_varlen_backward_without_copies_and_cats_is_exact(monkeypatch, lengths):
    """VarlenRowsAttention.backward with the gradient copied once into KV-head-major order and
    the key/value gradients reduced straight into their slices (no per-head copies inside the
    kernel, no concatenation): the same gradients in the same layout. Bit for bit on the RTX 4080;
    on the T4 the efficient-attention backward is not run-to-run exact (its dQ reduction differs
    by an ulp between layouts and even between identical launches: it failed once in a Kaggle
    notebook and passed in the next two sessions), so the bound is twice the spread between two
    identical launches, at least one fp16 ulp of the largest element (atol = max|g| * 2^-10)."""
    from solution import unpad
    from solution.unpad import VarlenRowsAttention, rows_context, _cu

    torch.manual_seed(35)
    bounds, n = _row_bounds(lengths)
    q = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    k, v = (torch.randn(1, n, 2, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
            for _ in range(2))
    grad = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16)
    results = {}
    for name, nocat in (("cat", False), ("nocat", True), ("cat_again", False)):
        monkeypatch.setattr(unpad, "_NOCAT", nocat)
        q1, k1, v1 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        with rows_context(bounds, device="cuda"):
            cu, max_len, _ = _cu[-1]
            out = VarlenRowsAttention.apply(q1, k1, v1, cu, max_len, 128 ** -0.5)
        dq, dk, dv = torch.autograd.grad(out, (q1, k1, v1), grad)
        results[name] = (dq, dk, dv)
    for ours, theirs, again in zip(results["nocat"], results["cat"], results["cat_again"]):
        assert ours.shape == theirs.shape and ours.stride() == theirs.stride()
        spread = float((theirs.float() - again.float()).abs().max())
        if spread == 0.0 and torch.equal(ours, theirs):
            continue
        ulp = float(theirs.float().abs().max()) * 2.0 ** -10                                       
        torch.testing.assert_close(ours, theirs, rtol=0, atol=max(2 * spread, ulp))


def test_rows_sdpa_takes_the_varlen_path(monkeypatch):
    """Through the registered attention interface, with the offsets on the GPU: the
    single-launch path runs (forced here; on its own only below compute capability 8) and
    gives the block-diagonal causal attention."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from solution import unpad

    monkeypatch.setenv("SOLUTION_VARLEN", "1")
    unpad.patch_sdpa_rows()
    interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
    calls = {"n": 0}
    apply = unpad.VarlenRowsAttention.apply

    def counted(*args):
        calls["n"] += 1
        return apply(*args)

    monkeypatch.setattr(unpad.VarlenRowsAttention, "apply", counted)
    torch.manual_seed(34)
    lengths = [37, 5, 128, 64]
    bounds, n = _row_bounds(lengths)
    q = torch.randn(1, 12, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)

    class Module:
        num_key_value_groups = 6
        training = True

    with unpad.rows_context(bounds, device="cuda"):
        out, _ = interface(Module(), q, k, v, None, dropout=0.0, scaling=128 ** -0.5,
                           sliding_window=None)
    assert calls["n"] == 1
    row = torch.cat([torch.full((length,), i, device="cuda") for i, length in enumerate(lengths)])
    pos = torch.cat([torch.arange(length, device="cuda") for length in lengths])
    mask = (row[:, None] == row[None, :]) & (pos[None, :] <= pos[:, None])
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(6, dim=1),
                                         v.repeat_interleave(6, dim=1), attn_mask=mask,
                                         scale=128 ** -0.5).transpose(1, 2)
                                                                                
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3,
                               atol=4e-4 * float(ref.detach().abs().max()))
