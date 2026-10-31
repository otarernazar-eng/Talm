"""The LoRA projection node (`solution/lora.py`) against PEFT's own `Linear4bit.forward`:
bit-identical outputs, gradients and RNG stream with the dropout on, PEFT's path outside the
plain case.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _lora_layer(rank=16, hidden=256, kv=64):
    """A decoder-layer-like namespace: seven 4-bit projections, q and v wrapped by PEFT LoRA."""
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

    from solution.test_kernels import _fake_layer

    layer = _fake_layer(hidden=hidden, kv=kv)
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05)
    for name in ("q_proj", "v_proj"):
        wrapped = LoraLinear4bit(getattr(layer.self_attn, name), "default", config=config, r=rank,
                                 lora_alpha=2 * rank)
        wrapped.lora_A.float()  # fp32 adapter weights, as get_peft_model leaves them in training
        wrapped.lora_B.float()
        torch.nn.init.normal_(wrapped.lora_B["default"].weight, std=0.05)  # PEFT starts at 0
        setattr(layer.self_attn, name, wrapped)
    return layer


def _group_forward(self, x):
    """The training path of a packed 4-bit base (qgroup's Linear4bit patch), pinned here
    whatever other tests patched on bitsandbytes' Linear4bit."""
    from solution.qgroup import GroupLinear

    group, idx = self._solution_group
    if torch.is_grad_enabled() and x.requires_grad:
        group.expected_bwd += 1
    return GroupLinear.apply(x, group, idx)


@pytest.mark.parametrize("n_tokens", [1, 150, 1427])
@pytest.mark.parametrize("train", [True, False])
def test_lora_node_is_bit_identical_to_peft(n_tokens, train):
    """As the attention uses it: q, k, v from one norm output, dropout on (train) or off. The
    outputs, the norm input's gradient (five contributions accumulated by autograd), the adapter
    gradients and the CUDA generator state afterwards must equal PEFT's bit for bit."""
    import bitsandbytes as bnb

    from solution import lora
    from solution.qgroup import setup_layer

    assert lora.patch_peft_lora() is True
    torch.manual_seed(23)
    layer = _lora_layer(hidden=1536, kv=256)
    setup_layer(layer)
    attn = layer.self_attn
    q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
    assert hasattr(q.base_layer, "_solution_group"), "layer was not packed"
    for module in (q, v):
        module.train(train)
    original = type(q).forward._original
    params = [m.lora_A["default"].weight for m in (q, v)] + \
             [m.lora_B["default"].weight for m in (q, v)]
    leaf = torch.randn(1, n_tokens, 1536, device="cuda", dtype=torch.float16)
    grads = [torch.randn(1, n_tokens, n, device="cuda", dtype=torch.float16)
             for n in (1536, 256, 256)]

    def run(lora_forward):
        for prm in params:
            prm.grad = None
        x = leaf.clone().requires_grad_(True)
        torch.cuda.manual_seed(11)
        with torch.autocast("cuda", dtype=torch.float16):
            h = x * 1.0  # a non-leaf: autograd accumulates the five input gradients in a buffer
            outs = [lora_forward(q, h), k(h), lora_forward(v, h)]
        sum((o * g).float().sum() for o, g in zip(outs, grads)).backward()
        return outs, x.grad, [prm.grad.clone() for prm in params], torch.cuda.get_rng_state()

    calls = {"n": 0}
    node_apply = lora.LoraProjectionTorch.apply

    def counted(*args):
        calls["n"] += 1
        return node_apply(*args)

    saved = bnb.nn.Linear4bit.forward
    bnb.nn.Linear4bit.forward = _group_forward
    lora.LoraProjectionTorch.apply = counted
    try:
        ref_outs, ref_dx, ref_grads, ref_state = run(original)
        assert calls["n"] == 0
        outs, dx, ours, state = run(lambda mod, inp: mod(inp))
        assert calls["n"] == 2  # q_proj and v_proj took the node
    finally:
        bnb.nn.Linear4bit.forward = saved
        lora.LoraProjectionTorch.apply = node_apply
    for o, r in zip(outs, ref_outs):
        assert o.dtype == torch.float16 and torch.equal(o, r)
    assert torch.equal(dx, ref_dx)
    for g, r in zip(ours, ref_grads):
        assert g.dtype == torch.float32 and torch.equal(g, r)
    assert torch.equal(state, ref_state)


def test_lora_node_saves_only_the_input():
    """What the node keeps for the backward: the layer input (one tensor for q and v) and the
    [N, r] adapter product, not the dropout output and mask PEFT keeps."""
    from solution import lora
    from solution.qgroup import setup_layer

    assert lora.patch_peft_lora() is True
    torch.manual_seed(24)
    layer = _lora_layer(hidden=1536, kv=256)
    setup_layer(layer)
    q, v = layer.self_attn.q_proj, layer.self_attn.v_proj
    x = torch.randn(1, 300, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    big = []

    def pack(t):
        if t.numel() >= x.numel():
            big.append((t.untyped_storage().data_ptr(), t.dtype))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cuda", dtype=torch.float16):
            q(x)
            v(x)
    assert big and all(p == x.untyped_storage().data_ptr() and d == torch.float16
                       for p, d in big)


def test_half_copies_are_exact_and_follow_every_update():
    """`HalfCopies`: the same bits as `.to(float16)` (subnormals, ties, overflow to inf), one
    refresh for all tensors, a refresh after an in-place update (version counter), after a
    write that bypasses the version counter once an optimizer has stepped (the post-hook), and
    copies handed out earlier are never overwritten."""
    from solution import lora

    lora._install_cast_cache()
    torch.manual_seed(26)
    cache = lora.HalfCopies()
    a = torch.randn(16, 1536, device="cuda") * 0.02
    a.view(-1)[:7] = torch.tensor([1e-8, 6e-8, 2.98e-8, 65519.0, 65520.0, 1e6, -0.1],
                                  device="cuda")
    b = torch.nn.Parameter(torch.randn(256, 16, device="cuda"))
    first = cache.get(a)
    assert first.dtype == torch.float16 and torch.equal(first, a.to(torch.float16))
    assert torch.equal(cache.get(b), b.detach().to(torch.float16))
    assert cache.get(a) is first  # no change: the same copy
    with torch.no_grad():
        b.mul_(3.0)  # in place: the version counter moves
    assert torch.equal(cache.get(b), b.detach().to(torch.float16))
    kept = first.clone()
    a.data.add_(1.0)  # through .data: the version counter of `a` does not move ...
    assert cache.get(a) is first
    cache.invalidate()  # ... but every optimizer step invalidates the copies
    assert torch.equal(cache.get(a), a.to(torch.float16))
    assert torch.equal(first, kept)  # the earlier copy is untouched
    # the post-hook, through a real optimizer step
    p = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    p.grad = torch.ones(4, device="cuda")
    lora._half.get(a)
    torch.optim.SGD([p], lr=0.1).step()
    assert all(entry[1] is None and entry[2] is None for entry in lora._half.entries)


def test_half_copies_hold_no_memory_after_an_optimizer_step():
    """The copies are stale once the optimizer steps: the step drops them (no CUDA memory is
    held from the step to the next forward, or after the last step of a run), the next `get`
    makes them again, bit-exact, and a copy still referenced elsewhere (as a pending backward
    references it) stays as it was."""
    import weakref

    from solution import lora

    lora._install_cast_cache()
    torch.manual_seed(28)
    w = torch.nn.Parameter(torch.randn(16, 1536, device="cuda") * 0.02)
    w.grad = torch.randn_like(w)
    copy = lora._half.get(w)  # held here as a forward's ctx holds it through its backward
    ref, before = weakref.ref(copy), copy.clone()
    torch.optim.SGD([w], lr=0.1).step()
    assert lora._half.entries[lora._half.slots[id(w)]][2] is None
    assert ref() is copy and torch.equal(copy, before)  # the holder's copy is untouched
    del copy
    assert ref() is None  # freed: the cache keeps no reference to it
    assert torch.equal(lora._half.get(w), w.detach().to(torch.float16))


def test_lora_node_follows_the_optimizer():
    """After an AdamW step on the adapter weights the node's output equals PEFT's with the new
    weights, bit for bit (the fp16 copies were refreshed)."""
    import bitsandbytes as bnb

    from solution import lora
    from solution.qgroup import setup_layer

    assert lora.patch_peft_lora() is True
    torch.manual_seed(27)
    layer = _lora_layer(hidden=1536, kv=256)
    setup_layer(layer)
    q = layer.self_attn.q_proj
    q.eval()
    original = type(q).forward._original
    x = torch.randn(1, 200, 1536, device="cuda", dtype=torch.float16)
    params = [q.lora_A["default"].weight, q.lora_B["default"].weight]
    saved = bnb.nn.Linear4bit.forward
    bnb.nn.Linear4bit.forward = _group_forward
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            before = q(x)
            for prm in params:
                prm.grad = torch.randn_like(prm)
            torch.optim.AdamW(params, lr=0.05).step()
            after, ref = q(x), original(q, x)
    finally:
        bnb.nn.Linear4bit.forward = saved
    assert not torch.equal(before, after)
    assert torch.equal(after, ref)


def test_lora_patch_falls_back_to_peft_outside_the_plain_case():
    from solution import lora
    from solution.qgroup import setup_layer

    assert lora.patch_peft_lora() is True
    torch.manual_seed(25)
    layer = _lora_layer()
    setup_layer(layer)
    q = layer.self_attn.q_proj
    x = torch.randn(3, 50, 256, device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        assert lora.lora_plan(q, x) is not None
        assert lora.lora_plan(q, x.float()) is None  # fp32 input
        q.enable_adapters(False)
        assert lora.lora_plan(q, x) is None
        q.enable_adapters(True)
        assert lora.lora_plan(q, x) is not None
    assert lora.lora_plan(q, x) is None  # no autocast: PEFT casts to fp32 itself
    # training mode with lora_B = 0 leaves exactly the base output
    q.train()
    torch.nn.init.zeros_(q.lora_B["default"].weight)
    with torch.autocast("cuda", dtype=torch.float16):
        torch.testing.assert_close(q(x), q.base_layer(x), rtol=0, atol=0)
