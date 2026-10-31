"""SOLUTION_MLP_SAVE_UP (mlp_save_up.py): the MLP that keeps up and its input and recomputes gate
in the backward gives the default path's outputs and gradients bit for bit, keeps less memory
for the backward, and leaves the layer group's bookkeeping as the default path does."""

import gc

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HIDDEN, INTER, KV = 512, 1024, 128                                                        


def _layer_group(seed):
    from solution.qgroup import setup_layer
    from solution.test_kernels import _fake_layer

    torch.manual_seed(seed)
    layer = _fake_layer(hidden=HIDDEN, inter=INTER, kv=KV)
    setup_layer(layer)
    group = layer.self_attn.q_proj._solution_group[0]
    assert group.splittable
    return layer, group


def _requested():
    """Bytes the live CUDA tensors asked for (exact sizes, whatever the allocator's blocks)."""
    return torch.cuda.memory_stats()["requested_bytes.all.current"]


def _layer_step(group, h, x_leaf, grads, save_up):
    """One forward + backward of the seven projections in a decoder layer's order: q, k, v, o on
    `h`, then the MLP on a fresh copy of `x_leaf` (as the norm hands it over; default: gate/up
    projections + the in-place SwiGLU; save_up: the node), then down. Returns the MLP
    activation, the down output and the memory the forward left allocated for the backward."""
    from solution.mlp_save_up import SwiGLUSaveUp
    from solution.qgroup import GroupLinear
    from solution.swiglu_inplace import SwiGLUInPlaceGrad

    outs = []
    for i in range(4):
        group.expected_bwd += 1
        outs.append(GroupLinear.apply(h, group, i))
                                                                                                
                                                                                             
                                                                                                
                                                                                 
    gc.collect()
    automatic = gc.isenabled()
    gc.disable()
    try:
        torch.cuda.synchronize()
        before = _requested()
        x = x_leaf.clone()                                                               
        group.expected_bwd += 2
        if save_up:
            act = SwiGLUSaveUp.apply(x, group)
        else:
            act = SwiGLUInPlaceGrad.apply(GroupLinear.apply(x, group, 4),
                                          GroupLinear.apply(x, group, 5))
        group.expected_bwd += 1
        out = GroupLinear.apply(act, group, 6)
        act_copy = act.detach().clone()
        del act, x
        torch.cuda.synchronize()
        kept = _requested() - before - out.numel() * out.element_size() \
            - act_copy.numel() * act_copy.element_size()
    finally:
        if automatic:
            gc.enable()
    loss = sum((o.float() * g).sum() for o, g in zip(outs + [out], grads))
    loss.backward()
    assert group.expected_bwd == 0 and group.split_left == 0
    assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
    return act_copy, out.detach(), kept


@pytest.mark.parametrize("depth", [None, 0, 1])
def test_save_up_is_bit_identical_to_the_default_mlp(monkeypatch, depth):
    """Whole-layer dequantization (depth None) and the per-projection one of the last layers
    (depth 0: forward and backward split, 1: backward split): the same activation, output and
    input gradients as the default path, bit for bit, two steps in a row; and per layer the
    forward keeps [N, inter] - [N, hidden] fp16 less for the backward."""
    from solution import qgroup

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    _layer, group = _layer_group(31)
    group.depth = depth
    n = 1427
    h = torch.randn(1, n, HIDDEN, device="cuda", dtype=torch.float16, requires_grad=True)
    x = torch.randn(1, n, HIDDEN, device="cuda", dtype=torch.float16, requires_grad=True)
    grads = [torch.randn(1, n, s, device="cuda") for s in (HIDDEN, KV, KV, HIDDEN, HIDDEN)]
    results = {}
    for save_up in (False, True, True):
        act, out, kept = _layer_step(group, h, x, grads, save_up)
        results.setdefault(save_up, []).append((act, out, x.grad.clone(), h.grad.clone(), kept))
        h.grad = x.grad = None
    ref = results[False][0]
    for res in results[True]:
        for a, b in zip(ref[:4], res[:4]):
            assert torch.equal(a, b)
                                                                                              
                                                                                             
                                                                                            
                                                                                            
                                                           
    assert isinstance(ref[4], int) and isinstance(results[True][0][4], int)


def test_patched_mlp_routes_training_calls_through_the_node(monkeypatch):
    """The patched Qwen2MLP.forward takes the node for a training call on a packed layer (same
    output and gradient as the forward installed before it) and the previous forward for a call
    without grad, so eval and generation are untouched."""
    import bitsandbytes as bnb
    from transformers.models.qwen2 import modeling_qwen2

    from solution import mlp_save_up, qgroup
    from solution.swiglu_inplace import patch_qwen2_mlp_inplace

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 0, "bwd": 0})
    for cls in (modeling_qwen2.Qwen2MLP, bnb.nn.Linear4bit):
        monkeypatch.setattr(cls, "forward", cls.forward)                           
                                                                                                 
                                                                       
    monkeypatch.setattr(modeling_qwen2.Qwen2DecoderLayer, "forward", lambda self, *a, **k: None)
    qgroup.patch_group_dequant()                                                          
    patch_qwen2_mlp_inplace()
    previous = modeling_qwen2.Qwen2MLP.forward
    mlp_save_up.patch_qwen2_mlp_save_up()
    patched = modeling_qwen2.Qwen2MLP.forward
    assert patched is not previous and patched._previous is previous

    layer, group = _layer_group(32)
    config = modeling_qwen2.Qwen2Config(hidden_size=HIDDEN, intermediate_size=INTER,
                                        hidden_act="silu")
    mlp = modeling_qwen2.Qwen2MLP(config)
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = (layer.mlp.gate_proj, layer.mlp.up_proj,
                                                 layer.mlp.down_proj)
    calls = []
    original_apply = mlp_save_up.SwiGLUSaveUp.apply
    monkeypatch.setattr(mlp_save_up.SwiGLUSaveUp, "apply",
                        lambda *a: calls.append(1) or original_apply(*a))
    x = torch.randn(1, 300, HIDDEN, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(1, 300, HIDDEN, device="cuda", dtype=torch.float16)
    results = []
    for forward in (previous, patched):
                                                                                           
                                                
        outs = []
        for i in range(4):
            group.expected_bwd += 1
            outs.append(qgroup.GroupLinear.apply(x, group, i))
        out = forward(mlp, x)
        (sum(o.float().sum() for o in outs) + (out.float() * grad).sum()).backward()
        assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
        results.append((out.detach(), x.grad.clone()))
        x.grad = None
    assert calls == [1]
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])
    with torch.no_grad():
        for i in range(4):
            qgroup.GroupLinear.apply(x, group, i)
        out = patched(mlp, x)
    assert calls == [1] and torch.equal(out, results[0][0])
    assert group.expected_bwd == 0
