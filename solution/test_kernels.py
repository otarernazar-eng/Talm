"""Correctness of our custom Triton kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = F.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 33, 512)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.rmsnorm import triton_rmsnorm

    torch.manual_seed(1)
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn((shape[-1],), device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref = ((x_f * torch.rsqrt(var + eps)) * w.float()).to(dtype)
    ref.backward(grad)
    ref_dx, ref_dw = x.grad.float().clone(), w.grad.float().clone()
    x.grad = w.grad = None

    out = triton_rmsnorm(x, w, eps)
    assert out.dtype == dtype
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)
    torch.testing.assert_close(w.grad.float(), ref_dw, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch_cos", [1, 4])
def test_rope_matches_pytorch(dtype, batch_cos):
    from solution.rope import triton_apply_rotary_pos_emb

    torch.manual_seed(2)
    B, H_q, H_k, S, D = 4, 12, 2, 128, 128
    q = torch.randn((B, H_q, S, D), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((B, H_k, S, D), device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn((batch_cos, S, D), device="cuda", dtype=dtype)
    sin = torch.randn((batch_cos, S, D), device="cuda", dtype=dtype)
    dq_out = torch.randn_like(q)
    dk_out = torch.randn_like(k)

    def rotate_half(t):
        t1 = t[..., : D // 2]
        t2 = t[..., D // 2 :]
        return torch.cat((-t2, t1), dim=-1)

    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    ref_q = (q * cos_u) + (rotate_half(q) * sin_u)
    ref_k = (k * cos_u) + (rotate_half(k) * sin_u)
    (ref_q * dq_out + ref_k.sum(dim=1, keepdim=True) * dk_out[:, :1]).sum().backward()
    ref_dq, ref_dk = q.grad.float().clone(), k.grad.float().clone()
    q.grad = k.grad = None

    out_q, out_k = triton_apply_rotary_pos_emb(q, k, cos, sin)
    (out_q * dq_out + out_k.sum(dim=1, keepdim=True) * dk_out[:, :1]).sum().backward()
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out_q.float(), ref_q.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(out_k.float(), ref_k.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(q.grad.float(), ref_dq, rtol=tol, atol=tol)
    torch.testing.assert_close(k.grad.float(), ref_dk, rtol=tol, atol=tol)


def test_nf4_dequant_matches_bitsandbytes():
    import bitsandbytes as bnb
    from solution.nf4_linear import triton_dequantize_nf4

    torch.manual_seed(3)
    layer = bnb.nn.Linear4bit(
        1536, 1536, bias=False, compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4"
    )
    layer.weight = bnb.nn.Params4bit(
        torch.randn(1536, 1536, dtype=torch.float16),
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
    ).to("cuda")

    ref = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).to(torch.float16)
    out = triton_dequantize_nf4(layer.weight, layer.weight.quant_state, dtype=torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_chunked_ce_matches_pytorch():
    from solution.fused_cross_entropy import triton_chunked_causal_lm_loss

    torch.manual_seed(4)
    B, S, H, V = 2, 64, 128, 4096
    hidden = torch.randn((B, S, H), device="cuda", dtype=torch.float16, requires_grad=True)
    head = torch.nn.Linear(H, V, bias=False, device="cuda", dtype=torch.float16)
    head.weight.requires_grad_(False)
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[:, :16] = -100
    labels[:, -8:] = -100

    # Reference
    logits = head(hidden)
    log_probs = F.log_softmax(logits, dim=-1)
    ref_entropy = -(torch.exp(log_probs) * log_probs).sum(-1)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    ref_loss = F.cross_entropy(
        shift_logits.reshape(-1, V), shift_labels.reshape(-1), ignore_index=-100, reduction="mean"
    )
    ref_loss.backward()
    ref_grad = hidden.grad.float().clone()
    hidden.grad = None

    loss, step = triton_chunked_causal_lm_loss(hidden, head, labels, chunk=32)
    loss.backward()
    torch.testing.assert_close(loss.float(), ref_loss.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(step.token_entropy.float(), ref_entropy.float(), rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(hidden.grad.float(), ref_grad, rtol=5e-3, atol=5e-3)
