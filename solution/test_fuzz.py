"""Randomized (fuzz) tests of every Triton kernel and autograd node of the solution against
plain-PyTorch references, on random shapes, dtypes and strides inside the task's envelope and
beyond it.

Each test draws its cases from a seeded `random.Random` (the same cases on every run; a failure
names its case and parameters) and checks each kernel against the claim its module makes:
`torch.equal` where the module claims bit-exactness, else a bound derived from the arithmetic the
module documents (one rounding to the output dtype, fp32 accumulation, fp16 softmax weights, ...),
computed per element from a float64 reference. SOLUTION_FUZZ_SCALE multiplies the number of
cases (default 1).

Run on a GPU: `python -m pytest solution/test_fuzz.py -q`.
"""

import contextlib
import math
import os
import random

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SCALE = float(os.environ.get("SOLUTION_FUZZ_SCALE", "1"))
F16, BF16, F32 = torch.float16, torch.bfloat16, torch.float32


# --- helpers --------------------------------------------------------------------------------

def _cases(name, n):
    """(index, random.Random of that case) for the n cases of a test (string seeds: the same
    cases on every run, whatever PYTHONHASHSEED is)."""
    for i in range(max(1, round(n * SCALE))):
        yield i, random.Random(f"{name}/{i}")


def _gen(r):
    return torch.Generator(device="cuda").manual_seed(r.getrandbits(62))


def _log_int(r, lo, hi):
    """An integer in [lo, hi], log-uniform: small and large sizes alike."""
    return min(hi, max(lo, int(math.exp(r.uniform(math.log(lo), math.log(hi + 1))))))


@contextlib.contextmanager
def _patched(obj, **values):
    saved = {name: getattr(obj, name) for name in values}
    for name, value in values.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(obj, name, value)


@contextlib.contextmanager
def _exact_fp32_matmul():
    """References use plain fp32 GEMMs (no TF32), whatever another test set."""
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _ulp(x, dtype):
    """The spacing of `dtype` at |x| (x float64), subnormals included."""
    info = torch.finfo(dtype)
    return torch.exp2(torch.floor(torch.log2(x.abs().clamp(min=info.tiny)))) * info.eps


def _unit(dtype):
    """Unit roundoff (half an ulp at 1) of a floating dtype."""
    return torch.finfo(dtype).eps / 2


def _assert_within(ours, ref, tol, what, dtype=None):
    """|ours - ref| <= tol elementwise (float64; NaN fails), naming the first bad element. With
    `dtype`, an infinity where the reference itself rounds to that infinity in `dtype` (an
    overflow of the output dtype) is right too."""
    ours, ref = ours.detach().double(), ref.detach().double()
    assert ours.shape == ref.shape, f"{what}: shape {tuple(ours.shape)} != {tuple(ref.shape)}"
    diff = (ours - ref).abs()
    tol = torch.broadcast_to(torch.as_tensor(tol, dtype=torch.float64, device=diff.device),
                             diff.shape)
    bad = ~(diff <= tol)
    if dtype is not None:
        rounded = ref.to(dtype).double()
        bad &= ~(torch.isinf(rounded) & (ours == rounded))
    if bool(bad.any()):
        flat = int(torch.nonzero(bad.reshape(-1))[0])
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {diff.numel()} elements off, first at "
            f"{tuple(int(v) for v in torch.unravel_index(torch.tensor(flat), diff.shape))}: "
            f"ours {float(ours.reshape(-1)[flat])!r}, reference {float(ref.reshape(-1)[flat])!r}, "
            f"tolerance {float(tol.reshape(-1)[flat]):.3g} (max excess "
            f"{float((diff - tol).nan_to_num(float('inf')).max()):.3g})")


def _assert_equal(a, b, what):
    if a is None or b is None:
        assert a is None and b is None, f"{what}: one side is None"
        return
    assert a.dtype == b.dtype and a.shape == b.shape, \
        f"{what}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
    if not torch.equal(a, b):
        diff = (a.double() - b.double()).abs()
        raise AssertionError(f"{what}: not bit-identical ({int((a != b).sum())} of {a.numel()} "
                             f"elements differ, max |diff| {float(diff.nan_to_num(0).max()):.3g})")


def _strided(t, r):
    """A copy of `t`'s values (new storage, no graph) in a random memory layout: contiguous,
    padded rows, or the last two dimensions swapped in memory."""
    t = t.detach()
    kind = r.choice(["contiguous", "contiguous", "padded", "swapped"])
    if kind == "padded":
        pad = r.choice([1, 3, 8, 16])
        big = torch.empty(*t.shape[:-1], t.shape[-1] + pad, dtype=t.dtype, device=t.device)
        big.fill_(float("nan") if t.is_floating_point() else 0)
        view = big[..., :t.shape[-1]]
        view.copy_(t)
        return view
    if kind == "swapped" and t.dim() >= 2:
        return t.transpose(-1, -2).clone(memory_format=torch.contiguous_format).transpose(-1, -2)
    return t.clone()


def _gemm_tol(a, w, factor=2.0**-7):
    """Bound for the difference of two fp16 GEMMs a @ w^T (each may reduce split-K partials in
    fp16, PyTorch's default): a few fp16 roundings of the magnitudes summed."""
    # fp32 GEMM (a bound needs no more; cuBLAS's fp64 GEMM raised "misaligned address" on odd
    # shapes on the RTX 4080 with CUDA 13)
    with _exact_fp32_matmul():
        return factor * F.linear(a.detach().abs().float(), w.detach().abs().float()).double()


# --- fused head: FusedHeadLossVocab / FusedHeadLoss / fused_head_vocab_kernel / rows kernel --

def _head_reference(hidden, weight, targets, num_items):
    """The base run's head from the fp16 logits of an fp16 GEMM, in float64: per-token loss,
    entropy and first-maximum correctness, the max |logit| per row, the labelled rows whose
    correctness a last-bit logit difference could flip (the target within 2 fp16 ulps of the
    row maximum, next to another logit as close), and the hidden gradient of sum(token loss)
    (float64 softmax, fp32 GEMM with the fp16 weight values)."""
    vocab = weight.shape[0]
    h16 = hidden.detach().to(F16)
    w16 = weight.detach().to(F16)
    w32 = w16.float()
    labelled = targets >= 0
    out = {key: [] for key in ("token_loss", "entropy", "correct", "zmax", "fragile", "grad")}
    rows = max(1, (1 << 22) // vocab)
    cols = torch.arange(vocab, device=hidden.device)
    with _exact_fp32_matmul():
        for s in range(0, hidden.shape[0], rows):
            z16 = h16[s:s + rows] @ w16.t()
            z = z16.double()
            t = targets[s:s + rows]
            lab = t >= 0
            tc = t.clamp(min=0)
            m = z.max(-1).values
            lse = m + torch.log(torch.exp(z - m[:, None]).sum(-1))
            p = torch.exp(z - lse[:, None])
            out["entropy"].append(lse - (p * z).sum(-1))
            zt = z.gather(1, tc[:, None])[:, 0]
            out["token_loss"].append(torch.where(lab, lse - zt, torch.zeros_like(lse)))
            first = torch.where(z == m[:, None], cols, vocab).min(-1).values
            out["correct"].append(lab & (first == t))
            out["zmax"].append(z.abs().max(-1).values)
            near = z >= (m - 8 * _ulp(m, F16))[:, None]
            out["fragile"].append(lab & near.gather(1, tc[:, None])[:, 0] & (near.sum(-1) >= 2))
            p[torch.arange(len(t), device=p.device), tc] -= lab.double()
            p[~lab] = 0
            out["grad"].append((p.float() @ w32).double())
    ref = {key: torch.cat(value) for key, value in out.items()}
    n_lab = int(labelled.sum())
    ref["denom"] = float(num_items) if num_items is not None else float(max(n_lab, 1))
    ref["labelled"] = labelled
    ref["max_w"] = float(w32.abs().max()) if w32.numel() else 0.0
    return ref


def _head_case(r):
    """Random parameters of one head case (sizes, chunking, labels, dtypes, layouts)."""
    big_ok = torch.cuda.mem_get_info()[0] > 3 * 2**30
    c = {"vocab": r.choice([151936, 151936] if big_ok else [4480])
         if r.random() < 0.18 else r.choice([r.randint(1, 4480), _log_int(r, 1, 4480), 1, 2,
                                               4095, 4096, 4097, 4480, 8191])}
    if c["vocab"] == 151936:
        c["size"] = r.choice([8, 64, 77, 256, 1536 if big_ok else 256])
        c["n"] = _log_int(r, 1, 640 if c["size"] == 1536 else 2100)
    else:
        c["size"] = r.choice([1, 3, 8, 16, 40, 64, 100, 128, 255, 256, 1536, r.randint(1, 300)])
        c["n"] = r.choice([1, 2, 7, 8, 9, 63, 64, 65, _log_int(r, 1, 2100), _log_int(r, 1, 2100)])
    vocab, n = c["vocab"], c["n"]
    min_chunk = max(1, vocab // 160)  # at most ~160 vocabulary slices per case
    c["vchunk"] = r.choice([None, 4096, 8192, max(min_chunk, r.randint(1, vocab + 9)),
                            max(min_chunk, 8 * r.randint(1, max(1, vocab // 8 + 1))),
                            max(min_chunk, r.choice([1, 7, 384, 1028, 4100, 20000]))])
    c["tchunk"] = max(max(1, n // 200), r.choice([1, 2, 37, 256, 512, n, n + 5,
                                                   r.randint(1, n + 3)]))
    c["path"] = r.choice(["vocab", "vocab", "vocab", "tokens"])
    c["labels"] = r.choice(["none", "all", "partial", "partial", "partial"])
    c["frac"] = r.uniform(0.05, 0.95)
    c["num_items"] = r.choice([None, "int", "tensor", "cpu"])
    c["style"] = r.choice(["exact", "exact", "random"])
    c["peak"] = r.choice([0, 0, 2, 3])
    c["hidden_dtype"] = r.choice([F16, F32, F32, BF16])
    c["weight_dtype"] = r.choice([F32, F16])
    c["layout"] = r.choice(["contiguous", "contiguous", "contiguous", "padded", "transposed"])
    c["need_grad"] = r.random() < 0.85
    c["grad_scale"] = r.choice([1.0, 3.0, 1024.0, 0.5])
    c["compact"] = r.random() < 0.7
    c["given"] = r.random() < 0.5
    c["unit"] = r.choice([1, 8, 8])
    c["variant"] = r.choice(["compact", "count", "storage", "storage", "unit"])
    return c


def _head_build(r, gen, c):
    """hidden [n, size], weight [vocab, size] (fp16 values), targets [n], num_items.

    style "exact": hidden and weight are {-1, 0, 1} times a power of two per row, so every
    partial sum of a logit is an integer (|int| <= size < 2048) times one power of two, exact in
    fp16 and fp32: every GEMM gives the same logits whatever its tiling or reduction order, and
    the statistics can be compared to fp32 rounding. style "random": Gaussian values (the GEMM
    may round a logit differently from the reference's in the last bit)."""
    n, vocab, size = c["n"], c["vocab"], c["size"]
    dev = "cuda"
    targets = torch.randint(0, vocab, (n,), generator=gen, device=dev)
    # targets at the vocabulary edges and on slice boundaries
    chunk = c["vchunk"] or 8192
    special = torch.tensor(sorted({min(vocab - 1, v) for v in (0, vocab - 1, chunk, chunk - 1,
                                                                4095, 4096, 2 * chunk)}),
                           device=dev)
    pick = torch.rand(n, generator=gen, device=dev) < 0.15
    targets[pick] = special[torch.randint(0, len(special), (int(pick.sum()),), generator=gen,
                                          device=dev)]
    if c["labels"] == "none":
        targets.fill_(-100)
    elif c["labels"] == "partial":
        targets[torch.rand(n, generator=gen, device=dev) > c["frac"]] = -100
    labelled = targets >= 0
    if c["style"] == "exact":
        a = torch.randint(-3, 1 + c["peak"], (n, 1), generator=gen, device=dev).float()
        b = torch.randint(-6, -1, (vocab, 1), generator=gen, device=dev).float()
        hidden = torch.randint(-1, 2, (n, size), generator=gen, device=dev).float() * torch.exp2(a)
        weight = torch.randint(-1, 2, (vocab, size), generator=gen, device=dev).float() \
            * torch.exp2(b)
        sure = labelled & (torch.rand(n, generator=gen, device=dev) < 0.4)
        hidden[sure] = torch.sign(weight[targets[sure]]) * torch.exp2(a[sure])
    else:
        hidden = torch.randn(n, size, generator=gen, device=dev) * r.choice([0.5, 1.0, 2.0])
        weight = (torch.randn(vocab, size, generator=gen, device=dev)
                  * r.choice([0.02, 0.05, 0.2])).half().float()
        sure = labelled & (torch.rand(n, generator=gen, device=dev) < 0.3)
        hidden[sure] = weight[targets[sure]] * r.choice([30.0, 300.0])
    hidden = hidden.to(c["hidden_dtype"])
    weight = weight.to(c["weight_dtype"])
    if c["layout"] == "padded":  # rows not contiguous (stride size + 8): the copied slices
        weight = torch.cat([weight, torch.zeros(vocab, 8, dtype=weight.dtype, device=dev)],
                           1)[:, :size]
    elif c["layout"] == "transposed":  # stride(1) != 1: the token-chunked path
        weight = weight.t().contiguous().t()
    if r.random() < 0.2:  # a strided targets view
        full = torch.full((2 * n,), -7, dtype=torch.int64, device=dev)
        full[::2] = targets
        targets = full[::2]
    n_lab = int(labelled.sum())
    items = max(1, n_lab + r.randint(0, 5))
    num_items = {None: None, "int": items, "tensor": torch.tensor(items, device=dev),
                 "cpu": torch.tensor(items)}[c["num_items"]]
    return hidden, weight, targets, num_items


def _head_run(c, hidden, weight, targets, num_items, **override):
    from solution.fused_head import fused_head_loss

    c = {**c, **override}
    x = hidden.detach().clone().requires_grad_(c["need_grad"])
    if c["path"] == "vocab":
        n_lab = int((targets >= 0).sum()) if c["given"] else None
        kwargs = {"vocab_chunk": c["vchunk"], "n_labelled": n_lab}
    else:
        kwargs = {"token_chunk": c["tchunk"]}
    with torch.set_grad_enabled(c["need_grad"]):
        loss, token_loss, entropy, correct = fused_head_loss(x, weight, targets, num_items,
                                                             **kwargs)
    grad = None
    if c["need_grad"]:
        (loss * c["grad_scale"]).backward()
        grad = x.grad
        assert grad.dtype == hidden.dtype and grad.shape == hidden.shape
    return loss.detach(), token_loss, entropy, correct, grad


def _head_check(c, ref, res, what):
    """Our head against the float64 reference, to the arithmetic fused_head.py documents."""
    loss, token_loss, entropy, correct, grad = res
    zmax = ref["zmax"]
    # fp32 statistics: log-sum-exp and sum(p z) over the row (relative ~1e-6 each)
    tol_loss = 2e-5 + 2e-6 * zmax
    tol_ent = 2e-5 + 1e-5 * zmax
    if c["style"] == "random":  # two fp16 GEMMs may round a logit differently (split-K)
        tol_loss = tol_loss + 8 * _ulp(zmax, F16)
        tol_ent = tol_ent + 16 * _ulp(zmax, F16)
    assert token_loss.dtype == entropy.dtype == F32 and correct.dtype == torch.bool
    _assert_within(token_loss, ref["token_loss"], tol_loss, f"{what}: per-token loss")
    _assert_within(entropy, ref["entropy"], tol_ent, f"{what}: entropy")
    unlabelled = ~ref["labelled"]
    assert bool((token_loss[unlabelled] == 0).all()), f"{what}: loss on an unlabelled row"
    flips = correct != ref["correct"]
    if c["style"] == "exact":
        assert not bool(flips.any()), f"{what}: correctness differs on {int(flips.sum())} rows"
    else:
        assert not bool((flips & ~ref["fragile"]).any()), \
            f"{what}: correctness differs on rows without a near tie"
    denom = ref["denom"]
    ref_loss = ref["token_loss"].sum() / denom
    _assert_within(loss, ref_loss, float(tol_loss.sum()) / denom + 1e-6 * abs(float(ref_loss)),
                   f"{what}: loss")
    if grad is None:
        return
    scale = c["grad_scale"] / denom
    ref_grad = ref["grad"] * scale
    # vocabulary path: fp16 softmax weights (2^-11 relative each, |W| <= max_w) summed by an
    # fp32-output GEMM, the fp16 row gradient (2^-11 of |dH| <= 2 max_w), one rounding to the
    # hidden dtype. Token path: fp16 dlogits into an fp16-OUTPUT GEMM, whose split-K partials
    # cuBLAS may reduce in fp16 (PyTorch's default allow_fp16_reduced_precision_reduction, as
    # in the base run's own head): a few more fp16 roundings of |dH|.
    bound = 2.0**-6 if c["took_tokens"] else 2.0**-8
    tol = scale * ref["max_w"] * bound + _ulp(ref_grad, grad.dtype)
    if c["style"] == "random":
        tol = tol + scale * ref["max_w"] * 2.0**-7 * zmax[:, None]
    _assert_within(grad, ref_grad, tol, f"{what}: hidden gradient")
    assert bool((grad[unlabelled] == 0).all()), f"{what}: gradient on an unlabelled row"
    assert bool(torch.isfinite(grad).all()), f"{what}: non-finite gradient"


def test_fuzz_fused_head():
    """The fused head (vocabulary-sliced by default, token-chunked with `token_chunk=` or a
    weight whose rows are not contiguous) on random sizes (N 1..2100, vocabulary 151936 or
    1..4480, hidden 1..1536), slice and chunk widths (including 1, non-multiples of BLOCK_V and
    of 8, wider than the vocabulary), labelled fractions 0 / partial / 1, num_items None / int /
    tensor (GPU or CPU), fp16/fp32 weights (the fp16 one sliced without a copy), padded and
    transposed weights, fp16/fp32/bf16 hidden states, with and without gradient, against the
    float64 reference; then one claim per case: the compacted gradient rows vs all rows, the
    labelled count given vs counted, the fp16 vs fp32 weight storage (bit-identical), row units
    of 8 vs 1 (to fp32 rounding / one fp16 ulp)."""
    from solution import fused_head

    for i, r in _cases("fused-head", 120):
        c = _head_case(r)
        gen = _gen(r)
        hidden, weight, targets, num_items = _head_build(r, gen, c)
        c["took_tokens"] = c["path"] == "tokens" or weight.stride(1) != 1 \
            or not fused_head.VOCAB_PATH
        what = f"case {i} {c}"
        ref = _head_reference(hidden, weight, targets, num_items)
        with _patched(fused_head, COMPACT=c["compact"], ROW_UNIT=c["unit"]):
            res = _head_run(c, hidden, weight, targets, num_items)
        _head_check(c, ref, res, what)
        variant = c["variant"]
        if variant == "compact" and c["path"] == "vocab" and weight.stride(1) == 1:
            # the accumulator over the labelled rows only: statistics bit-identical, gradient
            # within one fp16 ulp of its largest element (fused_head.py)
            with _patched(fused_head, COMPACT=not c["compact"], ROW_UNIT=c["unit"]):
                other = _head_run(c, hidden, weight, targets, num_items)
            for name, a, b in zip(("loss", "token loss", "entropy", "correct"), res, other):
                _assert_equal(a, b, f"{what}: compact vs full, {name}")
            if res[4] is not None:
                full = other[4].double()
                _assert_within(res[4], full,
                               1e-3 * full.abs() + float(full.abs().max()) * 2.0**-10,
                               f"{what}: compact vs full gradient")
        elif variant == "count" and c["path"] == "vocab":
            with _patched(fused_head, COMPACT=c["compact"], ROW_UNIT=c["unit"]):
                other = _head_run(c, hidden, weight, targets, num_items, given=not c["given"])
            for name, a, b in zip(("loss", "token loss", "entropy", "correct", "grad"), res,
                                  other):
                _assert_equal(a, b, f"{what}: labelled count given vs counted, {name}")
        elif variant == "storage":
            # the same fp16 values stored in the other dtype (fp16_embed.py): bit-identical
            other_w = weight.to(F32 if weight.dtype == F16 else F16)
            with _patched(fused_head, COMPACT=c["compact"], ROW_UNIT=c["unit"]):
                other = _head_run(c, hidden, other_w, targets, num_items)
            for name, a, b in zip(("loss", "token loss", "entropy", "correct", "grad"), res,
                                  other):
                _assert_equal(a, b, f"{what}: fp16 vs fp32 weight storage, {name}")
        elif variant == "unit" and c["path"] == "vocab":
            with _patched(fused_head, COMPACT=c["compact"], ROW_UNIT=9 - c["unit"]):
                other = _head_run(c, hidden, weight, targets, num_items)
            _assert_equal(res[3], other[3], f"{what}: row units 8 vs 1, correct")
            # another in-row summation order: fp32 rounding, which grows with |logits| (the
            # entropy is lse - sum(p z) / s, two numbers close to the largest logit)
            zmax = ref["zmax"]
            row_tol = 2 * (2e-5 + 1e-5 * zmax)
            for name, a, b, tol in (("loss", res[0], other[0], float(row_tol.sum()) / ref["denom"]),
                                    ("token loss", res[1], other[1], row_tol),
                                    ("entropy", res[2], other[2], row_tol)):
                _assert_within(a, b, tol + 1e-6 * b.double().abs(),
                               f"{what}: row units 8 vs 1, {name}")
            if res[4] is not None:
                ref_g = other[4].double()
                _assert_within(res[4], ref_g,
                               1e-3 * ref_g.abs() + float(ref_g.abs().max()) * 2.0**-10,
                               f"{what}: row units 8 vs 1, gradient")


def _structured_logits(r, gen, n, vocab, pad):
    """fp16 logits [n, vocab] (a view with row stride vocab + pad) with ties, constant rows,
    maxima at the edges and on block boundaries, and extreme finite values."""
    dev = "cuda"
    scale = r.choice([0.1, 1.0, 8.0, 100.0])
    z = torch.randn(n, vocab, generator=gen, device=dev) * scale
    if r.random() < 0.4:
        z = (z * 4).round() / 4  # many ties
    kind = torch.randint(0, 8, (n,), generator=gen, device=dev)
    z[kind == 0] = z[kind == 0, :1]  # constant rows: the first index wins every tie
    for col in {0, vocab - 1, min(vocab - 1, 4095), min(vocab - 1, 4096), vocab // 2}:
        rows = kind == 1 + (col % 3)
        z[rows, col] = z[rows].max(-1).values + r.choice([0.0, 0.25, 1000.0])
    z[kind == 5] = z[kind == 5].clamp(-1, 1) * 60000.0
    z = z.clamp(-65504, 65504).half()
    full = torch.full((n, vocab + pad), float("nan"), dtype=F16, device=dev)
    full[:, :vocab] = z
    return full[:, :vocab]


def _stats_reference(z16, targets):
    vocab = z16.shape[1]
    z = z16.double()
    lab = targets >= 0
    tc = targets.clamp(min=0)
    m = z.max(-1).values
    s = torch.exp(z - m[:, None]).sum(-1)
    lse = m + torch.log(s)
    entropy = lse - (torch.exp(z - lse[:, None]) * z).sum(-1)
    loss = torch.where(lab, lse - z.gather(1, tc[:, None])[:, 0], torch.zeros_like(lse))
    first = torch.where(z == m[:, None], torch.arange(vocab, device=z.device), vocab).min(-1).values
    return loss, entropy, lab & (first == targets), z.abs().max(-1).values


def test_fuzz_head_kernels_on_strided_logits():
    """fused_head_vocab_kernel (statistics pass, slices of any width and row stride, sizes in
    units of 8 when they all allow it) and fused_head_rows_kernel on random fp16 logits with
    ties, constant rows and extreme values: per-token loss and entropy to fp32 rounding and the
    FIRST maximum exactly (ties across blocks and slices), against float64 statistics of the
    same logits."""
    from solution import fused_head

    dev = "cuda"
    for i, r in _cases("head-kernels", 200):
        n = _log_int(r, 1, 2100)
        vocab = r.choice([_log_int(r, 1, 20000), r.randint(1, 9000), 8192,
                          151936 if n <= 200 else 4096])
        pad = r.choice([0, 0, 8, 3, 16, 1])
        gen = _gen(r)
        z = _structured_logits(r, gen, n, vocab, pad)
        targets = torch.randint(0, vocab, (n,), generator=gen, device=dev)
        targets[torch.rand(n, generator=gen, device=dev) < 0.3] = -100
        hit = torch.rand(n, generator=gen, device=dev) < 0.3
        targets[hit] = z[hit].float().argmax(-1)
        ref_loss, ref_ent, ref_correct, zmax = _stats_reference(z, targets)
        unit = r.choice([1, 8])
        chunk = r.choice([vocab, 4096, 8192, 8 * r.randint(1, max(1, vocab // 8)),
                          max(1, vocab // 150, r.randint(1, vocab))])
        what = f"case {i}: n={n} vocab={vocab} pad={pad} chunk={chunk} unit={unit}"
        outs = [torch.full((n,), float("nan"), device=dev), torch.full((n,), float("nan"),
                                                                       device=dev),
                torch.full((n,), -1, dtype=torch.int32, device=dev)]
        stats = torch.empty(n, 5, device=dev)
        best = torch.empty(n, dtype=torch.int32, device=dev)
        dummy = torch.zeros(1, 8, device=dev)
        with _patched(fused_head, ROW_UNIT=unit):
            for start in range(0, vocab, chunk):
                cols = min(chunk, vocab - start)
                fused_head.launch_vocab(n, z[:, start:start + cols], targets, stats, best,
                                        dummy, dummy, dummy, *outs, cols, start, z.stride(0), 8,
                                        False, start == 0, start + cols >= vocab, False)
        tol_loss, tol_ent = 2e-5 + 2e-6 * zmax, 2e-5 + 1e-5 * zmax
        _assert_within(outs[0], ref_loss, tol_loss, f"{what}: vocab kernel loss")
        _assert_within(outs[1], ref_ent, tol_ent, f"{what}: vocab kernel entropy")
        _assert_equal(outs[2].bool(), ref_correct, f"{what}: vocab kernel argmax hits")
        if z.stride(0) == vocab or r.random() < 0.5:  # the rows kernel takes any row stride too
            rows_out = [torch.empty(n, device=dev), torch.empty(n, device=dev),
                        torch.empty(n, dtype=torch.int32, device=dev)]
            logits = z.clone() if z.stride(0) == vocab else z.contiguous()
            fused_head.launch_rows(logits, targets.contiguous(), *rows_out, False)
            _assert_within(rows_out[0], ref_loss, tol_loss, f"{what}: rows kernel loss")
            _assert_within(rows_out[1], ref_ent, tol_ent, f"{what}: rows kernel entropy")
            _assert_equal(rows_out[2].bool(), ref_correct, f"{what}: rows kernel argmax hits")


# --- RMSNorm and the residual add folded into it (rmsnorm.py) -------------------------------

def _norm_reference(s, weight, eps):
    """float64 RMSNorm of the values of `s`: (y, xhat, rstd)."""
    xd = s.detach().double()
    rstd = torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps)
    xhat = xd * rstd
    return weight.detach().double() * xhat, xhat, rstd


def _norm_dx(dy, weight, xhat, rstd, out_dtype):
    """float64 input gradient of the norm and its bound for our kernel: xhat is kept in the
    norm's output dtype (one rounding, unit u, entering through xhat * c and through c), the
    rest is fp32 arithmetic."""
    g = dy.detach().double() * weight.detach().double()
    c = (g * xhat).mean(-1, keepdim=True)
    m_abs = (g * xhat).abs().mean(-1, keepdim=True)
    dx = rstd * (g - xhat * c)
    tol = rstd * (2 * _unit(out_dtype) * xhat.abs() * (c.abs() + m_abs)
                  + 2.0**-20 * (g.abs() + (xhat * c).abs()))
    return dx, tol


def _norm_sizes(r, budget=3_000_000):
    h = r.choice([1, 2, 3, 7, 8, 64, 1535, 1536, 2047, 2048, r.randint(1, 2048),
                  _log_int(r, 1, 2048)])
    n = max(1, min(_log_int(r, 1, 3000), budget // h))
    shape = r.choice([(n, h), (1, n, h), (n, 1, h)])
    if n % 4 == 0 and r.random() < 0.3:
        shape = (4, n // 4, h)
    return shape


def test_fuzz_rmsnorm():
    """RMSNorm forward/backward (one kernel, runtime `backward` flag) on hidden sizes 1..2048,
    1..3000 rows, fp32/fp16/bf16 inputs and outputs, fp32 or 16-bit weights, padded / swapped /
    3-D inputs and gradients, eps from 1e-12 to 0.5, zero rows: y within one rounding to the
    output dtype (plus fp32 slack) of the float64 norm, dx within the bound of an xhat kept in the
    output dtype."""
    from solution.rmsnorm import RMSNorm

    for i, r in _cases("rmsnorm", 200):
        shape = _norm_sizes(r)
        x_dtype = r.choice([F32, F32, F16, BF16])
        out_dtype = r.choice([F16, BF16, F32]) if x_dtype == F32 else r.choice([x_dtype, F32])
        w_dtype = r.choice([F32, x_dtype])
        eps = r.choice([1e-6, 1e-5, 1e-12, 1e-3, 0.5])
        gen = _gen(r)
        x = (torch.randn(shape, generator=gen, device="cuda")
             * r.choice([1e-3, 1.0, 30.0, 1000.0])).to(x_dtype)
        if r.random() < 0.15:
            x[..., 0, :] = 0  # a zero row: rstd = 1 / sqrt(eps)
        x = _strided(x, r).requires_grad_(True)
        weight = (torch.rand(shape[-1], generator=gen, device="cuda") * 4 - 2).to(w_dtype)
        dy = _strided(torch.randn(shape, generator=gen, device="cuda").to(out_dtype), r)
        what = f"case {i}: shape={shape} x={x_dtype} out={out_dtype} w={w_dtype} eps={eps}"
        y = RMSNorm.apply(x, weight, eps, out_dtype)
        y.backward(dy)
        assert y.dtype == out_dtype and y.shape == x.shape and x.grad.dtype == x_dtype, what
        y_ref, xhat, rstd = _norm_reference(x, weight, eps)
        _assert_within(y, y_ref, _ulp(y_ref, out_dtype) + 2.0**-19 * y_ref.abs(), f"{what}: y",
                       out_dtype)
        dx_ref, tol = _norm_dx(dy, weight, xhat, rstd, out_dtype)
        _assert_within(x.grad, dx_ref, tol + _ulp(dx_ref, x_dtype), f"{what}: dx", x_dtype)


def test_rmsnorm_rejects_rows_wider_than_its_block():
    """One program normalizes a whole row of at most BLOCK columns: a wider row must raise
    (it used to normalize only the first BLOCK columns and leave the rest of y unwritten), and
    the Qwen2 patch keeps such norms on transformers' path."""
    from solution.rmsnorm import BLOCK, AddRMSNorm, _norm_applies, rmsnorm

    x = torch.randn(3, BLOCK + 8, device="cuda")
    w = torch.ones(BLOCK + 8, device="cuda")
    with pytest.raises(ValueError):
        rmsnorm(x, w, 1e-6)
    with pytest.raises(ValueError):
        AddRMSNorm.apply(x, x.half(), w, 1e-6, F16)

    class Norm:
        weight = w

    assert not _norm_applies(Norm, x)


def test_fuzz_add_rmsnorm():
    """AddRMSNorm (the residual add inside the norm's kernel): with an fp32 residual, the sum,
    the norm output and both input gradients bit-identical to the separate fp32 add followed by
    our RMSNorm (rmsnorm.py's claim), for gradients from the sum, from the norm or from both;
    for every dtype within the float64 bounds of test_fuzz_rmsnorm."""
    from solution.rmsnorm import AddRMSNorm, RMSNorm

    for i, r in _cases("add-rmsnorm", 150):
        shape = _norm_sizes(r, budget=2_000_000)
        x_dtype = r.choice([F32, F32, F32, F16, BF16])
        out_dtype = r.choice([F16, BF16, F32]) if x_dtype == F32 else x_dtype
        b_dtype = out_dtype if r.random() < 0.7 else r.choice([F32, F16, BF16])
        eps = r.choice([1e-6, 1e-5, 1e-3])
        uses = r.choice(["both", "both", "sum", "norm"])
        gen = _gen(r)
        scale = r.choice([0.01, 1.0, 30.0])
        x0 = (torch.randn(shape, generator=gen, device="cuda") * scale).to(x_dtype)
        b0 = (torch.randn(shape, generator=gen, device="cuda") * scale).to(b_dtype)
        weight = torch.rand(shape[-1], generator=gen, device="cuda") + 0.5
        g_s = _strided(torch.randn(shape, generator=gen, device="cuda").to(x_dtype), r)
        g_y = _strided(torch.randn(shape, generator=gen, device="cuda").to(out_dtype), r)
        layout_x, layout_b = random.Random(r.random()), random.Random(r.random())
        what = (f"case {i}: shape={shape} x={x_dtype} out={out_dtype} branch={b_dtype} "
                f"uses={uses}")

        def run(fn):
            x = _strided(x0, random.Random(layout_x.random())).requires_grad_(True)
            b = _strided(b0, random.Random(layout_b.random())).requires_grad_(True)
            total, y = fn(x, b)
            loss = 0
            if uses != "norm":
                loss = loss + (total * g_s).float().sum()
            if uses != "sum":
                loss = loss + (y * g_y).float().sum()
            loss.backward()
            return total.detach(), y.detach(), x.grad, b.grad

        ours = run(lambda x, b: AddRMSNorm.apply(x, b, weight, eps, out_dtype))
        assert ours[0].dtype == x_dtype and ours[1].dtype == out_dtype, what
        assert ours[2].dtype == x_dtype and ours[3].dtype == b_dtype, what
        if x_dtype == F32:
            def separate(x, b):
                total = x + b
                return total, RMSNorm.apply(total, weight, eps, out_dtype)

            theirs = run(separate)
            for name, a, t in zip(("sum", "norm output", "dx", "dbranch"), ours, theirs):
                _assert_equal(a, t, f"{what}: {name} vs separate add + RMSNorm")
        s32 = x0.float() + b0.float()  # the kernel's fp32 sum (rounded to x's dtype in `total`)
        _assert_equal(ours[0], s32.to(x_dtype), f"{what}: sum")
        y_ref, xhat, rstd = _norm_reference(s32, weight, eps)
        _assert_within(ours[1], y_ref, _ulp(y_ref, out_dtype) + 2.0**-19 * y_ref.abs(),
                       f"{what}: norm output")
        dx_ref = torch.zeros(shape, dtype=torch.float64, device="cuda")
        tol = torch.zeros_like(dx_ref)
        if uses != "sum":
            dx_ref, tol = _norm_dx(g_y, weight, xhat, rstd, out_dtype)
        if uses != "norm":
            dx_ref = dx_ref + g_s.double()
        _assert_within(ours[2], dx_ref, tol + _ulp(dx_ref, x_dtype), f"{what}: dx", x_dtype)
        _assert_within(ours[3], dx_ref, tol + _ulp(dx_ref, x_dtype) + _ulp(dx_ref, b_dtype),
                       f"{what}: dbranch", b_dtype)


def test_fuzz_add_norm_across_layers():
    """Small random Qwen2 models (1-4 layers, head sizes 16..128, 1-8 heads, grouped KV) with
    every residual add folded into the norm that reads it, across layers: the same output bit
    for bit as transformers' layer forward with separate adds, and the same input gradient (bit
    for bit where the attention backward is deterministic, else within its own spread)."""
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution import rmsnorm

    rmsnorm.patch_qwen2_rmsnorm()
    rmsnorm.patch_qwen2_add_norm()
    cls = modeling_qwen2.Qwen2DecoderLayer
    for i, r in _cases("add-norm-layers", 12):
        heads = r.choice([1, 2, 4, 8])
        kv = r.choice([d for d in (1, 2, 4, 8) if heads % d == 0])
        head_dim = r.choice([16, 32, 64, 128, 48, 80])
        layers, seq = r.randint(1, 4), _log_int(r, 1, 96)
        config = Qwen2Config(hidden_size=heads * head_dim, intermediate_size=r.choice([64, 256]),
                             num_attention_heads=heads, num_key_value_heads=kv,
                             num_hidden_layers=layers, vocab_size=64, max_position_embeddings=256,
                             attn_implementation="sdpa")
        torch.manual_seed(r.getrandbits(31))
        model = modeling_qwen2.Qwen2Model(config).cuda().train()
        for module in model.modules():
            if isinstance(module, modeling_qwen2.Qwen2RMSNorm):
                module.weight.data.uniform_(0.5, 1.5)
        model.requires_grad_(False)
        embeds = torch.randn(1, seq, config.hidden_size, device="cuda")
        grad = torch.randn(1, seq, config.hidden_size, device="cuda", dtype=F16)
        what = f"case {i}: heads={heads} kv={kv} head_dim={head_dim} layers={layers} seq={seq}"

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
                with torch.autocast("cuda", dtype=F16):
                    out = model(inputs_embeds=inp).last_hidden_state
                (out.float() * grad.float()).sum().backward()
            finally:
                cls.forward = saved
            return out.detach(), inp.grad

        ref_out, ref_dx = run(rmsnorm.LAYER_FORWARDS["original"], False)
        _, again_dx = run(rmsnorm.LAYER_FORWARDS["original"], False)
        out, dx = run(rmsnorm.LAYER_FORWARDS["add_norm"], True)
        _assert_equal(out, ref_out, f"{what}: output")
        spread = float((ref_dx - again_dx).abs().max())
        if spread == 0.0:
            _assert_equal(dx, ref_dx, f"{what}: input gradient")
        else:
            _assert_within(dx, ref_dx, 2e-3 * ref_dx.abs() + 2 * spread, f"{what}: input gradient")


# --- RoPE (rope.py) ---------------------------------------------------------------------------

def _rope_ref(x, cos, sin, backward):
    """float64 rotation of x [1, H, N, D] by cos/sin [1, N, D] (backward: the transpose), and the
    magnitude of its terms (for the fp32 slack)."""
    half = x.shape[-1] // 2
    xd, c, s = x.detach().double(), cos.double()[:, None], sin.double()[:, None]
    x1, x2 = xd[..., :half], xd[..., half:]
    c1, c2, s1, s2 = c[..., :half], c[..., half:], s[..., :half], s[..., half:]
    if backward:
        y = torch.cat((x1 * c1 + x2 * s2, x2 * c2 - x1 * s1), -1)
        mag = torch.cat(((x1 * c1).abs() + (x2 * s2).abs(), (x2 * c2).abs() + (x1 * s1).abs()), -1)
    else:
        y = torch.cat((x1 * c1 - x2 * s1, x2 * c2 + x1 * s2), -1)
        mag = torch.cat(((x1 * c1).abs() + (x2 * s1).abs(), (x2 * c2).abs() + (x1 * s2).abs()), -1)
    return y, mag


def _rope_input(r, gen, heads, n, dim, other_heads):
    """[1, heads, n, dim] fp16 requiring grad, in one of the layouts q/k come in: a view of the
    projection output (the task's), head-major, a slice of a fused qkv output (strides that may
    not be whole rows), or rows padded in memory."""
    kind = r.choice(["proj", "proj", "head", "fused", "padded"])
    dev = "cuda"
    if kind == "proj":
        base = torch.randn(1, n, heads * dim, generator=gen, device=dev).half().requires_grad_()
        return base.view(1, n, heads, dim).transpose(1, 2), kind
    if kind == "head":
        return torch.randn(1, heads, n, dim, generator=gen, device=dev).half().requires_grad_(), kind
    if kind == "fused":
        extra = r.choice([0, 8, dim, 3])
        width = (heads + other_heads) * dim + extra
        base = torch.randn(1, n, width, generator=gen, device=dev).half().requires_grad_()
        start = r.choice([0, other_heads * dim])
        return base[..., start:start + heads * dim].view(1, n, heads, dim).transpose(1, 2), kind
    pad = r.choice([2, 8, 16])
    base = torch.randn(1, heads, n, dim + pad, generator=gen, device=dev).half().requires_grad_()
    return base[..., :dim], kind


def test_fuzz_rope():
    """RoPE forward and backward (one kernel) on every even head size 2..256 (powers of two and
    not), 1-16 query and key heads, 1..3000 tokens, the layouts q/k come in, real rotary tables
    at random positions or arbitrary cos/sin, gradients token-major / head-major / padded /
    broadcast: within one fp16 rounding (plus fp32 slack) of the float64 rotation both ways;
    and bit-identical across the token-major / head-major gradient and row / element strides
    (rope.py's claim)."""
    from solution import rope
    from solution.rope import RoPE

    dev = "cuda"
    for i, r in _cases("rope", 200):
        dim = r.choice([2, 4, 6, 8, 10, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128, 128, 160, 192,
                        256, 2 * r.randint(1, 128)])
        hq, hk = r.randint(1, 16), r.randint(1, 16)
        n = max(1, min(_log_int(r, 1, 3000), 8_000_000 // ((hq + hk) * dim)))
        gen = _gen(r)
        q, q_kind = _rope_input(r, gen, hq, n, dim, hk)
        k, k_kind = _rope_input(r, gen, hk, n, dim, hq)
        if r.random() < 0.6:  # transformers' table at per-row positions
            pos = torch.cat([torch.arange(length, device=dev) for length in
                             _split_lengths(r, n)])[None].float()
            inv = 1.0 / (r.choice([1e4, 1e6]) ** (torch.arange(0, dim, 2, device=dev).float()
                                                  / dim))
            emb = torch.cat(((pos[..., None] * inv),) * 2, -1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos = torch.randn(1, n, dim, generator=gen, device=dev)
            sin = torch.randn(1, n, dim, generator=gen, device=dev)
        if r.random() < 0.3:
            cos = torch.cat((cos, cos), -1)[..., :dim]  # strided tables (made contiguous)
        grads = []
        for heads in (hq, hk):
            kind = r.choice(["token", "head", "padded", "broadcast"])
            if kind == "token":
                g = torch.randn(1, n, heads, dim, generator=gen, device=dev).half().transpose(1, 2)
            elif kind == "head":
                g = torch.randn(1, heads, n, dim, generator=gen, device=dev).half()
            elif kind == "padded":
                g = torch.randn(1, heads, n, dim + 8, generator=gen, device=dev).half()[..., :dim]
            else:
                g = torch.randn(1, 1, 1, dim, generator=gen, device=dev).half().expand(
                    1, heads, n, dim)
            grads.append(g)
        what = f"case {i}: dim={dim} hq={hq} hk={hk} n={n} q={q_kind} k={k_kind}"
        combos = [(rope.TOKEN_MAJOR_GRAD, rope.ROW_STRIDES)]
        combos.append(r.choice([(a, b) for a in (True, False) for b in (True, False)
                                if (a, b) != combos[0]]))
        results = []
        for token_major, rows in combos:
            with _patched(rope, TOKEN_MAJOR_GRAD=token_major, ROW_STRIDES=rows):
                oq, ok = RoPE.apply(q, k, cos, sin)
                dq, dk = torch.autograd.grad((oq, ok), (q, k), grads)
            results.append((oq.detach(), ok.detach(), dq, dk))
        oq, ok, dq, dk = results[0]
        assert oq.dtype == ok.dtype == F16 and oq.shape == q.shape and ok.shape == k.shape, what
        for name, ours, x, backward in (("q", oq, q, False), ("k", ok, k, False),
                                        ("dq", dq, grads[0], True), ("dk", dk, grads[1], True)):
            ref, mag = _rope_ref(x, cos, sin, backward)
            _assert_within(ours, ref, _ulp(ref, F16) + 2.0**-21 * mag, f"{what}: {name}")
        for name, a, b in zip(("q", "k", "dq", "dk"), results[0], results[1]):
            _assert_equal(a, b, f"{what}: {name}, {combos[0]} vs {combos[1]} "
                                "(token-major grads, row strides)")


def _split_lengths(r, n):
    """n tokens as 1-8 rows of random lengths (the unpadded batch)."""
    cuts = sorted(r.sample(range(1, n), min(n - 1, r.randint(0, 7)))) if n > 1 else []
    bounds = [0] + cuts + [n]
    return [b - a for a, b in zip(bounds, bounds[1:])]


def test_fuzz_rope_patch_dispatch():
    """The patched apply_rotary_pos_emb: the fused kernel exactly when `_applies` says so
    (within one fp16 rounding of the float64 rotation), transformers' own function bit for bit
    otherwise (fp32 or bf16 q/k, batches, a head size above 256, unsqueeze_dim 2, ...)."""
    from transformers.models.qwen2 import modeling_qwen2

    from solution.rope import _applies, patch_qwen2_rope

    patch_qwen2_rope()
    patched = modeling_qwen2.apply_rotary_pos_emb
    original = patched._original
    dev = "cuda"
    for i, r in _cases("rope-dispatch", 60):
        dim = r.choice([8, 64, 96, 128, 258, 2 * r.randint(1, 140)])
        hq, hk, n = r.randint(1, 8), r.randint(1, 4), _log_int(r, 1, 600)
        batch = r.choice([1, 1, 1, 2])
        dtype = r.choice([F16, F16, F32, BF16])
        gen = _gen(r)
        q = torch.randn(batch, n, hq, dim, generator=gen, device=dev).to(dtype).transpose(1, 2)
        k = torch.randn(batch, n, hk, dim, generator=gen, device=dev).to(dtype).transpose(1, 2)
        cos = torch.randn(batch, n, dim, generator=gen, device=dev)
        sin = torch.randn(batch, n, dim, generator=gen, device=dev)
        unsqueeze = 1 if r.random() < 0.85 else 2
        if unsqueeze == 2:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        what = f"case {i}: dim={dim} hq={hq} hk={hk} n={n} batch={batch} {dtype} u={unsqueeze}"
        oq, ok = patched(q, k, cos, sin, unsqueeze_dim=unsqueeze)
        if _applies(q, k, cos, sin, unsqueeze):
            for name, ours, x in (("q", oq, q), ("k", ok, k)):
                ref, mag = _rope_ref(x, cos, sin, False)
                _assert_within(ours, ref, _ulp(ref, F16) + 2.0**-21 * mag, f"{what}: {name}")
        else:
            rq, rk = original(q, k, cos, sin, unsqueeze_dim=unsqueeze)
            _assert_equal(oq, rq, f"{what}: fallback q")
            _assert_equal(ok, rk, f"{what}: fallback k")


# --- NF4 dequantization: per projection (nf4_dequant.py), per layer (qgroup.py), qlinear.py ---

def _nf4_manual(packed, absmax, blocksize, numel, dtype=F16):
    """The NF4 format in PyTorch: high nibble first, level times the block's fp32 absmax, one
    rounding to `dtype` (what bitsandbytes' kernel computes)."""
    from solution.nf4_dequant import NF4_TABLE

    codes = packed.reshape(-1).to(torch.int64)
    levels = torch.tensor(NF4_TABLE, device=packed.device)
    vals = torch.stack([levels[codes >> 4], levels[codes & 15]], 1).reshape(-1)[:numel]
    return (vals * absmax.float().repeat_interleave(blocksize)[:numel]).to(dtype)


def _absmax32(state):
    import bitsandbytes.functional as bnbf

    if state.nested:
        return bnbf.dequantize_blockwise(state.absmax, state.state2) + state.offset
    return state.absmax


def _nf4_weight(r, gen, shape, dtype=F16):
    w = torch.randn(shape, generator=gen, device="cuda") * r.choice([0.01, 0.05, 1.0])
    flat = w.view(-1)
    if r.random() < 0.3:
        flat[:64] = 0  # an all-zero block (absmax 0)
    if r.random() < 0.3:
        flat[torch.randint(0, flat.numel(), (8,), generator=gen, device="cuda")] *= 50
    return w.to(dtype)


def test_fuzz_nf4_dequant_kernel():
    """The per-projection NF4 kernel (the fallback for unpackable layers) on random shapes and
    every bitsandbytes block size, with and without double quantization, through the patched
    bitsandbytes dispatch (odd sizes and non-fp16 outputs take bitsandbytes' own kernel): bit for
    bit the NF4 format (nf4_dequant.py's claim) and bitsandbytes' dequantize_4bit."""
    import bitsandbytes.functional as bnbf

    from solution.nf4_dequant import nf4_dequantize, patch_bnb_dequant

    assert patch_bnb_dequant() is True
    for i, r in _cases("nf4", 100):
        blocksize = r.choice([64, 64, 128, 256, 512, 1024, 2048, 4096])
        rows = _log_int(r, 1, 4096)
        cols = max(1, min(_log_int(r, 1, 4096), 4_000_000 // rows))
        dtype = r.choice([F16, F16, F16, BF16])
        nested = r.random() < 0.7
        gen = _gen(r)
        w = _nf4_weight(r, gen, (rows, cols), dtype)
        packed, state = bnbf.quantize_4bit(w, blocksize=blocksize, quant_type="nf4",
                                           compress_statistics=nested)
        absmax = _absmax32(state)
        what = f"case {i}: shape=({rows}, {cols}) blocksize={blocksize} {dtype} nested={nested}"
        ref = _nf4_manual(packed, absmax, blocksize, rows * cols, dtype).view(rows, cols)
        _assert_equal(bnbf.dequantize_4bit(packed, state), ref, f"{what}: patched dispatch")
        if dtype == F16 and (rows * cols) % 2 == 0:
            out = torch.full((rows, cols), float("nan"), dtype=F16, device="cuda")
            nf4_dequantize(packed, absmax, blocksize, out)
            _assert_equal(out, ref, f"{what}: kernel")


def _fake_proj(w16, blocksize, bias):
    """What LayerGroup reads of a bitsandbytes Linear4bit: weight (packed, with quant_state)
    and bias."""
    import bitsandbytes.functional as bnbf
    from types import SimpleNamespace

    packed, state = bnbf.quantize_4bit(w16, blocksize=blocksize, quant_type="nf4",
                                       compress_statistics=True)
    packed.quant_state = state
    return SimpleNamespace(weight=packed, bias=bias)


def _group_shapes(r, qblock, conforming):
    """Seven (out, in) shapes; conforming: every size a multiple of the absmax double block."""
    double = qblock * 256
    shapes = []
    for _ in range(7):
        n_in = r.choice([64, 128, 256, 512, 1536, 192, 320])
        step = double // math.gcd(n_in, double)
        shapes.append((step * r.randint(1, max(1, 1024 // step)), n_in))
    shapes[5] = shapes[4]  # up_proj has gate_proj's shape (the save-up MLP node needs it)
    if not conforming:  # breaks the multiple (sizes stay multiples of 8: bitsandbytes' own
        # Linear4bit forward faulted with "misaligned address" on out_features 515, in 128,
        # fp16, 234 tokens on the RTX 4080 / CUDA 13, without any patch of ours)
        idx = r.randrange(7)
        out, n_in = shapes[idx]
        shapes[idx] = (out + r.choice([8, 16, 24, 40]), n_in)
    return shapes


def test_fuzz_group_dequant():
    """The per-layer NF4 kernel (qgroup.py) on seven random projection shapes per layer and
    absmax block sizes 64/128/256: the whole-layer launch and the per-projection slice launches
    bit for bit the NF4 format and bitsandbytes' dequantize_4bit (the claimed torch.equal);
    GroupLinear forward/backward over random split depths and inputs that need no gradient,
    with the per-pass bookkeeping back to rest after each step, and the MLP node that keeps up
    (mlp_save_up.py) bit-identical to the default MLP on the same group; and a layer whose sizes
    are not a multiple of the double block raises ValueError before touching its parameters
    (the per-projection fallback then serves it)."""
    import bitsandbytes.functional as bnbf

    from solution import qgroup
    from solution.mlp_save_up import GATE, UP, SwiGLUSaveUp
    from solution.qgroup import GroupLinear, LayerGroup
    from solution.swiglu_inplace import SwiGLUInPlaceGrad

    dev = "cuda"
    for i, r in _cases("group-dequant", 50):
        qblock = r.choice([64, 64, 128, 256])
        conforming = r.random() < 0.8
        shapes = _group_shapes(r, qblock, conforming)
        gen = _gen(r)
        modules = [_fake_proj(_nf4_weight(r, gen, shape), qblock,
                              torch.randn(shape[0], generator=gen, device=dev).half()
                              if r.random() < 0.3 else None) for shape in shapes]
        refs = [_nf4_manual(m.weight, _absmax32(m.weight.quant_state), qblock,
                            s[0] * s[1]).view(s) for m, s in zip(modules, shapes)]
        what = f"case {i}: qblock={qblock} shapes={shapes}"
        if not conforming:
            data = [(m.weight.data_ptr(), m.weight.quant_state.absmax.data_ptr())
                    for m in modules]
            with pytest.raises(ValueError):
                LayerGroup(modules)
            assert data == [(m.weight.data_ptr(), m.weight.quant_state.absmax.data_ptr())
                            for m in modules], f"{what}: parameters repacked before the error"
            continue
        group = LayerGroup(modules)
        full = group.dequantize_all()
        for idx, (m, ref) in enumerate(zip(modules, refs)):
            ours = full[group.elem_bounds[idx]:group.elem_bounds[idx + 1]].view(shapes[idx])
            _assert_equal(ours, ref, f"{what}: whole-layer launch, projection {idx}")
            again = bnbf.dequantize_4bit(m.weight.data, m.weight.quant_state).half()
            _assert_equal(again, ref, f"{what}: the packed parameter {idx} for bitsandbytes")
            if group.splittable:
                _assert_equal(group.dequantize_one(idx), ref, f"{what}: slice launch {idx}")
        split = {"fwd": r.randint(0, 2), "bwd": r.randint(0, 3)}
        group.depth = r.choice([None, 0, 1, 2, 5])
        with _patched(qgroup, SPLIT_LAYERS=split):
            for step in range(2):
                inputs = []
                for idx, (out_f, in_f) in enumerate(shapes):
                    x = torch.randn(r.choice([(1, 37), (3, 11), (1, 1)]) + (in_f,),
                                    generator=gen, device=dev).half()
                    inputs.append(x.requires_grad_(r.random() < 0.8))
                outs = []
                for idx, x in enumerate(inputs):
                    if x.requires_grad:
                        group.expected_bwd += 1  # as the patched Linear4bit.forward does
                    outs.append(GroupLinear.apply(x, group, idx))
                for idx, (o, x) in enumerate(zip(outs, inputs)):
                    b = modules[idx].bias
                    ref_o = F.linear(x.detach(), refs[idx], b)
                    _assert_within(o, ref_o.double(), _gemm_tol(x, refs[idx])
                                   + _ulp(ref_o.double(), F16),
                                   f"{what}: GroupLinear output {idx} step {step}")
                live = [(o, x) for o, x in zip(outs, inputs) if x.requires_grad]
                if live:
                    sum((o.float() ** 2).mean() for o, _ in live).backward()
                for idx, x in enumerate(inputs):
                    if x.requires_grad:
                        xr = x.detach().clone().requires_grad_(True)
                        o_ref = F.linear(xr, refs[idx], modules[idx].bias)
                        (o_ref.float() ** 2).mean().backward()
                        g_out = (2 * o_ref.detach().float() / o_ref.numel()).half()
                        _assert_within(x.grad, xr.grad.double(),
                                       _gemm_tol(g_out, refs[idx].t())
                                       + _ulp(xr.grad.double(), F16),
                                       f"{what}: GroupLinear input gradient {idx} step {step}")
                assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None, \
                    f"{what}: a dequantized buffer outlived its pass (step {step})"
                assert group.expected_bwd == 0 and group.split_left == 0, \
                    f"{what}: bookkeeping not at rest after step {step}"
            # mlp_save_up.SwiGLUSaveUp against the default MLP path on the same group
            n_in = shapes[GATE][1]
            if shapes[UP][1] == n_in and shapes[UP][0] == shapes[GATE][0]:
                x0 = torch.randn(r.choice([(1, 29), (2, 9)]) + (n_in,), generator=gen,
                                 device=dev).half()
                grad = torch.randn(x0.shape[:-1] + (shapes[GATE][0],), generator=gen,
                                   device=dev).half()
                res = []
                for kind in ("default", "save_up"):
                    x = x0.clone().requires_grad_(True)
                    if kind == "default":
                        group.expected_bwd += 2
                        act = SwiGLUInPlaceGrad.apply(GroupLinear.apply(x, group, GATE),
                                                      GroupLinear.apply(x, group, UP))
                    else:
                        group.expected_bwd += 2
                        act = SwiGLUSaveUp.apply(x, group)
                    act.backward(grad)
                    res.append((act.detach(), x.grad))
                _assert_equal(res[0][0], res[1][0], f"{what}: save-up MLP activation")
                _assert_equal(res[0][1], res[1][1], f"{what}: save-up MLP input gradient")


def test_fuzz_unpackable_layers_take_the_per_projection_path():
    """Real bitsandbytes layers whose sizes the per-layer kernel cannot tile: setup_layer leaves
    them unpacked, installs the per-projection kernel (FALLBACK_DEQUANT), and their outputs are
    those of the dequantized weights."""
    import bitsandbytes as bnb
    import bitsandbytes.backends.cuda.ops as ops

    from solution import qgroup
    from solution.qgroup import PROJECTIONS, setup_layer

    for i, r in _cases("unpackable", 6):
        shapes = _group_shapes(r, 64, conforming=False)
        torch.manual_seed(r.getrandbits(31))
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        mods = []
        for path, (out_f, in_f) in zip(PROJECTIONS, shapes):
            lin = bnb.nn.Linear4bit(in_f, out_f, bias=False, quant_type="nf4",
                                    compress_statistics=True, compute_dtype=F16)
            lin.weight = bnb.nn.Params4bit((torch.randn(out_f, in_f) * 0.05).half(),
                                           requires_grad=False, quant_type="nf4",
                                           compress_statistics=True)
            lin = lin.cuda().requires_grad_(False)
            owner, name = path.split(".")
            setattr(getattr(layer, owner), name, lin)
            mods.append(lin)
        with _patched(qgroup, FALLBACK_DEQUANT=True):
            setup_layer(layer)
        assert not any(hasattr(m, "_solution_group") for m in mods), f"case {i}: packed"
        assert getattr(ops._dequantize_4bit_impl, "_solution_patched", False), f"case {i}"
        for m, (out_f, in_f) in zip(mods, shapes):
            state = m.weight.quant_state
            w = _nf4_manual(m.weight.data, _absmax32(state), state.blocksize, out_f * in_f)
            x = torch.randn(1, r.randint(1, 600), in_f, device="cuda").half()
            ref = F.linear(x, w.view(out_f, in_f)).double()
            _assert_within(m(x), ref, _gemm_tol(x, w.view(out_f, in_f)) + _ulp(ref, F16),
                           f"case {i}: output {shapes}")


def test_fuzz_qlinear_dequant():
    """The opt-in fused 4-bit linear's dequantization (qlinear.py): within one fp16 ulp of the
    NF4 format on a small fraction of the weights at most (the fused multiply-add of the scale),
    as claimed, on random shapes and block sizes."""
    import bitsandbytes.functional as bnbf

    from solution.qlinear import dequantize_weight

    for i, r in _cases("qlinear", 40):
        blocksize = r.choice([64, 128, 256, 512])
        rows = _log_int(r, 1, 4096)
        cols = 2 * max(1, min(_log_int(r, 1, 2048), 2_000_000 // rows))
        gen = _gen(r)
        w = _nf4_weight(r, gen, (rows, cols))
        packed, state = bnbf.quantize_4bit(w, blocksize=blocksize, quant_type="nf4",
                                           compress_statistics=True)
        ref = _nf4_manual(packed, _absmax32(state), blocksize, rows * cols).view(rows, cols)
        ours = dequantize_weight(packed, state)
        what = f"case {i}: shape=({rows}, {cols}) blocksize={blocksize}"
        _assert_within(ours, ref.double(), _ulp(ref.double(), F16), what)
        assert float((ours != ref).float().mean()) < 1e-3, f"{what}: too many 1-ulp differences"


# --- per-row causal attention (unpad.py) ----------------------------------------------------

def _rows_reference(q, k, v, bounds, scale):
    """Per-row causal SDPA with the memory-efficient kernel on repeat_kv copies: [1, N, Hq, D]."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from solution.unpad import _repeat_kv

    rep = q.shape[1] // k.shape[1]
    k_rep, v_rep = _repeat_kv(k, rep), _repeat_kv(v, rep)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        outs = [F.scaled_dot_product_attention(q[:, :, s:e], k_rep[:, :, s:e], v_rep[:, :, s:e],
                                               is_causal=True, scale=scale)
                for s, e in bounds if e > s]
    return torch.cat(outs, 2).transpose(1, 2)


def test_fuzz_rows_attention():
    """RowsAttention (one launch per row) and VarlenRowsAttention (all rows in one launch per KV
    head) on 1-8 rows of 1..512 tokens (empty rows too), 1-6 query heads per KV head, 1-4 KV
    heads, head sizes 8..256, fp16/bf16, head-major or token-major inputs, against per-row SDPA
    on the memory-efficient kernel with repeated K/V (unpad.py's claim): the forward and dK/dV
    bit for bit (within the run-to-run spread where that backward is not deterministic), dQ
    within one ulp of the 16-bit dtype (its key-block accumulation order follows the launch
    layout; this test found the earlier "bit-identical" claim too strong for dQ)."""
    from solution import unpad
    from solution.unpad import RowsAttention, VarlenRowsAttention, rows_context

    dev = "cuda"
    for i, r in _cases("rows-attention", 100):
        lengths = [r.choice([1, 2, 3, 512, _log_int(r, 1, 512), r.randint(1, 512)])
                   for _ in range(r.randint(1, 8))]
        if r.random() < 0.2:
            lengths.insert(r.randint(0, len(lengths)), 0)
        rep, hkv = r.randint(1, 6), r.randint(1, 4)
        hq = rep * hkv
        d = r.choice([8, 16, 32, 64, 64, 128, 128, 24, 40, 80, 96, 160, 256])
        dtype = r.choice([F16, F16, BF16])
        if dtype == BF16 and torch.cuda.get_device_capability()[0] < 8:
            # below sm80 (the T4) no SDPA kernel takes bf16 (the reference would raise "No
            # available kernel"), and the patch only sends fp16 there: drawn, then fp16, so the
            # other parameters of every case stay those of newer GPUs
            dtype = F16
        scale = r.choice([d ** -0.5, d ** -0.5, 0.05, 1.0])
        bounds, start = [], 0
        for length in lengths:
            bounds.append((start, start + length))
            start += length
        n = start
        gen = _gen(r)

        def make(heads):
            if r.random() < 0.5:
                return torch.randn(1, heads, n, d, generator=gen, device=dev).to(dtype)
            return torch.randn(1, n, heads, d, generator=gen, device=dev).to(dtype).transpose(1, 2)

        q0, k0, v0 = make(hq), make(hkv), make(hkv)
        grad = torch.randn(1, n, hq, d, generator=gen, device=dev).to(dtype)
        rows = [(s, e) for s, e in bounds if e > s]
        what = f"case {i}: lengths={lengths} rep={rep} hkv={hkv} d={d} {dtype} scale={scale:.3g}"

        def run(fn):
            q, k, v = (t.detach().clone().requires_grad_(True) for t in (q0, k0, v0))
            out = fn(q, k, v)
            grads = torch.autograd.grad(out, (q, k, v), grad)
            return out.detach(), grads

        ref, ref_grads = run(lambda q, k, v: _rows_reference(q, k, v, bounds, scale))
        _, again = run(lambda q, k, v: _rows_reference(q, k, v, bounds, scale))
        candidates = [("rows", lambda q, k, v: RowsAttention.apply(q, k, v, rows, scale))]

        def varlen(q, k, v):
            with rows_context(bounds, device=dev):
                cu, max_len, total = unpad._cu[-1]
                assert total == n and max_len == max(lengths)
                return VarlenRowsAttention.apply(q, k, v, cu, max_len, scale)

        if d <= 128:
            candidates.append(("varlen", varlen))
        for name, fn in candidates:
            out, grads = run(fn)
            _, grads_again = run(fn)
            _assert_equal(out.reshape(ref.shape), ref, f"{what}: {name} forward")
            for gname, ours, ours_again, theirs, theirs_again in zip("qkv", grads, grads_again,
                                                                     ref_grads, again):
                spread = max(float((theirs.float() - theirs_again.float()).abs().max()),
                             float((ours.float() - ours_again.float()).abs().max()))
                if gname == "q":  # accumulated over key blocks in a layout-dependent order:
                    # one ulp of the element, or fp32 rounding of the summed terms near zero
                    top = torch.maximum(ours.double().abs(), theirs.double().abs())
                    _assert_within(ours, theirs.double(), _ulp(top, dtype) + 2 * spread
                                   + 2.0**-18 * float(theirs.abs().max()),
                                   f"{what}: {name} dq (one ulp)")
                elif spread == 0.0:
                    _assert_equal(ours, theirs, f"{what}: {name} d{gname}")
                else:
                    _assert_within(ours, theirs.double(), 2e-3 * theirs.double().abs()
                                   + 2 * spread, f"{what}: {name} d{gname}")


# --- LoRA projection node and its fp16 adapter copies (lora.py) -------------------------------

def _lora_layer(hidden, kv, rank):
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

    from solution.qgroup import setup_layer
    from solution.test_kernels import _fake_layer

    layer = _fake_layer(hidden=hidden, kv=kv)
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05)
    for name in ("q_proj", "v_proj"):
        wrapped = LoraLinear4bit(getattr(layer.self_attn, name), "default", config=config,
                                 r=rank, lora_alpha=2 * rank, lora_dropout=0.05)
        wrapped.lora_A.float()
        wrapped.lora_B.float()
        torch.nn.init.normal_(wrapped.lora_B["default"].weight, std=0.05)
        setattr(layer.self_attn, name, wrapped)
    setup_layer(layer)
    assert hasattr(layer.self_attn.q_proj.base_layer, "_solution_group"), "layer not packed"
    return layer


def test_fuzz_lora_node():
    """The q/v projection + adapter node against PEFT's own Linear4bit.forward on random token
    counts and batch shapes, ranks, scalings (powers of two and not), dropout p in {0, 0.05, 0.5}
    in train and eval mode, inputs and adapters with or without gradient: outputs, the shared
    input's gradient, the adapter gradients and the CUDA generator state afterwards bit for bit
    (lora.py's claim; the dropout replayed from the saved generator state)."""
    import bitsandbytes as bnb

    from solution import lora
    from solution.test_lora import _group_forward

    assert lora.patch_peft_lora() is True
    dev = "cuda"
    layers = {}
    for i, r in _cases("lora", 80):
        hidden, kv, rank = r.choice([(256, 64, 16), (256, 64, 1), (512, 128, 8),
                                     (1536, 256, 16), (1536, 256, 64)])
        if (hidden, kv, rank) not in layers:
            torch.manual_seed(len(layers) + 40)
            layers[hidden, kv, rank] = _lora_layer(hidden, kv, rank)
        attn = layers[hidden, kv, rank].self_attn
        q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
        p = r.choice([0.0, 0.05, 0.5])
        scaling = r.choice([2.0, 2.0, 1.0, 0.5, 4.0, 1.5, 0.3])
        train = r.random() < 0.7
        for module in (q, v):
            module.lora_dropout["default"] = torch.nn.Dropout(p) if p > 0 else torch.nn.Identity()
            module.scaling["default"] = scaling
            module.train(train)
        params = [m.lora_A["default"].weight for m in (q, v)] + \
                 [m.lora_B["default"].weight for m in (q, v)]
        adapters_grad = r.random() < 0.85
        for prm in params:
            prm.requires_grad_(adapters_grad)
        x_grad = r.random() < 0.85
        n = _log_int(r, 1, 2000)
        shape = r.choice([(1, n, hidden), (1, n, hidden), (2, max(1, n // 2), hidden)])
        gen = _gen(r)
        leaf = torch.randn(shape, generator=gen, device=dev).half() * r.choice([0.5, 1.0, 3.0])
        grads = [torch.randn(shape[:-1] + (width,), generator=gen, device=dev).half()
                 for width in (hidden, kv, kv)]
        seed = r.getrandbits(31)
        what = (f"case {i}: hidden={hidden} rank={rank} p={p} scaling={scaling} train={train} "
                f"shape={shape} x_grad={x_grad} adapters_grad={adapters_grad}")

        def run(forward):
            for prm in params:
                prm.grad = None
            x = leaf.clone().requires_grad_(x_grad)
            torch.cuda.manual_seed(seed)
            with torch.autocast("cuda", dtype=F16):
                h = x * 1.0
                outs = [forward(q, h), k(h), forward(v, h)]
            if x_grad or adapters_grad:
                sum((o * g).float().sum() for o, g in zip(outs, grads)).backward()
            return ([o.detach() for o in outs], x.grad,
                    [None if prm.grad is None else prm.grad.clone() for prm in params],
                    torch.cuda.get_rng_state())

        saved = bnb.nn.Linear4bit.forward
        bnb.nn.Linear4bit.forward = _group_forward
        try:
            ref = run(type(q).forward._original)
            ours = run(lambda mod, inp: mod(inp))
        finally:
            bnb.nn.Linear4bit.forward = saved
        for name, a, b in zip(("q", "k", "v"), ours[0], ref[0]):
            _assert_equal(a, b, f"{what}: output {name}")
        _assert_equal(ours[1], ref[1], f"{what}: input gradient")
        for j, (a, b) in enumerate(zip(ours[2], ref[2])):
            _assert_equal(a, b, f"{what}: adapter gradient {j}")
        _assert_equal(ours[3], ref[3], f"{what}: CUDA generator state")
    for layer in layers.values():
        for module in (layer.self_attn.q_proj, layer.self_attn.v_proj):
            for prm in module.parameters():
                prm.requires_grad_(prm.dtype == F32)


def test_fuzz_half_copies_follow_the_adapter_weights():
    """The adapters' fp16 copies (lora.HalfCopies) through a random sequence of updates: AdamW
    steps, in-place updates, load_state_dict, new storage assigned to the Parameter's .data
    (several times in a row: the allocator hands freed blocks out again), a new Parameter
    object: after each, the node's output equals PEFT's (computed from the current fp32
    weights) bit for bit."""
    import bitsandbytes as bnb

    from solution import lora
    from solution.test_lora import _group_forward

    assert lora.patch_peft_lora() is True
    torch.manual_seed(77)
    layer = _lora_layer(256, 64, 16)
    q = layer.self_attn.q_proj
    q.eval()
    x = torch.randn(1, 123, 256, device="cuda").half()
    opt = torch.optim.AdamW([q.lora_A["default"].weight, q.lora_B["default"].weight], lr=0.01)
    r = random.Random("half-copies")
    saved = bnb.nn.Linear4bit.forward
    bnb.nn.Linear4bit.forward = _group_forward
    try:
        for step in range(max(1, round(80 * SCALE))):
            lin = r.choice([q.lora_A["default"], q.lora_B["default"]])
            op = r.choice(["step", "inplace", "load", "assign", "assign2", "replace", "none"])
            prm = lin.weight
            with torch.no_grad():
                if op == "step":
                    for p_ in opt.param_groups[0]["params"]:
                        p_.grad = torch.randn_like(p_)
                    opt.step()
                elif op == "inplace":
                    prm.mul_(r.uniform(0.5, 1.5))
                elif op == "load":
                    lin.load_state_dict({"weight": torch.randn_like(prm) * 0.05})
                elif op in ("assign", "assign2"):
                    for _ in range(1 if op == "assign" else 2):
                        prm.data = torch.randn_like(prm) * 0.05
                elif op == "replace":
                    lin.weight = torch.nn.Parameter(torch.randn_like(prm) * 0.05)
            with torch.autocast("cuda", dtype=F16):
                ours = q(x)
                ref = type(q).forward._original(q, x)
            _assert_equal(ours, ref, f"step {step} after {op!r}")
    finally:
        bnb.nn.Linear4bit.forward = saved


# --- fused projections on the packed layer (fused_proj.py) -----------------------------------

def test_fuzz_fused_projections():
    """q|k|v as one FusedProjections node and the opt-in FusedMLP, against the separate path
    (PEFT's LoRA forward on GroupLinear bases, GroupLinear, the in-place SwiGLU) on random token
    counts (not multiples of 8 too), ranks, scalings, dropout p in {0, 0.05, 0.5}, train/eval and
    backward passes split per projection: the adapter weight gradients and the CUDA generator
    state bit for bit (fused_proj.py's claim), outputs and input gradients within a few fp16
    roundings of the GEMM magnitudes (another accumulation order, fused_proj.py's numerics)."""
    import bitsandbytes as bnb

    from solution import lora, qgroup
    from solution.fused_proj import GATE, Q, V, FusedMLP, FusedProjections, qkv_plan
    from solution.swiglu_inplace import SwiGLUInPlaceGrad
    from solution.test_lora import _group_forward

    assert lora.patch_peft_lora() is True
    dev = "cuda"
    layers = {}
    for i, r in _cases("fused-proj", 40):
        hidden, kv, rank = r.choice([(256, 64, 16), (512, 128, 8), (1536, 256, 16)])
        if (hidden, kv, rank) not in layers:
            torch.manual_seed(len(layers) + 60)
            layers[hidden, kv, rank] = _lora_layer(hidden, kv, rank)
        layer = layers[hidden, kv, rank]
        attn, mlp = layer.self_attn, layer.mlp
        q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
        group = q.base_layer._solution_group[0]
        p = r.choice([0.0, 0.05, 0.5])
        scaling = r.choice([2.0, 2.0, 1.0, 0.5, 1.5])
        train = r.random() < 0.7
        for module in (q, v):
            module.lora_dropout["default"] = torch.nn.Dropout(p) if p > 0 else torch.nn.Identity()
            module.scaling["default"] = scaling
            module.train(train)
        n = _log_int(r, 1, 1500)
        shape = r.choice([(1, n, hidden), (2, max(1, n // 2), hidden)])
        gen = _gen(r)
        leaf = torch.randn(shape, generator=gen, device=dev).half()
        grads = [torch.randn(shape[:-1] + (w,), generator=gen, device=dev).half()
                 for w in (hidden, kv, kv)]
        split = r.random() < 0.3  # the backward of the layer before the last: per projection
        seed = r.getrandbits(31)
        params = [m.lora_A["default"].weight for m in (q, v)] + \
                 [m.lora_B["default"].weight for m in (q, v)]
        what = (f"case {i}: hidden={hidden} rank={rank} p={p} scaling={scaling} train={train} "
                f"shape={shape} split={split}")

        def run(fused):
            for prm in params:
                prm.grad = None
            x = leaf.clone().requires_grad_(True)
            torch.cuda.manual_seed(seed)
            with torch.autocast("cuda", dtype=F16):
                h = x * 1.0
                if fused:
                    plan = qkv_plan(attn, h)
                    assert plan is not None, what
                    grp, adapters, weights = plan
                    grp.expected_bwd += 3
                    outs = FusedProjections.apply(h, grp, Q, V + 1, adapters, (V - Q,), *weights)
                else:
                    original = type(q).forward._original
                    outs = (original(q, h), k(h), original(v, h))
            sum((o * g).float().sum() for o, g in zip(outs, grads)).backward()
            return ([o.detach() for o in outs], x.grad, [prm.grad.clone() for prm in params],
                    torch.cuda.get_rng_state())

        saved = bnb.nn.Linear4bit.forward
        bnb.nn.Linear4bit.forward = _group_forward
        group.depth = 1 if split else None
        try:
            with _patched(qgroup, SPLIT_LAYERS={"fwd": 1, "bwd": 2}):
                ref = run(False)
                ours = run(True)
        finally:
            bnb.nn.Linear4bit.forward = saved
            group.depth = None
        h16 = leaf.reshape(-1, hidden)
        for name, idx, a, b in zip("qkv", range(3), ours[0], ref[0]):
            w = group.weight("fwd", idx)
            _assert_within(a, b.double(), _gemm_tol(h16, w).view(b.shape) + 2 * _ulp(b.double(),
                                                                                     F16),
                           f"{what}: output {name}")
        group.cache["fwd"] = [None, 0]
        w_cat = torch.cat([group.dequantize_all()[group.elem_bounds[j]:group.elem_bounds[j + 1]]
                           .view(group.shapes[j]) for j in range(3)])
        dy_cat = torch.cat([g.reshape(-1, g.shape[-1]) for g in grads], 1)
        _assert_within(ours[1], ref[1].double(),
                       _gemm_tol(dy_cat, w_cat.t(), 2.0**-6).view(ref[1].shape)
                       + 4 * _ulp(ref[1].double(), F16), f"{what}: input gradient")
        for j, (a, b) in enumerate(zip(ours[2], ref[2])):
            _assert_equal(a, b, f"{what}: adapter gradient {j}")
        _assert_equal(ours[3], ref[3], f"{what}: CUDA generator state")
        group.expected_bwd = group.split_left = 0
        group.cache = {"fwd": [None, 0], "bwd": [None, 0]}
        # the opt-in fused MLP against the default one on the same group
        if r.random() < 0.5:
            x0 = torch.randn(shape, generator=gen, device=dev).half()
            gy = torch.randn(shape, generator=gen, device=dev).half()
            res = []
            for fused in (False, True):
                x = x0.clone().requires_grad_(True)
                group.expected_bwd += 3
                if fused:
                    y = FusedMLP.apply(x, group, True)
                else:
                    act = SwiGLUInPlaceGrad.apply(qgroup.GroupLinear.apply(x, group, GATE),
                                                  qgroup.GroupLinear.apply(x, group, GATE + 1))
                    y = qgroup.GroupLinear.apply(act, group, GATE + 2)
                y.backward(gy)
                res.append((y.detach(), x.grad))
                group.expected_bwd = group.split_left = 0
                group.cache = {"fwd": [None, 0], "bwd": [None, 0]}
            # opt-in path without a stated bound: a sanity check relative to the largest element
            # (gate/up and down accumulated in other orders, rounding differences carried
            # through the activation)
            _assert_within(res[1][0], res[0][0].double(),
                           2.0**-5 * float(res[0][0].abs().max()) + 4 * _ulp(res[0][0].double(),
                                                                             F16),
                           f"{what}: fused MLP output")
            _assert_within(res[1][1], res[0][1].double(),
                           2.0**-5 * float(res[0][1].abs().max()) + 4 * _ulp(res[0][1].double(),
                                                                             F16),
                           f"{what}: fused MLP input gradient")
    for layer in layers.values():
        for module in (layer.self_attn.q_proj, layer.self_attn.v_proj):
            module.scaling["default"] = 2.0


# --- SwiGLU with the backward written in place (swiglu_inplace.py) ----------------------------

def test_fuzz_swiglu_inplace_matches_the_example():
    """Our in-place-backward wrapper around the example's kernels on random sizes, fp16/bf16/
    fp32, gate/up as fresh tensors, as non-contiguous halves of a fused projection or as two
    contiguous views of one buffer: output and input gradients bit for bit the example's
    (swiglu_inplace.py's claim), the output within one rounding of float64 silu(gate) * up."""
    from solution.swiglu import swiglu
    from solution.swiglu_inplace import SwiGLUInPlaceGrad

    dev = "cuda"
    for i, r in _cases("swiglu-inplace", 150):
        inter = _log_int(r, 1, 9000)
        n = max(1, min(_log_int(r, 1, 3000), 6_000_000 // inter))
        dtype = r.choice([F16, F16, BF16, F32])
        kind = r.choice(["fresh", "fresh", "halves", "stacked"])
        gen = _gen(r)
        scale = r.choice([1.0, 4.0, 20.0])
        if kind == "fresh":
            leaves = [torch.randn(n, inter, generator=gen, device=dev).to(dtype) * scale
                      for _ in range(2)]
        elif kind == "halves":
            leaves = [torch.randn(n, 2 * inter, generator=gen, device=dev).to(dtype) * scale]
        else:
            leaves = [torch.randn(2, n, inter, generator=gen, device=dev).to(dtype) * scale]
        grad = _strided(torch.randn(n, inter, generator=gen, device=dev).to(dtype), r)
        what = f"case {i}: n={n} inter={inter} {dtype} {kind}"

        def run(fn):
            ls = [t.detach().clone().requires_grad_(True) for t in leaves]
            if kind == "fresh":
                gate, up = ls[0] * 1.0, ls[1] * 1.0
            elif kind == "halves":
                gate, up = (ls[0] * 1.0).chunk(2, -1)
            else:
                fused = ls[0] * 1.0
                gate, up = fused[0], fused[1]
            out = fn(gate, up)
            out.backward(grad)
            return out.detach(), [t.grad for t in ls]

        ref_out, ref_grads = run(swiglu)
        out, grads = run(SwiGLUInPlaceGrad.apply)
        _assert_equal(out, ref_out, f"{what}: output")
        for j, (a, b) in enumerate(zip(grads, ref_grads)):
            _assert_equal(a, b, f"{what}: gradient {j}")
        gate, up = (leaves[0], leaves[1]) if kind == "fresh" else (
            leaves[0].chunk(2, -1) if kind == "halves" else (leaves[0][0], leaves[0][1]))
        g, u = gate.double(), up.double()
        exact = g * torch.sigmoid(g) * u
        # the example's fp32 sigmoid flushes below fp32's normal range (gate < -87)
        flush = torch.finfo(F32).tiny * (g.abs() * u.abs() + 1)
        _assert_within(out, exact, _ulp(exact, dtype) + 2.0**-18 * exact.abs() + flush,
                       f"{what}: value")
