"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import warnings

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

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


def test_swiglu_inplace_backward_matches_example_bitwise():
    """Our wrapper around the example's kernels: the same output and gradients bit for bit,
    and its backward allocates no [N, 8960] gradient buffers (they overwrite gate and up)."""
    from solution.swiglu import swiglu
    from solution.swiglu_inplace import SwiGLUInPlaceGrad

    torch.manual_seed(3)
    shape = (1427, 8960)
    x = torch.randn(shape[0], 64, device="cuda", dtype=torch.float16, requires_grad=True)
    w_gate = torch.randn(64, shape[1], device="cuda", dtype=torch.float16) * 0.2
    w_up = torch.randn(64, shape[1], device="cuda", dtype=torch.float16) * 0.2
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    results = {}
    for name, fn in (("example", swiglu), ("inplace", SwiGLUInPlaceGrad.apply)):
        out = fn(x @ w_gate, x @ w_up)                                               
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out.backward(grad)
        torch.cuda.synchronize()
        results[name] = (out.detach(), x.grad.clone(), torch.cuda.max_memory_allocated() - base)
        x.grad = None
    for a, b in zip(results["example"][:2], results["inplace"][:2]):
        assert torch.equal(a, b)
    buffer = shape[0] * shape[1] * 2
    assert results["inplace"][2] <= results["example"][2] - 2 * buffer + 2**20


                                                                                              

def reference_head(hidden, weight, targets, num_items):
    """What the base run computes: fp16 logits, transformers ForCausalLMLoss (fp32 upcast,
    sum / num_items), TRL entropy and accuracy, Soup per-token losses."""
    logits16 = hidden.to(torch.float16) @ weight.to(torch.float16).t()
    logits = logits16.float()
    labelled = targets >= 0
    denom = float(num_items) if num_items is not None else max(int(labelled.sum()), 1)
    loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum") / denom
    with torch.no_grad():
        token_loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
        lse = torch.logsumexp(logits, dim=-1)
        entropy = lse - (torch.softmax(logits, dim=-1) * logits).sum(-1)
        correct = (logits16.argmax(-1) == targets) & labelled
    return loss, token_loss, entropy, correct


def _head_chunks(path, token_chunk, vocab_chunk):
    """Keyword arguments selecting the vocabulary-sliced (default) or token-chunked head."""
    return {"token_chunk": token_chunk} if path == "tokens" else {"vocab_chunk": vocab_chunk}


def apply_quietly():
    """solution.apply() without its warm-up thread (it would go on compiling into later tests:
    only the first fused forward of the process joins it) and without gc.freeze (everything
    alive now would stay out of the collector for the rest of the session). Both have tests of
    their own (test_threads.py, test_startup.py)."""
    import solution

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SOLUTION_WARMUP_THREAD", "0")
        mp.setenv("SOLUTION_DISABLE_GC_FREEZE", "1")
        solution.apply()


def boundary_columns(vocab, chunk):
    """Columns at the edges the head's loops cut the vocabulary at: the first, either side of
    the first BLOCK_V block boundary and of the first slice boundary, the first column of the
    last slice, the last column."""
    from solution import fused_head

    block = fused_head.BLOCK_V
    last = (vocab - 1) // chunk * chunk
    columns = [0, block - 1, block, chunk - 1, chunk, last, vocab - 1]
    return list(dict.fromkeys(c for c in columns if 0 <= c < vocab))


def plant_hits(hidden, weight, targets, rows, columns, scale):
    """With random targets the argmax hits about 1 target in V, so an accuracy that is always
    False would pass. Here each row gets a hidden state along a weight row (`scale` times it):
    that column is the clear argmax. Every other planted row gets that column as target (a
    hit), the others the column next to it (a miss). Returns the number of planted hits."""
    vocab = weight.shape[0]
    hits = 0
    with torch.no_grad():
        for i, (row, col) in enumerate(zip(rows, columns)):
            hidden[row] = (weight[col].float() * scale).to(hidden.dtype)
            if i % 2 == 0:
                targets[row] = col
                hits += 1
            else:
                targets[row] = col + 1 if col + 1 < vocab else col - 1
    return hits


@pytest.mark.parametrize("path", ["vocab", "tokens"])
@pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("n_tokens,vocab,token_chunk,vocab_chunk",
                         [(2048, 151936, 512, None), (37, 151936, 512, 20000),
                          (300, 1000, 128, 384), (5, 4097, 2, 4096)])
@pytest.mark.parametrize("num_items", ["tensor", "int", None])
def test_fused_head_matches_reference(path, hidden_dtype, n_tokens, vocab, token_chunk,
                                      vocab_chunk, num_items):
    from solution import fused_head
    from solution.fused_head import fused_head_loss

    torch.manual_seed(0)
    hidden_size = 1536 if vocab > 10000 else 64
    hidden = torch.randn(n_tokens, hidden_size, device="cuda", dtype=hidden_dtype) * 2
    weight = (torch.randn(vocab, hidden_size, device="cuda") * 0.05).requires_grad_(False)
    targets = torch.randint(0, vocab, (n_tokens,), device="cuda")
    targets[torch.rand(n_tokens, device="cuda") < 0.45] = -100                           
    targets[0] = -100
    if n_tokens > 2:
        targets[-1] = -100
    columns = boundary_columns(vocab, vocab_chunk or fused_head.VOCAB_CHUNK)
    planted = plant_hits(hidden, weight, targets, range(1, n_tokens - 1), columns, 40.0)
    hidden.requires_grad_(True)
    labelled = int((targets >= 0).sum())
    items = {"tensor": torch.tensor(labelled + 3, device="cuda"), "int": labelled + 3,
             None: None}[num_items]

    ref_loss, ref_token_loss, ref_entropy, ref_correct = reference_head(hidden, weight, targets,
                                                                        items)
    grad_scale = 3.0                                           
    (ref_loss * grad_scale).backward()
    ref_grad = hidden.grad.clone()
    hidden.grad = None

    loss, token_loss, entropy, correct = fused_head_loss(
        hidden, weight, targets, items, **_head_chunks(path, token_chunk, vocab_chunk))
    (loss * grad_scale).backward()

    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(token_loss, ref_token_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    assert int(ref_correct.sum()) >= planted > 0                             
    assert torch.equal(correct, ref_correct)
    assert (token_loss[targets < 0] == 0).all()
                                                                                              
                                                                                            
                                                                                         
    scale = float(ref_grad.abs().max())
    torch.testing.assert_close(hidden.grad.float(), ref_grad.float(), rtol=2e-3,
                               atol=2 ** -7 * scale)
    assert (hidden.grad[targets < 0] == 0).all()


@pytest.mark.parametrize("path", ["vocab", "tokens"])
def test_fused_head_no_grad_path(path):
    from solution.fused_head import fused_head_loss

    torch.manual_seed(1)
    hidden = torch.randn(64, 1536, device="cuda", dtype=torch.float16)
    weight = torch.randn(151936, 1536, device="cuda") * 0.05
    targets = torch.randint(0, 151936, (64,), device="cuda")
    with torch.no_grad():
        loss, token_loss, entropy, correct = fused_head_loss(
            hidden, weight, targets, None, **_head_chunks(path, 256, None))
    ref_loss, *_ = reference_head(hidden, weight, targets, None)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)


def test_vocab_kernel_statistics_match_rows_kernel():
    """On the same logits, the vocabulary-sliced kernel computes the row statistics in the
    token-chunked kernel's order (slices a multiple of BLOCK_V): the same per-token loss and
    entropy up to fp32 rounding (bit-identical on the cards we tried) and the same argmax hits.
    (End to end the two paths' logits come from GEMMs of different shapes, which cuBLAS may
    round differently in the last bit: compared with a tolerance below.)"""
    from solution import fused_head

    assert fused_head.VOCAB_CHUNK % fused_head.BLOCK_V == 0
    torch.manual_seed(12)
    n, size, vocab = 300, 256, 151936
    hidden = torch.randn(n, size, device="cuda", dtype=torch.float16)
    weight = (torch.randn(vocab, size, device="cuda") * 0.2).half()
    targets = torch.randint(0, vocab, (n,), device="cuda")
    targets[torch.rand(n, device="cuda") < 0.3] = -100
    planted = plant_hits(hidden, weight, targets, range(1, 8),
                         boundary_columns(vocab, fused_head.VOCAB_CHUNK), 2.0)
    logits = hidden @ weight.t()

    def outputs():
        return (torch.empty(n, dtype=torch.float32, device="cuda"),
                torch.empty(n, dtype=torch.float32, device="cuda"),
                torch.empty(n, dtype=torch.int32, device="cuda"))

    ref = outputs()
    fused_head.fused_head_rows_kernel[(n,)](logits.clone(), targets, *ref, n, vocab, vocab, 0,
                                           BLOCK=fused_head.BLOCK_V,
                                           num_warps=fused_head.NUM_WARPS)
    ours = outputs()
    stats = torch.empty(n, 5, dtype=torch.float32, device="cuda")
    best = torch.empty(n, dtype=torch.int32, device="cuda")
    unused_w = torch.zeros(1, size, dtype=torch.float32, device="cuda")                  
    chunk = fused_head.VOCAB_CHUNK
    for start in range(0, vocab, chunk):
        cols = min(chunk, vocab - start)
        piece = logits[:, start:start + cols].contiguous()
        fused_head.fused_head_vocab_kernel[(n,)](
            piece, targets, stats, best, stats, unused_w, hidden, ours[0], ours[1], ours[2],
            cols, start, cols, size, 0, size, 0, int(start == 0), int(start + cols >= vocab), 0,
            BLOCK=fused_head.BLOCK_V, BLOCK_H=fused_head.BLOCK_H, UNIT=1,
            num_warps=fused_head.NUM_WARPS)
    torch.testing.assert_close(ours[0], ref[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ours[1], ref[1], rtol=1e-6, atol=1e-6)
    hits = (logits.argmax(-1) == targets) & (targets >= 0)
    assert int(hits.sum()) >= planted == 4 and torch.equal(ref[2].bool(), hits)
    assert torch.equal(ours[2], ref[2])


@pytest.mark.parametrize("path", ["vocab", "tokens"])
def test_head_argmax_ties_report_the_first_column(path):
    """Two equal largest logits: the head's prediction is the first of the two columns, as
    torch.argmax (and so TRL's accuracy) has it, for a tie inside one BLOCK_V block, across
    blocks, across vocabulary slices and inside the narrower last slice. The tied logits are
    exact in any summation order (+-2^-4 weights, +-4 hidden states: 64 exactly)."""
    from solution import fused_head
    from solution.fused_head import fused_head_loss

    torch.manual_seed(13)
    n, size, vocab = 16, 256, 151936
    block, chunk = fused_head.BLOCK_V, fused_head.VOCAB_CHUNK
    pairs = [(block + 10, block + 20), (1000, block + 1000), (100, chunk + 50),
             (vocab - 4000, vocab - 1)]
    hidden = torch.randn(n, size, device="cuda", dtype=torch.float16) * 2
    weight = torch.randn(vocab, size, device="cuda") * 0.05                          
    targets = torch.full((n,), -100, dtype=torch.long, device="cuda")
    targets[2 * len(pairs):] = torch.randint(0, vocab, (n - 2 * len(pairs),), device="cuda")
    signs = torch.randint(0, 2, (len(pairs), size), device="cuda").float() * 2 - 1
    for i, (first, second) in enumerate(pairs):
        weight[first] = weight[second] = signs[i] / 16
        hidden[2 * i] = hidden[2 * i + 1] = (signs[i] * 4).half()
        targets[2 * i], targets[2 * i + 1] = first, second                                      
    logits = hidden @ weight.half().t()
    for i, (first, second) in enumerate(pairs):
        for row in (2 * i, 2 * i + 1):
            assert logits[row, first] == logits[row, second] == logits[row].max() == 64
    ref_correct = (logits.argmax(-1) == targets) & (targets >= 0)
    assert ref_correct[:2 * len(pairs)].tolist() == [True, False] * len(pairs)
    with torch.no_grad():
        *_, correct = fused_head_loss(hidden, weight, targets, None,
                                      **_head_chunks(path, 256, None))
    assert torch.equal(correct, ref_correct)


def test_vocab_head_matches_token_head_and_holds_no_full_fp16_weight():
    """End to end the vocabulary-sliced head gives the token-chunked head's loss, per-token loss,
    entropy and accuracy (up to last-bit logit differences between the two GEMM shapes), a
    gradient at least as close to an fp32 reference, and its forward never holds a full fp16
    copy of the weight."""
    from solution.fused_head import fused_head_loss

    torch.manual_seed(11)
    n, size, vocab = 700, 1536, 151936
    weight = torch.randn(vocab, size, device="cuda") * 0.03
    targets = torch.randint(0, vocab, (n,), device="cuda")
    targets[torch.rand(n, device="cuda") < 0.35] = -100
    hidden = torch.randn(n, size, device="cuda", dtype=torch.float16) * 1.5
    confident = (torch.rand(n, device="cuda") < 0.3) & (targets >= 0)
    hidden[confident] = (weight[targets[confident]] * 400).half()                        
    hidden.requires_grad_(True)
    outs = {}
    for path in ("tokens", "vocab"):
        hidden.grad = None
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss, token_loss, entropy, correct = fused_head_loss(
            hidden, weight, targets, None, **_head_chunks(path, 256, None))
        transient = torch.cuda.max_memory_allocated() - base
        (loss * 1024.0).backward()
        outs[path] = (loss.detach(), token_loss, entropy, correct, hidden.grad.float(), transient)
                                                                                             
                                                                                               
    for a, b in zip(outs["tokens"][:3], outs["vocab"][:3]):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    assert int((outs["tokens"][3] != outs["vocab"][3]).sum()) <= 1
    assert outs["vocab"][5] < vocab * size                                             
    with torch.no_grad():
        logits = hidden.float() @ weight.half().float().t()
        probs = torch.softmax(logits, -1)
        labelled = targets >= 0
        probs[labelled, targets[labelled]] -= 1
        probs[~labelled] = 0
        exact = (probs @ weight.half().float()) * (1024.0 / labelled.sum())
    err_tokens = (outs["tokens"][4] - exact).abs().max()
    err_vocab = (outs["vocab"][4] - exact).abs().max()
    assert err_vocab <= err_tokens * 1.25 + 1e-6                             
    assert (outs["vocab"][4][~labelled] == 0).all()


def test_labelled_first_order():
    from solution.fused_head import labelled_first

    torch.manual_seed(3)
    labelled = torch.rand(1000, device="cuda") < 0.4
    n_lab = int(labelled.sum())
    dest, src = labelled_first(labelled, n_lab)
    rows = torch.arange(1000, device="cuda")
    assert torch.equal(src[dest], rows) and torch.equal(dest[src], rows)
    assert labelled[src[:n_lab]].all() and not labelled[src[n_lab:]].any()
                                         
    assert (src[1:n_lab] > src[:n_lab - 1]).all() and (src[n_lab + 1:] > src[n_lab:-1]).all()


@pytest.mark.parametrize("n_tokens,vocab,vocab_chunk,unlabelled",
                         [(700, 151936, None, 0.45), (37, 151936, 20000, 0.3),
                          (300, 1000, 384, 0.0), (300, 1000, 384, 1.0), (5, 4097, 4096, 0.5)])
def test_head_compaction_matches_full_accumulator(monkeypatch, n_tokens, vocab, vocab_chunk,
                                                   unlabelled):
    """The accumulator GEMM over the labelled rows only (rows reordered labelled-first): loss,
    per-token loss, entropy and accuracy bit-identical to the accumulator over all rows (the
    logits GEMM and the kernel see the same rows, only reordered), the gradient equal up to the
    last-bit rounding of a GEMM with fewer rows, exact zeros on unlabelled rows, and the same
    result whether the caller passes the labelled count or the head counts it."""
    from solution import fused_head
    from solution.fused_head import fused_head_loss

    torch.manual_seed(13)
    size = 1536 if vocab > 10000 else 64
    weight = torch.randn(vocab, size, device="cuda") * 0.03
    hidden = (torch.randn(n_tokens, size, device="cuda") * 1.5).half().requires_grad_(True)
    targets = torch.randint(0, vocab, (n_tokens,), device="cuda")
    targets[torch.rand(n_tokens, device="cuda") < unlabelled] = -100
    n_lab = int((targets >= 0).sum())
    outs = {}
    for name, compact, given in (("full", False, None), ("compact", True, n_lab),
                                 ("counted", True, None)):
        monkeypatch.setattr(fused_head, "COMPACT", compact)
        hidden.grad = None
        loss, token_loss, entropy, correct = fused_head_loss(hidden, weight, targets, None,
                                                             vocab_chunk, n_labelled=given)
        (loss * 1024.0).backward()
        outs[name] = (loss.detach(), token_loss, entropy, correct, hidden.grad.clone())
    for a, b in zip(outs["full"][:4], outs["compact"][:4]):
        assert torch.equal(a, b)
    for a, b in zip(outs["compact"], outs["counted"]):
        assert torch.equal(a, b)
    full, compact = outs["full"][4].float(), outs["compact"][4].float()
                                                
    torch.testing.assert_close(compact, full, rtol=1e-3, atol=float(full.abs().max()) * 2**-10)
    assert (outs["compact"][4][targets < 0] == 0).all()


@pytest.mark.parametrize("n_tokens,vocab,vocab_chunk", [(300, 151936, None), (37, 4100, 1028)])
def test_head_row_units_match(monkeypatch, n_tokens, vocab, vocab_chunk):
    """The default sizes and strides in units of 8 elements (vectorized rows) against units of 1
    (SOLUTION_DISABLE_HEAD_ROWS=1): the same accuracy, loss / per-token loss / entropy to fp32
    rounding (the
    vectorized layout sums a row in another order) and the gradient to the last fp16 bit (a
    slice width that is not a multiple of 8, as the last of 4100 columns in slices of 1028,
    takes units of 1)."""
    from solution import fused_head
    from solution.fused_head import fused_head_loss

    torch.manual_seed(14)
    size = 1536 if vocab > 10000 else 64
    weight = torch.randn(vocab, size, device="cuda") * 0.03
    hidden = (torch.randn(n_tokens, size, device="cuda") * 1.5).half().requires_grad_(True)
    targets = torch.randint(0, vocab, (n_tokens,), device="cuda")
    targets[torch.rand(n_tokens, device="cuda") < 0.4] = -100
    outs = {}
    for unit in (1, 8):
        monkeypatch.setattr(fused_head, "ROW_UNIT", unit)
        hidden.grad = None
        loss, *per_token = fused_head_loss(hidden, weight, targets, None, vocab_chunk)
        (loss * 512.0).backward()
        outs[unit] = (loss.detach(), *per_token, hidden.grad.clone())
    for a, b in zip(outs[1][:3], outs[8][:3]):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-5)
    assert torch.equal(outs[1][3], outs[8][3])
    grad = outs[1][4].float()
    torch.testing.assert_close(outs[8][4].float(), grad, rtol=1e-3,
                               atol=float(grad.abs().max()) * 2**-10)


def test_unpad_counts_labelled_in_the_same_sync():
    from solution.unpad import unpad_batch

    torch.manual_seed(4)
    batch, seq = 4, 50
    lengths = [50, 13, 31, 1]
    mask = (torch.arange(seq, device="cuda")[None]
            < torch.tensor(lengths, device="cuda")[:, None]).long()
    shift = torch.randint(0, 100, (batch * seq,), device="cuda")
    shift[torch.rand(batch * seq, device="cuda") < 0.4] = -100
    idx, positions, bounds, got_lengths = unpad_batch(mask)
    idx2, positions2, bounds2, got_lengths2, counts = unpad_batch(mask, shift)
    assert torch.equal(idx, idx2) and torch.equal(positions, positions2)
    assert bounds == bounds2 and got_lengths == got_lengths2 == lengths
    assert counts == [int((shift[idx] >= 0).sum()), int((shift >= 0).sum())]


def test_patched_forward_compacts_the_head_without_an_extra_sync(monkeypatch):
    """Through the patched causal LM (padding removal on): the head gets the exact labelled
    count from the unpad sync, so the compaction adds no host sync, and the input gradient
    matches the full accumulator."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution import fused_head, patches

    apply_quietly()
    config = Qwen2Config(vocab_size=600, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=False,
                         attn_implementation="sdpa")
    torch.manual_seed(5)
    model = Qwen2ForCausalLM(config).cuda().train()
    model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)                               
    batch, seq = 3, 24
    input_ids = torch.randint(0, 600, (batch, seq), device="cuda")
    lengths = [24, 9, 17]
    attention_mask = (torch.arange(seq, device="cuda")[None]
                      < torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :6] = -100
    seen, drop = [], [False]
    real = patches.fused_head_loss

    def spy(hidden, weight, targets, num_items=None, vocab_chunk=None, **kw):
        seen.append((int((targets >= 0).sum()), kw.get("n_labelled")))
        if drop[0]:
            kw["n_labelled"] = None                                                 
        return real(hidden, weight, targets, num_items, vocab_chunk, **kw)

    monkeypatch.setattr(patches, "fused_head_loss", spy)
    monkeypatch.setitem(patches._unpad, "checked", 3)                                           
    grads, syncs = {}, {}
                                                           
    for compact in (True, False, True, "counted"):
        drop[0] = compact == "counted"
        monkeypatch.setattr(fused_head, "COMPACT", bool(compact))
        model.model.embed_tokens.weight.grad = None
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        syncs[compact] = sum("synchroniz" in str(w.message) for w in caught)
        out.loss.backward()
        grads[compact] = model.model.embed_tokens.weight.grad.float().clone()
    assert seen[-1][1] == seen[-1][0] > 0                                        
    assert syncs[True] == syncs[False]
    assert syncs["counted"] == syncs[True] + 1                              
    assert torch.equal(grads["counted"], grads[True])
                                                                               
    torch.testing.assert_close(grads[True], grads[False], rtol=1e-3,
                               atol=float(grads[False].abs().max()) * 2 ** -10)


def test_batch_shift_and_rewind_layout():
    """The model patch shifts labels per row: position t scores label t+1, last position and
    pads get -100; Soup's rewind reads [B, T] token losses in exactly that layout."""
    from solution.fused_head import fused_head_loss

    torch.manual_seed(2)
    batch, seq, size, vocab = 3, 16, 64, 1000
    hidden = torch.randn(batch, seq, size, device="cuda")
    weight = torch.randn(vocab, size, device="cuda") * 0.1
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, :5] = -100                  
    labels[1, 10:] = -100                                 
    shift = F.pad(labels, (0, 1), value=-100)[:, 1:]
    _, token_loss, _, _ = fused_head_loss(hidden.reshape(-1, size), weight, shift.reshape(-1),
                                          None, 7)                              
    token_loss = token_loss.view(batch, seq)
    assert (token_loss[:, -1] == 0).all()
    assert (token_loss[1, 9:] == 0).all()
    assert (token_loss[1, 4:9] > 0).all()
                                                             
    tokens = (labels[:, 1:] != -100).sum(1)
    row_loss = token_loss.sum(1) / tokens.clamp(min=1)
    assert row_loss.shape == (batch,) and torch.isfinite(row_loss).all()


                                                                                               

def reference_rmsnorm(x, weight, eps, out_dtype):
    """transformers Qwen2RMSNorm.forward followed by the autocast cast of its consumer."""
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return (weight * h.to(x.dtype)).to(out_dtype)


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1536), (1, 1, 1536), (5, 1000)])
def test_rmsnorm_matches_pytorch(out_dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(3)
    eps = 1e-6
    x = (torch.randn(shape, device="cuda") * 3).requires_grad_(True)
    weight = (1 + 0.1 * torch.randn(shape[-1], device="cuda")).requires_grad_(False)
    grad = torch.randn(shape, device="cuda", dtype=out_dtype)

    ref = reference_rmsnorm(x, weight, eps, out_dtype)
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    out = rmsnorm(x, weight, eps, out_dtype)
    out.backward(grad)
    assert out.dtype == out_dtype and out.shape == x.shape
    tol = 2e-3 if out_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
                                                                                          
    dx_tol = 3e-3 if out_dtype == torch.float16 else 1e-4
    torch.testing.assert_close(x.grad, ref_dx, rtol=dx_tol, atol=dx_tol * float(ref_dx.abs().max()))


def test_rmsnorm_patch_falls_back_and_emits_autocast_dtype():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution.rmsnorm import patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    norm = Qwen2RMSNorm(64, eps=1e-6).cuda()
    norm.weight.requires_grad_(False)
    x = torch.randn(2, 5, 64, device="cuda")
    assert norm(x).dtype == torch.float32
    with torch.autocast("cuda", dtype=torch.float16):
        assert norm(x).dtype == torch.float16
    norm.weight.requires_grad_(True)                                       
    with torch.autocast("cuda", dtype=torch.float16):
        assert norm(x).dtype == torch.float32
    cpu_norm = Qwen2RMSNorm(64, eps=1e-6)
    cpu_norm.weight.requires_grad_(False)
    assert cpu_norm(x.cpu()).device.type == "cpu"                              


                                                                                              

@pytest.mark.parametrize("shape", [(8960, 1536), (256, 1536), (1536, 8960), (64, 4094)])
def test_nf4_dequant_matches_bitsandbytes(shape):
    import bitsandbytes.functional as bnbf

    from solution.nf4_dequant import NF4_TABLE, nf4_dequantize

    torch.manual_seed(4)
    w = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.02
    packed, state = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    ref = bnbf.dequantize_4bit(packed, state)
    assert ref.dtype == torch.float16
    absmax = bnbf.dequantize_blockwise(state.absmax, state.state2) + state.offset
    out = torch.empty(shape, dtype=torch.float16, device="cuda")
    nf4_dequantize(packed, absmax, 64, out)
    assert torch.equal(out, ref)
                                                                                             
    codes = packed.view(-1).to(torch.int64)
    levels = torch.tensor(NF4_TABLE, device="cuda")
    vals = torch.stack([levels[codes >> 4], levels[codes & 15]], dim=1).reshape(-1)
    manual = (vals * absmax.repeat_interleave(64)[: vals.numel()]).to(torch.float16)
    assert torch.equal(out.view(-1), manual)


def test_nf4_patch_self_check_and_linear():
    import bitsandbytes as bnb
    import bitsandbytes.backends.cuda.ops as ops

    from solution.nf4_dequant import patch_bnb_dequant

    assert patch_bnb_dequant() is True
    assert getattr(ops._dequantize_4bit_impl, "_solution_patched", False)
    layer = bnb.nn.Linear4bit(1536, 512, bias=False, quant_type="nf4", compress_statistics=True,
                              compute_dtype=torch.float16).cuda()
    x = torch.randn(2048, 1536, device="cuda", dtype=torch.float16)
    y = layer(x)
    w = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)
    ref = (x @ w.to(x.dtype).t()).to(y.dtype)
    torch.testing.assert_close(y, ref, rtol=2e-3, atol=2e-3)


                                                                                             

def test_unpadded_causal_lm_matches_padded_forward():
    """The patched Qwen2ForCausalLM.forward (fused head + padding removal + per-row SDPA) gives
    the same loss and per-token losses as the original forward on a tiny random model."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution import patches

    apply_quietly()
    config = Qwen2Config(vocab_size=600, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=False,
                         attn_implementation="sdpa")
    torch.manual_seed(5)
    model = Qwen2ForCausalLM(config).cuda().train()
    model.requires_grad_(False)
    original = patches.ORIGINAL_FORWARD[Qwen2ForCausalLM]

    batch, seq = 3, 24
    input_ids = torch.randint(0, 600, (batch, seq), device="cuda")
    lengths = [24, 9, 17]
    attention_mask = (torch.arange(seq, device="cuda")[None]
                      < torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :4] = -100
    num_items = int((labels[:, 1:] != -100).sum())

    with torch.autocast("cuda", dtype=torch.float16):
        ref = original(model, input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                       num_items_in_batch=num_items)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    num_items_in_batch=num_items)
    assert out.logits is None
    torch.testing.assert_close(out.loss.float(), ref.loss.float(), rtol=2e-2, atol=1e-3)
    step = patches._pending_step[-1]
    shift = F.pad(labels, (0, 1), value=-100)[:, 1:]
    ref_token = F.cross_entropy(ref.logits.float().reshape(-1, 600), shift.reshape(-1),
                                ignore_index=-100, reduction="none").view(batch, seq)
    torch.testing.assert_close(step.token_loss, ref_token, rtol=3e-2, atol=2e-2)
    assert (step.token_loss[attention_mask == 0] == 0).all()
    assert int(step.labelled) == num_items


                                                                                              

@pytest.mark.parametrize("path", ["vocab", "tokens"])
def test_fused_head_unlabelled_rows_never_produce_nan(path):
    """Large logits on unlabelled rows must not leak inf/NaN into the hidden gradient."""
    from solution.fused_head import fused_head_loss

    torch.manual_seed(6)
    hidden = (torch.randn(512, 1536, device="cuda") * 60).requires_grad_(True)
    weight = torch.randn(151936, 1536, device="cuda") * 0.5
    targets = torch.randint(0, 151936, (512,), device="cuda")
    targets[::2] = -100
    loss, *_ = fused_head_loss(hidden, weight, targets, None, **_head_chunks(path, 256, None))
    (loss * 4096.0).backward()                             
    assert torch.isfinite(hidden.grad).all()
    assert (hidden.grad[targets < 0] == 0).all()


def test_rows_sdpa_matches_block_diagonal_causal_attention():
    from solution.unpad import patch_sdpa_rows, rows_context
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    patch_sdpa_rows()
    interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
    torch.manual_seed(7)
    lengths = [37, 5, 128, 64]
    n = sum(lengths)
    q = torch.randn(1, 12, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16)

    class Module:                                                  
        num_key_value_groups = 6
        training = True

    bounds, start = [], 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length
    with rows_context(bounds):
        out, _ = interface(Module(), q, k, v, None, dropout=0.0, scaling=128 ** -0.5,
                           sliding_window=None)
    out.backward(grad)
    grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None

    row = torch.cat([torch.full((length,), i, device="cuda") for i, length in enumerate(lengths)])
    pos = torch.cat([torch.arange(length, device="cuda") for length in lengths])
    mask = (row[:, None] == row[None, :]) & (pos[None, :] <= pos[:, None])
    k_rep = k.repeat_interleave(6, dim=1)
    v_rep = v.repeat_interleave(6, dim=1)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask, scale=128 ** -0.5)
    ref = ref.transpose(1, 2)
    ref.backward(grad)
                                                                                       
                                 
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3,
                               atol=4e-4 * float(ref.detach().abs().max()))
    for ours, t in zip(grads, (q, k, v)):
        torch.testing.assert_close(ours.float(), t.grad.float(), rtol=2e-3,
                                   atol=4e-4 * float(t.grad.abs().max()))


def _saved_bytes(fn):
    """Bytes of the distinct storages the autograd graph built by `fn` keeps for backward."""
    storages = {}

    def pack(t):
        storage = t.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = fn()
    return result, sum(storages.values())


def test_rows_attention_matches_repeated_kv_sdpa():
    """RowsAttention (K/V broadcast over the query heads with stride 0, nothing repeated kept
    for backward) gives exactly the outputs of the plain per-row path on repeat_kv copies with
    the same memory-efficient SDPA kernel, the same gradients (bit for bit where that backward
    is deterministic; on a T4 it may split the keys and accumulate dQ/dK in a varying order, so
    there the reference is only as exact as it is with itself), and saves far less."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from solution.unpad import RowsAttention, _repeat_kv

    torch.manual_seed(8)
    lengths = [37, 5, 128, 64, 1, 300]
    n = sum(lengths)
    scale = 128 ** -0.5
    q = torch.randn(1, 12, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    v_flat = torch.randn(1, n, 2, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16)
    bounds, start = [], 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length

    def plain():
        k_rep, v_rep = _repeat_kv(k, 6), _repeat_kv(v_flat.transpose(1, 2), 6)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            outs = [F.scaled_dot_product_attention(q[:, :, s:e], k_rep[:, :, s:e],
                                                   v_rep[:, :, s:e], is_causal=True, scale=scale)
                    for s, e in bounds]
        return torch.cat(outs, 2).transpose(1, 2).contiguous()

    results = {}
    for name, fn in (("plain", plain), ("plain again", plain),
                     ("rows", lambda: RowsAttention.apply(q, k, v_flat.transpose(1, 2), bounds,
                                                          scale))):
        out, saved = _saved_bytes(fn)
        out.backward(grad)
        results[name] = (out.detach(), saved, [t.grad.clone() for t in (q, k, v_flat)])
        for t in (q, k, v_flat):
            t.grad = None
    assert torch.equal(results["rows"][0], results["plain"][0])                           
    for ours, ref, again in zip(results["rows"][2], results["plain"][2],
                                results["plain again"][2]):
                                                                                 
        spread = float((ref.float() - again.float()).abs().max())
        torch.testing.assert_close(ours, ref, rtol=2e-3,
                                   atol=2e-3 * float(ref.abs().max()) + 2 * spread)
    repeated_kv = 2 * 12 * n * 128 * 2
    assert results["rows"][1] <= results["plain"][1] - repeated_kv * 0.8


                                                                                              

@pytest.mark.parametrize("quantized", [False, True])
def test_mlp_recompute_matches_stored_activations(quantized):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from transformers import Qwen2Config

    from solution.mlp_recompute import MLPRecompute
    from solution.swiglu import swiglu

    torch.manual_seed(8)
    config = Qwen2Config(hidden_size=256, intermediate_size=1024)
    mlp = Qwen2MLP(config).cuda().half()
    if quantized:
        import bitsandbytes as bnb

        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, name)
            q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, quant_type="nf4",
                                  compress_statistics=True, compute_dtype=torch.float16)
            q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False,
                                         quant_type="nf4", compress_statistics=True)
            setattr(mlp, name, q.cuda())
    mlp.requires_grad_(False).train()
    x = torch.randn(3, 40, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(3, 40, 256, device="cuda", dtype=torch.float16)

    ref = mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x)))                      
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    out = MLPRecompute.apply(x, mlp)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)                          
    torch.testing.assert_close(x.grad.float(), ref_dx.float(), rtol=2e-3, atol=2e-3)


                                                                                              

@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (256, 1536), (1536, 8960)])
def test_qlinear_dequant_matches_bitsandbytes(shape):
    import bitsandbytes.functional as bnbf

    from solution.qlinear import dequantize_weight

    torch.manual_seed(9)
    w = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.03
    packed, state = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    ref = bnbf.dequantize_4bit(packed, state)
    out = dequantize_weight(packed, state)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=1e-6)
    assert float((out != ref).float().mean()) < 1e-3                                        


def test_qlinear_patch_forward_backward():
    import bitsandbytes as bnb

    from solution.qlinear import patch_bnb_linear4bit

    torch.manual_seed(10)
    layer = bnb.nn.Linear4bit(1536, 512, bias=True, quant_type="nf4", compress_statistics=True,
                              compute_dtype=torch.float16).cuda()
    layer.requires_grad_(False)
    x = torch.randn(300, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(300, 512, device="cuda", dtype=torch.float16)
                                                        
    w = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)
    ref = F.linear(x, w.to(torch.float16), layer.bias.to(torch.float16))
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    assert patch_bnb_linear4bit() is True
    out = layer(x)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref_dx, rtol=2e-3, atol=2e-3)
    assert layer(x.float()).dtype == torch.float32                                  


                                                                                              

@pytest.mark.parametrize("n_tokens", [1, 37, 700, 2048])
def test_rope_matches_transformers(n_tokens):
    from transformers.models.qwen2 import modeling_qwen2

    from solution.rope import RoPE, patch_qwen2_rope

    torch.manual_seed(11)
    hq, hk, dim = 12, 2, 128
                                                                                              
    q_flat = torch.randn(1, n_tokens, hq * dim, device="cuda", dtype=torch.float16,
                         requires_grad=True)
    k_flat = torch.randn(1, n_tokens, hk * dim, device="cuda", dtype=torch.float16,
                         requires_grad=True)
    pos = torch.arange(n_tokens, device="cuda")[None]
    inv = 1.0 / (1e6 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    freqs = (pos.float()[..., None] * inv)              
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    grad_q = torch.randn(1, hq, n_tokens, dim, device="cuda", dtype=torch.float16)
    grad_k = torch.randn(1, hk, n_tokens, dim, device="cuda", dtype=torch.float16)

    original = modeling_qwen2.apply_rotary_pos_emb
    original = getattr(original, "_original", original)                                         
    q = q_flat.view(1, n_tokens, hq, dim).transpose(1, 2)
    k = k_flat.view(1, n_tokens, hk, dim).transpose(1, 2)
    rq, rk = original(q, k, cos, sin)
                                                                
    ((rq.to(torch.float16) * grad_q.float()).sum()
     + (rk.to(torch.float16) * grad_k.float()).sum()).backward()
    ref_gq, ref_gk = q_flat.grad.clone(), k_flat.grad.clone()
    q_flat.grad = k_flat.grad = None

    oq, ok = RoPE.apply(q, k, cos, sin)
    assert oq.dtype == torch.float16 and oq.is_contiguous() and oq.shape == (1, hq, n_tokens, dim)
    ((oq * grad_q).float().sum() + (ok * grad_k).float().sum()).backward()
    torch.testing.assert_close(oq.float(), rq.to(torch.float16).float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(ok.float(), rk.to(torch.float16).float(), rtol=1e-3, atol=2e-3)
                                                                                               
    for ours, ref in ((q_flat.grad, ref_gq), (k_flat.grad, ref_gk)):
        torch.testing.assert_close(ours.float(), ref.float(), rtol=1e-3,
                                   atol=1e-3 * float(ref.abs().max()))

    patch_qwen2_rope()
    pq, pk = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)
    assert pq.dtype == torch.float16              
    fq, fk = modeling_qwen2.apply_rotary_pos_emb(q.float(), k.float(), cos, sin)
    assert fq.dtype == torch.float32                                  


def test_rope_copies_inputs_whose_strides_are_not_whole_rows():
    """q/k views whose row stride is no multiple of the head dimension (rope._row_strided copies
    them before the kernel) give the values of contiguous inputs, forward and backward, bit for
    bit; the usual q/k views are not copied."""
    from solution import rope

    torch.manual_seed(14)
    n, hq, hk, dim, pad = 37, 12, 2, 128, 8
    inv = 1.0 / (1e6 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    freqs = torch.arange(n, device="cuda")[None].float()[..., None] * inv
    cos, sin = torch.cat((freqs, freqs), -1).cos(), torch.cat((freqs, freqs), -1).sin()
    wide = [torch.randn(1, n, h * dim + pad, device="cuda", dtype=torch.float16) for h in (hq, hk)]
    grads = [torch.randn(1, h, n, dim, device="cuda", dtype=torch.float16) for h in (hq, hk)]
    results = {}
    for strided in (True, False):
        flats = [w[..., :h * dim] for w, h in zip(wide, (hq, hk))]
        flats = [(f if strided else f.contiguous()).detach().requires_grad_(True) for f in flats]
        q, k = (f.view(1, n, h, dim).transpose(1, 2) for f, h in zip(flats, (hq, hk)))
        assert (rope._row_strided(q, dim) is q) is not strided                            
        oq, ok = rope.RoPE.apply(q, k, cos, sin)
        ((oq * grads[0]).float().sum() + (ok * grads[1]).float().sum()).backward()
        results[strided] = (oq, ok, flats[0].grad, flats[1].grad)
    for a, b in zip(results[True], results[False]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("n_tokens", [1, 37, 700])
def test_rope_backward_token_major_is_exact_and_copy_free(monkeypatch, n_tokens):
    """The backward written token-major and the strides passed in whole rows (UNIT = D): the
    same outputs and input gradients bit for bit as head-major with element strides, and the
    gradient reaching q/k is the transpose of a contiguous [1, N, H, D] tensor, so the view
    back to the projection output [1, N, H * D] copies nothing."""
    from solution import rope
    from solution.rope import RoPE

    torch.manual_seed(12)
    hq, hk, dim = 12, 2, 128
    cos = torch.randn(1, n_tokens, dim, device="cuda")
    sin = torch.randn(1, n_tokens, dim, device="cuda")
    grad_q = torch.randn(1, n_tokens, hq, dim, device="cuda", dtype=torch.float16).transpose(1, 2)
    grad_k = torch.randn(1, n_tokens, hk, dim, device="cuda", dtype=torch.float16).transpose(1, 2)
    results = {}
    for token_major, rows in ((False, False), (True, True), (True, False), (False, True)):
        monkeypatch.setattr(rope, "TOKEN_MAJOR_GRAD", token_major)
        monkeypatch.setattr(rope, "ROW_STRIDES", rows)
        torch.manual_seed(13)
        q_flat = torch.randn(1, n_tokens, hq * dim, device="cuda", dtype=torch.float16,
                             requires_grad=True)
        k_flat = torch.randn(1, n_tokens, hk * dim, device="cuda", dtype=torch.float16,
                             requires_grad=True)
        q = q_flat.view(1, n_tokens, hq, dim).transpose(1, 2)
        k = k_flat.view(1, n_tokens, hk, dim).transpose(1, 2)
        seen = []
        q.register_hook(seen.append)                                               
        k.register_hook(seen.append)
        oq, ok = RoPE.apply(q, k, cos, sin)
        assert oq.is_contiguous() and ok.is_contiguous()                                
        torch.autograd.backward((oq, ok), (grad_q, grad_k))
        results[token_major, rows] = (oq.detach(), ok.detach(), q_flat.grad, k_flat.grad, seen)
    reference = results[False, False]
    for key, result in results.items():
        for a, b in zip(reference[:4], result[:4]):
            assert torch.equal(a, b), key
        if key[0]:
            assert all(g.transpose(1, 2).is_contiguous() for g in result[4])


                                                                                              

def _fake_layer(hidden=256, inter=512, kv=64):
    import bitsandbytes as bnb
    from types import SimpleNamespace

    def lin(out_f, in_f, bias):
        layer = bnb.nn.Linear4bit(in_f, out_f, bias=bias, quant_type="nf4",
                                  compress_statistics=True, compute_dtype=torch.float16)
        layer.weight = bnb.nn.Params4bit((torch.randn(out_f, in_f) * 0.05).half(),
                                         requires_grad=False, quant_type="nf4",
                                         compress_statistics=True)
        return layer.cuda().requires_grad_(False)

    return SimpleNamespace(
        self_attn=SimpleNamespace(q_proj=lin(hidden, hidden, True), k_proj=lin(kv, hidden, True),
                                  v_proj=lin(kv, hidden, True), o_proj=lin(hidden, hidden, False)),
        mlp=SimpleNamespace(gate_proj=lin(inter, hidden, False), up_proj=lin(inter, hidden, False),
                            down_proj=lin(hidden, inter, False)))


@pytest.mark.parametrize("seed", [3, 4])
def test_group_dequant_is_bit_exact(seed):
    import bitsandbytes.functional as bnbf

    from solution.qgroup import PROJECTIONS, LayerGroup

    torch.manual_seed(seed)
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    modules = []
    for path in PROJECTIONS:
        obj = layer
        for part in path.split("."):
            obj = getattr(obj, part)
        modules.append(obj)
    refs = [bnbf.dequantize_4bit(m.weight.data, m.weight.quant_state).half() for m in modules]
    group = LayerGroup(modules)
    full = group.dequantize_all()
    for idx, ref in enumerate(refs):
        ours = full[group.elem_bounds[idx]:group.elem_bounds[idx + 1]].view(group.shapes[idx])
        assert torch.equal(ours, ref), f"projection {idx} differs from bitsandbytes"


def test_group_dequant_matches_bitsandbytes_and_launches_once_per_pass(monkeypatch):
    import bitsandbytes.functional as bnbf

    from solution import qgroup
    from solution.qgroup import PROJECTIONS, GroupLinear, LayerGroup, setup_layer

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 0, "bwd": 0})                     
    torch.manual_seed(12)
    layer = _fake_layer()
    modules = []
    for path in PROJECTIONS:
        obj = layer
        for part in path.split("."):
            obj = getattr(obj, part)
        modules.append(obj)
    refs = [bnbf.dequantize_4bit(m.weight.data, m.weight.quant_state).half() for m in modules]
    setup_layer(layer)
    assert all(hasattr(m, "_solution_group") for m in modules), "layer was not packed"
    group = modules[0]._solution_group[0]
                                                                                        
    for m, ref in zip(modules, refs):
        again = bnbf.dequantize_4bit(m.weight.data, m.weight.quant_state).half()
        assert torch.equal(again, ref)
    launches = {"n": 0}
    original = LayerGroup.dequantize_all

    def counted(self):
        launches["n"] += 1
        return original(self)

    LayerGroup.dequantize_all = counted
    try:
        x = torch.randn(3, 40, 256, device="cuda", dtype=torch.float16, requires_grad=True)
        h = torch.randn(3, 40, 512, device="cuda", dtype=torch.float16, requires_grad=True)
        inputs = [x] * 6 + [h]
        outs = []
        for i, inp in enumerate(inputs):                                          
            group.expected_bwd += 1
            outs.append(GroupLinear.apply(inp, group, i))
        assert launches["n"] == 1 and group.cache["fwd"][0] is None                               
        ref_outs = [F.linear(inp, ref, None if m.bias is None else m.bias.half())
                    for inp, ref, m in zip([x] * 6 + [h], refs, modules)]
        for o, r in zip(outs, ref_outs):                                         
            assert torch.equal(o, r)
        loss = sum((o.float() ** 2).mean() for o in outs)
        loss.backward()
        assert launches["n"] == 2 and group.cache["bwd"][0] is None                          
                                                            
        x2 = x.detach().clone().requires_grad_(True)
        h2 = h.detach().clone().requires_grad_(True)
        ref_loss = sum((F.linear(inp, ref, None if m.bias is None else m.bias.half()).float() ** 2)
                       .mean() for inp, ref, m in zip([x2] * 6 + [h2], refs, modules))
        ref_loss.backward()
        assert torch.equal(x.grad, x2.grad) and torch.equal(h.grad, h2.grad)
                                                                        
        with torch.no_grad():
            for i in range(7):
                GroupLinear.apply(x if i < 6 else h, group, i)
        assert group.cache["fwd"][0] is None and group.expected_bwd == 0
        assert launches["n"] == 3
    finally:
        LayerGroup.dequantize_all = original


                                                                                              

def _compiled(fn):
    """{specialization key: compiled kernel} of a @triton.jit function on the current device."""
    caches = fn.device_caches.get(torch.cuda.current_device())
    return dict(caches[0]) if caches else {}


def _warm_up_fp16():
    """The warm-up's compiles on the T4 path (fp16 autocast), on any GPU, in this thread."""
    from solution import warmup

    original = warmup._half_dtype
    warmup._half_dtype = lambda: torch.float16
    try:
        warmup.compile_all()
    finally:
        warmup._half_dtype = original


def test_rmsnorm_forward_and_backward_are_one_specialization():
    from solution import rmsnorm, warmup

    warmup.join()                                                             
    rmsnorm.rmsnorm_kernel.device_caches.clear()
    _warm_up_fp16()
    assert len(_compiled(rmsnorm.rmsnorm_kernel)) == 1                                     


def test_warmup_compiles_exactly_the_training_specializations():
    """After the warm-up, the training code paths (T4 dtypes: fp32 residual, fp16 activations;
    bitsandbytes NF4 with double quantization, whole-layer and per-projection slice launches;
    the vocabulary-sliced head over several slices; the example's SwiGLU kernels through our
    in-place wrapper and through fused_proj's MLP node) compile nothing new: the warm-up built
    the very specializations the training steps use."""
    from solution import fp16_embed, fused_head, qgroup, rmsnorm, rope, swiglu, warmup
    from solution.fused_head import fused_head_loss
    from solution.qgroup import PROJECTIONS, setup_layer
    from solution.rope import RoPE
    from solution.swiglu_inplace import SwiGLUInPlaceGrad

    head = (fused_head.fused_head_vocab_kernel if fused_head.VOCAB_PATH
            else fused_head.fused_head_rows_kernel)
    kernels = (rmsnorm.rmsnorm_kernel, head, rope.rope_kernel, qgroup.nf4_group_dequant_kernel,
               swiglu.example_swiglu_fwd_kernel, swiglu.example_swiglu_bwd_kernel)
    warmup.join()                                                             
    for fn in kernels:
        fn.device_caches.clear()
    _warm_up_fp16()
    warmed = {fn.fn.__name__: set(_compiled(fn)) for fn in kernels}
    assert all(len(keys) == 1 for keys in warmed.values()), warmed

    torch.manual_seed(21)
                                                                                               
    x = torch.randn(1, 300, 1536, device="cuda", requires_grad=True)
    w = torch.ones(1536, device="cuda")
    y = rmsnorm.RMSNorm.apply(x, w, 1e-6, torch.float16)
    y.backward(torch.randn_like(y))
                                                                                       
    branch = torch.randn(1, 300, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    total, y = rmsnorm.AddRMSNorm.apply(x, branch, w, 1e-6, torch.float16)
    (total.sum() + y.float().sum()).backward()
                                                                       
    q = torch.randn(1, 300, 12 * 128, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 300, 2 * 128, device="cuda", dtype=torch.float16, requires_grad=True)
    cos = torch.randn(1, 300, 128, device="cuda")
    oq, ok = RoPE.apply(q.view(1, 300, 12, 128).transpose(1, 2),
                        k.view(1, 300, 2, 128).transpose(1, 2), cos, cos.sin())
    (oq.float().sum() + ok.float().sum()).backward()
                                                                                                
    hidden = torch.randn(300, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, 1000, (300,), device="cuda")
    targets[::7] = -100
                                                                                             
                                                           
    weight = (torch.randn(1000, 1536, device="cuda") * 0.02).to(
        fp16_embed.head_weight_dtype(torch.float16))
    fused_head_loss(hidden, weight, targets, vocab_chunk=384)[0].backward()
                                                                                           
    gate = torch.randn(300, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    swiglu.swiglu(gate, up).float().sum().backward()
    SwiGLUInPlaceGrad.apply(gate.detach().clone().requires_grad_(True),
                            up.detach().clone().requires_grad_(True)).float().sum().backward()
                                                                                          
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    setup_layer(layer)
    obj = layer
    for part in PROJECTIONS[0].split("."):
        obj = getattr(obj, part)
    group = obj._solution_group[0]
    group.dequantize_all()
    assert group.splittable
    for idx in range(qgroup.NSEG):
        group.dequantize_one(idx)
                                                                                            
                                                                                          
    from solution.fused_proj import FusedMLP

    xm = torch.randn(1, 301, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    FusedMLP.apply(xm, group, True).float().sum().backward()
                                                                                           
    assert not qgroup.LayerGroup(
        [getattr(getattr(_fake_layer(), p.split(".")[0]), p.split(".")[1])
         for p in PROJECTIONS]).splittable
    torch.cuda.synchronize()
    after = {fn.fn.__name__: set(_compiled(fn)) for fn in kernels}
    assert after == warmed, {name: after[name] - warmed[name] for name in after}


def test_unpackable_layer_installs_the_per_projection_dequant_once():
    import bitsandbytes as bnb

    from solution import nf4_dequant, qgroup

    calls = []
    original = nf4_dequant.patch_bnb_dequant
    nf4_dequant.patch_bnb_dequant = lambda *a, **k: calls.append(1) or True
    saved = qgroup.FALLBACK_DEQUANT
    try:
        qgroup.FALLBACK_DEQUANT = True
        qgroup.setup_layer(_fake_layer())                                     
        assert calls == []
        layer = torch.nn.Module()
        layer.proj = bnb.nn.Linear4bit(64, 64, bias=False, quant_type="nf4",
                                       compute_dtype=torch.float16).cuda()
        qgroup.setup_layer(layer)                                                            
        assert calls == [1]
        qgroup.FALLBACK_DEQUANT = False
        other = torch.nn.Module()
        other.proj = layer.proj
        qgroup.setup_layer(other)
        assert calls == [1]
    finally:
        nf4_dequant.patch_bnb_dequant = original
        qgroup.FALLBACK_DEQUANT = saved


def test_triton_start_up_options():
    """-O0 helper builds go through Triton's own build hook; Triton's cache key is left alone."""
    from triton import knobs
    from triton.runtime import cache

    from solution import triton_build

    assert triton_build.install() and knobs.build.impl is triton_build._build
    assert cache.triton_key.__module__.startswith("triton")


def test_warmup_is_on_by_default(monkeypatch):
    import solution

    monkeypatch.delenv("SOLUTION_WARMUP_THREAD", raising=False)
    assert solution._warmup_mode() == "1"                                                
    monkeypatch.setenv("SOLUTION_WARMUP_THREAD", "0")
    assert solution._warmup_mode() == "0"
    monkeypatch.setenv("SOLUTION_WARMUP_THREAD", "late")
    assert solution._warmup_mode() == "late"


@pytest.mark.parametrize("depth,expected", [(0, {"all": 0, "one": 14}),
                                            (1, {"all": 1, "one": 7}),
                                            (2, {"all": 2, "one": 0}),
                                            (None, {"all": 2, "one": 0})])
def test_split_dequant_on_the_last_layers(monkeypatch, depth, expected):
    """In the forward of the last layer and the backward of the last two, each projection is
    dequantized alone: the same fp16 weights bit for bit (a slice launch of the same kernel),
    the same outputs and gradients, and the bookkeeping is reset for the next step."""
    from solution import qgroup
    from solution.qgroup import PROJECTIONS, GroupLinear, LayerGroup, setup_layer

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    torch.manual_seed(13)
                                                                                        
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    modules = []
    for path in PROJECTIONS:
        obj = layer
        for part in path.split("."):
            obj = getattr(obj, part)
        modules.append(obj)
    setup_layer(layer)
    group = modules[0]._solution_group[0]
    full = group.dequantize_all()
    for idx in range(7):
        ours = group.dequantize_one(idx)
        ref = full[group.elem_bounds[idx]:group.elem_bounds[idx + 1]].view(group.shapes[idx])
        assert torch.equal(ours, ref)

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
    x = torch.randn(3, 40, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    h = torch.randn(3, 40, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
    results = []
    for group_depth in (depth, None):                                    
        group.depth = group_depth
        for _ in range(2):                                                       
            calls.update(all=0, one=0)
            outs = []
            for i, inp in enumerate([x] * 6 + [h]):
                group.expected_bwd += 1
                outs.append(GroupLinear.apply(inp, group, i))
            sum((o.float() ** 2).mean() for o in outs).backward()
            assert group.split_left == 0 and group.expected_bwd == 0
            assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
        results.append((calls.copy(), [o.detach() for o in outs], x.grad.clone(), h.grad.clone()))
        x.grad = h.grad = None
    assert results[0][0] == expected
    for o, r in zip(results[0][1], results[1][1]):
        assert torch.equal(o, r)
    assert torch.equal(results[0][2], results[1][2]) and torch.equal(results[0][3], results[1][3])


                                                                                              

def _metric_inputs():
    from types import SimpleNamespace

    torch.manual_seed(13)
    batch, seq = 4, 50
    steps = []
    for labelled in (37, 0, 112):                                                  
        mask = (torch.arange(seq)[None] < torch.randint(1, seq + 1, (batch, 1))).long().cuda()
        steps.append(({"attention_mask": mask}, SimpleNamespace(
            token_entropy=torch.rand(batch, seq, device="cuda") * 3,
            correct=torch.tensor(min(labelled, 23), device="cuda"),
            labelled=torch.tensor(labelled, device="cuda"))))
    return steps


def test_deferred_trl_metrics_match_immediate():
    """TRL's per-step metrics fetched when they are logged are exactly the values the immediate
    path appends: same keys in the same order, same floats, same running token count."""
    import collections
    from types import SimpleNamespace

    from solution import host, patches

    steps = _metric_inputs()
    saved = patches._host["defer"]
    results = []
    try:
        for defer in (False, True):
            patches._host["defer"] = defer
            trainer = SimpleNamespace(
                _metrics={"train": collections.defaultdict(list),
                          "eval": collections.defaultdict(list)},
                _total_train_tokens=5, accelerator=SimpleNamespace(gather_for_metrics=lambda t: t))
            for inputs, step in steps:
                patches._log_trl_metrics(trainer, "train", inputs, step)
            if defer:
                assert not trainer._metrics["train"]                                           
                host.resolve_metrics()
            results.append((list(trainer._metrics["train"].items()), trainer._total_train_tokens))
    finally:
        patches._host["defer"] = saved
    assert results[0] == results[1]
    assert results[1][0][2][1][1] == 0.0                                    


def test_deferred_rewind_records_match_soup():
    """Soup's rewind recorder with the row losses left on the GPU pops the same row ids, numbers
    the micro-batches the same way and writes the same records, in the same order."""
    pytest.importorskip("soup_cli")
    from soup_cli.trainer import rewind_hf

    from solution import host

    host._patch_rewind()
    rows_fn = rewind_hf.row_losses_from_token_losses
    original_rows, original_micro = rows_fn._original, rewind_hf.RewindState.on_micro_batch._original

    class Sink:
        def __init__(self):
            self.records = []

        def record_batch(self, **kwargs):
            self.records.append(kwargs)

    torch.manual_seed(14)
    batches = []
    for step in (1, 1, 2, 3):                               
        labels = torch.randint(0, 100, (4, 30), device="cuda")
        labels[:, :5] = -100
        labels[2, 18:] = -100
        labels[3] = -100                                  
        token_loss = torch.rand(4, 30, device="cuda")
        token_loss[F.pad(labels, (0, 1), value=-100)[:, 1:] < 0] = 0
        batches.append((step, labels, token_loss))

    states = []
    for deferred in (False, True):
        state = rewind_hf.RewindState(Sink())
        for index in range(15):                                                  
            state.on_index(index)
        for step, labels, token_loss in batches:
            if deferred:
                host.mark_fused_token_loss(token_loss)
                row_loss, row_tokens = rows_fn(token_loss, labels)
                assert isinstance(row_loss, host.LazyRows)
                state.on_micro_batch(step=step, row_loss=row_loss, row_tokens=row_tokens)
            else:
                row_loss, row_tokens = original_rows(token_loss, labels)
                original_micro(state, step=step, row_loss=row_loss, row_tokens=row_tokens)
        if deferred:
            assert not state._sink.records
            host.flush_rewind()
        states.append(state)
    ref, ours = states
    assert ours._sink.records == ref._sink.records and len(ref._sink.records) == 3
    assert ours.dropped == ref.dropped == 1 and list(ours._pending) == list(ref._pending)
    host.mark_fused_token_loss(None)                                                  
    assert isinstance(rows_fn(batches[0][2], batches[0][1])[0], list)


def test_flush_releases_the_last_fused_token_loss():
    """The log's flush leaves no tensor of the step referenced by host.py: Soup's rewind
    recorder read the fused step's token losses before it, and the reference would otherwise
    keep the last step's tensor alive after the run."""
    import weakref

    from solution import host

    token_loss = torch.zeros(4, 8, device="cuda")
    host.mark_fused_token_loss(token_loss)
    ref = weakref.ref(token_loss)
    del token_loss
    host.flush_all()
    assert host._fused_token_loss[0] is None and ref() is None


def test_trainer_bookkeeping_fastpaths_are_exact():
    """Flat-list train()/zero_grad()/clip and the cached parameter count give what the
    module-tree walks give."""
    from types import SimpleNamespace

    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution import host

    def make():
        torch.manual_seed(15)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Sequential(
            torch.nn.Linear(8, 4), torch.nn.Dropout(0.1)), torch.nn.Linear(4, 2)).cuda()
        model[0].requires_grad_(False)
        return model

    ref, fast = make(), make()
    host.install_model_fastpaths(fast)
    x = torch.randn(5, 8, device="cuda")
    for model in (ref, fast):
        torch.manual_seed(16)
        model(x).square().sum().backward()
    host.trainer_bookkeeping()
    clip = transformers.Trainer._clip_grad_norm
    accelerator = SimpleNamespace(clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    args = SimpleNamespace(max_grad_norm=0.5, fp16=False)
    norm_ref = clip._original(SimpleNamespace(args=args, accelerator=accelerator), ref)
    norm_fast = clip(SimpleNamespace(args=args, accelerator=accelerator), fast)
    assert torch.equal(norm_ref, norm_fast)
    for a, b in zip(ref.parameters(), fast.parameters()):
        assert (a.grad is None) == (b.grad is None)
        assert a.grad is None or torch.equal(a.grad, b.grad)
    fast.zero_grad()
    assert all(p.grad is None for p in fast.parameters())
    fast.train()
    fast[1][1].eval()                                                 
    fast.train()
    assert all(m.training for m in fast.modules())
    fast.eval()
    assert not any(m.training for m in fast.modules())
    fast.train()
    assert all(m.training for m in fast.modules())

    config = Qwen2Config(vocab_size=600, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    trainer = SimpleNamespace(model=Qwen2ForCausalLM(config))
    inputs = {"input_ids": torch.ones(3, 17, dtype=torch.long)}
    flops = transformers.Trainer.floating_point_ops
    assert flops(trainer, inputs) == flops._original(trainer, inputs) == flops(trainer, inputs)


