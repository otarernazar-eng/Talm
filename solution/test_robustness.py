"""Robustness on batches the public rows do not contain.

The jury trains and evaluates on hidden rows of the same dataset with another seed, so every
part of the step must give the values of the path it replaces (or fall back to it cleanly) on
any batch TRL can build from them, not only on the public rows. Covered here, on synthetic
tensors:

- the patched causal LM end to end (padding removal, variable-length or per-row attention, the
  fused head with its labelled-rows compaction) against the original forward on edge batches:
  a row with no labelled token, all rows of one length (no padding at all), a 1-token row,
  rows of the full window, no labelled token in the whole batch (num_items_in_batch = 0), every
  position labelled, a batch of one row, token counts that are no multiple of any block size,
  and masks that are not right-padded (left padding, a hole), where the padding removal must
  switch itself off instead of crashing;
- the fused head, the variable-length attention, the LoRA node, add+norm, the split
  dequantization and the deferred metrics on the same shapes.

Soup itself rejects a row whose response is fully truncated (`ensure_causal_loss_target`), so in
a real run every row keeps at least one labelled target; the zero-label rows and batches below
are stricter than any jury batch can be.

Run on a GPU: `python -m pytest solution -q`.
"""

import collections
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

VOCAB = 600


# --- the patched causal LM end to end -------------------------------------------------------

def _tiny_lm(seed=5):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.test_kernels import apply_quietly

    apply_quietly()
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=1024, tie_word_embeddings=False,
                         attn_implementation="sdpa")
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(config).cuda().train()
    model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)  # a gradient through every layer
    return model


def _batch(lengths, seq=None, prompt=4, left=False, labelled_rows=None, all_labelled=False,
           hole=False, seed=0):
    """input_ids, attention_mask and labels as TRL's collator builds them (right padding, -100
    on the prompt and the padding) for rows of the given lengths; `left` pads on the left,
    `hole` masks one position inside the first row, `labelled_rows` keeps labels only on those
    rows, `all_labelled` labels every real token."""
    gen = torch.Generator().manual_seed(seed)
    seq = seq or max(lengths)
    batch = len(lengths)
    input_ids = torch.randint(0, VOCAB, (batch, seq), generator=gen)
    mask = torch.zeros(batch, seq, dtype=torch.long)
    labels = torch.full((batch, seq), -100, dtype=torch.long)
    for b, length in enumerate(lengths):
        cols = slice(seq - length, seq) if left else slice(0, length)
        mask[b, cols] = 1
        row = input_ids[b, cols].clone()
        if not all_labelled:
            row[:prompt] = -100
        if labelled_rows is not None and b not in labelled_rows:
            row[:] = -100
        labels[b, cols] = row
    if hole:
        mask[0, 5] = 0
        labels[0, 5] = -100
    input_ids[mask == 0] = VOCAB - 1  # the pad token
    return input_ids.cuda(), mask.cuda(), labels.cuda()


CASES = {
    # name: (lengths, keyword arguments of _batch)
    "row_without_labels": ([24, 9, 17, 20], {"labelled_rows": {0, 1, 3}}),
    "equal_lengths_no_padding": ([20, 20, 20, 20], {}),
    "one_token_row": ([24, 1, 17, 5], {}),
    "full_window_rows": ([512, 512, 512, 512], {"prompt": 500}),
    "full_and_short_rows": ([512, 2, 512, 7], {"prompt": 1}),
    "no_label_in_batch": ([24, 9, 17, 20], {"labelled_rows": set()}),
    "every_position_labelled": ([24, 9, 17, 20], {"all_labelled": True}),
    "single_row": ([37], {}),
    "single_one_token_row": ([1], {"all_labelled": True}),
    "odd_token_counts": ([17, 3, 29, 1], {"prompt": 2}),
    "left_padded": ([24, 9, 17, 20], {"left": True}),
    "mask_with_hole": ([24, 9, 17, 20], {"hole": True}),
}


def _reference(model, input_ids, mask, labels, num_items):
    """The original forward (padded batch, SDPA with the mask, full logits), its loss gradient
    and the per-token values TRL and Soup read from its logits."""
    from transformers import Qwen2ForCausalLM

    from solution import patches

    original = patches.ORIGINAL_FORWARD[Qwen2ForCausalLM]
    model.model.embed_tokens.weight.grad = None
    with torch.autocast("cuda", dtype=torch.float16):
        out = original(model, input_ids=input_ids, attention_mask=mask, labels=labels,
                       num_items_in_batch=num_items)
    out.loss.backward()
    batch, seq = labels.shape
    shift = F.pad(labels, (0, 1), value=-100)[:, 1:]
    logits = out.logits.detach().float()
    token_loss = F.cross_entropy(logits.reshape(-1, VOCAB), shift.reshape(-1), ignore_index=-100,
                                 reduction="none").view(batch, seq)
    lse = torch.logsumexp(logits, -1)
    entropy = lse - (torch.softmax(logits, -1) * logits).sum(-1)
    correct = ((out.logits.detach().argmax(-1) == shift) & (shift >= 0)).sum()
    return (out.loss.detach(), token_loss, entropy, correct, int((shift >= 0).sum()),
            model.model.embed_tokens.weight.grad.clone())


@pytest.mark.parametrize("attention", ["varlen", "rows"])
@pytest.mark.parametrize("case", list(CASES))
def test_patched_forward_matches_original_on_edge_batches(monkeypatch, case, attention):
    """Loss, per-token loss (Soup's rewind), entropy and accuracy (TRL), the labelled count and
    the input gradient of the patched forward equal the original forward's on each edge batch
    (fp16 tolerances: the two paths round differently), the gradient is finite (zeros where
    nothing is labelled, as the base gives), and a mask that is not right-padded switches the
    padding removal off instead of failing."""
    from solution import patches, unpad

    lengths, kwargs = CASES[case]
    model = _tiny_lm()
    if attention == "varlen":  # the T4 path, forced on any GPU
        monkeypatch.setenv("SOLUTION_VARLEN", "1")
        monkeypatch.setattr(unpad, "_VARLEN", True)
    else:
        monkeypatch.setattr(unpad, "_VARLEN", False)
    monkeypatch.setitem(patches._unpad, "enabled", True)
    monkeypatch.setitem(patches._unpad, "checked", 0)  # the right-padding check runs
    calls = collections.Counter()
    for name in ("VarlenRowsAttention", "RowsAttention"):
        cls = getattr(unpad, name)
        monkeypatch.setattr(cls, "apply",
                            (lambda fn, key: lambda *a: calls.update([key]) or fn(*a))(
                                cls.apply, name))
    input_ids, mask, labels = _batch(lengths, **kwargs)
    num_items = (labels[:, 1:] != -100).sum()
    ref_loss, ref_token, ref_entropy, ref_correct, n_labelled, ref_grad = _reference(
        model, input_ids, mask, labels, num_items)

    model.model.embed_tokens.weight.grad = None
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=input_ids, attention_mask=mask, labels=labels,
                    num_items_in_batch=num_items)
    assert out.logits is None
    step = patches._pending_step[-1]
    out.loss.backward()
    grad = model.model.embed_tokens.weight.grad

    right_padded = not (kwargs.get("left") or kwargs.get("hole"))
    assert patches._unpad["enabled"] is right_padded  # off for good after such a batch
    expected = {"varlen": "VarlenRowsAttention", "rows": "RowsAttention"}[attention]
    assert calls[expected] == (model.config.num_hidden_layers if right_padded else 0)
    assert int(step.labelled) == n_labelled
    if n_labelled == 0:
        assert torch.isnan(out.loss) and torch.isnan(ref_loss)  # 0 / 0, as the base
    else:
        torch.testing.assert_close(out.loss.float(), ref_loss.float(), rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(step.token_loss, ref_token, rtol=3e-2, atol=2e-2)
    assert (step.token_loss[F.pad(labels, (0, 1), value=-100)[:, 1:] < 0] == 0).all()
    torch.testing.assert_close(step.token_entropy * mask, ref_entropy * mask, rtol=3e-2,
                               atol=2e-2)
    assert abs(int(step.correct) - int(ref_correct)) <= max(1, n_labelled // 50)
    assert torch.isfinite(grad).all(), "non-finite input gradient"
    if n_labelled == 0:
        assert (ref_grad == 0).all() and (grad == 0).all()
    else:
        torch.testing.assert_close(grad, ref_grad, rtol=3e-2,
                                   atol=3e-2 * float(ref_grad.abs().max()))


@pytest.fixture(scope="module")
def tiny_qlora(tmp_path_factory):
    """A 2-layer Qwen2 in the task's setup: NF4 4-bit projections with double quantization and
    fp16 compute (bitsandbytes), prepare_model_for_kbit_training, PEFT LoRA r=16 / alpha=32 on
    q_proj and v_proj, tied frozen embeddings. Sizes chosen so that qgroup packs both layers and
    splits them (the last layer's forward and both backwards dequantize per projection), with
    GQA (8 query heads on 2 KV heads). LoRA dropout 0 so both paths draw nothing."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2Config, Qwen2ForCausalLM

    from solution.test_kernels import apply_quietly

    apply_quietly()
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=512, intermediate_size=1024,
                         num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                         max_position_embeddings=1024, tie_word_embeddings=True,
                         attn_implementation="sdpa")
    torch.manual_seed(6)
    path = tmp_path_factory.mktemp("tiny_qwen2")
    Qwen2ForCausalLM(config).save_pretrained(path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2ForCausalLM.from_pretrained(path, quantization_config=quant, device_map={"": 0},
                                             dtype=torch.float32, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    torch.manual_seed(7)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)  # PEFT starts B at 0: no gradient for A
    model.train()
    return model


QLORA_CASES = ["row_without_labels", "equal_lengths_no_padding", "one_token_row",
               "full_and_short_rows", "no_label_in_batch", "single_row", "odd_token_counts",
               "left_padded"]


@pytest.mark.parametrize("case", QLORA_CASES)
def test_qlora_forward_matches_original_on_edge_batches(monkeypatch, tiny_qlora, case):
    """The task's module stack (packed NF4 layers with split dequantization, LoRA nodes,
    add+norm, RoPE, padding removal, variable-length attention, fused head) on the edge
    batches: the same loss, per-token losses and LoRA gradients as the original causal-LM
    forward on the padded batch, finite gradients, and the per-layer dequantization
    bookkeeping back to rest after every step."""
    from solution import lora, patches, qgroup, unpad

    model = tiny_qlora
    inner = model.get_base_model()
    monkeypatch.setenv("SOLUTION_VARLEN", "1")
    monkeypatch.setattr(unpad, "_VARLEN", True)
    monkeypatch.setitem(patches._unpad, "enabled", True)
    monkeypatch.setitem(patches._unpad, "checked", 0)
    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    from solution import fused_proj

    calls = collections.Counter()
    node = lora.LoraProjectionTorch.apply
    monkeypatch.setattr(lora.LoraProjectionTorch, "apply",
                        lambda *a: calls.update(["lora"]) or node(*a))
    fused = fused_proj.FusedProjections.apply  # q|k|v as one node (fused_proj.py)
    monkeypatch.setattr(fused_proj.FusedProjections, "apply",
                        lambda *a: calls.update(["qkv"]) or fused(*a))
    lengths, kwargs = CASES[case]
    input_ids, mask, labels = _batch(lengths, **kwargs)
    num_items = (labels[:, 1:] != -100).sum()
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    assert params and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)

    def run(fn):
        for p in params:
            p.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            out = fn()
        out.loss.backward()
        groups = {m._solution_group[0] for m in inner.modules() if hasattr(m, "_solution_group")}
        assert len(groups) == 2 and all(g.splittable for g in groups)
        for g in groups:  # nothing cached or pending between steps
            assert g.cache["fwd"][0] is None and g.cache["bwd"][0] is None
            assert g.expected_bwd == 0 and g.split_left == 0
        return out, [p.grad.clone() for p in params]

    original = patches.ORIGINAL_FORWARD[type(inner)]
    ref, ref_grads = run(lambda: original(inner, input_ids=input_ids, attention_mask=mask,
                                          labels=labels, num_items_in_batch=num_items))
    calls.clear()
    out, grads = run(lambda: model(input_ids=input_ids, attention_mask=mask, labels=labels,
                                   num_items_in_batch=num_items))
    # q and v of both layers took the LoRA node, or the fused q|k|v node (which runs both
    # adapters) where the layer's forward dequantizes the whole layer (not the last one)
    assert out.logits is None and calls["lora"] + 2 * calls["qkv"] == 4
    step = patches._pending_step[-1]
    shift = F.pad(labels, (0, 1), value=-100)[:, 1:]
    ref_token = F.cross_entropy(ref.logits.float().reshape(-1, VOCAB), shift.reshape(-1),
                                ignore_index=-100, reduction="none").view(labels.shape)
    n_labelled = int((shift >= 0).sum())
    assert int(step.labelled) == n_labelled
    if n_labelled == 0:
        assert torch.isnan(out.loss) and torch.isnan(ref.loss)
    else:
        torch.testing.assert_close(out.loss.float(), ref.loss.float(), rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(step.token_loss, ref_token, rtol=3e-2, atol=2e-2)
    for g, r in zip(grads, ref_grads):
        assert torch.isfinite(g).all()
        if n_labelled == 0:
            assert (g == 0).all() and (r == 0).all()
        else:
            torch.testing.assert_close(g, r, rtol=3e-2, atol=3e-2 * float(r.abs().max()) + 1e-8)


def test_padding_removal_stays_off_and_correct_after_left_padding(monkeypatch):
    """After a left-padded batch the padding removal is off for the rest of the run; the next
    (right-padded) batch then takes the padded path through the fused head and still gives the
    original forward's values."""
    from solution import patches

    model = _tiny_lm()
    monkeypatch.setitem(patches._unpad, "enabled", True)
    monkeypatch.setitem(patches._unpad, "checked", 0)
    for left in (True, False):
        input_ids, mask, labels = _batch([24, 9, 17, 20], left=left, seed=int(left))
        num_items = (labels[:, 1:] != -100).sum()
        ref_loss, ref_token, _, _, n_labelled, ref_grad = _reference(model, input_ids, mask,
                                                                     labels, num_items)
        model.model.embed_tokens.weight.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels,
                        num_items_in_batch=num_items)
        out.loss.backward()
        step = patches._pending_step[-1]
        assert patches._unpad["enabled"] is False
        assert int(step.labelled) == n_labelled
        torch.testing.assert_close(out.loss.float(), ref_loss.float(), rtol=2e-2, atol=1e-3)
        torch.testing.assert_close(step.token_loss, ref_token, rtol=3e-2, atol=2e-2)
        grad = model.model.embed_tokens.weight.grad
        torch.testing.assert_close(grad, ref_grad, rtol=3e-2,
                                   atol=3e-2 * float(ref_grad.abs().max()))


def test_right_padding_check():
    from solution.unpad import is_right_padded, unpad_batch

    def check(rows):
        mask = torch.tensor(rows, device="cuda")
        _, _, bounds, lengths = unpad_batch(mask)
        assert bounds[-1][1] == sum(lengths) == int(mask.sum())
        return is_right_padded(mask, lengths)

    assert check([[1, 1, 1], [1, 0, 0], [1, 1, 0]])      # right padding
    assert check([[1, 1, 1], [1, 1, 1]])                 # no padding at all
    assert check([[1, 0, 0]])                            # one row
    assert check([[1, 1], [0, 0]])                       # an empty row
    assert not check([[1, 1, 1], [0, 1, 1]])             # left padding
    assert not check([[1, 0, 1], [1, 1, 1]])             # a hole
    assert not check([[0, 0, 1]])                        # one left-padded row


def test_row_bounds_with_empty_and_one_token_rows():
    """Row offsets for the variable-length kernel skip empty rows wherever they are."""
    from solution.unpad import _cu, rows_context, unpad_batch

    lengths = [0, 7, 1, 0, 9]
    mask = (torch.arange(9, device="cuda")[None]
            < torch.tensor(lengths, device="cuda")[:, None]).long()
    idx, positions, bounds, got = unpad_batch(mask)
    assert got == lengths and idx.numel() == 17
    assert positions.tolist()[0] == list(range(7)) + [0] + list(range(9))
    with rows_context(bounds, device="cuda"):
        cu, max_len, total = _cu[-1]
        assert cu.tolist() == [0, 7, 8, 17] and max_len == 9 and total == 17


# --- fused head -----------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["vocab", "tokens"])
@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("num_items", ["zero_tensor", "zero_int"])
def test_head_without_labels_and_zero_num_items_gives_zero_gradient(monkeypatch, path, compact,
                                                                     num_items):
    """No labelled token at all and num_items_in_batch = 0: the base's loss is 0 / 0 = NaN, and
    its gradient is exactly 0 (cross-entropy writes no gradient on ignored rows). The fused head
    must give the same NaN loss and the same zero gradient, not 0 * inf = NaN: a NaN gradient
    makes the GradScaler skip the step and halve its scale where the base steps."""
    from solution import fused_head
    from solution.fused_head import fused_head_loss

    monkeypatch.setattr(fused_head, "COMPACT", compact)
    torch.manual_seed(40)
    n, size, vocab = 37, 64, 8195
    hidden = torch.randn(n, size, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(vocab, size, device="cuda") * 0.05
    targets = torch.full((n,), -100, device="cuda")
    items = torch.tensor(0, device="cuda") if num_items == "zero_tensor" else 0
    kwargs = {"token_chunk": 16} if path == "tokens" else {"vocab_chunk": 4096,
                                                           "n_labelled": 0}

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    logits = (ref_hidden @ weight.half().t()).float()
    ref_loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum") / (
        items.float() if torch.is_tensor(items) else items)
    (ref_loss * 1024.0).backward()  # a GradScaler-sized factor

    loss, token_loss, entropy, correct = fused_head_loss(hidden, weight, targets, items, **kwargs)
    (loss * 1024.0).backward()
    assert torch.isnan(loss) and torch.isnan(ref_loss)
    assert (token_loss == 0).all() and not correct.any() and torch.isfinite(entropy).all()
    assert (ref_hidden.grad == 0).all()
    assert torch.isfinite(hidden.grad).all() and (hidden.grad == 0).all()


@pytest.mark.parametrize("path", ["vocab", "tokens"])
@pytest.mark.parametrize("n_tokens,labelled", [(1, 1.0), (1, 0.0), (2, 0.5), (3, 1.0),
                                               (4097, 0.6), (2048, 1.0)])
def test_head_odd_sizes_and_label_fractions(path, n_tokens, labelled):
    """1, 2, 3 and 4097 rows, a vocabulary whose last slice is 3 columns wide, 0% and 100%
    labelled, the labelled count given exactly as the padding removal gives it: the base's
    loss, per-token values and gradient."""
    from solution.fused_head import fused_head_loss
    from solution.test_kernels import boundary_columns, plant_hits, reference_head

    torch.manual_seed(41)
    size, vocab = 64, 8195
    hidden = (torch.randn(n_tokens, size, device="cuda") * 2).half()
    weight = torch.randn(vocab, size, device="cuda") * 0.05
    targets = torch.randint(0, vocab, (n_tokens,), device="cuda")
    targets[torch.rand(n_tokens, device="cuda") >= labelled] = -100
    rows = (targets >= 0).nonzero().flatten().tolist()  # hits and misses among the labelled
    planted = plant_hits(hidden, weight, targets, rows, boundary_columns(vocab, 4096) * 2, 40.0)
    hidden.requires_grad_(True)
    n_lab = int((targets >= 0).sum())
    items = torch.tensor(max(n_lab, 1), device="cuda")
    ref_loss, ref_token, ref_entropy, ref_correct = reference_head(hidden, weight, targets, items)
    (ref_loss * 512.0).backward()
    ref_grad = hidden.grad.clone()
    hidden.grad = None
    kwargs = {"token_chunk": 1000} if path == "tokens" else {"vocab_chunk": 4096,
                                                             "n_labelled": n_lab}
    loss, token_loss, entropy, correct = fused_head_loss(hidden, weight, targets, items, **kwargs)
    (loss * 512.0).backward()
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(token_loss, ref_token, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    assert int(ref_correct.sum()) >= planted  # the planted hits are hits
    assert torch.equal(correct, ref_correct)
    scale = max(float(ref_grad.abs().max()), 1e-6)
    torch.testing.assert_close(hidden.grad.float(), ref_grad.float(), rtol=2e-2, atol=2e-2 * scale)
    assert (hidden.grad[targets < 0] == 0).all()


# --- variable-length attention --------------------------------------------------------------

@pytest.mark.parametrize("lengths", [[512, 512, 512, 512], [1], [1, 1, 1, 1], [0, 7, 1, 0, 9],
                                     [3, 5, 2], [511, 1, 512, 13]])
def test_varlen_attention_on_edge_lengths(lengths):
    """One launch per KV head over rows of the full window, of one token, empty rows and odd
    lengths: the per-row kernel's output bit for bit, the block-diagonal causal attention, and
    gradients within the per-row reference's own run-to-run spread."""
    from solution.unpad import RowsAttention, VarlenRowsAttention, _cu, rows_context
    from solution.test_fusions import _row_bounds

    torch.manual_seed(42)
    bounds, n = _row_bounds(lengths)
    rows = [(s, e) for s, e in bounds if e > s]
    q = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    k, v = (torch.randn(1, n, 2, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
            for _ in range(2))
    grad = torch.randn(1, n, 12, 128, device="cuda", dtype=torch.float16)
    scale = 128 ** -0.5

    def run(fn):
        q1, k1, v1 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        out = fn(q1, k1, v1)
        return out.detach(), torch.autograd.grad(out, (q1, k1, v1), grad)

    def varlen(q1, k1, v1):
        with rows_context(bounds, device="cuda"):
            cu, max_len, total = _cu[-1]
            assert total == n and max_len == max(lengths)
            return VarlenRowsAttention.apply(q1, k1, v1, cu, max_len, scale)

    ref, ref_grads = run(lambda *t: RowsAttention.apply(*t, rows, scale))
    _, again = run(lambda *t: RowsAttention.apply(*t, rows, scale))
    out, grads = run(varlen)
    assert torch.equal(out, ref)
    for ours, theirs, theirs_again in zip(grads, ref_grads, again):
        spread = float((theirs.float() - theirs_again.float()).abs().max())
        torch.testing.assert_close(ours, theirs, rtol=2e-3,
                                   atol=2e-3 * float(theirs.abs().max()) + 2 * spread)
    row = torch.cat([torch.full((e - s,), i, device="cuda") for i, (s, e) in enumerate(rows)])
    pos = torch.cat([torch.arange(e - s, device="cuda") for s, e in rows])
    allowed = (row[:, None] == row[None, :]) & (pos[None, :] <= pos[:, None])
    dense = F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(6, 1),
                                           v.float().repeat_interleave(6, 1), attn_mask=allowed,
                                           scale=scale).transpose(1, 2)
    torch.testing.assert_close(out.float(), dense, rtol=1e-2, atol=1e-2)


# --- LoRA node and add+norm on the edge token counts ----------------------------------------

@pytest.mark.parametrize("n_tokens", [2, 3, 2048])
def test_lora_node_bit_identical_on_edge_token_counts(n_tokens):
    from solution.test_lora import test_lora_node_is_bit_identical_to_peft

    test_lora_node_is_bit_identical_to_peft(n_tokens, True)


@pytest.mark.parametrize("n_rows", [2, 3, 2048])
def test_add_norm_on_edge_token_counts(n_rows):
    from solution.test_fusions import test_add_rmsnorm_matches_pytorch

    test_add_rmsnorm_matches_pytorch(n_rows)


# --- split dequantization: bookkeeping when only part of a layer needs gradients -------------

@pytest.mark.parametrize("depth", [0, 1, None])
def test_split_dequant_when_the_attention_inputs_need_no_gradient(monkeypatch, depth):
    """As in the first decoder layer, where the norm output of the frozen embeddings needs no
    gradient: only o/gate/up/down ask for their weight in the backward. The per-projection
    (split) and whole-layer passes must count those uses, drop their buffers and reset for the
    next step, and give the same gradients; a forward without gradients between two steps (an
    evaluation) must leave no bookkeeping behind."""
    from solution import qgroup
    from solution.qgroup import PROJECTIONS, GroupLinear, setup_layer
    from solution.test_kernels import _fake_layer

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    torch.manual_seed(43)
    layer = _fake_layer(hidden=512, inter=1024, kv=128)
    modules = []
    for path in PROJECTIONS:
        obj = layer
        for part in path.split("."):
            obj = getattr(obj, part)
        modules.append(obj)
    setup_layer(layer)
    group = modules[0]._solution_group[0]
    assert group.splittable
    x = torch.randn(1, 37, 512, device="cuda", dtype=torch.float16)  # no gradient
    h = torch.randn(1, 37, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    m = torch.randn(1, 37, 1024, device="cuda", dtype=torch.float16, requires_grad=True)

    def step():
        outs = []
        for i, inp in enumerate([x, x, x, h, h, h, m]):
            if inp.requires_grad:
                group.expected_bwd += 1  # as the patched Linear4bit.forward does
            outs.append(GroupLinear.apply(inp, group, i))
        sum((o.float() ** 2).mean() for o in outs).backward()
        assert group.split_left == 0 and group.expected_bwd == 0
        assert group.cache["fwd"][0] is None and group.cache["bwd"][0] is None
        grads = (h.grad.clone(), m.grad.clone())
        h.grad = m.grad = None
        return grads

    results = []
    for group_depth in (depth, None):
        group.depth = group_depth
        first = step()
        with torch.no_grad():  # an evaluation forward in between
            for i, inp in enumerate([x, x, x, h, h, h, m]):
                GroupLinear.apply(inp, group, i)
        assert group.expected_bwd == 0 and group.cache["fwd"][0] is None
        results.append((first, step()))
    for (a1, a2), (b1, b2) in zip(*results):
        assert torch.equal(a1, b1) and torch.equal(a2, b2)


def _qlora_step_tools(monkeypatch, model):
    """The task's module stack on `tiny_qlora` with the padding removal on, every dequantization
    launch counted, and a helper that runs one step and checks that the layer groups are back to
    rest (nothing cached, nothing pending) after it."""
    from solution import patches, qgroup

    monkeypatch.setitem(patches._unpad, "enabled", True)
    monkeypatch.setitem(patches._unpad, "checked", 0)
    launches = collections.Counter()
    launch = qgroup.launch
    monkeypatch.setattr(qgroup, "launch", lambda *a, **k: launches.update(["n"]) or launch(*a, **k))
    inner = model.get_base_model()
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    input_ids, mask, labels = _batch([24, 9, 17, 20])
    num_items = (labels[:, 1:] != -100).sum()

    def groups():
        found = {m._solution_group[0] for m in inner.modules() if hasattr(m, "_solution_group")}
        assert len(found) == 2 and all(g.splittable for g in found)
        return found

    def forward(fn=None):
        with torch.autocast("cuda", dtype=torch.float16):
            if fn is not None:
                return fn(inner, input_ids=input_ids, attention_mask=mask, labels=labels,
                          num_items_in_batch=num_items)
            return model(input_ids=input_ids, attention_mask=mask, labels=labels,
                         num_items_in_batch=num_items)

    def step(fn=None):
        for p in params:
            p.grad = None
        launches.clear()
        out = forward(fn)
        out.loss.backward()
        for g in groups():  # nothing cached or pending between steps
            assert g.cache["fwd"][0] is None and g.cache["bwd"][0] is None
            assert g.expected_bwd == 0 and g.split_left == 0
        return out, launches["n"], [p.grad.clone() for p in params]

    return SimpleNamespace(inner=inner, forward=forward, step=step, groups=groups)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("interrupt", ["backward_never_runs", "exception_in_backward"])
def test_group_starts_afresh_after_an_interrupted_step(monkeypatch, tiny_qlora, interrupt, split):
    """A step cut short (a loss whose backward never runs, an exception in the middle of the
    backward) leaves a layer group's counters and buffers mid-pass. The layer's next forward is
    the step boundary: the next step dequantizes as a clean step does (exactly once per layer and
    pass without the split), gives the same gradients and holds no buffer at the end."""
    from solution import qgroup

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS",
                        {"fwd": 1, "bwd": 2} if split else {"fwd": 0, "bwd": 0})
    tools = _qlora_step_tools(monkeypatch, tiny_qlora)
    tools.step()  # packs the layers on a first run (and compiles)
    _, n_clean, clean = tools.step()
    if not split:
        assert n_clean == 2 * 2  # 2 layers x (forward, backward)
    if interrupt == "backward_never_runs":
        tools.forward()  # the loss is dropped
    else:
        weight = qgroup.LayerGroup.weight

        def failing(self, which, idx):
            if which == "bwd" and (self.cache["bwd"][0] is not None or self.split_left):
                raise RuntimeError("interrupted")  # the second weight use of a backward pass
            return weight(self, which, idx)

        monkeypatch.setattr(qgroup.LayerGroup, "weight", failing)
        with pytest.raises(RuntimeError, match="interrupted"):
            tools.forward().loss.backward()
        monkeypatch.setattr(qgroup.LayerGroup, "weight", weight)
    assert any(g.expected_bwd or g.split_left or g.cache["bwd"][0] is not None
               for g in tools.groups()), "the interruption left nothing behind: no test"
    _, n_next, grads = tools.step()
    assert n_next == n_clean
    for g, c in zip(grads, clean):
        assert torch.equal(g, c)


@pytest.mark.parametrize("reentrant", [False, True])
def test_group_and_add_norm_under_gradient_checkpointing(monkeypatch, tiny_qlora, reentrant):
    """Trainer gradient checkpointing on (TRL's default; Soup turns it off in this task): every
    decoder layer runs again inside the backward, outside the causal-LM forward (with
    use_reentrant, its first run is under no_grad). The patched causal LM then takes the original
    path (the padding removal's row bounds would be gone in the replay), a checkpointed layer's
    group keeps no per-pass state, and no add+norm output crosses a checkpointed layer's
    boundary. Enabled after a fused step (the norms are linked), the step gives the loss and the
    LoRA gradients of the original forward without checkpointing and leaves nothing behind."""
    from solution import patches, qgroup

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    tools = _qlora_step_tools(monkeypatch, tiny_qlora)
    inner = tools.inner
    tools.step()  # a fused step: every layer is linked to the norm that reads its output
    assert all(getattr(layer, "_solution_next_norm", None) is not None
               for layer in inner.model.layers)
    ref, _, ref_grads = tools.step(patches.ORIGINAL_FORWARD[type(inner)])
    inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    try:
        assert inner.model.gradient_checkpointing
        assert all(layer.gradient_checkpointing for layer in inner.model.layers)
        out, n_launches, grads = tools.step()
    finally:
        inner.gradient_checkpointing_disable()
        inner.disable_input_require_grads()
    assert out.logits is not None  # the original causal-LM path
    assert n_launches > 0  # the 4-bit projections still went through the layer groups
    torch.testing.assert_close(out.loss.float(), ref.loss.float(), rtol=1e-4, atol=1e-5)
    for g, r in zip(grads, ref_grads):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g, r, rtol=1e-3, atol=1e-3 * float(r.abs().max()) + 1e-8)


def test_soup_selective_checkpointing(monkeypatch, tiny_qlora):
    """Soup's `gradient_checkpointing: selective` tier (soup_cli.utils.gradient_ckpt): each
    decoder layer's attention gets an instance-level `forward` that runs it under
    torch.utils.checkpoint (use_reentrant=False), and HF's own flag stays off. The attention then
    runs again inside the backward, outside the causal-LM forward: the padding removal's row
    bounds are gone there (the replay took another attention path than the forward: PyTorch's
    CheckpointError), and the replay's weight uses belong to no counted pass (a dequantized layer
    left cached until the layer's next forward). The patched causal LM must take the original
    path, the groups keep no per-pass state, and the step gives the original forward's loss and
    LoRA gradients. The gradient bound is 5e-3 of the largest element: on the RTX 4080 the
    replayed step matches within 1e-3, on the Kaggle T4 the attention replay (sm75's
    efficient-attention kernel is not run-to-run exact) moved 6 of 8 192 LoRA gradient elements
    by up to 1.3e-3 of the largest."""
    pytest.importorskip("soup_cli")
    from soup_cli.utils.gradient_ckpt import install_selective_hooks

    from solution import patches, qgroup

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    tools = _qlora_step_tools(monkeypatch, tiny_qlora)
    inner = tools.inner
    # as in training (transformers' Trainer sets it from its args, TRL's compute_loss passes
    # use_cache=False): without it the replayed attention would extend a KV cache a second time
    monkeypatch.setattr(inner.config, "use_cache", False)
    tools.step()  # a fused step: every layer is linked to the norm that reads its output
    ref, _, ref_grads = tools.step(patches.ORIGINAL_FORWARD[type(inner)])
    attentions = [layer.self_attn for layer in inner.model.layers]
    try:
        assert install_selective_hooks(tiny_qlora, "selective") == len(attentions)
        assert not inner.model.gradient_checkpointing  # HF's flag stays off with this tier
        out, n_launches, grads = tools.step()  # checks the groups are at rest after the step
    finally:
        for attention in attentions:
            attention.__dict__.pop("forward", None)  # back to the class-level forward
    assert out.logits is not None  # the original causal-LM path
    assert n_launches > 0  # the 4-bit projections still went through the layer groups
    torch.testing.assert_close(out.loss.float(), ref.loss.float(), rtol=1e-4, atol=1e-5)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r, rtol=1e-3, atol=5e-3 * float(r.abs().max()) + 1e-8)


# --- deferred metrics -------------------------------------------------------------------------

def _edge_metric_steps():
    """(inputs, fused step) pairs: one row, rows of one length (no padding), every position
    labelled, nothing labelled."""
    torch.manual_seed(44)
    steps = []
    for batch, seq, lengths, labelled in ((1, 37, [37], 30), (4, 20, [20] * 4, 76),
                                          (4, 50, [50, 3, 1, 22], 72), (4, 50, [9, 1, 50, 4], 0)):
        mask = (torch.arange(seq)[None] < torch.tensor(lengths)[:, None]).long().cuda()
        steps.append(({"attention_mask": mask}, SimpleNamespace(
            token_entropy=torch.rand(batch, seq, device="cuda") * 3,
            correct=torch.tensor(labelled // 3, device="cuda"),
            labelled=torch.tensor(labelled, device="cuda"))))
    return steps


def test_deferred_metrics_on_edge_batches():
    """TRL's per-step metrics fetched when they are logged equal the immediate ones on the edge
    batches: the same keys in the same order, the same floats, the same token count."""
    from solution import host, patches

    steps = _edge_metric_steps()
    saved = patches._host["defer"]
    results = []
    try:
        for defer in (False, True):
            patches._host["defer"] = defer
            trainer = SimpleNamespace(
                _metrics={"train": collections.defaultdict(list),
                          "eval": collections.defaultdict(list)},
                _total_train_tokens=0,
                accelerator=SimpleNamespace(gather_for_metrics=lambda t: t))
            for inputs, step in steps:
                patches._log_trl_metrics(trainer, "train", inputs, step)
            if defer:
                assert not trainer._metrics["train"]
                host.resolve_metrics()
            results.append((list(trainer._metrics["train"].items()), trainer._total_train_tokens))
    finally:
        patches._host["defer"] = saved
    assert results[0] == results[1]
    assert results[1][1] == 37 + 80 + 76 + 64
    assert results[1][0][2][1][-1] == 0.0  # nothing labelled: accuracy 0.0, as TRL logs it


def test_deferred_rewind_rows_on_edge_batches():
    """Soup's rewind row losses left on the GPU: a batch of one row, a row of one token, a row
    without a supervised token; the same numbers as Soup's own function."""
    pytest.importorskip("soup_cli")
    from soup_cli.trainer import rewind_hf

    from solution import host

    host._patch_rewind()
    rows_fn = rewind_hf.row_losses_from_token_losses
    original = rows_fn._original
    torch.manual_seed(45)
    for batch, seq in ((1, 30), (4, 1), (4, 512)):
        labels = torch.randint(0, 100, (batch, seq), device="cuda")
        labels[0, : seq // 2] = -100
        if batch > 1:
            labels[1] = -100
        token_loss = torch.rand(batch, seq, device="cuda")
        token_loss[F.pad(labels, (0, 1), value=-100)[:, 1:] < 0] = 0
        host.mark_fused_token_loss(token_loss)
        lazy_loss, lazy_tokens = rows_fn(token_loss, labels)
        assert isinstance(lazy_loss, host.LazyRows) and len(lazy_loss) == batch
        ref_loss, ref_tokens = original(token_loss, labels)
        assert list(lazy_loss) == list(ref_loss) and list(lazy_tokens) == list(ref_tokens)
    host.mark_fused_token_loss(None)


def test_no_host_sync_added_by_the_edge_paths(monkeypatch):
    """A batch with a row without labels and a 1-token row, or without any label, makes no more
    host syncs (forward and backward) than a plain batch: the edge cases, and the guard of the
    head's zero-denominator gradient, cost no sync."""
    from solution import patches

    model = _tiny_lm()
    monkeypatch.setitem(patches._unpad, "enabled", True)
    monkeypatch.setitem(patches._unpad, "checked", 3)  # past the right-padding check
    batches = {"plain": _batch([24, 9, 17, 20]),
               "edge": _batch([24, 1, 17, 20], labelled_rows={0, 2}),
               "no_label": _batch([24, 9, 17, 20], labelled_rows=set())}
    counts = {}
    for name in ("plain", "plain", "edge", "no_label"):  # the first forward does one-off work
        input_ids, mask, labels = batches[name]
        num_items = (labels[:, 1:] != -100).sum()
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(input_ids=input_ids, attention_mask=mask, labels=labels,
                                num_items_in_batch=num_items)
                out.loss.backward()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        counts[name] = sum("synchroniz" in str(w.message) for w in caught)
    assert counts["edge"] == counts["plain"] == counts["no_label"] >= 1, counts
