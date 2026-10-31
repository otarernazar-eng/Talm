"""The tied embedding / lm_head weight stored in fp16 after PEFT's upcast (`fp16_embed.py`): the
same values, the same lookups, the same head outputs and the same training step, bit for bit.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _fp16_values(rows, cols, seed):
    """fp16 weights with the awkward cases in them: subnormals, signed zeros, large values."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(rows, cols, device="cuda", generator=gen) * 0.05).half()
    w[0, :4] = torch.tensor([0.0, -0.0, 6.0e-8, -3.0e-5], dtype=torch.float16)  # zeros, subnormals
    w[1, :2] = torch.tensor([65504.0, -1024.5], dtype=torch.float16)
    return w


class _TiedToy(torch.nn.Module):
    """The shape of a transformers causal LM as far as fp16_embed.py looks: an input embedding
    and an output head sharing one Parameter."""

    def __init__(self, weight, tied=True, embedding_cls=torch.nn.Embedding):
        super().__init__()
        vocab, hidden = weight.shape
        self.embed = embedding_cls(vocab, hidden, padding_idx=3, device=weight.device,
                                   dtype=weight.dtype)
        self.embed.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.head = torch.nn.Linear(hidden, vocab, bias=False, device=weight.device,
                                    dtype=weight.dtype)
        self.head.weight = self.embed.weight if tied else torch.nn.Parameter(weight.clone(),
                                                                            requires_grad=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def test_store_fp16_keeps_the_values_the_tie_and_the_fp32_lookup():
    """The stored weight converts back to the fp32 weight bit for bit, stays the same (tied,
    frozen) Parameter, frees the fp32 storage, and every lookup returns the fp32 rows bit for
    bit, inside and outside autocast."""
    from solution import fp16_embed

    w16 = _fp16_values(5000, 192, seed=1)
    reference = _TiedToy(w16.float())  # PEFT's upcast: fp32 copies of the fp16 values
    toy = _TiedToy(w16.float().clone())
    param = toy.embed.weight
    ids = torch.randint(0, 5000, (3, 257), device="cuda")
    ids[0, :6] = torch.tensor([0, 1, 3, 3, 4999, 0])  # repeats, the padding row, the edges
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    saved = fp16_embed.store_fp16(toy)
    torch.cuda.synchronize()
    assert saved == pytest.approx(5000 * 192 * 2 / 2**20)
    assert before - torch.cuda.memory_allocated() >= 5000 * 192 * 2  # the fp32 storage is gone
    assert toy.embed.weight is param and toy.head.weight is param  # one Parameter, still tied
    assert param.dtype == torch.float16 and not param.requires_grad
    assert torch.equal(_bits(param.float()), _bits(reference.embed.weight))
    for autocast in (False, True):
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            ours, ref = toy.embed(ids), reference.embed(ids)
        assert ours.dtype == ref.dtype == torch.float32
        assert torch.equal(_bits(ours), _bits(ref))
    # under autocast the output head computes from the same fp16 values as from the fp32 weight
    hidden = torch.randn(64, 192, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        assert torch.equal(toy.head(hidden), reference.head(hidden))


def test_store_fp16_leaves_everything_else_alone():
    """Nothing changes unless the fp16 copy is exact and the weight is a frozen, tied, plain
    nn.Embedding weight on CUDA (and anything that is not a model is ignored)."""
    from unittest import mock

    from solution import fp16_embed

    class MyEmbedding(torch.nn.Embedding):
        pass

    exact = _fp16_values(300, 64, seed=2).float()
    cases = {
        "not fp16 values": _TiedToy(torch.randn(300, 64, device="cuda")),
        "untied": _TiedToy(exact.clone(), tied=False),
        "subclassed embedding": _TiedToy(exact.clone(), embedding_cls=MyEmbedding),
        "on the CPU": _TiedToy(exact.cpu()),
    }
    trainable = _TiedToy(exact.clone())
    trainable.embed.weight.requires_grad_(True)
    cases["trainable"] = trainable
    for name, toy in cases.items():
        dtype = toy.embed.weight.dtype
        assert fp16_embed.store_fp16(toy) == 0, name
        assert toy.embed.weight.dtype == dtype == torch.float32, name
        assert not toy.embed._forward_hooks, name
    for not_a_model in (object(), mock.MagicMock(), torch.nn.Linear(4, 4)):
        assert fp16_embed.tied_embedding(not_a_model) is None
        assert fp16_embed.store_fp16(not_a_model) == 0


@pytest.mark.parametrize("path,hidden_dtype,n_tokens,vocab,vocab_chunk,compact", [
    ("vocab", torch.float16, 1427, 151936, None, True),   # the task's sizes and default path
    ("vocab", torch.float16, 1427, 151936, None, False),
    ("vocab", torch.float32, 1427, 151936, None, True),
    ("tokens", torch.float16, 1427, 151936, None, True),
    ("vocab", torch.float16, 37, 4100, 1028, True),       # a last slice of another width
    ("tokens", torch.float32, 37, 4100, None, True),
    ("vocab", torch.float32, 300, 1000, 384, True),
    ("tokens", torch.float16, 300, 1000, None, True)])
def test_fused_head_is_bit_identical_with_the_fp16_weight(monkeypatch, path, hidden_dtype,
                                                          n_tokens, vocab, vocab_chunk, compact):
    """The fused head from the fp16-stored weight (slices used as they are) against the same
    values in fp32 (slices cast to fp16): loss, per-token loss, entropy, accuracy and the hidden
    gradient bit for bit, on both head paths, with and without the labelled-row compaction."""
    from solution import fused_head
    from solution.fused_head import fused_head_loss

    monkeypatch.setattr(fused_head, "COMPACT", compact)
    torch.manual_seed(31)
    size = 1536 if vocab > 10000 else 64
    w16 = (torch.randn(vocab, size, device="cuda") * 0.03).half()
    hidden = (torch.randn(n_tokens, size, device="cuda") * 1.5).to(hidden_dtype)
    hidden.requires_grad_(True)
    targets = torch.randint(0, vocab, (n_tokens,), device="cuda")
    targets[torch.rand(n_tokens, device="cuda") < 0.45] = -100
    n_lab = int((targets >= 0).sum())
    chunks = {"token_chunk": 256} if path == "tokens" else {"vocab_chunk": vocab_chunk}
    outs = {}
    for name, weight in (("fp32", w16.float()), ("fp16", w16)):
        hidden.grad = None
        loss, token_loss, entropy, correct = fused_head_loss(hidden, weight, targets, n_lab + 5,
                                                             n_labelled=n_lab, **chunks)
        (loss * 1024.0).backward()
        outs[name] = (loss.detach(), token_loss, entropy, correct, hidden.grad.clone())
    for a, b in zip(outs["fp32"], outs["fp16"]):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_fused_head_needs_no_slice_buffer_for_the_fp16_weight():
    """From the fp16 weight the vocabulary-sliced head neither allocates the fp16 slice buffer
    (24 MiB at this task's sizes) nor launches the per-slice casts (19 here)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    from solution.fused_head import VOCAB_CHUNK, fused_head_loss

    torch.manual_seed(32)
    n, size, vocab = 700, 1536, 151936
    w16 = (torch.randn(vocab, size, device="cuda") * 0.03).half()
    hidden = (torch.randn(n, size, device="cuda") * 1.5).half().requires_grad_(True)
    targets = torch.randint(0, vocab, (n,), device="cuda")
    targets[::3] = -100
    transient, copies = {}, {}
    for name, weight in (("fp32", w16.float()), ("fp16", w16)):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            loss = fused_head_loss(hidden, weight, targets)[0]
            torch.cuda.synchronize()
        transient[name] = torch.cuda.max_memory_allocated() - base
        copies[name] = sum(e.count for e in prof.key_averages()
                           if e.device_type == DeviceType.CUDA and "copy" in e.key.lower())
        del loss
    assert copies["fp32"] - copies["fp16"] == -(-vocab // VOCAB_CHUNK)  # one cast per slice
    assert transient["fp32"] - transient["fp16"] >= VOCAB_CHUNK * size * 2


def _tiny_checkpoint(path):
    """A small tied Qwen2 saved in bf16, as the real checkpoint is, with sizes the solution's
    kernels take (the group dequant packs its seven 4-bit projections)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=256, tie_word_embeddings=True,
                         attn_implementation="sdpa")
    torch.manual_seed(41)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        for param in model.parameters():  # trained-looking scales, not the init's
            param.normal_(0, 0.08)
    model.to(torch.bfloat16).save_pretrained(path)


def _qlora_model(path, prepare):
    """Soup's set-up of the task on this checkpoint: 4-bit NF4 load in fp16 (the T4 path),
    `prepare` (PEFT's prepare_model_for_kbit_training, original or ours), LoRA r=16 on q/v."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(str(path), quantization_config=quant,
                                                 device_map={"": 0}, dtype=torch.float16)
    model = prepare(model, use_gradient_checkpointing=False)
    torch.manual_seed(42)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    torch.manual_seed(43)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:  # PEFT starts B at 0: give the adapter a non-zero output
                param.normal_(0, 0.05)
    return model


def test_qlora_step_is_bit_identical_with_the_patched_prepare(tmp_path):
    """The task's pipeline on a small tied Qwen2 through `solution.apply()`: 4-bit fp16 load,
    PEFT's prepare (original vs ours), LoRA, then a training step through the fused path and an
    eval forward through transformers' own head. Only the tied weight's storage differs (fp16,
    the same values, still tied, frozen); loss, logits and every LoRA gradient are bit for bit
    the same, and the adapter state (names, shapes, dtypes, values) too."""
    import peft

    from solution.test_kernels import apply_quietly

    apply_quietly()
    patched = peft.prepare_model_for_kbit_training
    assert getattr(patched, "_solution_fp16_embed", False)
    assert peft.utils.prepare_model_for_kbit_training is patched
    _tiny_checkpoint(tmp_path)
    models = {"fp32": _qlora_model(tmp_path, patched._original),
              "fp16": _qlora_model(tmp_path, patched)}

    weights = {}
    for name, model in models.items():
        base = model.get_base_model()
        weight = base.get_input_embeddings().weight
        assert base.get_output_embeddings().weight is weight  # tied
        assert not weight.requires_grad
        weights[name] = weight
    assert weights["fp32"].dtype == torch.float32 and weights["fp16"].dtype == torch.float16
    assert torch.equal(_bits(weights["fp16"].float()), _bits(weights["fp32"]))
    dtypes = {name: {n: p.dtype for n, p in model.named_parameters()}
              for name, model in models.items()}
    differs = [n for n in dtypes["fp32"] if dtypes["fp32"][n] != dtypes["fp16"][n]]
    assert differs == ["base_model.model.model.embed_tokens.weight"]  # nothing else changed
    assert models["fp16"].get_base_model().dtype == torch.float16  # transformers' model.dtype
    trainable = {name: [(n, p.dtype, tuple(p.shape)) for n, p in model.named_parameters()
                        if p.requires_grad] for name, model in models.items()}
    assert trainable["fp32"] == trainable["fp16"]
    assert all(dtype == torch.float32 for _, dtype, _ in trainable["fp16"])

    batch, seq = 3, 40
    gen = torch.Generator(device="cuda").manual_seed(44)
    input_ids = torch.randint(0, 1000, (batch, seq), device="cuda", generator=gen)
    lengths = [40, 17, 29]
    attention_mask = (torch.arange(seq, device="cuda")[None]
                      < torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[:, :7] = -100
    results = {}
    for name, model in models.items():
        model.train()
        torch.manual_seed(45)  # the LoRA dropout masks
        embeds = model.get_base_model().get_input_embeddings()(input_ids)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert out.logits is None  # the fused head ran
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        results[name] = (embeds, out.loss.detach(), grads, logits)
    (e32, loss32, g32, l32), (e16, loss16, g16, l16) = results["fp32"], results["fp16"]
    assert e32.dtype == e16.dtype == torch.float32 and torch.equal(_bits(e32), _bits(e16))
    assert torch.equal(loss32, loss16)
    assert g32.keys() == g16.keys() and len(g32) == 8  # q/v A/B in two layers
    for n in g32:
        assert torch.equal(g32[n], g16[n]), n
    assert torch.equal(l32, l16)

    from peft import get_peft_model_state_dict

    states = {name: get_peft_model_state_dict(model) for name, model in models.items()}
    assert states["fp32"].keys() == states["fp16"].keys()
    for n in states["fp32"]:
        a, b = states["fp32"][n], states["fp16"][n]
        assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b), n


def test_head_weight_dtype_follows_the_switch_and_the_path(monkeypatch):
    from solution import fp16_embed

    monkeypatch.delenv("SOLUTION_DISABLE_FP16_EMBED", raising=False)
    assert fp16_embed.head_weight_dtype(torch.float16) == torch.float16
    assert fp16_embed.head_weight_dtype(torch.bfloat16) == torch.float32  # bf16 GPUs: untouched
    monkeypatch.setenv("SOLUTION_DISABLE_FP16_EMBED", "1")
    assert fp16_embed.head_weight_dtype(torch.float16) == torch.float32


def test_switch_leaves_peft_alone(monkeypatch):
    """apply() wraps PEFT's function in every namespace that exports it, once; with
    SOLUTION_DISABLE_FP16_EMBED=1 it leaves PEFT's own function in place."""
    import peft
    import peft.utils.other

    import solution
    from solution import fp16_embed

    monkeypatch.setenv("SOLUTION_WARMUP_THREAD", "0")
    monkeypatch.setenv("SOLUTION_DISABLE_STARTUP", "1")  # gc.freeze included
    namespaces = (peft, peft.utils, peft.utils.other)
    solution.apply()
    wrapped = peft.prepare_model_for_kbit_training
    assert getattr(wrapped, "_solution_fp16_embed", False)
    assert all(ns.prepare_model_for_kbit_training is wrapped for ns in namespaces)
    assert fp16_embed.patch_prepare_model_for_kbit_training() is True  # no second wrapper
    assert peft.prepare_model_for_kbit_training is wrapped
    for ns in namespaces:
        monkeypatch.setattr(ns, "prepare_model_for_kbit_training", wrapped._original)
    monkeypatch.setenv("SOLUTION_DISABLE_FP16_EMBED", "1")
    solution.apply()
    assert all(ns.prepare_model_for_kbit_training is wrapped._original for ns in namespaces)
