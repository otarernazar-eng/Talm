"""Nothing of ours outlives a training step or a training run.

Two complete training runs, one after the other in one process, through `solution.apply()`, a
tiny Qwen2 loaded in 4-bit as Soup loads the real one (bitsandbytes NF4 with double
quantization, fp16, PEFT's k-bit preparation, LoRA on q_proj / v_proj, TRL's SFTTrainer with fp16
autocast and right padding), so every patched path runs: padding removal, the variable-length
attention, the per-layer dequantization (whole-layer and per-projection), the LoRA node with its
fp16 copies, the tied weight stored in fp16 (fp16_embed), the fused head with the deferred
metrics. After every optimizer step no per-pass
buffer, counter, pending step or row context of ours is left, the LoRA copies hold no CUDA
memory, and the memory left allocated is the same from step to step; after each run the deferred
queues are empty, the model, the trainer and our per-layer groups are freed, and the second run
leaves exactly as much allocated as the first.

Run on a GPU: `python -m pytest solution -q`.
"""

import gc
import json
import weakref

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

VOCAB = 512
STEPS = 5
ROWS = 4 * STEPS
SLACK = 64 * 1024                                                                      


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"w{i}": i for i in range(3, VOCAB)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                   unk_token="<unk>")


def _dataset():
    """Pre-tokenized prompt/completion rows of different lengths (so every batch is padded)."""
    from datasets import Dataset

    gen = torch.Generator().manual_seed(7)
    rows = []
    for _ in range(ROWS):
        prompt = int(torch.randint(4, 20, (1,), generator=gen))
        completion = int(torch.randint(3, 40, (1,), generator=gen))
        ids = torch.randint(3, VOCAB, (prompt + completion,), generator=gen).tolist() + [1]
        rows.append({"input_ids": ids, "completion_mask": [0] * prompt + [1] * (completion + 1)})
    return Dataset.from_list(rows)


def _save_tiny_model(path):
    """Shapes the per-layer packing takes, including the per-projection slices of the last
    layers (every weight a multiple of 1024 absmax blocks); a tied fp32 head as Qwen2.5's."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=VOCAB, hidden_size=512, intermediate_size=1024,
                         num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(path)


def _load(path):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant,
                                                 dtype=torch.float16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                            target_modules=["q_proj", "v_proj"],
                                            task_type="CAUSAL_LM"))


def _groups(model):
    found = {}
    for module in model.modules():
        tag = getattr(module, "_solution_group", None)
        if tag is not None:
            found[id(tag[0])] = tag[0]
    return list(found.values())


def _half_cuda_bytes():
    """CUDA bytes the LoRA fp16 copies (lora._half) keep alive."""
    from solution import lora

    storages = {}
    for entry in lora._half.entries:
        copy = entry[2]
        if copy is not None and copy.is_cuda:
            storages[copy.untyped_storage().data_ptr()] = copy.untyped_storage().nbytes()
    return sum(storages.values())


@pytest.mark.parametrize("mlp", ["default", "save_up"])
def test_two_training_runs_leave_nothing_behind(tmp_path, monkeypatch, mlp):
    """`mlp="save_up"`: the same with SOLUTION_MLP_SAVE_UP=1 (gate recomputed in the backward
    from the layer group's dequantized weight)."""
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    import transformers
    from transformers.models.qwen2 import modeling_qwen2
    from trl import SFTConfig, SFTTrainer

    import solution
    from solution import fused_proj, host, lora, mlp_save_up, patches, unpad

    monkeypatch.setenv("SOLUTION_VARLEN", "1")                                                 
                                                                                    
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", modeling_qwen2.Qwen2MLP.forward)
    if mlp == "save_up":
        monkeypatch.setenv("SOLUTION_MLP_SAVE_UP", "1")
    solution.apply()
    if mlp == "default" and getattr(modeling_qwen2.Qwen2MLP.forward, "_solution_save_up", False):
        monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward",
                            modeling_qwen2.Qwen2MLP.forward._previous)
    calls = {"lora": 0, "qkv": 0, "save_up": 0}

    def counter(key, real):
        def counted(*args):
            calls[key] += 1
            return real(*args)
        return counted

    monkeypatch.setattr(lora.LoraProjectionTorch, "apply",
                        counter("lora", lora.LoraProjectionTorch.apply))
    monkeypatch.setattr(fused_proj.FusedProjections, "apply",
                        counter("qkv", fused_proj.FusedProjections.apply))
    monkeypatch.setattr(mlp_save_up.SwiGLUSaveUp, "apply",
                        counter("save_up", mlp_save_up.SwiGLUSaveUp.apply))
    tokenizer, dataset = _tokenizer(), _dataset()
    path = tmp_path / "tiny"
    _save_tiny_model(path)
    runs, after_run = [], []
    for run in range(2):
        steps, at_log = [], []

        class Check(transformers.TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kwargs):
                torch.cuda.synchronize()
                groups = _groups(model)
                steps.append({
                    "alloc": torch.cuda.memory_allocated(), "groups": len(groups),
                    "held": [(i, which) for i, g in enumerate(groups) for which in ("fwd", "bwd")
                             if g.cache[which][0] is not None],
                    "counters": [(i, g.expected_bwd, g.split_left) for i, g in enumerate(groups)
                                 if g.expected_bwd or g.split_left],
                    "half_bytes": _half_cuda_bytes(), "pending": len(patches._pending_step),
                    "row_contexts": len(unpad._rows) + len(unpad._cu)})

            def on_log(self, args, state, control, **kwargs):
                                                                         
                at_log.append((len(host._metric_queue), len(host._rewind_queue),
                               host._fused_token_loss[0] is None))

        model = _load(path)
        config = SFTConfig(output_dir=str(tmp_path / f"run{run}"), max_steps=STEPS,
                           per_device_train_batch_size=4, gradient_accumulation_steps=1,
                           learning_rate=2e-4, fp16=True, logging_steps=1, save_strategy="no",
                           report_to="none", seed=1, max_length=64, completion_only_loss=True,
                           gradient_checkpointing=False,                                       
                           dataloader_num_workers=0, disable_tqdm=True)
        trainer = SFTTrainer(model=model, args=config, train_dataset=dataset,
                             processing_class=tokenizer, callbacks=[Check()])
        for name, param in trainer.model.named_parameters():
                                                                                              
                                                                                      
            if param.requires_grad and "lora_" in name and param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
        calls.update(lora=0, qkv=0, save_up=0)
        trainer.train()
        assert getattr(trainer, patches.STEP_ATTR, None) is not None, "the fused step never ran"
                                                                                               
                                                                                              
                                                                                       
        assert calls["lora"] + 2 * calls["qkv"] == 2 * 3 * STEPS, calls
        assert calls["qkv"] == 2 * STEPS, calls
        assert calls["save_up"] == (3 * STEPS if mlp == "save_up" else 0), calls
        assert len(steps) == STEPS and all(s["groups"] == 3 for s in steps), "layers not packed"
        assert at_log and all(entry == (0, 0, True) for entry in at_log), at_log
        embedding = model.get_input_embeddings()
                                                                                        
        assert embedding.weight.dtype == torch.float16
        refs = [weakref.ref(obj) for obj in (trainer, model, model.get_base_model(), embedding)]
        refs += [weakref.ref(group) for group in _groups(model)]
        del embedding
        del trainer, model, Check
        gc.collect()
        torch.cuda.synchronize()
        alive = [ref() for ref in refs if ref() is not None]
        assert not alive, f"kept alive after the run: {[type(obj).__name__ for obj in alive]}"
        assert not host._metric_queue and not host._rewind_queue
        assert host._fused_token_loss[0] is None and not patches._pending_step
        assert not unpad._rows and not unpad._cu
        assert _half_cuda_bytes() == 0, "LoRA fp16 copies of a finished run still hold memory"
        runs.append(steps)
        after_run.append(torch.cuda.memory_allocated())

    report = json.dumps({"runs": runs, "after_run": after_run}, indent=1)
    for steps in runs:
        for s in steps:
            assert not s["held"] and not s["counters"], report                                 
            assert s["half_bytes"] == 0, report                                         
            assert s["pending"] == 0 and s["row_contexts"] == 0, report
        allocs = [s["alloc"] for s in steps]
        assert max(allocs) - min(allocs) <= SLACK, report                                   
    for first, second in zip(*runs):                                            
        assert abs(first["alloc"] - second["alloc"]) <= SLACK, report
    assert abs(after_run[1] - after_run[0]) <= SLACK, report                               


@pytest.mark.parametrize("depth", [None, 0, 1])
def test_save_up_layer_group_trace(monkeypatch, depth):
    """SOLUTION_MLP_SAVE_UP's node on a packed layer group, three steps in a row, every weight
    use traced: per pass exactly the dequantizations of the default path (one whole-layer launch;
    on the last layers one per projection: depth 0 both passes, depth 1 the backward), seven uses
    per pass with the cached buffer's count going 6 .. 0 and the buffer dropped at the seventh,
    the backward counters back at 0 after every step: no buffer kept and none dequantized twice."""
    from solution import qgroup
    from solution.qgroup import LayerGroup
    from solution.test_mlp_save_up import HIDDEN, KV, _layer_group, _layer_step

    monkeypatch.setattr(qgroup, "SPLIT_LAYERS", {"fwd": 1, "bwd": 2})
    _layer, group = _layer_group(32)
    group.depth = depth
    trace = []
    real_weight, real_all = LayerGroup.weight, LayerGroup.dequantize_all
    real_one = LayerGroup.dequantize_one

    def weight(self, which, idx):
        out = real_weight(self, which, idx)
        trace.append((which, idx, self.cache[which][1], self.cache[which][0] is not None))
        return out

    def dequantize_all(self):
        trace.append(("all",))
        return real_all(self)

    def dequantize_one(self, idx):
        trace.append(("one", idx))
        return real_one(self, idx)

    monkeypatch.setattr(LayerGroup, "weight", weight)
    monkeypatch.setattr(LayerGroup, "dequantize_all", dequantize_all)
    monkeypatch.setattr(LayerGroup, "dequantize_one", dequantize_one)
    n = 300
    h = torch.randn(1, n, HIDDEN, device="cuda", dtype=torch.float16, requires_grad=True)
    x = torch.randn(1, n, HIDDEN, device="cuda", dtype=torch.float16, requires_grad=True)
    grads = [torch.randn(1, n, s, device="cuda") for s in (HIDDEN, KV, KV, HIDDEN, HIDDEN)]
    split = {"fwd": depth is not None and depth < 1, "bwd": depth is not None and depth < 2}
    for _ in range(3):
        trace.clear()
        _layer_step(group, h, x, grads, save_up=True)                                    
        h.grad = x.grad = None
        launches = [t for t in trace if t[0] in ("all", "one")]
        expected = []
        for which in ("fwd", "bwd"):
            uses = [t for t in trace if t[0] == which]
            assert sorted(t[1] for t in uses) == list(range(7)), (which, uses)                
            if split[which]:
                assert all(t[2] == 0 and not t[3] for t in uses), (which, uses)                  
                expected += [("one", t[1]) for t in uses]
            else:
                assert [t[2] for t in uses] == [6, 5, 4, 3, 2, 1, 0], (which, uses)
                assert all(t[3] for t in uses[:-1]) and not uses[-1][3], (which, uses)
                expected.append(("all",))
        assert sorted(launches) == sorted(expected), (launches, trace)


def test_fp16_embedding_is_not_kept_alive():
    """fp16_embed keeps no strong reference to an embedding it stored in fp16: the module and its
    fp16 weight go with the model (a strong one kept 445 MiB alive per model for the process)."""
    from solution import fp16_embed
    from solution.test_fp16_embed import _fp16_values, _TiedToy

    toy = _TiedToy(_fp16_values(4096, 256, seed=3).float())
    assert fp16_embed.store_fp16(toy) > 0
    module, weight = weakref.ref(toy.embed), weakref.ref(toy.embed.weight)
    del toy
    gc.collect()
    assert module() is None and weight() is None


def test_training_run_with_gradient_checkpointing(tmp_path):
    """TRL's default gradient checkpointing (Soup sets it off in this task): the run takes the
    original causal-LM forward and TRL's own loss and metrics (in the backward's replay of a
    layer the padding removal's row bounds would be gone), the checkpointed layers' groups keep
    no per-pass state, and every step ends with nothing held or pending."""
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    import transformers
    from trl import SFTConfig, SFTTrainer

    import solution
    from solution import patches, unpad

    solution.apply()
    path = tmp_path / "tiny"
    _save_tiny_model(path)
    steps, logged = [], []

    class Check(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            groups = _groups(model)
            steps.append({
                "groups": len(groups), "checkpointed": all(g.checkpointed for g in groups),
                "held": [(i, which) for i, g in enumerate(groups) for which in ("fwd", "bwd")
                         if g.cache[which][0] is not None],
                "counters": [(i, g.expected_bwd, g.split_left) for i, g in enumerate(groups)
                             if g.expected_bwd or g.split_left],
                "row_contexts": len(unpad._rows) + len(unpad._cu)})

        def on_log(self, args, state, control, logs=None, **kwargs):
            logged.append(dict(logs or {}))

    model = _load(path)
    config = SFTConfig(output_dir=str(tmp_path / "run"), max_steps=3,
                       per_device_train_batch_size=4, learning_rate=2e-4, fp16=True,
                       logging_steps=1, save_strategy="no", report_to="none", seed=1,
                       max_length=64, completion_only_loss=True, dataloader_num_workers=0,
                       disable_tqdm=True)
    assert config.gradient_checkpointing                 
    trainer = SFTTrainer(model=model, args=config, train_dataset=_dataset(),
                         processing_class=_tokenizer(), callbacks=[Check()])
    for name, param in trainer.model.named_parameters():
        if param.requires_grad and "lora_" in name and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)                                  
    trainer.train()
    assert model.get_base_model().model.gradient_checkpointing                          
    assert getattr(trainer, patches.STEP_ATTR, None) is None                              
    assert len(steps) == 3 and all(s["groups"] == 3 and s["checkpointed"] for s in steps), steps
    assert all(not s["held"] and not s["counters"] and not s["row_contexts"] for s in steps), steps
    losses = [entry for entry in logged if "loss" in entry]
    assert len(losses) == 3 and all("mean_token_accuracy" in entry for entry in losses), logged
    assert torch.isfinite(torch.tensor([entry["loss"] for entry in losses])).all(), logged
