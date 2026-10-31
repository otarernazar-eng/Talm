"""Threads, process lifecycle and global state: the warm-up thread (warmup.py) against the main
thread, apply() called twice or without CUDA, interrupts, forks, and the global swaps and
module state that must survive exceptions (startup.py, lora.py).

Run on a GPU: `python -m pytest solution -q` (the CPU-only cases run anywhere).
"""

import collections
import os
import signal
import subprocess
import sys
import threading
import time
import types
import warnings
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SOLUTION_DIR = str(REPO / "solution")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _warmup_threads_alive():
    return any(t.name == "solution-warmup" and t.is_alive() for t in threading.enumerate())


@pytest.fixture
def fresh_warmup(monkeypatch):
    """apply() done, and warmup.py as in a process whose thread has not started yet."""
    import solution
    from solution import warmup

    solution.apply()
    warmup.join()                                                     
    monkeypatch.setattr(warmup, "_thread", None)
    monkeypatch.setattr(warmup, "_report", {})
    monkeypatch.setitem(warmup._state, "started", False)
    yield warmup
    if warmup._thread is not None:
        warmup._thread.join(120)
    warmup._unhook()


def _tiny_lm():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=600, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=False,
                         attn_implementation="sdpa")
    torch.manual_seed(5)
    model = Qwen2ForCausalLM(config).cuda().train()
    model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    seq, lengths = 24, [24, 9, 17]
    input_ids = torch.randint(0, 600, (len(lengths), seq), device="cuda")
    mask = (torch.arange(seq, device="cuda")[None]
            < torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = input_ids.clone()
    labels[mask == 0] = -100
    labels[:, :4] = -100
    return model, {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


def _step(model, batch):
    model.model.embed_tokens.weight.grad = None
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(**batch)
    out.loss.backward()
    return out.loss.detach().clone(), model.model.embed_tokens.weight.grad.clone()


                                                                                                

def _solution_layers(fn):
    """How many functions defined under solution/ are stacked in `fn` (each wrapper keeps the
    function it wraps in a closure cell)."""
    count, seen = 0, set()
    while isinstance(fn, types.FunctionType) and id(fn) not in seen:
        seen.add(id(fn))
        if not os.path.abspath(fn.__code__.co_filename).startswith(SOLUTION_DIR):
            break
        count += 1
        inner = [c.cell_contents for c in fn.__closure__ or ()
                 if isinstance(c.cell_contents, types.FunctionType)]
        same = [f for f in inner if f.__name__ == fn.__name__]
        fn = same[0] if same else (inner[0] if len(inner) == 1 else None)
    return count


def _patched():
    import importlib.metadata

    import bitsandbytes as bnb
    import transformers
    import trl
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2 import modeling_qwen2 as qwen2

    found = {
        "Qwen2MLP.forward": qwen2.Qwen2MLP.forward,
        "Qwen2RMSNorm.forward": qwen2.Qwen2RMSNorm.forward,
        "Qwen2DecoderLayer.forward": qwen2.Qwen2DecoderLayer.forward,
        "Qwen2ForCausalLM.forward": qwen2.Qwen2ForCausalLM.forward,
        "apply_rotary_pos_emb": qwen2.apply_rotary_pos_emb,
        "Linear4bit.forward": bnb.nn.Linear4bit.forward,
        "sdpa": ALL_ATTENTION_FUNCTIONS["sdpa"],
        "SFTTrainer.compute_loss": trl.SFTTrainer.compute_loss,
        "SFTTrainer.log": trl.SFTTrainer.log,
        "Trainer.train": transformers.Trainer.train,
        "Trainer.floating_point_ops": transformers.Trainer.floating_point_ops,
        "Trainer._clip_grad_norm": transformers.Trainer._clip_grad_norm,
        "torch.compile": torch.compile,
        "packages_distributions": importlib.metadata.packages_distributions,
    }
    try:
        import peft
        from peft.tuners.lora import bnb as lora_bnb

        found["lora Linear4bit.forward"] = lora_bnb.Linear4bit.forward
        found["prepare_model_for_kbit_training"] = peft.prepare_model_for_kbit_training
    except ImportError:
        pass
    return found


def test_apply_twice_installs_nothing_twice(monkeypatch):
    """A second apply() stacks no wrapper (each patched attribute is the same object with the
    same number of our layers), starts no second warm-up thread and registers no second exit
    handler; nor does calling our patch functions again directly (test_fusions calls the
    add+norm patch after apply(): its saved original stays transformers' forward). A later
    apply() with an opt-in switch on (test_steady_state's SOLUTION_MLP_SAVE_UP=1) installs that
    patch, on top, and nothing else again."""
    import atexit

    from transformers.models.qwen2 import modeling_qwen2

    import solution
    from solution import mlp_save_up, patches, qgroup, rmsnorm, swiglu_inplace, warmup

    monkeypatch.delenv("SOLUTION_MLP_SAVE_UP", raising=False)
    monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward", modeling_qwen2.Qwen2MLP.forward)
    solution.apply()
    if getattr(modeling_qwen2.Qwen2MLP.forward, "_solution_save_up", False):                     
        monkeypatch.setattr(modeling_qwen2.Qwen2MLP, "forward",
                            modeling_qwen2.Qwen2MLP.forward._previous)
    before = _patched()
    layers = {name: _solution_layers(fn) for name, fn in before.items()}
    assert layers["Qwen2MLP.forward"] >= 2 and layers["Qwen2DecoderLayer.forward"] >= 2
    thread, callbacks = warmup._thread, getattr(atexit, "_ncallbacks", lambda: 0)()
    solution.apply()
    for patch in (rmsnorm.patch_qwen2_rmsnorm, rmsnorm.patch_qwen2_add_norm,
                  swiglu_inplace.patch_qwen2_mlp_inplace, qgroup.patch_group_dequant,
                  patches.patch_qwen2_causal_lm, patches.patch_sft_compute_loss):
        patch()
    after = _patched()
    assert {name: _solution_layers(fn) for name, fn in after.items()} == layers
    assert all(after[name] is before[name] for name in before), \
        [name for name in before if after[name] is not before[name]]
    assert warmup._thread is thread
    assert getattr(atexit, "_ncallbacks", lambda: 0)() == callbacks
    original = rmsnorm.LAYER_FORWARDS.get("original")
    if original is not None:
        assert not os.path.abspath(original.__code__.co_filename).startswith(SOLUTION_DIR)
    monkeypatch.setenv("SOLUTION_MLP_SAVE_UP", "1")
    solution.apply()
    solution.apply()
    mlp = modeling_qwen2.Qwen2MLP.forward
    assert getattr(mlp, "_solution_save_up", False) and mlp._previous is before["Qwen2MLP.forward"]
    assert mlp_save_up.patch_qwen2_mlp_save_up() is None and modeling_qwen2.Qwen2MLP.forward is mlp
    again = _patched()
    assert all(again[name] is before[name] for name in before if name != "Qwen2MLP.forward")
    warmup.join()


                                                                                             

@cuda
def test_main_thread_launches_while_the_warmup_builds_the_same_kernels(monkeypatch,
                                                                        fresh_warmup):
    """Both threads at the same kernels: the warm-up thread is inside the first load of its
    first kernel when the main thread launches the same kernels (same specializations, via
    compile_all) 50 times. No error, one cache entry per kernel, and each kernel compiled and
    loaded once: the main thread's first launch waits for the warm-up instead of compiling or
    loading a kernel that the warm-up thread is still building (Triton has no lock there)."""
    from triton.runtime import driver

    warmup = fresh_warmup
    kernels = warmup._kernels()
    compiles, loads = collections.Counter(), collections.Counter()
    for fn in kernels:
        fn.device_caches.clear()

        def counting(*args, _real=fn._do_compile, _name=fn.fn.__name__):
            compiles[_name] += 1
            return _real(*args)

        monkeypatch.setitem(fn.__dict__, "_do_compile", counting)
    utils = driver.active.utils
    real_load = utils.load_binary
    loading = threading.Event()

    def slow_load(name, *args):
        loads[name] += 1
        if threading.current_thread().name == "solution-warmup":
            loading.set()
            time.sleep(0.2)                                                                
        return real_load(name, *args)

    monkeypatch.setattr(utils, "load_binary", slow_load)
    warmup.start()
    assert loading.wait(120), warmup._report
    for _ in range(50):
        warmup.compile_all()
    torch.cuda.synchronize()
    assert warmup._thread is None and not _warmup_threads_alive()                              
    assert "error" not in warmup._report, warmup._report
    device = torch.cuda.current_device()
    built = {fn.fn.__name__: len(fn.device_caches[device][0]) for fn in kernels
             if fn.device_caches}
    assert built and all(n == 1 for n in built.values()), built
    assert dict(compiles) == dict.fromkeys(built, 1), compiles
    assert dict(loads) == dict.fromkeys(built, 1), loads
    assert not any(warmup._wait_for_warmup in fn.pre_run_hooks for fn in kernels)


@cuda
def test_first_forward_waits_for_a_slow_warmup_with_the_same_values(monkeypatch, fresh_warmup):
    """A warm-up thread slower than the start-up: the first fused forward waits for it before
    its first decoder layer, and loss and gradient are bit for bit those of a step with no
    warm-up thread."""
    from transformers.models.qwen2 import modeling_qwen2

    from solution import patches

    warmup = fresh_warmup
    model, batch = _tiny_lm()
    monkeypatch.setattr(patches, "_first_forward", [])
    ref = _step(model, batch)
    real = warmup.compile_all

    def slow(*args):
        time.sleep(1.0)
        return real(*args)

    monkeypatch.setattr(warmup, "compile_all", slow)
    monkeypatch.setattr(patches, "_first_forward", [True])
    alive = []
    layer_forward = modeling_qwen2.Qwen2DecoderLayer.forward

    def spy(self, *args, **kwargs):
        if not alive:
            alive.append(_warmup_threads_alive())
        return layer_forward(self, *args, **kwargs)

    monkeypatch.setattr(modeling_qwen2.Qwen2DecoderLayer, "forward", spy)
    warmup.start()
    got = _step(model, batch)
    assert alive == [False]
    assert warmup._thread is None and warmup._report["seconds"] >= 1.0
    assert "error" not in warmup._report, warmup._report
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


@cuda
def test_warmup_error_is_reported_at_the_join_and_kernels_compile_at_first_launch(
        monkeypatch, capsys, fresh_warmup):
    from solution import rmsnorm

    warmup = fresh_warmup

    def boom(*args):
        raise RuntimeError("warm-up failed on purpose")

    monkeypatch.setattr(warmup, "compile_all", boom)
    rmsnorm.rmsnorm_kernel.device_caches.clear()
    warmup.start()
    warmup.join()                           
    assert "warm-up failed on purpose" in warmup._report["error"]
    assert "warm-up failed on purpose" in capsys.readouterr().out
    assert warmup._wait_for_warmup not in rmsnorm.rmsnorm_kernel.pre_run_hooks
    torch.manual_seed(7)
    x = (torch.randn(3, 50, 1536, device="cuda") * 2).requires_grad_(True)
    w = 1 + 0.1 * torch.randn(1536, device="cuda")
    y = rmsnorm.RMSNorm.apply(x, w, 1e-6, torch.float16)                                 
    h = x.detach().float()
    ref = (w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6))).to(torch.float16)
    torch.testing.assert_close(y.float(), ref.float(), rtol=2e-3, atol=2e-3)
    y.float().sum().backward()
    assert torch.isfinite(x.grad).all()


@cuda
@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="POSIX signals")
def test_keyboard_interrupt_during_the_first_forward_join_propagates(monkeypatch,
                                                                     fresh_warmup):
    """Ctrl-C while the first fused forward waits for the warm-up: KeyboardInterrupt leaves
    the forward (from the join), the thread stays registered and the first-forward join stays
    armed, so the next forward (or the exit) still waits for it."""
    from solution import patches

    warmup = fresh_warmup
    model, batch = _tiny_lm()
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        release.wait(60)
        return {}

    monkeypatch.setattr(warmup, "compile_all", blocked)
    monkeypatch.setattr(patches, "_first_forward", [True])
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    warmup.start()
    assert entered.wait(60)
    main = threading.main_thread().ident

    def interrupt():
        time.sleep(0.5)                                              
        signal.pthread_kill(main, signal.SIGINT)

    killer = threading.Thread(target=interrupt)
    try:
        killer.start()
        with pytest.raises(KeyboardInterrupt) as info:
            with torch.autocast("cuda", dtype=torch.float16):
                model(**batch)
    finally:
        killer.join()
        signal.signal(signal.SIGINT, previous)
        release.set()
    assert any(entry.name == "join" and Path(entry.path).name == "warmup.py"
               for entry in info.traceback), info.traceback
    assert patches._first_forward == [True]
    assert warmup._thread is not None
    _step(model, batch)                          
    assert warmup._thread is None and patches._first_forward == []


@cuda
def test_join_timeout_keeps_the_thread_for_the_exit_wait(monkeypatch, fresh_warmup):
    """A thread still running when the join times out stays registered (the exit and fork
    waits join it), and launches stop waiting for it."""
    warmup = fresh_warmup
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        release.wait(60)
        return {}

    monkeypatch.setattr(warmup, "compile_all", blocked)
    warmup.start()
    assert entered.wait(60)
    thread = warmup._thread
    assert any(warmup._wait_for_warmup in fn.pre_run_hooks for fn in warmup._kernels())
    warmup.join(timeout=0.2)
    assert thread.is_alive() and warmup._thread is thread
    assert not any(warmup._wait_for_warmup in fn.pre_run_hooks for fn in warmup._kernels())
    release.set()
    warmup._finish()
    assert not thread.is_alive()


@cuda
def test_warmup_leaves_rng_and_current_stream_alone(fresh_warmup):
    """The thread draws no random numbers (training's dropout stream is untouched) and runs
    on its own default stream (the main thread's current stream is not changed)."""
    warmup = fresh_warmup
    for fn in warmup._kernels():
        fn.device_caches.clear()                                
    torch.manual_seed(123)
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        warmup.start()
        warmup.join(timeout=120)
        assert torch.cuda.current_stream() == side
                                                                                             
                                                                                   
    assert warmup._thread is None and not _warmup_threads_alive()
    assert "error" not in warmup._report, warmup._report
    assert torch.equal(torch.get_rng_state(), cpu) and torch.equal(torch.cuda.get_rng_state(), gpu)


@cuda
@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork")
def test_fork_waits_for_the_warmup_thread(monkeypatch, fresh_warmup):
    """os.fork() while the thread runs first waits for it (a before-fork hook), so no child
    inherits a lock held by a thread that does not exist there."""
    warmup = fresh_warmup

    def slow(*args):
        time.sleep(1.0)
        return {}

    monkeypatch.setattr(warmup, "compile_all", slow)
    warmup.start()
    thread = warmup._thread
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)                                   
        pid = os.fork()
    if pid == 0:
        os._exit(0)
    assert not thread.is_alive()
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


                                                                                              

NO_CUDA = """
import torch
import solution
from solution import patches, warmup
assert not torch.cuda.is_available()
solution.apply()
solution.apply()
assert warmup._thread is None and not warmup._state["started"]
from transformers import Qwen2Config, Qwen2ForCausalLM
config = Qwen2Config(vocab_size=600, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                     tie_word_embeddings=False, attn_implementation="sdpa")
torch.manual_seed(5)
model = Qwen2ForCausalLM(config).train()
ids = torch.randint(0, 600, (2, 12))
batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
out = model(**batch)
ref = patches.ORIGINAL_FORWARD[Qwen2ForCausalLM](model, **batch)
assert torch.equal(out.loss, ref.loss) and out.logits is not None
out.loss.backward()
print("NO_CUDA_OK")
"""


def test_apply_without_cuda_patches_and_falls_back():
    """In a process without a visible GPU apply() (twice) installs its patches, starts no
    thread, and a CPU training forward and backward take the original paths."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "-c", NO_CUDA], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0 and "NO_CUDA_OK" in proc.stdout, proc.stdout[-2000:] + \
        proc.stderr[-4000:]


                                                                                              

@pytest.mark.parametrize("error", [ImportError("broken on purpose"), KeyboardInterrupt()])
def test_check_swap_is_restored_when_the_import_raises(monkeypatch, error):
    """startup._import_with_check_off: the real `is_*_available` is back (also in a module the
    failed import left behind holding the stand-in), whatever the import raised."""
    import importlib

    pytest.importorskip("transformers")
    import transformers.utils as utils

    from solution import startup

    def real():
        return True

    monkeypatch.setattr(utils, "is_solution_test_available", real, raising=False)
    partial = types.ModuleType("solution_test_partial")
    monkeypatch.setitem(sys.modules, "solution_test_partial", partial)

    def failing(name):
        partial.is_solution_test_available = utils.is_solution_test_available                
        raise error

    with monkeypatch.context() as m:
        m.setattr(importlib, "import_module", failing)
        if isinstance(error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                startup._import_with_check_off("solution_test_mod", {},
                                               "is_solution_test_available")
        else:
            assert startup._import_with_check_off("solution_test_mod", {},
                                                  "is_solution_test_available") is False
    assert utils.is_solution_test_available is real
    assert partial.is_solution_test_available is real


def test_context_swaps_are_restored_when_the_block_raises():
    import importlib.metadata as metadata

    pytest.importorskip("transformers")
    import transformers.utils.import_utils                                            

    from solution import startup

    real_compile, real_mapping = torch.compile, metadata.packages_distributions
    with pytest.raises(KeyboardInterrupt):
        with startup.deferred_torch_compile():
            assert torch.compile is not real_compile
            raise KeyboardInterrupt
    assert torch.compile is real_compile
    with pytest.raises(ImportError):
        with startup.shared_package_mapping():
            raise ImportError("trl failed on purpose")
    assert metadata.packages_distributions is real_mapping


@pytest.mark.parametrize("error", [ImportError, KeyboardInterrupt])
def test_boto3_placeholder_is_removed_when_accelerate_fails(monkeypatch, error):
    """startup.import_accelerate_without_boto3: sys.modules["boto3"] = None is undone whatever
    the accelerate import raises."""
    import importlib.util

    from solution import startup

    class Refuse:
        def find_spec(self, name, path=None, target=None):
            if name == "accelerate":
                raise error("accelerate broken on purpose")
            return None

    real_find_spec = importlib.util.find_spec
    monkeypatch.delitem(sys.modules, "accelerate", raising=False)
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "boto3" else real_find_spec(name, *a))
    monkeypatch.setattr(sys, "meta_path", [Refuse()] + sys.meta_path)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            startup.import_accelerate_without_boto3()
    else:
        assert startup.import_accelerate_without_boto3() is False
    assert "boto3" not in sys.modules


@cuda
def test_lora_backward_restores_the_rng_state_when_the_replay_raises(monkeypatch):
    """The LoRA node's backward sets the CUDA generator back to the forward's state to replay
    the dropout; if the replay raises, the generator is still put back where it was."""
    from solution.lora import LoraProjectionTorch

    torch.manual_seed(11)
    x = torch.randn(32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    base = torch.randn(48, 64, device="cuda", dtype=torch.float16) * 0.1
    group = types.SimpleNamespace(weight=lambda which, idx: base, biases=[None])
    lora_a = torch.randn(8, 64, device="cuda", requires_grad=True)
    lora_b = torch.randn(48, 8, device="cuda", requires_grad=True)
    out = LoraProjectionTorch.apply(x, x, lora_a, lora_b, group, 0, 2.0, 0.1)
    real_dropout = torch.native_dropout

    def failing(*args, **kwargs):
        raise RuntimeError("replay failed on purpose")

    torch.cuda.manual_seed(99)
    torch.randn(5, device="cuda")                                                    
    before = torch.cuda.get_rng_state()
    monkeypatch.setattr(torch, "native_dropout", failing)
    with pytest.raises(RuntimeError, match="replay failed on purpose"):
        out.float().sum().backward()
    monkeypatch.setattr(torch, "native_dropout", real_dropout)
    assert torch.equal(torch.cuda.get_rng_state(), before)
