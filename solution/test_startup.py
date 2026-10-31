"""Tests of solution/startup.py: the patches outside the training loop change no result."""

import sys

import pytest
import torch

from solution import startup


def test_startup_table_names_exist():
    import solution

    for switch, name, _label in solution.STARTUP:
        assert switch.startswith("SOLUTION_DISABLE_") and callable(getattr(startup, name))


                                                                                                  

def test_shared_package_mapping_is_the_real_mapping():
    import importlib.metadata as metadata

    pytest.importorskip("transformers")
    import transformers.utils.import_utils as import_utils

    real = metadata.packages_distributions
    with startup.shared_package_mapping() as shared:
        assert shared
        inside = metadata.packages_distributions()
        assert metadata.packages_distributions is not real
    assert metadata.packages_distributions is real
    assert inside == import_utils.PACKAGE_DISTRIBUTION_MAPPING
    assert inside == real()                                                        
    inside["solution-test"] = ["x"]                                             
    assert "solution-test" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING


def test_import_trl_is_a_noop_when_loaded():
    if "trl" in sys.modules:
        assert startup.import_trl() is False


                                                                                                  

def _fake_tokenizer(cls, n, shift=0):
    tok = object.__new__(cls)
    tok.__dict__.update({
        "_vocab": {f"tok{i}": i + shift for i in range(n)},
        "_merges": [(f"a{i}", f"b{i}") for i in range(n)],
        "name_or_path": "fake",
        "chat_template": "{{ messages }}",
    })
    return tok


def _fake_tokenizer_class():
    transformers = pytest.importorskip("transformers")

    class FakeTokenizer(transformers.PreTrainedTokenizerBase):
        def __repr__(self):                                                       
            return "FakeTokenizer()"

    return FakeTokenizer


def test_tokenizer_fingerprint_is_a_content_digest():
    pytest.importorskip("datasets")
    from datasets.fingerprint import Hasher

    FakeTokenizer = _fake_tokenizer_class()
    assert startup.patch_tokenizer_fingerprint()
    big = _fake_tokenizer(FakeTokenizer, 5000)
    vocab, merges = big._vocab, big._merges
    first = Hasher.hash(big)
    assert big._vocab is vocab and big._merges is merges                                
    same = _fake_tokenizer(FakeTokenizer, 5000)                                
    assert Hasher.hash(same) == first
    assert Hasher.hash(_fake_tokenizer(FakeTokenizer, 5000, shift=1)) != first
    other = _fake_tokenizer(FakeTokenizer, 5000)
    other._merges = other._merges[::-1]                       
    assert Hasher.hash(other) != first
                                                                      
    from datasets.utils import _dill

    original, patched = startup._state["tokenizer_reducer"]
    small = _fake_tokenizer(FakeTokenizer, 10)
    patched_hash = Hasher.hash(small)
    _dill.Pickler.dispatch[FakeTokenizer] = original
    try:
        assert Hasher.hash(small) == patched_hash
    finally:
        _dill.Pickler.dispatch[FakeTokenizer] = patched


def test_map_with_patched_fingerprint_gives_the_same_rows():
    datasets = pytest.importorskip("datasets")
    FakeTokenizer = _fake_tokenizer_class()
    assert startup.patch_tokenizer_fingerprint()
    tok = _fake_tokenizer(FakeTokenizer, 5000)

    def fmt(row):
        return {"ids": [tok._vocab.get(w, -1) for w in row["text"].split()]}

    rows = [{"text": "tok1 tok2 tok4999 x"}, {"text": "tok0"}]
    out = datasets.Dataset.from_list(rows).map(fmt, remove_columns=["text"])
    assert out.to_list() == [{"ids": [1, 2, 4999, -1]}, {"ids": [0]}]


                                                                                                  

def test_deferred_compile_builds_on_first_call(monkeypatch):
    calls = []

    def fake_compile(model=None, **kwargs):
        calls.append(kwargs)
        return lambda *a, **k: ("compiled", model(*a, **k))

    monkeypatch.setattr(torch, "compile", fake_compile)

    def square(x):
        return x * x

    with startup.deferred_torch_compile():
        wrapped = torch.compile(square, dynamic=False)
        decorated = torch.compile(dynamic=True)(square)
        module = torch.nn.Linear(2, 2)
        compiled_module = torch.compile(module)                                           
    assert torch.compile is fake_compile            
    assert calls == [{}]                                        
    assert wrapped.__name__ == "square" and decorated.__name__ == "square"
    assert wrapped(3) == ("compiled", 9)
    assert decorated(4) == ("compiled", 16)
    assert wrapped(5) == ("compiled", 25)
    assert calls == [{}, {"dynamic": False}, {"dynamic": True}]                                  
    assert compiled_module(torch.ones(2))[0] == "compiled"


def test_deferred_compile_restores_on_error():
    real = torch.compile
    with pytest.raises(RuntimeError):
        with startup.deferred_torch_compile():
            raise RuntimeError("boom")
    assert torch.compile is real


def test_import_bitsandbytes_is_a_noop_when_loaded():
    if "bitsandbytes" in sys.modules:
        assert startup.import_bitsandbytes() is False


                                                                                                  

def test_lazy_callable_imports_on_first_call():
    import math

    sqrt = startup._lazy_callable("math", "sqrt")
    assert sqrt.__name__ == "sqrt"
    assert sqrt(16.0) == math.sqrt(16.0)


def test_import_without_sklearn_defers_and_restores(tmp_path, monkeypatch):
    """A module shaped like transformers' sklearn importers, with a stand-in for scikit-learn:
    imported without the heavy import, then the real check and a working lazy name."""
    pytest.importorskip("transformers")
    import transformers.utils as utils

    (tmp_path / "solution_fake_heavy.py").write_text("def score(a, b):\n    return a * 10 + b\n")
    (tmp_path / "solution_fake_importer.py").write_text(
        "from transformers.utils import is_sklearn_available\n"
        "if is_sklearn_available():\n"
        "    from solution_fake_heavy import score\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def installed():
        return True

    monkeypatch.setattr(utils, "is_sklearn_available", installed)
    for name in ("solution_fake_heavy", "solution_fake_importer"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert startup._import_without_sklearn("solution_fake_importer",
                                           {"score": "solution_fake_heavy"})
    module = sys.modules["solution_fake_importer"]
    assert "solution_fake_heavy" not in sys.modules                                   
    assert utils.is_sklearn_available is installed                                
    assert module.is_sklearn_available is installed                                  
    assert module.score(4, 2) == 42                                                     
    assert "solution_fake_heavy" in sys.modules
    assert startup._import_without_sklearn("solution_fake_importer", {}) is False          


def test_transformers_modules_keep_the_real_sklearn_check():
    pytest.importorskip("transformers")
    from transformers.generation import candidate_generator as cg
    from transformers.utils import import_utils

                                                                                    
    assert cg.is_sklearn_available is import_utils.is_sklearn_available
    if import_utils.is_sklearn_available():
        assert callable(cg.roc_curve)


                                                                                                  

def test_import_with_scipy_check_off_defers_and_restores(tmp_path, monkeypatch):
    """A module shaped like transformers' DETR losses, with a stand-in for scipy.optimize."""
    pytest.importorskip("transformers")
    import transformers.utils as utils

    (tmp_path / "solution_fake_optimize.py").write_text(
        "def linear_sum_assignment(cost):\n    return ('matched', cost)\n")
    (tmp_path / "solution_fake_loss.py").write_text(
        "from transformers.utils import is_scipy_available\n"
        "if is_scipy_available():\n"
        "    from solution_fake_optimize import linear_sum_assignment\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def installed():
        return True

    monkeypatch.setattr(utils, "is_scipy_available", installed)
    for name in ("solution_fake_optimize", "solution_fake_loss"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert startup._import_with_check_off(
        "solution_fake_loss", {"linear_sum_assignment": "solution_fake_optimize"},
        "is_scipy_available")
    module = sys.modules["solution_fake_loss"]
    assert "solution_fake_optimize" not in sys.modules
    assert utils.is_scipy_available is installed and module.is_scipy_available is installed
    assert module.linear_sum_assignment(3) == ("matched", 3)
    assert "solution_fake_optimize" in sys.modules


def test_scipy_table_covers_every_transformers_loss():
    """Every module of transformers.loss that imports SciPy at import time is in the table, with
    the names it imports."""
    import ast
    import pathlib

    transformers = pytest.importorskip("transformers")
    found = {}
    for path in sorted((pathlib.Path(transformers.__file__).parent / "loss").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:                                                         
            if isinstance(node, ast.If):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.ImportFrom) and (inner.module or "").startswith("scipy"):
                        found.setdefault(f"transformers.loss.{path.stem}", {}).update(
                            {alias.asname or alias.name: inner.module for alias in inner.names})
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else \
                    [alias.name for alias in node.names]
                assert not any(n.startswith("scipy") for n in names), path                        
    assert found == startup.SCIPY_IMPORTERS


def test_transformers_losses_keep_the_real_scipy_check():
    pytest.importorskip("transformers")
    import importlib

    from transformers.utils import import_utils

    for name, names in startup.SCIPY_IMPORTERS.items():
        module = importlib.import_module(name)
        assert module.is_scipy_available is import_utils.is_scipy_available
        if import_utils.is_scipy_available():
            assert all(callable(getattr(module, attr)) for attr in names)


                                                                                                  

def test_lazy_module_imports_on_first_use():
    import json
    import types

    owner = types.SimpleNamespace()
    owner.json = startup._LazyModule("json", owner)
    assert owner.json.dumps([1]) == json.dumps([1])
    assert owner.json is json                                                   


def test_import_accelerate_without_boto3_in_a_fresh_process(tmp_path):
    """In a process where accelerate is not imported yet, with a stand-in boto3 installed:
    accelerate is imported without it, and its SageMaker config module imports it when used."""
    import pathlib
    import subprocess

    pytest.importorskip("accelerate")
    (tmp_path / "boto3").mkdir()
    (tmp_path / "boto3" / "__init__.py").write_text(
        "def client(name):\n    return ('client', name)\n")
    (tmp_path / "boto3-0.0.0.dist-info").mkdir()                                           
    (tmp_path / "boto3-0.0.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: boto3\nVersion: 0.0.0\n")
    repo = pathlib.Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r}); sys.path.insert(0, {str(repo)!r})\n"
        "from solution import startup\n"
        "assert 'accelerate' not in sys.modules\n"
        "assert startup.import_accelerate_without_boto3() is True\n"
        "assert 'boto3' not in sys.modules, 'boto3 was imported'\n"
        "from accelerate.utils import is_boto3_available\n"
        "assert is_boto3_available()  # the real answer again\n"
        "from accelerate.commands.config import sagemaker\n"
        "assert type(sagemaker.boto3).__name__ == '_LazyModule'\n"
        "assert sagemaker.boto3.client('iam') == ('client', 'iam')\n"
        "assert sagemaker.boto3 is sys.modules['boto3']\n"
        "assert startup.import_accelerate_without_boto3() is False  # loaded\n"
        "print('ok')\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stdout + proc.stderr


                                                                                                  

def test_freeze_gc_keeps_collecting_new_garbage():
    import gc
    import weakref

    before = gc.get_freeze_count()
    try:
        assert startup.freeze_gc() >= before
        assert gc.get_freeze_count() > 0

        class Node:
            pass

        a, b = Node(), Node()
        a.other, b.other = b, a                                              
        ref = weakref.ref(a)
        del a, b
        gc.collect()
        assert ref() is None                   
    finally:
        if before == 0:
            gc.unfreeze()                                    


def test_freeze_gc_does_nothing_when_disabled():
    import gc

    if not gc.isenabled():
        pytest.skip("collector disabled in this process")
    before = gc.get_freeze_count()
    gc.disable()
    try:
        assert startup.freeze_gc() == 0
        assert gc.get_freeze_count() == before
    finally:
        gc.enable()


                                                                                                  

def test_share_gpu_query_runs_the_query_once(monkeypatch):
    pytest.importorskip("accelerate")
    from accelerate.utils import environment

    calls = []

    def fake_get_gpu_info():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("nvidia-smi failed")                               
        return ["Tesla T4", "Tesla T4"], 2

    monkeypatch.setattr(environment, "get_gpu_info", fake_get_gpu_info)
    assert startup.share_gpu_query()
    shared = environment.get_gpu_info
    assert shared is not fake_get_gpu_info and shared._original is fake_get_gpu_info
    with pytest.raises(RuntimeError):
        shared()
    first = shared()
    first[0].append("changed by a caller")                      
    assert shared() == (["Tesla T4", "Tesla T4"], 2)
    assert len(calls) == 2
                                                                                           
    assert environment.check_cuda_p2p_ib_support() is True
    assert len(calls) == 2


                                                                                                  

def test_share_4bit_code_equals_bitsandbytes_tables():
    pytest.importorskip("bitsandbytes")
    from bitsandbytes import functional

    assert startup.share_4bit_code()
    shared = functional.get_4bit_type
    original = shared._original
    devices = ["cpu"] + (["cuda", torch.device("cuda:0")] if torch.cuda.is_available() else [])
    for device in devices:
        for name in ("nf4", "fp4", "af4"):
            first, second = shared(name, device=device), shared(name, device=device)
            want = original(name, device=device)
            assert first.data_ptr() != second.data_ptr()                                 
            for got in (first, second):
                assert got.device == want.device and got.dtype == want.dtype
                assert torch.equal(got.view(torch.int32), want.view(torch.int32))
            first.mul_(0)                                                               
            assert torch.equal(shared(name, device=device), want)
    with pytest.raises(NotImplementedError):
        shared("nf3")
    with pytest.raises(NotImplementedError):
        shared("af4", blocksize=128)
    with pytest.raises(RuntimeError):                                                   
        shared("int4", device="cpu")


                                                                                                  

class _FakeChatTokenizer:
    """Renders a conversation deterministically, counts the renders; no generation block in
    its template, like Qwen2.5's."""

    chat_template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
                     "{%- if add_generation_prompt %}<|assistant|>{% endif %}")
    eos_token_id = 0

    def __init__(self):
        self.calls = []

    def _ids(self, messages):
        ids = []
        for message in messages:
            ids.append(1000 + len(message["role"]))
            ids.extend(10 + (ord(ch) % 50) for ch in message["content"])
        return ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_assistant_tokens_mask=False, return_dict=False, **kwargs):
        self.calls.append((len(messages), return_assistant_tokens_mask))
        ids = self._ids(messages)
        if return_dict:
            out = {"input_ids": ids}
            if return_assistant_tokens_mask:
                out["assistant_masks"] = [0] * len(ids)                                  
            return out
        return ids


_CONVERSATIONS = [
    [{"role": "user", "content": "add 2 and 3"}, {"role": "assistant", "content": "5"}],
    [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
     {"role": "assistant", "content": "hello"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
     {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
    [{"role": "user", "content": "no answer yet"}],                                            
]


def test_share_chat_renders_gives_the_same_labels_with_fewer_renders():
    pytest.importorskip("soup_cli")
    from soup_cli.data import loss_mask, sft_format

    assert startup.share_chat_renders()
    patched = loss_mask.build_assistant_only_labels
    assert sft_format.build_assistant_only_labels is patched
    original = patched._original                                                             
    for kwargs in ({}, {"include_eot": True}, {"mask_history": True}, {"max_length": 5}):
        ref_tok, new_tok = _FakeChatTokenizer(), _FakeChatTokenizer()
        for messages in _CONVERSATIONS * 2:
            want = original(messages, ref_tok, **kwargs)
            got = patched(messages, new_tok, **kwargs)
            assert got == want
        assert len(new_tok.calls) < len(ref_tok.calls)
                                                                                          
                                                                     
        assert 2 <= sum(1 for _, mask in new_tok.calls if mask) <= 3
    assert startup._render_state["active"] is None                                


def test_share_chat_renders_real_tokenizer_and_rows():
    """The task's rows with the Qwen tokenizer: identical labels, half the renders."""
    import json
    import pathlib

    pytest.importorskip("soup_cli")
    transformers = pytest.importorskip("transformers")
    from soup_cli.data import loss_mask

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:                                      
        pytest.skip("Qwen tokenizer not in the local cache")
    path = pathlib.Path(__file__).resolve().parents[1] / "data" / "train.jsonl"
    if not path.exists():
        pytest.skip("data/train.jsonl missing (task/prepare_data.py)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:64]]
    assert startup.share_chat_renders()
    patched = loss_mask.build_assistant_only_labels
    calls = []
    real = type(tokenizer).apply_chat_template

    def counting(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    type(tokenizer).apply_chat_template = counting
    try:
        n_ref = n_new = 0
        for row in rows:
            user = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
            messages = [{"role": "user", "content": user},
                        {"role": "assistant", "content": row["output"]}]
            start = len(calls)
            want = patched._original(messages, tokenizer, max_length=512)
            n_ref += len(calls) - start
            start = len(calls)
            got = patched(messages, tokenizer, max_length=512)
            n_new += len(calls) - start
            assert got == want
    finally:
        type(tokenizer).apply_chat_template = real
    assert n_new <= n_ref // 2 + 1


                                                                                                  

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("src, dst", [(torch.bfloat16, torch.float16),
                                      (torch.bfloat16, torch.float32),
                                      (torch.float16, torch.float32),
                                      (torch.float32, torch.float16)])
def test_gpu_conversion_equals_cpu_conversion(src, dst):
    """Every bf16 / fp16 bit pattern (and 2^20 random fp32 ones), converted on the CPU and on
    the GPU: bit-identical except the payload of NaNs (the GPU writes the canonical NaN; the
    checkpoint has none: the whole-model comparison found 0 differing tensors of 1,320)."""
    if src == torch.float32:
        bits = torch.randint(-2**31, 2**31 - 1, (2**20,), dtype=torch.int32)
        bits[:4] = torch.tensor([0, 0x7F800000, -0x00800000, 0x477FE000], dtype=torch.int32)
    else:
        bits = torch.arange(-2**15, 2**15, dtype=torch.int32).to(torch.int16)
    values = bits.view(src)
    cpu = values.to(device="cuda", dtype=dst).cpu()                                          
    gpu = values.to(device="cuda").to(dtype=dst).cpu()                    
    width = torch.int16 if dst == torch.float16 else torch.int32
    nan = torch.isnan(values.float())
    assert torch.equal(torch.isnan(cpu), nan) and torch.equal(torch.isnan(gpu), nan)
    assert torch.equal(cpu.view(width)[~nan], gpu.view(width)[~nan])
    assert (~nan).sum() > 0.9 * nan.numel()


def test_materialize_copy_patch_matches_the_original():
    pytest.importorskip("transformers")
    from transformers import core_model_loading as loading

    original = loading._materialize_copy
    assert startup.patch_load_conversion()
    patched = loading._materialize_copy
    if getattr(original, "_solution_patched", False):
        original = None
    tensor = torch.randn(64, 48).to(torch.bfloat16)
    if torch.cuda.is_available():
        out = patched(tensor, "cuda:0", torch.float16)
        assert out.device.type == "cuda" and out.dtype == torch.float16
        assert torch.equal(out.cpu(), tensor.to(torch.float16))
    out = patched(tensor, "cpu", torch.float32)                                        
    assert torch.equal(out, tensor.to(torch.float32))
    assert patched(tensor, None, None) is not None
    if original is not None:
        loading._materialize_copy = original                                         
