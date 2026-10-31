"""Work outside the training loop that the run pays for and never uses.

About 40% of the measured `soup train` wall time is outside the 64 training steps: imports,
dataset preparation, model loading, saving. Each patch here removes work whose result the run
never reads; none of them changes the model, the data, the batches, the LoRA set-up, the
optimizer or anything Soup writes. They run from `solution.apply()` (table `STARTUP` in
`__init__.py`), each behind its own switch, and `SOLUTION_DISABLE_STARTUP=1` turns all off.
Seconds are from the Kaggle T4 machine (2 vCPUs, about 1,000 installed Python packages), where
these costs are several times larger than on a desktop.

Measured on the T4 in one Kaggle session (st5-a2b9): `soup train` takes 44.36 s with patches
1-5 (6 and 7 came later, measured below) and 50.67 s with SOLUTION_DISABLE_STARTUP=1 (-6.3 s;
time outside the training loop 17.1 vs 23.5 s), with the same peak memory (4,793 vs
4,791 MiB); in that session's jury pair all ten checks pass (speedup 1.718x, eval loss
+0.007%). The logged metrics with and without the patches differ only by the run-to-run noise
of identical code (largest relative loss difference 7.7e-4, against 1.2e-3 between two runs of
the same code). The per-patch numbers below come from the wall-clock timeline (tools/timeline.py,
a dev tool) and the Python import times (-X importtime) of the run before and after; the run's
total import time went from 12.97 to 8.80 s with patches 1, 3 and 4.

1. Shared package mapping (`import_trl`, SOLUTION_DISABLE_SHARED_MAPPING). When trl is imported,
   `trl.import_utils` calls `importlib.metadata.packages_distributions()`, which opens the
   metadata and the file list of every installed distribution: 2.6 s on the T4 machine.
   transformers makes the same call when it is imported (by the harness, before `apply()`) and
   keeps the result in `transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING`. While trl
   is imported, the function returns a copy of that mapping. Nothing is installed between the
   two imports (a few seconds apart in the same process), so trl gets exactly the mapping the
   call would compute (asserted in test_startup.py); trl only reads it to look up versions of
   optional packages. Measured: the trl import 2.57 -> 0.0 s.
2. Tokenizer digest for dataset fingerprints (`patch_tokenizer_fingerprint`,
   SOLUTION_DISABLE_FAST_FINGERPRINT). `Dataset.map` fingerprints the mapped function with a
   pure-Python dill pickler, and Soup's row formatter closes over the tokenizer, so the map that
   tokenizes the 256 rows first pickles the tokenizer's 151,643-entry vocabulary and 151,387
   merges object by object: 2.7 s for that map on the T4 machine, most of it this hashing. The
   fingerprint is only a cache key: a dataset built in memory, as Soup's is, is never written to
   or read from the cache, and nothing in Soup or TRL reads the key. The patched reducer pickles
   the same tokenizer state with those two containers replaced by an xxh64 digest of their C
   pickle, so the key is still a deterministic function of the tokenizer's content (a different
   string). The mapped rows are identical (test_startup.py; all 256 rows of the task's data
   with the Qwen tokenizer as well). Measured: Soup's tokenizing map 2.75 -> 0.79 s.
3. Deferred `torch.compile` in bitsandbytes (`import_bitsandbytes`,
   SOLUTION_DISABLE_LAZY_COMPILE). At import, bitsandbytes wraps its pure-PyTorch fallback
   kernels (the "default" backend, for CPU tensors) in `torch.compile`; building those wrappers
   imports TorchInductor and functorch (`compile_fx`, `select_algorithm`, `lowering`, ...),
   about 1.1 s there, and on CUDA the wrapped functions are never called. While bitsandbytes is
   imported, `torch.compile(fn, **kw)` returns a function that calls the real
   `torch.compile(fn, **kw)` on its first call. `torch.compile` itself compiles on the first
   call anyway, so a call behaves exactly as before; only the import of the compiler stack moves
   to the moment it is first needed (never, in this run). Restored right after the import.
   Measured: `torch._inductor.compile_fx` and functorch's AOT autograd are no longer imported,
   about 1.0 s (TorchDynamo itself is still imported, by transformers).
4. scikit-learn deferred (`import_without_sklearn`, SOLUTION_DISABLE_LAZY_SKLEARN). When
   scikit-learn is installed (it is on Kaggle), transformers' assisted-generation module
   (`roc_curve`, for a speculative-decoding heuristic) and its deprecated GLUE metrics
   (`f1_score`, `matthews_corrcoef`, `pearsonr`, `spearmanr`) import `sklearn.metrics` and
   `scipy.stats` at import: 1.1 s there, for code a training run never calls. Those two modules
   are imported once while `is_sklearn_available()` answers False, then get the real check back
   and, for each name they would have imported, a function that imports the real one on its
   first call and calls it, so any later use behaves as before. Measured: `sklearn.metrics` and
   `scipy.stats` are no longer imported, about 1.1 s.
5. Checkpoint conversion on the GPU (`patch_load_conversion`, SOLUTION_DISABLE_GPU_CONVERT). On a
   T4 Soup loads the bf16 checkpoint as fp16. transformers' loader reads each tensor into CPU
   memory and calls `tensor.to(device="cuda", dtype=float16)`, and for a blocking host-to-device
   copy PyTorch converts the dtype on the CPU first: 1.5 billion conversions on two vCPUs. The
   patched loader copies the tensor as stored and converts it on the GPU. bf16 -> fp16 rounds to
   nearest-even on both devices: every bf16 bit pattern converts identically (NaN payloads
   aside; the checkpoint has no NaN) and the loaded 4-bit model is identical in all 1,320
   parameter, buffer and quant-state tensors (checked against the unpatched load). Costs a
   transient GPU copy of one tensor in its stored dtype (at most the 445 MiB embedding) while
   loading, far below the training peak: right after the load the T4 had 1,133 MiB peak
   allocated and 1,154 MiB peak reserved. Measured: the model load 2.26 -> 1.33 s.
6. SciPy deferred in the object-detection losses (`import_losses_without_scipy`,
   SOLUTION_DISABLE_LAZY_SCIPY). Every model import goes through `transformers.modeling_utils`
   -> `transformers.loss.loss_utils`, which imports the DETR-family losses, and six of those
   import SciPy's `linear_sum_assignment` (the Hungarian matcher of object detection) when
   SciPy is installed (it is on Kaggle): `scipy.optimize` with `scipy.linalg` and `scipy.sparse`
   behind it, for losses a causal LM never computes. As in 4: the six modules are imported once
   while `is_scipy_available()` answers False, then get the real check back and a lazy
   `linear_sum_assignment`. `scipy.optimize` is then never imported in the run (only SciPy's
   14 ms top-level package still is, by nibabel, which `datasets` imports).
7. boto3 deferred in accelerate (`import_accelerate_without_boto3`, SOLUTION_DISABLE_LAZY_BOTO3).
   `accelerate.commands.config.sagemaker` (imported with every `import accelerate`) imports
   boto3 when it is installed (it is on Kaggle), with botocore, urllib3's pyOpenSSL layer and
   cryptography behind it; it only uses boto3 to create IAM roles in `accelerate config`. While
   accelerate is imported, `sys.modules["boto3"] = None` makes `find_spec("boto3")` answer None
   (nothing else in accelerate reads it; restored right after), and the module's `boto3` name
   is a stand-in that imports the real boto3 on first use. boto3 is then never imported.
   Measured for 6 and 7 together in one Kaggle session (hx1-66e7, 8 runs alternating with and
   without them, ABBA ABBA): apply() 5.26 -> 4.68 s (every run with them, 4.52-4.80 s, below
   every run without them, 5.01-5.36 s); Soup's set-up after apply() unchanged (5.09 vs
   5.18 s, so the imports did not move later); 537 fewer modules imported and 0.67 s less
   import time (-X importtime runs); median wall 42.3 vs 42.7 s (the first, cold run of the
   session excluded), within the wall-time noise of single runs. The same logged metrics,
   eval loss 2.0498 in both arms.

Patches 8-11 come from a cProfile of the set-up window between `apply()` and the first training
step (tools/startup_profile.py, a dev tool: the harness's own sequence, stopped after step 0,
with gc.callbacks timing every garbage collection). Measured together on the T4 in one Kaggle
session (w3su-abd2, 8 runs alternating with and without the four, ABBA ABBA): `soup train`
35.27 s median with them (35.25-35.29 s; the session's first, cold run 37.40 s) and 36.86 s
without (36.45-37.00 s): -1.6 s, every warm run with them below every run without. Of that,
apply() 3.90 -> 3.64 s, the set-up (first-forward mark minus apply-done mark) 4.41 -> 4.08 s,
and about 1.1 s at the interpreter's shutdown after the last [solution] mark (its full
collections skip the frozen objects, patch 8); the training loop is unchanged (21.6 s). Same
peak in all 8 runs (3,921 MiB nvidia-smi, 3,659.2 MiB max allocated, 3,778 MiB reserved);
identical batches in all 8 (num_tokens, rewind rows and row tokens), and the logged floats
differ between the arms by no more than between runs of the same code (loss: 8.9e-4 relative
across, 8.4e-4 within an arm; T4 kernels with atomics), eval loss 2.0498 in both arms.

8. Import-time objects out of the collector's scans (`freeze_gc`, SOLUTION_DISABLE_GC_FREEZE;
   called at the start and at the end of `apply()`). A full (generation-2) collection walks
   every container object Python tracks, and after the imports those are about 400,000
   objects (modules, functions, classes, their dicts) that live until the process exits. Two
   full collections run in the set-up window (one is accelerate's explicit `gc.collect()` in
   `free_memory()` at the start of `Trainer.train`, one automatic) and one during `apply()`.
   `gc.freeze()` moves the objects alive at that moment to the collector's permanent
   generation, which collections skip; objects created later are tracked and collected as
   before, at the same collections (the full-collection trigger counts are not changed by a
   freeze). What it cannot do any more is free a cycle of objects that were alive at the
   freeze and become garbage later: host memory only (no CUDA tensor exists at the start of
   apply(); CUDA memory at the first step is the same, 1,980.9 vs 1,981.0 MiB max allocated,
   1,540.3 MiB allocated, 1,578 MiB reserved in both, locally). No value is computed
   differently. The interpreter's own full collections at shutdown skip them too (the process
   exits; the operating system takes the memory back), after every output is written.
   Measured on the T4 (startup_profile runs, gc.callbacks, 2 runs per arm): collections during
   apply() 0.45 -> 0.19 s, in the set-up window 0.28 -> 0.05 s (accelerate's `gc.collect()`
   0.24 -> 0.02 s); tracked objects at the first step 561,456 -> 60,600; plus the shutdown
   share of the -1.6 s above (wall minus the process-exit mark: 6.80 -> 5.69 s).
9. One nvidia-smi query per process (`share_gpu_query`, SOLUTION_DISABLE_GPU_QUERY_ONCE).
   Soup builds a `TrainingArguments` and converts it to an `SFTConfig`; each one's
   `__post_init__` resets accelerate's state and re-creates `PartialState`, whose CUDA P2P
   check (`check_cuda_p2p_ib_support`) spawns `nvidia-smi --query-gpu=count,name` to list the
   machine's GPUs: two subprocesses (0.12 s each locally). `get_gpu_info()` now answers the
   second time with a copy of the first answer (the GPUs of a machine do not change between
   two calls a second apart; a failing call raises as before and is not remembered).
   Measured on the T4: the two configs 0.07 -> 0.035 s (nvidia-smi is fast there): within the
   wall's noise.
10. One NF4 code table (`share_4bit_code`, SOLUTION_DISABLE_NF4_CODE_ONCE). For each of the 196
   weights it quantizes, bitsandbytes' `quantize_4bit` calls `get_4bit_type("nf4", device)`,
   which builds the same 16-value table from a Python list with `torch.tensor(..., device=
   "cuda")` (a blocking host-to-device copy, a wait for the GPU) and normalizes it with three
   more launches. The table is built once per (type, device, blocksize) by that same function
   and every call returns its own clone (a device-to-device copy, no wait): bit-identical
   tables, still one tensor per quantization state, so any kernel that reads one (training
   dequantizes with tables built into the kernels; bitsandbytes' 4-bit GEMV reads it) sees
   the same values. Measured on the T4: the model load 1.41 -> 1.32 s (2 runs per arm;
   `quantize_4bit` 0.24 -> 0.12 s under cProfile).
11. One render per conversation (`share_chat_renders`, SOLUTION_DISABLE_RENDER_ONCE). Soup's
   `build_assistant_only_labels` (one call per training row) first asks the chat template for
   an assistant mask; Qwen2.5's template has no `{% generation %}` block, so the mask is all
   zeros, the result is discarded (`None`) and the incremental fallback renders and tokenizes
   the whole conversation, then each growing prefix, the last of which is the whole
   conversation again: four renders per row, two of them discarded or repeated. Within one
   call, a conversation rendered again gets a copy of its first render (the template is a pure
   function of the conversation; templates that read the clock are left alone), and the mask
   attempt is skipped when its result is known to be `None`: the template has no generation
   block, the conversation has an assistant turn, and the attempt was made for real once in
   this process (so transformers' one-time warning is printed as before). Conversations
   without an assistant turn take the mask path as before. Labels identical for all 256 rows
   (test_startup.py); locally the 256 rows' label building 0.20 -> 0.09 s. On the T4 the
   session above measured a first version with plain function wrappers: label building 0.85
   -> 0.39 s under cProfile, but datasets' fingerprint of the row formatter 0.43 -> 0.99 s
   (dill's recursive global search walked into the wrappers and the originals they closed
   over), so the map was only 0.80 -> 0.77 s. The wrappers are now `_LossMaskStandIn` objects,
   which that search does not enter (locally the formatter's fingerprint 0.17-0.25 s ->
   0.15-0.17 s, no longer above the unpatched one); not re-measured on the T4.
"""

import contextlib
import functools
import sys

from .state import DISABLED, fallback

_state = {"tokenizer_reducer": None}


# 1. Shared package mapping ----------------------------------------------------------------------

@contextlib.contextmanager
def shared_package_mapping():
    """Within the block, `importlib.metadata.packages_distributions()` returns a copy of the
    mapping transformers computed with it at import time (yields whether it does)."""
    import importlib.metadata as metadata

    import_utils = sys.modules.get("transformers.utils.import_utils")
    mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    real = metadata.packages_distributions
    if not isinstance(mapping, dict) or not mapping or getattr(real, "_solution_shared", False):
        yield False
        return

    def packages_distributions():
        return {name: list(dists) for name, dists in mapping.items()}

    packages_distributions._solution_shared = True
    metadata.packages_distributions = packages_distributions
    try:
        yield True
    finally:
        metadata.packages_distributions = real


def import_trl():
    """Import trl with `shared_package_mapping`; False if it was loaded or is missing."""
    if "trl" in sys.modules:
        return False
    try:
        with shared_package_mapping() as shared:
            import trl  # noqa: F401
    except Exception:  # noqa: BLE001 - no trl here: nothing to import
        return False
    return shared


# 2. Tokenizer digest for dataset fingerprints ---------------------------------------------------

_BIG = 4096  # containers smaller than this are pickled as datasets does


def _content_digest(value):
    """xxh64 of the C pickle of `value` (a dict in its insertion order, i.e. the order of the
    tokenizer file it was read from)."""
    import pickle

    import xxhash

    return xxhash.xxh64(pickle.dumps(value, protocol=4)).hexdigest()


def patch_tokenizer_fingerprint():
    """Make datasets' fingerprint of a tokenizer digest `_vocab` / `_merges` with the C pickler."""
    try:
        from datasets.utils import _dill
    except Exception:  # noqa: BLE001 - no datasets in this process
        return False
    original = getattr(_dill, "_save_transformersPreTrainedTokenizerBase", None)
    if original is None or not hasattr(_dill, "Pickler"):
        return False
    if getattr(original, "_solution_patched", False):
        return True

    def save_tokenizer(pickler, obj):
        state = getattr(obj, "__dict__", None)
        big = {}
        if isinstance(state, dict):
            for key in ("_vocab", "_merges"):
                value = state.get(key)
                if isinstance(value, (dict, list)) and len(value) >= _BIG:
                    big[key] = value
        if not big or "startup.patch_tokenizer_fingerprint" in DISABLED:
            return original(pickler, obj)
        try:
            digests = {key: f"<{key} {len(value)} xxh64:{_content_digest(value)}>"
                       for key, value in big.items()}
        except Exception:  # noqa: BLE001 - unpicklable content: pickle it as before
            return original(pickler, obj)
        # datasets' reducer pickles a copy of `obj.__dict__`: put the digests in for that copy.
        state.update(digests)
        try:
            return original(pickler, obj)
        except Exception as exc:  # noqa: BLE001 - state.py: pickled as before, below
            if not fallback("startup.patch_tokenizer_fingerprint", exc):
                raise
        finally:
            state.update(big)
        return original(pickler, obj)

    save_tokenizer._solution_patched = True
    _dill._save_transformersPreTrainedTokenizerBase = save_tokenizer
    for key, value in list(_dill.Pickler.dispatch.items()):
        if value is original:  # tokenizer classes registered before the patch
            _dill.Pickler.dispatch[key] = save_tokenizer
    _state["tokenizer_reducer"] = (original, save_tokenizer)
    return True


# 3. Deferred torch.compile in bitsandbytes ------------------------------------------------------

@contextlib.contextmanager
def deferred_torch_compile():
    """Within the block, `torch.compile(fn, **kw)` on a plain function is created on first call."""
    import torch

    real = torch.compile
    if getattr(real, "_solution_deferred", False):
        yield
        return

    def compile(model=None, **kwargs):  # noqa: A001 - stands in for torch.compile
        if model is None:  # @torch.compile(...) used as a decorator factory
            return lambda fn: compile(fn, **kwargs)
        if isinstance(model, torch.nn.Module) or not callable(model):
            return real(model, **kwargs)
        compiled = []

        @functools.wraps(model)
        def lazy(*args, **kw):
            if not compiled:
                compiled.append(real(model, **kwargs))
            return compiled[0](*args, **kw)

        lazy._solution_deferred_compile = True
        return lazy

    compile._solution_deferred = True
    torch.compile = compile
    try:
        yield
    finally:
        torch.compile = real


def import_bitsandbytes():
    """Import bitsandbytes with its import-time `torch.compile` decorations deferred."""
    if "bitsandbytes" in sys.modules:
        return False
    try:
        with deferred_torch_compile():
            import bitsandbytes  # noqa: F401
    except Exception:  # noqa: BLE001 - no bitsandbytes here: nothing to defer
        return False
    return True


# 4. scikit-learn deferred -----------------------------------------------------------------------

def _lazy_callable(module_name, attr):
    """A function that imports `module_name` on its first call and calls its `attr`."""

    def call(*args, **kwargs):
        import importlib

        return getattr(importlib.import_module(module_name), attr)(*args, **kwargs)

    call.__name__ = call.__qualname__ = attr
    call._solution_lazy = (module_name, attr)
    return call


# Modules that import scikit-learn (and SciPy) at import time when scikit-learn is installed,
# with the names they import under that check; training calls none of them.
SKLEARN_IMPORTERS = {
    # assisted generation: confidence threshold of speculative decoding
    "transformers.generation.candidate_generator": {"roc_curve": "sklearn.metrics"},
    # deprecated GLUE/XNLI metrics, imported with `transformers.data` (DataCollator, by TRL)
    "transformers.data.metrics": {"pearsonr": "scipy.stats", "spearmanr": "scipy.stats",
                                  "f1_score": "sklearn.metrics",
                                  "matthews_corrcoef": "sklearn.metrics"},
}


def _import_with_check_off(name, lazy_names, check):
    """Import module `name` while transformers' `check()` (an `is_*_available` of
    `transformers.utils`) answers False, then give the module the real check back and, for
    each name it would have imported under that check, a function importing the real one on
    its first call."""
    import importlib

    if name in sys.modules:
        return False
    try:
        import transformers.utils as utils

        real = getattr(utils, check)
        if not real():
            return False  # the module skips those imports by itself
    except Exception:  # noqa: BLE001 - another transformers layout: leave it alone
        return False

    def unavailable():
        return False

    setattr(utils, check, unavailable)
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - let the normal import path raise it later
        return False
    finally:
        setattr(utils, check, real)
        # Every module imported meanwhile that kept a reference to the stand-in (normally only
        # `module`) gets the real check back, so its run-time checks see the real answer.
        for loaded in list(sys.modules.values()):
            try:
                namespace = getattr(loaded, "__dict__", None)  # no module __getattr__: no imports
                if isinstance(namespace, dict) and namespace.get(check) is unavailable:
                    namespace[check] = real
            except Exception:  # noqa: BLE001 - an exotic sys.modules entry: not ours
                continue
    for attr, source in lazy_names.items():
        if not hasattr(module, attr):
            setattr(module, attr, _lazy_callable(source, attr))
    return True


def _import_without_sklearn(name, lazy_names):
    return _import_with_check_off(name, lazy_names, "is_sklearn_available")


def import_without_sklearn():
    """Import the modules of SKLEARN_IMPORTERS without their import-time scikit-learn import
    (number of modules imported that way: 0 when scikit-learn is not installed)."""
    return sum(_import_without_sklearn(name, names) for name, names in SKLEARN_IMPORTERS.items())


# 6. SciPy deferred in the object-detection losses -----------------------------------------------

# transformers' DETR-family losses, imported with every model (transformers.modeling_utils ->
# transformers.loss.loss_utils), import SciPy's Hungarian matcher when SciPy is installed. In the
# order they import each other (the first is imported by the others).
SCIPY_IMPORTERS = {
    f"transformers.loss.{name}": {"linear_sum_assignment": "scipy.optimize"}
    for name in ("loss_for_object_detection", "loss_rt_detr", "loss_deformable_detr",
                 "loss_grounding_dino", "loss_lw_detr", "loss_rf_detr")
}


def import_losses_without_scipy():
    """Import the modules of SCIPY_IMPORTERS without their import-time `scipy.optimize` import
    (number of modules imported that way: 0 when SciPy is not installed)."""
    return sum(_import_with_check_off(name, names, "is_scipy_available")
               for name, names in SCIPY_IMPORTERS.items())


# 7. boto3 deferred in accelerate ----------------------------------------------------------------

class _LazyModule:
    """Stands for module `name` in one module's namespace: the first attribute read imports the
    real module (and replaces the stand-in there with it)."""

    def __init__(self, name, owner):
        self._name, self._owner = name, owner

    def __getattr__(self, attr):
        import importlib

        module = importlib.import_module(self._name)
        if getattr(self._owner, self._name, None) is self:
            setattr(self._owner, self._name, module)
        return getattr(module, attr)


def import_accelerate_without_boto3():
    """Import accelerate without the import of boto3 its SageMaker config module makes when
    boto3 is installed; that module (the only user of boto3 in accelerate, for `accelerate
    config` on SageMaker) gets a stand-in that imports boto3 when first used."""
    import importlib.util

    if "accelerate" in sys.modules or "boto3" in sys.modules:
        return False
    try:
        if importlib.util.find_spec("boto3") is None:
            return False  # not installed: accelerate skips the import by itself
    except (ImportError, ValueError):
        return False
    sys.modules["boto3"] = None  # find_spec("boto3") answers None while accelerate is imported
    try:
        import accelerate  # noqa: F401
    except Exception:  # noqa: BLE001 - let the normal import path raise it later
        return False
    finally:
        if "boto3" in sys.modules and sys.modules["boto3"] is None:
            del sys.modules["boto3"]
    sagemaker = sys.modules.get("accelerate.commands.config.sagemaker")
    if sagemaker is not None and "boto3" not in vars(sagemaker):
        sagemaker.boto3 = _LazyModule("boto3", sagemaker)
    return True


# 8. Import-time objects out of the cyclic garbage collector's scans -----------------------------

def freeze_gc():
    """`gc.freeze()`: every object the collector tracks now moves to its permanent generation,
    which later collections do not scan (returns how many objects are frozen; 0 when the
    collector is disabled, then nothing is done)."""
    import gc

    if not gc.isenabled():
        return 0
    gc.freeze()
    return gc.get_freeze_count()


# 9. One nvidia-smi query per process in accelerate ----------------------------------------------

def share_gpu_query():
    """accelerate's `get_gpu_info()` (an `nvidia-smi` subprocess listing the machine's GPUs)
    runs once; later calls get a copy of that first answer."""
    try:
        from accelerate.utils import environment
    except Exception:  # noqa: BLE001 - no accelerate here: nothing to share
        return False
    original = getattr(environment, "get_gpu_info", None)
    if original is None:
        return False
    if getattr(original, "_solution_once", False):
        return True
    answer = []

    @functools.wraps(original)
    def get_gpu_info():
        if not answer:  # a failing first call raises as before and is not remembered
            names, count = original()
            answer.append((list(names), count))
        names, count = answer[0]
        return list(names), count

    get_gpu_info._solution_once = True
    get_gpu_info._original = original
    environment.get_gpu_info = get_gpu_info
    return True


# 10. One NF4 code table per device in bitsandbytes ----------------------------------------------

def share_4bit_code():
    """bitsandbytes' `get_4bit_type(name, device)`: the 16-value code table of a 4-bit type is
    built once per (name, device, blocksize) the way bitsandbytes builds it; every call returns
    its own copy of it (a device-to-device clone)."""
    try:
        from bitsandbytes import functional
    except Exception:  # noqa: BLE001 - no bitsandbytes here
        return False
    original = getattr(functional, "get_4bit_type", None)
    if original is None:
        return False
    if getattr(original, "_solution_once", False):
        return True
    import torch

    tables = {}

    @functools.wraps(original)
    def get_4bit_type(typename, device=None, blocksize=64):
        try:
            key = (typename, torch.device("cuda" if device is None else device), blocksize)
            hash(key)
        except (TypeError, RuntimeError):
            return original(typename, device, blocksize)
        table = tables.get(key)
        if table is None:  # unknown names raise as before and are not remembered
            table = tables[key] = original(typename, device, blocksize)
            return table.clone()
        return table.clone()

    get_4bit_type._solution_once = True
    get_4bit_type._original = original
    get_4bit_type._tables = tables
    functional.get_4bit_type = get_4bit_type
    return True


# 11. One render per conversation in Soup's assistant-only labels ---------------------------------

# A `{% generation %}` block of a chat template (with any whitespace control), as Jinja parses it.
_GENERATION_TAG = r"\{%[-+]?\s*generation\b"
_render_state = {"active": None, "seen": set()}


def _conversation_key(tokenizer, messages):
    """Hashable key of (tokenizer, chat template, conversation), or None if there is none (or
    the template reads the clock, so two renders may differ)."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template or "strftime" in template:
        return None
    try:
        key = (id(tokenizer), template,
               tuple(tuple(sorted(message.items())) for message in messages))
        hash(key)
    except Exception:  # noqa: BLE001 - unhashable message content: render as before
        return None
    return key


def _loss_mask_function(name):
    """`soup_cli.data.loss_mask.<name>` (how a `_LossMaskStandIn` is pickled)."""
    from soup_cli.data import loss_mask

    return getattr(loss_mask, name)


class _LossMaskStandIn:
    """Stands for function `name` of Soup's loss_mask module. A callable object, not a
    function, and pickled by name: datasets fingerprints Soup's row formatter with dill, whose
    recursive search for the globals a function uses walks (and disassembles) every function
    it reaches, including those inside closures; it does not walk into objects. Plain function
    wrappers made that walk reach both the wrappers and the originals they close over (0.43 ->
    0.99 s of fingerprinting under cProfile on the T4, more than the renders saved); with this
    stand-in it stops here."""

    _solution_once = True

    def __init__(self, name, call, original):
        self._name, self._call, self._original = name, call, original
        self.__name__ = self.__qualname__ = name
        self.__doc__ = getattr(original, "__doc__", None)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __reduce__(self):
        return _loss_mask_function, (self._name,)


def share_chat_renders():
    """Soup's `build_assistant_only_labels` (one call per training row): within one call, a
    conversation rendered and tokenized again gets a copy of the first result, and the
    mask-template attempt is not repeated when it cannot succeed (see below)."""
    import re

    try:
        from soup_cli.data import loss_mask  # standard library imports only
    except Exception:  # noqa: BLE001 - no Soup here
        return False
    build = getattr(loss_mask, "build_assistant_only_labels", None)
    tokenize_only = getattr(loss_mask, "_tokenize_only", None)
    with_mask = getattr(loss_mask, "_apply_template_with_mask", None)
    if build is None or tokenize_only is None or with_mask is None:
        return False
    if getattr(build, "_solution_once", False):
        return True
    state = _render_state

    def _tokenize_only(tokenizer, messages):
        renders = state["active"]
        key = None if renders is None else _conversation_key(tokenizer, messages)
        if key is not None and key in renders:
            return list(renders[key])
        ids = tokenize_only(tokenizer, messages)
        if key is not None:
            renders[key] = list(ids)
        return ids

    def _apply_template_with_mask(tokenizer, messages):
        template = getattr(tokenizer, "chat_template", None)
        skip = (state["active"] is not None and isinstance(template, str) and template
                and not re.search(_GENERATION_TAG, template)
                and any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages)
                and (id(tokenizer), template) in state["seen"])
        if skip:
            # Without a generation block the template's assistant mask is all zeros, and an
            # all-zero mask for a conversation with an assistant turn returns None (the
            # incremental fallback runs); the first such call of this run was made for real.
            return None
        out = with_mask(tokenizer, messages)
        if (out is None and state["active"] is not None and isinstance(template, str)
                and any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages)):
            state["seen"].add((id(tokenizer), template))
        return out

    def build_assistant_only_labels(*args, **kwargs):
        outer = state["active"]
        state["active"] = {}  # renders of this row only
        try:
            return build(*args, **kwargs)
        finally:
            state["active"] = outer

    build_assistant_only_labels = _LossMaskStandIn(
        "build_assistant_only_labels", build_assistant_only_labels, build)
    _tokenize_only = _LossMaskStandIn("_tokenize_only", _tokenize_only, tokenize_only)
    _apply_template_with_mask = _LossMaskStandIn(
        "_apply_template_with_mask", _apply_template_with_mask, with_mask)
    loss_mask._tokenize_only = _tokenize_only
    loss_mask._apply_template_with_mask = _apply_template_with_mask
    loss_mask.build_assistant_only_labels = build_assistant_only_labels
    # Soup's row formatter imports the name from loss_mask when it is first imported (after
    # apply() in a run, so it gets this one); rebind it if it was imported already.
    sft_format = sys.modules.get("soup_cli.data.sft_format")
    if getattr(sft_format, "build_assistant_only_labels", None) is build:
        sft_format.build_assistant_only_labels = build_assistant_only_labels
    _state["chat_renders"] = (build, tokenize_only, with_mask)
    return True


# 5. Checkpoint conversion on the GPU ------------------------------------------------------------

def patch_load_conversion():
    """transformers' loader: copy each checkpoint tensor to the GPU in its stored dtype and
    convert it there, instead of converting on the CPU before the copy."""
    import torch

    try:
        from transformers import core_model_loading as loading
    except Exception:  # noqa: BLE001 - another transformers layout: leave it alone
        return False
    original = getattr(loading, "_materialize_copy", None)
    if original is None:
        return False
    if getattr(original, "_solution_patched", False):
        return True

    def _materialize_copy(tensor, device=None, dtype=None):
        if device is None or dtype is None or "startup.patch_load_conversion" in DISABLED:
            return original(tensor, device, dtype)
        try:
            data = tensor[...]  # reads the tensor from the safetensors slice, as the original
            try:
                target = torch.device(device)
            except (TypeError, RuntimeError):
                target = None
            if (target is not None and target.type == "cuda" and data.device.type == "cpu"
                    and data.dtype != dtype and data.dtype.is_floating_point
                    and dtype.is_floating_point):
                return data.to(device=target).to(dtype=dtype)
            return data.to(device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001 - state.py: the original conversion instead
            if not fallback("startup.patch_load_conversion", exc):
                raise
        return original(tensor, device, dtype)

    _materialize_copy._solution_patched = True
    loading._materialize_copy = _materialize_copy
    return True
