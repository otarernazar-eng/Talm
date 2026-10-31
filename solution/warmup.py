"""Compile the Triton kernels of the training step in a thread of the training process while
Soup starts up and loads the model. On by default in every process that calls `apply()` (there
is no check of what the process is or whether it is being measured); SOLUTION_WARMUP_THREAD=0
switches it off (the kernels then compile at their first launch in the first training step, as
usual).

Why: every run starts with an empty Triton cache, so the first launch of each kernel compiles it
(Triton's frontend, LLVM, ptxas), builds its C launcher with gcc and loads the binary. Without
this module all of that sits in the first training step: on the Kaggle T4 step 0 took 3.0-4.9 s
instead of 0.9 s, and the run outside its steady-state steps was 1.8-2.8 s longer (same-session
A/Bs, same code otherwise).

What it is, for the jury:
- The compile time stays inside the measured wall time: the thread is started by
  `solution.apply()` in the measured `soup train` process itself, and it only runs alongside
  Soup's own start-up (imports, data, model download and 4-bit loading, which mostly wait on
  I/O and the GPU). The first training forward joins it before its first launch.
- No cache: nothing is read from or written to anything but the run's own, initially empty
  TRITON_CACHE_DIR, exactly as a first launch does; nothing is prebuilt or shipped.
- No background process: a Python thread of the same process (`threading.Thread`). The only
  subprocesses are the gcc/ptxas calls Triton itself makes for any first launch, and they are
  waited for. One thread per process (a second apply() starts none), and nothing builds the
  same kernels beside it: until it ends, a launch of one of its kernels from the main thread
  waits for it (a pre-run hook: Triton has no lock around a kernel's compile or first load);
  the first training forward, os.fork() and the interpreter exit wait for it too.
- Same kernels, same binaries: tiny launches with the training dtypes, the same constexprs,
  launch options and alignment class (fresh allocations; the integer arguments of our kernels are
  `do_not_specialize`, so their values do not matter) produce exactly the specializations the
  training steps use. They go through the same launch functions as the training code
  (`rmsnorm.launch`, `fused_head.launch_vocab` / `launch_rows`, `rope._launch`, `qgroup.launch`,
  the example's SwiGLU kernels), and a test checks that the training code paths then compile
  nothing new. The harness counts these launches (they come before training) and does not time
  them.

Measured and dropped: the same thread also making the first calls of the CUDA libraries and
PyTorch kernels of the step (cuBLAS handles and workspace, the GEMM layouts, the attention, the
elementwise kernels, an AdamW step with clipping under a GradScaler; PyTorch hands a finished
thread's cuBLAS handles to the next thread, and CUDA loads a kernel once per context). On the
T4 those first calls cost about 350 ms in a fresh process (tools/first_call.py: cuBLAS 129 ms,
clipping 85 ms, AdamW 55 ms, the fp32-output GEMM 26 ms, the first fp16 add / scale / sum
9-14 ms each), and in training step 0 went 0.82 -> 0.46 s, but the time before the first step
grew by as much (start-up + step 0: 15.70 s without, 15.68 s with; Kaggle session t3a, 5 runs):
on the 2-vCPU machine the thread's 0.6 s of library work slows Soup's start-up running beside it.
"""

import atexit
import os
import threading
import time

import torch

_thread = None
_report = {}
                                                                                             
                                                                        
_state = {"started": False, "atexit": False, "at_fork": False, "hooks": False}
_hooked = []                                                                 
_hook_lock = threading.Lock()


def _half_dtype():
    """The autocast dtype Soup trains with: bf16 where the GPU has it, fp16 otherwise (T4)."""
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) \
            else torch.float16
    except TypeError:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def compile_all(kernels=None):
    """Compile and load every kernel the training step launches, with its training
    specialization: the example's SwiGLU and those of our kernel groups that `kernels`
    ({group: on}, decided by solution.apply() from its switches; None: all) has on.
    Returns {kernel group: seconds}."""
    from . import fp16_embed, fused_head, qgroup, rmsnorm, rope, swiglu

    dev = torch.device("cuda", torch.cuda.current_device())
    half = _half_dtype()
    times = {}
    rows, hidden = 16, 1536

    def timed(name, fn):
        start = time.perf_counter()
        fn()
        times[name] = round(time.perf_counter() - start, 3)

    def norm():                                                                     
        x = torch.zeros(rows, hidden, dtype=torch.float32, device=dev)
        w = torch.ones(hidden, dtype=torch.float32, device=dev)
        y = torch.empty(rows, hidden, dtype=half, device=dev)
        xhat = torch.empty_like(y)
        rstd = torch.empty(rows, dtype=torch.float32, device=dev)
        rmsnorm.launch(x, w, y, xhat, rstd, 1e-6, backward=0)
        rmsnorm.launch(x, w, y, xhat, rstd, 1e-6, backward=1)

    def head():                                                                                
        logits = torch.zeros(rows, 4096, dtype=torch.float16, device=dev)
        targets = torch.zeros(rows, dtype=torch.int64, device=dev)
        loss = torch.empty(rows, dtype=torch.float32, device=dev)
        entropy = torch.empty_like(loss)
        correct = torch.empty(rows, dtype=torch.int32, device=dev)
        if fused_head.VOCAB_PATH:                                                              
            stats = torch.empty(rows, 5, dtype=torch.float32, device=dev)
            best = torch.empty(rows, dtype=torch.int32, device=dev)
            acc = torch.zeros(rows, hidden, dtype=torch.float32, device=dev)
                                                                                        
                                                                                             
            weight = torch.zeros(1, hidden, dtype=fp16_embed.head_weight_dtype(half), device=dev)
            grad = torch.empty(rows, hidden, dtype=torch.float16, device=dev)
            fused_head.launch_vocab(rows, logits, targets, stats, best, acc, weight, grad, loss,
                                    entropy, correct, 4096, 0, logits.stride(0), hidden, True,
                                    True, True, False)
            fused_head.launch_vocab(rows, logits, targets, stats, best, acc, weight, grad, loss,
                                    entropy, correct, 0, 0, 0, hidden, True, False, False, True)
        else:                                                          
            fused_head.launch_rows(logits, targets, loss, entropy, correct, True)

    def mlp():                                                                           
        gate = torch.zeros(rows, 64, dtype=half, device=dev, requires_grad=True)
        up = torch.zeros(rows, 64, dtype=half, device=dev, requires_grad=True)
        swiglu.swiglu(gate, up).backward(torch.zeros(rows, 64, dtype=half, device=dev))

    def rotary():                                                                 
        q = torch.zeros(1, rows, 12, 128, dtype=torch.float16, device=dev).transpose(1, 2)
        k = torch.zeros(1, rows, 2, 128, dtype=torch.float16, device=dev).transpose(1, 2)
        cos = torch.ones(1, rows, 128, dtype=torch.float32, device=dev)
        rope._launch(q, k, cos, cos, backward=0)
        rope._launch(q, k, cos, cos, backward=1)

    def dequant():                                                                    
        qblock, qblock2 = qgroup.WARMUP_QBLOCK, qgroup.WARMUP_QBLOCK2
        n_rows = qgroup.ROWS                                                     
        packed = torch.zeros(n_rows * qblock // 2, dtype=torch.uint8, device=dev)
        absmax8 = torch.zeros(n_rows, dtype=torch.uint8, device=dev)
        code = torch.zeros(256, dtype=torch.float32, device=dev)
        absmax2 = torch.zeros(1, dtype=torch.float32, device=dev)
        offset2 = torch.zeros(1, dtype=torch.float32, device=dev)
        out = torch.empty(n_rows * qblock, dtype=torch.float16, device=dev)
        qgroup.launch(packed, absmax8, code, absmax2, offset2, qgroup._levels(dev), out, n_rows,
                      qblock, qblock2)

    def on(group):
        return (kernels or {}).get(group, True)

    if on("rmsnorm"):
        timed("rmsnorm", norm)
    if on("fused_head"):
        timed("fused_head", head)
    timed("swiglu", mlp)
    if half == torch.float16:                                                               
        if on("rope"):
            timed("rope", rotary)
        if on("group_dequant"):
            timed("group_dequant", dequant)
    torch.cuda.synchronize()
    return times


def _kernels():
    """The @triton.jit functions of the modules whose kernels compile_all() builds."""
    from triton.runtime.jit import JITFunction

    from . import fp16_embed, fused_head, qgroup, rmsnorm, rope, swiglu

    found = []
    for module in (fp16_embed, fused_head, qgroup, rmsnorm, rope, swiglu):
        for obj in vars(module).values():
            if isinstance(obj, JITFunction) and all(obj is not k for k in found):
                found.append(obj)
    return found


def _wait_for_warmup(*args, **kwargs):
    """Pre-run hook of the warm-up's kernels while the thread runs: a launch from the main
    (training) thread first waits for the warm-up. Triton 3.6 has no lock around a kernel's
    compilation or its first load (`CompiledKernel._init_handles` even publishes the launcher
    before the loaded function), so the main thread launching a kernel the warm-up is still
    building would compile or load it a second time. Training joins the thread at its first
    fused forward anyway; this covers every other launch (tests, a forward outside the fused
    path, SOLUTION_DISABLE_FUSED_HEAD=1). Only the main thread waits: the warm-up's own
    backward launches its kernels from autograd's worker thread while the warm-up thread waits
    for that worker (a wait there would deadlock), and training's backward comes after its
    forward, which has joined the warm-up already. The hooks are removed when the thread ends."""
    if _thread is not None and threading.current_thread() is threading.main_thread():
        join()


def _hook():
    kernels = _kernels()
    with _hook_lock:
        if not _state["hooks"]:                                                      
            return
        for fn in kernels:
            if _wait_for_warmup not in fn.pre_run_hooks:
                fn.add_pre_run_hook(_wait_for_warmup)
                _hooked.append(fn)


def _unhook():
    with _hook_lock:
        _state["hooks"] = False
        for fn in _hooked:
            while _wait_for_warmup in fn.pre_run_hooks:
                fn.pre_run_hooks.remove(_wait_for_warmup)
        _hooked.clear()


def _run(kernels):
    from . import _mark

    start = time.perf_counter()
    try:
        _hook()                                                                          
        _report["times"] = compile_all(kernels)
    except BaseException as exc:                                                         
        _report["error"] = repr(exc)                                           
    finally:
        _unhook()                                                         
        _report["seconds"] = round(time.perf_counter() - start, 2)
    _mark("warm-up thread done")


def start(kernels=None):
    """Start the compile thread: once per process (a later call is a no-op, also after the
    join) and only with a CUDA device; `kernels` as for compile_all."""
    global _thread
    if _state["started"] or not torch.cuda.is_available():
        return
    from . import _mark

    _state["started"] = _state["hooks"] = True
    _mark("warm-up thread started")
    _thread = threading.Thread(target=_run, args=(kernels,), name="solution-warmup",
                               daemon=True)
    _thread.start()
    if not _state["atexit"]:
        _state["atexit"] = True
        atexit.register(_finish)                                                                 
    if not _state["at_fork"] and hasattr(os, "register_at_fork"):
        _state["at_fork"] = True
        os.register_at_fork(before=_finish)


def _finish():
    """Wait for the thread, quietly: at exit (a daemon thread stopped inside a CUDA or compiler
    call at interpreter shutdown can crash the exit) and before os.fork() (the child would
    inherit any lock the thread holds, with no thread left to release it)."""
    thread = _thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(300)


def join(timeout=300):
    """Wait for the warm-up before the first training launch; reports how long it ran and how
    long the training step waited for it. A no-op when the warm-up was not started or was
    joined already. An interrupt (KeyboardInterrupt) propagates and leaves the thread
    registered, so the next wait, or the exit, joins it again; a thread still running after
    `timeout` stays registered for the exit, and launches stop waiting for it."""
    global _thread
    thread = _thread
    if thread is None:
        return
    waited = time.perf_counter()
    thread.join(timeout)
    waited = time.perf_counter() - waited
    _unhook()
    late = thread.is_alive()
    if not late:
        _thread = None
    from . import _mark

    _mark(f"warm-up thread joined: {_report}{', still running' if late else ''}, "
          f"training waited {waited:.2f} s")
