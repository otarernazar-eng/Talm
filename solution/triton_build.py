"""Triton's own start-up, which every run pays inside the measured wall time because it starts
with an empty Triton cache.

On by default (SOLUTION_FAST_TRITON_BUILD=0 switches it off): Triton's C helpers are built with
`gcc -O0` instead of `-O3`. At the first launches Triton compiles, with gcc, its `cuda_utils`
module (once per process) and one small C launcher per distinct list of kernel argument types
(six here). Both are thin glue around CUDA driver calls (argument parsing, cuLaunchKernelEx,
module loading), so the optimization level does not matter for them at run time.
- Measured on the Kaggle T4 (tools/bench_triton_startup.py, same session): cuda_utils
  0.39-0.44 s -> 0.19-0.21 s, each launcher 0.27-0.33 s -> 0.15-0.18 s; host time per launch
  of each kernel unchanged (19-26 us either way, within noise); GPU time of our kernels
  unchanged.
- Same process, nothing cached or prebuilt: the builds still happen at the first launches (or
  in the warm-up thread, warmup.py), into the run's own empty TRITON_CACHE_DIR, through
  Triton's own hook for custom builds, `triton.knobs.build.impl`; the command is Triton's
  (`triton/runtime/build.py`, 3.6) with only the optimization level changed.
- Triton's cache keys and cache handling are not touched: the first compile of every process
  still hashes Triton's library and sources as Triton does (about 1.2 s on Kaggle's CPU, done
  inside the warm-up thread).
"""

import os
import shutil
import subprocess
import sysconfig

OPT_LEVEL = "-O0"


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
    """The -O0 build; on an unexpected error (state.py) Triton's own build of the same module,
    and Triton's own builds from then on (the hook is removed)."""
    from triton import knobs
    from triton.runtime import build

    try:
        return _build_o0(name, src, srcdir, library_dirs, include_dirs, libraries)
    except Exception as exc:                               
        from .engaged import FELL_BACK
        from .state import fallback

        if not fallback("triton_build", exc):
            raise
        FELL_BACK["build"] += 1
    if knobs.build.impl is _build:
        knobs.build.impl = None
                                                                                      
    return build._build(name, src, srcdir, library_dirs, include_dirs, libraries, [])


def _build_o0(name, src, srcdir, library_dirs, include_dirs, libraries):
    from triton import knobs
    from triton.runtime import build

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so = os.path.join(srcdir, f"{name}{suffix}")
    cc = os.environ.get("CC")
    if cc is None:
        gcc, clang = shutil.which("gcc"), shutil.which("clang")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler")
    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    include_dirs = include_dirs + [srcdir, py_include_dir, *knobs.build.backend_dirs]
    cmd = [cc, src, OPT_LEVEL, "-shared", "-fPIC", "-Wno-psabi", "-o", so]
    cmd += [build._library_flag(lib) for lib in libraries]
    cmd += [f"-L{d}" for d in library_dirs]
    cmd += [f"-I{d}" for d in include_dirs if d is not None]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    return so


def install():
    """Route Triton's C helper builds through `_build`; False if this Triton has no such hook."""
    try:
        from triton import knobs
        from triton.runtime import build
    except ImportError:
        return False
    if not hasattr(knobs, "build") or not hasattr(build, "_library_flag"):
        return False
    if knobs.build.impl is None:
        knobs.build.impl = _build
    return knobs.build.impl is _build
