from __future__ import annotations

import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import threading

import jax

_CUDA_TARGET_PREPARE = "rodas5_cudss_prepare"
_CUDA_TARGET_SOLVE = "rodas5_cudss_solve"
_CUDA_TARGET_PREPARE_DYNAMIC = "rodas5_cudss_prepare_dynamic"
_CUDA_TARGET_SOLVE_DYNAMIC = "rodas5_cudss_solve_dynamic"
_CUDA_TARGET_STEP = "rodas5_cudss_step"
_CUDA_TARGET_STEP_DYNAMIC = "rodas5_cudss_step_dynamic"

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BUILD_DIR = _ROOT / "build" / "rodas5_cudss_ffi"
_LIB_NAME = "librodas5_cudss_ffi.so"
_LIB_PATH = _BUILD_DIR / _LIB_NAME

_REGISTER_LOCK = threading.Lock()
_REGISTERED = False
_LIB = None
_LIVE_CONTEXTS = {}
_CONTEXTS_BY_KEY = {}


def is_enabled() -> bool:
    return os.environ.get("RODAS5_V2_ENABLE_CUDSS", "").lower() not in {"0", "false", "no", "off"}


def _native_sources() -> tuple[pathlib.Path, ...]:
    sources = [
        _ROOT / "CMakeLists.txt",
        _ROOT / "native" / "rodas5_cudss_ffi.cc",
    ]
    cuda_helper = _ROOT / "native" / "cudss_diagonal_kernel.cu"
    if cuda_helper.exists():
        sources.append(cuda_helper)
    return tuple(sources)


def _needs_rebuild() -> bool:
    if not _LIB_PATH.exists():
        return True
    lib_mtime = _LIB_PATH.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > lib_mtime for path in _native_sources())


def _site_packages() -> pathlib.Path:
    return pathlib.Path(next(path for path in sys.path if "site-packages" in path))


def _cuda_include_dirs() -> list[pathlib.Path]:
    site = _site_packages()
    candidates = [
        site / "jaxlib" / "include",
        site / "nvidia" / "cu12" / "include",
        site / "nvidia" / "cuda_runtime" / "include",
        site / "nvidia" / "cuda_nvcc" / "include",
        site / "nvidia" / "cusparse" / "include",
    ]
    return [path for path in candidates if path.exists()]


def _cuda_library_dirs() -> list[pathlib.Path]:
    site = _site_packages()
    candidates = [
        site / "nvidia" / "cu12" / "lib",
        site / "nvidia" / "cuda_runtime" / "lib",
        site / "nvidia" / "cusparse" / "lib",
        site / "nvidia" / "cublas" / "lib",
    ]
    return [path for path in candidates if path.exists()]


def _cmake_executable() -> str | None:
    direct = pathlib.Path(sys.executable).resolve().parent / "cmake"
    if direct.exists():
        return str(direct)
    return shutil.which("cmake")


def _nvcc_executable() -> str | None:
    direct = pathlib.Path("/usr/bin/nvcc")
    if direct.exists():
        return str(direct)
    direct = pathlib.Path("/usr/local/cuda/bin/nvcc")
    if direct.exists():
        return str(direct)
    return shutil.which("nvcc")


def _cuda_host_compiler() -> str | None:
    candidates = [
        pathlib.Path("/usr/bin/g++-13"),
        pathlib.Path("/usr/bin/x86_64-linux-gnu-g++-13"),
        pathlib.Path("/usr/bin/g++-14"),
        pathlib.Path("/usr/bin/x86_64-linux-gnu-g++-14"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return shutil.which("g++-13") or shutil.which("g++-14")


def _build_native_library() -> pathlib.Path | None:
    cmake = _cmake_executable()
    ninja = shutil.which("ninja") or str(pathlib.Path(sys.executable).resolve().parent / "ninja")
    include_dirs = _cuda_include_dirs()
    lib_dirs = _cuda_library_dirs()
    nvcc = _nvcc_executable()
    host_compiler = _cuda_host_compiler()
    if cmake is None or not include_dirs or not lib_dirs:
        return _LIB_PATH if _LIB_PATH.exists() else None

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    configure_cmd = [
        cmake,
        "-S",
        str(_ROOT),
        "-B",
        str(_BUILD_DIR),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        f"-DJAX_FFI_INCLUDE_DIR={include_dirs[0]}",
        f"-DCUDSS_INCLUDE_DIR={include_dirs[1]}",
        f"-DCUDA_RUNTIME_INCLUDE_DIR={include_dirs[2]}",
        f"-DCUDA_NVCC_INCLUDE_DIR={include_dirs[3]}",
        f"-DCUSPARSE_INCLUDE_DIR={include_dirs[4]}",
        f"-DCUDSS_LIBRARY_DIR={lib_dirs[0]}",
        f"-DCUDA_RUNTIME_LIBRARY_DIR={lib_dirs[1]}",
        f"-DCUSPARSE_LIBRARY_DIR={lib_dirs[2]}",
        f"-DCUBLAS_LIBRARY_DIR={lib_dirs[3]}",
    ]
    if nvcc is not None:
        configure_cmd.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    if host_compiler is not None:
        configure_cmd.append(f"-DCMAKE_CUDA_HOST_COMPILER={host_compiler}")
    subprocess.run(configure_cmd, check=True, cwd=_ROOT)
    subprocess.run([cmake, "--build", str(_BUILD_DIR), "--target", "rodas5_cudss_ffi"], check=True, cwd=_ROOT)
    return _LIB_PATH if _LIB_PATH.exists() else None


def _load_library():
    global _LIB
    if _LIB is not None:
        return _LIB
    if _needs_rebuild():
        built = _build_native_library()
        if built is None:
            return None
    lib = ctypes.cdll.LoadLibrary(str(_LIB_PATH))
    lib.rodas5_cudss_create_context.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
    lib.rodas5_cudss_create_context.restype = ctypes.c_uint64
    lib.rodas5_cudss_reset_active.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_reset_active.restype = None
    lib.rodas5_cudss_destroy_context.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_destroy_context.restype = None
    lib.rodas5_cudss_analysis_count.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_analysis_count.restype = ctypes.c_uint64
    lib.rodas5_cudss_factorization_count.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_factorization_count.restype = ctypes.c_uint64
    lib.rodas5_cudss_solve_count.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_solve_count.restype = ctypes.c_uint64
    lib.rodas5_cudss_host_prepare_slow_path_count.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_host_prepare_slow_path_count.restype = ctypes.c_uint64
    lib.rodas5_cudss_device_prepare_fast_path_count.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_device_prepare_fast_path_count.restype = ctypes.c_uint64
    lib.rodas5_cudss_graph_capture_count.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_graph_capture_count.restype = ctypes.c_uint64
    lib.rodas5_cudss_graph_replay_count.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_graph_replay_count.restype = ctypes.c_uint64
    lib.rodas5_cudss_mt_analysis_enabled.argtypes = [ctypes.c_uint64]
    lib.rodas5_cudss_mt_analysis_enabled.restype = ctypes.c_int
    lib.rodas5_cudss_has_fused_step.argtypes = []
    lib.rodas5_cudss_has_fused_step.restype = ctypes.c_int
    _LIB = lib
    return lib


def register_cuda_targets() -> bool:
    global _REGISTERED
    with _REGISTER_LOCK:
        if _REGISTERED:
            return True
        lib = _load_library()
        if lib is None:
            return False
        jax.ffi.register_ffi_target(
            _CUDA_TARGET_PREPARE,
            jax.ffi.pycapsule(lib.rodas5_cudss_prepare),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _CUDA_TARGET_PREPARE_DYNAMIC,
            jax.ffi.pycapsule(lib.rodas5_cudss_prepare_dynamic),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _CUDA_TARGET_SOLVE,
            jax.ffi.pycapsule(lib.rodas5_cudss_solve),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _CUDA_TARGET_SOLVE_DYNAMIC,
            jax.ffi.pycapsule(lib.rodas5_cudss_solve),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _CUDA_TARGET_STEP,
            jax.ffi.pycapsule(lib.rodas5_cudss_step),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _CUDA_TARGET_STEP_DYNAMIC,
            jax.ffi.pycapsule(lib.rodas5_cudss_step_dynamic),
            platform="CUDA",
        )
        _REGISTERED = True
        return True


class CuDSSContext:
    def __init__(self, handle: int, lib) -> None:
        self.handle = int(handle)
        self._lib = lib

    @property
    def counters(self) -> dict[str, int]:
        return {
            "analysis": int(self._lib.rodas5_cudss_analysis_count(self.handle)),
            "factorization": int(self._lib.rodas5_cudss_factorization_count(self.handle)),
            "solve": int(self._lib.rodas5_cudss_solve_count(self.handle)),
            "host_prepare_slow_path_count": int(
                self._lib.rodas5_cudss_host_prepare_slow_path_count(self.handle)
            ),
            "device_prepare_fast_path_count": int(
                self._lib.rodas5_cudss_device_prepare_fast_path_count(self.handle)
            ),
            "graph_capture_count": int(
                self._lib.rodas5_cudss_graph_capture_count(self.handle)
            ),
            "graph_replay_count": int(
                self._lib.rodas5_cudss_graph_replay_count(self.handle)
            ),
            "mt_analysis_enabled": bool(
                self._lib.rodas5_cudss_mt_analysis_enabled(self.handle)
            ),
        }


def create_context(n_vars: int, batch_size: int, precision_code: int) -> CuDSSContext | None:
    if not is_enabled() or jax.default_backend() != "gpu":
        return None
    if not register_cuda_targets():
        return None
    key = (int(n_vars), int(batch_size), int(precision_code))
    if key in _CONTEXTS_BY_KEY:
        return _CONTEXTS_BY_KEY[key]
    lib = _load_library()
    if lib is None:
        return None
    handle = int(lib.rodas5_cudss_create_context(n_vars, batch_size, precision_code))
    if handle == 0:
        return None
    ctx = CuDSSContext(handle, lib)
    _LIVE_CONTEXTS[handle] = ctx
    _CONTEXTS_BY_KEY[key] = ctx
    return ctx


def reset_active(handle: int) -> None:
    lib = _load_library()
    if lib is None:
        return
    lib.rodas5_cudss_reset_active(int(handle))


def has_fused_step() -> bool:
    lib = _load_library()
    if lib is None:
        return False
    return bool(lib.rodas5_cudss_has_fused_step())
