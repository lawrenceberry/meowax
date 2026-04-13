"""Kvaerno5 solver via diffrax — ESDIRK method of order 5 for stiff ODEs."""

import hashlib
import inspect
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import diffrax
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402

_CACHE_ROOT = (
    Path(__file__).resolve().parents[3]
    / ".cached_reference_outputs"
    / "diffrax_kvaerno5"
)


def _hash_array(arr):
    """Hash an array by dtype, shape, and raw bytes."""
    arr_np = np.ascontiguousarray(np.asarray(arr))
    hasher = hashlib.sha256()
    hasher.update(str(arr_np.dtype).encode("utf-8"))
    hasher.update(repr(arr_np.shape).encode("utf-8"))
    hasher.update(arr_np.tobytes())
    return hasher.hexdigest()


def _hash_value(value):
    """Hash builtin Python/JAX/numpy values recursively."""
    hasher = hashlib.sha256()
    _update_hash(hasher, value, seen=set())
    return hasher.hexdigest()


def _update_hash(hasher, value, *, seen):
    value_type = type(value)
    hasher.update(
        f"type:{value_type.__module__}.{value_type.__qualname__}".encode("utf-8")
    )

    if value is None:
        hasher.update(b"none")
        return

    if isinstance(value, (bool, int, float, str)):
        hasher.update(repr(value).encode("utf-8"))
        return

    if isinstance(value, bytes):
        hasher.update(value)
        return

    if isinstance(value, types.CodeType):
        hasher.update(value.co_code)
        hasher.update(repr(value.co_names).encode("utf-8"))
        hasher.update(repr(value.co_varnames).encode("utf-8"))
        hasher.update(repr(value.co_freevars).encode("utf-8"))
        hasher.update(repr(value.co_cellvars).encode("utf-8"))
        hasher.update(repr(value.co_argcount).encode("utf-8"))
        hasher.update(repr(value.co_kwonlyargcount).encode("utf-8"))
        hasher.update(repr(value.co_posonlyargcount).encode("utf-8"))
        for const in value.co_consts:
            _update_hash(hasher, const, seen=seen)
        return

    if isinstance(value, (np.ndarray, jax.Array)):
        hasher.update(_hash_array(value).encode("utf-8"))
        return

    if inspect.isfunction(value):
        hasher.update(_hash_callable(value, seen=seen).encode("utf-8"))
        return

    object_id = id(value)
    if object_id in seen:
        hasher.update(f"recursion:{object_id}".encode("utf-8"))
        return

    seen.add(object_id)

    if isinstance(value, (tuple, list)):
        hasher.update(repr(len(value)).encode("utf-8"))
        for item in value:
            _update_hash(hasher, item, seen=seen)
        return

    if isinstance(value, dict):
        for key in sorted(value.keys(), key=repr):
            _update_hash(hasher, key, seen=seen)
            _update_hash(hasher, value[key], seen=seen)
        return

    if isinstance(value, (set, frozenset)):
        for item in sorted(value, key=repr):
            _update_hash(hasher, item, seen=seen)
        return

    hasher.update(repr(value).encode("utf-8"))


def _hash_callable(fn, *, seen=None):
    """Hash a Python callable by source, code object, defaults, and closure."""
    if seen is None:
        seen = set()

    hasher = hashlib.sha256()
    hasher.update(fn.__module__.encode("utf-8"))
    hasher.update(fn.__qualname__.encode("utf-8"))

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = None
    hasher.update(repr(source).encode("utf-8"))

    _update_hash(hasher, fn.__code__, seen=seen)
    _update_hash(hasher, fn.__defaults__, seen=seen)
    _update_hash(hasher, fn.__kwdefaults__, seen=seen)

    closure = []
    if fn.__closure__ is not None:
        for cell in fn.__closure__:
            try:
                closure.append(cell.cell_contents)
            except ValueError:
                closure.append("<empty-cell>")
    _update_hash(hasher, tuple(closure), seen=seen)
    return hasher.hexdigest()


def _make_cache_key_data(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    rtol,
    atol,
    first_step,
    max_steps,
):
    """Create canonical metadata for the cache key."""
    y0_arr = np.asarray(y0, dtype=np.float64)
    times_arr = np.asarray(t_span, dtype=np.float64)
    params_arr = np.asarray(params)
    return {
        "solver_id": "diffrax_kvaerno5",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "diffrax_version": diffrax.__version__,
        "ode_fn_hash": _hash_callable(ode_fn),
        "y0": {
            "dtype": str(y0_arr.dtype),
            "shape": list(y0_arr.shape),
            "sha256": _hash_array(y0_arr),
        },
        "t_span": {
            "dtype": str(times_arr.dtype),
            "shape": list(times_arr.shape),
            "sha256": _hash_array(times_arr),
        },
        "params": {
            "dtype": str(params_arr.dtype),
            "shape": list(params_arr.shape),
            "sha256": _hash_array(params_arr),
        },
        "solve_kwargs": {
            "rtol": float(rtol),
            "atol": float(atol),
            "first_step": None if first_step is None else float(first_step),
            "max_steps": int(max_steps),
        },
    }


def _cache_key_from_data(key_data):
    """Hash canonical metadata into a cache key string."""
    payload = json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path, payload):
    """Write JSON data atomically."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _write_npz_atomic(path, ys):
    """Write compressed NumPy data atomically."""
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, suffix=".tmp"
    ) as tmp:
        np.savez_compressed(tmp, ys=ys)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def _solve_with_diffrax(
    ode_fn,
    y0_arr,
    save_times,
    params_arr,
    *,
    rtol,
    atol,
    first_step,
    max_steps,
    backend=None,
):
    """Solve an ensemble using Diffrax Kvaerno5."""
    t0 = float(save_times[0])
    tf = float(save_times[-1])
    dt0 = first_step if first_step is not None else (tf - t0) * 1e-6

    def _solve_one(p):
        term = diffrax.ODETerm(lambda t, y, args: ode_fn(y, t, p))
        solver = diffrax.Kvaerno5()
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=tf,
            dt0=dt0,
            y0=y0_arr,
            stepsize_controller=controller,
            max_steps=max_steps,
            saveat=diffrax.SaveAt(ts=save_times),
        )
        return sol.ys

    return jax.jit(jax.vmap(_solve_one), backend=backend)(params_arr)


def make_solver(ode_fn):
    """Create a reusable Kvaerno5 ensemble solver.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``.
    """

    def _solve(
        y0,
        t_span,
        params,
        *,
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
    ):
        """Solve an ensemble of ODEs.

        Parameters
        ----------
        y0 : array, shape [n_vars]
            Shared initial state.
        t_span : array-like, shape [n_save]
            Strictly increasing array of save times (len >= 2).
        params : array, shape [N, ...]
            Per-trajectory parameters.

        Returns
        -------
        array, shape [N, n_save, n_vars]
        """
        y0_arr = jnp.asarray(y0, dtype=jnp.float64)
        params_arr = jnp.asarray(params)
        save_times = jnp.asarray(
            np.asarray(t_span, dtype=np.float64), dtype=jnp.float64
        )
        return _solve_with_diffrax(
            ode_fn,
            y0_arr,
            save_times,
            params_arr,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
        )

    return _solve


def make_cached_solver(ode_fn, cache_dir=None):
    """Create a cached Kvaerno5 ensemble solver with the same solve signature."""
    cache_root = Path(cache_dir) if cache_dir is not None else _CACHE_ROOT

    def _solve(
        y0,
        t_span,
        params,
        *,
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
    ):
        y0_arr = np.asarray(y0, dtype=np.float64)
        params_arr = np.asarray(params)
        save_times_arr = np.asarray(t_span, dtype=np.float64)
        key_data = _make_cache_key_data(
            ode_fn,
            y0_arr,
            save_times_arr,
            params_arr,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
        )
        cache_key = _cache_key_from_data(key_data)
        cache_root.mkdir(parents=True, exist_ok=True)
        npz_path = cache_root / f"{cache_key}.npz"
        json_path = cache_root / f"{cache_key}.json"

        if npz_path.exists():
            with np.load(npz_path) as data:
                ys_np = data["ys"]
        else:
            ys = _solve_with_diffrax(
                ode_fn,
                jnp.asarray(y0_arr, dtype=jnp.float64),
                jnp.asarray(save_times_arr, dtype=jnp.float64),
                jnp.asarray(params_arr),
                rtol=rtol,
                atol=atol,
                first_step=first_step,
                max_steps=max_steps,
            )
            ys_np = np.asarray(ys)

            _write_npz_atomic(npz_path, ys_np)
            _write_json_atomic(
                json_path,
                {
                    "cache_key": cache_key,
                    **key_data,
                    "output": {
                        "dtype": str(ys_np.dtype),
                        "shape": list(ys_np.shape),
                    },
                },
            )

        return jnp.asarray(ys_np, dtype=jnp.float64)

    return _solve
