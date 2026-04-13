"""Unit tests for the Diffrax Kvaerno5 cached reference wrapper."""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np

from tests.reference_solvers.python import diffrax_kvaerno5


def _make_ode(scale):
    """Create a simple linear ODE with a closure-dependent scale."""

    def ode_fn(y, t, p):
        del t
        return scale * p[0] * y

    return ode_fn


def _make_fake_solver(call_counter):
    """Create a fake Diffrax backend for cache-key tests."""

    def _fake_solve_with_diffrax(
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
        del ode_fn, rtol, atol, first_step, max_steps, backend
        call_counter.append(1)
        params_np = np.asarray(params_arr, dtype=np.float64)
        times_np = np.asarray(save_times, dtype=np.float64)
        y0_np = np.asarray(y0_arr, dtype=np.float64)
        output = (
            params_np[:, 0][:, None, None]
            + times_np[None, :, None]
            + y0_np[None, None, :]
        )
        return jnp.asarray(output, dtype=jnp.float64)

    return _fake_solve_with_diffrax


def test_make_cached_solver_hits_cache(monkeypatch, tmp_path):
    """Repeated identical solves should reuse the on-disk cache entry."""
    calls = []
    monkeypatch.setattr(
        diffrax_kvaerno5, "_solve_with_diffrax", _make_fake_solver(calls)
    )

    solver = diffrax_kvaerno5.make_cached_solver(
        _make_ode(1.0), cache_dir=tmp_path / "cache"
    )
    kwargs = {
        "y0": jnp.array([1.0, 2.0], dtype=jnp.float64),
        "t_span": jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64),
        "params": jnp.array([[1.0], [2.0]], dtype=jnp.float64),
        "first_step": 1e-4,
        "rtol": 1e-8,
        "atol": 1e-10,
    }

    y1 = np.asarray(solver(**kwargs))
    y2 = np.asarray(solver(**kwargs))

    assert len(calls) == 1
    np.testing.assert_allclose(y1, y2)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_make_cached_solver_key_changes_with_inputs(monkeypatch, tmp_path):
    """Changing solver inputs or kwargs should invalidate the cache key."""
    calls = []
    monkeypatch.setattr(
        diffrax_kvaerno5, "_solve_with_diffrax", _make_fake_solver(calls)
    )

    solver = diffrax_kvaerno5.make_cached_solver(
        _make_ode(1.0), cache_dir=tmp_path / "cache"
    )
    base_kwargs = {
        "y0": jnp.array([1.0, 2.0], dtype=jnp.float64),
        "t_span": jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64),
        "params": jnp.array([[1.0], [2.0]], dtype=jnp.float64),
        "first_step": 1e-4,
        "rtol": 1e-8,
        "atol": 1e-10,
    }

    solver(**base_kwargs)
    solver(**{**base_kwargs, "params": jnp.array([[1.0], [3.0]], dtype=jnp.float64)})
    solver(**{**base_kwargs, "y0": jnp.array([1.0, 3.0], dtype=jnp.float64)})
    solver(**{**base_kwargs, "t_span": jnp.array([0.0, 1.0], dtype=jnp.float64)})
    solver(**{**base_kwargs, "first_step": 1e-3})

    assert len(calls) == 5
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 5


def test_make_cached_solver_key_changes_with_ode_closure(monkeypatch, tmp_path):
    """Changing the ODE function closure should invalidate the cache key."""
    calls = []
    monkeypatch.setattr(
        diffrax_kvaerno5, "_solve_with_diffrax", _make_fake_solver(calls)
    )

    solver_a = diffrax_kvaerno5.make_cached_solver(
        _make_ode(1.0), cache_dir=tmp_path / "cache"
    )
    solver_b = diffrax_kvaerno5.make_cached_solver(
        _make_ode(2.0), cache_dir=tmp_path / "cache"
    )
    kwargs = {
        "y0": jnp.array([1.0, 2.0], dtype=jnp.float64),
        "t_span": jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64),
        "params": jnp.array([[1.0], [2.0]], dtype=jnp.float64),
        "first_step": 1e-4,
        "rtol": 1e-8,
        "atol": 1e-10,
    }

    solver_a(**kwargs)
    solver_b(**kwargs)

    assert len(calls) == 2
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2
