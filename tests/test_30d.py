"""Tests for a synthetic stiff 30D mass-conserving ODE system."""

import time

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.rodas5 import solve_ensemble as rodas5_solve_ensemble
from solvers.rodas5_custom_kernel_v2 import make_solver as make_rodas5_v2_solver
from solvers.rodas5_custom_kernel_v3 import make_solver as make_rodas5_v3_solver
from solvers.rodas5_custom_kernel_v4 import make_solver as make_rodas5_v4_solver

_N_VARS = 30
_T_SPAN = (0.0, 1.0)

# Forward/backward nearest-neighbor reaction rates with a large stiffness spread.
_EDGE_COUNT = _N_VARS - 1
_KF = tuple(10.0 ** (-2.0 + 8.0 * i / (_EDGE_COUNT - 1)) for i in range(_EDGE_COUNT))
_KB = tuple(10.0 ** (6.0 - 8.0 * i / (_EDGE_COUNT - 1)) for i in range(_EDGE_COUNT))


def _build_linear_chain_matrix():
    """Build matrix M for the 30D linear chain dy/dt = M y (scale s=1)."""
    M = np.zeros((_N_VARS, _N_VARS), dtype=np.float64)
    M[0, 0] = -_KF[0]
    M[0, 1] = _KB[0]

    for i in range(1, _N_VARS - 1):
        M[i, i - 1] = _KF[i - 1]
        M[i, i] = -(_KB[i - 1] + _KF[i])
        M[i, i + 1] = _KB[i]

    M[_N_VARS - 1, _N_VARS - 2] = _KF[_N_VARS - 2]
    M[_N_VARS - 1, _N_VARS - 1] = -_KB[_N_VARS - 2]
    return jnp.array(M)


_M_30D = _build_linear_chain_matrix()


def _stiff_30d_ode(y, p):
    """30D stiff linear chain with conserved total mass.

    p[0] scales all rates per trajectory for ensemble variability.
    Works with tuple or array inputs by relying on index-based elementwise ops.
    """
    s = p[0]

    dy = [None] * _N_VARS
    dy[0] = -s * _KF[0] * y[0] + s * _KB[0] * y[1]

    for i in range(1, _N_VARS - 1):
        dy[i] = (
            s * _KF[i - 1] * y[i - 1]
            - s * (_KB[i - 1] + _KF[i]) * y[i]
            + s * _KB[i] * y[i + 1]
        )

    dy[_N_VARS - 1] = (
        s * _KF[_N_VARS - 2] * y[_N_VARS - 2] - s * _KB[_N_VARS - 2] * y[_N_VARS - 1]
    )
    return tuple(dy)


@jax.jit
def _stiff_30d_array(y, p):
    """Array-returning adapter for non-Pallas ensemble reference solver."""
    return jnp.array(_stiff_30d_ode(y, p))


def test_rodas5_v2_30d_matches_rodas5_reference():
    """Validate Pallas v2 solver on a new stiff 30D system."""
    N = 256
    rng = np.random.default_rng(0)

    # Start with all mass in the first species; conservation implies sum(y)=1 always.
    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    y0_batch = jnp.broadcast_to(y0, (N, _N_VARS))

    # Small per-trajectory perturbation around scale 1.0.
    params_batch = jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((N, 1)) - 1.0),
        dtype=jnp.float64,
    )

    solve_v2 = make_rodas5_v2_solver(_stiff_30d_ode)

    y_v2 = solve_v2(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        _stiff_30d_array,
        y0=y0,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v2.shape == (N, _N_VARS)
    assert y_ref.shape == (N, _N_VARS)

    # Conservative linear chain: mass should remain near 1 for each trajectory.
    np.testing.assert_allclose(y_v2.sum(axis=1), 1.0, atol=3e-6)

    # Compare against the existing Rodas5 implementation as reference.
    np.testing.assert_allclose(y_v2, y_ref, rtol=3e-5, atol=3e-8)


def test_rodas5_v3_30d_matches_rodas5_reference_for_linear_matrix():
    """Validate matrix-specialized Pallas v3 solver on the 30D stiff chain."""
    N = 256

    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    y0_batch = jnp.broadcast_to(y0, (N, _N_VARS))

    solve_v3 = make_rodas5_v3_solver(_M_30D)
    y_v3 = solve_v3(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    params_batch = jnp.ones((N, 1), dtype=jnp.float64)
    y_ref = rodas5_solve_ensemble(
        _stiff_30d_array,
        y0=y0,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v3.shape == (N, _N_VARS)
    np.testing.assert_allclose(y_v3.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v3, y_ref, rtol=3e-5, atol=3e-8)


@pytest.fixture
def params_batch_30d(request):
    """30D rate-scale parameter sets with +/-10% uniform perturbation."""
    N = request.param
    rng = np.random.default_rng(42)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((N, 1)) - 1.0),
        dtype=jnp.float64,
    )


@pytest.mark.parametrize(
    "params_batch_30d", [2, 100, 1000, 10000, 100_000], indirect=True
)
def test_rodas5_30d_ensemble_N(benchmark, params_batch_30d):
    """Rodas5 vmap ensemble benchmark on the stiff 30D system."""
    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    results = benchmark.pedantic(
        lambda: rodas5_solve_ensemble(
            _stiff_30d_array,
            y0=y0,
            t_span=_T_SPAN,
            params_batch=params_batch_30d,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (params_batch_30d.shape[0], _N_VARS)
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize(
    "params_batch_30d", [2, 100, 1000, 10000, 100_000], indirect=True
)
def test_rodas5_v2_30d_pallas_ensemble_N(benchmark, params_batch_30d):
    """Rodas5 v2 Pallas custom kernel ensemble benchmark on the stiff 30D system."""
    solve_v2 = make_rodas5_v2_solver(_stiff_30d_ode)
    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    y0_batch = jnp.broadcast_to(y0, (params_batch_30d.shape[0], _N_VARS))
    results = benchmark.pedantic(
        lambda: solve_v2(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            params_batch=params_batch_30d,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (params_batch_30d.shape[0], _N_VARS)
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.fixture
def y0_batch_30d(request):
    """30D initial condition batch (all mass in first species)."""
    N = request.param
    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    return jnp.broadcast_to(y0, (N, _N_VARS))


@pytest.mark.parametrize("y0_batch_30d", [2, 100, 1000, 10000, 100_000], indirect=True)
def test_rodas5_v3_30d_pallas_ensemble_N(benchmark, y0_batch_30d):
    """Rodas5 v3 (matrix) Pallas custom kernel ensemble benchmark on the stiff 30D system."""
    solve_v3 = make_rodas5_v3_solver(_M_30D)
    results = benchmark.pedantic(
        lambda: solve_v3(
            y0_batch=y0_batch_30d,
            t_span=_T_SPAN,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (y0_batch_30d.shape[0], _N_VARS)
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


def test_rodas5_v3_30d_pallas_compile_time(benchmark):
    """Measure approximate compile time for v3 (matrix) Pallas solver.

    Uses first-call latency minus steady-state latency as estimate.
    """
    N = 1234

    solve_v3 = make_rodas5_v3_solver(_M_30D)
    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    y0_batch = jnp.broadcast_to(y0, (N, _N_VARS))

    # Use uncommon static args to avoid reusing an already-compiled cache entry.
    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v3(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v3(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, _N_VARS)
        assert y_second.shape == (N, _N_VARS)
        np.testing.assert_allclose(y_first.sum(axis=1), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


def test_rodas5_v2_30d_pallas_compile_time(benchmark):
    """Measure approximate compile time as first-call latency minus steady-state latency.

    We use a unique static configuration so this test triggers a fresh compile,
    then subtract a second call with identical inputs to estimate compile cost.
    """
    N = 1234
    rng = np.random.default_rng(123)

    solve_v2 = make_rodas5_v2_solver(_stiff_30d_ode)
    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    y0_batch = jnp.broadcast_to(y0, (N, _N_VARS))
    params_batch = jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((N, 1)) - 1.0),
        dtype=jnp.float64,
    )

    # Use uncommon static args to avoid reusing an already-compiled cache entry.
    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v2(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v2(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, _N_VARS)
        assert y_second.shape == (N, _N_VARS)
        np.testing.assert_allclose(y_first.sum(axis=1), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


# ---------------------------------------------------------------------------
# v4 Numba CUDA tests
# ---------------------------------------------------------------------------


def test_rodas5_v4_30d_matches_rodas5_reference_for_linear_matrix():
    """Validate Numba CUDA v4 solver on the 30D stiff chain."""
    N = 256

    y0 = np.array([1.0] + [0.0] * (_N_VARS - 1), dtype=np.float64)
    y0_batch = np.broadcast_to(y0, (N, _N_VARS)).copy()

    solve_v4 = make_rodas5_v4_solver(np.asarray(_M_30D))
    y_v4 = solve_v4(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )

    params_batch = jnp.ones((N, 1), dtype=jnp.float64)
    y_ref = rodas5_solve_ensemble(
        _stiff_30d_array,
        y0=jnp.array(y0),
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v4.shape == (N, _N_VARS)
    np.testing.assert_allclose(y_v4.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v4, np.asarray(y_ref), rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("y0_batch_30d", [2, 100, 1000, 10000, 100_000], indirect=True)
def test_rodas5_v4_30d_numba_ensemble_N(benchmark, y0_batch_30d):
    """Rodas5 v4 Numba CUDA ensemble benchmark on the stiff 30D system."""
    solve_v4 = make_rodas5_v4_solver(np.asarray(_M_30D))
    y0_np = np.asarray(y0_batch_30d, dtype=np.float64).copy()
    results = benchmark.pedantic(
        lambda: solve_v4(
            y0_batch=y0_np,
            t_span=_T_SPAN,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (y0_np.shape[0], _N_VARS)
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


def test_rodas5_v4_30d_numba_compile_time(benchmark):
    """Measure approximate compile time for v4 Numba CUDA solver.

    Uses first-call latency minus steady-state latency as estimate.
    """
    N = 1234

    solve_v4 = make_rodas5_v4_solver(np.asarray(_M_30D))
    y0 = np.array([1.0] + [0.0] * (_N_VARS - 1), dtype=np.float64)
    y0_batch = np.broadcast_to(y0, (N, _N_VARS)).copy()

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v4(**kwargs)
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v4(**kwargs)
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, _N_VARS)
        assert y_second.shape == (N, _N_VARS)
        np.testing.assert_allclose(y_first.sum(axis=1), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0
