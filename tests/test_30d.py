"""Tests for a synthetic stiff 30D mass-conserving ODE system."""

import time

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest
from jax.experimental import pallas as pl

from solvers.rodas5 import solve_ensemble as rodas5_solve_ensemble
from solvers.rodas5_custom_kernel_v2 import make_solver as make_rodas5_v2_solver

_N_VARS = 30
_T_SPAN = (0.0, 1.0)

# Forward/backward nearest-neighbor reaction rates with a large stiffness spread.
_EDGE_COUNT = _N_VARS - 1
_KF = tuple(10.0 ** (-2.0 + 8.0 * i / (_EDGE_COUNT - 1)) for i in range(_EDGE_COUNT))
_KB = tuple(10.0 ** (6.0 - 8.0 * i / (_EDGE_COUNT - 1)) for i in range(_EDGE_COUNT))

# Constants array for Ref-based ODE: [KF_0..KF_28, KB_0..KB_28]
_KF_KB = jnp.array(list(_KF) + list(_KB), dtype=jnp.float64)


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


def _stiff_30d_ode_ref(y_ref, p_ref, dy_ref, c_ref):
    """Ref-based 30D chain ODE using fori_loop for compact Triton IR.

    Constants layout in c_ref: [KF_0..KF_28, KB_0..KB_28] (58 values).
    """
    nv = _N_VARS
    edge = _EDGE_COUNT  # 29

    s = p_ref.at[:, 0][...]

    # Boundary: dy[0] = -s*KF[0]*y[0] + s*KB[0]*y[1]
    y0 = y_ref.at[:, 0][...]
    y1 = y_ref.at[:, 1][...]
    kf0 = c_ref.at[:, 0][...]
    kb0 = c_ref.at[:, edge][...]
    dy_ref.at[:, 0][...] = -s * kf0 * y0 + s * kb0 * y1

    # Interior via fori_loop (i = 1..28)
    def interior(i, carry):
        ym = y_ref.at[:, pl.ds(i - 1, 1)][...][:, 0]
        yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
        yp = y_ref.at[:, pl.ds(i + 1, 1)][...][:, 0]
        kf_im1 = c_ref.at[:, pl.ds(i - 1, 1)][...][:, 0]
        kf_i = c_ref.at[:, pl.ds(i, 1)][...][:, 0]
        kb_im1 = c_ref.at[:, pl.ds(edge + i - 1, 1)][...][:, 0]
        kb_i = c_ref.at[:, pl.ds(edge + i, 1)][...][:, 0]
        val = s * kf_im1 * ym - s * (kb_im1 + kf_i) * yi + s * kb_i * yp
        dy_ref.at[:, pl.ds(i, 1)][...] = val[:, None]
        return carry

    jax.lax.fori_loop(1, nv - 1, interior, jnp.int32(0))

    # Boundary: dy[29] = s*KF[28]*y[28] - s*KB[28]*y[29]
    y28 = y_ref.at[:, nv - 2][...]
    y29 = y_ref.at[:, nv - 1][...]
    kf28 = c_ref.at[:, edge - 1][...]
    kb28 = c_ref.at[:, 2 * edge - 1][...]
    dy_ref.at[:, nv - 1][...] = s * kf28 * y28 - s * kb28 * y29


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


def test_rodas5_v2_30d_ref_matches_rodas5_reference():
    """Validate Ref-based ODE (ode_ref_fn + constants) against reference."""
    N = 256
    rng = np.random.default_rng(0)

    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    y0_batch = jnp.broadcast_to(y0, (N, _N_VARS))
    params_batch = jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((N, 1)) - 1.0),
        dtype=jnp.float64,
    )

    solve_ref_ode = make_rodas5_v2_solver(
        _stiff_30d_ode,
        ode_ref_fn=_stiff_30d_ode_ref,
        constants=_KF_KB,
    )

    y_ref_ode = solve_ref_ode(
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

    assert y_ref_ode.shape == (N, _N_VARS)
    np.testing.assert_allclose(y_ref_ode.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_ref_ode, y_ref, rtol=3e-5, atol=3e-8)


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


def test_rodas5_v2_30d_ref_compile_time(benchmark):
    """Compile time for Ref-based ODE variant (ode_ref_fn + constants).

    Should be faster than tuple-only because the 9 stage ODE evaluations
    use a single fori_loop instead of unrolling 28 Python iterations each.
    """
    N = 1234
    rng = np.random.default_rng(456)

    solve_ref = make_rodas5_v2_solver(
        _stiff_30d_ode,
        ode_ref_fn=_stiff_30d_ode_ref,
        constants=_KF_KB,
    )
    y0 = jnp.array([1.0] + [0.0] * (_N_VARS - 1), dtype=jnp.float64)
    y0_batch = jnp.broadcast_to(y0, (N, _N_VARS))
    params_batch = jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((N, 1)) - 1.0),
        dtype=jnp.float64,
    )

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=7.1e-7,
        rtol=1.24e-6,
        atol=4.57e-8,
        max_steps=654322,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_ref(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_ref(**kwargs).block_until_ready()
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
