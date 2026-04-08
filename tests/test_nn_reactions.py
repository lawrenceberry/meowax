"""Tests for synthetic stiff nearest-neighbor mass-conserving ODE systems."""

import time

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest
from scipy.linalg import expm

from solvers.rodas5 import solve_ensemble as rodas5_solve_ensemble
from solvers.rodas5_custom_kernel_v2 import make_solver as make_rodas5_v2_solver
from solvers.rodas5_custom_kernel_v3 import make_solver as make_rodas5_v3_solver
from solvers.rodas5_custom_kernel_v4 import make_solver as make_rodas5_v4_solver
from solvers.rodas5_custom_kernel_v5 import make_solver as make_rodas5_v5_solver
from solvers.rodas5_custom_kernel_v6 import make_solver as make_rodas5_v6_solver
from solvers.rodas5_custom_kernel_v7_tc import make_solver as make_rodas5_v7_tc_solver
from solvers.rodas5_v2_linear import make_solver as make_rodas5_v2_linear_solver
from solvers.rodas5_v2_nonlinear import make_solver as make_rodas5_v2_nonlinear_solver

_T_SPAN = (0.0, 1.0)
_V6_SAVE_TIMES = jnp.array(_T_SPAN, dtype=jnp.float64)
_V6_MULTI_SAVE_TIMES = jnp.array((0.0, 0.125, 0.25, 0.5, 1.0), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]
_LINEAR_SOLVER_PRECISIONS = ("fp64", "fp32")
_V6_LINEAR_SOLVERS = _LINEAR_SOLVER_PRECISIONS


def _time_linear_scale(t):
    """Scalar multiplier for the time-dependent linear systems."""
    return 1.0 + 0.2 * t


def _time_linear_scale_integral(t0, tf):
    """Integral of ``_time_linear_scale`` over ``[t0, tf]``."""
    return (tf - t0) + 0.1 * (tf**2 - t0**2)


def _build_linear_chain_matrix(n_vars, kf, kb):
    """Build matrix M for the D-dimensional linear chain dy/dt = M y."""
    M = np.zeros((n_vars, n_vars), dtype=np.float64)
    M[0, 0] = -kf[0]
    M[0, 1] = kb[0]

    for i in range(1, n_vars - 1):
        M[i, i - 1] = kf[i - 1]
        M[i, i] = -(kb[i - 1] + kf[i])
        M[i, i + 1] = kb[i]

    M[n_vars - 1, n_vars - 2] = kf[n_vars - 2]
    M[n_vars - 1, n_vars - 1] = -kb[n_vars - 2]
    return M


def _make_nn_reaction_system(n_vars):
    """Construct a configurable D-dimensional nearest-neighbor reaction system."""
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    edge_count = n_vars - 1
    kf = tuple(10.0 ** (-2.0 + 8.0 * i / (edge_count - 1)) for i in range(edge_count))
    kb = tuple(10.0 ** (6.0 - 8.0 * i / (edge_count - 1)) for i in range(edge_count))
    M_np = _build_linear_chain_matrix(n_vars, kf, kb)
    M = jnp.array(M_np, dtype=jnp.float64)
    y0 = jnp.array([1.0] + [0.0] * (n_vars - 1), dtype=jnp.float64)
    y0_np = np.asarray(y0, dtype=np.float64)

    def ode(y, p):
        """D-dimensional stiff linear chain with conserved total mass."""
        s = p[0]
        dy = [None] * n_vars
        dy[0] = -s * kf[0] * y[0] + s * kb[0] * y[1]

        for i in range(1, n_vars - 1):
            dy[i] = (
                s * kf[i - 1] * y[i - 1]
                - s * (kb[i - 1] + kf[i]) * y[i]
                + s * kb[i] * y[i + 1]
            )

        dy[n_vars - 1] = (
            s * kf[n_vars - 2] * y[n_vars - 2] - s * kb[n_vars - 2] * y[n_vars - 1]
        )
        return tuple(dy)

    @jax.jit
    def array(y, p):
        """Array-returning adapter for the non-Pallas reference solver."""
        return jnp.array(ode(y, p))

    def jac(y, p):
        """Explicit Jacobian for the D-dimensional stiff linear chain."""
        del y
        s = p[0]
        return tuple(
            tuple(s * M_np[i, j] for j in range(n_vars)) for i in range(n_vars)
        )

    def time_jac(t):
        """Time-dependent matrix callback for the v6 solver."""
        s = _time_linear_scale(t)
        return tuple(
            tuple(s * M_np[i, j] for j in range(n_vars)) for i in range(n_vars)
        )

    def jac_array(t, p):
        """Jacobian as a JAX array, function of time and params."""
        s = p[0]
        return s * M

    def time_exact(y_init, t_span):
        """Closed-form solution for M(t) = a(t) * M with commuting matrices."""
        scale_int = _time_linear_scale_integral(*t_span)
        return jnp.array(expm(scale_int * M_np) @ np.asarray(y_init), dtype=jnp.float64)

    return {
        "n_vars": n_vars,
        "matrix": M,
        "matrix_np": M_np,
        "ode": ode,
        "array": array,
        "jac": jac,
        "time_jac": time_jac,
        "jac_array": jac_array,
        "time_exact": time_exact,
        "y0": y0,
        "y0_np": y0_np,
    }


def _dim_id(n_vars):
    return f"{n_vars}d"


def _linear_solver_id(linear_solver):
    return linear_solver


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _broadcast_y0(y0, size):
    return jnp.broadcast_to(y0, (size, y0.shape[0]))


def _time_exact_trajectory(system, save_times):
    save_times_np = np.asarray(save_times, dtype=np.float64)
    t0 = float(save_times_np[0])
    return np.stack(
        [
            np.asarray(
                system["time_exact"](system["y0"], (t0, float(tf))), dtype=np.float64
            )
            for tf in save_times_np
        ],
        axis=0,
    )


def _reference_trajectory(system, params_batch, save_times):
    save_times_np = np.asarray(save_times, dtype=np.float64)
    traj = []
    for i, tf in enumerate(save_times_np):
        if i == 0:
            traj.append(np.broadcast_to(np.asarray(system["y0"]), (params_batch.shape[0], system["n_vars"])))
            continue
        y_ref = rodas5_solve_ensemble(
            system["array"],
            y0=system["y0"],
            t_span=(float(save_times_np[0]), float(tf)),
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready()
        traj.append(np.asarray(y_ref))
    return np.stack(traj, axis=1)


@pytest.fixture
def nn_reaction_system(request):
    """Configurable nearest-neighbor reaction system parameterized by dimension."""
    return _make_nn_reaction_system(request.param)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v2_matches_rodas5_reference(nn_reaction_system):
    """Validate Pallas v2 solver on the nearest-neighbor reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=0)

    solve_v2 = make_rodas5_v2_solver(system["ode"])
    y_v2 = solve_v2(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v2.shape == (N, system["n_vars"])
    assert y_ref.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v2.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v2, y_ref, rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v3_matches_rodas5_reference_for_linear_matrix(nn_reaction_system):
    """Validate matrix-specialized Pallas v3 solver on the reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)

    solve_v3 = make_rodas5_v3_solver(system["matrix"])
    y_v3 = solve_v3(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    params_batch = jnp.ones((N, 1), dtype=jnp.float64)
    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v3.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v3.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v3, y_ref, rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 vmap ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: rodas5_solve_ensemble(
            system["array"],
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v2_pallas_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v2 Pallas custom kernel ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    solve_v2 = make_rodas5_v2_solver(system["ode"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    params_batch = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: solve_v2(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v3_pallas_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v3 (matrix) Pallas custom kernel ensemble benchmark."""
    system = nn_reaction_system
    solve_v3 = make_rodas5_v3_solver(system["matrix"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    results = benchmark.pedantic(
        lambda: solve_v3(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v3_pallas_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v3 (matrix) Pallas solver."""
    N = 1234
    system = nn_reaction_system

    solve_v3 = make_rodas5_v3_solver(system["matrix"])
    y0_batch = _broadcast_y0(system["y0"], N)

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

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
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


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v2_pallas_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v2 (JVP) Pallas solver."""
    N = 1234
    system = nn_reaction_system

    solve_v2 = make_rodas5_v2_solver(system["ode"])
    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=123)

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

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
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


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v4_matches_rodas5_reference_for_linear_matrix(nn_reaction_system):
    """Validate Numba CUDA v4 solver on the reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = np.broadcast_to(system["y0_np"], (N, system["n_vars"])).copy()

    solve_v4 = make_rodas5_v4_solver(np.asarray(system["matrix"]))
    y_v4 = solve_v4(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )

    params_batch = jnp.ones((N, 1), dtype=jnp.float64)
    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v4.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v4.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v4, np.asarray(y_ref), rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v4_numba_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v4 Numba CUDA ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    solve_v4 = make_rodas5_v4_solver(np.asarray(system["matrix"]))
    y0_np = np.broadcast_to(system["y0_np"], (ensemble_size, system["n_vars"])).copy()
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

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v4_numba_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v4 Numba CUDA solver."""
    N = 1234
    system = nn_reaction_system

    solve_v4 = make_rodas5_v4_solver(np.asarray(system["matrix"]))
    y0_batch = np.broadcast_to(system["y0_np"], (N, system["n_vars"])).copy()

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

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
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


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v5_matches_rodas5_reference(nn_reaction_system):
    """Validate v5 solver (explicit Jacobian) on the reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=0)

    solve_v5 = make_rodas5_v5_solver(system["jac"])
    y_v5 = solve_v5(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v5.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v5.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v5, y_ref, rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v5_pallas_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v5 Pallas ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    solve_v5 = make_rodas5_v5_solver(system["jac"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    params_batch = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: solve_v5(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v5_pallas_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v5 (explicit Jacobian)."""
    N = 1234
    system = nn_reaction_system

    solve_v5 = make_rodas5_v5_solver(system["jac"])
    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=123)

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
        y_first = solve_v5(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v5(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
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


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_matches_closed_form_time_dependent_reference(
    nn_reaction_system, linear_solver
):
    """Validate v6 solver on a time-dependent nearest-neighbor reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)

    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)
    y_v6 = solve_v6(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_v6_np = np.asarray(y_v6)

    y_exact_traj = _time_exact_trajectory(system, _V6_SAVE_TIMES)
    y_exact_batch = np.broadcast_to(
        y_exact_traj, (N, y_exact_traj.shape[0], y_exact_traj.shape[1])
    )

    assert y_v6.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(y_v6_np[:, 0, :], np.asarray(y0_batch), atol=0.0)
    np.testing.assert_allclose(y_v6_np.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v6_np, y_exact_batch, rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_saves_multiple_times(nn_reaction_system, linear_solver):
    """Validate v6 solver on several requested save times."""
    N = 64
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)
    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)

    y_v6 = solve_v6(
        y0_batch=y0_batch,
        t_span=_V6_MULTI_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_v6_np = np.asarray(y_v6)

    y_exact_traj = _time_exact_trajectory(system, _V6_MULTI_SAVE_TIMES)
    y_exact_batch = np.broadcast_to(
        y_exact_traj, (N, y_exact_traj.shape[0], y_exact_traj.shape[1])
    )

    assert y_v6.shape == (N, len(_V6_MULTI_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(y_v6_np[:, 0, :], np.asarray(y0_batch), atol=0.0)
    np.testing.assert_allclose(y_v6_np.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v6_np, y_exact_batch, rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_pallas_ensemble_N(
    benchmark, nn_reaction_system, ensemble_size, linear_solver
):
    """Rodas5 v6 Pallas ensemble benchmark on the time-dependent system."""
    system = nn_reaction_system
    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    results = benchmark.pedantic(
        lambda: solve_v6(
            y0_batch=y0_batch,
            t_span=_V6_SAVE_TIMES,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(np.asarray(results).sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_pallas_compile_time(benchmark, nn_reaction_system, linear_solver):
    """Measure approximate compile time for v6 (time-dependent Jacobian)."""
    N = 1234
    system = nn_reaction_system

    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)
    y0_batch = _broadcast_y0(system["y0"], N)

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v6(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v6(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        assert y_second.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        np.testing.assert_allclose(np.asarray(y_first).sum(axis=2), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


@pytest.mark.parametrize(
    ("t_span", "match"),
    [
        (jnp.array([[0.0, 1.0]], dtype=jnp.float64), "1D array"),
        (jnp.array([0.0], dtype=jnp.float64), "at least 2"),
        (jnp.array([0.0, 0.5, 0.5], dtype=jnp.float64), "strictly increasing"),
    ],
)
def test_rodas5_v6_rejects_invalid_save_times(t_span, match):
    """Validate v6 save-time input checks."""
    system = _make_nn_reaction_system(30)
    solve_v6 = make_rodas5_v6_solver(system["time_jac"])
    y0_batch = _broadcast_y0(system["y0"], 4)

    with pytest.raises(ValueError, match=match):
        solve_v6(
            y0_batch=y0_batch,
            t_span=t_span,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v6_fp32_matches_fp64_baseline(nn_reaction_system):
    """Validate the FP32 LU v6 path against the FP64 baseline."""
    N = 256
    system = nn_reaction_system
    y0_batch = _broadcast_y0(system["y0"], N)

    solve_fp64 = make_rodas5_v6_solver(system["time_jac"], linear_solver="fp64")
    solve_fp32 = make_rodas5_v6_solver(system["time_jac"], linear_solver="fp32")

    y_fp64 = solve_fp64(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_fp32 = solve_fp32(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    np.testing.assert_allclose(
        np.asarray(y_fp32), np.asarray(y_fp64), rtol=2e-4, atol=3e-8
    )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v7_tc_matches_closed_form_time_dependent_reference(nn_reaction_system):
    """Validate the experimental blocked-LU Tensor Core-style solver on 50D."""
    N = 128
    system = nn_reaction_system
    y0_batch = _broadcast_y0(system["y0"], N)

    solve_v7 = make_rodas5_v7_tc_solver(system["time_jac"])
    y_v7 = solve_v7(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_v7_np = np.asarray(y_v7)

    y_exact_traj = _time_exact_trajectory(system, _V6_SAVE_TIMES)
    y_exact_batch = np.broadcast_to(
        y_exact_traj, (N, y_exact_traj.shape[0], y_exact_traj.shape[1])
    )

    assert y_v7.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(y_v7_np[:, 0, :], np.asarray(y0_batch), atol=0.0)
    np.testing.assert_allclose(y_v7_np.sum(axis=2), 1.0, atol=3e-5)
    np.testing.assert_allclose(y_v7_np, y_exact_batch, rtol=8e-4, atol=6e-8)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v7_tc_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Benchmark the experimental blocked-LU Tensor Core-style solver on 50D."""
    system = nn_reaction_system
    solve_v7 = make_rodas5_v7_tc_solver(system["time_jac"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    results = benchmark.pedantic(
        lambda: solve_v7(
            y0_batch=y0_batch,
            t_span=_V6_SAVE_TIMES,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(np.asarray(results).sum(axis=2), 1.0, atol=3e-5)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v7_tc_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for the experimental blocked-LU solver."""
    N = 1234
    system = nn_reaction_system

    solve_v7 = make_rodas5_v7_tc_solver(system["time_jac"])
    y0_batch = _broadcast_y0(system["y0"], N)

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v7(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v7(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        assert y_second.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        np.testing.assert_allclose(np.asarray(y_first).sum(axis=2), 1.0, atol=3e-5)
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
# Rodas5 v2 — single-loop batched ensemble
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("batch_size", [1, 7, None], ids=["bs1", "bs7", "bsN"])
def test_rodas5_v2_matches_reference(nn_reaction_system, batch_size):
    """Validate single-loop v2 solver against vmap reference."""
    N = 100
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    solve_v2 = make_rodas5_v2_nonlinear_solver(system["array"], batch_size=batch_size)
    y_v2 = solve_v2(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v2.shape == (N, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(y_v2[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0)
    np.testing.assert_allclose(y_v2.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v2[:, -1, :], y_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v2_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v2 single-loop ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)
    solve_v2 = make_rodas5_v2_nonlinear_solver(system["array"])
    results = benchmark.pedantic(
        lambda: solve_v2(
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


# ---------------------------------------------------------------------------
# Rodas5 v2 — jac_fn path (linear systems, no AD)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v2_jac_fn_matches_reference(nn_reaction_system):
    """Validate jac_fn (linear) path against the f (AD) path."""
    N = 256
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    solve_linear = make_rodas5_v2_linear_solver(system["jac_array"])
    y_jac = solve_linear(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_jac.shape == (N, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(y_jac[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0)
    np.testing.assert_allclose(y_jac.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_jac[:, -1, :], y_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v2_fp32_linear_solver_matches_fp64_baseline(nn_reaction_system):
    """Validate FP32 LU solves against the FP64 baseline on the jac_fn path."""
    N = 256
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    solve_kwargs = dict(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )

    solve_fp64 = make_rodas5_v2_linear_solver(system["jac_array"], linear_solver_precision="fp64")
    y_fp64 = solve_fp64(**solve_kwargs).block_until_ready()
    solve_fp32 = make_rodas5_v2_linear_solver(system["jac_array"], linear_solver_precision="fp32")
    y_fp32 = solve_fp32(**solve_kwargs).block_until_ready()

    np.testing.assert_allclose(
        np.asarray(y_fp32), np.asarray(y_fp64), rtol=2e-4, atol=3e-8
    )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v2_jac_fn_saves_multiple_times(nn_reaction_system):
    """Validate multiple save times on the jac_fn path."""
    N = 64
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    solve_linear = make_rodas5_v2_linear_solver(system["jac_array"], linear_solver_precision="fp32")
    y_hist = solve_linear(
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_ref = _reference_trajectory(system, params_batch, _V6_MULTI_SAVE_TIMES)

    assert y_hist.shape == (N, len(_V6_MULTI_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(y_hist[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0)
    np.testing.assert_allclose(y_hist.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(y_hist), y_ref, rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v2_jac_fn_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v2 jac_fn (linear, no AD) ensemble benchmark."""
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)
    solve_linear = make_rodas5_v2_linear_solver(
        system["jac_array"],
        linear_solver_precision="fp32",
    )
    results = benchmark.pedantic(
        lambda: solve_linear(
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "linear_solver_precision", _LINEAR_SOLVER_PRECISIONS, ids=_linear_solver_id
)
def test_rodas5_v2_jac_fn_linear_solver_precision_timing(
    benchmark, nn_reaction_system, linear_solver_precision
):
    """Benchmark the jac_fn path with FP64 vs FP32 LU precision."""
    ensemble_size = 10_000
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)

    solve_linear = make_rodas5_v2_linear_solver(
        system["jac_array"],
        linear_solver_precision=linear_solver_precision,
    )
    results = benchmark.pedantic(
        lambda: solve_linear(
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    benchmark.extra_info["backend"] = jax.default_backend()

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)
