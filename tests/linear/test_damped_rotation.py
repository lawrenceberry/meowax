"""Tests for Tsit5 linear and nonlinear solvers on damped rotation systems.

Damped rotation — nonstiff linear dynamics with an exact closed-form solution

Each equation pair (x_i, y_i) is an independent damped oscillator:

    d/dt [x_i] = s * [-lambda_i  -omega_i] [x_i]
         [y_i]       [ omega_i  -lambda_i] [y_i]

with global time-scale factor ``s = p[0]``. The eigenvalues are
``s * (-lambda_i ± i omega_i)``, so all modes decay smoothly while rotating.
This makes the problem well suited to an explicit RK method, and the exact
solution is available in closed form for direct validation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from solvers.linear.kencarp5_linear import make_solver as make_kencarp5_linear
from solvers.linear.tsit5_linear import make_solver as make_tsit5_linear
from solvers.nonlinear.kencarp5_nonlinear import make_solver as make_kencarp5_nonlinear
from solvers.nonlinear.tsit5_nonlinear import make_solver as make_tsit5_nonlinear
from tests.reference_solvers.python.diffrax_kencarp5 import (
    make_solver as make_diffrax_kencarp5_solver,
)
from tests.reference_solvers.python.diffrax_tsit5 import (
    make_solver as make_diffrax_tsit5_solver,
)
from tests.reference_solvers.python.julia_common import (
    JULIA_ENSEMBLE_BACKENDS,
    benchmark_julia_solver,
    julia_backend_id,
    maybe_mark_large_ensemble_sizes,
)
from tests.reference_solvers.python.julia_kencarp5 import (
    make_solver as make_julia_kencarp5_solver,
)
from tests.reference_solvers.python.julia_kvaerno5 import (
    make_solver as make_julia_kvaerno5_solver,
)
from tests.reference_solvers.python.julia_rodas5 import (
    make_solver as make_julia_rodas5_solver,
)
from tests.reference_solvers.python.julia_tsit5 import (
    make_solver as make_julia_tsit5_solver,
)

_TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
_N_PAIRS = [15, 25, 35]  # equation pairs → 30D, 50D, 70D
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]


def _make_damped_rotation_system(n_pairs):
    """Construct n_pairs independent damped rotation blocks."""
    n_vars = 2 * n_pairs
    lambdas = jnp.array(
        [0.2 + 1.8 * i / max(n_pairs - 1, 1) for i in range(n_pairs)],
        dtype=jnp.float64,
    )
    omegas = jnp.array(
        [0.5 + 1.0 * i / max(n_pairs - 1, 1) for i in range(n_pairs)],
        dtype=jnp.float64,
    )

    y0 = jnp.array([1.0, 0.0] * n_pairs, dtype=jnp.float64)

    blocks = [
        np.array([[-lam, -omega], [omega, -lam]], dtype=np.float64)
        for lam, omega in zip(np.asarray(lambdas), np.asarray(omegas))
    ]
    matrix_np = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i, block in enumerate(blocks):
        matrix_np[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    matrix = jnp.array(matrix_np, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        return p[0] * (matrix @ y)

    def jac_fn(t, p):
        del t
        return p[0] * matrix

    def explicit_jac_fn(t, p):
        return jac_fn(t, p)

    def implicit_jac_fn(t, p):
        del t, p
        return jnp.zeros_like(matrix)

    def explicit_ode_fn(y, t, p):
        return ode_fn(y, t, p)

    def implicit_ode_fn(y, t, p):
        del y, t, p
        return jnp.zeros_like(y0)

    return {
        "n_pairs": n_pairs,
        "n_vars": n_vars,
        "lambdas": np.asarray(lambdas, dtype=np.float64),
        "omegas": np.asarray(omegas, dtype=np.float64),
        "ode_fn": ode_fn,
        "jac_fn": jac_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "explicit_jac_fn": explicit_jac_fn,
        "implicit_jac_fn": implicit_jac_fn,
        "y0": y0,
    }


def _exact_solution(system, t_span, params):
    """Exact solution for repeated damped rotation blocks."""
    t = np.asarray(t_span, dtype=np.float64)
    s = np.asarray(params, dtype=np.float64)[:, 0]
    lambdas = system["lambdas"][None, None, :]
    omegas = system["omegas"][None, None, :]
    phase = s[:, None, None] * t[None, :, None]
    decay = np.exp(-lambdas * phase)
    cos_term = np.cos(omegas * phase)
    sin_term = np.sin(omegas * phase)
    pair = np.stack([decay * cos_term, decay * sin_term], axis=-1)
    return pair.reshape(len(s), len(t), 2 * system["n_pairs"])


@pytest.fixture
def damped_rotation_system(request):
    """Damped rotation system parameterized by number of blocks."""
    return _make_damped_rotation_system(request.param)


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _run_julia_damped_rotation(
    benchmark, solver_factory, damped_rotation_system, ensemble_size, ensemble_backend
):
    system = damped_rotation_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = solver_factory(
        "damped_rotation",
        system_config={"n_pairs": system["n_pairs"]},
        ensemble_backend=ensemble_backend,
    )
    results_np = benchmark_julia_solver(
        benchmark,
        solve,
        y0=system["y0"],
        t_span=_TIMES,
        params=params,
        first_step=1e-4,
        rtol=1e-6,
        atol=1e-8,
    )
    return system, results_np, params


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("mv_precision", ["fp32", "fp64"])
def test_tsit5_linear(benchmark, damped_rotation_system, ensemble_size, mv_precision):
    """Tsit5 linear benchmark with exact-solution validation."""
    system = damped_rotation_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_tsit5_linear(
        jac_fn=system["jac_fn"],
        mv_precision=mv_precision,
    )
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system, _TIMES, params)
    if mv_precision == "fp32":
        rtol = 1e-3
        atol = 6e-4
    else:
        rtol = 2e-4
        atol = 1e-6

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5_linear(benchmark, damped_rotation_system, ensemble_size, lu_precision):
    """KenCarp5 linear benchmark with exact-solution validation."""
    system = damped_rotation_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_kencarp5_linear(
        explicit_jac_fn=system["explicit_jac_fn"],
        implicit_jac_fn=system["implicit_jac_fn"],
        lu_precision=lu_precision,
    )
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system, _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_tsit5_nonlinear_on_linear_system(benchmark, damped_rotation_system, ensemble_size):
    """Tsit5 nonlinear benchmark on the same linear system."""
    system = damped_rotation_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_tsit5_nonlinear(ode_fn=system["ode_fn"])
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system, _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5_nonlinear_on_linear_system(
    benchmark, damped_rotation_system, ensemble_size, lu_precision
):
    """KenCarp5 nonlinear benchmark on the same linear system."""
    system = damped_rotation_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_kencarp5_nonlinear(
        explicit_ode_fn=system["explicit_ode_fn"],
        implicit_ode_fn=system["implicit_ode_fn"],
        lu_precision=lu_precision,
    )
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system, _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, damped_rotation_system, ensemble_size):
    """Diffrax KenCarp5 benchmark on the same damped rotation systems."""
    system = damped_rotation_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_diffrax_kencarp5_solver(
        system["explicit_ode_fn"], system["implicit_ode_fn"]
    )
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system, _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_tsit5(benchmark, damped_rotation_system, ensemble_size):
    """Diffrax Tsit5 benchmark on the same damped rotation systems."""
    system = damped_rotation_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_diffrax_tsit5_solver(system["ode_fn"])
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system, _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_tsit5(benchmark, damped_rotation_system, ensemble_size, ensemble_backend):
    """Julia Tsit5 benchmark on the same damped rotation systems."""
    system, results_np, params = _run_julia_damped_rotation(
        benchmark,
        make_julia_tsit5_solver,
        damped_rotation_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system, _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(
    benchmark, damped_rotation_system, ensemble_size, ensemble_backend
):
    """Julia KenCarp5 benchmark on the same damped rotation systems."""
    system, results_np, params = _run_julia_damped_rotation(
        benchmark,
        make_julia_kencarp5_solver,
        damped_rotation_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system, _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, damped_rotation_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark on the same damped rotation systems."""
    system, results_np, params = _run_julia_damped_rotation(
        benchmark,
        make_julia_rodas5_solver,
        damped_rotation_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system, _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "damped_rotation_system",
    _N_PAIRS,
    indirect=True,
    ids=lambda n_pairs: f"{2 * n_pairs}d",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(
    benchmark, damped_rotation_system, ensemble_size, ensemble_backend
):
    """Julia Kvaerno5 benchmark on the same damped rotation systems."""
    system, results_np, params = _run_julia_damped_rotation(
        benchmark,
        make_julia_kvaerno5_solver,
        damped_rotation_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system, _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np, y_exact, rtol=2e-4, atol=1e-6)
