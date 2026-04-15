"""Tests for the Rodas5 linear and nonlinear solvers on semi-discretized 1D heat equation systems.

Heat equation — stiffness from grid refinement

M is tridiagonal with uniform entries: −2/Δx² on the diagonal and 1/Δx² on the off-diagonals.
Eigenvalues range from about −π² (slowest mode) to −4/Δx² ≈ −4(N+1)² (fastest mode), giving
a stiffness ratio of O(N²). With N=70 that's only ~2000 — far less stiff than nn_reactions —
but it grows without bound as you refine the grid.

PDE:  ∂u/∂t = α ∂²u/∂x²,  x ∈ (0, 1),  u(0, t) = u(1, t) = 0
Grid: N interior points,  Δx = 1/(N+1),  x_i = (i+1)·Δx

The N×N system matrix M = (1/Δx²) tridiag(1, −2, 1) is symmetric tridiagonal.
Eigenvalues span ≈ [−4/Δx², −π²], so the stiffness ratio scales as O(N²) — a
demanding scalability test for implicit solvers.

Initial condition y0_i = sin(π x_i) is the first discrete eigenmode of M, so the
exact solution is y_i(t) = y0_i · exp(λ₁ α t) where λ₁ = (2/Δx²)(cos(πΔx) − 1).
This lets the reference tests validate against the analytic answer rather than
a second numerical solver.

fp32 cancellation hazard — mv_precision must stay fp64
------------------------------------------------------
Each row of M computes a second difference: (M@u)_i = inv_dx2*(u_{i-1} − 2u_i + u_{i+1}).
For a smooth solution like sin(πx), the three terms nearly cancel: individual terms are
O((N+1)²·u_i) ≈ O(1000) but the result is O(π²·u_i) ≈ O(10), a ~100× cancellation.
In fp32 (ε ≈ 1.2×10⁻⁷), each term carries absolute error ~1000 × 1.2×10⁻⁷ ≈ 1.2×10⁻⁴,
which survives the cancellation and gives a relative error in f_eval of ~1.2×10⁻⁴/10 = 1.2×10⁻⁵.
That is ten times larger than rtol=1e-6, so the Rosenbrock error estimator sees inflated
residuals in k8, rejects most steps, and the solver crawls (tested: fp32 MV on 30D/N=2
took ~3 s vs ~4 ms for the nonlinear solver whose f_eval stays in fp64).  The fix is to
keep mv_precision="fp64" and vary only lu_precision, which is why the benchmark tests are
parameterised on lu_precision alone.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from solvers.nonlinear.kencarp5_nonlinear import make_solver as make_kencarp5_nonlinear
from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear
from tests.reference_solvers.python.diffrax_kencarp5 import (
    make_solver as make_diffrax_kencarp5_solver,
)
from tests.reference_solvers.python.diffrax_kvaerno5 import (
    make_solver as make_kvaerno5_solver,
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

_TIMES = jnp.array((0.0, 0.025, 0.05, 0.075, 0.1), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]


def _make_heat_system(n_vars):
    """Construct a semi-discretized 1D heat equation with n_vars interior grid points.

    Grid: Δx = 1/(n_vars + 1), interior points x_i = (i+1)·Δx for i = 0, …, n_vars−1.
    Diffusion matrix M: symmetric tridiagonal, (1/Δx²)(δᵢ,ⱼ₋₁ − 2δᵢⱼ + δᵢ,ⱼ₊₁).
    Stiffness: eigenvalues span [λ_N, λ_1] ≈ [−4/Δx², −π²], ratio ∝ N².
    """
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    dx = 1.0 / (n_vars + 1)
    inv_dx2 = 1.0 / dx**2

    M_np = (
        np.diag(-2.0 * inv_dx2 * np.ones(n_vars))
        + np.diag(inv_dx2 * np.ones(n_vars - 1), 1)
        + np.diag(inv_dx2 * np.ones(n_vars - 1), -1)
    )
    M = jnp.array(M_np, dtype=jnp.float64)

    x = np.arange(1, n_vars + 1) * dx
    y0 = jnp.array(np.sin(np.pi * x), dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        return p[0] * (M @ y)

    def jac_fn(t, p):
        del t
        return p[0] * M

    def explicit_jac_fn(t, p):
        del t, p
        return jnp.zeros_like(M)

    def implicit_jac_fn(t, p):
        return jac_fn(t, p)

    def explicit_ode_fn(y, t, p):
        del y, t, p
        return jnp.zeros_like(y0)

    def implicit_ode_fn(y, t, p):
        return ode_fn(y, t, p)

    return {
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "jac_fn": jac_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "explicit_jac_fn": explicit_jac_fn,
        "implicit_jac_fn": implicit_jac_fn,
        "y0": y0,
    }


def _exact_solution(n_vars, times, params_batch):
    """Exact discrete solution for sin(π·x) IC: y_i(t) = y0_i · exp(λ₁ · α · t).

    Since y0 is the first eigenmode of M, the time evolution is a pure exponential
    decay with the first eigenvalue λ₁ = (2/Δx²)(cos(π·Δx) − 1).

    Returns array of shape (ensemble_size, n_times, n_vars).
    """
    dx = 1.0 / (n_vars + 1)
    x = np.arange(1, n_vars + 1) * dx
    y0 = np.sin(np.pi * x)  # (n_vars,)
    lambda_1 = (2.0 / dx**2) * (np.cos(np.pi * dx) - 1.0)  # first eigenvalue of M
    alpha = np.asarray(params_batch)[:, 0]  # (ensemble_size,)
    times_np = np.asarray(times)  # (n_times,)
    decay = np.exp(lambda_1 * alpha[:, None] * times_np[None, :])  # (N, n_times)
    return y0[None, None, :] * decay[:, :, None]  # (N, n_times, n_vars)


def _dim_id(n_vars):
    return f"{n_vars}d"


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _run_julia_heat(
    benchmark, solver_factory, heat_system, ensemble_size, ensemble_backend
):
    system = heat_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = solver_factory(
        "heat_equation",
        system_config={"n_vars": system["n_vars"]},
        ensemble_backend=ensemble_backend,
    )
    results_np = benchmark_julia_solver(
        benchmark,
        solve,
        y0=system["y0"],
        t_span=_TIMES,
        params=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )
    return system, results_np, params


@pytest.fixture
def heat_system(request):
    """Configurable heat equation system parameterized by grid dimension."""
    return _make_heat_system(request.param)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(benchmark, heat_system, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with exact-solution validation."""
    system = heat_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_nonlinear(ode_fn=system["ode_fn"], lu_precision=lu_precision)
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5_nonlinear(benchmark, heat_system, ensemble_size, lu_precision):
    """KenCarp5 nonlinear benchmark with exact-solution validation."""
    system = heat_system
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
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, heat_system, ensemble_size):
    """Diffrax KenCarp5 benchmark with exact-solution validation."""
    system = heat_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_diffrax_kencarp5_solver(
        system["explicit_ode_fn"], system["implicit_ode_fn"]
    )
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Diffrax Kvaerno5 (reference solver timing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size",
    [
        pytest.param(n, marks=pytest.mark.slow) if n >= 10000 else n
        for n in _ENSEMBLE_SIZES
    ],
)
def test_diffrax_kvaerno5(benchmark, heat_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark with exact-solution validation."""
    system = heat_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_kvaerno5_solver(system["ode_fn"])
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    results_np = np.asarray(results)
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_tsit5(benchmark, heat_system, ensemble_size, ensemble_backend):
    """Julia Tsit5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_heat(
        benchmark, make_julia_tsit5_solver, heat_system, ensemble_size, ensemble_backend
    )
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, heat_system, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_heat(
        benchmark,
        make_julia_kencarp5_solver,
        heat_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, heat_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_heat(
        benchmark,
        make_julia_rodas5_solver,
        heat_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("heat_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES)
)
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(benchmark, heat_system, ensemble_size, ensemble_backend):
    """Julia Kvaerno5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_heat(
        benchmark,
        make_julia_kvaerno5_solver,
        heat_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system["n_vars"], _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    assert np.all(results_np >= -1e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)
