"""Tests for the Rodas5 nonlinear solver on coupled Kaps singular-perturbation systems.

Kaps' problem — controllable stiffness with a known exact solution

Each equation pair (y1_i, y2_i) is an independent copy of the original Kaps system
with a different stiffness parameter ε_i. The stiffness ratio of pair i is ≈ 1/ε_i,
so ε_min controls the maximum stiffness across the system. The initial conditions lie
exactly on the slow manifold, so there is no fast initial transient — only the stable
fast modes that implicit solvers must handle without being forced into tiny steps.

The analytical solution is independent of ε_i for all pairs:
    y1_i(t) = exp(−2t),  y2_i(t) = exp(−t)

which allows precise validation at arbitrary stiffness without a reference solver.
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

_TIMES = jnp.array((0.0, 0.5, 1.0, 2.0), dtype=jnp.float64)
_N_PAIRS = [15, 25, 35]  # equation pairs → 30D, 50D, 70D
_EPSILON_MIN = [1e-2, 1e-4, 1e-6]  # smallest ε: stiffness ratio ≈ 1/ε_min
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]


def _make_kaps_system(n_pairs, epsilon_min):
    """Construct n_pairs coupled Kaps singular-perturbation equation pairs (2*n_pairs state variables).

    State ordering: (y1_0, y2_0, y1_1, y2_1, ..., y1_{n-1}, y2_{n-1}).
    ODE:
        dy1_i/dt = p[0] · (−(1/ε_i + 2) · y1_i + (1/ε_i) · y2_i²)
        dy2_i/dt = p[0] · (y1_i − y2_i − y2_i²)

    ε_i    — stiffness parameters on a log scale from 1 (mild) down to epsilon_min (very stiff).
              Stiffness of the i-th pair scales as 1/ε_i.
    p[0]   — global time-scale factor (ensemble parameter, ≈1 ± 10%).

    Exact solution (independent of ε_i for all i):
        y1_i(t) = exp(−2·p[0]·t)
        y2_i(t) = exp(−p[0]·t)
    """
    n_vars = 2 * n_pairs

    # Stiffness: ε = 1 (mild) down to epsilon_min (very stiff) on a log scale
    epsilon = jnp.array(
        [10.0 ** (np.log10(epsilon_min) * i / (n_pairs - 1)) for i in range(n_pairs)],
        dtype=jnp.float64,
    )

    y0 = jnp.array([1.0, 1.0] * n_pairs, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        s = p[0]
        y1 = y[0::2]
        y2 = y[1::2]
        dy1 = s * (-(1.0 / epsilon + 2.0) * y1 + (1.0 / epsilon) * y2**2)
        dy2 = s * (y1 - y2 - y2**2)
        return jnp.stack([dy1, dy2], axis=1).ravel()

    def explicit_ode_fn(y, t, p):
        del t
        s = p[0]
        y1 = y[0::2]
        y2 = y[1::2]
        dy1 = s * (-2.0 * y1)
        dy2 = s * (y1 - y2 - y2**2)
        return jnp.stack([dy1, dy2], axis=1).ravel()

    def implicit_ode_fn(y, t, p):
        del t
        s = p[0]
        y1 = y[0::2]
        y2 = y[1::2]
        dy1 = s * (-(1.0 / epsilon) * (y1 - y2**2))
        dy2 = jnp.zeros_like(y2)
        return jnp.stack([dy1, dy2], axis=1).ravel()

    return {
        "n_pairs": n_pairs,
        "epsilon_min": epsilon_min,
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "y0": y0,
    }


def _exact_solution(t_span, params, n_pairs):
    """Exact solution for n_pairs Kaps equation pairs.

    y1_i(t) = exp(−2·p[0]·t),  y2_i(t) = exp(−p[0]·t)  for all i (independent of ε_i).

    Returns array of shape (N, n_save, 2*n_pairs).
    """
    t = np.asarray(t_span, dtype=np.float64)  # (n_save,)
    s = np.asarray(params)[:, 0]  # (N,)
    y1 = np.exp(-2.0 * np.outer(s, t))  # (N, n_save)
    y2 = np.exp(-1.0 * np.outer(s, t))  # (N, n_save)
    pair = np.stack([y1, y2], axis=-1)  # (N, n_save, 2)
    return np.tile(pair, (1, 1, n_pairs))  # (N, n_save, 2*n_pairs)


@pytest.fixture
def kaps_system(request):
    """Kaps system parameterized by (n_pairs, epsilon_min)."""
    n_pairs, epsilon_min = request.param
    return _make_kaps_system(n_pairs, epsilon_min)


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _run_julia_kaps(
    benchmark, solver_factory, kaps_system, ensemble_size, ensemble_backend
):
    system = kaps_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = solver_factory(
        "kaps",
        system_config={
            "n_pairs": system["n_pairs"],
            "epsilon_min": system["epsilon_min"],
        },
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


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(benchmark, kaps_system, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with exact-solution validation."""
    system = kaps_system
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

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5_nonlinear(benchmark, kaps_system, ensemble_size, lu_precision):
    """KenCarp5 nonlinear benchmark with exact-solution validation."""
    system = kaps_system
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

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, kaps_system, ensemble_size):
    """Diffrax KenCarp5 benchmark on coupled Kaps singular-perturbation systems."""
    system = kaps_system
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

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Diffrax Kvaerno5 (reference solver timing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kvaerno5(benchmark, kaps_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark on coupled Kaps singular-perturbation systems."""
    system = kaps_system
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

    assert results.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_tsit5(benchmark, kaps_system, ensemble_size, ensemble_backend):
    """Julia Tsit5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_kaps(
        benchmark, make_julia_tsit5_solver, kaps_system, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, kaps_system, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_kaps(
        benchmark,
        make_julia_kencarp5_solver,
        kaps_system,
        ensemble_size,
        ensemble_backend,
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, kaps_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_kaps(
        benchmark, make_julia_rodas5_solver, kaps_system, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "kaps_system",
    [(n, e) for n in _N_PAIRS for e in _EPSILON_MIN],
    indirect=True,
    ids=lambda p: f"{p[0]}pairs-eps{p[1]:.0e}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(benchmark, kaps_system, ensemble_size, ensemble_backend):
    """Julia Kvaerno5 benchmark with exact-solution validation."""
    system, results_np, params = _run_julia_kaps(
        benchmark,
        make_julia_kvaerno5_solver,
        kaps_system,
        ensemble_size,
        ensemble_backend,
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    y_exact = _exact_solution(_TIMES, params, system["n_pairs"])
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)
