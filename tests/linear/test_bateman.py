"""Tests for Rodas5 linear and nonlinear solvers on Bateman radioactive decay chains.

Physical background
-------------------
The Bateman equations (H. Bateman, 1910) describe a sequential radioactive decay
chain A₁ → A₂ → … → Aₙ, where each species Aᵢ decays into Aᵢ₊₁ with first-order
rate constant λᵢ.  The final species Aₙ is stable (no further decay).  Starting
from a pure parent population — N₁(0) = 1, all others zero — the system evolves as
the parent feeds the chain, each intermediate species is produced by its predecessor
and destroyed by its own decay, and the stable end product accumulates irreversibly.

ODE system
----------
    dN₁/dt =                  −λ₁ N₁
    dNᵢ/dt = λᵢ₋₁ Nᵢ₋₁     − λᵢ Nᵢ    (i = 2, …, n − 1)
    dNₙ/dt = λₙ₋₁ Nₙ₋₁                  (stable end product)

The coefficient matrix M is lower bidiagonal: diagonal entries −λᵢ, subdiagonal
entries +λᵢ.  A global rate-scale factor p[0] ≈ 1 ± 10% multiplies M as the
ensemble parameter, so the ensemble explores slightly different reaction speeds.

Total population is conserved: Σᵢ Nᵢ(t) = 1 for all t.

Analytical solution
-------------------
Because M is constant the exact solution is N(t) = expm(p[0] · M · t) · N₀, where
expm denotes the matrix exponential.  This is equivalent to the classic Bateman
formula (a sum of exponentials), but computed via scipy.linalg.expm to avoid the
catastrophic cancellation that affects the direct formula for long chains with
widely separated decay constants.  This exact solution is used for correctness
validation in the tests below.

Stiffness character — CONSTANT
-------------------------------
The Jacobian of the Bateman system equals p[0] · M identically — it is independent
of both t and the state y.  The stiffness ratio is therefore fixed at λ_max / λ_min
throughout the entire integration; it does not grow or shrink with time.  The
``stiffness`` parameter sets this ratio directly.

An explicit solver requires Δt < 2 / λ_max to remain stable; a step of that size
over an integration window of O(1 / λ_min) requires O(stiffness) steps.  An
implicit L-stable solver such as Rodas5 can take O(1) steps regardless of stiffness.

What this tests
---------------
- Rodas5 linear solver (jac_fn path): validates that the constant-Jacobian fast path
  produces accurate results across three chain lengths and three stiffness ratios.
- Rodas5 nonlinear solver (ode_fn path): validates the JVP-based Jacobian on a
  problem with a known exact solution.
- Diffrax Kvaerno5: provides a timing baseline for comparison.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from solvers.linear.kencarp5_linear import make_solver as make_kencarp5_linear
from solvers.linear.rodas5_linear import make_solver as make_rodas5_linear
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

_TIMES = jnp.array((0.0, 0.2, 0.5, 1.0, 2.0), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]  # chain lengths (species count including stable)
_STIFFNESS_RATIOS = [1e2, 1e4, 1e6]  # λ_max / λ_min
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]


def _make_bateman_system(n_vars, stiffness):
    """Construct an n_vars Bateman decay-chain system.

    Decay constants λᵢ are log-spaced from 1 (slowest parent) to ``stiffness``
    (fastest intermediate) across the n_vars − 1 radioactive species.
    """
    n_radioactive = n_vars - 1  # species that decay; last species is stable

    # λ₀ = 1 (slowest), λₙ₋₂ = stiffness (fastest), log-uniform in between
    lambdas = np.array(
        [
            10.0 ** (np.log10(stiffness) * i / (n_radioactive - 1))
            for i in range(n_radioactive)
        ],
        dtype=np.float64,
    )

    # Lower-bidiagonal coefficient matrix M
    M_np = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_radioactive):
        M_np[i, i] = -lambdas[i]  # loss from species i
        M_np[i + 1, i] = lambdas[i]  # gain into species i+1
    # M_np[n_vars-1, n_vars-1] = 0: stable end product, no decay term

    M = jnp.array(M_np, dtype=jnp.float64)
    y0 = jnp.array([1.0] + [0.0] * n_radioactive, dtype=jnp.float64)

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
        "stiffness": stiffness,
        "ode_fn": ode_fn,
        "jac_fn": jac_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "explicit_jac_fn": explicit_jac_fn,
        "implicit_jac_fn": implicit_jac_fn,
        "y0": y0,
        "M_np": M_np,
    }


def _exact_solution(M_np, y0_np, t_span, params):
    """Exact solution via eigendecomposition: N(t) = L · diag(exp(p[0]·t·d)) · L⁻¹ · N₀.

    M is lower-triangular with distinct real eigenvalues, so the decomposition is
    computed once (O(n³)) and then applied to the full ensemble in a single batched
    matmul (O(N · n_save · n²)), making this cheap even for large ensembles.

    Parameters
    ----------
    M_np : array, shape [n_vars, n_vars]
    y0_np : array, shape [n_vars]
    t_span : array, shape [n_save]
    params : array, shape [N, 1]

    Returns
    -------
    array, shape [N, n_save, n_vars]
    """
    d, L = np.linalg.eig(M_np)
    d = np.real(d)                       # eigenvalues are −λᵢ, all real
    L = np.real(L)                       # eigenvectors are all real
    w = np.linalg.solve(L, y0_np)       # L⁻¹ @ y0, shape (n_vars,)

    t_arr = np.asarray(t_span, dtype=np.float64)
    s_arr = np.asarray(params, dtype=np.float64)[:, 0]

    alpha = np.outer(s_arr, t_arr)                   # (N, n_save): p[0]·t per trajectory
    exp_vals = np.exp(alpha[:, :, None] * d)         # (N, n_save, n_vars)
    return (exp_vals * w) @ L.T                      # (N, n_save, n_vars)


@pytest.fixture
def bateman_system(request):
    """Bateman decay-chain system parameterized by (n_vars, stiffness)."""
    n_vars, stiffness = request.param
    return _make_bateman_system(n_vars, stiffness)


def _system_id(p):
    return f"{p[0]}vars-stiff1e{int(round(np.log10(p[1])))}"


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _run_julia_bateman(benchmark, solver_factory, bateman_system, ensemble_size, ensemble_backend):
    system = bateman_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = solver_factory(
        "bateman",
        system_config={
            "n_vars": system["n_vars"],
            "stiffness": system["stiffness"],
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


# ---------------------------------------------------------------------------
# Linear solver (jac_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_linear(benchmark, bateman_system, ensemble_size, lu_precision):
    """Rodas5 linear benchmark with conservation and exact-solution validation."""
    system = bateman_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_linear(jac_fn=system["jac_fn"], lu_precision=lu_precision)
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-2, atol=3e-5)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5_linear(benchmark, bateman_system, ensemble_size, lu_precision):
    """KenCarp5 linear benchmark with conservation and exact-solution validation."""
    system = bateman_system
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Nonlinear solver (ode_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(benchmark, bateman_system, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with conservation and exact-solution validation."""
    system = bateman_system
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5_nonlinear(benchmark, bateman_system, ensemble_size, lu_precision):
    """KenCarp5 nonlinear benchmark with conservation and exact-solution validation."""
    system = bateman_system
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, bateman_system, ensemble_size):
    """Diffrax KenCarp5 benchmark on Bateman decay chains."""
    system = bateman_system
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Diffrax Kvaerno5 (reference solver timing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kvaerno5(benchmark, bateman_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark on Bateman decay chains."""
    system = bateman_system
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
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_tsit5(benchmark, bateman_system, ensemble_size, ensemble_backend):
    """Julia Tsit5 benchmark on Bateman decay chains."""
    system, results_np, params = _run_julia_bateman(
        benchmark, make_julia_tsit5_solver, bateman_system, ensemble_size, ensemble_backend
    )
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, bateman_system, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark on Bateman decay chains."""
    system, results_np, params = _run_julia_bateman(
        benchmark,
        make_julia_kencarp5_solver,
        bateman_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, bateman_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark on Bateman decay chains."""
    system, results_np, params = _run_julia_bateman(
        benchmark, make_julia_rodas5_solver, bateman_system, ensemble_size, ensemble_backend
    )
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "bateman_system",
    [(n, s) for n in _SYSTEM_DIMS for s in _STIFFNESS_RATIOS],
    indirect=True,
    ids=_system_id,
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(benchmark, bateman_system, ensemble_size, ensemble_backend):
    """Julia Kvaerno5 benchmark on Bateman decay chains."""
    system, results_np, params = _run_julia_bateman(
        benchmark,
        make_julia_kvaerno5_solver,
        bateman_system,
        ensemble_size,
        ensemble_backend,
    )
    y_exact = _exact_solution(system["M_np"], np.asarray(system["y0"]), _TIMES, params)
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
    np.testing.assert_allclose(results_np.sum(axis=-1), 1.0, atol=3e-6)
    np.testing.assert_allclose(results_np, y_exact, rtol=1e-3, atol=1e-6)
