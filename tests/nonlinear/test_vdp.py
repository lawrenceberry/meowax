"""Tests for the Rodas5 nonlinear solver on coupled van der Pol oscillator systems."""

import jax.numpy as jnp
import numpy as np
import pytest

from solvers.nonlinear.kencarp5_nonlinear import make_solver as make_kencarp5_nonlinear
from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear
from tests.reference_solvers.python.diffrax_kencarp5 import (
    make_solver as make_diffrax_kencarp5_solver,
)
from tests.reference_solvers.python.diffrax_kvaerno5 import (
    make_cached_solver as make_cached_kvaerno5_solver,
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

_TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
_OSC_PAIRS = [15, 25, 35]  # oscillator pairs → 30D, 50D, 70D
_MU_SCALES = [1, 10, 100]  # mu_max: stiffness scales as mu²
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]
_REFERENCE_ENSEMBLE_SIZES = [2]


def _make_vdp_system(n_osc, mu_max):
    """Construct n_osc van der Pol oscillator pairs (2*n_osc state variables).

    State ordering: (x_0, v_0, x_1, v_1, ..., x_{n-1}, v_{n-1}).
    ODE:
        dx_i/dt = v_i
        dv_i/dt = p[0] · mu_i · (1 − x_i²) · v_i  −  omega_i² · x_i

    mu_i    — damping coefficients on a log scale from 1 to mu_max.
              Stiffness of the i-th oscillator scales as mu_i², so mu_max
              controls the overall stiffness of the system.
    omega_i — natural frequencies on a log scale from ~0.32 to ~3.16 rad/time
              (geometric mean = 1, 10× range, fixed across all mu_max values).
    p[0]    — global damping scale factor (ensemble parameter, ≈1 ± 10%).
    """
    n_vars = 2 * n_osc

    # Damping: 1 to mu_max on a log scale (uniform when mu_max = 1)
    mu = jnp.array(
        [10.0 ** (np.log10(max(mu_max, 1)) * i / (n_osc - 1)) for i in range(n_osc)],
        dtype=jnp.float64,
    )

    # Natural frequencies: 10^(-0.5) to 10^(+0.5) ≈ 0.32 to 3.16 rad/time unit
    omega = jnp.array(
        [10.0 ** (-0.5 + i / (n_osc - 1)) for i in range(n_osc)],
        dtype=jnp.float64,
    )

    y0 = jnp.array([2.0, 0.0] * n_osc, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        s = p[0]
        x = y[0::2]
        v = y[1::2]
        return jnp.stack([v, s * mu * (1.0 - x * x) * v - omega**2 * x], axis=1).ravel()

    def explicit_ode_fn(y, t, p):
        del t
        x = y[0::2]
        v = y[1::2]
        return jnp.stack([v, -(omega**2) * x], axis=1).ravel()

    def implicit_ode_fn(y, t, p):
        del t
        s = p[0]
        x = y[0::2]
        v = y[1::2]
        return jnp.stack(
            [jnp.zeros_like(x), s * mu * (1.0 - x * x) * v], axis=1
        ).ravel()

    return {
        "n_osc": n_osc,
        "mu_max": mu_max,
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "explicit_ode_fn": explicit_ode_fn,
        "implicit_ode_fn": implicit_ode_fn,
        "y0": y0,
    }


@pytest.fixture
def vdp_system(request):
    """Van der Pol system parameterized by (n_osc, mu_max)."""
    n_osc, mu_max = request.param
    return _make_vdp_system(n_osc, mu_max)


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _run_julia_vdp(benchmark, solver_factory, vdp_system, ensemble_size, ensemble_backend):
    system = vdp_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = solver_factory(
        "vdp",
        system_config={"n_osc": system["n_osc"], "mu_max": system["mu_max"]},
        ensemble_backend=ensemble_backend,
    )
    return system, benchmark_julia_solver(
        benchmark,
        solve,
        y0=system["y0"],
        t_span=_TIMES,
        params=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(benchmark, vdp_system, ensemble_size, lu_precision):
    """Rodas5 nonlinear benchmark with cached Diffrax validation on practical ensemble sizes."""
    system = vdp_system
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

    if ensemble_size in _REFERENCE_ENSEMBLE_SIZES:
        solve_ref = make_cached_kvaerno5_solver(system["ode_fn"])
        y_ref = solve_ref(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready()
        np.testing.assert_allclose(results_np, np.asarray(y_ref), rtol=5e-4, atol=3e-8)


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_kencarp5_nonlinear(benchmark, vdp_system, ensemble_size, lu_precision):
    """KenCarp5 nonlinear benchmark with cached Diffrax validation on practical ensemble sizes."""
    system = vdp_system
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

    if ensemble_size in _REFERENCE_ENSEMBLE_SIZES:
        solve_ref = make_cached_kvaerno5_solver(system["ode_fn"])
        y_ref = solve_ref(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready()
        np.testing.assert_allclose(results_np, np.asarray(y_ref), rtol=5e-4, atol=3e-8)


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kencarp5(benchmark, vdp_system, ensemble_size):
    """Diffrax KenCarp5 benchmark with cached Diffrax validation on practical ensemble sizes."""
    system = vdp_system
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

    if ensemble_size in _REFERENCE_ENSEMBLE_SIZES:
        solve_ref = make_cached_kvaerno5_solver(system["ode_fn"])
        y_ref = solve_ref(
            y0=system["y0"],
            t_span=_TIMES,
            params=params,
            first_step=1e-6,
            rtol=1e-8,
            atol=1e-10,
        ).block_until_ready()
        np.testing.assert_allclose(results_np, np.asarray(y_ref), rtol=5e-4, atol=3e-8)


# ---------------------------------------------------------------------------
# Diffrax Kvaerno5 (reference solver timing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize(
    "ensemble_size",
    [
        pytest.param(n, marks=pytest.mark.slow) if n >= 10000 else n
        for n in _ENSEMBLE_SIZES
    ],
)
def test_diffrax_kvaerno5(benchmark, vdp_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark on coupled van der Pol oscillators."""
    system = vdp_system
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


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_tsit5(benchmark, vdp_system, ensemble_size, ensemble_backend):
    """Julia Tsit5 benchmark on coupled van der Pol oscillators."""
    system, results_np = _run_julia_vdp(
        benchmark, make_julia_tsit5_solver, vdp_system, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kencarp5(benchmark, vdp_system, ensemble_size, ensemble_backend):
    """Julia KenCarp5 benchmark on coupled van der Pol oscillators."""
    system, results_np = _run_julia_vdp(
        benchmark,
        make_julia_kencarp5_solver,
        vdp_system,
        ensemble_size,
        ensemble_backend,
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5(benchmark, vdp_system, ensemble_size, ensemble_backend):
    """Julia Rodas5 benchmark on coupled van der Pol oscillators."""
    system, results_np = _run_julia_vdp(
        benchmark, make_julia_rodas5_solver, vdp_system, ensemble_size, ensemble_backend
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
@pytest.mark.parametrize("ensemble_size", maybe_mark_large_ensemble_sizes(_ENSEMBLE_SIZES))
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_kvaerno5(benchmark, vdp_system, ensemble_size, ensemble_backend):
    """Julia Kvaerno5 benchmark on coupled van der Pol oscillators."""
    system, results_np = _run_julia_vdp(
        benchmark,
        make_julia_kvaerno5_solver,
        vdp_system,
        ensemble_size,
        ensemble_backend,
    )
    assert results_np.shape == (ensemble_size, len(_TIMES), system["n_vars"])
    assert np.all(np.isfinite(results_np))
