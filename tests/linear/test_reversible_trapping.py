"""Tests for the Rodas5 linear and nonlinear solvers on reversible trapping systems.

Reversible trapping — constant multiscale stiffness from local exchange plus slow transport

This is a linear mobile/immobile transport model on a 1D chain with zero-flux boundaries:

    m_t = D L m - k_on m + k_off s
    s_t = k_on m - k_off s

where m is the mobile concentration, s is the trapped concentration, and L is the
Neumann Laplacian. The diffusion block is slow and spatially coupled, while the
trapping/untrapping block is fast and local to each cell.

Why this is a good future IMEX test
-----------------------------------
The stiff exchange block is block-diagonal: each cell contributes only a local 2x2
trapping operator. A split IMEX method can therefore treat that local exchange
implicitly and keep the slow global diffusion explicit. A fully implicit solver,
by contrast, factorizes the full coupled 2N x 2N system every step even though the
stiffness comes from local reactions rather than long-range transport.

Mass conservation
-----------------
The Neumann diffusion matrix has zero column sum, and the trapping block transfers
mass between mobile and trapped states without creating or destroying it. Therefore
sum(m) + sum(s) stays exactly constant in the continuous system; the tests use this
as a robust invariant.

fp32 cancellation hazard — mv_precision should stay fp64
--------------------------------------------------------
Once the fast exchange transient relaxes, the mobile and trapped populations sit near a
local quasi-equilibrium. The exchange term then evaluates differences of large, nearly
cancelling contributions such as -k_on m + k_off s, whose net effect is slow even though
the individual terms are O(10^4). In fp32 this cancellation inflates the residual seen
by the Rosenbrock error estimator and makes the linear solver take far too many tiny
steps. As with the heat-equation test, the stable benchmark path is to keep the
matrix-vector products in fp64 and vary only the LU precision.
"""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.linear.rodas5_linear import make_solver as make_rodas5_linear
from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear
from tests.reference_solvers.python.diffrax_kvaerno5 import (
    make_cached_solver as make_cached_kvaerno5_solver,
)

_DIFFUSION_COEFF = 2e-2
_ADSORPTION_RATE = 3e4
_DESORPTION_RATE = 3e3

_TIMES = jnp.array((0.0, 1e-4, 1e-3, 1e-2, 5e-1), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000]
_REFERENCE_ENSEMBLE_SIZES = [2]


def _make_neumann_laplacian(n_cells, dx):
    """Construct the 1D Neumann Laplacian on n_cells cell centers."""
    laplacian = np.zeros((n_cells, n_cells), dtype=np.float64)
    for i in range(n_cells):
        laplacian[i, i] = -2.0
        if i > 0:
            laplacian[i, i - 1] = 1.0
        if i < n_cells - 1:
            laplacian[i, i + 1] = 1.0

    laplacian[0, 0] = -1.0
    laplacian[-1, -1] = -1.0
    return laplacian / dx**2


def _make_reversible_trapping_system(n_vars):
    """Construct a reversible trapping system with n_vars = 2 * n_cells."""
    if n_vars < 4:
        raise ValueError(f"n_vars must be at least 4, got {n_vars}")
    if n_vars % 2 != 0:
        raise ValueError(f"n_vars must be even, got {n_vars}")

    n_cells = n_vars // 2
    dx = 1.0 / (n_cells - 1)
    laplacian = _make_neumann_laplacian(n_cells, dx)

    diffusion_np = np.zeros((n_vars, n_vars), dtype=np.float64)
    diffusion_np[:n_cells, :n_cells] = _DIFFUSION_COEFF * laplacian

    exchange_np = np.zeros((n_vars, n_vars), dtype=np.float64)
    exchange_np[:n_cells, :n_cells] = -_ADSORPTION_RATE * np.eye(n_cells)
    exchange_np[:n_cells, n_cells:] = _DESORPTION_RATE * np.eye(n_cells)
    exchange_np[n_cells:, :n_cells] = _ADSORPTION_RATE * np.eye(n_cells)
    exchange_np[n_cells:, n_cells:] = -_DESORPTION_RATE * np.eye(n_cells)

    diffusion = jnp.array(diffusion_np, dtype=jnp.float64)
    exchange = jnp.array(exchange_np, dtype=jnp.float64)

    x = np.linspace(0.0, 1.0, n_cells)
    mobile0 = np.exp(-120.0 * (x - 0.15) ** 2)
    mobile0 /= mobile0.sum()
    trapped0 = np.zeros(n_cells, dtype=np.float64)
    y0 = jnp.array(np.concatenate([mobile0, trapped0]), dtype=jnp.float64)

    def jac_nonstiff_fn(t, p):
        del t
        return p[0] * diffusion

    def jac_stiff_fn(t, p):
        del t
        return p[0] * exchange

    def jac_fn(t, p):
        return jac_nonstiff_fn(t, p) + jac_stiff_fn(t, p)

    def ode_fn(y, t, p):
        return jac_fn(t, p) @ y

    return {
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "jac_fn": jac_fn,
        "jac_nonstiff_fn": jac_nonstiff_fn,
        "jac_stiff_fn": jac_stiff_fn,
        "y0": y0,
    }


def _dim_id(n_vars):
    return f"{n_vars}d"


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


@pytest.fixture
def reversible_trapping_system(request):
    """Configurable reversible trapping system parameterized by state dimension."""
    return _make_reversible_trapping_system(request.param)


# ---------------------------------------------------------------------------
# Linear solver (jac_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reversible_trapping_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_linear(
    benchmark, reversible_trapping_system, ensemble_size, lu_precision
):
    """Rodas5 linear benchmark with cached Diffrax validation on practical ensemble sizes."""
    system = reversible_trapping_system
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
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=3e-6)
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
        np.testing.assert_allclose(
            results_np, np.asarray(y_ref), rtol=2e-4, atol=3e-8
        )


# ---------------------------------------------------------------------------
# Nonlinear solver (ode_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reversible_trapping_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(
    benchmark, reversible_trapping_system, ensemble_size, lu_precision
):
    """Rodas5 nonlinear benchmark with cached Diffrax validation on practical ensemble sizes."""
    system = reversible_trapping_system
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
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=3e-6)
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
        np.testing.assert_allclose(
            results_np, np.asarray(y_ref), rtol=2e-4, atol=3e-8
        )
