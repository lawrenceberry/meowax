"""
Tests for the Rodas5 linear and nonlinear solvers on nearest-neighbor reaction systems.

NN reactions — stiffness from rate constant heterogeneity

M is tridiagonal with entries that span 8 orders of magnitude. kf goes from 1e-2 to 1e6 and kb goes from 1e6 to 1e-2 across the chain, so diagonal entries like -(kb[i-1] + kf[i]) vary wildly. The stiffness ratio (largest / smallest |eigenvalue|) is roughly 10⁸ — fixed regardless of N. Adding more variables doesn't make the problem harder; the matrix just gets larger but equally stiff.
"""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.linear.rodas5_linear import make_solver as make_rodas5_linear
from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear
from tests.reference_solvers.python.diffrax_kvaerno5 import (
    make_solver as make_kvaerno5_solver,
)

_T_SPAN = (0.0, 1.0)
_MULTI_SAVE_TIMES = jnp.array((0.0, 0.125, 0.25, 0.5, 1.0), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]


def _make_nn_reaction_system(n_vars):
    """Construct a configurable D-dimensional nearest-neighbor reaction system."""
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    edge_count = n_vars - 1
    kf = np.array(
        [10.0 ** (-2.0 + 8.0 * i / (edge_count - 1)) for i in range(edge_count)]
    )
    kb = np.array(
        [10.0 ** (6.0 - 8.0 * i / (edge_count - 1)) for i in range(edge_count)]
    )

    M_np = np.zeros((n_vars, n_vars), dtype=np.float64)
    M_np[0, 0] = -kf[0]
    M_np[0, 1] = kb[0]
    for i in range(1, n_vars - 1):
        M_np[i, i - 1] = kf[i - 1]
        M_np[i, i] = -(kb[i - 1] + kf[i])
        M_np[i, i + 1] = kb[i]
    M_np[n_vars - 1, n_vars - 2] = kf[n_vars - 2]
    M_np[n_vars - 1, n_vars - 1] = -kb[n_vars - 2]

    M = jnp.array(M_np, dtype=jnp.float64)
    y0 = jnp.array([1.0] + [0.0] * (n_vars - 1), dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        return p[0] * (M @ y)

    def jac_fn(t, p):
        del t
        return p[0] * M

    return {
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "jac_fn": jac_fn,
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
def nn_reaction_system(request):
    """Configurable nearest-neighbor reaction system parameterized by dimension."""
    return _make_nn_reaction_system(request.param)


# ---------------------------------------------------------------------------
# Linear solver (jac_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_rodas5_linear(benchmark, nn_reaction_system, ensemble_size, precision):
    """Rodas5 linear (jac_fn) ensemble benchmark parameterised by dim, size, and precision."""
    system = nn_reaction_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_linear(
        jac_fn=system["jac_fn"], lu_precision=precision, mv_precision=precision
    )
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", [70], indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", [2])
@pytest.mark.parametrize("precision", ["fp32"])
def test_rodas5_linear_matches_reference(nn_reaction_system, ensemble_size, precision):
    """Validate rodas5 linear against Kvaerno5 (diffrax) on a 70D system, N=2, fp32."""
    system = nn_reaction_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_linear(
        jac_fn=system["jac_fn"], lu_precision=precision, mv_precision=precision
    )
    solve_ref = make_kvaerno5_solver(system["ode_fn"])

    y = solve(
        y0=system["y0"],
        t_span=_MULTI_SAVE_TIMES,
        params=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = solve_ref(
        y0=system["y0"],
        t_span=_MULTI_SAVE_TIMES,
        params=params,
        first_step=1e-6,
        rtol=1e-8,
        atol=1e-10,
    ).block_until_ready()

    assert y.shape == (2, len(_MULTI_SAVE_TIMES), 70)
    np.testing.assert_allclose(y.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=2e-4, atol=3e-8)


# ---------------------------------------------------------------------------
# Nonlinear solver (ode_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(benchmark, nn_reaction_system, ensemble_size, precision):
    """Rodas5 nonlinear (ode_fn) ensemble benchmark parameterised by dim, size, and precision."""
    system = nn_reaction_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_nonlinear(ode_fn=system["ode_fn"], lu_precision=precision)
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", [70], indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", [2])
@pytest.mark.parametrize("precision", ["fp32"])
def test_rodas5_nonlinear_matches_reference(
    nn_reaction_system, ensemble_size, precision
):
    """Validate rodas5 nonlinear against Kvaerno5 (diffrax) on a 70D system, N=2, fp32."""
    system = nn_reaction_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_nonlinear(ode_fn=system["ode_fn"], lu_precision=precision)
    solve_ref = make_kvaerno5_solver(system["ode_fn"])

    y = solve(
        y0=system["y0"],
        t_span=_MULTI_SAVE_TIMES,
        params=params,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = solve_ref(
        y0=system["y0"],
        t_span=_MULTI_SAVE_TIMES,
        params=params,
        first_step=1e-6,
        rtol=1e-8,
        atol=1e-10,
    ).block_until_ready()

    assert y.shape == (2, len(_MULTI_SAVE_TIMES), 70)
    np.testing.assert_allclose(y.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=2e-4, atol=3e-8)
