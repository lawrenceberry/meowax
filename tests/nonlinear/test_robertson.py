"""Tests for the Rodas5 nonlinear solver on the Robertson stiff ODE system."""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear

_T_SPAN = (0.0, 1e5)
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]


def _make_robertson_system():
    """Construct the Robertson stiff ODE system (3-variable).

    Stiff ODE system (Appendix A.1.3, arXiv:2304.06835).
    Standard parameters: k1=0.04, k2=1e4, k3=3e7.
    """
    y0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        return jnp.array([
            -p[0] * y[0] + p[1] * y[1] * y[2],
            p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
            p[2] * y[1] ** 2,
        ])

    return {"n_vars": 3, "ode_fn": ode_fn, "y0": y0}


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    base = np.array([0.04, 1e4, 3e7])
    return jnp.array(
        base * (1.0 + 0.1 * (2.0 * rng.random((size, 3)) - 1.0)),
        dtype=jnp.float64,
    )


@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(benchmark, ensemble_size, precision):
    """Rodas5 nonlinear ensemble benchmark on the Robertson stiff system."""
    system = _make_robertson_system()
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_nonlinear(ode_fn=system["ode_fn"], lu_precision=precision)
    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_T_SPAN,
            params=params,
            first_step=1e-4,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=1e-6)
