"""Tests for the Rodas5 nonlinear solver on coupled van der Pol oscillator systems."""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear

_T_SPAN = (0.0, 1.0)
_OSC_PAIRS = [15, 25, 35]  # oscillator pairs → 30D, 50D, 70D
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]


def _make_vdp_system(n_osc):
    """Construct n_osc van der Pol oscillator pairs (2*n_osc state variables).

    State ordering: (x_0, v_0, x_1, v_1, ..., x_{n-1}, v_{n-1}).
    Damping coefficients span ~0.5 to 100 on a log scale.
    Each oscillator i: dx_i/dt = v_i, dv_i/dt = p[0]*mu_i*(1 - x_i^2)*v_i - x_i.
    """
    n_vars = 2 * n_osc
    mu = jnp.array(
        [10.0 ** (-0.3 + 2.3 * i / (n_osc - 1)) for i in range(n_osc)],
        dtype=jnp.float64,
    )
    y0 = jnp.array([2.0, 0.0] * n_osc, dtype=jnp.float64)

    def ode_fn(y, t, p):
        del t
        s = p[0]
        x = y[0::2]
        v = y[1::2]
        return jnp.stack([v, s * mu * (1.0 - x * x) * v - x], axis=1).ravel()

    return {"n_osc": n_osc, "n_vars": n_vars, "ode_fn": ode_fn, "y0": y0}


def _osc_id(n_osc):
    return f"{n_osc}osc"


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


@pytest.fixture
def vdp_system(request):
    """Configurable van der Pol system parameterized by number of oscillator pairs."""
    return _make_vdp_system(request.param)


@pytest.mark.parametrize("vdp_system", _OSC_PAIRS, indirect=True, ids=_osc_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(benchmark, vdp_system, ensemble_size, precision):
    """Rodas5 nonlinear ensemble benchmark on the van der Pol system."""
    system = vdp_system
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
    assert np.all(np.isfinite(results))
