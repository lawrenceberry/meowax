"""Tests for the Rodas5 nonlinear solver on coupled van der Pol oscillator systems."""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np
import pytest

from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear

_T_SPAN = (0.0, 1.0)
_OSC_PAIRS = [15, 25, 35]   # oscillator pairs → 30D, 50D, 70D
_MU_SCALES = [1, 10, 100]   # mu_max: stiffness scales as mu²
_ENSEMBLE_SIZES = [2, 100, 1000, 10000]


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

    return {"n_osc": n_osc, "mu_max": mu_max, "n_vars": n_vars, "ode_fn": ode_fn, "y0": y0}


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


@pytest.mark.parametrize(
    "vdp_system",
    [(n, m) for n in _OSC_PAIRS for m in _MU_SCALES],
    indirect=True,
    ids=lambda p: f"{p[0]}osc-mu{p[1]}",
)
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
