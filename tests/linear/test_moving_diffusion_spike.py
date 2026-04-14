"""Tests for the Rodas5 linear and nonlinear solvers on transient diffusion spike systems.

Transient diffusion spike — transient multiscale stiffness from a localized fast zone

This is a linear diffusion problem with time-varying face diffusivity:

    u_t = d/dx ( D(x, t) u_x )

The background diffusion is weak everywhere, while a narrow high-conductivity band
is localized in space and turns on only during a finite time window. The problem is
therefore only stiff for a subset of edges and only over a subset of the trajectory.

Why this is a good future adaptive-IMEX test
--------------------------------------------
A static IMEX split can mark the whole spike contribution implicit up front, but it
still pays that implicit cost even when the localized fast band is effectively off.
An adaptive IMEX method that inspects the Jacobian each step can identify when the
fast zone is currently active and keep only those transiently stiff couplings in the
implicit partition, leaving the background diffusion explicit elsewhere.

Mass conservation
-----------------
The operator is assembled from conservative face fluxes with zero-flux boundary
conditions. Boundary faces are set to zero flux, internal face contributions are
shared symmetrically, and the resulting Jacobian has zero column sum. The tests
therefore verify conservation of total discrete mass.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from solvers.linear.rodas5_linear import make_solver as make_rodas5_linear
from solvers.nonlinear.rodas5_nonlinear import make_solver as make_rodas5_nonlinear
from tests.reference_solvers.python.diffrax_kvaerno5 import (
    make_cached_solver as make_cached_kvaerno5_solver,
    make_solver as make_kvaerno5_solver,
)

_BACKGROUND_DIFFUSIVITY = 1e-4
_SPIKE_DIFFUSIVITY = 2e-1
_SPIKE_WIDTH = 5e-2
_SPIKE_CENTER = 0.55
_SPIKE_TIME_CENTER = 0.5
_SPIKE_TIME_WIDTH = 0.2
_FINAL_TIME = 1.0
_REFERENCE_RTOL = 7e-2

_TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000]
_REFERENCE_ENSEMBLE_SIZES = [2]


def _matrix_from_face_diffusivity(face_diffusivity, dx):
    """Assemble the conservative diffusion matrix from face diffusivities."""
    diag = -(face_diffusivity[:-1] + face_diffusivity[1:]) / dx**2
    offdiag = face_diffusivity[1:-1] / dx**2
    return jnp.diag(diag) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)


def _make_moving_diffusion_spike_system(n_vars):
    """Construct a diffusion system with a transient localized diffusivity spike."""
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    dx = 1.0 / (n_vars + 1)
    x = np.arange(1, n_vars + 1) * dx
    x_faces = jnp.array(np.arange(n_vars + 1) * dx, dtype=jnp.float64)

    y0 = np.exp(-4.0 * (x - 0.25) ** 2)
    y0 /= y0.sum()
    y0 = jnp.array(y0, dtype=jnp.float64)

    background_faces = jnp.full(
        (n_vars + 1,), _BACKGROUND_DIFFUSIVITY, dtype=jnp.float64
    )
    background_faces = background_faces.at[0].set(0.0).at[-1].set(0.0)
    background_matrix = _matrix_from_face_diffusivity(background_faces, dx)
    spatial_profile = jnp.exp(-(((x_faces - _SPIKE_CENTER) / _SPIKE_WIDTH) ** 2))
    spatial_profile = spatial_profile.at[0].set(0.0).at[-1].set(0.0)

    def _spike_faces(t, p):
        pulse = p[0] * jnp.exp(-(((t - _SPIKE_TIME_CENTER) / _SPIKE_TIME_WIDTH) ** 2))
        return _SPIKE_DIFFUSIVITY * pulse * spatial_profile

    def jac_nonstiff_fn(t, p):
        del t, p
        return background_matrix

    def jac_stiff_fn(t, p):
        return _matrix_from_face_diffusivity(_spike_faces(t, p), dx)

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
def moving_diffusion_spike_system(request):
    """Configurable moving diffusion spike system parameterized by dimension."""
    return _make_moving_diffusion_spike_system(request.param)


# ---------------------------------------------------------------------------
# Linear solver (jac_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moving_diffusion_spike_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_linear(
    benchmark, moving_diffusion_spike_system, ensemble_size, lu_precision
):
    """Rodas5 linear benchmark with cached Diffrax validation on practical ensemble sizes."""
    system = moving_diffusion_spike_system
    params = _make_params_batch(ensemble_size, seed=42)
    solve = make_rodas5_linear(
        jac_fn=system["jac_fn"],
        lu_precision=lu_precision,
        batch_size=1,
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
            results_np, np.asarray(y_ref), rtol=_REFERENCE_RTOL, atol=3e-8
        )


# ---------------------------------------------------------------------------
# Nonlinear solver (ode_fn path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moving_diffusion_spike_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_rodas5_nonlinear(
    benchmark, moving_diffusion_spike_system, ensemble_size, lu_precision
):
    """Rodas5 nonlinear benchmark with cached Diffrax validation on practical ensemble sizes."""
    system = moving_diffusion_spike_system
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
            results_np, np.asarray(y_ref), rtol=_REFERENCE_RTOL, atol=3e-8
        )


# ---------------------------------------------------------------------------
# Diffrax Kvaerno5 (reference solver timing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moving_diffusion_spike_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id
)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_diffrax_kvaerno5(benchmark, moving_diffusion_spike_system, ensemble_size):
    """Diffrax Kvaerno5 benchmark with mass-conservation validation."""
    system = moving_diffusion_spike_system
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
    np.testing.assert_allclose(results_np.sum(axis=2), 1.0, atol=3e-6)
