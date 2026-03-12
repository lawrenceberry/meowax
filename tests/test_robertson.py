import json
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvers.rodas5 import solve as rodas5_solve
from solvers.rodas5 import solve_ensemble as rodas5_solve_ensemble
from solvers.scipy_bdf import solve as scipy_bdf_solve
from solvers.scipy_bdf import solve_ensemble as scipy_bdf_solve_ensemble

_JULIA_SCRIPT = "benchmarks/robertson_julia.jl"
_HAS_JULIA = shutil.which("julia") is not None


def _run_julia(n=2, timeout=300):
    """Run the Julia GPU ensemble Robertson benchmark and return parsed JSON result."""
    cmd = ["julia", _JULIA_SCRIPT, str(n)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Julia failed:\n{result.stderr}")
    return json.loads(result.stdout.strip())

_STANDARD_PARAMS = jnp.array([0.04, 1e4, 3e7])
_EXPECTED_FINAL = jnp.array([1.786592e-02, 7.274753e-08, 9.821340e-01])


@pytest.fixture
def params_batch(request):
    """Robertson parameter sets with ±10% uniform perturbation."""
    N = request.param
    rng = np.random.default_rng(42)
    return jnp.array(_STANDARD_PARAMS * (1.0 + 0.1 * (2 * rng.random((N, 3)) - 1)))


@jax.jit
def robertson(y, params=_STANDARD_PARAMS):
    """Robertson equation parameterized by rate constants (k1, k2, k3).

    Stiff ODE system (Appendix A.1.3, arXiv:2304.06835).
    Standard parameters: k1=0.04, k2=1e4, k3=3e7.
    """
    k1, k2, k3 = params
    y1, y2, y3 = y
    return jnp.array(
        [
            -k1 * y1 + k2 * y2 * y3,
            k1 * y1 - k2 * y2 * y3 - k3 * y2**2,
            k3 * y2**2,
        ]
    )


def test_scipy_bdf(benchmark):
    y = benchmark.pedantic(
        lambda: scipy_bdf_solve(
            robertson, y0=[1.0, 0.0, 0.0], t_span=(0.0, 1e5), first_step=1e-4
        ),
        warmup_rounds=1,
        rounds=1,
    )

    # Conservation: y1 + y2 + y3 = 1 (the system is conservative)
    total = y[0] + y[1] + y[2]
    np.testing.assert_allclose(total, 1.0, atol=1e-6)

    # Check final state against known values
    np.testing.assert_allclose(y[0, -1], 1.786592e-02, rtol=1e-4)
    np.testing.assert_allclose(y[1, -1], 7.274753e-08, rtol=1e-4)
    np.testing.assert_allclose(y[2, -1], 9.821340e-01, rtol=1e-4)


def test_rodas5(benchmark):
    y = benchmark.pedantic(
        jax.jit(
            lambda: rodas5_solve(
                robertson, y0=[1.0, 0.0, 0.0], t_span=(0.0, 1e5), first_step=1e-4
            )
        ),
        warmup_rounds=1,
        rounds=1,
    )

    # Conservation: y1 + y2 + y3 = 1 (the system is conservative)
    np.testing.assert_allclose(y.sum(), 1.0, atol=1e-6)

    # Check final state against known values
    np.testing.assert_allclose(y[0], 1.786592e-02, rtol=1e-4)
    np.testing.assert_allclose(y[1], 7.274753e-08, rtol=1e-4)
    np.testing.assert_allclose(y[2], 9.821340e-01, rtol=1e-4)


def test_scipy_bdf_ensemble(benchmark):
    params_batch = jnp.array(
        [
            [0.04, 1e4, 3e7],
            [0.04, 1e4, 3e7],
        ]
    )
    results = benchmark.pedantic(
        lambda: scipy_bdf_solve_ensemble(
            robertson,
            y0=[1.0, 0.0, 0.0],
            t_span=(0.0, 1e5),
            params_batch=params_batch,
            first_step=1e-4,
        ),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (2, 3)
    for i in range(2):
        np.testing.assert_allclose(results[i], _EXPECTED_FINAL, rtol=1e-4)


def test_rodas5_ensemble(benchmark):
    params_batch = jnp.array(
        [
            [0.04, 1e4, 3e7],
            [0.04, 1e4, 3e7],
        ]
    )
    results = benchmark.pedantic(
        lambda: rodas5_solve_ensemble(
            robertson,
            y0=[1.0, 0.0, 0.0],
            t_span=(0.0, 1e5),
            params_batch=params_batch,
            first_step=1e-4,
        ),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (2, 3)
    for i in range(2):
        np.testing.assert_allclose(results[i], _EXPECTED_FINAL, rtol=1e-4)


@pytest.mark.parametrize("params_batch", [100], indirect=True)
def test_scipy_bdf_ensemble_N(benchmark, params_batch):
    results = benchmark.pedantic(
        lambda: scipy_bdf_solve_ensemble(
            robertson,
            y0=[1.0, 0.0, 0.0],
            t_span=(0.0, 1e5),
            params_batch=params_batch,
            first_step=1e-4,
        ),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (params_batch.shape[0], 3)
    # Conservation should hold for every member
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("params_batch", [100], indirect=True)
def test_rodas5_ensemble_N(benchmark, params_batch):
    results = benchmark.pedantic(
        lambda: rodas5_solve_ensemble(
            robertson,
            y0=[1.0, 0.0, 0.0],
            t_span=(0.0, 1e5),
            params_batch=params_batch,
            first_step=1e-4,
        ),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (params_batch.shape[0], 3)
    # Conservation should hold for every member
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Julia / DiffEqGPU.jl GPU ensemble benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JULIA, reason="Julia not installed")
def test_julia_gpu_ensemble(benchmark):
    """Ensemble (N=2): Julia GPURosenbrock23 + EnsembleGPUKernel (CUDA, Float64)."""
    data = benchmark.pedantic(
        lambda: _run_julia(2),
        warmup_rounds=0,
        rounds=1,
    )
    benchmark.extra_info["julia_elapsed"] = data["elapsed_seconds"]

    assert data["converged"]
    conservations = np.array(data["conservations"])
    np.testing.assert_allclose(conservations, 1.0, atol=1e-6)


@pytest.mark.skipif(not _HAS_JULIA, reason="Julia not installed")
@pytest.mark.parametrize("N", [100])
def test_julia_gpu_ensemble_N(benchmark, N):
    """Ensemble (N=100): Julia GPURosenbrock23 + EnsembleGPUKernel (CUDA, Float64)."""
    data = benchmark.pedantic(
        lambda: _run_julia(N),
        warmup_rounds=0,
        rounds=1,
    )
    benchmark.extra_info["julia_elapsed"] = data["elapsed_seconds"]
    benchmark.extra_info["n_trajectories"] = N

    assert data["converged"]
    conservations = np.array(data["conservations"])
    np.testing.assert_allclose(conservations, 1.0, atol=1e-6)
