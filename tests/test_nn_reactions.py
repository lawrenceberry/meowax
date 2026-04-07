"""Tests for synthetic stiff nearest-neighbor mass-conserving ODE systems."""

import time

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
from jax import lax  # isort: skip  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
from jax.experimental import sparse as jsparse  # isort: skip  # noqa: E402
import numpy as np
import pytest
import scipy.sparse
from scipy.linalg import expm

import solvers.rodas5_v2 as rodas5_v2_mod
from solvers import _cudss_pallas
from solvers.rodas5 import solve_ensemble as rodas5_solve_ensemble
from solvers.rodas5_custom_kernel_v2 import make_solver as make_rodas5_v2_solver
from solvers.rodas5_custom_kernel_v3 import make_solver as make_rodas5_v3_solver
from solvers.rodas5_custom_kernel_v4 import make_solver as make_rodas5_v4_solver
from solvers.rodas5_custom_kernel_v5 import make_solver as make_rodas5_v5_solver
from solvers.rodas5_custom_kernel_v6 import make_solver as make_rodas5_v6_solver
from solvers.rodas5_custom_kernel_v7_tc import make_solver as make_rodas5_v7_tc_solver
from solvers.rodas5_v2 import SparseJacobianCSR
from solvers.rodas5_v2 import make_solver as make_rodas5_v2_dense_solver
from solvers.rodas5_v2 import solve_ensemble as rodas5_v2_solve_ensemble

_T_SPAN = (0.0, 1.0)
_V6_SAVE_TIMES = jnp.array(_T_SPAN, dtype=jnp.float64)
_V6_MULTI_SAVE_TIMES = jnp.array((0.0, 0.125, 0.25, 0.5, 1.0), dtype=jnp.float64)
_SYSTEM_DIMS = [30, 50, 70]
_ENSEMBLE_SIZES = [2, 100, 1000, 10000, 100_000]
_LINEAR_SOLVER_PRECISIONS = ("fp64", "fp32")
_V6_LINEAR_SOLVERS = _LINEAR_SOLVER_PRECISIONS


def _has_cuda_backend():
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def _has_fused_cudss_runtime():
    return _has_cuda_backend() and rodas5_v2_mod._cudss_ffi.has_fused_step()


def _time_linear_scale(t):
    """Scalar multiplier for the time-dependent linear systems."""
    return 1.0 + 0.2 * t


def _time_linear_scale_integral(t0, tf):
    """Integral of ``_time_linear_scale`` over ``[t0, tf]``."""
    return (tf - t0) + 0.1 * (tf**2 - t0**2)


def _build_linear_chain_matrix(n_vars, kf, kb):
    """Build matrix M for the D-dimensional linear chain dy/dt = M y."""
    M = np.zeros((n_vars, n_vars), dtype=np.float64)
    M[0, 0] = -kf[0]
    M[0, 1] = kb[0]

    for i in range(1, n_vars - 1):
        M[i, i - 1] = kf[i - 1]
        M[i, i] = -(kb[i - 1] + kf[i])
        M[i, i + 1] = kb[i]

    M[n_vars - 1, n_vars - 2] = kf[n_vars - 2]
    M[n_vars - 1, n_vars - 1] = -kb[n_vars - 2]
    return M


def _coo_to_bcoo(matrix):
    """Convert a SciPy COO/CSR/CSC matrix to a JAX BCOO matrix."""
    coo = matrix.tocoo()
    indices = jnp.asarray(np.stack([coo.row, coo.col], axis=1), dtype=jnp.int32)
    data = jnp.asarray(coo.data, dtype=jnp.float64)
    return jsparse.BCOO((data, indices), shape=coo.shape)


def _csr_descriptor_from_scipy(matrix):
    """Convert a SciPy sparse matrix to the static SparseJacobianCSR descriptor."""
    csr = matrix.tocsr()
    return SparseJacobianCSR(
        indptr=jnp.asarray(csr.indptr, dtype=jnp.int32),
        indices=jnp.asarray(csr.indices, dtype=jnp.int32),
        values=jnp.asarray(csr.data, dtype=jnp.float64),
        nnz=jnp.asarray(csr.nnz, dtype=jnp.int32),
        shape=csr.shape,
    )


def _align_csr_values_to_union(union_matrix, source_matrix):
    """Align source CSR values onto a fixed union CSR sparsity pattern."""
    union = union_matrix.tocsr()
    source = source_matrix.tocsr()
    values = np.zeros(union.nnz, dtype=np.float64)
    for row in range(union.shape[0]):
        union_start, union_end = union.indptr[row], union.indptr[row + 1]
        source_start, source_end = source.indptr[row], source.indptr[row + 1]
        source_cols = source.indices[source_start:source_end]
        source_vals = source.data[source_start:source_end]
        source_map = {
            int(col): float(val)
            for col, val in zip(source_cols, source_vals, strict=False)
        }
        for offset in range(union_start, union_end):
            values[offset] = source_map.get(int(union.indices[offset]), 0.0)
    return values


def _build_generator_matrix(n_vars, directed_edges):
    """Build a mass-conserving generator matrix from directed edges i -> j."""
    M = np.zeros((n_vars, n_vars), dtype=np.float64)
    for src, dst, rate in directed_edges:
        M[dst, src] += rate
        M[src, src] -= rate
    return M


def _make_nn_reaction_system(n_vars):
    """Construct a configurable D-dimensional nearest-neighbor reaction system."""
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    edge_count = n_vars - 1
    kf = tuple(10.0 ** (-2.0 + 8.0 * i / (edge_count - 1)) for i in range(edge_count))
    kb = tuple(10.0 ** (6.0 - 8.0 * i / (edge_count - 1)) for i in range(edge_count))
    M_np = _build_linear_chain_matrix(n_vars, kf, kb)
    M = jnp.array(M_np, dtype=jnp.float64)
    M_sparse = scipy.sparse.coo_matrix(M_np)
    M_sparse_csr = _csr_descriptor_from_scipy(M_sparse)
    y0 = jnp.array([1.0] + [0.0] * (n_vars - 1), dtype=jnp.float64)
    y0_np = np.asarray(y0, dtype=np.float64)

    def ode(y, p):
        """D-dimensional stiff linear chain with conserved total mass."""
        s = p[0]
        dy = [None] * n_vars
        dy[0] = -s * kf[0] * y[0] + s * kb[0] * y[1]

        for i in range(1, n_vars - 1):
            dy[i] = (
                s * kf[i - 1] * y[i - 1]
                - s * (kb[i - 1] + kf[i]) * y[i]
                + s * kb[i] * y[i + 1]
            )

        dy[n_vars - 1] = (
            s * kf[n_vars - 2] * y[n_vars - 2] - s * kb[n_vars - 2] * y[n_vars - 1]
        )
        return tuple(dy)

    @jax.jit
    def array(y, p):
        """Array-returning adapter for the non-Pallas reference solver."""
        return jnp.array(ode(y, p))

    def jac(y, p):
        """Explicit Jacobian for the D-dimensional stiff linear chain."""
        del y
        s = p[0]
        return tuple(
            tuple(s * M_np[i, j] for j in range(n_vars)) for i in range(n_vars)
        )

    def time_jac(t):
        """Time-dependent matrix callback for the v6 solver."""
        s = _time_linear_scale(t)
        return tuple(
            tuple(s * M_np[i, j] for j in range(n_vars)) for i in range(n_vars)
        )

    def jac_array(p):
        """Jacobian as a JAX array, function of params only (linear systems)."""
        s = p[0]
        return s * M

    def jac_sparse_array(p):
        """Jacobian as a static CSR descriptor, function of params only."""
        s = p[0]
        return SparseJacobianCSR(
            indptr=M_sparse_csr.indptr,
            indices=M_sparse_csr.indices,
            values=s * M_sparse_csr.values,
            nnz=M_sparse_csr.nnz,
            shape=M_sparse_csr.shape,
        )

    def time_exact(y_init, t_span):
        """Closed-form solution for M(t) = a(t) * M with commuting matrices."""
        scale_int = _time_linear_scale_integral(*t_span)
        return jnp.array(expm(scale_int * M_np) @ np.asarray(y_init), dtype=jnp.float64)

    return {
        "n_vars": n_vars,
        "matrix": M,
        "matrix_np": M_np,
        "ode": ode,
        "array": array,
        "jac": jac,
        "time_jac": time_jac,
        "jac_array": jac_array,
        "jac_sparse_array": jac_sparse_array,
        "time_exact": time_exact,
        "y0": y0,
        "y0_np": y0_np,
    }


def _dim_id(n_vars):
    return f"{n_vars}d"


def _linear_solver_id(linear_solver):
    return linear_solver


def _make_params_batch(size, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def _broadcast_y0(y0, size):
    return jnp.broadcast_to(y0, (size, y0.shape[0]))


def _time_exact_trajectory(system, save_times):
    save_times_np = np.asarray(save_times, dtype=np.float64)
    t0 = float(save_times_np[0])
    return np.stack(
        [
            np.asarray(
                system["time_exact"](system["y0"], (t0, float(tf))), dtype=np.float64
            )
            for tf in save_times_np
        ],
        axis=0,
    )


def _reference_trajectory(system, params_batch, save_times):
    save_times_np = np.asarray(save_times, dtype=np.float64)
    traj = []
    for i, tf in enumerate(save_times_np):
        if i == 0:
            traj.append(
                np.broadcast_to(
                    np.asarray(system["y0"]), (params_batch.shape[0], system["n_vars"])
                )
            )
            continue
        y_ref = rodas5_solve_ensemble(
            system["array"],
            y0=system["y0"],
            t_span=(float(save_times_np[0]), float(tf)),
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready()
        traj.append(np.asarray(y_ref))
    return np.stack(traj, axis=1)


def _make_nonbanded_sparse_linear_system():
    """Build a small non-banded sparse linear system for jac_sparse_fn tests."""
    n_vars = 8
    edges = (
        (0, 4, 1.8),
        (4, 0, 0.9),
        (1, 6, 1.2),
        (6, 2, 0.7),
        (2, 7, 0.6),
        (7, 3, 0.5),
        (3, 5, 0.4),
        (5, 1, 0.8),
        (0, 7, 0.3),
        (6, 0, 0.25),
    )
    M_np = _build_generator_matrix(n_vars, edges)
    M = jnp.asarray(M_np, dtype=jnp.float64)
    M_sparse_csr = _csr_descriptor_from_scipy(scipy.sparse.coo_matrix(M_np))
    y0 = jnp.asarray([1.0] + [0.0] * (n_vars - 1), dtype=jnp.float64)

    @jax.jit
    def array(y, p):
        return (p[0] * M) @ y

    def jac_array(p):
        return p[0] * M

    def jac_sparse_array(p):
        return SparseJacobianCSR(
            indptr=M_sparse_csr.indptr,
            indices=M_sparse_csr.indices,
            values=p[0] * M_sparse_csr.values,
            nnz=M_sparse_csr.nnz,
            shape=M_sparse_csr.shape,
        )

    return {
        "n_vars": n_vars,
        "y0": y0,
        "array": array,
        "jac_array": jac_array,
        "jac_sparse_array": jac_sparse_array,
    }


def _make_switching_sparse_system():
    """Build a sparse system whose sparsity pattern changes with state."""
    n_vars = 5
    edges_a = (
        (0, 1, 4.0),
        (1, 2, 2.5),
        (2, 4, 1.5),
        (4, 3, 0.8),
        (3, 0, 0.6),
    )
    edges_b = (
        (0, 2, 2.2),
        (2, 1, 1.3),
        (1, 4, 1.1),
        (4, 0, 0.9),
        (3, 2, 0.7),
        (0, 3, 0.5),
    )
    M_a_np = _build_generator_matrix(n_vars, edges_a)
    M_b_np = _build_generator_matrix(n_vars, edges_b)
    M_a = jnp.asarray(M_a_np, dtype=jnp.float64)
    M_b = jnp.asarray(M_b_np, dtype=jnp.float64)
    M_a_csr = scipy.sparse.csr_matrix(M_a_np)
    M_b_csr = scipy.sparse.csr_matrix(M_b_np)
    union_csr = ((M_a_csr != 0.0) + (M_b_csr != 0.0)).astype(np.float64).tocsr()
    union_csr.data[:] = 1.0
    union_desc = _csr_descriptor_from_scipy(union_csr)
    union_values_a = jnp.asarray(
        _align_csr_values_to_union(union_csr, M_a_csr), dtype=jnp.float64
    )
    union_values_b = jnp.asarray(
        _align_csr_values_to_union(union_csr, M_b_csr), dtype=jnp.float64
    )
    y0 = jnp.asarray([1.0] + [0.0] * (n_vars - 1), dtype=jnp.float64)

    def _select_dense(y):
        return lax.cond(y[0] > 0.55, lambda _: M_a, lambda _: M_b, operand=None)

    @jax.jit
    def array(y, p):
        scale = p[0]
        return scale * (_select_dense(y) @ y)

    def jac_sparse(y, p):
        scale = p[0]
        use_a = y[0] > 0.55
        values = jnp.where(use_a, union_values_a, union_values_b)
        return SparseJacobianCSR(
            indptr=union_desc.indptr,
            indices=union_desc.indices,
            values=scale * values,
            nnz=union_desc.nnz,
            shape=union_desc.shape,
        )

    return {
        "n_vars": n_vars,
        "y0": y0,
        "array": array,
        "jac_sparse": jac_sparse,
    }


def _make_fixed_structure_dynamic_sparse_system():
    """Build a dynamic-values sparse system with a fixed CSR structure."""
    n_vars = 5
    edges_a = (
        (0, 1, 4.0),
        (1, 2, 2.5),
        (2, 4, 1.5),
        (4, 3, 0.8),
        (3, 0, 0.6),
        (1, 4, 0.7),
    )
    edges_b = (
        (0, 1, 1.9),
        (1, 2, 1.7),
        (2, 4, 0.9),
        (4, 3, 1.3),
        (3, 0, 1.1),
        (1, 4, 1.4),
    )
    M_a_np = _build_generator_matrix(n_vars, edges_a)
    M_b_np = _build_generator_matrix(n_vars, edges_b)
    M_a = jnp.asarray(M_a_np, dtype=jnp.float64)
    M_b = jnp.asarray(M_b_np, dtype=jnp.float64)
    desc = _csr_descriptor_from_scipy(scipy.sparse.coo_matrix(M_a_np))
    values_a = desc.values
    values_b = _csr_descriptor_from_scipy(scipy.sparse.coo_matrix(M_b_np)).values
    y0 = jnp.asarray([1.0] + [0.0] * (n_vars - 1), dtype=jnp.float64)

    def _select_dense(y):
        return lax.cond(y[0] > 0.55, lambda _: M_a, lambda _: M_b, operand=None)

    @jax.jit
    def array(y, p):
        return p[0] * (_select_dense(y) @ y)

    def jac_sparse(y, p):
        values = jnp.where(y[0] > 0.55, values_a, values_b)
        return SparseJacobianCSR(
            indptr=desc.indptr,
            indices=desc.indices,
            values=p[0] * values,
            nnz=desc.nnz,
            shape=desc.shape,
        )

    return {
        "n_vars": n_vars,
        "y0": y0,
        "array": array,
        "jac_sparse": jac_sparse,
    }


@pytest.fixture
def nn_reaction_system(request):
    """Configurable nearest-neighbor reaction system parameterized by dimension."""
    return _make_nn_reaction_system(request.param)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v2_matches_rodas5_reference(nn_reaction_system):
    """Validate Pallas v2 solver on the nearest-neighbor reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=0)

    solve_v2 = make_rodas5_v2_solver(system["ode"])
    y_v2 = solve_v2(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v2.shape == (N, system["n_vars"])
    assert y_ref.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v2.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v2, y_ref, rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v3_matches_rodas5_reference_for_linear_matrix(nn_reaction_system):
    """Validate matrix-specialized Pallas v3 solver on the reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)

    solve_v3 = make_rodas5_v3_solver(system["matrix"])
    y_v3 = solve_v3(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    params_batch = jnp.ones((N, 1), dtype=jnp.float64)
    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v3.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v3.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v3, y_ref, rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 vmap ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: rodas5_solve_ensemble(
            system["array"],
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v2_pallas_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v2 Pallas custom kernel ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    solve_v2 = make_rodas5_v2_solver(system["ode"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    params_batch = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: solve_v2(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v3_pallas_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v3 (matrix) Pallas custom kernel ensemble benchmark."""
    system = nn_reaction_system
    solve_v3 = make_rodas5_v3_solver(system["matrix"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    results = benchmark.pedantic(
        lambda: solve_v3(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v3_pallas_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v3 (matrix) Pallas solver."""
    N = 1234
    system = nn_reaction_system

    solve_v3 = make_rodas5_v3_solver(system["matrix"])
    y0_batch = _broadcast_y0(system["y0"], N)

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v3(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v3(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
        np.testing.assert_allclose(y_first.sum(axis=1), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v2_pallas_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v2 (JVP) Pallas solver."""
    N = 1234
    system = nn_reaction_system

    solve_v2 = make_rodas5_v2_solver(system["ode"])
    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=123)

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v2(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v2(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
        np.testing.assert_allclose(y_first.sum(axis=1), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v4_matches_rodas5_reference_for_linear_matrix(nn_reaction_system):
    """Validate Numba CUDA v4 solver on the reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = np.broadcast_to(system["y0_np"], (N, system["n_vars"])).copy()

    solve_v4 = make_rodas5_v4_solver(np.asarray(system["matrix"]))
    y_v4 = solve_v4(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )

    params_batch = jnp.ones((N, 1), dtype=jnp.float64)
    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v4.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v4.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v4, np.asarray(y_ref), rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v4_numba_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v4 Numba CUDA ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    solve_v4 = make_rodas5_v4_solver(np.asarray(system["matrix"]))
    y0_np = np.broadcast_to(system["y0_np"], (ensemble_size, system["n_vars"])).copy()
    results = benchmark.pedantic(
        lambda: solve_v4(
            y0_batch=y0_np,
            t_span=_T_SPAN,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v4_numba_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v4 Numba CUDA solver."""
    N = 1234
    system = nn_reaction_system

    solve_v4 = make_rodas5_v4_solver(np.asarray(system["matrix"]))
    y0_batch = np.broadcast_to(system["y0_np"], (N, system["n_vars"])).copy()

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v4(**kwargs)
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v4(**kwargs)
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
        np.testing.assert_allclose(y_first.sum(axis=1), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v5_matches_rodas5_reference(nn_reaction_system):
    """Validate v5 solver (explicit Jacobian) on the reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=0)

    solve_v5 = make_rodas5_v5_solver(system["jac"])
    y_v5 = solve_v5(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v5.shape == (N, system["n_vars"])
    np.testing.assert_allclose(y_v5.sum(axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v5, y_ref, rtol=3e-5, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v5_pallas_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v5 Pallas ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    solve_v5 = make_rodas5_v5_solver(system["jac"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    params_batch = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: solve_v5(
            y0_batch=y0_batch,
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=1), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v5_pallas_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for v5 (explicit Jacobian)."""
    N = 1234
    system = nn_reaction_system

    solve_v5 = make_rodas5_v5_solver(system["jac"])
    y0_batch = _broadcast_y0(system["y0"], N)
    params_batch = _make_params_batch(N, seed=123)

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v5(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v5(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, system["n_vars"])
        assert y_second.shape == (N, system["n_vars"])
        np.testing.assert_allclose(y_first.sum(axis=1), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_matches_closed_form_time_dependent_reference(
    nn_reaction_system, linear_solver
):
    """Validate v6 solver on a time-dependent nearest-neighbor reaction system."""
    N = 256
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)

    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)
    y_v6 = solve_v6(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_v6_np = np.asarray(y_v6)

    y_exact_traj = _time_exact_trajectory(system, _V6_SAVE_TIMES)
    y_exact_batch = np.broadcast_to(
        y_exact_traj, (N, y_exact_traj.shape[0], y_exact_traj.shape[1])
    )

    assert y_v6.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(y_v6_np[:, 0, :], np.asarray(y0_batch), atol=0.0)
    np.testing.assert_allclose(y_v6_np.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v6_np, y_exact_batch, rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_saves_multiple_times(nn_reaction_system, linear_solver):
    """Validate v6 solver on several requested save times."""
    N = 64
    system = nn_reaction_system

    y0_batch = _broadcast_y0(system["y0"], N)
    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)

    y_v6 = solve_v6(
        y0_batch=y0_batch,
        t_span=_V6_MULTI_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_v6_np = np.asarray(y_v6)

    y_exact_traj = _time_exact_trajectory(system, _V6_MULTI_SAVE_TIMES)
    y_exact_batch = np.broadcast_to(
        y_exact_traj, (N, y_exact_traj.shape[0], y_exact_traj.shape[1])
    )

    assert y_v6.shape == (N, len(_V6_MULTI_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(y_v6_np[:, 0, :], np.asarray(y0_batch), atol=0.0)
    np.testing.assert_allclose(y_v6_np.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v6_np, y_exact_batch, rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_pallas_ensemble_N(
    benchmark, nn_reaction_system, ensemble_size, linear_solver
):
    """Rodas5 v6 Pallas ensemble benchmark on the time-dependent system."""
    system = nn_reaction_system
    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    results = benchmark.pedantic(
        lambda: solve_v6(
            y0_batch=y0_batch,
            t_span=_V6_SAVE_TIMES,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(np.asarray(results).sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("linear_solver", _V6_LINEAR_SOLVERS, ids=_linear_solver_id)
def test_rodas5_v6_pallas_compile_time(benchmark, nn_reaction_system, linear_solver):
    """Measure approximate compile time for v6 (time-dependent Jacobian)."""
    N = 1234
    system = nn_reaction_system

    solve_v6 = make_rodas5_v6_solver(system["time_jac"], linear_solver=linear_solver)
    y0_batch = _broadcast_y0(system["y0"], N)

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v6(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v6(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        assert y_second.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        np.testing.assert_allclose(np.asarray(y_first).sum(axis=2), 1.0, atol=3e-6)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


@pytest.mark.parametrize(
    ("t_span", "match"),
    [
        (jnp.array([[0.0, 1.0]], dtype=jnp.float64), "1D array"),
        (jnp.array([0.0], dtype=jnp.float64), "at least 2"),
        (jnp.array([0.0, 0.5, 0.5], dtype=jnp.float64), "strictly increasing"),
    ],
)
def test_rodas5_v6_rejects_invalid_save_times(t_span, match):
    """Validate v6 save-time input checks."""
    system = _make_nn_reaction_system(30)
    solve_v6 = make_rodas5_v6_solver(system["time_jac"])
    y0_batch = _broadcast_y0(system["y0"], 4)

    with pytest.raises(ValueError, match=match):
        solve_v6(
            y0_batch=y0_batch,
            t_span=t_span,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v6_fp32_matches_fp64_baseline(nn_reaction_system):
    """Validate the FP32 LU v6 path against the FP64 baseline."""
    N = 256
    system = nn_reaction_system
    y0_batch = _broadcast_y0(system["y0"], N)

    solve_fp64 = make_rodas5_v6_solver(system["time_jac"], linear_solver="fp64")
    solve_fp32 = make_rodas5_v6_solver(system["time_jac"], linear_solver="fp32")

    y_fp64 = solve_fp64(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_fp32 = solve_fp32(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    np.testing.assert_allclose(
        np.asarray(y_fp32), np.asarray(y_fp64), rtol=2e-4, atol=3e-8
    )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v7_tc_matches_closed_form_time_dependent_reference(nn_reaction_system):
    """Validate the experimental blocked-LU Tensor Core-style solver on 50D."""
    N = 128
    system = nn_reaction_system
    y0_batch = _broadcast_y0(system["y0"], N)

    solve_v7 = make_rodas5_v7_tc_solver(system["time_jac"])
    y_v7 = solve_v7(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_v7_np = np.asarray(y_v7)

    y_exact_traj = _time_exact_trajectory(system, _V6_SAVE_TIMES)
    y_exact_batch = np.broadcast_to(
        y_exact_traj, (N, y_exact_traj.shape[0], y_exact_traj.shape[1])
    )

    assert y_v7.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(y_v7_np[:, 0, :], np.asarray(y0_batch), atol=0.0)
    np.testing.assert_allclose(y_v7_np.sum(axis=2), 1.0, atol=3e-5)
    np.testing.assert_allclose(y_v7_np, y_exact_batch, rtol=8e-4, atol=6e-8)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v7_tc_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Benchmark the experimental blocked-LU Tensor Core-style solver on 50D."""
    system = nn_reaction_system
    solve_v7 = make_rodas5_v7_tc_solver(system["time_jac"])
    y0_batch = _broadcast_y0(system["y0"], ensemble_size)
    results = benchmark.pedantic(
        lambda: solve_v7(
            y0_batch=y0_batch,
            t_span=_V6_SAVE_TIMES,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_V6_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(np.asarray(results).sum(axis=2), 1.0, atol=3e-5)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v7_tc_compile_time(benchmark, nn_reaction_system):
    """Measure approximate compile time for the experimental blocked-LU solver."""
    N = 1234
    system = nn_reaction_system

    solve_v7 = make_rodas5_v7_tc_solver(system["time_jac"])
    y0_batch = _broadcast_y0(system["y0"], N)

    kwargs = dict(
        y0_batch=y0_batch,
        t_span=_V6_SAVE_TIMES,
        first_step=7e-7,
        rtol=1.23e-6,
        atol=4.56e-8,
        max_steps=654321,
    )

    def measure_compile_estimate():
        t0 = time.perf_counter()
        y_first = solve_v7(**kwargs).block_until_ready()
        first_call_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_second = solve_v7(**kwargs).block_until_ready()
        second_call_s = time.perf_counter() - t1

        assert y_first.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        assert y_second.shape == (N, len(_V6_SAVE_TIMES), system["n_vars"])
        np.testing.assert_allclose(np.asarray(y_first).sum(axis=2), 1.0, atol=3e-5)
        return max(first_call_s - second_call_s, 0.0)

    compile_estimate_s = benchmark.pedantic(
        measure_compile_estimate,
        warmup_rounds=0,
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["compile_estimate_s"] = float(compile_estimate_s)
    assert compile_estimate_s >= 0.0


# ---------------------------------------------------------------------------
# Rodas5 v2 — single-loop batched ensemble
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("batch_size", [1, 7, None], ids=["bs1", "bs7", "bsN"])
def test_rodas5_v2_matches_reference(nn_reaction_system, batch_size):
    """Validate single-loop v2 solver against vmap reference."""
    N = 100
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    y_v2 = rodas5_v2_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
        batch_size=batch_size,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_v2.shape == (N, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(
        y_v2[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0
    )
    np.testing.assert_allclose(y_v2.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_v2[:, -1, :], y_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v2_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v2 single-loop ensemble benchmark on the reaction system."""
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)
    results = benchmark.pedantic(
        lambda: rodas5_v2_solve_ensemble(
            system["array"],
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


# ---------------------------------------------------------------------------
# Rodas5 v2 — jac_fn path (linear systems, no AD)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v2_jac_fn_matches_reference(nn_reaction_system):
    """Validate jac_fn (linear) path against the f (AD) path."""
    N = 256
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    y_jac = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_fn=system["jac_array"],
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    y_ref = rodas5_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_jac.shape == (N, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(
        y_jac[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0
    )
    np.testing.assert_allclose(y_jac.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(y_jac[:, -1, :], y_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
def test_rodas5_v2_jac_sparse_fn_matches_dense_jac_fn(nn_reaction_system):
    """Validate jac_sparse_fn against the dense jac_fn path."""
    N = 128
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=1)

    y_sparse = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_sparse_fn=system["jac_sparse_array"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_dense = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_fn=system["jac_array"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_sparse.shape == y_dense.shape
    np.testing.assert_allclose(
        y_sparse[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0
    )
    np.testing.assert_allclose(y_sparse.sum(axis=2), 1.0, atol=5e-6)
    np.testing.assert_allclose(
        np.asarray(y_sparse), np.asarray(y_dense), rtol=2e-4, atol=3e-8
    )


def test_rodas5_v2_jac_sparse_fn_nonbanded_matches_dense_reference():
    """Validate jac_sparse_fn on a non-banded sparse system."""
    system = _make_nonbanded_sparse_linear_system()
    N = 64
    params_batch = _make_params_batch(N, seed=3)

    y_sparse = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_sparse_fn=system["jac_sparse_array"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_dense = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_fn=system["jac_array"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_sparse.shape == (N, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(
        np.asarray(y_sparse), np.asarray(y_dense), rtol=4e-2, atol=4e-4
    )


def test_rodas5_v2_jac_sparse_fn_changing_pattern_matches_explicit_f():
    """Validate a jac_sparse_fn whose sparsity pattern changes with state."""
    system = _make_switching_sparse_system()
    N = 48
    params_batch = _make_params_batch(N, seed=4)

    y_sparse = rodas5_v2_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        jac_sparse_fn=system["jac_sparse"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_ref = rodas5_v2_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_sparse.shape == (N, len(_V6_MULTI_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(
        np.asarray(y_sparse), np.asarray(y_ref), rtol=8e-3, atol=1e-6
    )


def test_rodas5_v2_jac_sparse_fn_static_callback_batches_host_preprocessing():
    """Static JAX sparse callbacks should not execute once per ensemble member."""
    system = _make_nn_reaction_system(30)
    params_batch = _make_params_batch(64, seed=11)
    base = system["jac_sparse_array"](jnp.asarray([1.0], dtype=jnp.float64))
    callback_calls = {"count": 0}

    def counted_jac_sparse(p):
        callback_calls["count"] += 1
        return SparseJacobianCSR(
            indptr=base.indptr,
            indices=base.indices,
            values=p[0] * base.values,
            nnz=base.nnz,
            shape=base.shape,
        )

    solve = make_rodas5_v2_dense_solver(
        jac_sparse_fn=counted_jac_sparse,
        linear_solver_precision="fp32",
    )
    y_hist = solve(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    assert y_hist.shape == (64, len(_T_SPAN), system["n_vars"])
    assert callback_calls["count"] < 8


def test_cudss_pallas_batched_csr_matvec_matches_reference():
    """The fixed-structure Pallas CSR helper should match the reference matvec."""
    system = _make_nn_reaction_system(30)
    params_batch = _make_params_batch(32, seed=21)
    sparse_items = [
        system["jac_sparse_array"](np.asarray(params))
        for params in np.asarray(params_batch)
    ]
    sparse_batch = rodas5_v2_mod._stack_sparse_batch(sparse_items)
    vec = jnp.broadcast_to(system["y0"], (params_batch.shape[0], system["n_vars"]))

    y_pallas = _cudss_pallas.batched_csr_matvec(
        sparse_batch.indptr,
        sparse_batch.indices,
        sparse_batch.values,
        sparse_batch.nnz,
        vec,
    )
    y_ref = rodas5_v2_mod._csr_matvec_batch(
        sparse_batch,
        vec,
        out_dtype=sparse_batch.values.dtype,
    )

    np.testing.assert_allclose(
        np.asarray(y_pallas), np.asarray(y_ref), rtol=1e-7, atol=1e-9
    )


@pytest.mark.skipif(
    not _has_fused_cudss_runtime(),
    reason="Fused cuDSS step requires a CUDA backend and native fused-step support",
)
@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v2_fused_cudss_static_matches_dense_jac_fn(nn_reaction_system):
    """Validate the fused static cuDSS step against the dense jac_fn path."""
    system = nn_reaction_system
    params_batch = _make_params_batch(96, seed=12)
    solve_sparse = make_rodas5_v2_dense_solver(
        jac_sparse_fn=system["jac_sparse_array"],
        linear_solver_precision="fp32",
    )
    solve_dense = make_rodas5_v2_dense_solver(
        jac_fn=system["jac_array"],
        linear_solver_precision="fp32",
    )

    y_sparse = solve_sparse(
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_dense = solve_dense(
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    counters = solve_sparse.get_cudss_counters(system["n_vars"], params_batch.shape[0])
    assert counters is not None and counters["solve"] > 0
    assert counters["device_prepare_fast_path_count"] > 0
    assert counters["host_prepare_slow_path_count"] == 0
    np.testing.assert_allclose(
        np.asarray(y_sparse), np.asarray(y_dense), rtol=2e-4, atol=3e-8
    )


@pytest.mark.skipif(
    not _has_fused_cudss_runtime(),
    reason="Fused cuDSS step requires a CUDA backend and native fused-step support",
)
def test_rodas5_v2_fused_cudss_dynamic_values_matches_explicit_f():
    """Validate fused cuDSS on a fixed-structure dynamic-values sparse system."""
    system = _make_fixed_structure_dynamic_sparse_system()
    params_batch = _make_params_batch(48, seed=13)
    solve_sparse = make_rodas5_v2_dense_solver(
        jac_sparse_fn=system["jac_sparse"],
        linear_solver_precision="fp32",
    )

    y_sparse = solve_sparse(
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_ref = rodas5_v2_solve_ensemble(
        system["array"],
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    counters = solve_sparse.get_cudss_counters(system["n_vars"], params_batch.shape[0])
    assert counters is not None and counters["solve"] > 0
    assert counters["device_prepare_fast_path_count"] > 0
    assert counters["host_prepare_slow_path_count"] == 0
    np.testing.assert_allclose(
        np.asarray(y_sparse), np.asarray(y_ref), rtol=8e-3, atol=1e-6
    )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v2_fp32_linear_solver_matches_fp64_baseline(nn_reaction_system):
    """Validate FP32 LU solves against the FP64 baseline on the jac_fn path."""
    N = 256
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    common_kwargs = dict(
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_fn=system["jac_array"],
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    )

    y_fp64 = rodas5_v2_solve_ensemble(
        None,
        linear_solver_precision="fp64",
        **common_kwargs,
    ).block_until_ready()
    y_fp32 = rodas5_v2_solve_ensemble(
        None,
        linear_solver_precision="fp32",
        **common_kwargs,
    ).block_until_ready()

    np.testing.assert_allclose(
        np.asarray(y_fp32), np.asarray(y_fp64), rtol=2e-4, atol=3e-8
    )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v2_jac_sparse_fn_dense_fallback_matches_sparse_backend(
    monkeypatch, nn_reaction_system
):
    """Validate the explicit dense fallback when cuDSS is unavailable."""
    N = 64
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=5)

    y_sparse = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_sparse_fn=system["jac_sparse_array"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    monkeypatch.setattr(
        rodas5_v2_mod._cudss_ffi, "create_context", lambda *args, **kwargs: None
    )
    assert (
        rodas5_v2_mod._resolve_sparse_backend(None)
        == rodas5_v2_mod._SPARSE_BACKEND_DENSE_FALLBACK
    )

    y_fallback = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_T_SPAN,
        params_batch=params_batch,
        jac_sparse_fn=system["jac_sparse_array"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()

    np.testing.assert_allclose(
        np.asarray(y_fallback), np.asarray(y_sparse), rtol=2e-4, atol=3e-8
    )


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
def test_rodas5_v2_jac_fn_saves_multiple_times(nn_reaction_system):
    """Validate multiple save times on the jac_fn path."""
    N = 64
    system = nn_reaction_system
    params_batch = _make_params_batch(N, seed=0)

    y_hist = rodas5_v2_solve_ensemble(
        None,
        y0=system["y0"],
        t_span=_V6_MULTI_SAVE_TIMES,
        params_batch=params_batch,
        jac_fn=system["jac_array"],
        linear_solver_precision="fp32",
        first_step=1e-6,
        rtol=1e-6,
        atol=1e-8,
    ).block_until_ready()
    y_ref = _reference_trajectory(system, params_batch, _V6_MULTI_SAVE_TIMES)

    assert y_hist.shape == (N, len(_V6_MULTI_SAVE_TIMES), system["n_vars"])
    np.testing.assert_allclose(
        y_hist[:, 0, :], np.asarray(_broadcast_y0(system["y0"], N)), atol=0.0
    )
    np.testing.assert_allclose(y_hist.sum(axis=2), 1.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(y_hist), y_ref, rtol=2e-4, atol=3e-8)


@pytest.mark.parametrize("nn_reaction_system", _SYSTEM_DIMS, indirect=True, ids=_dim_id)
@pytest.mark.parametrize("ensemble_size", _ENSEMBLE_SIZES)
def test_rodas5_v2_jac_fn_ensemble_N(benchmark, nn_reaction_system, ensemble_size):
    """Rodas5 v2 jac_fn (linear, no AD) ensemble benchmark."""
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)
    solve_v2 = make_rodas5_v2_dense_solver(
        jac_fn=system["jac_array"],
        linear_solver_precision="fp32",
    )
    results = benchmark.pedantic(
        lambda: solve_v2(
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "linear_solver_precision", _LINEAR_SOLVER_PRECISIONS, ids=_linear_solver_id
)
def test_rodas5_v2_jac_fn_linear_solver_precision_timing(
    benchmark, nn_reaction_system, linear_solver_precision
):
    """Benchmark the jac_fn path with FP64 vs FP32 LU precision."""
    ensemble_size = 10_000
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)

    results = benchmark.pedantic(
        lambda: rodas5_v2_solve_ensemble(
            None,
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            jac_fn=system["jac_array"],
            linear_solver_precision=linear_solver_precision,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    benchmark.extra_info["backend"] = jax.default_backend()

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=3e-6)


@pytest.mark.parametrize("nn_reaction_system", [50], indirect=True, ids=_dim_id)
@pytest.mark.parametrize(
    "solver_mode",
    ("dense_jac_fn", "sparse_cudss", "sparse_dense_fallback"),
)
def test_rodas5_v2_jac_sparse_fn_timing(
    benchmark, monkeypatch, nn_reaction_system, solver_mode
):
    """Benchmark dense jac_fn, sparse cuDSS, and explicit dense fallback."""
    ensemble_size = 1000
    system = nn_reaction_system
    params_batch = _make_params_batch(ensemble_size, seed=42)

    if solver_mode == "dense_jac_fn":
        solve = make_rodas5_v2_dense_solver(
            jac_fn=system["jac_array"],
            linear_solver_precision="fp32",
        )
        sparse_backend = "n/a"
    else:
        if solver_mode == "sparse_dense_fallback":
            monkeypatch.setattr(
                rodas5_v2_mod._cudss_ffi, "create_context", lambda *args, **kwargs: None
            )
        solve = make_rodas5_v2_dense_solver(
            jac_sparse_fn=system["jac_sparse_array"],
            linear_solver_precision="fp32",
        )
        sparse_backend = "cudss" if solver_mode == "sparse_cudss" else "dense_fallback"

    results = benchmark.pedantic(
        lambda: solve(
            y0=system["y0"],
            t_span=_T_SPAN,
            params_batch=params_batch,
            first_step=1e-6,
            rtol=1e-6,
            atol=1e-8,
        ).block_until_ready(),
        warmup_rounds=1,
        rounds=1,
    )
    benchmark.extra_info["backend"] = jax.default_backend()
    benchmark.extra_info["solver_mode"] = solver_mode
    benchmark.extra_info["sparse_backend"] = sparse_backend

    assert results.shape == (ensemble_size, len(_T_SPAN), system["n_vars"])
    mass_atol = 6e-6 if solver_mode == "sparse_cudss" else 3e-6
    np.testing.assert_allclose(results.sum(axis=2), 1.0, atol=mass_atol)
