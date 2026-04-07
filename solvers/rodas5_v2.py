"""Rodas5 solver — single-loop batched variant.

Same Rodas5 math and float64 precision as rodas5.py, but solve_ensemble
uses a single jax.lax.while_loop with the batch dimension inside the loop
body instead of vmap-over-while-loop.

The batch_size parameter controls how many trajectories share a while loop.
batch_size=N (default) puts all trajectories in one loop; batch_size=1
recovers the vmap-over-while-loop behaviour of rodas5.py.

For linear systems dy/dt = J(p) y, pass jac_fn instead of f to avoid
automatic differentiation entirely. For sparse linear systems, pass
jac_sparse_fn and return either a static-shape SparseJacobianCSR descriptor
or a fixed-shape BCOO/BCSR matrix. The sparse path uses cuDSS via JAX FFI on
CUDA when available and otherwise falls back to the dense JAX LU path.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass

import jax
import numpy as np
import scipy.sparse

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
from jax import lax  # isort: skip  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
from jax import tree_util  # isort: skip  # noqa: E402
from jax.experimental import sparse as jsparse  # isort: skip  # noqa: E402

from solvers import _cudss_ffi  # noqa: E402

# fmt: off
# Rodas5 W-transformed coefficients
_gamma = 0.19

_a21 = 2.0
_a31 = 3.040894194418781;  _a32 = 1.041747909077569
_a41 = 2.576417536461461;  _a42 = 1.622083060776640;  _a43 = -0.9089668560264532
_a51 = 2.760842080225597;  _a52 = 1.446624659844071;  _a53 = -0.3036980084553738;  _a54 = 0.2877498600325443
_a61 = -14.09640773051259; _a62 = 6.925207756232704;  _a63 = -41.47510893210728;   _a64 = 2.343771018586405;  _a65 = 24.13215229196062
_a71 = _a61;               _a72 = _a62;               _a73 = _a63;                  _a74 = _a64;              _a75 = _a65;              _a76 = 1.0
_a81 = _a61;               _a82 = _a62;               _a83 = _a63;                  _a84 = _a64;              _a85 = _a65;              _a86 = 1.0;  _a87 = 1.0

_C21 = -10.31323885133993
_C31 = -21.04823117650003; _C32 = -7.234992135176716
_C41 = 32.22751541853323;  _C42 = -4.943732386540191;  _C43 = 19.44922031041879
_C51 = -20.69865579590063; _C52 = -8.816374604402768;  _C53 = 1.260436877740897;   _C54 = -0.7495647613787146
_C61 = -46.22004352711257; _C62 = -17.49534862857472;  _C63 = -289.6389582892057;  _C64 = 93.60855400400906;  _C65 = 318.3822534212147
_C71 = 34.20013733472935;  _C72 = -14.15535402717690;  _C73 = 57.82335640988400;   _C74 = 25.83362985412365;  _C75 = 1.408950972071624;  _C76 = -6.551835421242162
_C81 = 42.57076742291101;  _C82 = -13.80770672017997;  _C83 = 93.98938432427124;   _C84 = 18.77919633714503;  _C85 = -31.58359187223370;  _C86 = -6.685968952921985;  _C87 = -5.810979938412932
# fmt: on

_SUPPORTED_LINEAR_SOLVER_PRECISIONS = ("fp64", "fp32")
_MATVEC_DOT_PRECISION = lax.DotAlgorithmPreset.TF32_TF32_F32
_BATCHED_DOT_DIMENSION_NUMBERS = (((2,), (1,)), ((0,), (0,)))
_SPARSE_BACKEND_CUDSS = "cudss_ffi"
_SPARSE_BACKEND_DENSE_FALLBACK = "dense_fallback"
_SPARSE_CALLBACK_TYPES = (jsparse.BCOO, jsparse.BCSR)


@functools.partial(
    tree_util.register_dataclass,
    data_fields=["indptr", "indices", "values", "nnz"],
    meta_fields=["shape"],
)
@dataclass(frozen=True)
class SparseJacobianCSR:
    """Static-shape CSR descriptor for JAX-traceable sparse Jacobians."""

    indptr: object
    indices: object
    values: object
    nnz: object
    shape: tuple[int, int]


def _batched_matvec(mat, vec, *, precision):
    out = lax.dot(
        mat,
        vec[..., None],
        dimension_numbers=_BATCHED_DOT_DIMENSION_NUMBERS,
        precision=precision,
    )
    return out[..., 0]


def _count_provided(*callbacks):
    return sum(callback is not None for callback in callbacks)


def _validate_solver_args(f, jac_fn, jac_sparse_fn, linear_solver_precision):
    if jac_fn is not None and (f is not None or jac_sparse_fn is not None):
        raise ValueError("jac_fn cannot be combined with f or jac_sparse_fn")
    if jac_sparse_fn is None and f is None and jac_fn is None:
        raise ValueError("At least one of f, jac_fn, or jac_sparse_fn must be provided")
    if jac_sparse_fn is not None and f is None and jac_fn is None:
        pass
    elif jac_sparse_fn is not None and f is not None:
        pass
    elif jac_sparse_fn is None and f is not None:
        pass
    elif jac_sparse_fn is None and jac_fn is not None:
        pass
    else:
        raise ValueError("Unsupported callback combination for rodas5_v2")
    if linear_solver_precision not in _SUPPORTED_LINEAR_SOLVER_PRECISIONS:
        raise ValueError(
            "linear_solver_precision must be one of "
            f"{_SUPPORTED_LINEAR_SOLVER_PRECISIONS}, got {linear_solver_precision!r}"
        )


def _wrap_sparse_callback(jac_sparse_fn):
    params = list(inspect.signature(jac_sparse_fn).parameters.values())
    positional = [
        param
        for param in params
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    argcount = len(positional)
    if argcount not in (1, 2, 3):
        raise TypeError(
            "jac_sparse_fn must accept 1, 2, or 3 positional parameters: "
            "(params), (t, params), (y, params), or (t, y, params)"
        )

    first_name = positional[0].name.lower() if positional else ""
    depends_on_time = argcount in (2, 3) and first_name in {"t", "time"}
    depends_on_state = argcount == 3 or (
        argcount == 2 and first_name not in {"t", "time"}
    )

    if argcount == 1:

        def call(t, y, p):
            del t, y
            return jac_sparse_fn(p)

    elif argcount == 2 and first_name in {"t", "time"}:

        def call(t, y, p):
            del y
            return jac_sparse_fn(t, p)

    elif argcount == 2:

        def call(t, y, p):
            del t
            return jac_sparse_fn(y, p)

    else:

        def call(t, y, p):
            return jac_sparse_fn(t, y, p)

    return call, depends_on_time, depends_on_state


def _coerce_concrete_sparse_csr(matrix, *, expected_shape=None):
    if isinstance(matrix, SparseJacobianCSR):
        shape = tuple(matrix.shape)
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(
                f"Expected sparse Jacobian shape {expected_shape}, got {shape}"
            )
        indptr = np.asarray(matrix.indptr, dtype=np.int32)
        indices = np.asarray(matrix.indices, dtype=np.int32)
        values = np.asarray(matrix.values)
        nnz = int(np.asarray(matrix.nnz))
        return SparseJacobianCSR(
            indptr=indptr, indices=indices, values=values, nnz=nnz, shape=shape
        )

    if isinstance(matrix, jsparse.BCOO):
        indices = np.asarray(matrix.indices, dtype=np.int32)
        data = np.asarray(matrix.data)
        coo = scipy.sparse.coo_matrix(
            (data, (indices[:, 0], indices[:, 1])), shape=matrix.shape
        )
        matrix = coo.tocsr()
    elif isinstance(matrix, jsparse.BCSR):
        matrix = scipy.sparse.csr_matrix(np.asarray(matrix.todense()))
    elif scipy.sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError(
                f"jac_sparse_fn output must be 2D, got shape {dense.shape}"
            )
        matrix = scipy.sparse.csr_matrix(dense)

    shape = tuple(matrix.shape)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"Expected sparse Jacobian shape {expected_shape}, got {shape}"
        )
    return SparseJacobianCSR(
        indptr=np.asarray(matrix.indptr, dtype=np.int32),
        indices=np.asarray(matrix.indices, dtype=np.int32),
        values=np.asarray(matrix.data),
        nnz=int(matrix.nnz),
        shape=shape,
    )


def _stack_sparse_batch(items):
    first = items[0]
    return SparseJacobianCSR(
        indptr=jnp.asarray(
            np.stack([np.asarray(item.indptr) for item in items], axis=0),
            dtype=jnp.int32,
        ),
        indices=jnp.asarray(
            np.stack([np.asarray(item.indices) for item in items], axis=0),
            dtype=jnp.int32,
        ),
        values=jnp.asarray(
            np.stack([np.asarray(item.values) for item in items], axis=0)
        ),
        nnz=jnp.asarray(np.array([int(item.nnz) for item in items], dtype=np.int32)),
        shape=first.shape,
    )


def _sparse_structure_matches(items):
    first = items[0]
    first_indptr = np.asarray(first.indptr)
    first_indices = np.asarray(first.indices)
    first_nnz = int(first.nnz)
    return all(
        np.array_equal(np.asarray(item.indptr), first_indptr)
        and np.array_equal(np.asarray(item.indices), first_indices)
        and int(item.nnz) == first_nnz
        for item in items[1:]
    )


def _pad_sparse_batch(batch, target_size):
    current = int(batch.nnz.shape[0])
    if current == target_size:
        return batch
    pad = target_size - current
    return SparseJacobianCSR(
        indptr=jnp.concatenate(
            [
                batch.indptr,
                jnp.broadcast_to(batch.indptr[-1:], (pad,) + batch.indptr.shape[1:]),
            ],
            axis=0,
        ),
        indices=jnp.concatenate(
            [
                batch.indices,
                jnp.broadcast_to(batch.indices[-1:], (pad,) + batch.indices.shape[1:]),
            ],
            axis=0,
        ),
        values=jnp.concatenate(
            [
                batch.values,
                jnp.broadcast_to(batch.values[-1:], (pad,) + batch.values.shape[1:]),
            ],
            axis=0,
        ),
        nnz=jnp.concatenate(
            [batch.nnz, jnp.broadcast_to(batch.nnz[-1:], (pad,))], axis=0
        ),
        shape=batch.shape,
    )


def _reshape_sparse_groups(batch, n_chunks, bs):
    return SparseJacobianCSR(
        indptr=batch.indptr.reshape((n_chunks, bs) + batch.indptr.shape[1:]),
        indices=batch.indices.reshape((n_chunks, bs) + batch.indices.shape[1:]),
        values=batch.values.reshape((n_chunks, bs) + batch.values.shape[1:]),
        nnz=batch.nnz.reshape((n_chunks, bs)),
        shape=batch.shape,
    )


def _csr_matvec_single(csr: SparseJacobianCSR, vec, *, out_dtype):
    max_nnz = csr.indices.shape[0]
    nnz = jnp.asarray(csr.nnz, dtype=jnp.int32)
    positions = jnp.arange(max_nnz, dtype=jnp.int32)
    mask = positions < nnz
    row_ids = jnp.searchsorted(csr.indptr[1:], positions, side="right")
    row_ids = jnp.where(mask, row_ids, 0)
    col_ids = jnp.where(mask, csr.indices, 0)
    vals = jnp.where(mask, csr.values, 0)
    contrib = vals.astype(out_dtype) * vec[col_ids].astype(out_dtype)
    return jnp.zeros((csr.shape[0],), dtype=out_dtype).at[row_ids].add(contrib)


def _csr_matvec_batch(csr_batch: SparseJacobianCSR, vec, *, out_dtype):
    return jax.vmap(lambda csr, v: _csr_matvec_single(csr, v, out_dtype=out_dtype))(
        csr_batch, vec
    )


def _csr_to_dense_single(csr: SparseJacobianCSR, *, dtype):
    max_nnz = csr.indices.shape[0]
    nnz = jnp.asarray(csr.nnz, dtype=jnp.int32)
    positions = jnp.arange(max_nnz, dtype=jnp.int32)
    mask = positions < nnz
    row_ids = jnp.searchsorted(csr.indptr[1:], positions, side="right")
    row_ids = jnp.where(mask, row_ids, 0)
    col_ids = jnp.where(mask, csr.indices, 0)
    vals = jnp.where(mask, csr.values, 0).astype(dtype)
    return jnp.zeros(csr.shape, dtype=dtype).at[row_ids, col_ids].add(vals)


def _csr_to_dense_batch(csr_batch: SparseJacobianCSR, *, dtype):
    return jax.vmap(lambda csr: _csr_to_dense_single(csr, dtype=dtype))(csr_batch)


def _resolve_sparse_backend(context):
    if context is None:
        return _SPARSE_BACKEND_DENSE_FALLBACK
    return _SPARSE_BACKEND_CUDSS


def make_solver(
    f=None,
    *,
    jac_fn=None,
    jac_sparse_fn=None,
    linear_solver_precision="fp64",
    batch_size=None,
):
    """Create a reusable Rodas5 ensemble solver."""
    _validate_solver_args(f, jac_fn, jac_sparse_fn, linear_solver_precision)

    lu_dtype = jnp.float32 if linear_solver_precision == "fp32" else jnp.float64
    matvec_dtype = jnp.float32 if linear_solver_precision == "fp32" else jnp.float64
    dot_precision = (
        _MATVEC_DOT_PRECISION
        if linear_solver_precision == "fp32" and jax.default_backend() != "cpu"
        else lax.Precision.DEFAULT
    )
    precision_code = 1 if linear_solver_precision == "fp32" else 0

    lu_factor_batched = jax.vmap(jax.scipy.linalg.lu_factor)
    lu_solve_batched = jax.vmap(jax.scipy.linalg.lu_solve)

    sparse_call = None
    depends_on_time = False
    depends_on_state = False
    if jac_sparse_fn is not None:
        sparse_call, depends_on_time, depends_on_state = _wrap_sparse_callback(
            jac_sparse_fn
        )

    if f is not None:
        _f_batched = jax.vmap(f)

    if jac_fn is not None:
        _jac_fn_batched = jax.vmap(jac_fn)
    elif jac_sparse_fn is None:
        _jac_batched = jax.vmap(lambda y, p: jax.jacobian(lambda y_: f(y_, p))(y))
    elif depends_on_time or depends_on_state:
        _dynamic_sparse_batched = jax.vmap(sparse_call)
        _dynamic_sparse_values_batched = jax.vmap(lambda t, y, p: sparse_call(t, y, p).values)

    context_cache = {}

    def _get_context(n_vars, bs):
        key = (int(n_vars), int(bs))
        if key not in context_cache:
            context_cache[key] = _cudss_ffi.create_context(
                int(n_vars), int(bs), precision_code
            )
        return context_cache[key]

    @functools.partial(
        jax.jit,
        static_argnames=(
            "n_save",
            "max_steps",
            "sparse_mode",
            "use_cudss",
            "cudss_handle",
        ),
    )
    def _solve_impl(
        y0_arr,
        params_groups,
        times,
        sparse_indptr_groups,
        sparse_indices_groups,
        sparse_values_groups,
        sparse_nnz_groups,
        *,
        sparse_mode,
        use_cudss,
        cudss_handle,
        n_save,
        max_steps,
        dt0,
        rtol,
        atol,
    ):
        n_vars = y0_arr.shape[0]
        bs = params_groups.shape[1]
        eye_lu = jnp.eye(n_vars, dtype=lu_dtype)[None, :, :]
        tf = times[-1]

        def _factorize_from_jacobian(jac_lu, dt):
            dtgamma_inv = (1.0 / (dt * _gamma)).astype(lu_dtype)[:, None, None]
            return lu_factor_batched(dtgamma_inv * eye_lu - jac_lu)

        def _solve_factorized(factorized, rhs):
            sol = lu_solve_batched(factorized, rhs.astype(lu_dtype))
            return sol.astype(jnp.float64)

        def _step_with_dense_factorization(y, dt, f_eval, jac_lu):
            factorized = _factorize_from_jacobian(jac_lu, dt)
            inv_dt = (1.0 / dt)[:, None]

            def lu_solve(rhs):
                return _solve_factorized(factorized, rhs)

            dy = f_eval(y)
            k1 = lu_solve(dy)

            u = y + _a21 * k1
            du = f_eval(u)
            k2 = lu_solve(du + _C21 * k1 * inv_dt)

            u = y + _a31 * k1 + _a32 * k2
            du = f_eval(u)
            k3 = lu_solve(du + (_C31 * k1 + _C32 * k2) * inv_dt)

            u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
            du = f_eval(u)
            k4 = lu_solve(du + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt)

            u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
            du = f_eval(u)
            k5 = lu_solve(du + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt)

            u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
            du = f_eval(u)
            k6 = lu_solve(
                du
                + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5) * inv_dt
            )

            u = u + k6
            du = f_eval(u)
            k7 = lu_solve(
                du
                + (
                    _C71 * k1
                    + _C72 * k2
                    + _C73 * k3
                    + _C74 * k4
                    + _C75 * k5
                    + _C76 * k6
                )
                * inv_dt
            )

            u = u + k7
            du = f_eval(u)
            k8 = lu_solve(
                du
                + (
                    _C81 * k1
                    + _C82 * k2
                    + _C83 * k3
                    + _C84 * k4
                    + _C85 * k5
                    + _C86 * k6
                    + _C87 * k7
                )
                * inv_dt
            )

            return u + k8, k8

        def _cudss_prepare(dt, sparse_batch):
            token = jax.ShapeDtypeStruct((1,), jnp.uint8)
            target = (
                _cudss_ffi._CUDA_TARGET_PREPARE
                if sparse_mode == "static"
                else _cudss_ffi._CUDA_TARGET_PREPARE_DYNAMIC
            )
            return jax.ffi.ffi_call(
                target,
                token,
                has_side_effect=True,
                vmap_method="sequential",
            )(
                dt,
                sparse_batch.indptr,
                sparse_batch.indices,
                sparse_batch.values.astype(lu_dtype),
                sparse_batch.nnz,
                context_handle=np.uint64(cudss_handle),
            )

        def _cudss_solve(token, rhs):
            target = (
                _cudss_ffi._CUDA_TARGET_SOLVE
                if sparse_mode == "static"
                else _cudss_ffi._CUDA_TARGET_SOLVE_DYNAMIC
            )
            return jax.ffi.ffi_call(
                target,
                jax.ShapeDtypeStruct(rhs.shape, lu_dtype),
                has_side_effect=True,
                vmap_method="sequential",
            )(
                token,
                rhs.astype(lu_dtype),
                context_handle=np.uint64(cudss_handle),
            ).astype(jnp.float64)

        def _step_with_sparse_system(y, dt, sparse_batch, f_eval):
            if use_cudss:
                prepare_token = _cudss_prepare(dt, sparse_batch)

                def lu_solve(rhs):
                    return _cudss_solve(prepare_token, rhs)

            else:
                jac_dense = _csr_to_dense_batch(sparse_batch, dtype=lu_dtype)
                factorized = _factorize_from_jacobian(jac_dense, dt)

                def lu_solve(rhs):
                    return _solve_factorized(factorized, rhs)

            inv_dt = (1.0 / dt)[:, None]

            dy = f_eval(y)
            k1 = lu_solve(dy)

            u = y + _a21 * k1
            du = f_eval(u)
            k2 = lu_solve(du + _C21 * k1 * inv_dt)

            u = y + _a31 * k1 + _a32 * k2
            du = f_eval(u)
            k3 = lu_solve(du + (_C31 * k1 + _C32 * k2) * inv_dt)

            u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
            du = f_eval(u)
            k4 = lu_solve(du + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt)

            u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
            du = f_eval(u)
            k5 = lu_solve(du + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt)

            u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
            du = f_eval(u)
            k6 = lu_solve(
                du
                + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5) * inv_dt
            )

            u = u + k6
            du = f_eval(u)
            k7 = lu_solve(
                du
                + (
                    _C71 * k1
                    + _C72 * k2
                    + _C73 * k3
                    + _C74 * k4
                    + _C75 * k5
                    + _C76 * k6
                )
                * inv_dt
            )

            u = u + k7
            du = f_eval(u)
            k8 = lu_solve(
                du
                + (
                    _C81 * k1
                    + _C82 * k2
                    + _C83 * k3
                    + _C84 * k4
                    + _C85 * k5
                    + _C86 * k6
                    + _C87 * k7
                )
                * inv_dt
            )

            return u + k8, k8

        def _solve_chunk(chunk_index, params_chunk):
            y_init = jnp.broadcast_to(y0_arr, (bs, n_vars)).copy()
            hist_init = (
                jnp.zeros((bs, n_save, n_vars), dtype=jnp.float64)
                .at[:, 0, :]
                .set(y_init)
            )
            t_init = jnp.full((bs,), times[0], dtype=jnp.float64)
            dt_init = jnp.full((bs,), dt0, dtype=jnp.float64)
            save_idx_init = jnp.ones((bs,), dtype=jnp.int32)

            if jac_fn is not None:
                jac = _jac_fn_batched(params_chunk)
                jac_matvec = jac.astype(matvec_dtype)
                jac_lu = jac.astype(lu_dtype)

                def _step_batch(t, y, dt):
                    del t

                    def f_eval(u):
                        out = _batched_matvec(
                            jac_matvec,
                            u.astype(matvec_dtype),
                            precision=dot_precision,
                        )
                        return out.astype(jnp.float64)

                    return _step_with_dense_factorization(y, dt, f_eval, jac_lu)

            elif jac_sparse_fn is not None:
                if sparse_mode == "static":
                    sparse_chunk = SparseJacobianCSR(
                        indptr=sparse_indptr_groups[chunk_index],
                        indices=sparse_indices_groups[chunk_index],
                        values=sparse_values_groups[chunk_index],
                        nnz=sparse_nnz_groups[chunk_index],
                        shape=(n_vars, n_vars),
                    )

                    def _step_batch(t, y, dt):
                        del t
                        if f is not None:
                            f_eval = lambda u: _f_batched(u, params_chunk)
                        else:
                            f_eval = lambda u: _csr_matvec_batch(
                                sparse_chunk,
                                u.astype(matvec_dtype),
                                out_dtype=matvec_dtype,
                            ).astype(jnp.float64)
                        return _step_with_sparse_system(y, dt, sparse_chunk, f_eval)

                elif sparse_mode == "dynamic_values":

                    def _step_batch(t, y, dt):
                        sparse_chunk = SparseJacobianCSR(
                            indptr=sparse_indptr_groups[chunk_index],
                            indices=sparse_indices_groups[chunk_index],
                            values=_dynamic_sparse_values_batched(t, y, params_chunk),
                            nnz=sparse_nnz_groups[chunk_index],
                            shape=(n_vars, n_vars),
                        )
                        if f is not None:
                            f_eval = lambda u: _f_batched(u, params_chunk)
                        else:
                            f_eval = lambda u: _csr_matvec_batch(
                                sparse_chunk,
                                u.astype(matvec_dtype),
                                out_dtype=matvec_dtype,
                            ).astype(jnp.float64)
                        return _step_with_sparse_system(y, dt, sparse_chunk, f_eval)

                else:

                    def _step_batch(t, y, dt):
                        sparse_chunk = _dynamic_sparse_batched(t, y, params_chunk)
                        if f is not None:
                            f_eval = lambda u: _f_batched(u, params_chunk)
                        else:
                            f_eval = lambda u: _csr_matvec_batch(
                                sparse_chunk,
                                u.astype(matvec_dtype),
                                out_dtype=matvec_dtype,
                            ).astype(jnp.float64)
                        return _step_with_sparse_system(y, dt, sparse_chunk, f_eval)

            else:

                def _step_batch(t, y, dt):
                    del t
                    jac = _jac_batched(y, params_chunk)
                    jac_lu = jac.astype(lu_dtype)

                    def f_eval(u):
                        return _f_batched(u, params_chunk)

                    return _step_with_dense_factorization(y, dt, f_eval, jac_lu)

            def cond_fn(state):
                t, _, _, _, save_idx, n_steps = state
                active = save_idx < n_save
                return (jnp.min(jnp.where(active, t, tf)) < tf) & (n_steps < max_steps)

            def body_fn(state):
                t, y, dt, hist, save_idx, n_steps = state
                active = save_idx < n_save
                next_target = times[save_idx]
                dt_use = jnp.where(
                    active,
                    jnp.maximum(jnp.minimum(dt, next_target - t), 1e-30),
                    1e-30,
                )

                y_new, err_est = _step_batch(t, y, dt_use)

                scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
                err_norm = jnp.sqrt(jnp.mean((err_est / scale) ** 2, axis=1))

                accept = active & (err_norm <= 1.0) & ~jnp.isnan(err_norm)
                t_new = jnp.where(accept, t + dt_use, t)
                y_out = jnp.where(accept[:, None], y_new, y)

                reached = accept & (
                    jnp.abs(t_new - next_target)
                    <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
                )
                slot_mask = (
                    jax.nn.one_hot(save_idx, n_save, dtype=jnp.bool_) & reached[:, None]
                )
                hist_new = jnp.where(slot_mask[:, :, None], y_out[:, None, :], hist)
                save_idx_new = save_idx + reached.astype(jnp.int32)

                safe_err = jnp.where(
                    jnp.isnan(err_norm) | (err_norm > 1e18),
                    1e18,
                    jnp.where(err_norm == 0.0, 1e-18, err_norm),
                )
                factor = jnp.clip(0.9 * safe_err ** (-1.0 / 6.0), 0.2, 6.0)
                dt_new = jnp.where(active, dt_use * factor, dt)

                return (t_new, y_out, dt_new, hist_new, save_idx_new, n_steps + 1)

            init = (t_init, y_init, dt_init, hist_init, save_idx_init, jnp.int32(0))
            _, _, _, hist_final, _, _ = lax.while_loop(cond_fn, body_fn, init)
            return hist_final

        if jac_sparse_fn is not None:
            return lax.map(
                lambda idx_params: _solve_chunk(idx_params[0], idx_params[1]),
                (jnp.arange(params_groups.shape[0]), params_groups),
            )
        return jax.vmap(_solve_chunk)(jnp.arange(params_groups.shape[0]), params_groups)

    def _solve(
        y0,
        t_span,
        params_batch,
        *,
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
    ):
        y0_arr = jnp.asarray(y0, dtype=jnp.float64)
        params_batch_arr = jnp.asarray(params_batch)
        n_vars = int(y0_arr.shape[0])
        N = int(params_batch_arr.shape[0])
        times = np.asarray(t_span, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError(
                f"t_span must be a 1D array of save times, got shape {times.shape}"
            )
        if times.size < 2:
            raise ValueError(
                f"t_span must contain at least 2 save times, got {times.size}"
            )
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("t_span must be strictly increasing")

        times_jnp = jnp.asarray(times, dtype=jnp.float64)
        dt0 = jnp.float64(
            first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
        )
        bs = N if batch_size is None else batch_size
        n_chunks = (N + bs - 1) // bs
        n_padded = n_chunks * bs

        if n_padded > N:
            pad_rows = jnp.broadcast_to(
                params_batch_arr[-1:],
                (n_padded - N,) + params_batch_arr.shape[1:],
            )
            params_padded = jnp.concatenate([params_batch_arr, pad_rows], axis=0)
        else:
            params_padded = params_batch_arr

        params_groups = params_padded.reshape(
            (n_chunks, bs) + params_batch_arr.shape[1:]
        )
        sparse_mode = "none"
        use_cudss = False
        cudss_handle = np.uint64(0)
        sparse_groups = SparseJacobianCSR(
            indptr=jnp.zeros((0, 0, 0), dtype=jnp.int32),
            indices=jnp.zeros((0, 0, 0), dtype=jnp.int32),
            values=jnp.zeros((0, 0, 0), dtype=lu_dtype),
            nnz=jnp.zeros((0, 0), dtype=jnp.int32),
            shape=(n_vars, n_vars),
        )

        if jac_sparse_fn is not None:
            context = _get_context(n_vars, bs)
            if not depends_on_time and not depends_on_state:
                sparse_items = [
                    _coerce_concrete_sparse_csr(
                        sparse_call(times[0], y0_arr, np.asarray(params_padded[i])),
                        expected_shape=(n_vars, n_vars),
                    )
                    for i in range(n_padded)
                ]
                sparse_batch = _stack_sparse_batch(sparse_items)
                sparse_groups = _reshape_sparse_groups(sparse_batch, n_chunks, bs)
                sparse_mode = "static"
                use_cudss = _resolve_sparse_backend(context) == _SPARSE_BACKEND_CUDSS
                if use_cudss:
                    _cudss_ffi.reset_active(context.handle)
                    cudss_handle = np.uint64(context.handle)
            else:
                sparse_items = [
                    _coerce_concrete_sparse_csr(
                        sparse_call(times[0], y0_arr, np.asarray(params_padded[i])),
                        expected_shape=(n_vars, n_vars),
                    )
                    for i in range(n_padded)
                ]
                if _sparse_structure_matches(sparse_items):
                    sparse_batch = _stack_sparse_batch(sparse_items)
                    sparse_groups = _reshape_sparse_groups(sparse_batch, n_chunks, bs)
                    sparse_mode = "dynamic_values"
                    use_cudss = _resolve_sparse_backend(context) == _SPARSE_BACKEND_CUDSS
                    if use_cudss:
                        _cudss_ffi.reset_active(context.handle)
                        cudss_handle = np.uint64(context.handle)
                else:
                    sparse_mode = "dynamic"
                    use_cudss = False

        results = _solve_impl(
            y0_arr,
            params_groups,
            times_jnp,
            sparse_groups.indptr,
            sparse_groups.indices,
            sparse_groups.values.astype(lu_dtype),
            sparse_groups.nnz,
            sparse_mode=sparse_mode,
            use_cudss=bool(use_cudss),
            cudss_handle=int(cudss_handle),
            n_save=int(times.size),
            max_steps=int(max_steps),
            dt0=float(dt0),
            rtol=float(rtol),
            atol=float(atol),
        )
        return results.reshape(n_padded, int(times.size), n_vars)[:N]

    def _get_cudss_counters(n_vars, bs):
        ctx = context_cache.get((int(n_vars), int(bs)))
        return None if ctx is None else ctx.counters

    _solve.get_cudss_counters = _get_cudss_counters
    return _solve


def solve_ensemble(
    f,
    y0,
    t_span,
    params_batch,
    *,
    jac_fn=None,
    jac_sparse_fn=None,
    linear_solver_precision="fp64",
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    batch_size=None,
):
    """Solve an ensemble of ODEs with configurable batch grouping."""
    solver = make_solver(
        f,
        jac_fn=jac_fn,
        jac_sparse_fn=jac_sparse_fn,
        linear_solver_precision=linear_solver_precision,
        batch_size=batch_size,
    )
    return solver(
        y0,
        t_span,
        params_batch,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )
