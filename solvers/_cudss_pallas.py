from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

try:
    from jax._src.pallas.triton import core as pltriton
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - import fallback for unsupported installs
    pl = None
    pltriton = None

_BLOCK = 32


def _pad_cols_pow2(n_cols: int) -> int:
    return 1 << max(0, int(n_cols - 1)).bit_length()


def _pad_batch(arr, target_rows):
    pad = target_rows - int(arr.shape[0])
    if pad <= 0:
        return arr
    return jnp.concatenate(
        [arr, jnp.zeros((pad,) + arr.shape[1:], dtype=arr.dtype)], axis=0
    )


def _pad_last_dim(arr, target_cols):
    pad = target_cols - int(arr.shape[-1])
    if pad <= 0:
        return arr
    pad_shape = arr.shape[:-1] + (pad,)
    return jnp.concatenate([arr, jnp.zeros(pad_shape, dtype=arr.dtype)], axis=-1)


def _can_use_pallas():
    return pl is not None and pltriton is not None and jax.default_backend() == "gpu"


def add_vectors(lhs, rhs):
    lhs = jnp.asarray(lhs)
    rhs = jnp.asarray(rhs)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Expected matching shapes, got {lhs.shape} and {rhs.shape}")
    if lhs.ndim != 2:
        raise ValueError(f"Expected rank-2 arrays, got {lhs.ndim}")
    if not _can_use_pallas():
        return lhs + rhs

    rows, n_cols = lhs.shape
    n_pad = ((rows + _BLOCK - 1) // _BLOCK) * _BLOCK
    cols_pad = _pad_cols_pow2(n_cols)
    lhs_pad = _pad_last_dim(_pad_batch(lhs, n_pad), cols_pad)
    rhs_pad = _pad_last_dim(_pad_batch(rhs, n_pad), cols_pad)

    def kernel(lhs_ref, rhs_ref, out_ref):
        def body(i, carry):
            lv = lhs_ref.at[:, pl.ds(i, 1)][...][:, 0]
            rv = rhs_ref.at[:, pl.ds(i, 1)][...][:, 0]
            out_ref.at[:, pl.ds(i, 1)][...] = (lv + rv)[:, None]
            return carry

        jax.lax.fori_loop(0, n_cols, body, jnp.int32(0))

    block = pl.BlockSpec((_BLOCK, cols_pad), lambda i: (i, 0))
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_pad, cols_pad), lhs.dtype),
        in_specs=[block, block],
        out_specs=block,
        grid=(n_pad // _BLOCK,),
        compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
    )(lhs_pad, rhs_pad)
    return out[:rows, :n_cols]


def stage_state(base, terms):
    base = jnp.asarray(base)
    if base.ndim != 2:
        raise ValueError(f"Expected rank-2 base array, got {base.ndim}")
    stages = []
    coeffs = []
    for coeff, arr in terms:
        coeffs.append(float(coeff))
        stages.append(jnp.asarray(arr, dtype=base.dtype))
    if len(stages) > 7:
        raise ValueError("stage_state supports at most 7 stage inputs")
    if not _can_use_pallas():
        out = base
        for coeff, arr in terms:
            out = out + coeff * jnp.asarray(arr, dtype=base.dtype)
        return out

    zeros = jnp.zeros_like(base)
    while len(stages) < 7:
        coeffs.append(0.0)
        stages.append(zeros)

    rows, n_cols = base.shape
    n_pad = ((rows + _BLOCK - 1) // _BLOCK) * _BLOCK
    cols_pad = _pad_cols_pow2(n_cols)
    base_pad = _pad_last_dim(_pad_batch(base, n_pad), cols_pad)
    stage_pads = [_pad_last_dim(_pad_batch(stage, n_pad), cols_pad) for stage in stages]

    def kernel(base_ref, s1_ref, s2_ref, s3_ref, s4_ref, s5_ref, s6_ref, s7_ref, out_ref):
        def body(i, carry):
            value = base_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[0] * s1_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[1] * s2_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[2] * s3_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[3] * s4_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[4] * s5_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[5] * s6_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[6] * s7_ref.at[:, pl.ds(i, 1)][...][:, 0]
            out_ref.at[:, pl.ds(i, 1)][...] = value[:, None]
            return carry

        jax.lax.fori_loop(0, n_cols, body, jnp.int32(0))

    block = pl.BlockSpec((_BLOCK, cols_pad), lambda i: (i, 0))
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_pad, cols_pad), base.dtype),
        in_specs=[block] * 8,
        out_specs=block,
        grid=(n_pad // _BLOCK,),
        compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
    )(base_pad, *stage_pads)
    return out[:rows, :n_cols]


def stage_rhs(du, dt, terms):
    du = jnp.asarray(du)
    dt = jnp.asarray(dt)
    if du.ndim != 2:
        raise ValueError(f"Expected rank-2 du array, got {du.ndim}")
    if dt.ndim != 1 or dt.shape[0] != du.shape[0]:
        raise ValueError(f"Expected dt shape {(du.shape[0],)}, got {dt.shape}")
    stages = []
    coeffs = []
    for coeff, arr in terms:
        coeffs.append(float(coeff))
        stages.append(jnp.asarray(arr, dtype=du.dtype))
    if len(stages) > 7:
        raise ValueError("stage_rhs supports at most 7 stage inputs")
    if not _can_use_pallas():
        inv_dt = (1.0 / dt).astype(du.dtype)[:, None]
        out = du
        for coeff, arr in terms:
            out = out + coeff * jnp.asarray(arr, dtype=du.dtype) * inv_dt
        return out

    zeros = jnp.zeros_like(du)
    while len(stages) < 7:
        coeffs.append(0.0)
        stages.append(zeros)

    rows, n_cols = du.shape
    n_pad = ((rows + _BLOCK - 1) // _BLOCK) * _BLOCK
    cols_pad = _pad_cols_pow2(n_cols)
    du_pad = _pad_last_dim(_pad_batch(du, n_pad), cols_pad)
    dt_pad = _pad_batch(dt[:, None], n_pad)[:, 0]
    stage_pads = [_pad_last_dim(_pad_batch(stage, n_pad), cols_pad) for stage in stages]

    def kernel(du_ref, dt_ref, s1_ref, s2_ref, s3_ref, s4_ref, s5_ref, s6_ref, s7_ref, out_ref):
        inv_dt = 1.0 / dt_ref[...]

        def body(i, carry):
            value = du_ref.at[:, pl.ds(i, 1)][...][:, 0]
            value = value + coeffs[0] * s1_ref.at[:, pl.ds(i, 1)][...][:, 0] * inv_dt
            value = value + coeffs[1] * s2_ref.at[:, pl.ds(i, 1)][...][:, 0] * inv_dt
            value = value + coeffs[2] * s3_ref.at[:, pl.ds(i, 1)][...][:, 0] * inv_dt
            value = value + coeffs[3] * s4_ref.at[:, pl.ds(i, 1)][...][:, 0] * inv_dt
            value = value + coeffs[4] * s5_ref.at[:, pl.ds(i, 1)][...][:, 0] * inv_dt
            value = value + coeffs[5] * s6_ref.at[:, pl.ds(i, 1)][...][:, 0] * inv_dt
            value = value + coeffs[6] * s7_ref.at[:, pl.ds(i, 1)][...][:, 0] * inv_dt
            out_ref.at[:, pl.ds(i, 1)][...] = value[:, None]
            return carry

        jax.lax.fori_loop(0, n_cols, body, jnp.int32(0))

    block = pl.BlockSpec((_BLOCK, cols_pad), lambda i: (i, 0))
    dt_block = pl.BlockSpec((_BLOCK,), lambda i: (i,))
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_pad, cols_pad), du.dtype),
        in_specs=[block, dt_block] + [block] * 7,
        out_specs=block,
        grid=(n_pad // _BLOCK,),
        compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
    )(du_pad, dt_pad, *stage_pads)
    return out[:rows, :n_cols]


def copy_base_add_diagonal(base, diag_offsets, dt, gamma):
    base = jnp.asarray(base)
    diag_offsets = jnp.asarray(diag_offsets, dtype=jnp.int32)
    dt = jnp.asarray(dt)
    out = base
    valid = diag_offsets >= 0
    if not jnp.any(valid):
        return out
    batch_ids = jnp.arange(base.shape[0], dtype=jnp.int32)[:, None]
    diag_ids = jnp.broadcast_to(diag_offsets[None, :], (base.shape[0], diag_offsets.shape[0]))
    updates = (1.0 / (dt[:, None] * gamma)).astype(base.dtype)
    mask = jnp.broadcast_to(valid[None, :], updates.shape)
    return out.at[(batch_ids, jnp.where(mask, diag_ids, 0))].add(jnp.where(mask, updates, 0))


@functools.partial(jax.jit, static_argnames=("n_vars", "max_nnz"))
def _csr_matvec_uniform_impl(indptr, indices, values, nnz, vec, *, n_vars, max_nnz):
    indptr0 = indptr[0]
    indices0 = indices[0]
    nnz0 = nnz[0]
    if not _can_use_pallas():
        positions = jnp.arange(max_nnz, dtype=jnp.int32)
        mask = positions < nnz0
        row_ids = jnp.where(mask, jnp.searchsorted(indptr0[1:], positions, side="right"), 0)
        col_ids = jnp.where(mask, indices0, 0)
        contrib = values * vec[jnp.arange(vec.shape[0])[:, None], col_ids[None, :]]
        return jnp.zeros((vec.shape[0], n_vars), dtype=vec.dtype).at[
            jnp.arange(vec.shape[0])[:, None], row_ids[None, :]
        ].add(contrib)

    rows = vec.shape[0]
    n_pad = ((rows + _BLOCK - 1) // _BLOCK) * _BLOCK
    cols_pad = _pad_cols_pow2(n_vars)
    max_nnz_pad = _pad_cols_pow2(max_nnz)
    indptr_pad = _pad_last_dim(indptr0[None, :], _pad_cols_pow2(n_vars + 1))[0]
    indices_pad = _pad_last_dim(indices0[None, :], max_nnz_pad)[0]
    values_pad = _pad_last_dim(_pad_batch(values, n_pad), max_nnz_pad)
    nnz_pad = _pad_batch(nnz[:, None], n_pad)[:, 0]
    vec_pad = _pad_last_dim(_pad_batch(vec, n_pad), cols_pad)

    def kernel(indptr_ref, indices_ref, values_ref, nnz_ref, vec_ref, out_ref):
        def row(i, carry):
            start = indptr_ref.at[pl.ds(i, 1)][...][0]
            end = indptr_ref.at[pl.ds(i + 1, 1)][...][0]
            limit = jnp.minimum(end, nnz_ref[0])
            acc = vec_ref.at[:, 0][...] * 0.0

            def elem(k, acc):
                col_idx = indices_ref.at[pl.ds(k, 1)][...][0]
                val = values_ref.at[:, pl.ds(k, 1)][...][:, 0]
                yj = vec_ref.at[:, pl.ds(col_idx, 1)][...][:, 0]
                active = (k >= start) & (k < limit)
                return acc + jnp.where(active, val * yj, 0)

            acc = jax.lax.fori_loop(0, max_nnz, elem, acc)
            out_ref.at[:, pl.ds(i, 1)][...] = acc[:, None]
            return carry

        jax.lax.fori_loop(0, n_vars, row, jnp.int32(0))

    indptr_block = pl.BlockSpec((indptr_pad.shape[0],), lambda i: (0,))
    nnz_block = pl.BlockSpec((_BLOCK,), lambda i: (i,))
    index_block = pl.BlockSpec((max_nnz_pad,), lambda i: (0,))
    values_block = pl.BlockSpec((_BLOCK, max_nnz_pad), lambda i: (i, 0))
    vec_block = pl.BlockSpec((_BLOCK, cols_pad), lambda i: (i, 0))
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_pad, cols_pad), vec.dtype),
        in_specs=[indptr_block, index_block, values_block, nnz_block, vec_block],
        out_specs=vec_block,
        grid=(n_pad // _BLOCK,),
        compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
    )(indptr_pad, indices_pad, values_pad, nnz_pad, vec_pad)
    return out[:rows, :n_vars]


def batched_csr_matvec(indptr, indices, values, nnz, vec):
    """Batched CSR matvec for fixed-structure sparse batches.

    The Rodas sparse fast paths only call this helper when every batch item shares
    the same CSR structure and only the values vary. That lets us keep the helper
    JIT-safe without inspecting traced arrays on the Python side.
    """
    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    values = jnp.asarray(values)
    nnz = jnp.asarray(nnz, dtype=jnp.int32)
    vec = jnp.asarray(vec, dtype=values.dtype)
    n_vars = int(vec.shape[1])
    max_nnz = int(values.shape[1])
    return _csr_matvec_uniform_impl(
        indptr, indices, values, nnz, vec, n_vars=n_vars, max_nnz=max_nnz
    )
