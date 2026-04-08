"""Experimental blocked-LU Rodas5 solver using dense batched JAX arrays.

This variant targets Tensor Core-style acceleration by performing the LU
factorization in float32 with blocked trailing updates expressed via
``jax.lax.dot`` using a TF32 precision preset. It is intentionally
separate from the Pallas v6 solver so we can benchmark the approach in
isolation.
"""

from __future__ import annotations

import functools

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402
from jax import lax  # isort: skip  # noqa: E402
import jax.numpy as jnp  # isort: skip  # noqa: E402
import numpy as np

from solvers.rodas5_custom_kernel_v6 import _FINAL_UPDATE
from solvers.rodas5_custom_kernel_v6 import _gamma
from solvers.rodas5_custom_kernel_v6 import _STAGE_SPECS

_BLOCK_SIZE = 16
_DOT_PRECISION = lax.DotAlgorithmPreset.TF32_TF32_F32


def _pad_dim(n: int, block: int = _BLOCK_SIZE) -> int:
    return ((n + block - 1) // block) * block


def _stack_jacobian(jac_fn, t, dtype):
    m = jac_fn(t)
    rows = [jnp.stack([jnp.asarray(v, dtype=dtype) for v in row], axis=-1) for row in m]
    return jnp.stack(rows, axis=-2)


def _pad_matrix(mat, n_pad):
    pad = n_pad - mat.shape[-1]
    return jnp.pad(mat, ((0, 0), (0, pad), (0, pad)))


def _pad_vector(vec, n_pad):
    pad = n_pad - vec.shape[-1]
    return jnp.pad(vec, ((0, 0), (0, pad)))


def _unblocked_lu_block(block):
    """No-pivot LU on a batch of small dense blocks."""
    bs = block.shape[-1]
    lu = block
    for j in range(bs):
        pivot = lu[:, j, j]
        if j + 1 < bs:
            l_col = lu[:, j + 1 :, j] / pivot[:, None]
            lu = lu.at[:, j + 1 :, j].set(l_col)
            trailing = (
                lu[:, j + 1 :, j + 1 :] - l_col[:, :, None] * lu[:, None, j, j + 1 :]
            )
            lu = lu.at[:, j + 1 :, j + 1 :].set(trailing)
    return lu


def _batched_matmul(a, b, *, precision):
    out = lax.dot(
        a,
        b,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=precision,
    )
    return out


def _blocked_lu_nopivot(a, *, precision):
    """Blocked no-pivot LU factorization with TF32 trailing updates."""
    n = a.shape[-1]
    lu = a
    for k in range(0, n, _BLOCK_SIZE):
        bs = min(_BLOCK_SIZE, n - k)
        diag = lu[:, k : k + bs, k : k + bs]
        diag_lu = _unblocked_lu_block(diag)
        lu = lu.at[:, k : k + bs, k : k + bs].set(diag_lu)
        if k + bs >= n:
            continue

        a12 = lu[:, k : k + bs, k + bs :]
        l11 = jnp.tril(diag_lu, -1) + jnp.eye(bs, dtype=lu.dtype)[None, :, :]
        u12 = lax.linalg.triangular_solve(
            l11, a12, left_side=True, lower=True, unit_diagonal=True
        )
        lu = lu.at[:, k : k + bs, k + bs :].set(u12)

        a21 = lu[:, k + bs :, k : k + bs]
        u11 = jnp.triu(diag_lu)
        l21_t = lax.linalg.triangular_solve(
            u11, jnp.swapaxes(a21, 1, 2), left_side=True, lower=False
        )
        l21 = jnp.swapaxes(l21_t, 1, 2)
        lu = lu.at[:, k + bs :, k : k + bs].set(l21)

        trailing = lu[:, k + bs :, k + bs :] - _batched_matmul(
            l21, u12, precision=precision
        )
        lu = lu.at[:, k + bs :, k + bs :].set(trailing)
    return lu


def _lu_solve(lu, rhs):
    l = jnp.tril(lu, -1) + jnp.eye(lu.shape[-1], dtype=lu.dtype)[None, :, :]
    u = jnp.triu(lu)
    y = lax.linalg.triangular_solve(
        l, rhs[..., None], left_side=True, lower=True, unit_diagonal=True
    )
    x = lax.linalg.triangular_solve(u, y, left_side=True, lower=False)
    return x[..., 0]


def make_solver(jac_fn):
    """Create the experimental blocked-LU Rodas5 solver."""
    m0 = jnp.asarray(jac_fn(jnp.float64(0.0)), dtype=jnp.float64)
    if m0.ndim != 2 or m0.shape[0] != m0.shape[1]:
        raise ValueError(
            f"jac_fn(0.0) must be square with shape (n, n), got {m0.shape}"
        )

    n_vars = int(m0.shape[0])
    n_pad = _pad_dim(n_vars)
    dot_precision = (
        _DOT_PRECISION if jax.default_backend() != "cpu" else lax.Precision.DEFAULT
    )

    @functools.partial(
        jax.jit,
        static_argnames=("n_save", "max_steps"),
    )
    def _solve_impl(y0_batch, times, *, n_save, max_steps, dt0, rtol, atol):
        n_batch = y0_batch.shape[0]
        y = y0_batch
        hist = (
            jnp.zeros((n_batch, n_save, n_vars), dtype=jnp.float64)
            .at[:, 0, :]
            .set(y0_batch)
        )
        t = jnp.full((n_batch,), times[0], dtype=jnp.float64)
        dt_v = jnp.full((n_batch,), dt0, dtype=jnp.float64)
        save_idx = jnp.ones((n_batch,), dtype=jnp.int32)
        tf = times[-1]

        def eval_f(m_dense, vec):
            vec32 = vec.astype(jnp.float32)
            out = lax.dot(
                m_dense,
                vec32[..., None],
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                precision=dot_precision,
            )
            return out[..., 0].astype(jnp.float64)

        def solve_stage(m_dense, lu, rhs):
            rhs_pad = _pad_vector(rhs.astype(jnp.float32), n_pad)
            sol = _lu_solve(lu, rhs_pad)
            return sol[:, :n_vars].astype(jnp.float64)

        def step_fn(t_step, dt_step, y_step, k_store):
            m_dense = _stack_jacobian(jac_fn, t_step, jnp.float32)
            w = _pad_matrix(-m_dense, n_pad)
            diag_idx = jnp.arange(n_pad)
            diag_add = jnp.broadcast_to(
                (1.0 / (dt_step * _gamma)).astype(jnp.float32)[:, None],
                (w.shape[0], n_pad),
            )
            w = w.at[:, diag_idx, diag_idx].add(diag_add)
            lu = _blocked_lu_nopivot(w, precision=dot_precision)
            inv_dt = 1.0 / dt_step

            def assemble_stage_state(base_state, coeffs):
                out = base_state
                for coeff, stage in coeffs:
                    out = out + coeff * k_store[:, stage, :]
                return out

            def assemble_stage_rhs(f_val, coeffs):
                out = f_val
                for coeff, stage in coeffs:
                    out = out + coeff * k_store[:, stage, :] * inv_dt[:, None]
                return out

            f1 = eval_f(m_dense, y_step)
            k_store = k_store.at[:, 0, :].set(solve_stage(m_dense, lu, f1))

            u = y_step
            for stage_idx, (use_u_base, state_coeffs, rhs_coeffs) in enumerate(
                _STAGE_SPECS, start=1
            ):
                base = u if use_u_base else y_step
                u = assemble_stage_state(base, state_coeffs)
                f_stage = eval_f(m_dense, u)
                rhs = assemble_stage_rhs(f_stage, rhs_coeffs)
                k_store = k_store.at[:, stage_idx, :].set(solve_stage(m_dense, lu, rhs))

            u = assemble_stage_state(u, _FINAL_UPDATE)
            return u, k_store

        def cond_fn(state):
            t_c, _dt_c, save_idx_c, _y_c, _hist_c, n_c = state
            active = save_idx_c < n_save
            return (jnp.min(jnp.where(active, t_c, tf)) < tf) & (n_c < max_steps)

        def body_fn(state):
            t_c, dt_c, save_idx_c, y_c, hist_c, n_c = state
            active = save_idx_c < n_save
            next_target = times[save_idx_c]
            dt_use = jnp.where(
                active,
                jnp.maximum(jnp.minimum(dt_c, next_target - t_c), 1e-30),
                1e-30,
            )
            t_use = jnp.where(active, t_c, tf)
            k_init = jnp.zeros((n_batch, 8, n_vars), dtype=jnp.float64)
            u_c, k_out = step_fn(t_use, dt_use, y_c, k_init)

            sc = atol + rtol * jnp.maximum(jnp.abs(y_c), jnp.abs(u_c))
            e_est = jnp.sqrt(jnp.mean((k_out[:, 7, :] / sc) ** 2, axis=1))
            accept = active & (e_est <= 1.0) & ~jnp.isnan(e_est)

            y_new = jnp.where(accept[:, None], u_c, y_c)
            t_new = jnp.where(accept, t_c + dt_use, t_c)
            reached = accept & (
                jnp.abs(t_new - next_target)
                <= 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
            )
            slot_mask = (
                jax.nn.one_hot(save_idx_c, n_save, dtype=jnp.bool_) & reached[:, None]
            )
            hist_new = jnp.where(slot_mask[:, :, None], y_new[:, None, :], hist_c)
            save_idx_new = save_idx_c + reached.astype(jnp.int32)

            safe = jnp.where(
                jnp.isnan(e_est) | (e_est > 1e18),
                1e18,
                jnp.where(e_est == 0.0, 1e-18, e_est),
            )
            factor = jnp.clip(0.9 * safe ** (-1.0 / 6.0), 0.2, 6.0)
            dt_new = jnp.where(active, dt_use * factor, dt_c)
            return t_new, dt_new, save_idx_new, y_new, hist_new, n_c + 1

        init = (t, dt_v, save_idx, y, hist, jnp.int32(0))
        _, _, _, y_final, hist_final, _ = lax.while_loop(cond_fn, body_fn, init)
        return y_final, hist_final

    def solve_ensemble(
        y0_batch,
        t_span,
        *,
        rtol=1e-6,
        atol=1e-8,
        first_step=None,
        max_steps=100000,
    ):
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
        dt0 = float(
            first_step if first_step is not None else (times[-1] - times[0]) * 1e-6
        )
        y0 = jnp.asarray(y0_batch, dtype=jnp.float64)
        _, hist = _solve_impl(
            y0,
            times_jnp,
            n_save=int(times.size),
            max_steps=int(max_steps),
            dt0=dt0,
            rtol=float(rtol),
            atol=float(atol),
        )
        return hist

    return solve_ensemble
