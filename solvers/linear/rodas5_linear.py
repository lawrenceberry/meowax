"""Rodas5 solver — linear ODE variant (jac_fn path).

Accepts a ``jac_fn(t, params) -> [n_vars, n_vars]`` whose Jacobian depends on
time and parameters but NOT on the current state y.  f_eval is computed as a
matrix-vector product using the supplied Jacobian.

Uses a single jax.lax.while_loop with the batch dimension inside the loop
body instead of vmap-over-while-loop.

The batch_size parameter controls how many trajectories share a while loop.
batch_size=N (default) puts all trajectories in one loop; batch_size=1
recovers the vmap-over-while-loop behaviour.
"""

import functools
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

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

_BATCHED_DOT_DIMENSION_NUMBERS = (((2,), (1,)), ((0,), (0,)))


def _batched_matvec(mat, vec, *, precision):
    out = lax.dot(
        mat,
        vec[..., None],
        dimension_numbers=_BATCHED_DOT_DIMENSION_NUMBERS,
        precision=precision,
    )
    return out[..., 0]


def make_solver(
    jac_fn,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    mv_precision: Literal["fp32", "fp64"] = "fp64",
    batch_size=None,
):
    """Create a reusable Rodas5 ensemble solver for linear ODEs.

    Parameters
    ----------
    jac_fn : callable
        Jacobian function with signature ``jac_fn(t, params) -> [n_vars, n_vars]``.
        The Jacobian may depend on time and parameters but must not depend on
        the current state y.  f_eval is implemented as a matrix-vector product.
    lu_precision :
        Precision for LU factorization and LU solve: ``"fp32"`` or ``"fp64"``.
    mv_precision :
        Precision for matrix-vector products: ``"fp32"`` or ``"fp64"``.
    batch_size : int or None
        Number of trajectories per while-loop batch.  ``None`` (default)
        puts all trajectories in a single loop.
    """
    lu_dtype = jnp.float32 if lu_precision == "fp32" else jnp.float64
    mv_dtype = jnp.float32 if mv_precision == "fp32" else jnp.float64

    lu_factor_batched = jax.vmap(jax.scipy.linalg.lu_factor)
    lu_solve_batched = jax.vmap(jax.scipy.linalg.lu_solve)

    _jac_batched = jax.vmap(lambda _, t, p: jac_fn(t, p), in_axes=(0, 0, 0))

    @functools.partial(
        jax.jit,
        static_argnames=("n_save", "max_steps"),
    )
    def _solve_impl(
        y0_arr, params_batches, times, *, n_save, max_steps, dt0, rtol, atol
    ):
        n_vars = y0_arr.shape[0]
        bs = params_batches.shape[1]
        eye = jnp.eye(n_vars, dtype=lu_dtype)[None, :, :]
        tf = times[-1]

        def _solve_batch(params_batch):
            y_init = jnp.broadcast_to(y0_arr, (bs, n_vars)).copy()
            hist_init = (
                jnp.zeros((bs, n_save, n_vars), dtype=jnp.float64)
                .at[:, 0, :]
                .set(y_init)
            )
            t_init = jnp.full((bs,), times[0], dtype=jnp.float64)
            dt_init = jnp.full((bs,), dt0, dtype=jnp.float64)
            save_idx_init = jnp.ones((bs,), dtype=jnp.int32)

            def _step_batch(y, t, dt):
                jac = _jac_batched(y, t, params_batch)
                jac_lu = jac.astype(lu_dtype)
                jac_mv = jac.astype(mv_dtype)
                dtgamma_inv = (1.0 / (dt * _gamma)).astype(lu_dtype)[:, None, None]
                lu = lu_factor_batched(dtgamma_inv * eye - jac_lu)
                inv_dt = (1.0 / dt)[:, None]

                def f_eval(u):
                    out = _batched_matvec(
                        jac_mv,
                        u.astype(mv_dtype),
                        precision=lax.DotAlgorithmPreset.TF32_TF32_F32
                        if mv_precision == "fp32" and jax.default_backend() != "cpu"
                        else lax.Precision.DEFAULT,
                    )
                    return out.astype(jnp.float64)

                def lu_solve(rhs):
                    sol = lu_solve_batched(lu, rhs.astype(lu_dtype))
                    return sol.astype(jnp.float64)

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
                k5 = lu_solve(
                    du + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt
                )

                u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
                du = f_eval(u)
                k6 = lu_solve(
                    du
                    + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5)
                    * inv_dt
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

                y_new, err_est = _step_batch(y, t, dt_use)

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
            _, _, _, hist_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
            return hist_final

        return jax.vmap(_solve_batch)(params_batches)

    def _solve(
        y0,
        t_span,
        params,
        *,
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
    ):
        y0_arr = jnp.asarray(y0, dtype=jnp.float64)
        params_arr = jnp.asarray(params)
        n_vars = int(y0_arr.shape[0])
        N = int(params_arr.shape[0])
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
                params_arr[-1:],
                (n_padded - N,) + params_arr.shape[1:],
            )
            params_padded = jnp.concatenate([params_arr, pad_rows], axis=0)
        else:
            params_padded = params_arr

        params_batches = params_padded.reshape((n_chunks, bs) + params_arr.shape[1:])
        results = _solve_impl(
            y0_arr,
            params_batches,
            times_jnp,
            n_save=int(times.size),
            max_steps=int(max_steps),
            dt0=float(dt0),
            rtol=float(rtol),
            atol=float(atol),
        )
        return results.reshape(n_padded, int(times.size), n_vars)[:N]

    return _solve
