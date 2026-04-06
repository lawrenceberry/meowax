"""Rodas5 solver — single-loop batched variant.

Same Rodas5 math and float64 precision as rodas5.py, but solve_ensemble
uses a single jax.lax.while_loop with the batch dimension inside the loop
body instead of vmap-over-while-loop.

The batch_size parameter controls how many trajectories share a while loop.
batch_size=N (default) puts all trajectories in one loop; batch_size=1
recovers the vmap-over-while-loop behaviour of rodas5.py.
"""

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402

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


def solve_ensemble(
    f,
    y0,
    t_span,
    params_batch,
    *,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    batch_size=None,
):
    """Solve an ensemble of ODEs with configurable batch grouping.

    Trajectories are split into chunks of ``batch_size``.  Within each chunk
    a single ``jax.lax.while_loop`` advances all trajectories together;
    chunks are parallelised with ``jax.vmap``.

    * ``batch_size=None`` (default) — all N trajectories in one loop.
    * ``batch_size=1`` — equivalent to ``vmap``-over-``while_loop``
      (same behaviour as ``rodas5.py``).

    Args:
        f: JAX function (y, params) -> dy/dt.
        y0: Initial conditions, shared across ensemble.
        t_span: Tuple (t0, tf) for integration bounds.
        params_batch: Array of shape (n_ensemble, ...) with parameters.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        first_step: Initial step size (optional).
        max_steps: Maximum number of steps.
        batch_size: Number of trajectories per while-loop chunk (default: N).

    Returns:
        Array of shape (n_ensemble, n_components) with final states.
    """
    y0_arr = jnp.asarray(y0, dtype=jnp.float64)
    n_vars = y0_arr.shape[0]
    N = params_batch.shape[0]
    t0, tf = t_span
    dt0 = jnp.float64(first_step if first_step is not None else (tf - t0) * 1e-6)
    bs = N if batch_size is None else batch_size

    # Batched helpers (all vmap over the chunk dimension)
    f_batched = jax.vmap(f)

    def _jac_one(y, p):
        return jax.jacobian(lambda y_: f(y_, p))(y)

    jac_batched = jax.vmap(_jac_one)
    lu_factor_batched = jax.vmap(jax.scipy.linalg.lu_factor)
    lu_solve_batched = jax.vmap(jax.scipy.linalg.lu_solve)

    def _step_batch(y, dt, params):
        """Batched Rodas5 step: y (bs, n), dt (bs,), params (bs, ...)."""
        J = jac_batched(y, params)
        dtgamma = dt * _gamma
        W = jnp.eye(n_vars)[None] / dtgamma[:, None, None] - J
        LU_piv = lu_factor_batched(W)
        inv_dt = (1.0 / dt)[:, None]

        def f_eval(u):
            return f_batched(u, params)

        def lu_solve(rhs):
            return lu_solve_batched(LU_piv, rhs)

        # Stage 1
        dy = f_eval(y)
        k1 = lu_solve(dy)

        # Stage 2
        u = y + _a21 * k1
        du = f_eval(u)
        k2 = lu_solve(du + _C21 * k1 * inv_dt)

        # Stage 3
        u = y + _a31 * k1 + _a32 * k2
        du = f_eval(u)
        k3 = lu_solve(du + (_C31 * k1 + _C32 * k2) * inv_dt)

        # Stage 4
        u = y + _a41 * k1 + _a42 * k2 + _a43 * k3
        du = f_eval(u)
        k4 = lu_solve(du + (_C41 * k1 + _C42 * k2 + _C43 * k3) * inv_dt)

        # Stage 5
        u = y + _a51 * k1 + _a52 * k2 + _a53 * k3 + _a54 * k4
        du = f_eval(u)
        k5 = lu_solve(du + (_C51 * k1 + _C52 * k2 + _C53 * k3 + _C54 * k4) * inv_dt)

        # Stage 6
        u = y + _a61 * k1 + _a62 * k2 + _a63 * k3 + _a64 * k4 + _a65 * k5
        du = f_eval(u)
        k6 = lu_solve(
            du + (_C61 * k1 + _C62 * k2 + _C63 * k3 + _C64 * k4 + _C65 * k5) * inv_dt
        )

        # Stage 7
        u = u + k6
        du = f_eval(u)
        k7 = lu_solve(
            du
            + (_C71 * k1 + _C72 * k2 + _C73 * k3 + _C74 * k4 + _C75 * k5 + _C76 * k6)
            * inv_dt
        )

        # Stage 8
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

        y_new = u + k8
        return y_new, k8

    def _solve_chunk(params_chunk):
        """Run a single while-loop for a chunk of ``bs`` trajectories."""

        def cond_fn(state):
            t, _, _, n_steps = state
            return jnp.any(t < tf) & (n_steps < max_steps)

        def body_fn(state):
            t, y, dt, n_steps = state
            active = t < tf
            dt_use = jnp.where(active, jnp.minimum(dt, tf - t), 1e-30)

            y_new, err_est = _step_batch(y, dt_use, params_chunk)

            scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
            err_norm = jnp.sqrt(jnp.mean((err_est / scale) ** 2, axis=1))

            accept = active & (err_norm <= 1.0) & ~jnp.isnan(err_norm)
            t_new = jnp.where(accept, t + dt_use, t)
            y_out = jnp.where(accept[:, None], y_new, y)

            safe_err = jnp.where(
                jnp.isnan(err_norm) | (err_norm > 1e18),
                1e18,
                jnp.where(err_norm == 0.0, 1e-18, err_norm),
            )
            factor = jnp.clip(0.9 * safe_err ** (-1.0 / 6.0), 0.2, 6.0)
            dt_new = jnp.where(active, dt_use * factor, dt)

            return (t_new, y_out, dt_new, n_steps + 1)

        y_init = jnp.broadcast_to(y0_arr, (bs, n_vars)).copy()
        t_init = jnp.full((bs,), t0, dtype=jnp.float64)
        dt_init = jnp.full((bs,), dt0, dtype=jnp.float64)

        init = (t_init, y_init, dt_init, jnp.int32(0))
        _, final_y, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return final_y

    # Pad N to a multiple of bs, split into chunks, vmap, trim.
    n_chunks = (N + bs - 1) // bs
    n_padded = n_chunks * bs

    if n_padded > N:
        # Repeat last row so padded trajectories have valid params
        pad_rows = jnp.broadcast_to(
            params_batch[-1:], (n_padded - N,) + params_batch.shape[1:]
        )
        params_padded = jnp.concatenate([params_batch, pad_rows], axis=0)
    else:
        params_padded = params_batch

    params_groups = params_padded.reshape((n_chunks, bs) + params_batch.shape[1:])
    results = jax.jit(jax.vmap(_solve_chunk))(params_groups)  # (n_chunks, bs, n_vars)
    return results.reshape(n_padded, n_vars)[:N]
