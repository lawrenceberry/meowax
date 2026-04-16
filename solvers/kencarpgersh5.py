"""KenCarp5 solver — nonlinear IMEX ODE variant (split ode_fn path).

Accepts split ODE functions
``explicit_ode_fn(y, t, params) -> dy/dt`` and
``implicit_ode_fn(y, t, params) -> dy/dt``.

The implicit DIRK stage equation is solved via a batched Newton iteration,
using ``jax.jacfwd`` to form the Jacobian of the implicit part.  When
``linear=True``, the implicit RHS is assumed linear in ``y`` and each implicit
stage is solved in one LU-backed linear solve instead of Newton iteration.

Uses a single jax.lax.while_loop with the batch dimension inside the loop
body instead of vmap-over-while-loop.
"""

import functools
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

# fmt: off
_GAMMA = 41.0 / 200.0

_B_SOL = np.array([
    -872700587467.0 / 9133579230613.0,
    0.0,
    0.0,
    22348218063261.0 / 9555858737531.0,
    -1143369518992.0 / 8141816002931.0,
    -39379526789629.0 / 19018526304540.0,
    32727382324388.0 / 42900044865799.0,
    _GAMMA,
], dtype=np.float64)
_B_EMBEDDED = np.array([
    -975461918565.0 / 9796059967033.0,
    0.0,
    0.0,
    78070527104295.0 / 32432590147079.0,
    -548382580838.0 / 3424219808633.0,
    -33438840321285.0 / 15594753105479.0,
    3629800801594.0 / 4656183773603.0,
    4035322873751.0 / 18575991585200.0,
], dtype=np.float64)
_B_ERROR = _B_SOL - _B_EMBEDDED
_C = np.array([
    0.0,
    41.0 / 100.0,
    2935347310677.0 / 11292855782101.0,
    1426016391358.0 / 7196633302097.0,
    92.0 / 100.0,
    24.0 / 100.0,
    3.0 / 5.0,
    1.0,
], dtype=np.float64)

_A_EXPLICIT = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [41.0 / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [367902744464.0 / 2072280473677.0, 677623207551.0 / 8224143866563.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1268023523408.0 / 10340822734521.0, 0.0, 1029933939417.0 / 13636558850479.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [14463281900351.0 / 6315353703477.0, 0.0, 66114435211212.0 / 5879490589093.0, -54053170152839.0 / 4284798021562.0, 0.0, 0.0, 0.0, 0.0],
    [14090043504691.0 / 34967701212078.0, 0.0, 15191511035443.0 / 11219624916014.0, -18461159152457.0 / 12425892160975.0, -281667163811.0 / 9011619295870.0, 0.0, 0.0, 0.0],
    [19230459214898.0 / 13134317526959.0, 0.0, 21275331358303.0 / 2942455364971.0, -38145345988419.0 / 4862620318723.0, -1.0 / 8.0, -1.0 / 8.0, 0.0, 0.0],
    [-19977161125411.0 / 11928030595625.0, 0.0, -40795976796054.0 / 6384907823539.0, 177454434618887.0 / 12078138498510.0, 782672205425.0 / 8267701900261.0, -69563011059811.0 / 9646580694205.0, 7356628210526.0 / 4942186776405.0, 0.0],
], dtype=np.float64)

_A_IMPLICIT = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, _GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [41.0 / 400.0, -567603406766.0 / 11931857230679.0, _GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0],
    [683785636431.0 / 9252920307686.0, 0.0, -110385047103.0 / 1367015193373.0, _GAMMA, 0.0, 0.0, 0.0, 0.0],
    [3016520224154.0 / 10081342136671.0, 0.0, 30586259806659.0 / 12414158314087.0, -22760509404356.0 / 11113319521817.0, _GAMMA, 0.0, 0.0, 0.0],
    [218866479029.0 / 1489978393911.0, 0.0, 638256894668.0 / 5436446318841.0, -1179710474555.0 / 5321154724896.0, -60928119172.0 / 8023461067671.0, _GAMMA, 0.0, 0.0],
    [1020004230633.0 / 5715676835656.0, 0.0, 25762820946817.0 / 25263940353407.0, -2161375909145.0 / 9755907335909.0, -211217309593.0 / 5846859502534.0, -4269925059573.0 / 7827059040719.0, _GAMMA, 0.0],
    [_B_SOL[0], _B_SOL[1], _B_SOL[2], _B_SOL[3], _B_SOL[4], _B_SOL[5], _B_SOL[6], _GAMMA],
], dtype=np.float64)

_PREDICTOR = (
    np.array([1.0], dtype=np.float64),
    np.array([1.0 - _C[2] / _C[1], _C[2] / _C[1]], dtype=np.float64),
    np.array([1.0 - _C[3] / _C[1], _C[3] / _C[1], 0.0], dtype=np.float64),
    np.array([1.0 - _C[4] / _C[2], 0.0, _C[4] / _C[2], 0.0], dtype=np.float64),
    np.array([1.0 - _C[5] / _C[4], 0.0, 0.0, 0.0, _C[5] / _C[4]], dtype=np.float64),
    np.array([1.0 - _C[6] / _C[4], 0.0, 0.0, 0.0, _C[6] / _C[4], 0.0], dtype=np.float64),
    np.array([1.0], dtype=np.float64) * 0.0 + np.array([
        1.0 - _C[7] / _C[4], 0.0, 0.0, 0.0, _C[7] / _C[4], 0.0, 0.0
    ], dtype=np.float64),
)
# fmt: on

_SAFETY = 0.9
_FACTOR_MIN = 0.2
_FACTOR_MAX = 10.0
_NEWTON_MAX_ITERS = 10


def _row_partition(mask):
    """Return a stable inactive-first permutation and its inverse.

    Rows where ``mask`` is False come first; True (active) rows come last.
    """
    perm = jnp.argsort(mask.astype(jnp.int32), axis=-1, stable=True)
    inv_perm = jnp.argsort(perm, axis=-1)
    n_active = jnp.sum(mask, axis=-1, dtype=jnp.int32)
    return perm, inv_perm, n_active


def _permute_vector(vec, perm):
    return jnp.take_along_axis(vec, perm, axis=-1)


def _unpermute_vector(vec_perm, inv_perm):
    return jnp.take_along_axis(vec_perm, inv_perm, axis=-1)


def _permute_matrix(mat, perm):
    mat_rows = jnp.take_along_axis(mat, perm[..., :, None], axis=-2)
    return jnp.take_along_axis(mat_rows, perm[..., None, :], axis=-1)


def _make_reduced_implicit_solver(n_vars, lu_dtype):
    """Solve ``(I - coeff * J) x = rhs`` with a per-trajectory reduced LU.

    Rows where ``mask`` is False are inactive: their implicit Jacobian rows
    are zero, so ``x_i = rhs_i`` by direct substitution.  Active rows form a
    k×k sub-system solved by LU, with coupling from inactive columns included
    in the RHS.  ``lax.switch`` is used to select the right branch at runtime
    while keeping XLA slice sizes static.
    """

    @functools.partial(jax.vmap, in_axes=(0, 0, 0, 0))
    def _solve_single(jac_perm, rhs_perm, n_active, coeff):
        branches = []
        for active_size in range(n_vars + 1):
            n_inactive = n_vars - active_size

            def _branch(args, *, _k=active_size, _ni=n_inactive):
                jac_perm, rhs_perm, coeff = args
                if _k == 0:
                    return rhs_perm
                x_inactive = rhs_perm[:_ni]
                jac_an = jac_perm[_ni:, :_ni]
                jac_aa = jac_perm[_ni:, _ni:]
                mat_aa = jnp.eye(_k, dtype=lu_dtype) - coeff.astype(
                    lu_dtype
                ) * jac_aa.astype(lu_dtype)
                rhs_active = rhs_perm[_ni:] + coeff * (
                    jac_an.astype(jnp.float64) @ x_inactive.astype(jnp.float64)
                )
                lu_piv = jax.scipy.linalg.lu_factor(mat_aa)
                x_active = jax.scipy.linalg.lu_solve(
                    lu_piv, rhs_active.astype(lu_dtype)
                ).astype(jnp.float64)
                return jnp.concatenate((x_inactive, x_active), axis=0)

            branches.append(_branch)

        return jax.lax.switch(n_active, branches, (jac_perm, rhs_perm, coeff))

    def _solve_batched(jac, rhs, mask, coeff):
        perm, inv_perm, n_active = _row_partition(mask)
        jac_perm = _permute_matrix(jac, perm)
        rhs_perm = _permute_vector(rhs, perm)
        x_perm = _solve_single(jac_perm, rhs_perm, n_active, coeff)
        return _unpermute_vector(x_perm, inv_perm)

    return _solve_batched


def make_solver(
    explicit_ode_fn,
    implicit_ode_fn,
    lu_precision: Literal["fp32", "fp64"] = "fp64",
    linear: bool = False,
    batch_size=None,
):
    """Create a reusable KenCarp5 ensemble solver for split ODEs.

    When ``linear=True``, ``implicit_ode_fn`` is treated as linear in ``y`` for
    fixed ``(t, params)`` and each implicit stage uses a single LU factorization.
    """
    lu_dtype = jnp.float32 if lu_precision == "fp32" else jnp.float64

    _explicit_batched = jax.vmap(explicit_ode_fn)
    _implicit_batched = jax.vmap(implicit_ode_fn)
    _implicit_jac_fn = jax.jacfwd(implicit_ode_fn, argnums=0)
    _implicit_jac_batched = jax.vmap(_implicit_jac_fn)

    @functools.partial(jax.jit, static_argnames=("n_save", "max_steps"))
    def _solve_impl(
        y0_arr, params_batches, times, *, n_save, max_steps, dt0, rtol, atol
    ):
        n_vars = y0_arr.shape[0]
        bs = params_batches.shape[1]
        solve_row_masked = _make_reduced_implicit_solver(n_vars, lu_dtype)
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

            if linear:

                def _newton_stage(base, t_stage, dt, predictor):
                    gamma_dt = dt * _GAMMA
                    jac = _implicit_jac_batched(predictor, t_stage, params_batch)
                    mask = jnp.any(jac != 0.0, axis=2)
                    u_stage = solve_row_masked(jac, base, mask, gamma_dt)
                    fi_stage = _implicit_batched(u_stage, t_stage, params_batch)
                    failed = (
                        jnp.any(~jnp.isfinite(jac), axis=(1, 2))
                        | jnp.any(~jnp.isfinite(u_stage), axis=1)
                        | jnp.any(~jnp.isfinite(fi_stage), axis=1)
                    )
                    return u_stage, fi_stage, failed

            else:

                def _newton_stage(base, t_stage, dt, predictor):
                    gamma_dt = dt * _GAMMA

                    # Evaluate J at the predictor once to detect which rows have
                    # nonzero implicit coupling.  Rows with all-zero Jacobian rows
                    # are "inactive": their implicit RHS is u-independent, so they
                    # are resolved by direct substitution without entering Newton.
                    jac_pred = _implicit_jac_batched(predictor, t_stage, params_batch)
                    mask = jnp.any(jac_pred != 0.0, axis=2)  # (bs, n) per-trajectory

                    def direct_fn(_):
                        # All rows inactive: one f_impl eval, no LU, no loop.
                        fi = _implicit_batched(predictor, t_stage, params_batch)
                        u = base + gamma_dt[:, None] * fi
                        return u, fi, jnp.zeros((bs,), jnp.bool_)

                    def newton_fn(_):
                        def cond_fn(state):
                            _, converged, failed, it = state
                            return (jnp.any(~(converged | failed))) & (
                                it < _NEWTON_MAX_ITERS
                            )

                        def body_fn(state):
                            u, converged, failed, it = state
                            fi = _implicit_batched(u, t_stage, params_batch)
                            res = u - base - gamma_dt[:, None] * fi
                            jac = _implicit_jac_batched(u, t_stage, params_batch)
                            # Reduced solve: inactive rows → delta_i = res_i (direct),
                            # active rows → k×k Newton step with coupling correction.
                            delta = solve_row_masked(jac, res, mask, gamma_dt)
                            u_new = jnp.where(
                                (converged | failed)[:, None],
                                u,
                                u - delta,
                            )
                            scale = atol + rtol * jnp.maximum(
                                jnp.abs(u), jnp.abs(u_new)
                            )
                            delta_norm = jnp.sqrt(
                                jnp.mean((delta / scale) ** 2, axis=1)
                            )
                            invalid = (
                                jnp.any(~jnp.isfinite(u_new), axis=1)
                                | jnp.any(~jnp.isfinite(delta), axis=1)
                                | jnp.isnan(delta_norm)
                            )
                            converged_new = converged | ((delta_norm <= 1.0) & ~invalid)
                            failed_new = failed | invalid
                            return (u_new, converged_new, failed_new, it + 1)

                        init = (
                            predictor,
                            jnp.zeros((bs,), dtype=jnp.bool_),
                            jnp.zeros((bs,), dtype=jnp.bool_),
                            jnp.int32(0),
                        )
                        u_final, converged, failed, _ = jax.lax.while_loop(
                            cond_fn, body_fn, init
                        )
                        fi_final = _implicit_batched(u_final, t_stage, params_batch)
                        failed = failed | ~converged | jnp.any(
                            ~jnp.isfinite(fi_final), axis=1
                        )
                        return u_final, fi_final, failed

                    return jax.lax.cond(
                        jnp.any(mask), newton_fn, direct_fn, None
                    )

            def _step_batch(y, t, dt):
                dt_col = dt[:, None]
                stage_y = []
                stage_fe = []
                stage_fi = []
                failed = jnp.zeros((bs,), dtype=jnp.bool_)

                y_stage = y
                t_stage = t
                fe_stage = _explicit_batched(y_stage, t_stage, params_batch)
                fi_stage = _implicit_batched(y_stage, t_stage, params_batch)
                stage_y.append(y_stage)
                stage_fe.append(fe_stage)
                stage_fi.append(fi_stage)
                failed = (
                    failed
                    | jnp.any(~jnp.isfinite(fe_stage), axis=1)
                    | jnp.any(~jnp.isfinite(fi_stage), axis=1)
                )

                for i in range(1, 8):
                    t_stage = t + _C[i] * dt
                    base = y
                    for j in range(i):
                        ae = _A_EXPLICIT[i, j]
                        ai = _A_IMPLICIT[i, j]
                        if ae != 0.0:
                            base = base + dt_col * ae * stage_fe[j]
                        if ai != 0.0:
                            base = base + dt_col * ai * stage_fi[j]

                    predictor = jnp.zeros_like(y)
                    predictor_coeff = _PREDICTOR[i - 1]
                    for j, coeff in enumerate(predictor_coeff):
                        if coeff != 0.0:
                            predictor = predictor + coeff * stage_y[j]

                    y_stage, fi_stage, stage_failed = _newton_stage(
                        base, t_stage, dt, predictor
                    )
                    fe_stage = _explicit_batched(y_stage, t_stage, params_batch)
                    failed = (
                        failed
                        | stage_failed
                        | jnp.any(~jnp.isfinite(y_stage), axis=1)
                        | jnp.any(~jnp.isfinite(fe_stage), axis=1)
                    )
                    stage_y.append(y_stage)
                    stage_fe.append(fe_stage)
                    stage_fi.append(fi_stage)

                y_new = stage_y[-1]
                err_est = jnp.zeros_like(y)
                for i in range(8):
                    total_stage = stage_fe[i] + stage_fi[i]
                    err_est = err_est + dt_col * _B_ERROR[i] * total_stage
                failed = (
                    failed
                    | jnp.any(~jnp.isfinite(y_new), axis=1)
                    | jnp.any(~jnp.isfinite(err_est), axis=1)
                )
                return y_new, err_est, failed

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

                y_new, err_est, stage_failed = _step_batch(y, t, dt_use)
                scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
                err_norm = jnp.sqrt(jnp.mean((err_est / scale) ** 2, axis=1))

                accept = (
                    active & (err_norm <= 1.0) & ~jnp.isnan(err_norm) & ~stage_failed
                )
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
                    stage_failed | jnp.isnan(err_norm) | (err_norm > 1e18),
                    1e18,
                    jnp.where(err_norm == 0.0, 1e-18, err_norm),
                )
                factor = jnp.clip(
                    _SAFETY * safe_err ** (-1.0 / 5.0), _FACTOR_MIN, _FACTOR_MAX
                )
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
