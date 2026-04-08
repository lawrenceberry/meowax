"""Matrix-specialized Rodas5 ensemble ODE solver with Pallas GPU kernel.

This variant solves linear systems of the form dy/dt = M(t) y, where the
Jacobian callback returns the dense state matrix for the current step time.
Each trajectory carries its own time value, so the Jacobian is evaluated
per-trajectory and stored into ``m_ref`` in flattened row-major form.
"""

import functools

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402
from jax._src.pallas.triton import core as pltriton
from jax.experimental import pallas as pl

# fmt: off
_gamma = 0.19

_a21 = 2.0
_a31 = 3.040894194418781;  _a32 = 1.041747909077569
_a41 = 2.576417536461461;  _a42 = 1.622083060776640;  _a43 = -0.9089668560264532
_a51 = 2.760842080225597;  _a52 = 1.446624659844071;  _a53 = -0.3036980084553738;  _a54 = 0.2877498600325443
_a61 = -14.09640773051259; _a62 = 6.925207756232704;  _a63 = -41.47510893210728;   _a64 = 2.343771018586405;  _a65 = 24.13215229196062

_C21 = -10.31323885133993
_C31 = -21.04823117650003; _C32 = -7.234992135176716
_C41 = 32.22751541853323;  _C42 = -4.943732386540191;  _C43 = 19.44922031041879
_C51 = -20.69865579590063; _C52 = -8.816374604402768;  _C53 = 1.260436877740897;   _C54 = -0.7495647613787146
_C61 = -46.22004352711257; _C62 = -17.49534862857472;  _C63 = -289.6389582892057;  _C64 = 93.60855400400906;  _C65 = 318.3822534212147
_C71 = 34.20013733472935;  _C72 = -14.15535402717690;  _C73 = 57.82335640988400;   _C74 = 25.83362985412365;  _C75 = 1.408950972071624;  _C76 = -6.551835421242162
_C81 = 42.57076742291101;  _C82 = -13.80770672017997;  _C83 = 93.98938432427124;   _C84 = 18.77919633714503;  _C85 = -31.58359187223370;  _C86 = -6.685968952921985;  _C87 = -5.810979938412932
# fmt: on

_BLOCK = 32

_STAGE_SPECS = (
    (False, ((_a21, 0),), ((_C21, 0),)),
    (False, ((_a31, 0), (_a32, 1)), ((_C31, 0), (_C32, 1))),
    (False, ((_a41, 0), (_a42, 1), (_a43, 2)), ((_C41, 0), (_C42, 1), (_C43, 2))),
    (
        False,
        ((_a51, 0), (_a52, 1), (_a53, 2), (_a54, 3)),
        ((_C51, 0), (_C52, 1), (_C53, 2), (_C54, 3)),
    ),
    (
        False,
        ((_a61, 0), (_a62, 1), (_a63, 2), (_a64, 3), (_a65, 4)),
        ((_C61, 0), (_C62, 1), (_C63, 2), (_C64, 3), (_C65, 4)),
    ),
    (
        True,
        ((1.0, 5),),
        ((_C71, 0), (_C72, 1), (_C73, 2), (_C74, 3), (_C75, 4), (_C76, 5)),
    ),
    (
        True,
        ((1.0, 6),),
        ((_C81, 0), (_C82, 1), (_C83, 2), (_C84, 3), (_C85, 4), (_C86, 5), (_C87, 6)),
    ),
)

_FINAL_UPDATE = ((1.0, 7),)
_SUPPORTED_LINEAR_SOLVERS = ("fp64", "fp32")


def _pad_cols_pow2(n_cols):
    """Return the next power of 2 >= n_cols."""
    return 1 << (n_cols - 1).bit_length()


def _make_rodas5_step(jac_fn, n_vars, linear_solver):
    """Create one Rodas5 step specialized for dy/dt = M(t) y."""
    nv = n_vars
    use_fp32 = linear_solver == "fp32"

    def _m_idx(i, j):
        return i * nv + j

    def _k_idx(stage, i):
        return stage * nv + i

    def _tuple_select(vals, idx):
        return jax.lax.switch(
            idx,
            tuple((lambda _, v=v: v) for v in vals),
            operand=None,
        )

    def _eval_F(m_ref, src_ref, dst_ref):
        def row(i, carry):
            acc = src_ref.at[:, 0][...] * 0.0

            def col(j, acc):
                yj = src_ref.at[:, pl.ds(j, 1)][...][:, 0]
                mij = m_ref.at[:, pl.ds(_m_idx(i, j), 1)][...][:, 0]
                return acc + mij * yj

            acc = jax.lax.fori_loop(0, nv, col, acc)
            dst_ref.at[:, pl.ds(i, 1)][...] = acc[:, None]
            return carry

        jax.lax.fori_loop(0, nv, row, jnp.int32(0))

    def _copy_vec_to_stage(src_ref, k_ref, stage):
        off = stage * nv

        def copy_row(i, carry):
            val = src_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k_ref.at[:, pl.ds(off + i, 1)][...] = val[:, None]
            return carry

        jax.lax.fori_loop(0, nv, copy_row, jnp.int32(0))

    def _copy_vec_to_low(src_ref, dst_ref):
        def copy_row(i, carry):
            xi = src_ref.at[:, pl.ds(i, 1)][...][:, 0]
            dst_ref.at[:, pl.ds(i, 1)][...] = xi.astype(jnp.float32)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, copy_row, jnp.int32(0))

    def _copy_low_to_vec(src_ref, dst_ref):
        def copy_row(i, carry):
            xi = src_ref.at[:, pl.ds(i, 1)][...][:, 0]
            dst_ref.at[:, pl.ds(i, 1)][...] = xi.astype(jnp.float64)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, copy_row, jnp.int32(0))

    def _lu_solve_into_ref(w_ref, vec_ref):
        def fwd_row(i, carry):
            def fwd_col(j, carry):
                xi = vec_ref.at[:, pl.ds(i, 1)][...][:, 0]
                xj = vec_ref.at[:, pl.ds(j, 1)][...][:, 0]
                lij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0]
                vec_ref.at[:, pl.ds(i, 1)][...] = (xi - lij * xj)[:, None]
                return carry

            return jax.lax.fori_loop(0, i, fwd_col, carry)

        jax.lax.fori_loop(0, nv, fwd_row, jnp.int32(0))

        def bwd_row(i_rev, carry):
            i = nv - 1 - i_rev

            def bwd_col(j_off, carry):
                j = i + 1 + j_off
                xi = vec_ref.at[:, pl.ds(i, 1)][...][:, 0]
                xj = vec_ref.at[:, pl.ds(j, 1)][...][:, 0]
                uij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0]
                vec_ref.at[:, pl.ds(i, 1)][...] = (xi - uij * xj)[:, None]
                return carry

            jax.lax.fori_loop(0, nv - 1 - i, bwd_col, carry)
            xi = vec_ref.at[:, pl.ds(i, 1)][...][:, 0]
            uii = w_ref.at[:, pl.ds(i * nv + i, 1)][...][:, 0]
            vec_ref.at[:, pl.ds(i, 1)][...] = (xi / uii)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, bwd_row, jnp.int32(0))

    def _factor_w(m_ref, w_ref, d):
        def build_w_row(i_s, carry):
            def build_w_col(j_s, carry):
                mij = m_ref.at[:, pl.ds(_m_idx(i_s, j_s), 1)][...][:, 0]
                val = -mij + jnp.where(i_s == j_s, d, 0.0)
                w_ref.at[:, pl.ds(_m_idx(i_s, j_s), 1)][...] = val.astype(w_ref.dtype)[
                    :, None
                ]
                return carry

            return jax.lax.fori_loop(0, nv, build_w_col, carry)

        jax.lax.fori_loop(0, nv, build_w_row, jnp.int32(0))

        def lu_col(j, carry):
            pivot = w_ref.at[:, pl.ds(j * nv + j, 1)][...][:, 0]

            def lu_row(i, carry):
                l_ij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0] / pivot
                w_ref.at[:, pl.ds(i * nv + j, 1)][...] = l_ij[:, None]

                def lu_elem(k, carry):
                    ik = w_ref.at[:, pl.ds(i * nv + k, 1)][...][:, 0]
                    jk = w_ref.at[:, pl.ds(j * nv + k, 1)][...][:, 0]
                    w_ref.at[:, pl.ds(i * nv + k, 1)][...] = (ik - l_ij * jk)[:, None]
                    return carry

                return jax.lax.fori_loop(j + 1, nv, lu_elem, carry)

            return jax.lax.fori_loop(j + 1, nv, lu_row, carry)

        jax.lax.fori_loop(0, nv, lu_col, jnp.int32(0))

    def _solve_linear_system(m_ref, w_ref, x_ref, x_low_ref, k_ref, u_ref, d, stage):
        del m_ref, u_ref, d
        if not use_fp32:
            _lu_solve_into_ref(w_ref, x_ref)
            _copy_vec_to_stage(x_ref, k_ref, stage)
            return

        _copy_vec_to_low(x_ref, x_low_ref)
        _lu_solve_into_ref(w_ref, x_low_ref)
        _copy_low_to_vec(x_low_ref, x_ref)
        _copy_vec_to_stage(x_ref, k_ref, stage)

    def step(t, dt, m_ref, w_ref, x_ref, x_low_ref, y_ref, k_ref, u_ref):
        dtgamma = dt * _gamma
        inv_dt = 1.0 / dt
        d = 1.0 / dtgamma
        m = jac_fn(t)

        def store_m_row(i_s, carry):
            row_i = _tuple_select(m, i_s)

            def store_m_col(j_s, carry):
                mij = _tuple_select(row_i, j_s)
                m_ref.at[:, pl.ds(_m_idx(i_s, j_s), 1)][...] = mij.astype(m_ref.dtype)[
                    :, None
                ]
                return carry

            return jax.lax.fori_loop(0, nv, store_m_col, carry)

        jax.lax.fori_loop(0, nv, store_m_row, jnp.int32(0))
        _factor_w(m_ref, w_ref, d)

        def _assemble_stage_state(base_ref, coeffs):
            def write_u(i, carry):
                ui = base_ref.at[:, pl.ds(i, 1)][...][:, 0]
                for coeff, stage in coeffs:
                    ki = k_ref.at[:, pl.ds(_k_idx(stage, i), 1)][...][:, 0]
                    ui = ui + coeff * ki
                u_ref.at[:, pl.ds(i, 1)][...] = ui[:, None]
                return carry

            jax.lax.fori_loop(0, nv, write_u, jnp.int32(0))

        def _assemble_stage_rhs(coeffs):
            def write_x(i, carry):
                xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
                for coeff, stage in coeffs:
                    ki = k_ref.at[:, pl.ds(_k_idx(stage, i), 1)][...][:, 0]
                    xi = xi + coeff * ki * inv_dt
                x_ref.at[:, pl.ds(i, 1)][...] = xi[:, None]
                return carry

            jax.lax.fori_loop(0, nv, write_x, jnp.int32(0))

        _eval_F(m_ref, y_ref, x_ref)
        _solve_linear_system(m_ref, w_ref, x_ref, x_low_ref, k_ref, u_ref, d, 0)

        for stage, (use_u_base, state_coeffs, rhs_coeffs) in enumerate(
            _STAGE_SPECS, start=1
        ):
            base_ref = u_ref if use_u_base else y_ref
            _assemble_stage_state(base_ref, state_coeffs)
            _eval_F(m_ref, u_ref, x_ref)
            _assemble_stage_rhs(rhs_coeffs)
            _solve_linear_system(m_ref, w_ref, x_ref, x_low_ref, k_ref, u_ref, d, stage)

        _assemble_stage_state(u_ref, _FINAL_UPDATE)

    return step


def make_solver(jac_fn, *, linear_solver="fp64"):
    """Create a Pallas ensemble Rodas5 solver for dy/dt = M(t) y.

    Args:
        jac_fn: ``t -> M(t)`` returning a square dense Jacobian/state matrix.
        linear_solver: ``"fp64"`` or ``"fp32"`` for the LU solve precision.
    """
    if linear_solver not in _SUPPORTED_LINEAR_SOLVERS:
        raise ValueError(
            f"linear_solver must be one of {_SUPPORTED_LINEAR_SOLVERS}, got {linear_solver!r}"
        )
    M0 = jnp.asarray(jac_fn(jnp.float64(0.0)), dtype=jnp.float64)
    if M0.ndim != 2 or M0.shape[0] != M0.shape[1]:
        raise ValueError(
            f"jac_fn(0.0) must be square with shape (n, n), got {M0.shape}"
        )

    n_vars = int(M0.shape[0])

    @functools.partial(
        jax.jit,
        static_argnames=(
            "n_pad",
            "y_cols",
            "w_cols",
            "x_cols",
            "k_cols",
            "t0",
            "tf",
            "dt0",
            "r_tol",
            "a_tol",
            "ms",
            "n_save",
        ),
    )
    def _rodas5_pallas_solve(
        times_arr,
        y0_arr,
        *,
        n_pad,
        y_cols,
        w_cols,
        x_cols,
        k_cols,
        n_save,
        t0,
        tf,
        dt0,
        r_tol,
        a_tol,
        ms,
    ):
        step_fn = _make_rodas5_step(jac_fn, n_vars, linear_solver)
        hist_cols = n_save * y_cols
        w_dtype = jnp.float32 if linear_solver == "fp32" else jnp.float64
        m_dtype = jnp.float32 if linear_solver == "fp32" else jnp.float64

        def kernel_body(
            times_ref,
            y0_ref,
            hist_ref,
            y_ref,
            w_ref,
            x_ref,
            x_low_ref,
            k_ref,
            u_ref,
            m_ref,
        ):
            for i in range(n_vars):
                y_ref.at[:, i][...] = y0_ref.at[:, i][...]

            z = y0_ref.at[:, 0][...] * 0.0
            save_idx0 = z.astype(jnp.int32) + 1
            t = z + t0
            dt_v = z + dt0

            def init_hist(col, carry):
                hist_ref.at[:, pl.ds(col, 1)][...] = z[:, None]
                return carry

            jax.lax.fori_loop(0, hist_cols, init_hist, jnp.int32(0))

            def save_initial(i, carry):
                yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
                hist_ref.at[:, pl.ds(i, 1)][...] = yi[:, None]
                return carry

            jax.lax.fori_loop(0, n_vars, save_initial, jnp.int32(0))

            def cond_fn(state):
                t, _dt_v, save_idx, n = state
                active = save_idx < n_save
                return (jnp.min(jnp.where(active, t, tf)) < tf) & (n < ms)

            def body_fn(state):
                t, dt_v, save_idx, n = state

                active = save_idx < n_save

                def select_target(s, target):
                    ts = times_ref.at[pl.ds(s, 1)][...][0]
                    return jnp.where(save_idx == s, ts, target)

                next_target = jax.lax.fori_loop(0, n_save, select_target, z + tf)
                dt_cap = next_target - t
                dt_use = jnp.where(
                    active,
                    jnp.maximum(jnp.minimum(dt_v, dt_cap), 1e-30),
                    1e-30,
                )
                t_use = jnp.where(active, t, tf)

                step_fn(
                    t_use,
                    dt_use,
                    m_ref,
                    w_ref,
                    x_ref,
                    x_low_ref,
                    y_ref,
                    k_ref,
                    u_ref,
                )

                def compute_err_sq(i, acc):
                    yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    k8i = k_ref.at[:, pl.ds(7 * n_vars + i, 1)][...][:, 0]
                    sc = a_tol + r_tol * jnp.maximum(jnp.abs(yi), jnp.abs(ui))
                    return acc + (k8i / sc) ** 2

                err_sq = jax.lax.fori_loop(0, n_vars, compute_err_sq, jnp.zeros_like(t))
                EEst = jnp.sqrt(err_sq / n_vars)

                accept = (EEst <= 1.0) & ~jnp.isnan(EEst)
                mask = active & accept

                t_new = jnp.where(mask, t + dt_use, t)

                def update_y(i, carry):
                    yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    y_ref.at[:, pl.ds(i, 1)][...] = jnp.where(mask, ui, yi)[:, None]
                    return carry

                jax.lax.fori_loop(0, n_vars, update_y, jnp.int32(0))

                target_tol = 1e-12 * jnp.maximum(1.0, jnp.abs(next_target))
                reached_target = mask & (jnp.abs(t_new - next_target) <= target_tol)

                def write_slot(s, save_idx_carry):
                    write_mask = reached_target & (save_idx == s)

                    def write_var(i, carry):
                        yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
                        hist_col = s * y_cols + i
                        prev = hist_ref.at[:, pl.ds(hist_col, 1)][...][:, 0]
                        hist_ref.at[:, pl.ds(hist_col, 1)][...] = jnp.where(
                            write_mask, yi, prev
                        )[:, None]
                        return carry

                    jax.lax.fori_loop(0, n_vars, write_var, jnp.int32(0))
                    return jnp.where(write_mask, save_idx_carry + 1, save_idx_carry)

                save_idx_new = jax.lax.fori_loop(0, n_save, write_slot, save_idx)

                safe_EEst = jnp.where(
                    jnp.isnan(EEst) | (EEst > 1e18),
                    1e18,
                    jnp.where(EEst == 0.0, 1e-18, EEst),
                )
                factor = jnp.clip(0.9 * safe_EEst ** (-1.0 / 6.0), 0.2, 6.0)
                new_dt = jnp.where(active, dt_use * factor, dt_v)

                return (t_new, new_dt, save_idx_new, n + 1)

            jax.lax.while_loop(cond_fn, body_fn, (t, dt_v, save_idx0, jnp.int32(0)))

        times_bs = pl.BlockSpec((n_save,), lambda i: (0,))
        hist_bs = pl.BlockSpec((_BLOCK, hist_cols), lambda i: (i, 0))
        y_bs = pl.BlockSpec((_BLOCK, y_cols), lambda i: (i, 0))
        w_bs = pl.BlockSpec((_BLOCK, w_cols), lambda i: (i, 0))
        x_bs = pl.BlockSpec((_BLOCK, x_cols), lambda i: (i, 0))
        x_low_bs = pl.BlockSpec((_BLOCK, x_cols), lambda i: (i, 0))
        k_bs = pl.BlockSpec((_BLOCK, k_cols), lambda i: (i, 0))
        m_cols = _pad_cols_pow2(n_vars * n_vars)
        m_bs = pl.BlockSpec((_BLOCK, m_cols), lambda i: (i, 0))
        results = pl.pallas_call(
            kernel_body,
            out_shape=[
                jax.ShapeDtypeStruct((n_pad, hist_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, y_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, w_cols), w_dtype),
                jax.ShapeDtypeStruct((n_pad, x_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, x_cols), jnp.float32),
                jax.ShapeDtypeStruct((n_pad, k_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, x_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, m_cols), m_dtype),
            ],
            grid=(n_pad // _BLOCK,),
            in_specs=(times_bs, y_bs),
            # Output refs match kernel_body(times_ref, y0_ref, hist_ref, y_ref, w_ref, x_ref, x_low_ref, k_ref, u_ref, m_ref):
            # hist_bs -> saved states over the requested time array
            # y_bs -> accepted state y
            # w_bs -> LU workspace for W = I/(dt*gamma) - M(t), dtype depends on linear_solver
            # x_bs -> stage RHS / solve workspace x (float64)
            # x_low_bs -> low-precision solve / correction workspace (float32)
            # k_bs -> stacked Rodas5 stage increments k1..k8
            # x_bs -> temporary stage state u (float64)
            # m_bs -> per-trajectory flattened Jacobian M(t)
            out_specs=(hist_bs, y_bs, w_bs, x_bs, x_low_bs, k_bs, x_bs, m_bs),
            compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
        )(times_arr, y0_arr)
        return jnp.reshape(results[0], (n_pad, n_save, y_cols))

    def solve_ensemble_pallas(
        y0_batch,
        t_span,
        *,
        rtol=1e-6,
        atol=1e-8,
        first_step=None,
        max_steps=100000,
    ):
        """Solve ensemble dy/dt = M(t) y with Rodas5 Pallas custom kernel."""
        if y0_batch.shape[1] != n_vars:
            raise ValueError(
                "y0_batch has "
                f"{y0_batch.shape[1]} vars but jac_fn(0.0) is {n_vars}x{n_vars}"
            )

        times_arr = np.asarray(t_span, dtype=np.float64)
        if times_arr.ndim != 1:
            raise ValueError(
                f"t_span must be a 1D array of save times, got shape {times_arr.shape}"
            )
        if times_arr.size < 2:
            raise ValueError(
                f"t_span must contain at least 2 save times, got {times_arr.size}"
            )
        if not np.all(np.diff(times_arr) > 0.0):
            raise ValueError("t_span must be strictly increasing")

        N = y0_batch.shape[0]
        N_pad = ((N + _BLOCK - 1) // _BLOCK) * _BLOCK

        tf = float(times_arr[-1])
        t0 = float(times_arr[0])
        dt0 = float(first_step if first_step is not None else (tf - t0) * 1e-6)
        n_save = int(times_arr.size)

        y_cols = _pad_cols_pow2(n_vars)
        w_cols = _pad_cols_pow2(n_vars * n_vars)
        x_cols = _pad_cols_pow2(n_vars)
        k_cols = _pad_cols_pow2(8 * n_vars)

        y0_arr = jnp.pad(y0_batch, ((0, N_pad - N), (0, y_cols - n_vars)))
        times_jnp = jnp.asarray(times_arr, dtype=jnp.float64)

        y_hist = _rodas5_pallas_solve(
            times_jnp,
            y0_arr,
            n_pad=N_pad,
            y_cols=y_cols,
            w_cols=w_cols,
            x_cols=x_cols,
            k_cols=k_cols,
            n_save=n_save,
            t0=t0,
            tf=tf,
            dt0=dt0,
            r_tol=float(rtol),
            a_tol=float(atol),
            ms=int(max_steps),
        )

        return y_hist[:N, :, :n_vars]

    return solve_ensemble_pallas
