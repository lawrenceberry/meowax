"""Matrix-specialized Rodas5 ensemble ODE solver with Pallas GPU kernel.

This variant solves linear systems of the form dy/dt = M y, where M is fixed
for the entire ensemble. The Jacobian is exactly M, so each step builds
W = I/(dt*gamma) - M directly and avoids JVP/Jacobian tracing.
"""

import functools

import jax

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


def _pad_cols_pow2(n_cols):
    """Return the next power of 2 >= n_cols."""
    return 1 << (n_cols - 1).bit_length()


def _make_rodas5_step(n_vars):
    """Create one Rodas5 step specialized for dy/dt = M y."""
    nv = n_vars

    def _eval_F(m_ref, src_ref, dst_ref):
        def row(i, carry):
            acc = src_ref.at[:, 0][...] * 0.0

            def col(j, acc):
                yj = src_ref.at[:, pl.ds(j, 1)][...][:, 0]
                mij = m_ref.at[pl.ds(i, 1), pl.ds(j, 1)][...][0, 0]
                return acc + mij * yj

            acc = jax.lax.fori_loop(0, nv, col, acc)
            dst_ref.at[:, pl.ds(i, 1)][...] = acc[:, None]
            return carry

        jax.lax.fori_loop(0, nv, row, jnp.int32(0))

    def _lu_solve(w_ref, x_ref, k_ref, stage):
        def fwd_row(i, carry):
            def fwd_col(j, carry):
                xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
                xj = x_ref.at[:, pl.ds(j, 1)][...][:, 0]
                lij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0]
                x_ref.at[:, pl.ds(i, 1)][...] = (xi - lij * xj)[:, None]
                return carry

            return jax.lax.fori_loop(0, i, fwd_col, carry)

        jax.lax.fori_loop(0, nv, fwd_row, jnp.int32(0))

        def bwd_row(i_rev, carry):
            i = nv - 1 - i_rev

            def bwd_col(j_off, carry):
                j = i + 1 + j_off
                xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
                xj = x_ref.at[:, pl.ds(j, 1)][...][:, 0]
                uij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0]
                x_ref.at[:, pl.ds(i, 1)][...] = (xi - uij * xj)[:, None]
                return carry

            jax.lax.fori_loop(0, nv - 1 - i, bwd_col, carry)
            xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            uii = w_ref.at[:, pl.ds(i * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (xi / uii)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, bwd_row, jnp.int32(0))

        off = stage * nv

        def copy_k(i, carry):
            val = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k_ref.at[:, pl.ds(off + i, 1)][...] = val[:, None]
            return carry

        jax.lax.fori_loop(0, nv, copy_k, jnp.int32(0))

    def step(m_ref, dt, w_ref, x_ref, y_ref, k_ref, u_ref):
        dtgamma = dt * _gamma
        inv_dt = 1.0 / dt
        d = 1.0 / dtgamma

        def build_w_row(i_s, carry):
            def build_w_col(j_s, carry):
                mij = m_ref.at[pl.ds(i_s, 1), pl.ds(j_s, 1)][...][0, 0]
                val = -mij + jnp.where(i_s == j_s, d, 0.0)
                w_ref.at[:, pl.ds(i_s * nv + j_s, 1)][...] = val[:, None]
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

        _eval_F(m_ref, y_ref, x_ref)
        _lu_solve(w_ref, x_ref, k_ref, 0)

        def s2_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a21 * k1i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s2_u, jnp.int32(0))
        _eval_F(m_ref, u_ref, x_ref)

        def s2_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (fi + _C21 * k1i * inv_dt)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s2_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 1)

        def s3_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a31 * k1i + _a32 * k2i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s3_u, jnp.int32(0))
        _eval_F(m_ref, u_ref, x_ref)

        def s3_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (fi + (_C31 * k1i + _C32 * k2i) * inv_dt)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s3_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 2)

        def s4_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a41 * k1i + _a42 * k2i + _a43 * k3i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s4_u, jnp.int32(0))
        _eval_F(m_ref, u_ref, x_ref)

        def s4_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (fi + (_C41 * k1i + _C42 * k2i + _C43 * k3i) * inv_dt)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s4_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 3)

        def s5_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a51 * k1i + _a52 * k2i + _a53 * k3i + _a54 * k4i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s5_u, jnp.int32(0))
        _eval_F(m_ref, u_ref, x_ref)

        def s5_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (fi + (_C51 * k1i + _C52 * k2i + _C53 * k3i + _C54 * k4i) * inv_dt)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s5_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 4)

        def s6_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            k5i = k_ref.at[:, pl.ds(4 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a61 * k1i + _a62 * k2i + _a63 * k3i + _a64 * k4i + _a65 * k5i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s6_u, jnp.int32(0))
        _eval_F(m_ref, u_ref, x_ref)

        def s6_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            k5i = k_ref.at[:, pl.ds(4 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (
                fi + (_C61 * k1i + _C62 * k2i + _C63 * k3i + _C64 * k4i + _C65 * k5i) * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s6_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 5)

        def s7_u(i, carry):
            ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k6i = k_ref.at[:, pl.ds(5 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (ui + k6i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s7_u, jnp.int32(0))
        _eval_F(m_ref, u_ref, x_ref)

        def s7_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            k5i = k_ref.at[:, pl.ds(4 * nv + i, 1)][...][:, 0]
            k6i = k_ref.at[:, pl.ds(5 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (
                fi + (_C71 * k1i + _C72 * k2i + _C73 * k3i + _C74 * k4i + _C75 * k5i + _C76 * k6i) * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s7_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 6)

        def s8_u(i, carry):
            ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k7i = k_ref.at[:, pl.ds(6 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (ui + k7i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s8_u, jnp.int32(0))
        _eval_F(m_ref, u_ref, x_ref)

        def s8_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            k5i = k_ref.at[:, pl.ds(4 * nv + i, 1)][...][:, 0]
            k6i = k_ref.at[:, pl.ds(5 * nv + i, 1)][...][:, 0]
            k7i = k_ref.at[:, pl.ds(6 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (
                fi
                + (_C81 * k1i + _C82 * k2i + _C83 * k3i + _C84 * k4i + _C85 * k5i + _C86 * k6i + _C87 * k7i)
                * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s8_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 7)

        def ynew(i, carry):
            ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k8i = k_ref.at[:, pl.ds(7 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (ui + k8i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, ynew, jnp.int32(0))

    return step


def make_solver(M):
    """Create a Pallas ensemble Rodas5 solver for dy/dt = M y.

    Args:
        M: Constant Jacobian/state matrix with shape (n_vars, n_vars).
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square with shape (n, n), got {M.shape}")

    n_vars = int(M.shape[0])

    @functools.partial(
        jax.jit,
        static_argnames=(
            "n_pad",
            "y_cols",
            "w_cols",
            "x_cols",
            "k_cols",
            "tf",
            "dt0",
            "r_tol",
            "a_tol",
            "ms",
        ),
    )
    def _rodas5_pallas_solve(
        m_arr,
        y0_arr,
        *,
        n_pad,
        y_cols,
        w_cols,
        x_cols,
        k_cols,
        tf,
        dt0,
        r_tol,
        a_tol,
        ms,
    ):
        step_fn = _make_rodas5_step(n_vars)

        def kernel_body(y0_ref, m_ref, y_ref, w_ref, x_ref, k_ref, u_ref):
            for i in range(n_vars):
                y_ref.at[:, i][...] = y0_ref.at[:, i][...]

            z = y0_ref.at[:, 0][...] * 0.0
            t = z + 0.0
            dt_v = z + dt0

            def cond_fn(state):
                t, _dt_v, n = state
                return (jnp.min(t) < tf) & (n < ms)

            def body_fn(state):
                t, dt_v, n = state

                active = t < tf
                dt_use = jnp.maximum(jnp.minimum(dt_v, tf - t), 1e-30)

                step_fn(
                    m_ref,
                    dt_use,
                    w_ref,
                    x_ref,
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

                safe_EEst = jnp.where(
                    jnp.isnan(EEst) | (EEst > 1e18),
                    1e18,
                    jnp.where(EEst == 0.0, 1e-18, EEst),
                )
                factor = jnp.clip(0.9 * safe_EEst ** (-1.0 / 6.0), 0.2, 6.0)
                new_dt = jnp.where(active, dt_use * factor, dt_v)

                return (t_new, new_dt, n + 1)

            jax.lax.while_loop(cond_fn, body_fn, (t, dt_v, jnp.int32(0)))

        y_bs = pl.BlockSpec((_BLOCK, y_cols), lambda i: (i, 0))
        m_bs = pl.BlockSpec((n_vars, n_vars), lambda i: (0, 0))
        w_bs = pl.BlockSpec((_BLOCK, w_cols), lambda i: (i, 0))
        x_bs = pl.BlockSpec((_BLOCK, x_cols), lambda i: (i, 0))
        k_bs = pl.BlockSpec((_BLOCK, k_cols), lambda i: (i, 0))
        results = pl.pallas_call(
            kernel_body,
            out_shape=[
                jax.ShapeDtypeStruct((n_pad, y_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, w_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, x_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, k_cols), jnp.float64),
                jax.ShapeDtypeStruct((n_pad, x_cols), jnp.float64),
            ],
            grid=(n_pad // _BLOCK,),
            in_specs=(y_bs, m_bs),
            out_specs=(y_bs, w_bs, x_bs, k_bs, x_bs),
            compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
        )(y0_arr, m_arr)
        return results[0]

    def solve_ensemble_pallas(
        y0_batch,
        t_span,
        *,
        rtol=1e-6,
        atol=1e-8,
        first_step=None,
        max_steps=100000,
    ):
        """Solve ensemble dy/dt = M y with Rodas5 Pallas custom kernel."""
        if y0_batch.shape[1] != n_vars:
            raise ValueError(
                f"y0_batch has {y0_batch.shape[1]} vars but M is {n_vars}x{n_vars}"
            )

        N = y0_batch.shape[0]
        N_pad = ((N + _BLOCK - 1) // _BLOCK) * _BLOCK

        tf = float(t_span[1])
        dt0 = float(first_step if first_step is not None else (tf - float(t_span[0])) * 1e-6)

        y_cols = _pad_cols_pow2(n_vars)
        w_cols = _pad_cols_pow2(n_vars * n_vars)
        x_cols = _pad_cols_pow2(n_vars)
        k_cols = _pad_cols_pow2(8 * n_vars)

        y0_arr = jnp.pad(y0_batch, ((0, N_pad - N), (0, y_cols - n_vars)))

        y_out = _rodas5_pallas_solve(
            M,
            y0_arr,
            n_pad=N_pad,
            y_cols=y_cols,
            w_cols=w_cols,
            x_cols=x_cols,
            k_cols=k_cols,
            tf=tf,
            dt0=dt0,
            r_tol=float(rtol),
            a_tol=float(atol),
            ms=int(max_steps),
        )

        return y_out[:N, :n_vars]

    return solve_ensemble_pallas
