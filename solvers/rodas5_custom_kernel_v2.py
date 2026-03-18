"""General-purpose Rodas5 ensemble ODE solver with Pallas GPU kernel.

Implements the Rodas5 algorithm — a Rosenbrock method of order 5(4) for stiff
ODEs — using the same general-purpose make_solver pattern as
rosenbrock23_custom_kernel.py.

The W-transformed variant is from Di Marzo (1993) and DISCO-EB (Hahn).
8 stages with adaptive step size control. Pallas/Triton custom GPU kernel
with 32 trajectories per block.

Stage vectors and arithmetic use Pallas Refs with fori_loop, producing compact
compiled Triton IR whose code size is O(1) regardless of n_vars. The
while_loop carries only (t, dt, step_count) — all vectors live in Refs.

The LU decomposition and forward/back substitution also use fori_loop with
Pallas Ref scratch memory and pl.ds dynamic indexing.

An optional ode_ref_fn can be supplied for Ref-based ODE evaluation with
fori_loop, further reducing compiled code size for large systems.

Reference: https://github.com/SciML/DiffEqGPU.jl
"""

import functools
import itertools

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E402 - must precede jax.numpy import
import jax.numpy as jnp  # isort: skip  # noqa: E402
from jax._src.pallas.triton import core as pltriton
from jax.experimental import pallas as pl

# fmt: off
# Rodas5 W-transformed coefficients
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

# The number of ODE trajectories solved per block.
_BLOCK = 32


def _pad_cols_pow2(n_cols):
    """Return the next power of 2 >= n_cols."""
    return 1 << (n_cols - 1).bit_length()


# ---------------------------------------------------------------------------
# Rodas5 step: all vectors in Refs, stage arithmetic via fori_loop
# ---------------------------------------------------------------------------


def _make_rodas5_step(ode_fn, ode_ref_fn, n_vars, n_params):
    """Create Rodas5 step function with Ref-based stage vectors.

    Stage vectors k1..k8 are stored in k_ref, intermediate state in u_ref,
    and ODE evaluations / linear solve RHS in x_ref. All per-element vector
    arithmetic uses fori_loop for compact Triton IR.

    The Jacobian is always computed via JVP on the tuple-based ode_fn.
    Stage ODE evaluations use ode_ref_fn if available, else ode_fn wrapped
    with Ref reads/writes.
    """
    nv = n_vars
    np_ = n_params

    # -- Helper: evaluate ODE from src_ref, write result to dst_ref ----------
    def _eval_F(src_ref, p_ref, dst_ref, c_ref):
        if ode_ref_fn is not None:
            ode_ref_fn(src_ref, p_ref, dst_ref, c_ref)
        else:
            y_t = tuple(src_ref.at[:, i][...] for i in range(nv))
            p_t = tuple(p_ref.at[:, i][...] for i in range(np_))
            dy = ode_fn(y_t, p_t)
            for i in range(nv):
                dst_ref.at[:, i][...] = dy[i]

    # -- Helper: LU forward/back solve, copy result to k_ref ----------------
    def _lu_solve(w_ref, x_ref, k_ref, stage):
        """Solve (LU) x = rhs in-place on x_ref, copy result to k_ref[stage]."""

        # Forward substitution
        def fwd_row(i, carry):
            def fwd_col(j, carry):
                xi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
                xj = x_ref.at[:, pl.ds(j, 1)][...][:, 0]
                lij = w_ref.at[:, pl.ds(i * nv + j, 1)][...][:, 0]
                x_ref.at[:, pl.ds(i, 1)][...] = (xi - lij * xj)[:, None]
                return carry

            return jax.lax.fori_loop(0, i, fwd_col, carry)

        jax.lax.fori_loop(0, nv, fwd_row, jnp.int32(0))

        # Back substitution
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

        # Copy result to k_ref at stage offset
        off = stage * nv

        def copy_k(i, carry):
            val = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k_ref.at[:, pl.ds(off + i, 1)][...] = val[:, None]
            return carry

        jax.lax.fori_loop(0, nv, copy_k, jnp.int32(0))

    # -- Main step function --------------------------------------------------
    def step(p_ref, dt, w_ref, x_ref, y_ref, k_ref, u_ref, c_ref):
        """One Rodas5 step.  Reads state from y_ref, writes y_new to u_ref,
        error estimate k8 to k_ref[7*nv:]."""
        dtgamma = dt * _gamma
        inv_dt = 1.0 / dt
        d = 1.0 / dtgamma

        # --- Jacobian via JVP (trace-time unrolled, uses tuple-based ode_fn) ---
        y_tuple = tuple(y_ref.at[:, i][...] for i in range(nv))
        p_tuple = tuple(p_ref.at[:, i][...] for i in range(np_))
        ones = y_tuple[0] * 0.0 + 1.0

        J_cols = []
        for j in range(nv):

            def f_j(yj, j=j):
                y_mod = tuple(yj if k == j else y_tuple[k] for k in range(nv))
                return ode_fn(y_mod, p_tuple)

            _, col = jax.jvp(f_j, (y_tuple[j],), (ones,))
            J_cols.append(col)

        # Fill W = I/(dt*gamma) - J into w_ref (trace-time unrolled)
        for i_s in range(nv):
            for j_s in range(nv):
                val = -J_cols[j_s][i_s]
                if i_s == j_s:
                    val = val + d
                w_ref.at[:, i_s * nv + j_s][...] = val

        # --- LU decomposition in-place on w_ref via fori_loop ---
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

        # ====================== 8 Rodas5 stages ========================

        # Stage 1: k1 = S(F(y))
        _eval_F(y_ref, p_ref, x_ref, c_ref)
        _lu_solve(w_ref, x_ref, k_ref, 0)

        # Stage 2: u = y + a21*k1, k2 = S(F(u) + C21*k1/dt)
        def s2_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a21 * k1i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s2_u, jnp.int32(0))
        _eval_F(u_ref, p_ref, x_ref, c_ref)

        def s2_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (fi + _C21 * k1i * inv_dt)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s2_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 1)

        # Stage 3
        def s3_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a31 * k1i + _a32 * k2i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s3_u, jnp.int32(0))
        _eval_F(u_ref, p_ref, x_ref, c_ref)

        def s3_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (fi + (_C31 * k1i + _C32 * k2i) * inv_dt)[
                :, None
            ]
            return carry

        jax.lax.fori_loop(0, nv, s3_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 2)

        # Stage 4
        def s4_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (yi + _a41 * k1i + _a42 * k2i + _a43 * k3i)[
                :, None
            ]
            return carry

        jax.lax.fori_loop(0, nv, s4_u, jnp.int32(0))
        _eval_F(u_ref, p_ref, x_ref, c_ref)

        def s4_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (
                fi + (_C41 * k1i + _C42 * k2i + _C43 * k3i) * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s4_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 3)

        # Stage 5
        def s5_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (
                yi + _a51 * k1i + _a52 * k2i + _a53 * k3i + _a54 * k4i
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s5_u, jnp.int32(0))
        _eval_F(u_ref, p_ref, x_ref, c_ref)

        def s5_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (
                fi + (_C51 * k1i + _C52 * k2i + _C53 * k3i + _C54 * k4i) * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s5_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 4)

        # Stage 6
        def s6_u(i, carry):
            yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            k5i = k_ref.at[:, pl.ds(4 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (
                yi + _a61 * k1i + _a62 * k2i + _a63 * k3i + _a64 * k4i + _a65 * k5i
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s6_u, jnp.int32(0))
        _eval_F(u_ref, p_ref, x_ref, c_ref)

        def s6_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            k5i = k_ref.at[:, pl.ds(4 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (
                fi
                + (_C61 * k1i + _C62 * k2i + _C63 * k3i + _C64 * k4i + _C65 * k5i)
                * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s6_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 5)

        # Stage 7: u = u_stage6 + k6
        def s7_u(i, carry):
            ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k6i = k_ref.at[:, pl.ds(5 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (ui + k6i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s7_u, jnp.int32(0))
        _eval_F(u_ref, p_ref, x_ref, c_ref)

        def s7_rhs(i, carry):
            fi = x_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k1i = k_ref.at[:, pl.ds(0 * nv + i, 1)][...][:, 0]
            k2i = k_ref.at[:, pl.ds(1 * nv + i, 1)][...][:, 0]
            k3i = k_ref.at[:, pl.ds(2 * nv + i, 1)][...][:, 0]
            k4i = k_ref.at[:, pl.ds(3 * nv + i, 1)][...][:, 0]
            k5i = k_ref.at[:, pl.ds(4 * nv + i, 1)][...][:, 0]
            k6i = k_ref.at[:, pl.ds(5 * nv + i, 1)][...][:, 0]
            x_ref.at[:, pl.ds(i, 1)][...] = (
                fi
                + (
                    _C71 * k1i
                    + _C72 * k2i
                    + _C73 * k3i
                    + _C74 * k4i
                    + _C75 * k5i
                    + _C76 * k6i
                )
                * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s7_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 6)

        # Stage 8: u = u_stage7 + k7
        def s8_u(i, carry):
            ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k7i = k_ref.at[:, pl.ds(6 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (ui + k7i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s8_u, jnp.int32(0))
        _eval_F(u_ref, p_ref, x_ref, c_ref)

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
                + (
                    _C81 * k1i
                    + _C82 * k2i
                    + _C83 * k3i
                    + _C84 * k4i
                    + _C85 * k5i
                    + _C86 * k6i
                    + _C87 * k7i
                )
                * inv_dt
            )[:, None]
            return carry

        jax.lax.fori_loop(0, nv, s8_rhs, jnp.int32(0))
        _lu_solve(w_ref, x_ref, k_ref, 7)

        # y_new = u + k8  ->  u_ref
        def ynew(i, carry):
            ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
            k8i = k_ref.at[:, pl.ds(7 * nv + i, 1)][...][:, 0]
            u_ref.at[:, pl.ds(i, 1)][...] = (ui + k8i)[:, None]
            return carry

        jax.lax.fori_loop(0, nv, ynew, jnp.int32(0))
        # After: u_ref = y_new, k_ref[7*nv:] = k8 (error estimate)

    return step


# ---------------------------------------------------------------------------
# Pallas/Triton custom kernel (general ODE, 32 trajectories per block)
# ---------------------------------------------------------------------------


def make_solver(ode_fn, *, ode_ref_fn=None, constants=None):
    """Create a Pallas ensemble solver for the given ODE using Rodas5.

    Args:
        ode_fn: ODE right-hand side function with signature
            (y, p) -> dy/dt
            where y and p are tuples of scalar-like values and the return
            is a tuple of the same length as y. The function must use only
            element-wise operations so it can run inside a Pallas/Triton
            kernel.  Always required (used for Jacobian computation via JVP).

        ode_ref_fn: Optional Ref-based ODE function with signature
            (y_ref, p_ref, dy_ref, c_ref) -> None
            that reads state from y_ref, parameters from p_ref, and writes
            derivatives to dy_ref.  c_ref provides the constants array.
            Can use fori_loop for compact Triton IR in large systems.
            If None, ode_fn is wrapped with Ref reads/writes instead.

        constants: Optional 1-D array of constants passed to ode_ref_fn
            via c_ref.  Broadcast to every block so that each trajectory
            sees the same constant values.

    Returns:
        solve_ensemble_pallas function.

    Example::

        def robertson(y, p):
            return (
                -p[0] * y[0] + p[1] * y[1] * y[2],
                 p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1]**2,
                 p[2] * y[1]**2,
            )

        solve = make_solver(robertson)
        results = solve(y0_batch, (0.0, 1e5), params_batch)
    """

    @functools.partial(
        jax.jit,
        static_argnames=(
            "n_pad",
            "p_cols",
            "y_cols",
            "w_cols",
            "x_cols",
            "k_cols",
            "c_cols",
            "n_vars",
            "n_params",
            "tf",
            "dt0",
            "r_tol",
            "a_tol",
            "ms",
        ),
    )
    def _rodas5_pallas_solve(
        params_arr,
        y0_arr,
        c_arr,
        *,
        n_pad,
        p_cols,
        y_cols,
        w_cols,
        x_cols,
        k_cols,
        c_cols,
        n_vars,
        n_params,
        tf,
        dt0,
        r_tol,
        a_tol,
        ms,
    ):
        step_fn = _make_rodas5_step(ode_fn, ode_ref_fn, n_vars, n_params)
        nv = n_vars

        def kernel_body(params_ref, y0_ref, c_ref, y_ref, w_ref, x_ref, k_ref, u_ref):
            # Copy initial conditions to working state
            for i in range(n_vars):
                y_ref.at[:, i][...] = y0_ref.at[:, i][...]

            z = params_ref.at[:, 0][...] * 0.0
            t = z + 0.0
            dt_v = z + dt0

            def cond_fn(state):
                t, _dt_v, n = state
                return (jnp.min(t) < tf) & (n < ms)

            def body_fn(state):
                t, dt_v, n = state

                active = t < tf
                dt_use = jnp.maximum(jnp.minimum(dt_v, tf - t), 1e-30)

                # Rodas5 step: y_ref -> u_ref (y_new), k_ref[7*nv:] (k8 err)
                step_fn(
                    params_ref,
                    dt_use,
                    w_ref,
                    x_ref,
                    y_ref,
                    k_ref,
                    u_ref,
                    c_ref,
                )

                # Error estimation via fori_loop
                def compute_err_sq(i, acc):
                    yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    k8i = k_ref.at[:, pl.ds(7 * nv + i, 1)][...][:, 0]
                    sc = a_tol + r_tol * jnp.maximum(jnp.abs(yi), jnp.abs(ui))
                    return acc + (k8i / sc) ** 2

                err_sq = jax.lax.fori_loop(0, nv, compute_err_sq, jnp.zeros_like(t))
                EEst = jnp.sqrt(err_sq / nv)

                accept = (EEst <= 1.0) & ~jnp.isnan(EEst)
                mask = active & accept

                t_new = jnp.where(mask, t + dt_use, t)

                # Accept/reject: update y_ref from u_ref where accepted
                def update_y(i, carry):
                    yi = y_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    ui = u_ref.at[:, pl.ds(i, 1)][...][:, 0]
                    y_ref.at[:, pl.ds(i, 1)][...] = jnp.where(mask, ui, yi)[:, None]
                    return carry

                jax.lax.fori_loop(0, nv, update_y, jnp.int32(0))

                safe_EEst = jnp.where(
                    jnp.isnan(EEst) | (EEst > 1e18),
                    1e18,
                    jnp.where(EEst == 0.0, 1e-18, EEst),
                )
                factor = jnp.clip(0.9 * safe_EEst ** (-1.0 / 6.0), 0.2, 6.0)
                new_dt = jnp.where(active, dt_use * factor, dt_v)

                return (t_new, new_dt, n + 1)

            jax.lax.while_loop(cond_fn, body_fn, (t, dt_v, jnp.int32(0)))
            # y_ref already holds the final result

        p_bs = pl.BlockSpec((_BLOCK, p_cols), lambda i: (i, 0))
        y_bs = pl.BlockSpec((_BLOCK, y_cols), lambda i: (i, 0))
        w_bs = pl.BlockSpec((_BLOCK, w_cols), lambda i: (i, 0))
        x_bs = pl.BlockSpec((_BLOCK, x_cols), lambda i: (i, 0))
        k_bs = pl.BlockSpec((_BLOCK, k_cols), lambda i: (i, 0))
        c_bs = pl.BlockSpec((_BLOCK, c_cols), lambda i: (i, 0))
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
            in_specs=(p_bs, y_bs, c_bs),
            out_specs=(y_bs, w_bs, x_bs, k_bs, x_bs),
            compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=2),
        )(params_arr, y0_arr, c_arr)
        return results[0]

    def solve_ensemble_pallas(
        y0_batch,
        t_span,
        params_batch,
        *,
        rtol=1e-6,
        atol=1e-8,
        first_step=None,
        max_steps=100000,
    ):
        """Solve ODE ensemble using Rodas5 in a Pallas/Triton custom GPU kernel.

        Args:
            y0_batch: Per-trajectory initial conditions, shape (N, n_vars).
            t_span: (t0, tf) time interval.
            params_batch: Per-trajectory parameters, shape (N, n_params).
        """
        N = params_batch.shape[0]
        n_vars = y0_batch.shape[1]
        n_params = params_batch.shape[1]

        # Verify ODE function dimensions via JAX abstract evaluation
        y_trace = tuple(itertools.repeat(jax.ShapeDtypeStruct((), jnp.float64), n_vars))
        p_trace = tuple(
            itertools.repeat(jax.ShapeDtypeStruct((), jnp.float64), n_params)
        )
        out_trace = jax.eval_shape(ode_fn, y_trace, p_trace)
        assert len(out_trace) == n_vars, (
            f"ODE function returns {len(out_trace)} components but y0 has {n_vars}"
        )

        N_pad = ((N + _BLOCK - 1) // _BLOCK) * _BLOCK
        tf = float(t_span[1])
        dt0 = float(
            first_step if first_step is not None else (tf - float(t_span[0])) * 1e-6
        )

        p_cols = _pad_cols_pow2(n_params)
        y_cols = _pad_cols_pow2(n_vars)
        w_cols = _pad_cols_pow2(n_vars * n_vars)
        x_cols = _pad_cols_pow2(n_vars)
        k_cols = _pad_cols_pow2(8 * n_vars)

        # Constants array (broadcast to every row)
        if constants is not None:
            n_const = len(constants)
            c_cols = _pad_cols_pow2(n_const)
            c_1d = jnp.pad(
                jnp.asarray(constants, dtype=jnp.float64),
                (0, c_cols - n_const),
            )
            c_arr = jnp.tile(c_1d[None, :], (N_pad, 1))
        else:
            c_cols = 1
            c_arr = jnp.zeros((N_pad, 1), dtype=jnp.float64)

        params_arr = jnp.pad(params_batch, ((0, N_pad - N), (0, p_cols - n_params)))
        y0_arr = jnp.pad(y0_batch, ((0, N_pad - N), (0, y_cols - n_vars)))

        y_out = _rodas5_pallas_solve(
            params_arr,
            y0_arr,
            c_arr,
            n_pad=N_pad,
            p_cols=p_cols,
            y_cols=y_cols,
            w_cols=w_cols,
            x_cols=x_cols,
            k_cols=k_cols,
            c_cols=c_cols,
            n_vars=n_vars,
            n_params=n_params,
            tf=tf,
            dt0=dt0,
            r_tol=float(rtol),
            a_tol=float(atol),
            ms=int(max_steps),
        )

        return y_out[:N, :n_vars]

    return solve_ensemble_pallas
