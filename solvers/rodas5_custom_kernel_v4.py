"""Matrix-specialized Rodas5 ensemble ODE solver with Numba CUDA kernel.

One-thread-per-trajectory approach. Each CUDA thread independently integrates
dy/dt = M y for a single trajectory using the Rodas5 method with adaptive
time stepping. No JAX dependency.

This code is not obviously faster than v3 JAX/Pallas/Triton implementation in
either compilation time or runtime, but it is simpler.
"""

import math

import numpy as np
from numba import cuda

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


@cuda.jit(device=True)
def _mat_vec(M, src, dst, nv, tid):
    """dst[tid, :] = M @ src[tid, :]."""
    for i in range(nv):
        s = 0.0
        for j in range(nv):
            s += M[i, j] * src[tid, j]
        dst[tid, i] = s


@cuda.jit(device=True)
def _lu_solve(W, x, k, nv, tid, stage):
    """Solve LU system in-place on x[tid,:], copy result to k[tid, stage*nv:]."""
    # Forward substitution (L)
    for i in range(nv):
        for j in range(i):
            x[tid, i] -= W[tid, i * nv + j] * x[tid, j]
    # Backward substitution (U)
    for i_rev in range(nv):
        i = nv - 1 - i_rev
        for j in range(i + 1, nv):
            x[tid, i] -= W[tid, i * nv + j] * x[tid, j]
        x[tid, i] /= W[tid, i * nv + i]
    # Copy to stage slot
    off = stage * nv
    for i in range(nv):
        k[tid, off + i] = x[tid, i]


@cuda.jit
def _rodas5_kernel(M, y0, y_out, W, k, x, u, N, nv, tf, dt0, rtol, atol, max_steps):
    tid = cuda.grid(1)
    if tid >= N:
        return

    # Initialize y from y0
    for i in range(nv):
        y_out[tid, i] = y0[tid, i]

    t = 0.0
    dt = dt0

    for _step in range(max_steps):
        if t >= tf:
            break

        dt_use = min(dt, tf - t)
        if dt_use < 1e-30:
            dt_use = 1e-30

        d = 1.0 / (dt_use * _gamma)
        inv_dt = 1.0 / dt_use

        # Build W = I*d - M and LU factorize in-place (no pivoting)
        for i in range(nv):
            for j in range(nv):
                w_val = -M[i, j]
                if i == j:
                    w_val += d
                W[tid, i * nv + j] = w_val

        for j in range(nv):
            piv = W[tid, j * nv + j]
            for i in range(j + 1, nv):
                lij = W[tid, i * nv + j] / piv
                W[tid, i * nv + j] = lij
                for kk in range(j + 1, nv):
                    W[tid, i * nv + kk] -= lij * W[tid, j * nv + kk]

        # ---- Stage 1: x = M*y, solve W*k1 = x ----
        _mat_vec(M, y_out, x, nv, tid)
        _lu_solve(W, x, k, nv, tid, 0)

        # ---- Stage 2 ----
        for i in range(nv):
            u[tid, i] = y_out[tid, i] + _a21 * k[tid, i]
        _mat_vec(M, u, x, nv, tid)
        for i in range(nv):
            x[tid, i] += _C21 * k[tid, i] * inv_dt
        _lu_solve(W, x, k, nv, tid, 1)

        # ---- Stage 3 ----
        for i in range(nv):
            u[tid, i] = (
                y_out[tid, i] + _a31 * k[tid, 0 * nv + i] + _a32 * k[tid, 1 * nv + i]
            )
        _mat_vec(M, u, x, nv, tid)
        for i in range(nv):
            x[tid, i] += (
                _C31 * k[tid, 0 * nv + i] + _C32 * k[tid, 1 * nv + i]
            ) * inv_dt
        _lu_solve(W, x, k, nv, tid, 2)

        # ---- Stage 4 ----
        for i in range(nv):
            u[tid, i] = (
                y_out[tid, i]
                + _a41 * k[tid, 0 * nv + i]
                + _a42 * k[tid, 1 * nv + i]
                + _a43 * k[tid, 2 * nv + i]
            )
        _mat_vec(M, u, x, nv, tid)
        for i in range(nv):
            x[tid, i] += (
                _C41 * k[tid, 0 * nv + i]
                + _C42 * k[tid, 1 * nv + i]
                + _C43 * k[tid, 2 * nv + i]
            ) * inv_dt
        _lu_solve(W, x, k, nv, tid, 3)

        # ---- Stage 5 ----
        for i in range(nv):
            u[tid, i] = (
                y_out[tid, i]
                + _a51 * k[tid, 0 * nv + i]
                + _a52 * k[tid, 1 * nv + i]
                + _a53 * k[tid, 2 * nv + i]
                + _a54 * k[tid, 3 * nv + i]
            )
        _mat_vec(M, u, x, nv, tid)
        for i in range(nv):
            x[tid, i] += (
                _C51 * k[tid, 0 * nv + i]
                + _C52 * k[tid, 1 * nv + i]
                + _C53 * k[tid, 2 * nv + i]
                + _C54 * k[tid, 3 * nv + i]
            ) * inv_dt
        _lu_solve(W, x, k, nv, tid, 4)

        # ---- Stage 6 ----
        for i in range(nv):
            u[tid, i] = (
                y_out[tid, i]
                + _a61 * k[tid, 0 * nv + i]
                + _a62 * k[tid, 1 * nv + i]
                + _a63 * k[tid, 2 * nv + i]
                + _a64 * k[tid, 3 * nv + i]
                + _a65 * k[tid, 4 * nv + i]
            )
        _mat_vec(M, u, x, nv, tid)
        for i in range(nv):
            x[tid, i] += (
                _C61 * k[tid, 0 * nv + i]
                + _C62 * k[tid, 1 * nv + i]
                + _C63 * k[tid, 2 * nv + i]
                + _C64 * k[tid, 3 * nv + i]
                + _C65 * k[tid, 4 * nv + i]
            ) * inv_dt
        _lu_solve(W, x, k, nv, tid, 5)

        # ---- Stage 7: u += k6 (reuses u from stage 6) ----
        for i in range(nv):
            u[tid, i] += k[tid, 5 * nv + i]
        _mat_vec(M, u, x, nv, tid)
        for i in range(nv):
            x[tid, i] += (
                _C71 * k[tid, 0 * nv + i]
                + _C72 * k[tid, 1 * nv + i]
                + _C73 * k[tid, 2 * nv + i]
                + _C74 * k[tid, 3 * nv + i]
                + _C75 * k[tid, 4 * nv + i]
                + _C76 * k[tid, 5 * nv + i]
            ) * inv_dt
        _lu_solve(W, x, k, nv, tid, 6)

        # ---- Stage 8: u += k7 (reuses u from stage 7) ----
        for i in range(nv):
            u[tid, i] += k[tid, 6 * nv + i]
        _mat_vec(M, u, x, nv, tid)
        for i in range(nv):
            x[tid, i] += (
                _C81 * k[tid, 0 * nv + i]
                + _C82 * k[tid, 1 * nv + i]
                + _C83 * k[tid, 2 * nv + i]
                + _C84 * k[tid, 3 * nv + i]
                + _C85 * k[tid, 4 * nv + i]
                + _C86 * k[tid, 5 * nv + i]
                + _C87 * k[tid, 6 * nv + i]
            ) * inv_dt
        _lu_solve(W, x, k, nv, tid, 7)

        # ---- y_new = u + k8 (stored in u for error estimation) ----
        for i in range(nv):
            u[tid, i] += k[tid, 7 * nv + i]

        # ---- Error estimation ----
        err_sq = 0.0
        for i in range(nv):
            k8i = k[tid, 7 * nv + i]
            sc = atol + rtol * max(abs(y_out[tid, i]), abs(u[tid, i]))
            err_sq += (k8i / sc) ** 2
        EEst = math.sqrt(err_sq / nv)

        accept = (EEst <= 1.0) and not math.isnan(EEst)
        if accept:
            t += dt_use
            for i in range(nv):
                y_out[tid, i] = u[tid, i]

        # ---- Step size adaptation ----
        if math.isnan(EEst) or EEst > 1e18:
            safe_EEst = 1e18
        elif EEst == 0.0:
            safe_EEst = 1e-18
        else:
            safe_EEst = EEst

        factor = 0.9 * safe_EEst ** (-1.0 / 6.0)
        if factor < 0.2:
            factor = 0.2
        elif factor > 6.0:
            factor = 6.0
        dt = dt_use * factor


_THREADS_PER_BLOCK = 256


def make_solver(M_host):
    """Create a Numba CUDA ensemble Rodas5 solver for dy/dt = M y.

    Args:
        M_host: Constant Jacobian/state matrix with shape (n_vars, n_vars).
    """
    M_np = np.asarray(M_host, dtype=np.float64)
    if M_np.ndim != 2 or M_np.shape[0] != M_np.shape[1]:
        raise ValueError(f"M must be square with shape (n, n), got {M_np.shape}")

    n_vars = M_np.shape[0]
    M_d = cuda.to_device(M_np)

    def solve_ensemble(
        y0_batch,
        t_span,
        *,
        rtol=1e-6,
        atol=1e-8,
        first_step=None,
        max_steps=100000,
    ):
        """Solve ensemble dy/dt = M y with Rodas5 Numba CUDA kernel."""
        y0_np = np.asarray(y0_batch, dtype=np.float64)
        if y0_np.shape[1] != n_vars:
            raise ValueError(
                f"y0_batch has {y0_np.shape[1]} vars but M is {n_vars}x{n_vars}"
            )

        N = y0_np.shape[0]
        tf = float(t_span[1])
        dt0 = float(
            first_step if first_step is not None else (tf - float(t_span[0])) * 1e-6
        )

        y0_d = cuda.to_device(y0_np)
        y_d = cuda.device_array((N, n_vars), dtype=np.float64)
        W_d = cuda.device_array((N, n_vars * n_vars), dtype=np.float64)
        k_d = cuda.device_array((N, 8 * n_vars), dtype=np.float64)
        x_d = cuda.device_array((N, n_vars), dtype=np.float64)
        u_d = cuda.device_array((N, n_vars), dtype=np.float64)

        threads = _THREADS_PER_BLOCK
        blocks = (N + threads - 1) // threads

        _rodas5_kernel[blocks, threads](
            M_d,
            y0_d,
            y_d,
            W_d,
            k_d,
            x_d,
            u_d,
            N,
            n_vars,
            tf,
            dt0,
            float(rtol),
            float(atol),
            max_steps,
        )
        cuda.synchronize()
        return y_d.copy_to_host()

    return solve_ensemble
