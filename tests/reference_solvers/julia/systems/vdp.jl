function make_vdp_spec(config)
    n_osc = require_config_int(config, "n_osc")
    mu_max = require_config_float(config, "mu_max")
    n_vars = 2 * n_osc
    mu = Tuple(logspace_from_one_to(max(mu_max, 1.0), n_osc))
    omega = Tuple(logspace(-0.5, 0.5, n_osc))

    function ode!(du, u, p, t)
        s = p[1]
        for osc_idx in 1:n_osc
            x_idx = 2 * osc_idx - 1
            v_idx = x_idx + 1
            x = u[x_idx]
            v = u[v_idx]
            du[x_idx] = v
            du[v_idx] = s * mu[osc_idx] * (1.0 - x^2) * v - omega[osc_idx]^2 * x
        end
        return nothing
    end

    function explicit_ode!(du, u, p, t)
        for osc_idx in 1:n_osc
            x_idx = 2 * osc_idx - 1
            v_idx = x_idx + 1
            x = u[x_idx]
            v = u[v_idx]
            du[x_idx] = v
            du[v_idx] = -omega[osc_idx]^2 * x
        end
        return nothing
    end

    function implicit_ode!(du, u, p, t)
        s = p[1]
        for osc_idx in 1:n_osc
            x_idx = 2 * osc_idx - 1
            v_idx = x_idx + 1
            x = u[x_idx]
            v = u[v_idx]
            du[x_idx] = 0.0
            du[v_idx] = s * mu[osc_idx] * (1.0 - x^2) * v
        end
        return nothing
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        s = p[1]
        for osc_idx in 1:n_osc
            x_idx = 2 * osc_idx - 1
            v_idx = x_idx + 1
            x = u[x_idx]
            v = u[v_idx]
            μ = mu[osc_idx]
            ω = omega[osc_idx]
            J[x_idx, v_idx] = 1.0
            J[v_idx, x_idx] = -2.0 * s * μ * x * v - ω^2
            J[v_idx, v_idx] = s * μ * (1.0 - x^2)
        end
        return nothing
    end

    function explicit_jac!(J, u, p, t)
        fill!(J, 0.0)
        for osc_idx in 1:n_osc
            x_idx = 2 * osc_idx - 1
            v_idx = x_idx + 1
            ω = omega[osc_idx]
            J[x_idx, v_idx] = 1.0
            J[v_idx, x_idx] = -ω^2
        end
        return nothing
    end

    function implicit_jac!(J, u, p, t)
        fill!(J, 0.0)
        s = p[1]
        for osc_idx in 1:n_osc
            x_idx = 2 * osc_idx - 1
            v_idx = x_idx + 1
            x = u[x_idx]
            v = u[v_idx]
            μ = mu[osc_idx]
            J[v_idx, x_idx] = -2.0 * s * μ * x * v
            J[v_idx, v_idx] = s * μ * (1.0 - x^2)
        end
        return nothing
    end

    function kernel_ode(u, p, t)
        s = p[1]
        return SVector{n_vars, Float64}(ntuple(n_vars) do idx
            osc_idx = fld(idx + 1, 2)
            x_idx = 2 * osc_idx - 1
            v_idx = x_idx + 1
            x = u[x_idx]
            v = u[v_idx]
            isodd(idx) ? v : s * mu[osc_idx] * (1.0 - x^2) * v - omega[osc_idx]^2 * x
        end)
    end

    return ReferenceSystemSpec(
        build_array_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(ode!; jac=jac!, tgrad=zero_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_array_split_problem=(y0, tspan, p0) -> SciMLBase.SplitODEProblem(
            SciMLBase.ODEFunction(implicit_ode!; jac=implicit_jac!, tgrad=zero_tgrad!),
            SciMLBase.ODEFunction(explicit_ode!; jac=explicit_jac!, tgrad=zero_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_kernel_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem{false}(
            kernel_ode,
            vector_to_svector(y0),
            tspan,
            vector_to_svector(p0),
        ),
    )
end
