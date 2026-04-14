function make_damped_rotation_spec(config)
    n_pairs = require_config_int(config, "n_pairs")
    n_vars = 2 * n_pairs
    lambdas = Tuple([0.2 + 1.8 * (i - 1) / max(n_pairs - 1, 1) for i in 1:n_pairs])
    omegas = Tuple([0.5 + 1.0 * (i - 1) / max(n_pairs - 1, 1) for i in 1:n_pairs])
    zero_jac! = make_zero_jac!(n_vars)

    function ode!(du, u, p, t)
        s = p[1]
        for pair_idx in 1:n_pairs
            x_idx = 2 * pair_idx - 1
            y_idx = x_idx + 1
            λ = lambdas[pair_idx]
            ω = omegas[pair_idx]
            du[x_idx] = s * (-λ * u[x_idx] - ω * u[y_idx])
            du[y_idx] = s * (ω * u[x_idx] - λ * u[y_idx])
        end
        return nothing
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        s = p[1]
        for pair_idx in 1:n_pairs
            x_idx = 2 * pair_idx - 1
            y_idx = x_idx + 1
            λ = lambdas[pair_idx]
            ω = omegas[pair_idx]
            J[x_idx, x_idx] = -s * λ
            J[x_idx, y_idx] = -s * ω
            J[y_idx, x_idx] = s * ω
            J[y_idx, y_idx] = -s * λ
        end
        return nothing
    end

    function explicit_ode!(du, u, p, t)
        return ode!(du, u, p, t)
    end

    function implicit_ode!(du, u, p, t)
        fill!(du, 0.0)
        return nothing
    end

    function explicit_jac!(J, u, p, t)
        return jac!(J, u, p, t)
    end

    function kernel_ode(u, p, t)
        s = p[1]
        return SVector{n_vars, Float64}(ntuple(n_vars) do i
            pair_idx = fld(i + 1, 2)
            λ = lambdas[pair_idx]
            ω = omegas[pair_idx]
            isodd(i) ? s * (-λ * u[i] - ω * u[i + 1]) : s * (ω * u[i - 1] - λ * u[i])
        end)
    end

    function kernel_explicit(u, p, t)
        return kernel_ode(u, p, t)
    end

    return ReferenceSystemSpec(
        build_array_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(ode!; jac=jac!, tgrad=zero_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_array_split_problem=(y0, tspan, p0) -> SciMLBase.SplitODEProblem(
            SciMLBase.ODEFunction(implicit_ode!; jac=zero_jac!, tgrad=zero_tgrad!),
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
