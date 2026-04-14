function make_heat_equation_spec(config)
    n_vars = require_config_int(config, "n_vars")
    dx = 1.0 / (n_vars + 1)
    inv_dx2 = 1.0 / dx^2
    zero_jac! = make_zero_jac!(n_vars)

    function apply_heat!(du, u, scale)
        for i in 1:n_vars
            left = i == 1 ? 0.0 : u[i - 1]
            right = i == n_vars ? 0.0 : u[i + 1]
            du[i] = scale * inv_dx2 * (left - 2.0 * u[i] + right)
        end
        return nothing
    end

    function ode!(du, u, p, t)
        return apply_heat!(du, u, p[1])
    end

    function implicit_ode!(du, u, p, t)
        return apply_heat!(du, u, p[1])
    end

    function explicit_ode!(du, u, p, t)
        fill!(du, 0.0)
        return nothing
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        scale = p[1]
        for i in 1:n_vars
            J[i, i] = -2.0 * scale * inv_dx2
            if i > 1
                J[i, i - 1] = scale * inv_dx2
            end
            if i < n_vars
                J[i, i + 1] = scale * inv_dx2
            end
        end
        return nothing
    end

    function kernel_ode(u, p, t)
        scale = p[1]
        return SVector{n_vars, Float64}(ntuple(n_vars) do i
            left = i == 1 ? 0.0 : u[i - 1]
            right = i == n_vars ? 0.0 : u[i + 1]
            scale * inv_dx2 * (left - 2.0 * u[i] + right)
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
            SciMLBase.ODEFunction(implicit_ode!; jac=jac!, tgrad=zero_tgrad!),
            SciMLBase.ODEFunction(explicit_ode!; jac=zero_jac!, tgrad=zero_tgrad!),
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
