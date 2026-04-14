function make_lorenz_spec(config)
    n_vars = 3
    zero_jac! = make_zero_jac!(n_vars)
    sigma = 10.0
    beta = 8.0 / 3.0

    function ode!(du, u, p, t)
        rho = p[1]
        du[1] = sigma * (u[2] - u[1])
        du[2] = u[1] * (rho - u[3]) - u[2]
        du[3] = u[1] * u[2] - beta * u[3]
        return nothing
    end

    function jac!(J, u, p, t)
        rho = p[1]
        fill!(J, 0.0)
        J[1, 1] = -sigma
        J[1, 2] = sigma
        J[2, 1] = rho - u[3]
        J[2, 2] = -1.0
        J[2, 3] = -u[1]
        J[3, 1] = u[2]
        J[3, 2] = u[1]
        J[3, 3] = -beta
        return nothing
    end

    function explicit_ode!(du, u, p, t)
        return ode!(du, u, p, t)
    end

    function implicit_ode!(du, u, p, t)
        fill!(du, 0.0)
        return nothing
    end

    function kernel_ode(u, p, t)
        rho = p[1]
        return @SVector [
            sigma * (u[2] - u[1]),
            u[1] * (rho - u[3]) - u[2],
            u[1] * u[2] - beta * u[3],
        ]
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
            SciMLBase.ODEFunction(explicit_ode!; jac=jac!, tgrad=zero_tgrad!),
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
