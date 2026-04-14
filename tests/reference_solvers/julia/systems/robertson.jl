function make_robertson_spec(config)
    n_vars = 3

    function ode!(du, u, p, t)
        k1, k2, k3 = p
        du[1] = -k1 * u[1] + k2 * u[2] * u[3]
        du[2] = k1 * u[1] - k2 * u[2] * u[3] - k3 * u[2]^2
        du[3] = k3 * u[2]^2
        return nothing
    end

    function explicit_ode!(du, u, p, t)
        k1 = p[1]
        du[1] = -k1 * u[1]
        du[2] = k1 * u[1]
        du[3] = 0.0
        return nothing
    end

    function implicit_ode!(du, u, p, t)
        k2, k3 = p[2], p[3]
        du[1] = k2 * u[2] * u[3]
        du[2] = -k2 * u[2] * u[3] - k3 * u[2]^2
        du[3] = k3 * u[2]^2
        return nothing
    end

    function jac!(J, u, p, t)
        k1, k2, k3 = p
        fill!(J, 0.0)
        J[1, 1] = -k1
        J[1, 2] = k2 * u[3]
        J[1, 3] = k2 * u[2]
        J[2, 1] = k1
        J[2, 2] = -k2 * u[3] - 2.0 * k3 * u[2]
        J[2, 3] = -k2 * u[2]
        J[3, 2] = 2.0 * k3 * u[2]
        return nothing
    end

    function explicit_jac!(J, u, p, t)
        k1 = p[1]
        fill!(J, 0.0)
        J[1, 1] = -k1
        J[2, 1] = k1
        return nothing
    end

    function implicit_jac!(J, u, p, t)
        k2, k3 = p[2], p[3]
        fill!(J, 0.0)
        J[1, 2] = k2 * u[3]
        J[1, 3] = k2 * u[2]
        J[2, 2] = -k2 * u[3] - 2.0 * k3 * u[2]
        J[2, 3] = -k2 * u[2]
        J[3, 2] = 2.0 * k3 * u[2]
        return nothing
    end

    function kernel_ode(u, p, t)
        k1, k2, k3 = p
        return @SVector [
            -k1 * u[1] + k2 * u[2] * u[3],
            k1 * u[1] - k2 * u[2] * u[3] - k3 * u[2]^2,
            k3 * u[2]^2,
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
