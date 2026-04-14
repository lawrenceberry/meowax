function make_nn_reactions_spec(config)
    n_vars = require_config_int(config, "n_vars")
    edge_count = n_vars - 1
    kf = Tuple(logspace(-2.0, 6.0, edge_count))
    kb = Tuple(logspace(6.0, -2.0, edge_count))
    zero_jac! = make_zero_jac!(n_vars)

    function apply_nn!(du, u, scale)
        du[1] = scale * (-kf[1] * u[1] + kb[1] * u[2])
        for i in 2:(n_vars - 1)
            du[i] = scale * (
                kf[i - 1] * u[i - 1] -
                (kb[i - 1] + kf[i]) * u[i] +
                kb[i] * u[i + 1]
            )
        end
        du[n_vars] = scale * (kf[end] * u[n_vars - 1] - kb[end] * u[n_vars])
        return nothing
    end

    function ode!(du, u, p, t)
        return apply_nn!(du, u, p[1])
    end

    function implicit_ode!(du, u, p, t)
        return apply_nn!(du, u, p[1])
    end

    function explicit_ode!(du, u, p, t)
        fill!(du, 0.0)
        return nothing
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        scale = p[1]
        J[1, 1] = -scale * kf[1]
        J[1, 2] = scale * kb[1]
        for i in 2:(n_vars - 1)
            J[i, i - 1] = scale * kf[i - 1]
            J[i, i] = -scale * (kb[i - 1] + kf[i])
            J[i, i + 1] = scale * kb[i]
        end
        J[n_vars, n_vars - 1] = scale * kf[end]
        J[n_vars, n_vars] = -scale * kb[end]
        return nothing
    end

    function kernel_ode(u, p, t)
        scale = p[1]
        return SVector{n_vars, Float64}(ntuple(n_vars) do i
            if i == 1
                scale * (-kf[1] * u[1] + kb[1] * u[2])
            elseif i == n_vars
                scale * (kf[end] * u[n_vars - 1] - kb[end] * u[n_vars])
            else
                scale * (
                    kf[i - 1] * u[i - 1] -
                    (kb[i - 1] + kf[i]) * u[i] +
                    kb[i] * u[i + 1]
                )
            end
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
