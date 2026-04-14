function make_reversible_trapping_spec(config)
    n_vars = require_config_int(config, "n_vars")
    n_cells = div(n_vars, 2)
    dx = 1.0 / (n_cells - 1)
    diffusion_scale = 2e-2 / dx^2
    adsorption_rate = 3e4
    desorption_rate = 3e3

    function diffusion_value(u, i)
        left = i == 1 ? u[i] : u[i - 1]
        right = i == n_cells ? u[i] : u[i + 1]
        return diffusion_scale * (left - 2.0 * u[i] + right)
    end

    function ode!(du, u, p, t)
        s = p[1]
        for i in 1:n_cells
            mobile = u[i]
            trapped = u[n_cells + i]
            du[i] = s * (diffusion_value(u, i) - adsorption_rate * mobile + desorption_rate * trapped)
            du[n_cells + i] = s * (adsorption_rate * mobile - desorption_rate * trapped)
        end
        return nothing
    end

    function explicit_ode!(du, u, p, t)
        s = p[1]
        for i in 1:n_cells
            du[i] = s * diffusion_value(u, i)
            du[n_cells + i] = 0.0
        end
        return nothing
    end

    function implicit_ode!(du, u, p, t)
        s = p[1]
        for i in 1:n_cells
            mobile = u[i]
            trapped = u[n_cells + i]
            du[i] = s * (-adsorption_rate * mobile + desorption_rate * trapped)
            du[n_cells + i] = s * (adsorption_rate * mobile - desorption_rate * trapped)
        end
        return nothing
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        s = p[1]
        for i in 1:n_cells
            J[i, i] += s * (-2.0 * diffusion_scale - adsorption_rate)
            if i > 1
                J[i, i - 1] = s * diffusion_scale
            end
            if i < n_cells
                J[i, i + 1] = s * diffusion_scale
            end
            if i == 1
                J[i, i] += s * diffusion_scale
            end
            if i == n_cells
                J[i, i] += s * diffusion_scale
            end
            J[i, n_cells + i] = s * desorption_rate
            J[n_cells + i, i] = s * adsorption_rate
            J[n_cells + i, n_cells + i] = -s * desorption_rate
        end
        return nothing
    end

    function explicit_jac!(J, u, p, t)
        fill!(J, 0.0)
        s = p[1]
        for i in 1:n_cells
            J[i, i] += s * (-2.0 * diffusion_scale)
            if i > 1
                J[i, i - 1] = s * diffusion_scale
            end
            if i < n_cells
                J[i, i + 1] = s * diffusion_scale
            end
            if i == 1
                J[i, i] += s * diffusion_scale
            end
            if i == n_cells
                J[i, i] += s * diffusion_scale
            end
        end
        return nothing
    end

    function implicit_jac!(J, u, p, t)
        fill!(J, 0.0)
        s = p[1]
        for i in 1:n_cells
            J[i, i] = -s * adsorption_rate
            J[i, n_cells + i] = s * desorption_rate
            J[n_cells + i, i] = s * adsorption_rate
            J[n_cells + i, n_cells + i] = -s * desorption_rate
        end
        return nothing
    end

    function kernel_ode(u, p, t)
        s = p[1]
        return SVector{n_vars, Float64}(ntuple(n_vars) do idx
            if idx <= n_cells
                left = idx == 1 ? u[idx] : u[idx - 1]
                right = idx == n_cells ? u[idx] : u[idx + 1]
                mobile = u[idx]
                trapped = u[n_cells + idx]
                s * (diffusion_scale * (left - 2.0 * mobile + right) - adsorption_rate * mobile + desorption_rate * trapped)
            else
                cell = idx - n_cells
                mobile = u[cell]
                trapped = u[idx]
                s * (adsorption_rate * mobile - desorption_rate * trapped)
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
