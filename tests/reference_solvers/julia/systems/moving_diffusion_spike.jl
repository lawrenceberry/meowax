function make_moving_diffusion_spike_spec(config)
    n_vars = require_config_int(config, "n_vars")
    dx = 1.0 / (n_vars + 1)
    inv_dx2 = 1.0 / dx^2
    background_diffusivity = 1e-4
    spike_diffusivity = 2e-1
    spike_width = 5e-2
    spike_center = 0.55
    spike_time_center = 0.5
    spike_time_width = 0.2

    spatial_profile = Tuple([
        begin
            x_face = (i - 1) * dx
            if i == 1 || i == n_vars + 1
                0.0
            else
                exp(-((x_face - spike_center) / spike_width)^2)
            end
        end
        for i in 1:(n_vars + 1)
    ])

    background_faces = Tuple([
        i == 1 || i == n_vars + 1 ? 0.0 : background_diffusivity
        for i in 1:(n_vars + 1)
    ])

    function pulse(t, p)
        return p[1] * exp(-((t - spike_time_center) / spike_time_width)^2)
    end

    function pulse_dt(t, p)
        exp_term = exp(-((t - spike_time_center) / spike_time_width)^2)
        return p[1] * exp_term * (-2.0 * (t - spike_time_center) / spike_time_width^2)
    end

    function apply_faces!(du, u, faces, scale)
        for i in 1:n_vars
            left_face = faces[i]
            right_face = faces[i + 1]
            left = i == 1 ? 0.0 : left_face * u[i - 1]
            right = i == n_vars ? 0.0 : right_face * u[i + 1]
            du[i] = scale * inv_dx2 * (left - (left_face + right_face) * u[i] + right)
        end
        return nothing
    end

    function spike_faces(t, p)
        amplitude = spike_diffusivity * pulse(t, p)
        return ntuple(i -> amplitude * spatial_profile[i], n_vars + 1)
    end

    function spike_faces_dt(t, p)
        amplitude = spike_diffusivity * pulse_dt(t, p)
        return ntuple(i -> amplitude * spatial_profile[i], n_vars + 1)
    end

    function ode!(du, u, p, t)
        faces = ntuple(i -> background_faces[i] + spike_faces(t, p)[i], n_vars + 1)
        return apply_faces!(du, u, faces, 1.0)
    end

    function explicit_ode!(du, u, p, t)
        return apply_faces!(du, u, background_faces, 1.0)
    end

    function implicit_ode!(du, u, p, t)
        return apply_faces!(du, u, spike_faces(t, p), 1.0)
    end

    function fill_face_jacobian!(J, faces)
        fill!(J, 0.0)
        for i in 1:n_vars
            left_face = faces[i]
            right_face = faces[i + 1]
            J[i, i] = -inv_dx2 * (left_face + right_face)
            if i > 1
                J[i, i - 1] = inv_dx2 * left_face
            end
            if i < n_vars
                J[i, i + 1] = inv_dx2 * right_face
            end
        end
        return nothing
    end

    function jac!(J, u, p, t)
        faces = ntuple(i -> background_faces[i] + spike_faces(t, p)[i], n_vars + 1)
        return fill_face_jacobian!(J, faces)
    end

    function explicit_jac!(J, u, p, t)
        return fill_face_jacobian!(J, background_faces)
    end

    function implicit_jac!(J, u, p, t)
        return fill_face_jacobian!(J, spike_faces(t, p))
    end

    function tgrad!(dT, u, p, t)
        return apply_faces!(dT, u, spike_faces_dt(t, p), 1.0)
    end

    function explicit_tgrad!(dT, u, p, t)
        fill!(dT, 0.0)
        return nothing
    end

    function implicit_tgrad!(dT, u, p, t)
        return apply_faces!(dT, u, spike_faces_dt(t, p), 1.0)
    end

    function kernel_faces(t, p, which)
        if which === :background
            return background_faces
        elseif which === :spike
            return spike_faces(t, p)
        end
        spike = spike_faces(t, p)
        return ntuple(i -> background_faces[i] + spike[i], n_vars + 1)
    end

    function kernel_apply(u, p, t, which)
        faces = kernel_faces(t, p, which)
        return SVector{n_vars, Float64}(ntuple(n_vars) do i
            left_face = faces[i]
            right_face = faces[i + 1]
            left = i == 1 ? 0.0 : left_face * u[i - 1]
            right = i == n_vars ? 0.0 : right_face * u[i + 1]
            inv_dx2 * (left - (left_face + right_face) * u[i] + right)
        end)
    end

    return ReferenceSystemSpec(
        build_array_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(ode!; jac=jac!, tgrad=tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_array_split_problem=(y0, tspan, p0) -> SciMLBase.SplitODEProblem(
            SciMLBase.ODEFunction(implicit_ode!; jac=implicit_jac!, tgrad=implicit_tgrad!),
            SciMLBase.ODEFunction(explicit_ode!; jac=explicit_jac!, tgrad=explicit_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_kernel_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem{false}(
            (u, p, t) -> kernel_apply(u, p, t, :full),
            vector_to_svector(y0),
            tspan,
            vector_to_svector(p0),
        ),
    )
end
