include("damped_rotation.jl")
include("heat_equation.jl")
include("bateman.jl")
include("nn_reactions.jl")
include("reversible_trapping.jl")
include("moving_diffusion_spike.jl")
include("lorenz.jl")
include("robertson.jl")
include("vdp.jl")
include("kaps.jl")

function make_system_spec(system_name::String, config)
    if system_name == "damped_rotation"
        return make_damped_rotation_spec(config)
    elseif system_name == "heat_equation"
        return make_heat_equation_spec(config)
    elseif system_name == "bateman"
        return make_bateman_spec(config)
    elseif system_name == "nn_reactions"
        return make_nn_reactions_spec(config)
    elseif system_name == "reversible_trapping"
        return make_reversible_trapping_spec(config)
    elseif system_name == "moving_diffusion_spike"
        return make_moving_diffusion_spike_spec(config)
    elseif system_name == "lorenz"
        return make_lorenz_spec(config)
    elseif system_name == "robertson"
        return make_robertson_spec(config)
    elseif system_name == "vdp"
        return make_vdp_spec(config)
    elseif system_name == "kaps"
        return make_kaps_spec(config)
    end
    error("Unknown reference ODE system '$system_name'")
end
