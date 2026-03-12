"""
Julia GPU ensemble benchmark for Robertson equation.

Usage:
    julia robertson_julia.jl [N]

N: number of trajectories (default 2)

Uses GPURosenbrock23 + EnsembleGPUKernel (CUDA, Float64).
Robertson is formulated as a DAE with mass matrix to enforce conservation.
Outputs JSON to stdout with timing and results.
"""

using OrdinaryDiffEq, StaticArrays, LinearAlgebra, JSON, Random, CUDA, DiffEqGPU

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2

# Robertson DAE form with mass matrix (enforces y1+y2+y3=1)
# Parameter convention matches DiffEqGPU.jl: k1=0.04, k2=3e7, k3=1e4
#   dy1/dt = -k1*y1 + k3*y2*y3
#   dy2/dt =  k1*y1 - k2*y2^2 - k3*y2*y3
#   0      =  y1 + y2 + y3 - 1
function robertson(u, p, t)
    y1, y2, y3 = u
    k1, k2, k3 = p
    return @SVector [
        -k1 * y1 + k3 * y2 * y3,
         k1 * y1 - k2 * y2^2 - k3 * y2 * y3,
         y1 + y2 + y3 - 1,
    ]
end

M = @SMatrix [
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 0.0
]
ff = ODEFunction(robertson, mass_matrix=M)

params = @SVector [0.04, 3e7, 1e4]
u0 = @SVector [1.0, 0.0, 0.0]
tspan = (0.0, 1e5)
prob = ODEProblem{false}(ff, u0, tspan, params)

# Generate perturbed parameters (±10%)
function make_perturbed_params(N::Int)
    base = [0.04, 3e7, 1e4]
    rng = MersenneTwister(42)
    return [SVector{3,Float64}(base .* (1 .+ 0.1 .* (2 .* rand(rng, 3) .- 1))) for _ in 1:N]
end

params_list = make_perturbed_params(N)
prob_func = (prob, i, repeat) -> remake(prob, p=params_list[i])
monteprob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)

backend = CUDA.CUDABackend()

# Warmup (compile GPU kernel)
warmup_monte = EnsembleProblem(prob, prob_func=(p,i,r)->p, safetycopy=false)
solve(warmup_monte, GPURosenbrock23(), EnsembleGPUKernel(backend),
      trajectories=2, adaptive=true, dt=0.1, abstol=1e-10, reltol=1e-8,
      save_everystep=false)
CUDA.synchronize()

# Benchmark
CUDA.synchronize()
t_start = time()
sol = solve(monteprob, GPURosenbrock23(), EnsembleGPUKernel(backend),
            trajectories=N, adaptive=true, dt=0.1, abstol=1e-10, reltol=1e-8,
            save_everystep=false)
CUDA.synchronize()
elapsed = time() - t_start

finals = [[s.u[end][j] for j in 1:3] for s in sol.u]
conservations = [sum(s.u[end]) for s in sol.u]
result = Dict(
    "mode" => "gpu_ensemble",
    "solver" => "GPURosenbrock23",
    "backend" => "CUDA",
    "dtype" => "Float64",
    "elapsed_seconds" => elapsed,
    "n_trajectories" => N,
    "y_finals" => finals,
    "conservations" => conservations,
    "converged" => sol.converged,
)
println(JSON.json(result))
