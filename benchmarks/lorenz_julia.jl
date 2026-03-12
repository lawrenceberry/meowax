"""
Julia GPU ensemble benchmark for Lorenz equation (paper baseline).

Reproduces the benchmark flow from GPUODEBenchmarks:
- DiffEqGPU.vectorized_solve (fixed dt)
- DiffEqGPU.vectorized_asolve (adaptive dt)

Usage:
    julia benchmarks/lorenz_julia.jl [N]

N: number of trajectories (default 8192)

Outputs JSON to stdout with minimum benchmark times in milliseconds.
"""

using DiffEqGPU, StaticArrays
using CUDA
using JSON
using SimpleDiffEq

numberOfParameters = length(ARGS) >= 1 ? parse(Int64, ARGS[1]) : 8192

const SOLVER = GPUTsit5()
const SOLVER_NAME = "GPUTsit5"

function lorenz(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - 2.666f0 * u[3]
    return @SVector [du1, du2, du3]
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 1.0f0)
p = @SArray [21.0f0]

lorenzProblem = DiffEqGPU.ODEProblem(lorenz, u0, tspan, p)
parameterList = range(0.0f0, stop = 21.0f0, length = numberOfParameters)
prob_func = (prob, i, repeat) -> DiffEqGPU.remake(prob, p = @SArray [parameterList[i]])
ensembleProb = DiffEqGPU.EnsembleProblem(lorenzProblem, prob_func = prob_func)

function min_time_ms(f; repeats = 5)
    best = Inf
    for _ in 1:repeats
        CUDA.synchronize()
        t0 = time_ns()
        f()
        CUDA.synchronize()
        dt_ms = (time_ns() - t0) / 1e6
        best = min(best, dt_ms)
    end
    return best
end

fixed_min_ms = 0.0
adaptive_min_ms = 0.0
fixed_allocs = -1
adaptive_allocs = -1

# Strict paper path: build per-trajectory problems and use vectorized APIs.
I = 1:numberOfParameters
if ensembleProb.safetycopy
    probs = map(I) do i
        p = ensembleProb.prob_func(deepcopy(ensembleProb.prob), i, 1)
        convert(DiffEqGPU.ImmutableODEProblem, p)
    end
else
    probs = map(I) do i
        p = ensembleProb.prob_func(ensembleProb.prob, i, 1)
        convert(DiffEqGPU.ImmutableODEProblem, p)
    end
end

probs = cu(probs)

fixed_min_ms = min_time_ms() do
    CUDA.@sync DiffEqGPU.vectorized_solve(probs, ensembleProb.prob,
                                          SOLVER;
                                          save_everystep = false,
                                          dt = 0.001f0)
end
adaptive_min_ms = min_time_ms() do
    CUDA.@sync DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob,
                                           SOLVER;
                                           dt = 0.001f0,
                                           reltol = 1.0f-8,
                                           abstol = 1.0f-8)
end

result = Dict(
    "mode" => "gpu_lorenz_ensemble",
    "backend" => "CUDA",
    "dtype" => "Float32",
    "solver" => SOLVER_NAME,
    "n_trajectories" => numberOfParameters,
    "fixed_dt" => Dict(
        "dt" => 0.001f0,
        "min_time_ms" => fixed_min_ms,
        "allocs" => fixed_allocs,
    ),
    "adaptive_dt" => Dict(
        "dt" => 0.001f0,
        "reltol" => 1.0f-8,
        "abstol" => 1.0f-8,
        "min_time_ms" => adaptive_min_ms,
        "allocs" => adaptive_allocs,
    ),
    "benchmark_backend" => "manual_min_time",
)
println(JSON.json(result))
