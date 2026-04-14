# Julia GPU Reference Solvers

This folder contains Julia-based reference solvers that the Python tests call through
`run_solver.jl`. The Julia code uses `DiffEqGPU.jl` so the reference trajectories are
computed on the GPU rather than by a CPU fallback.

## Backends

`EnsembleGPUArray`
uses the ordinary SciML solver implementations on GPU-backed arrays. It supports a
wider range of algorithms, including `KenCarp5`, but stiff methods generally need the
problem to provide analytical derivative helpers such as Jacobians and time
derivatives.

`EnsembleGPUKernel`
uses specialized GPU kernels for a smaller set of compatible solvers. It can be
faster when the ODE function fits the kernel restrictions, but it requires out-of-place
`StaticArrays`-style system definitions and it does not offer a direct `KenCarp5`
kernel implementation in DiffEqGPU today.

In this test harness:

- `Tsit5` runs as `Tsit5()` on `EnsembleGPUArray` and `GPUTsit5()` on `EnsembleGPUKernel`.
- `Kvaerno5` runs as `Kvaerno5()` on `EnsembleGPUArray` and `GPUKvaerno5()` on `EnsembleGPUKernel`.
- `Rodas5` runs only on `EnsembleGPUArray`; the kernel case is skipped from Python because
  `GPURodas5P()` is not compiling reliably for these reference problems under the current
  DiffEqGPU kernel path.
- `KenCarp5` runs only on `EnsembleGPUArray`; the kernel case is skipped from Python with
  an explicit reason.

## Environment

The Python wrappers launch Julia with:

```bash
julia --project=tests/reference_solvers/julia tests/reference_solvers/julia/run_solver.jl ...
```

If the local Julia environment has not been instantiated yet, activate this project and
run:

```julia
using Pkg
Pkg.instantiate()
```
