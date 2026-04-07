// Tiny CUDA kernels for adding 1/(dt*gamma) to diagonal positions in the
// batched union-CSR values buffer.  This keeps the entire static prepare
// path on-device, eliminating the host round-trip that previously dominated
// cuDSS solve time.

#include <cuda_runtime.h>

#include <cstdint>

template <typename Scalar>
__global__ void CopyBaseKernel(
    Scalar* __restrict__ out,
    const Scalar* __restrict__ base,
    int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t i = tid; i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = base[i];
  }
}

template <typename Scalar>
__global__ void AddDiagonalKernel(
    Scalar* __restrict__ out,
    const int32_t* __restrict__ diag_offsets,
    const double* __restrict__ dt,
    int64_t n_vars,
    int64_t union_nnz,
    int64_t batch_size,
    double gamma) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t diag_total = batch_size * n_vars;
  for (int64_t i = tid; i < diag_total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t b = i / n_vars;
    int64_t row = i % n_vars;
    int32_t diag = diag_offsets[row];
    if (diag >= 0) {
      out[b * union_nnz + diag] += static_cast<Scalar>(1.0 / (dt[b] * gamma));
    }
  }
}

// C-linkage launcher functions called from rodas5_cudss_ffi.cc.

extern "C" void launch_copy_base_add_diagonal_f32(
    float* out,
    const float* base,
    const int32_t* diag_offsets,
    const double* dt,
    int64_t n_vars,
    int64_t union_nnz,
    int64_t batch_size,
    double gamma,
    void* stream) {
  int64_t total = batch_size * union_nnz;
  int threads = 256;
  int copy_blocks = static_cast<int>((total + threads - 1) / threads);
  if (copy_blocks > 65535) copy_blocks = 65535;
  CopyBaseKernel<float><<<copy_blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      out, base, total);
  int64_t diag_total = batch_size * n_vars;
  int diag_blocks = static_cast<int>((diag_total + threads - 1) / threads);
  if (diag_blocks > 65535) diag_blocks = 65535;
  AddDiagonalKernel<float><<<diag_blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      out, diag_offsets, dt, n_vars, union_nnz, batch_size, gamma);
}

extern "C" void launch_copy_base_add_diagonal_f64(
    double* out,
    const double* base,
    const int32_t* diag_offsets,
    const double* dt,
    int64_t n_vars,
    int64_t union_nnz,
    int64_t batch_size,
    double gamma,
    void* stream) {
  int64_t total = batch_size * union_nnz;
  int threads = 256;
  int copy_blocks = static_cast<int>((total + threads - 1) / threads);
  if (copy_blocks > 65535) copy_blocks = 65535;
  CopyBaseKernel<double><<<copy_blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      out, base, total);
  int64_t diag_total = batch_size * n_vars;
  int diag_blocks = static_cast<int>((diag_total + threads - 1) / threads);
  if (diag_blocks > 65535) diag_blocks = 65535;
  AddDiagonalKernel<double><<<diag_blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      out, diag_offsets, dt, n_vars, union_nnz, batch_size, gamma);
}
