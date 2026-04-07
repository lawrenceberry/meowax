// CUDA helper kernels for the fixed-structure cuDSS sparse fast path.

#include <cuda_runtime.h>

#include <cstdint>

template <typename Scalar>
__global__ void CopyNegateKernel(
    Scalar* __restrict__ out,
    const Scalar* __restrict__ src,
    int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < total; i += stride) {
    out[i] = -src[i];
  }
}

template <typename Scalar>
__global__ void CopyKernel(
    Scalar* __restrict__ out,
    const Scalar* __restrict__ src,
    int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < total; i += stride) {
    out[i] = src[i];
  }
}

template <typename Scalar>
__global__ void AddDiagonalKernel(
    Scalar* __restrict__ out,
    const int32_t* __restrict__ diag_offsets,
    const double* __restrict__ dt,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size,
    double gamma) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t diag_total = batch_size * n_vars;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < diag_total; i += stride) {
    int64_t batch = i / n_vars;
    int64_t row = i % n_vars;
    int32_t diag = diag_offsets[row];
    if (diag >= 0) {
      out[batch * nnz + diag] += static_cast<Scalar>(1.0 / (dt[batch] * gamma));
    }
  }
}

template <typename Scalar>
__global__ void BatchedCsrMatvecKernel(
    Scalar* __restrict__ out,
    const int32_t* __restrict__ indptr,
    const int32_t* __restrict__ indices,
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ vec,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch_size * n_vars;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < total; i += stride) {
    int64_t batch = i / n_vars;
    int64_t row = i % n_vars;
    int32_t start = indptr[row];
    int32_t end = indptr[row + 1];
    Scalar acc = Scalar{0};
    for (int32_t k = start; k < end && k < nnz; ++k) {
      acc += values[batch * nnz + k] * vec[batch * n_vars + indices[k]];
    }
    out[i] = acc;
  }
}

template <typename Scalar>
__global__ void StageStateKernel(
    Scalar* __restrict__ out,
    const Scalar* __restrict__ y,
    const Scalar* __restrict__ k1,
    double a1,
    const Scalar* __restrict__ k2,
    double a2,
    const Scalar* __restrict__ k3,
    double a3,
    const Scalar* __restrict__ k4,
    double a4,
    const Scalar* __restrict__ k5,
    double a5,
    const Scalar* __restrict__ k6,
    double a6,
    const Scalar* __restrict__ k7,
    double a7,
    int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < total; i += stride) {
    Scalar value = y[i];
    if (k1 != nullptr) value += static_cast<Scalar>(a1) * k1[i];
    if (k2 != nullptr) value += static_cast<Scalar>(a2) * k2[i];
    if (k3 != nullptr) value += static_cast<Scalar>(a3) * k3[i];
    if (k4 != nullptr) value += static_cast<Scalar>(a4) * k4[i];
    if (k5 != nullptr) value += static_cast<Scalar>(a5) * k5[i];
    if (k6 != nullptr) value += static_cast<Scalar>(a6) * k6[i];
    if (k7 != nullptr) value += static_cast<Scalar>(a7) * k7[i];
    out[i] = value;
  }
}

template <typename Scalar>
__global__ void StageRhsKernel(
    Scalar* __restrict__ out,
    const Scalar* __restrict__ du,
    const double* __restrict__ dt,
    const Scalar* __restrict__ k1,
    double c1,
    const Scalar* __restrict__ k2,
    double c2,
    const Scalar* __restrict__ k3,
    double c3,
    const Scalar* __restrict__ k4,
    double c4,
    const Scalar* __restrict__ k5,
    double c5,
    const Scalar* __restrict__ k6,
    double c6,
    const Scalar* __restrict__ k7,
    double c7,
    int64_t n_vars,
    int64_t batch_size) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch_size * n_vars;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < total; i += stride) {
    int64_t batch = i / n_vars;
    Scalar inv_dt = static_cast<Scalar>(1.0 / dt[batch]);
    Scalar value = du[i];
    if (k1 != nullptr) value += static_cast<Scalar>(c1) * k1[i] * inv_dt;
    if (k2 != nullptr) value += static_cast<Scalar>(c2) * k2[i] * inv_dt;
    if (k3 != nullptr) value += static_cast<Scalar>(c3) * k3[i] * inv_dt;
    if (k4 != nullptr) value += static_cast<Scalar>(c4) * k4[i] * inv_dt;
    if (k5 != nullptr) value += static_cast<Scalar>(c5) * k5[i] * inv_dt;
    if (k6 != nullptr) value += static_cast<Scalar>(c6) * k6[i] * inv_dt;
    if (k7 != nullptr) value += static_cast<Scalar>(c7) * k7[i] * inv_dt;
    out[i] = value;
  }
}

template <typename Scalar>
__global__ void AddKernel(
    Scalar* __restrict__ out,
    const Scalar* __restrict__ lhs,
    const Scalar* __restrict__ rhs,
    int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < total; i += stride) {
    out[i] = lhs[i] + rhs[i];
  }
}

static inline int LaunchBlocks(int64_t total) {
  int blocks = static_cast<int>((total + 255) / 256);
  return blocks > 65535 ? 65535 : blocks;
}

extern "C" void launch_copy_negate_f32(
    float* out, const float* src, int64_t total, void* stream) {
  CopyNegateKernel<float><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, src, total);
}

extern "C" void launch_copy_negate_f64(
    double* out, const double* src, int64_t total, void* stream) {
  CopyNegateKernel<double><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, src, total);
}

extern "C" void launch_copy_base_add_diagonal_f32(
    float* out,
    const float* base,
    const int32_t* diag_offsets,
    const double* dt,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size,
    double gamma,
    void* stream) {
  int64_t total = batch_size * nnz;
  CopyKernel<float><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, base, total);
  AddDiagonalKernel<float><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, diag_offsets, dt, n_vars, nnz, batch_size, gamma);
}

extern "C" void launch_copy_base_add_diagonal_f64(
    double* out,
    const double* base,
    const int32_t* diag_offsets,
    const double* dt,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size,
    double gamma,
    void* stream) {
  int64_t total = batch_size * nnz;
  CopyKernel<double><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, base, total);
  AddDiagonalKernel<double><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, diag_offsets, dt, n_vars, nnz, batch_size, gamma);
}

extern "C" void launch_copy_negate_add_diagonal_f32(
    float* out,
    const float* src,
    const int32_t* diag_offsets,
    const double* dt,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size,
    double gamma,
    void* stream) {
  int64_t total = batch_size * nnz;
  CopyNegateKernel<float><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, src, total);
  AddDiagonalKernel<float><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, diag_offsets, dt, n_vars, nnz, batch_size, gamma);
}

extern "C" void launch_copy_negate_add_diagonal_f64(
    double* out,
    const double* src,
    const int32_t* diag_offsets,
    const double* dt,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size,
    double gamma,
    void* stream) {
  int64_t total = batch_size * nnz;
  CopyNegateKernel<double><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, src, total);
  AddDiagonalKernel<double><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, diag_offsets, dt, n_vars, nnz, batch_size, gamma);
}

extern "C" void launch_batched_csr_matvec_f32(
    float* out,
    const int32_t* indptr,
    const int32_t* indices,
    const float* values,
    const float* vec,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size,
    void* stream) {
  BatchedCsrMatvecKernel<float><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, indptr, indices, values, vec, n_vars, nnz, batch_size);
}

extern "C" void launch_batched_csr_matvec_f64(
    double* out,
    const int32_t* indptr,
    const int32_t* indices,
    const double* values,
    const double* vec,
    int64_t n_vars,
    int64_t nnz,
    int64_t batch_size,
    void* stream) {
  BatchedCsrMatvecKernel<double><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, indptr, indices, values, vec, n_vars, nnz, batch_size);
}

extern "C" void launch_stage_state_f32(
    float* out, const float* y, const float* k1, double a1,
    const float* k2, double a2, const float* k3, double a3,
    const float* k4, double a4, const float* k5, double a5,
    const float* k6, double a6, const float* k7, double a7,
    int64_t total, void* stream) {
  StageStateKernel<float><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, y, k1, a1, k2, a2, k3, a3, k4, a4, k5, a5, k6, a6, k7, a7, total);
}

extern "C" void launch_stage_state_f64(
    double* out, const double* y, const double* k1, double a1,
    const double* k2, double a2, const double* k3, double a3,
    const double* k4, double a4, const double* k5, double a5,
    const double* k6, double a6, const double* k7, double a7,
    int64_t total, void* stream) {
  StageStateKernel<double><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, y, k1, a1, k2, a2, k3, a3, k4, a4, k5, a5, k6, a6, k7, a7, total);
}

extern "C" void launch_stage_rhs_f32(
    float* out, const float* du, const double* dt, const float* k1, double c1,
    const float* k2, double c2, const float* k3, double c3,
    const float* k4, double c4, const float* k5, double c5,
    const float* k6, double c6, const float* k7, double c7,
    int64_t n_vars, int64_t batch_size, void* stream) {
  StageRhsKernel<float><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, du, dt, k1, c1, k2, c2, k3, c3, k4, c4, k5, c5, k6, c6, k7, c7, n_vars, batch_size);
}

extern "C" void launch_stage_rhs_f64(
    double* out, const double* du, const double* dt, const double* k1, double c1,
    const double* k2, double c2, const double* k3, double c3,
    const double* k4, double c4, const double* k5, double c5,
    const double* k6, double c6, const double* k7, double c7,
    int64_t n_vars, int64_t batch_size, void* stream) {
  StageRhsKernel<double><<<LaunchBlocks(batch_size * n_vars), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, du, dt, k1, c1, k2, c2, k3, c3, k4, c4, k5, c5, k6, c6, k7, c7, n_vars, batch_size);
}

extern "C" void launch_add_vectors_f32(
    float* out, const float* lhs, const float* rhs, int64_t total, void* stream) {
  AddKernel<float><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, lhs, rhs, total);
}

extern "C" void launch_add_vectors_f64(
    double* out, const double* lhs, const double* rhs, int64_t total, void* stream) {
  AddKernel<double><<<LaunchBlocks(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out, lhs, rhs, total);
}
