#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xla/ffi/api/ffi.h"

// Kernel launchers defined in cudss_diagonal_kernel.cu (linked when CUDA lang available).
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
extern "C" void launch_copy_base_add_diagonal_f32(
    float* out, const float* base, const int32_t* diag_offsets,
    const double* dt, int64_t n_vars, int64_t union_nnz,
    int64_t batch_size, double gamma, void* stream);
extern "C" void launch_copy_base_add_diagonal_f64(
    double* out, const double* base, const int32_t* diag_offsets,
    const double* dt, int64_t n_vars, int64_t union_nnz,
    int64_t batch_size, double gamma, void* stream);
#endif

namespace ffi = xla::ffi;

namespace {

constexpr int kPrecisionFp64 = 0;
constexpr int kPrecisionFp32 = 1;

struct PatternKey {
  int64_t n_vars;
  int64_t batch_size;
  std::vector<int32_t> indptr;
  std::vector<int32_t> indices;

  bool operator==(const PatternKey& other) const {
    return n_vars == other.n_vars && batch_size == other.batch_size &&
           indptr == other.indptr && indices == other.indices;
  }
};

struct PatternKeyHash {
  std::size_t operator()(const PatternKey& key) const {
    std::size_t h = std::hash<int64_t>{}(key.n_vars) ^ (std::hash<int64_t>{}(key.batch_size) << 1);
    for (int32_t v : key.indptr) {
      h ^= std::hash<int32_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    for (int32_t v : key.indices) {
      h ^= std::hash<int32_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t count = 0;

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  bool Resize(size_t new_count) {
    if (new_count == count && ptr != nullptr) {
      return true;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      count = 0;
    }
    if (new_count == 0) {
      return true;
    }
    if (cudaMalloc(&ptr, sizeof(T) * new_count) != cudaSuccess) {
      ptr = nullptr;
      count = 0;
      return false;
    }
    count = new_count;
    return true;
  }
};

struct PatternCache {
  PatternKey key;
  int64_t union_nnz = 0;
  std::vector<int32_t> diag_offsets;
  DeviceBuffer<int32_t> d_indptr;
  DeviceBuffer<int32_t> d_indices;
  DeviceBuffer<std::byte> d_values;
  cudssData_t solver_data = nullptr;
  cudssMatrix_t matrix = nullptr;
  bool analyzed = false;
  bool factorized = false;

  // Cached base values: scattered -J values WITHOUT the 1/(dt*gamma) diagonal.
  // Used by the static fast path to skip D->H copies and ScatterUnionValues.
  std::vector<float> host_base_f32;
  std::vector<double> host_base_f64;
  DeviceBuffer<std::byte> d_base_values;     // device copy of base values
  DeviceBuffer<int32_t> d_diag_offsets;      // diagonal offsets on device
  bool has_cached_base = false;

  ~PatternCache() {
    if (matrix != nullptr) {
      cudssMatrixDestroy(matrix);
    }
  }
};

struct DenseBatchMatrices {
  DeviceBuffer<std::byte> d_rhs_dummy;
  DeviceBuffer<std::byte> d_sol_dummy;
  cudssMatrix_t rhs = nullptr;
  cudssMatrix_t solution = nullptr;

  ~DenseBatchMatrices() {
    if (rhs != nullptr) {
      cudssMatrixDestroy(rhs);
    }
    if (solution != nullptr) {
      cudssMatrixDestroy(solution);
    }
  }
};

struct Context {
  int64_t n_vars;
  int64_t batch_size;
  int precision_code;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  DenseBatchMatrices dense;
  std::unordered_map<PatternKey, std::unique_ptr<PatternCache>, PatternKeyHash> cache;
  PatternCache* active = nullptr;
  std::mutex mu;
  uint64_t analysis_count = 0;
  uint64_t factorization_count = 0;
  uint64_t solve_count = 0;

  ~Context() {
    if (handle != nullptr && config != nullptr) {
      for (auto& it : cache) {
        if (it.second->solver_data != nullptr) {
          cudssDataDestroy(handle, it.second->solver_data);
        }
      }
    }
    if (config != nullptr) {
      cudssConfigDestroy(config);
    }
    if (handle != nullptr) {
      cudssDestroy(handle);
    }
  }
};

std::mutex g_context_mu;
std::unordered_map<uint64_t, std::unique_ptr<Context>> g_contexts;
std::atomic<uint64_t> g_next_context_id{1};

ffi::Error CuDssError(const std::string& where, cudssStatus_t status) {
  std::ostringstream os;
  os << where << " failed with cuDSS status " << static_cast<int>(status);
  return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

ffi::Error CudaError(const std::string& where, cudaError_t status) {
  std::ostringstream os;
  os << where << " failed with CUDA status " << static_cast<int>(status) << ": "
     << cudaGetErrorString(status);
  return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

cudaDataType_t ValueType(int precision_code) {
  return precision_code == kPrecisionFp32 ? CUDA_R_32F : CUDA_R_64F;
}

size_t ValueBytes(int precision_code) {
  return precision_code == kPrecisionFp32 ? sizeof(float) : sizeof(double);
}

Context* LookupContext(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_context_mu);
  auto it = g_contexts.find(handle);
  return it == g_contexts.end() ? nullptr : it->second.get();
}

ffi::Error EnsureDenseMatrices(Context& ctx) {
  if (ctx.dense.rhs != nullptr && ctx.dense.solution != nullptr) {
    return ffi::Error();
  }

  const int64_t rows = ctx.n_vars;
  const int64_t cols = 1;
  const int64_t ld = ctx.n_vars;
  const size_t dense_elems = static_cast<size_t>(ctx.batch_size * ctx.n_vars);
  if (!ctx.dense.d_rhs_dummy.Resize(dense_elems * ValueBytes(ctx.precision_code)) ||
      !ctx.dense.d_sol_dummy.Resize(dense_elems * ValueBytes(ctx.precision_code))) {
    return ffi::Error(ffi::ErrorCode::kResourceExhausted, "Failed to allocate cuDSS dense dummy buffers");
  }

  cudssStatus_t rhs_status = cudssMatrixCreateDn(
      &ctx.dense.rhs,
      rows,
      cols,
      ld,
      ctx.dense.d_rhs_dummy.ptr,
      ValueType(ctx.precision_code),
      CUDSS_LAYOUT_COL_MAJOR);
  if (rhs_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixCreateDn(rhs)", rhs_status);
  }

  cudssStatus_t sol_status = cudssMatrixCreateDn(
      &ctx.dense.solution,
      rows,
      cols,
      ld,
      ctx.dense.d_sol_dummy.ptr,
      ValueType(ctx.precision_code),
      CUDSS_LAYOUT_COL_MAJOR);
  if (sol_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixCreateDn(solution)", sol_status);
  }

  return ffi::Error();
}

PatternKey BuildUnionKey(
    int64_t n_vars,
    int64_t batch_size,
    const int32_t* indptr,
    const int32_t* indices,
    const int32_t* nnz,
    int64_t max_nnz) {
  std::vector<std::vector<int32_t>> rows(n_vars);
  for (int64_t row = 0; row < n_vars; ++row) {
    rows[row].push_back(static_cast<int32_t>(row));
  }
  for (int64_t b = 0; b < batch_size; ++b) {
    const int32_t* indptr_b = indptr + b * (n_vars + 1);
    const int32_t* indices_b = indices + b * max_nnz;
    const int32_t nnz_b = nnz[b];
    for (int64_t row = 0; row < n_vars; ++row) {
      int32_t start = std::min(indptr_b[row], nnz_b);
      int32_t end = std::min(indptr_b[row + 1], nnz_b);
      for (int32_t k = start; k < end; ++k) {
        rows[row].push_back(indices_b[k]);
      }
    }
  }

  PatternKey key;
  key.n_vars = n_vars;
  key.batch_size = batch_size;
  key.indptr.reserve(n_vars + 1);
  key.indptr.push_back(0);
  for (int64_t row = 0; row < n_vars; ++row) {
    auto& cols = rows[row];
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    key.indices.insert(key.indices.end(), cols.begin(), cols.end());
    key.indptr.push_back(static_cast<int32_t>(key.indices.size()));
  }
  return key;
}

ffi::Error EnsurePatternCache(
    Context& ctx,
    const PatternKey& key,
    PatternCache*& out_cache) {
  auto it = ctx.cache.find(key);
  if (it != ctx.cache.end()) {
    out_cache = it->second.get();
    return ffi::Error();
  }

  auto cache = std::make_unique<PatternCache>();
  cache->key = key;
  cache->union_nnz = static_cast<int64_t>(key.indices.size());
  cache->diag_offsets.resize(ctx.n_vars, -1);

  for (int64_t row = 0; row < ctx.n_vars; ++row) {
    int32_t start = key.indptr[row];
    int32_t end = key.indptr[row + 1];
    auto begin = key.indices.begin() + start;
    auto found = std::lower_bound(begin, key.indices.begin() + end, static_cast<int32_t>(row));
    if (found != key.indices.begin() + end && *found == row) {
      cache->diag_offsets[row] = static_cast<int32_t>(found - key.indices.begin());
    }
  }

  if (!cache->d_indptr.Resize(key.indptr.size()) ||
      !cache->d_indices.Resize(key.indices.size()) ||
      !cache->d_values.Resize(ctx.batch_size * cache->union_nnz * ValueBytes(ctx.precision_code))) {
    return ffi::Error(ffi::ErrorCode::kResourceExhausted, "Failed to allocate cuDSS cache buffers");
  }

  if (cudaMemcpy(cache->d_indptr.ptr, key.indptr.data(), sizeof(int32_t) * key.indptr.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    return CudaError("cudaMemcpy(union indptr)", cudaGetLastError());
  }
  if (cudaMemcpy(cache->d_indices.ptr, key.indices.data(), sizeof(int32_t) * key.indices.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    return CudaError("cudaMemcpy(union indices)", cudaGetLastError());
  }

  cudssStatus_t data_status = cudssDataCreate(ctx.handle, &cache->solver_data);
  if (data_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssDataCreate", data_status);
  }

  cudssStatus_t matrix_status = cudssMatrixCreateCsr(
      &cache->matrix,
      ctx.n_vars,
      ctx.n_vars,
      cache->union_nnz,
      cache->d_indptr.ptr,
      nullptr,
      cache->d_indices.ptr,
      cache->d_values.ptr,
      CUDA_R_32I,
      ValueType(ctx.precision_code),
      CUDSS_MTYPE_GENERAL,
      CUDSS_MVIEW_FULL,
      CUDSS_BASE_ZERO);
  if (matrix_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixCreateCsr", matrix_status);
  }

  out_cache = cache.get();
  ctx.cache.emplace(cache->key, std::move(cache));
  return ffi::Error();
}

// Scatter -J values into the union sparsity layout WITHOUT the diagonal 1/(dt*gamma) term.
template <typename Scalar>
void ScatterUnionValuesBase(
    Scalar* out_values,
    const PatternCache& cache,
    const int32_t* indptr,
    const int32_t* indices,
    const Scalar* values,
    const int32_t* nnz,
    int64_t max_nnz) {
  std::fill(out_values, out_values + (cache.key.batch_size * cache.union_nnz), Scalar{0});
  for (int64_t b = 0; b < cache.key.batch_size; ++b) {
    Scalar* out = out_values + b * cache.union_nnz;
    const int32_t* indptr_b = indptr + b * (cache.key.n_vars + 1);
    const int32_t* indices_b = indices + b * max_nnz;
    const Scalar* values_b = values + b * max_nnz;
    int32_t nnz_b = nnz[b];

    for (int64_t row = 0; row < cache.key.n_vars; ++row) {
      int32_t start = std::min(indptr_b[row], nnz_b);
      int32_t end = std::min(indptr_b[row + 1], nnz_b);
      int32_t union_start = cache.key.indptr[row];
      int32_t union_end = cache.key.indptr[row + 1];
      for (int32_t k = start; k < end; ++k) {
        int32_t col = indices_b[k];
        auto it = std::lower_bound(
            cache.key.indices.begin() + union_start,
            cache.key.indices.begin() + union_end,
            col);
        if (it != cache.key.indices.begin() + union_end && *it == col) {
          out[it - cache.key.indices.begin()] = -values_b[k];
        }
      }
    }
  }
}

// Add the 1/(dt*gamma) diagonal term to pre-scattered union values.
template <typename Scalar>
void AddDiagonal(
    Scalar* values,
    const PatternCache& cache,
    const double* dt,
    double gamma) {
  for (int64_t b = 0; b < cache.key.batch_size; ++b) {
    Scalar* out = values + b * cache.union_nnz;
    for (int64_t row = 0; row < cache.key.n_vars; ++row) {
      int32_t diag = cache.diag_offsets[row];
      if (diag >= 0) {
        out[diag] += static_cast<Scalar>(1.0 / (dt[b] * gamma));
      }
    }
  }
}

ffi::Error PrepareCommon(
    cudaStream_t stream,
    uint64_t context_handle,
    bool allow_cache,
    ffi::AnyBuffer dt_buf,
    ffi::AnyBuffer indptr_buf,
    ffi::AnyBuffer indices_buf,
    ffi::AnyBuffer values_buf,
    ffi::AnyBuffer nnz_buf,
    ffi::Result<ffi::AnyBuffer> token_buf) {
  Context* ctx = LookupContext(context_handle);
  if (ctx == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unknown cuDSS context handle");
  }

  std::lock_guard<std::mutex> lock(ctx->mu);
  cudssStatus_t stream_status = cudssSetStream(ctx->handle, stream);
  if (stream_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssSetStream", stream_status);
  }

  ffi::Error dense_status = EnsureDenseMatrices(*ctx);
  if (dense_status.failure()) {
    return dense_status;
  }

  auto dt_dims = dt_buf.dimensions();
  int64_t batch = dt_dims[0];

  // ---- FAST PATH: reuse cached base values (ideas 1-3) ----
  // When the caller is static and we already have cached base values for the
  // active pattern, skip ALL D->H copies of indptr/indices/values/nnz (idea 2),
  // skip BuildUnionKey (idea 3), and only transfer dt to compute the diagonal
  // update (idea 1).
  if (allow_cache && ctx->active != nullptr && ctx->active->has_cached_base) {
    PatternCache* cache = ctx->active;
    size_t scalar_count = static_cast<size_t>(batch * cache->union_nnz);

#ifdef HAVE_CUDA_DIAGONAL_KERNEL
    if (cache->d_base_values.ptr != nullptr && cache->d_diag_offsets.ptr != nullptr) {
      if (ctx->precision_code == kPrecisionFp32) {
        launch_copy_base_add_diagonal_f32(
            reinterpret_cast<float*>(cache->d_values.ptr),
            reinterpret_cast<const float*>(cache->d_base_values.ptr),
            cache->d_diag_offsets.ptr,
            static_cast<const double*>(dt_buf.untyped_data()),
            ctx->n_vars,
            cache->union_nnz,
            batch,
            0.19,
            stream);
      } else {
        launch_copy_base_add_diagonal_f64(
            reinterpret_cast<double*>(cache->d_values.ptr),
            reinterpret_cast<const double*>(cache->d_base_values.ptr),
            cache->d_diag_offsets.ptr,
            static_cast<const double*>(dt_buf.untyped_data()),
            ctx->n_vars,
            cache->union_nnz,
            batch,
            0.19,
            stream);
      }
      cudaError_t launch_status = cudaGetLastError();
      if (launch_status != cudaSuccess) {
        return CudaError("launch_copy_base_add_diagonal", launch_status);
      }
    } else
#endif
    {
    // Only copy dt from device — the single small D->H transfer needed.
    std::vector<double> host_dt(batch);
    cudaError_t dt_err = cudaMemcpyAsync(
        host_dt.data(), dt_buf.untyped_data(), sizeof(double) * batch,
        cudaMemcpyDeviceToHost, stream);
    if (dt_err != cudaSuccess) {
      return CudaError("cudaMemcpyAsync(dt, fast path)", dt_err);
    }
    cudaStreamSynchronize(stream);

    // Rebuild values from cached base + new diagonal, then upload to device.
    if (ctx->precision_code == kPrecisionFp32) {
      std::vector<float> host_work(cache->host_base_f32);
      AddDiagonal(host_work.data(), *cache, host_dt.data(), 0.19);
      cudaError_t val_err = cudaMemcpy(
          cache->d_values.ptr, host_work.data(),
          sizeof(float) * scalar_count, cudaMemcpyHostToDevice);
      if (val_err != cudaSuccess) {
        return CudaError("cudaMemcpy(values_f32, fast path)", val_err);
      }
    } else {
      std::vector<double> host_work(cache->host_base_f64);
      AddDiagonal(host_work.data(), *cache, host_dt.data(), 0.19);
      cudaError_t val_err = cudaMemcpy(
          cache->d_values.ptr, host_work.data(),
          sizeof(double) * scalar_count, cudaMemcpyHostToDevice);
      if (val_err != cudaSuccess) {
        return CudaError("cudaMemcpy(values_f64, fast path)", val_err);
      }
    }
    }

    // Refactorize with the updated values (analysis already done).
    cudssStatus_t factor_status = cudssExecute(
        ctx->handle,
        CUDSS_PHASE_REFACTORIZATION,
        ctx->config,
        cache->solver_data,
        cache->matrix,
        ctx->dense.solution,
        ctx->dense.rhs);
    if (factor_status != CUDSS_STATUS_SUCCESS) {
      return CuDssError("cudssExecute(REFACTORIZATION, fast path)", factor_status);
    }
    ctx->factorization_count += 1;

    if (token_buf->untyped_data() != nullptr) {
      cudaMemsetAsync(token_buf->untyped_data(), 0, 1, stream);
    }
    return ffi::Error();
  }

  // ---- FULL PATH: first call or dynamic values ----
  auto indptr_dims = indptr_buf.dimensions();
  auto indices_dims = indices_buf.dimensions();
  if (dt_dims.size() != 1 || indptr_dims.size() != 2 || indices_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unexpected cuDSS prepare input ranks");
  }

  int64_t n_vars = indptr_dims[1] - 1;
  int64_t max_nnz = indices_dims[1];
  if (batch != ctx->batch_size || n_vars != ctx->n_vars) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "cuDSS prepare buffers do not match solver context");
  }

  std::vector<double> host_dt(batch);
  std::vector<int32_t> host_indptr(batch * (n_vars + 1));
  std::vector<int32_t> host_indices(batch * max_nnz);
  std::vector<int32_t> host_nnz(batch);

  if (cudaMemcpy(host_dt.data(), dt_buf.untyped_data(), sizeof(double) * batch, cudaMemcpyDeviceToHost) != cudaSuccess) {
    return CudaError("cudaMemcpy(dt)", cudaGetLastError());
  }
  if (cudaMemcpy(host_indptr.data(), indptr_buf.untyped_data(), sizeof(int32_t) * host_indptr.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return CudaError("cudaMemcpy(indptr)", cudaGetLastError());
  }
  if (cudaMemcpy(host_indices.data(), indices_buf.untyped_data(), sizeof(int32_t) * host_indices.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return CudaError("cudaMemcpy(indices)", cudaGetLastError());
  }
  if (nnz_buf.element_type() == ffi::DataType::S32) {
    if (cudaMemcpy(host_nnz.data(), nnz_buf.untyped_data(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(nnz)", cudaGetLastError());
    }
  } else if (nnz_buf.element_type() == ffi::DataType::S64) {
    std::vector<int64_t> host_nnz64(batch);
    if (cudaMemcpy(host_nnz64.data(), nnz_buf.untyped_data(), sizeof(int64_t) * batch, cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(nnz64)", cudaGetLastError());
    }
    for (int64_t i = 0; i < batch; ++i) host_nnz[i] = static_cast<int32_t>(host_nnz64[i]);
  } else {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "nnz buffer must be int32 or int64");
  }

  PatternKey key = BuildUnionKey(n_vars, batch, host_indptr.data(), host_indices.data(), host_nnz.data(), max_nnz);
  PatternCache* cache = nullptr;
  ffi::Error cache_status = EnsurePatternCache(*ctx, key, cache);
  if (cache_status.failure()) {
    return cache_status;
  }

  size_t scalar_count = static_cast<size_t>(batch * cache->union_nnz);
  if (ctx->precision_code == kPrecisionFp32) {
    std::vector<float> host_values(batch * max_nnz);
    if (cudaMemcpy(host_values.data(), values_buf.untyped_data(), sizeof(float) * host_values.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(values_f32)", cudaGetLastError());
    }
    std::vector<float> host_union_values(scalar_count);
    ScatterUnionValuesBase(
        host_union_values.data(),
        *cache,
        host_indptr.data(),
        host_indices.data(),
        host_values.data(),
        host_nnz.data(),
        max_nnz);
    // Cache base values before adding diagonal (for static fast path).
    if (allow_cache) {
      cache->host_base_f32 = host_union_values;
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
      if (cache->d_base_values.Resize(scalar_count * sizeof(float)) &&
          cache->d_diag_offsets.Resize(cache->diag_offsets.size())) {
        if (cudaMemcpy(cache->d_base_values.ptr, host_union_values.data(), sizeof(float) * scalar_count, cudaMemcpyHostToDevice) != cudaSuccess) {
          return CudaError("cudaMemcpy(base_values_f32)", cudaGetLastError());
        }
        if (cudaMemcpy(cache->d_diag_offsets.ptr, cache->diag_offsets.data(), sizeof(int32_t) * cache->diag_offsets.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
          return CudaError("cudaMemcpy(diag_offsets)", cudaGetLastError());
        }
      }
#endif
      cache->has_cached_base = true;
    }
    AddDiagonal(host_union_values.data(), *cache, host_dt.data(), 0.19);
    if (cudaMemcpy(cache->d_values.ptr, host_union_values.data(), sizeof(float) * scalar_count, cudaMemcpyHostToDevice) != cudaSuccess) {
      return CudaError("cudaMemcpy(union_values_f32)", cudaGetLastError());
    }
  } else {
    std::vector<double> host_values(batch * max_nnz);
    if (cudaMemcpy(host_values.data(), values_buf.untyped_data(), sizeof(double) * host_values.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(values_f64)", cudaGetLastError());
    }
    std::vector<double> host_union_values(scalar_count);
    ScatterUnionValuesBase(
        host_union_values.data(),
        *cache,
        host_indptr.data(),
        host_indices.data(),
        host_values.data(),
        host_nnz.data(),
        max_nnz);
    if (allow_cache) {
      cache->host_base_f64 = host_union_values;
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
      if (cache->d_base_values.Resize(scalar_count * sizeof(double)) &&
          cache->d_diag_offsets.Resize(cache->diag_offsets.size())) {
        if (cudaMemcpy(cache->d_base_values.ptr, host_union_values.data(), sizeof(double) * scalar_count, cudaMemcpyHostToDevice) != cudaSuccess) {
          return CudaError("cudaMemcpy(base_values_f64)", cudaGetLastError());
        }
        if (cudaMemcpy(cache->d_diag_offsets.ptr, cache->diag_offsets.data(), sizeof(int32_t) * cache->diag_offsets.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
          return CudaError("cudaMemcpy(diag_offsets)", cudaGetLastError());
        }
      }
#endif
      cache->has_cached_base = true;
    }
    AddDiagonal(host_union_values.data(), *cache, host_dt.data(), 0.19);
    if (cudaMemcpy(cache->d_values.ptr, host_union_values.data(), sizeof(double) * scalar_count, cudaMemcpyHostToDevice) != cudaSuccess) {
      return CudaError("cudaMemcpy(union_values_f64)", cudaGetLastError());
    }
  }

  if (!cache->analyzed) {
    cudssStatus_t analysis_status = cudssExecute(
        ctx->handle,
        CUDSS_PHASE_ANALYSIS,
        ctx->config,
        cache->solver_data,
        cache->matrix,
        ctx->dense.solution,
        ctx->dense.rhs);
    if (analysis_status != CUDSS_STATUS_SUCCESS) {
      return CuDssError("cudssExecute(ANALYSIS)", analysis_status);
    }
    cache->analyzed = true;
    ctx->analysis_count += 1;
  }

  cudssStatus_t factor_status = cudssExecute(
      ctx->handle,
      cache->factorized ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION,
      ctx->config,
      cache->solver_data,
      cache->matrix,
      ctx->dense.solution,
      ctx->dense.rhs);
  if (factor_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssExecute(FACTORIZATION)", factor_status);
  }
  cache->factorized = true;
  ctx->factorization_count += 1;
  ctx->active = cache;
  if (token_buf->untyped_data() != nullptr) {
    cudaError_t token_status = cudaMemsetAsync(token_buf->untyped_data(), 0, 1, stream);
    if (token_status != cudaSuccess) {
      return CudaError("cudaMemsetAsync(prepare_token)", token_status);
    }
  }
  return ffi::Error();
}

// Static entry point: uses cached base values after the first call.
ffi::Error PrepareStaticImpl(
    cudaStream_t stream,
    uint64_t context_handle,
    ffi::AnyBuffer dt_buf,
    ffi::AnyBuffer indptr_buf,
    ffi::AnyBuffer indices_buf,
    ffi::AnyBuffer values_buf,
    ffi::AnyBuffer nnz_buf,
    ffi::Result<ffi::AnyBuffer> token_buf) {
  return PrepareCommon(stream, context_handle, /*allow_cache=*/true,
                       dt_buf, indptr_buf, indices_buf, values_buf, nnz_buf, token_buf);
}

// Dynamic entry point: always recomputes from input arrays (values may change).
ffi::Error PrepareDynamicImpl(
    cudaStream_t stream,
    uint64_t context_handle,
    ffi::AnyBuffer dt_buf,
    ffi::AnyBuffer indptr_buf,
    ffi::AnyBuffer indices_buf,
    ffi::AnyBuffer values_buf,
    ffi::AnyBuffer nnz_buf,
    ffi::Result<ffi::AnyBuffer> token_buf) {
  return PrepareCommon(stream, context_handle, /*allow_cache=*/false,
                       dt_buf, indptr_buf, indices_buf, values_buf, nnz_buf, token_buf);
}

ffi::Error SolveImpl(
    cudaStream_t stream,
    uint64_t context_handle,
    ffi::AnyBuffer,
    ffi::AnyBuffer rhs_buf,
    ffi::Result<ffi::AnyBuffer> out_buf) {
  Context* ctx = LookupContext(context_handle);
  if (ctx == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unknown cuDSS context handle");
  }

  std::lock_guard<std::mutex> lock(ctx->mu);
  PatternCache* cache = ctx->active;
  if (cache == nullptr || !cache->factorized) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "No active cuDSS factorization");
  }

  cudssStatus_t stream_status = cudssSetStream(ctx->handle, stream);
  if (stream_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssSetStream", stream_status);
  }

  auto rhs_dims = rhs_buf.dimensions();
  auto out_dims = out_buf->dimensions();
  if (rhs_dims.size() != 2 || out_dims.size() != 2 || rhs_dims[0] != ctx->batch_size || rhs_dims[1] != ctx->n_vars) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unexpected cuDSS solve buffer shapes");
  }

  cudssStatus_t rhs_status = cudssMatrixSetValues(ctx->dense.rhs, rhs_buf.untyped_data());
  if (rhs_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixSetValues(rhs)", rhs_status);
  }
  cudssStatus_t sol_status = cudssMatrixSetValues(ctx->dense.solution, out_buf->untyped_data());
  if (sol_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixSetValues(solution)", sol_status);
  }

  cudssStatus_t solve_status = cudssExecute(
      ctx->handle,
      CUDSS_PHASE_SOLVE,
      ctx->config,
      cache->solver_data,
      cache->matrix,
      ctx->dense.solution,
      ctx->dense.rhs);
  if (solve_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssExecute(SOLVE)", solve_status);
  }
  ctx->solve_count += 1;
  return ffi::Error();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rodas5_cudss_prepare,
    PrepareStaticImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<uint64_t>("context_handle")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rodas5_cudss_prepare_dynamic,
    PrepareDynamicImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<uint64_t>("context_handle")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rodas5_cudss_solve,
    SolveImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<uint64_t>("context_handle")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

extern "C" uint64_t rodas5_cudss_create_context(int64_t n_vars, int64_t batch_size, int precision_code) {
  auto ctx = std::make_unique<Context>();
  ctx->n_vars = n_vars;
  ctx->batch_size = batch_size;
  ctx->precision_code = precision_code;

  cudssStatus_t create_status = cudssCreate(&ctx->handle);
  if (create_status != CUDSS_STATUS_SUCCESS) {
    return 0;
  }
  cudssStatus_t config_status = cudssConfigCreate(&ctx->config);
  if (config_status != CUDSS_STATUS_SUCCESS) {
    return 0;
  }
  int ubatch_size = static_cast<int>(batch_size);
  int ubatch_index = -1;
  cudssAlgType_t reorder_alg = CUDSS_ALG_3;
  if (cudssConfigSet(ctx->config, CUDSS_CONFIG_UBATCH_SIZE, &ubatch_size, sizeof(ubatch_size)) != CUDSS_STATUS_SUCCESS) {
    return 0;
  }
  if (cudssConfigSet(ctx->config, CUDSS_CONFIG_UBATCH_INDEX, &ubatch_index, sizeof(ubatch_index)) != CUDSS_STATUS_SUCCESS) {
    return 0;
  }
  if (cudssConfigSet(ctx->config, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(reorder_alg)) != CUDSS_STATUS_SUCCESS) {
    return 0;
  }

  uint64_t handle = g_next_context_id.fetch_add(1);
  std::lock_guard<std::mutex> lock(g_context_mu);
  g_contexts.emplace(handle, std::move(ctx));
  return handle;
}

extern "C" void rodas5_cudss_reset_active(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    std::lock_guard<std::mutex> lock(ctx->mu);
    ctx->active = nullptr;
  }
}

extern "C" void rodas5_cudss_destroy_context(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_context_mu);
  g_contexts.erase(handle);
}

extern "C" uint64_t rodas5_cudss_analysis_count(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->analysis_count;
  }
  return 0;
}

extern "C" uint64_t rodas5_cudss_factorization_count(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->factorization_count;
  }
  return 0;
}

extern "C" uint64_t rodas5_cudss_solve_count(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->solve_count;
  }
  return 0;
}
