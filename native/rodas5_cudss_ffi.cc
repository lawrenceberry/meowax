#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include "xla/ffi/api/ffi.h"

// Optional native fused-step helpers. The current Pallas-based sparse path does
// not build these launchers, so this block remains disabled unless a separate
// native fused-step implementation is linked in again.
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
extern "C" void launch_copy_negate_f32(
    float* out, const float* src, int64_t total, void* stream);
extern "C" void launch_copy_negate_f64(
    double* out, const double* src, int64_t total, void* stream);
extern "C" void launch_copy_base_add_diagonal_f32(
    float* out, const float* base, const int32_t* diag_offsets,
    const double* dt, int64_t n_vars, int64_t union_nnz,
    int64_t batch_size, double gamma, void* stream);
extern "C" void launch_copy_base_add_diagonal_f64(
    double* out, const double* base, const int32_t* diag_offsets,
    const double* dt, int64_t n_vars, int64_t union_nnz,
    int64_t batch_size, double gamma, void* stream);
extern "C" void launch_copy_negate_add_diagonal_f32(
    float* out, const float* src, const int32_t* diag_offsets,
    const double* dt, int64_t n_vars, int64_t union_nnz,
    int64_t batch_size, double gamma, void* stream);
extern "C" void launch_copy_negate_add_diagonal_f64(
    double* out, const double* src, const int32_t* diag_offsets,
    const double* dt, int64_t n_vars, int64_t union_nnz,
    int64_t batch_size, double gamma, void* stream);
extern "C" void launch_batched_csr_matvec_f32(
    float* out, const int32_t* indptr, const int32_t* indices,
    const float* values, const float* vec,
    int64_t n_vars, int64_t max_nnz, int64_t batch_size, void* stream);
extern "C" void launch_batched_csr_matvec_f64(
    double* out, const int32_t* indptr, const int32_t* indices,
    const double* values, const double* vec,
    int64_t n_vars, int64_t max_nnz, int64_t batch_size, void* stream);
extern "C" void launch_stage_state_f32(
    float* out, const float* y, const float* k1, double a1,
    const float* k2, double a2, const float* k3, double a3,
    const float* k4, double a4, const float* k5, double a5,
    const float* k6, double a6, const float* k7, double a7,
    int64_t total, void* stream);
extern "C" void launch_stage_state_f64(
    double* out, const double* y, const double* k1, double a1,
    const double* k2, double a2, const double* k3, double a3,
    const double* k4, double a4, const double* k5, double a5,
    const double* k6, double a6, const double* k7, double a7,
    int64_t total, void* stream);
extern "C" void launch_stage_rhs_f32(
    float* out, const float* du, const double* dt, const float* k1, double c1,
    const float* k2, double c2, const float* k3, double c3,
    const float* k4, double c4, const float* k5, double c5,
    const float* k6, double c6, const float* k7, double c7,
    int64_t n_vars, int64_t batch_size, void* stream);
extern "C" void launch_stage_rhs_f64(
    double* out, const double* du, const double* dt, const double* k1, double c1,
    const double* k2, double c2, const double* k3, double c3,
    const double* k4, double c4, const double* k5, double c5,
    const double* k6, double c6, const double* k7, double c7,
    int64_t n_vars, int64_t batch_size, void* stream);
extern "C" void launch_add_vectors_f32(
    float* out, const float* lhs, const float* rhs, int64_t total, void* stream);
extern "C" void launch_add_vectors_f64(
    double* out, const double* lhs, const double* rhs, int64_t total, void* stream);
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
  DeviceBuffer<std::byte> graph_values_in;
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
  cudaGraph_t step_graph = nullptr;
  cudaGraphExec_t step_graph_exec = nullptr;
  bool graph_ready = false;
  bool graph_launched = false;

  ~PatternCache() {
    if (step_graph_exec != nullptr) {
      cudaGraphExecDestroy(step_graph_exec);
    }
    if (step_graph != nullptr) {
      cudaGraphDestroy(step_graph);
    }
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

struct StepScratchBuffers {
  DeviceBuffer<std::byte> rhs;
  DeviceBuffer<std::byte> state;
  DeviceBuffer<std::byte> du;
  DeviceBuffer<std::byte> k1;
  DeviceBuffer<std::byte> k2;
  DeviceBuffer<std::byte> k3;
  DeviceBuffer<std::byte> k4;
  DeviceBuffer<std::byte> k5;
  DeviceBuffer<std::byte> k6;
  DeviceBuffer<std::byte> k7;
};

struct Context {
  int64_t n_vars;
  int64_t batch_size;
  int precision_code;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  DenseBatchMatrices dense;
  StepScratchBuffers scratch;
  DeviceBuffer<double> graph_dt;
  DeviceBuffer<std::byte> graph_y;
  DeviceBuffer<std::byte> graph_out;
  std::unordered_map<PatternKey, std::unique_ptr<PatternCache>, PatternKeyHash> cache;
  PatternCache* active = nullptr;
  std::mutex mu;
  uint64_t analysis_count = 0;
  uint64_t factorization_count = 0;
  uint64_t solve_count = 0;
  uint64_t host_prepare_slow_path_count = 0;
  uint64_t device_prepare_fast_path_count = 0;
  uint64_t graph_capture_count = 0;
  uint64_t graph_replay_count = 0;
  bool mt_analysis_enabled = false;
  bool graph_enabled = false;

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

bool EnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return !(text.empty() || text == "0" || text == "false" || text == "no" || text == "off");
}

int EnvIntOr(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

std::string ResolveThreadingLibrary() {
  const char* explicit_path = std::getenv("RODAS5_CUDSS_THREADING_LIB");
  if (explicit_path != nullptr && *explicit_path != '\0') {
    return std::string(explicit_path);
  }
  explicit_path = std::getenv("CUDSS_THREADING_LIB");
  if (explicit_path != nullptr && *explicit_path != '\0') {
    return std::string(explicit_path);
  }

  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&cudssCreate), &info) == 0 || info.dli_fname == nullptr) {
    return {};
  }
  std::filesystem::path cudss_path(info.dli_fname);
  std::filesystem::path candidate = cudss_path.parent_path() / "libcudss_mtlayer_gomp.so";
  return std::filesystem::exists(candidate) ? candidate.string() : std::string();
}

int AsyncDeviceAlloc(void*, void** ptr, size_t size, cudaStream_t stream) {
  return cudaMallocAsync(ptr, size, stream) == cudaSuccess ? 0 : 1;
}

int AsyncDeviceFree(void*, void* ptr, size_t, cudaStream_t stream) {
  return cudaFreeAsync(ptr, stream) == cudaSuccess ? 0 : 1;
}

constexpr double kGamma = 0.19;
constexpr double kA21 = 2.0;
constexpr double kA31 = 3.040894194418781;
constexpr double kA32 = 1.041747909077569;
constexpr double kA41 = 2.576417536461461;
constexpr double kA42 = 1.622083060776640;
constexpr double kA43 = -0.9089668560264532;
constexpr double kA51 = 2.760842080225597;
constexpr double kA52 = 1.446624659844071;
constexpr double kA53 = -0.3036980084553738;
constexpr double kA54 = 0.2877498600325443;
constexpr double kA61 = -14.09640773051259;
constexpr double kA62 = 6.925207756232704;
constexpr double kA63 = -41.47510893210728;
constexpr double kA64 = 2.343771018586405;
constexpr double kA65 = 24.13215229196062;
constexpr double kC21 = -10.31323885133993;
constexpr double kC31 = -21.04823117650003;
constexpr double kC32 = -7.234992135176716;
constexpr double kC41 = 32.22751541853323;
constexpr double kC42 = -4.943732386540191;
constexpr double kC43 = 19.44922031041879;
constexpr double kC51 = -20.69865579590063;
constexpr double kC52 = -8.816374604402768;
constexpr double kC53 = 1.260436877740897;
constexpr double kC54 = -0.7495647613787146;
constexpr double kC61 = -46.22004352711257;
constexpr double kC62 = -17.49534862857472;
constexpr double kC63 = -289.6389582892057;
constexpr double kC64 = 93.60855400400906;
constexpr double kC65 = 318.3822534212147;
constexpr double kC71 = 34.20013733472935;
constexpr double kC72 = -14.15535402717690;
constexpr double kC73 = 57.82335640988400;
constexpr double kC74 = 25.83362985412365;
constexpr double kC75 = 1.408950972071624;
constexpr double kC76 = -6.551835421242162;
constexpr double kC81 = 42.57076742291101;
constexpr double kC82 = -13.80770672017997;
constexpr double kC83 = 93.98938432427124;
constexpr double kC84 = 18.77919633714503;
constexpr double kC85 = -31.58359187223370;
constexpr double kC86 = -6.685968952921985;
constexpr double kC87 = -5.810979938412932;

template <typename Scalar>
Scalar* BufferPtr(DeviceBuffer<std::byte>& buffer) {
  return reinterpret_cast<Scalar*>(buffer.ptr);
}

template <typename Scalar>
const Scalar* BufferPtr(const DeviceBuffer<std::byte>& buffer) {
  return reinterpret_cast<const Scalar*>(buffer.ptr);
}

ffi::Error EnsureStepScratch(Context& ctx) {
  const size_t elem_count = static_cast<size_t>(ctx.batch_size * ctx.n_vars * ValueBytes(ctx.precision_code));
  if (!ctx.scratch.rhs.Resize(elem_count) ||
      !ctx.scratch.state.Resize(elem_count) ||
      !ctx.scratch.du.Resize(elem_count) ||
      !ctx.scratch.k1.Resize(elem_count) ||
      !ctx.scratch.k2.Resize(elem_count) ||
      !ctx.scratch.k3.Resize(elem_count) ||
      !ctx.scratch.k4.Resize(elem_count) ||
      !ctx.scratch.k5.Resize(elem_count) ||
      !ctx.scratch.k6.Resize(elem_count) ||
      !ctx.scratch.k7.Resize(elem_count)) {
    return ffi::Error(ffi::ErrorCode::kResourceExhausted, "Failed to allocate cuDSS step scratch buffers");
  }
  return ffi::Error();
}

ffi::Error EnsureGraphBuffers(Context& ctx) {
  const size_t elem_count = static_cast<size_t>(ctx.batch_size * ctx.n_vars * ValueBytes(ctx.precision_code));
  if (!ctx.graph_dt.Resize(static_cast<size_t>(ctx.batch_size)) ||
      !ctx.graph_y.Resize(elem_count) ||
      !ctx.graph_out.Resize(elem_count * 2)) {
    return ffi::Error(ffi::ErrorCode::kResourceExhausted, "Failed to allocate cuDSS graph buffers");
  }
  return ffi::Error();
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

PatternKey BuildSharedKey(
    int64_t n_vars,
    int64_t batch_size,
    const int32_t* indptr,
    const int32_t* indices,
    int32_t nnz) {
  PatternKey key;
  key.n_vars = n_vars;
  key.batch_size = batch_size;
  key.indptr.assign(indptr, indptr + (n_vars + 1));
  key.indices.assign(indices, indices + nnz);
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

ffi::Error CheckLaunch(const char* where);

ffi::Error EnsureActiveCache(
    Context& ctx,
    ffi::AnyBuffer indptr_buf,
    ffi::AnyBuffer indices_buf,
    ffi::AnyBuffer nnz_buf,
    PatternCache*& cache) {
  if (ctx.active != nullptr) {
    cache = ctx.active;
    return ffi::Error();
  }

  auto indptr_dims = indptr_buf.dimensions();
  auto indices_dims = indices_buf.dimensions();
  if (indptr_dims.size() != 2 || indices_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unexpected cuDSS sparse structure ranks");
  }

  int64_t n_vars = indptr_dims[1] - 1;
  int64_t max_nnz = indices_dims[1];
  std::vector<int32_t> host_indptr(n_vars + 1);
  std::vector<int32_t> host_indices(max_nnz);
  int32_t shared_nnz = 0;
  if (cudaMemcpy(host_indptr.data(), indptr_buf.untyped_data(), sizeof(int32_t) * host_indptr.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return CudaError("cudaMemcpy(shared indptr)", cudaGetLastError());
  }
  if (cudaMemcpy(host_indices.data(), indices_buf.untyped_data(), sizeof(int32_t) * host_indices.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return CudaError("cudaMemcpy(shared indices)", cudaGetLastError());
  }
  if (nnz_buf.element_type() == ffi::DataType::S32) {
    if (cudaMemcpy(&shared_nnz, nnz_buf.untyped_data(), sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(shared nnz)", cudaGetLastError());
    }
  } else if (nnz_buf.element_type() == ffi::DataType::S64) {
    int64_t shared_nnz64 = 0;
    if (cudaMemcpy(&shared_nnz64, nnz_buf.untyped_data(), sizeof(int64_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(shared nnz64)", cudaGetLastError());
    }
    shared_nnz = static_cast<int32_t>(shared_nnz64);
  } else {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "nnz buffer must be int32 or int64");
  }

  PatternKey key = BuildSharedKey(n_vars, ctx.batch_size, host_indptr.data(), host_indices.data(), shared_nnz);
  ffi::Error cache_status = EnsurePatternCache(ctx, key, cache);
  if (cache_status.failure()) {
    return cache_status;
  }
  ctx.active = cache;
  return ffi::Error();
}

template <typename Scalar>
ffi::Error PrepareValues(
    Context& ctx,
    PatternCache& cache,
    bool allow_cache,
    const double* dt_ptr,
    const Scalar* values_ptr,
    cudaStream_t stream,
    bool count_prepare) {
  size_t scalar_count = static_cast<size_t>(ctx.batch_size * cache.union_nnz);
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
  if (!cache.d_diag_offsets.Resize(cache.diag_offsets.size())) {
    return ffi::Error(ffi::ErrorCode::kResourceExhausted, "Failed to allocate cuDSS diagonal offsets");
  }
  if (cudaMemcpy(cache.d_diag_offsets.ptr, cache.diag_offsets.data(), sizeof(int32_t) * cache.diag_offsets.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    return CudaError("cudaMemcpy(diag_offsets)", cudaGetLastError());
  }
  if (allow_cache && !cache.has_cached_base) {
    if (!cache.d_base_values.Resize(scalar_count * ValueBytes(ctx.precision_code))) {
      return ffi::Error(ffi::ErrorCode::kResourceExhausted, "Failed to allocate cuDSS base values");
    }
    if constexpr (std::is_same_v<Scalar, float>) {
      launch_copy_negate_f32(
          reinterpret_cast<float*>(cache.d_base_values.ptr),
          values_ptr,
          scalar_count,
          stream);
    } else {
      launch_copy_negate_f64(
          reinterpret_cast<double*>(cache.d_base_values.ptr),
          values_ptr,
          scalar_count,
          stream);
    }
    if (ffi::Error status = CheckLaunch("launch_copy_negate(base)"); status.failure()) {
      return status;
    }
    cache.has_cached_base = true;
  }

  if (allow_cache && cache.has_cached_base) {
    if constexpr (std::is_same_v<Scalar, float>) {
      launch_copy_base_add_diagonal_f32(
          reinterpret_cast<float*>(cache.d_values.ptr),
          reinterpret_cast<const float*>(cache.d_base_values.ptr),
          cache.d_diag_offsets.ptr,
          dt_ptr,
          ctx.n_vars,
          cache.union_nnz,
          ctx.batch_size,
          kGamma,
          stream);
    } else {
      launch_copy_base_add_diagonal_f64(
          reinterpret_cast<double*>(cache.d_values.ptr),
          reinterpret_cast<const double*>(cache.d_base_values.ptr),
          cache.d_diag_offsets.ptr,
          dt_ptr,
          ctx.n_vars,
          cache.union_nnz,
          ctx.batch_size,
          kGamma,
          stream);
    }
  } else if constexpr (std::is_same_v<Scalar, float>) {
    launch_copy_negate_add_diagonal_f32(
        reinterpret_cast<float*>(cache.d_values.ptr),
        values_ptr,
        cache.d_diag_offsets.ptr,
        dt_ptr,
        ctx.n_vars,
        cache.union_nnz,
        ctx.batch_size,
        kGamma,
        stream);
  } else {
    launch_copy_negate_add_diagonal_f64(
        reinterpret_cast<double*>(cache.d_values.ptr),
        values_ptr,
        cache.d_diag_offsets.ptr,
        dt_ptr,
        ctx.n_vars,
        cache.union_nnz,
        ctx.batch_size,
        kGamma,
        stream);
  }
  if (ffi::Error status = CheckLaunch("launch_prepare_values"); status.failure()) {
    return status;
  }
  if (count_prepare) {
    ctx.device_prepare_fast_path_count += 1;
  }
#else
  if (count_prepare) {
    ctx.host_prepare_slow_path_count += 1;
  }
  std::vector<double> host_dt(ctx.batch_size);
  if (cudaMemcpy(host_dt.data(), dt_ptr, sizeof(double) * ctx.batch_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
    return CudaError("cudaMemcpy(dt)", cudaGetLastError());
  }
  if constexpr (std::is_same_v<Scalar, float>) {
    std::vector<float> host_values(scalar_count);
    if (cudaMemcpy(host_values.data(), values_ptr, sizeof(float) * scalar_count, cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(values_f32)", cudaGetLastError());
    }
    for (float& value : host_values) value = -value;
    AddDiagonal(host_values.data(), cache, host_dt.data(), kGamma);
    if (cudaMemcpy(cache.d_values.ptr, host_values.data(), sizeof(float) * scalar_count, cudaMemcpyHostToDevice) != cudaSuccess) {
      return CudaError("cudaMemcpy(prepared_values_f32)", cudaGetLastError());
    }
  } else {
    std::vector<double> host_values(scalar_count);
    if (cudaMemcpy(host_values.data(), values_ptr, sizeof(double) * scalar_count, cudaMemcpyDeviceToHost) != cudaSuccess) {
      return CudaError("cudaMemcpy(values_f64)", cudaGetLastError());
    }
    for (double& value : host_values) value = -value;
    AddDiagonal(host_values.data(), cache, host_dt.data(), kGamma);
    if (cudaMemcpy(cache.d_values.ptr, host_values.data(), sizeof(double) * scalar_count, cudaMemcpyHostToDevice) != cudaSuccess) {
      return CudaError("cudaMemcpy(prepared_values_f64)", cudaGetLastError());
    }
  }
#endif
  return ffi::Error();
}

ffi::Error EnsureAnalyzed(Context& ctx, PatternCache& cache) {
  if (cache.analyzed) {
    return ffi::Error();
  }
  cudssStatus_t analysis_status = cudssExecute(
      ctx.handle,
      CUDSS_PHASE_ANALYSIS,
      ctx.config,
      cache.solver_data,
      cache.matrix,
      ctx.dense.solution,
      ctx.dense.rhs);
  if (analysis_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssExecute(ANALYSIS)", analysis_status);
  }
  cache.analyzed = true;
  ctx.analysis_count += 1;
  return ffi::Error();
}

ffi::Error FactorizeActive(Context& ctx, PatternCache& cache, bool count_factorization) {
  cudssStatus_t factor_status = cudssExecute(
      ctx.handle,
      cache.factorized ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION,
      ctx.config,
      cache.solver_data,
      cache.matrix,
      ctx.dense.solution,
      ctx.dense.rhs);
  if (factor_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssExecute(FACTORIZATION)", factor_status);
  }
  cache.factorized = true;
  if (count_factorization) {
    ctx.factorization_count += 1;
  }
  return ffi::Error();
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
    void* token_ptr) {
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
  auto indptr_dims = indptr_buf.dimensions();
  auto indices_dims = indices_buf.dimensions();
  if (dt_dims.size() != 1 || indptr_dims.size() != 2 || indices_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unexpected cuDSS prepare input ranks");
  }

  int64_t batch = dt_dims[0];
  int64_t n_vars = indptr_dims[1] - 1;
  if (batch != ctx->batch_size || n_vars != ctx->n_vars) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "cuDSS prepare buffers do not match solver context");
  }

  PatternCache* cache = nullptr;
  if (ffi::Error status = EnsureActiveCache(*ctx, indptr_buf, indices_buf, nnz_buf, cache); status.failure()) {
    return status;
  }

  if (values_buf.element_type() == ffi::DataType::F32) {
    if (ffi::Error status = PrepareValues<float>(
            *ctx,
            *cache,
            allow_cache,
            static_cast<const double*>(dt_buf.untyped_data()),
            static_cast<const float*>(values_buf.untyped_data()),
            stream,
            /*count_prepare=*/true);
        status.failure()) {
      return status;
    }
  } else {
    if (ffi::Error status = PrepareValues<double>(
            *ctx,
            *cache,
            allow_cache,
            static_cast<const double*>(dt_buf.untyped_data()),
            static_cast<const double*>(values_buf.untyped_data()),
            stream,
            /*count_prepare=*/true);
        status.failure()) {
      return status;
    }
  }

  if (ffi::Error status = EnsureAnalyzed(*ctx, *cache); status.failure()) {
    return status;
  }
  if (ffi::Error status = FactorizeActive(*ctx, *cache, /*count_factorization=*/true); status.failure()) {
    return status;
  }
  if (token_ptr != nullptr) {
    cudaError_t token_status = cudaMemsetAsync(token_ptr, 0, 1, stream);
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
                       dt_buf, indptr_buf, indices_buf, values_buf, nnz_buf, token_buf->untyped_data());
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
                       dt_buf, indptr_buf, indices_buf, values_buf, nnz_buf, token_buf->untyped_data());
}

ffi::Error CheckLaunch(const char* where) {
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return CudaError(where, status);
  }
  return ffi::Error();
}

template <typename Scalar>
void LaunchMatvec(
    Scalar* out,
    const int32_t* indptr,
    const int32_t* indices,
    const Scalar* values,
    const Scalar* vec,
    int64_t n_vars,
    int64_t max_nnz,
    int64_t batch_size,
    cudaStream_t stream) {
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
  if constexpr (std::is_same_v<Scalar, float>) {
    launch_batched_csr_matvec_f32(out, indptr, indices, values, vec, n_vars, max_nnz, batch_size, stream);
  } else {
    launch_batched_csr_matvec_f64(out, indptr, indices, values, vec, n_vars, max_nnz, batch_size, stream);
  }
#endif
}

template <typename Scalar>
void LaunchState(
    Scalar* out,
    const Scalar* y,
    const Scalar* k1,
    double a1,
    const Scalar* k2,
    double a2,
    const Scalar* k3,
    double a3,
    const Scalar* k4,
    double a4,
    const Scalar* k5,
    double a5,
    const Scalar* k6,
    double a6,
    const Scalar* k7,
    double a7,
    int64_t total,
    cudaStream_t stream) {
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
  if constexpr (std::is_same_v<Scalar, float>) {
    launch_stage_state_f32(out, y, k1, a1, k2, a2, k3, a3, k4, a4, k5, a5, k6, a6, k7, a7, total, stream);
  } else {
    launch_stage_state_f64(out, y, k1, a1, k2, a2, k3, a3, k4, a4, k5, a5, k6, a6, k7, a7, total, stream);
  }
#endif
}

template <typename Scalar>
void LaunchRhs(
    Scalar* out,
    const Scalar* du,
    const double* dt,
    const Scalar* k1,
    double c1,
    const Scalar* k2,
    double c2,
    const Scalar* k3,
    double c3,
    const Scalar* k4,
    double c4,
    const Scalar* k5,
    double c5,
    const Scalar* k6,
    double c6,
    const Scalar* k7,
    double c7,
    int64_t n_vars,
    int64_t batch_size,
    cudaStream_t stream) {
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
  if constexpr (std::is_same_v<Scalar, float>) {
    launch_stage_rhs_f32(out, du, dt, k1, c1, k2, c2, k3, c3, k4, c4, k5, c5, k6, c6, k7, c7, n_vars, batch_size, stream);
  } else {
    launch_stage_rhs_f64(out, du, dt, k1, c1, k2, c2, k3, c3, k4, c4, k5, c5, k6, c6, k7, c7, n_vars, batch_size, stream);
  }
#endif
}

template <typename Scalar>
void LaunchAdd(
    Scalar* out,
    const Scalar* lhs,
    const Scalar* rhs,
    int64_t total,
    cudaStream_t stream) {
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
  if constexpr (std::is_same_v<Scalar, float>) {
    launch_add_vectors_f32(out, lhs, rhs, total, stream);
  } else {
    launch_add_vectors_f64(out, lhs, rhs, total, stream);
  }
#endif
}

ffi::Error SolveTo(Context& ctx, PatternCache* cache, void* solution_ptr) {
  cudssStatus_t sol_status = cudssMatrixSetValues(ctx.dense.solution, solution_ptr);
  if (sol_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixSetValues(solution)", sol_status);
  }
  cudssStatus_t solve_status = cudssExecute(
      ctx.handle,
      CUDSS_PHASE_SOLVE,
      ctx.config,
      cache->solver_data,
      cache->matrix,
      ctx.dense.solution,
      ctx.dense.rhs);
  if (solve_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssExecute(SOLVE)", solve_status);
  }
  ctx.solve_count += 1;
  return ffi::Error();
}

ffi::Error SolveToNoCount(Context& ctx, PatternCache* cache, void* solution_ptr) {
  cudssStatus_t sol_status = cudssMatrixSetValues(ctx.dense.solution, solution_ptr);
  if (sol_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixSetValues(solution)", sol_status);
  }
  cudssStatus_t solve_status = cudssExecute(
      ctx.handle,
      CUDSS_PHASE_SOLVE,
      ctx.config,
      cache->solver_data,
      cache->matrix,
      ctx.dense.solution,
      ctx.dense.rhs);
  if (solve_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssExecute(SOLVE)", solve_status);
  }
  return ffi::Error();
}

template <typename Scalar>
ffi::Error StepCommon(
    cudaStream_t stream,
    uint64_t context_handle,
    bool allow_cache,
    ffi::AnyBuffer dt_buf,
    ffi::AnyBuffer y_buf,
    ffi::AnyBuffer indptr_buf,
    ffi::AnyBuffer indices_buf,
    ffi::AnyBuffer values_buf,
    ffi::AnyBuffer nnz_buf,
    ffi::Result<ffi::AnyBuffer> out_buf) {
#ifndef HAVE_CUDA_DIAGONAL_KERNEL
  (void)stream;
  (void)context_handle;
  (void)allow_cache;
  (void)dt_buf;
  (void)y_buf;
  (void)indptr_buf;
  (void)indices_buf;
  (void)values_buf;
  (void)nnz_buf;
  (void)out_buf;
  return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "Fused cuDSS step requires CUDA kernels");
#else
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
  ffi::Error scratch_status = EnsureStepScratch(*ctx);
  if (scratch_status.failure()) {
    return scratch_status;
  }

  auto dt_dims = dt_buf.dimensions();
  auto y_dims = y_buf.dimensions();
  auto indptr_dims = indptr_buf.dimensions();
  auto indices_dims = indices_buf.dimensions();
  auto out_dims = out_buf->dimensions();
  if (dt_dims.size() != 1 || y_dims.size() != 2 || indptr_dims.size() != 2 || indices_dims.size() != 2 ||
      out_dims.size() != 3 || out_dims[0] != 2 || y_dims[0] != ctx->batch_size || y_dims[1] != ctx->n_vars ||
      out_dims[1] != ctx->batch_size || out_dims[2] != ctx->n_vars) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Unexpected cuDSS fused step buffer shapes");
  }
  if (nnz_buf.element_type() != ffi::DataType::S32) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "cuDSS fused step expects int32 nnz");
  }

  PatternCache* cache = nullptr;
  if (ffi::Error status = EnsureActiveCache(*ctx, indptr_buf, indices_buf, nnz_buf, cache); status.failure()) {
    return status;
  }

  const int64_t batch_size = ctx->batch_size;
  const int64_t n_vars = ctx->n_vars;
  const int64_t total = batch_size * n_vars;
  const auto* dt = static_cast<const double*>(dt_buf.untyped_data());
  const auto* y = static_cast<const Scalar*>(y_buf.untyped_data());
  const auto* values = static_cast<const Scalar*>(values_buf.untyped_data());
  auto* out = static_cast<Scalar*>(out_buf->untyped_data());
  Scalar* out_y = out;
  Scalar* out_err = out + total;
  Scalar* rhs = BufferPtr<Scalar>(ctx->scratch.rhs);
  Scalar* state = BufferPtr<Scalar>(ctx->scratch.state);
  Scalar* du = BufferPtr<Scalar>(ctx->scratch.du);
  Scalar* k1 = BufferPtr<Scalar>(ctx->scratch.k1);
  Scalar* k2 = BufferPtr<Scalar>(ctx->scratch.k2);
  Scalar* k3 = BufferPtr<Scalar>(ctx->scratch.k3);
  Scalar* k4 = BufferPtr<Scalar>(ctx->scratch.k4);
  Scalar* k5 = BufferPtr<Scalar>(ctx->scratch.k5);
  Scalar* k6 = BufferPtr<Scalar>(ctx->scratch.k6);
  Scalar* k7 = BufferPtr<Scalar>(ctx->scratch.k7);

  cudssStatus_t rhs_status = cudssMatrixSetValues(ctx->dense.rhs, ctx->scratch.rhs.ptr);
  if (rhs_status != CUDSS_STATUS_SUCCESS) {
    return CuDssError("cudssMatrixSetValues(rhs)", rhs_status);
  }
  const Scalar* none = nullptr;
  bool prepared_current_values = false;

  auto run_step = [&](const double* dt_ptr, const Scalar* y_ptr, Scalar* out_y_ptr, Scalar* out_err_ptr, bool count_solves) -> ffi::Error {
    auto solve_into = [&](void* solution_ptr) -> ffi::Error {
      return count_solves ? SolveTo(*ctx, cache, solution_ptr) : SolveToNoCount(*ctx, cache, solution_ptr);
    };

    LaunchMatvec(rhs, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), y_ptr, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(k1)"); status.failure()) return status;
    if (ffi::Error status = solve_into(ctx->scratch.k1.ptr); status.failure()) return status;

    LaunchState(state, y_ptr, k1, kA21, none, 0.0, none, 0.0, none, 0.0, none, 0.0, none, 0.0, none, 0.0, total, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_state(k2)"); status.failure()) return status;
    LaunchMatvec(du, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), state, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(du2)"); status.failure()) return status;
    LaunchRhs(rhs, du, dt_ptr, k1, kC21, none, 0.0, none, 0.0, none, 0.0, none, 0.0, none, 0.0, none, 0.0, n_vars, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_rhs(k2)"); status.failure()) return status;
    if (ffi::Error status = solve_into(ctx->scratch.k2.ptr); status.failure()) return status;

    LaunchState(state, y_ptr, k1, kA31, k2, kA32, none, 0.0, none, 0.0, none, 0.0, none, 0.0, none, 0.0, total, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_state(k3)"); status.failure()) return status;
    LaunchMatvec(du, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), state, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(du3)"); status.failure()) return status;
    LaunchRhs(rhs, du, dt_ptr, k1, kC31, k2, kC32, none, 0.0, none, 0.0, none, 0.0, none, 0.0, none, 0.0, n_vars, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_rhs(k3)"); status.failure()) return status;
    if (ffi::Error status = solve_into(ctx->scratch.k3.ptr); status.failure()) return status;

    LaunchState(state, y_ptr, k1, kA41, k2, kA42, k3, kA43, none, 0.0, none, 0.0, none, 0.0, none, 0.0, total, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_state(k4)"); status.failure()) return status;
    LaunchMatvec(du, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), state, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(du4)"); status.failure()) return status;
    LaunchRhs(rhs, du, dt_ptr, k1, kC41, k2, kC42, k3, kC43, none, 0.0, none, 0.0, none, 0.0, none, 0.0, n_vars, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_rhs(k4)"); status.failure()) return status;
    if (ffi::Error status = solve_into(ctx->scratch.k4.ptr); status.failure()) return status;

    LaunchState(state, y_ptr, k1, kA51, k2, kA52, k3, kA53, k4, kA54, none, 0.0, none, 0.0, none, 0.0, total, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_state(k5)"); status.failure()) return status;
    LaunchMatvec(du, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), state, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(du5)"); status.failure()) return status;
    LaunchRhs(rhs, du, dt_ptr, k1, kC51, k2, kC52, k3, kC53, k4, kC54, none, 0.0, none, 0.0, none, 0.0, n_vars, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_rhs(k5)"); status.failure()) return status;
    if (ffi::Error status = solve_into(ctx->scratch.k5.ptr); status.failure()) return status;

    LaunchState(state, y_ptr, k1, kA61, k2, kA62, k3, kA63, k4, kA64, k5, kA65, none, 0.0, none, 0.0, total, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_state(k6)"); status.failure()) return status;
    LaunchMatvec(du, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), state, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(du6)"); status.failure()) return status;
    LaunchRhs(rhs, du, dt_ptr, k1, kC61, k2, kC62, k3, kC63, k4, kC64, k5, kC65, none, 0.0, none, 0.0, n_vars, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_rhs(k6)"); status.failure()) return status;
    if (ffi::Error status = solve_into(ctx->scratch.k6.ptr); status.failure()) return status;

    LaunchAdd(state, state, k6, total, stream);
    if (ffi::Error status = CheckLaunch("launch_add_vectors(state+k6)"); status.failure()) return status;
    LaunchMatvec(du, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), state, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(du7)"); status.failure()) return status;
    LaunchRhs(rhs, du, dt_ptr, k1, kC71, k2, kC72, k3, kC73, k4, kC74, k5, kC75, k6, kC76, none, 0.0, n_vars, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_rhs(k7)"); status.failure()) return status;
    if (ffi::Error status = solve_into(ctx->scratch.k7.ptr); status.failure()) return status;

    LaunchAdd(state, state, k7, total, stream);
    if (ffi::Error status = CheckLaunch("launch_add_vectors(state+k7)"); status.failure()) return status;
    LaunchMatvec(du, cache->d_indptr.ptr, cache->d_indices.ptr, reinterpret_cast<const Scalar*>(cache->d_values.ptr), state, n_vars, cache->union_nnz, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_batched_csr_matvec(du8)"); status.failure()) return status;
    LaunchRhs(rhs, du, dt_ptr, k1, kC81, k2, kC82, k3, kC83, k4, kC84, k5, kC85, k6, kC86, k7, kC87, n_vars, batch_size, stream);
    if (ffi::Error status = CheckLaunch("launch_stage_rhs(k8)"); status.failure()) return status;
    if (ffi::Error status = solve_into(out_err_ptr); status.failure()) return status;

    LaunchAdd(out_y_ptr, state, out_err_ptr, total, stream);
    if (ffi::Error status = CheckLaunch("launch_add_vectors(y_next)"); status.failure()) return status;
    return ffi::Error();
  };

  if (!cache->analyzed) {
    if (ffi::Error status = PrepareValues<Scalar>(
            *ctx,
            *cache,
            allow_cache,
            dt,
            values,
            stream,
            /*count_prepare=*/true);
        status.failure()) {
      return status;
    }
    prepared_current_values = true;
    if (ffi::Error status = EnsureAnalyzed(*ctx, *cache); status.failure()) {
      return status;
    }
  }

  if (ctx->graph_enabled) {
    if (ffi::Error status = EnsureGraphBuffers(*ctx); status.failure()) {
      return status;
    }
    if (!allow_cache &&
        !cache->graph_values_in.Resize(static_cast<size_t>(batch_size * cache->union_nnz * ValueBytes(ctx->precision_code)))) {
      return ffi::Error(ffi::ErrorCode::kResourceExhausted, "Failed to allocate cuDSS graph values buffer");
    }
    if (cudaMemcpyAsync(ctx->graph_dt.ptr, dt, sizeof(double) * batch_size, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return CudaError("cudaMemcpyAsync(graph_dt)", cudaGetLastError());
    }
    if (cudaMemcpyAsync(ctx->graph_y.ptr, y, total * sizeof(Scalar), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return CudaError("cudaMemcpyAsync(graph_y)", cudaGetLastError());
    }
    if (!allow_cache &&
        cudaMemcpyAsync(
            cache->graph_values_in.ptr,
            values,
            static_cast<size_t>(batch_size * cache->union_nnz * sizeof(Scalar)),
            cudaMemcpyDeviceToDevice,
            stream) != cudaSuccess) {
      return CudaError("cudaMemcpyAsync(graph_values_in)", cudaGetLastError());
    }

    Scalar* graph_y = reinterpret_cast<Scalar*>(ctx->graph_y.ptr);
    Scalar* graph_out_y = reinterpret_cast<Scalar*>(ctx->graph_out.ptr);
    Scalar* graph_out_err = graph_out_y + total;
    const Scalar* graph_values = allow_cache
        ? reinterpret_cast<const Scalar*>(cache->d_base_values.ptr)
        : reinterpret_cast<const Scalar*>(cache->graph_values_in.ptr);

    if (!cache->graph_ready) {
      cudaError_t capture_status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
      if (capture_status == cudaSuccess) {
        ffi::Error graph_status = PrepareValues<Scalar>(
            *ctx,
            *cache,
            allow_cache,
            ctx->graph_dt.ptr,
            graph_values,
            stream,
            /*count_prepare=*/false);
        if (!graph_status.failure()) {
          graph_status = FactorizeActive(*ctx, *cache, /*count_factorization=*/false);
        }
        if (!graph_status.failure()) {
          graph_status = run_step(ctx->graph_dt.ptr, graph_y, graph_out_y, graph_out_err, false);
        }
        if (graph_status.failure()) {
          cudaStreamEndCapture(stream, &cache->step_graph);
          return graph_status;
        }
        cudaError_t end_status = cudaStreamEndCapture(stream, &cache->step_graph);
        if (end_status == cudaSuccess) {
          cudaError_t inst_status = cudaGraphInstantiate(&cache->step_graph_exec, cache->step_graph, 0);
          if (inst_status == cudaSuccess) {
            cache->graph_ready = true;
            ctx->graph_capture_count += 1;
          } else {
            cudaGraphDestroy(cache->step_graph);
            cache->step_graph = nullptr;
          }
        }
      }
    }

    if (cache->graph_ready) {
      cudaError_t launch_status = cudaGraphLaunch(cache->step_graph_exec, stream);
      if (launch_status != cudaSuccess) {
        return CudaError("cudaGraphLaunch(step)", launch_status);
      }
      ctx->device_prepare_fast_path_count += 1;
      ctx->factorization_count += 1;
      ctx->solve_count += 8;
      if (cache->graph_launched) {
        ctx->graph_replay_count += 1;
      } else {
        cache->graph_launched = true;
      }
      if (cudaMemcpyAsync(out, ctx->graph_out.ptr, total * 2 * sizeof(Scalar), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return CudaError("cudaMemcpyAsync(graph_out)", cudaGetLastError());
      }
      return ffi::Error();
    }
  }

  if (!prepared_current_values) {
    if (ffi::Error status = PrepareValues<Scalar>(
            *ctx,
            *cache,
            allow_cache,
            dt,
            values,
            stream,
            /*count_prepare=*/true);
        status.failure()) {
      return status;
    }
  }
  if (ffi::Error status = EnsureAnalyzed(*ctx, *cache); status.failure()) {
    return status;
  }
  if (ffi::Error status = FactorizeActive(*ctx, *cache, /*count_factorization=*/true); status.failure()) {
    return status;
  }
  return run_step(dt, y, out_y, out_err, true);
#endif
}

ffi::Error StepStaticImpl(
    cudaStream_t stream,
    uint64_t context_handle,
    ffi::AnyBuffer dt_buf,
    ffi::AnyBuffer y_buf,
    ffi::AnyBuffer indptr_buf,
    ffi::AnyBuffer indices_buf,
    ffi::AnyBuffer values_buf,
    ffi::AnyBuffer nnz_buf,
    ffi::Result<ffi::AnyBuffer> out_buf) {
  if (values_buf.element_type() == ffi::DataType::F32) {
    return StepCommon<float>(stream, context_handle, true, dt_buf, y_buf, indptr_buf, indices_buf, values_buf, nnz_buf, out_buf);
  }
  return StepCommon<double>(stream, context_handle, true, dt_buf, y_buf, indptr_buf, indices_buf, values_buf, nnz_buf, out_buf);
}

ffi::Error StepDynamicImpl(
    cudaStream_t stream,
    uint64_t context_handle,
    ffi::AnyBuffer dt_buf,
    ffi::AnyBuffer y_buf,
    ffi::AnyBuffer indptr_buf,
    ffi::AnyBuffer indices_buf,
    ffi::AnyBuffer values_buf,
    ffi::AnyBuffer nnz_buf,
    ffi::Result<ffi::AnyBuffer> out_buf) {
  if (values_buf.element_type() == ffi::DataType::F32) {
    return StepCommon<float>(stream, context_handle, false, dt_buf, y_buf, indptr_buf, indices_buf, values_buf, nnz_buf, out_buf);
  }
  return StepCommon<double>(stream, context_handle, false, dt_buf, y_buf, indptr_buf, indices_buf, values_buf, nnz_buf, out_buf);
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rodas5_cudss_step,
    StepStaticImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<uint64_t>("context_handle")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rodas5_cudss_step_dynamic,
    StepDynamicImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<uint64_t>("context_handle")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
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
  if (EnvFlagEnabled("RODAS5_CUDSS_ENABLE_GRAPH")) {
    cudssDeviceMemHandler_t handler{};
    handler.ctx = nullptr;
    handler.device_alloc = AsyncDeviceAlloc;
    handler.device_free = AsyncDeviceFree;
    std::snprintf(handler.name, sizeof(handler.name), "rodas5_async");
    if (cudssSetDeviceMemHandler(ctx->handle, &handler) == CUDSS_STATUS_SUCCESS) {
      ctx->graph_enabled = true;
    }
  }
  std::string threading_lib = ResolveThreadingLibrary();
  if (!threading_lib.empty() &&
      cudssSetThreadingLayer(ctx->handle, threading_lib.c_str()) == CUDSS_STATUS_SUCCESS) {
    int host_nthreads = EnvIntOr("RODAS5_CUDSS_HOST_NTHREADS", EnvIntOr("CUDSS_CONFIG_HOST_NTHREADS", 0));
    if (host_nthreads > 0) {
      if (cudssConfigSet(ctx->config, CUDSS_CONFIG_HOST_NTHREADS, &host_nthreads, sizeof(host_nthreads)) == CUDSS_STATUS_SUCCESS) {
        ctx->mt_analysis_enabled = true;
      }
    } else {
      ctx->mt_analysis_enabled = true;
    }
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

extern "C" uint64_t rodas5_cudss_host_prepare_slow_path_count(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->host_prepare_slow_path_count;
  }
  return 0;
}

extern "C" uint64_t rodas5_cudss_device_prepare_fast_path_count(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->device_prepare_fast_path_count;
  }
  return 0;
}

extern "C" uint64_t rodas5_cudss_graph_capture_count(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->graph_capture_count;
  }
  return 0;
}

extern "C" uint64_t rodas5_cudss_graph_replay_count(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->graph_replay_count;
  }
  return 0;
}

extern "C" int rodas5_cudss_mt_analysis_enabled(uint64_t handle) {
  if (Context* ctx = LookupContext(handle)) {
    return ctx->mt_analysis_enabled ? 1 : 0;
  }
  return 0;
}

extern "C" int rodas5_cudss_has_fused_step() {
#ifdef HAVE_CUDA_DIAGONAL_KERNEL
  return 1;
#else
  return 0;
#endif
}
