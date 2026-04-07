cmake_minimum_required(VERSION 3.20)
project(ode_test_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import sys; print(next(p for p in sys.path if 'site-packages' in p))"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT JAX_FFI_INCLUDE_DIR)
  set(JAX_FFI_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/jaxlib/include")
endif()
if(NOT CUDSS_INCLUDE_DIR)
  set(CUDSS_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cu12/include")
endif()
if(NOT CUDA_RUNTIME_INCLUDE_DIR)
  set(CUDA_RUNTIME_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/include")
endif()
if(NOT CUDA_NVCC_INCLUDE_DIR)
  set(CUDA_NVCC_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvcc/include")
endif()
if(NOT CUSPARSE_INCLUDE_DIR)
  set(CUSPARSE_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cusparse/include")
endif()
if(NOT CUDSS_LIBRARY_DIR)
  set(CUDSS_LIBRARY_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cu12/lib")
endif()
if(NOT CUDA_RUNTIME_LIBRARY_DIR)
  set(CUDA_RUNTIME_LIBRARY_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib")
endif()
if(NOT CUSPARSE_LIBRARY_DIR)
  set(CUSPARSE_LIBRARY_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cusparse/lib")
endif()
if(NOT CUBLAS_LIBRARY_DIR)
  set(CUBLAS_LIBRARY_DIR "${PYTHON_SITE_PACKAGES}/nvidia/cublas/lib")
endif()
if(NOT CUDSS_LIBRARY)
  find_library(CUDSS_LIBRARY NAMES libcudss.so.0 cudss PATHS "${CUDSS_LIBRARY_DIR}" NO_DEFAULT_PATH)
endif()
if(NOT CUDA_RUNTIME_LIBRARY)
  find_library(CUDA_RUNTIME_LIBRARY NAMES libcudart.so.12 cudart PATHS "${CUDA_RUNTIME_LIBRARY_DIR}" NO_DEFAULT_PATH)
endif()
if(NOT CUSPARSE_LIBRARY)
  find_library(CUSPARSE_LIBRARY NAMES libcusparse.so.12 cusparse PATHS "${CUSPARSE_LIBRARY_DIR}" NO_DEFAULT_PATH)
endif()
if(NOT CUBLAS_LIBRARY)
  find_library(CUBLAS_LIBRARY NAMES libcublas.so.12 cublas PATHS "${CUBLAS_LIBRARY_DIR}" NO_DEFAULT_PATH)
endif()

set(RODAS5_CUDSS_READY TRUE)
foreach(
  required_var
  JAX_FFI_INCLUDE_DIR
  CUDSS_INCLUDE_DIR
  CUDA_RUNTIME_INCLUDE_DIR
  CUDA_NVCC_INCLUDE_DIR
  CUSPARSE_INCLUDE_DIR
  CUDSS_LIBRARY
  CUDA_RUNTIME_LIBRARY
  CUSPARSE_LIBRARY
  CUBLAS_LIBRARY
)
  if(NOT ${required_var} OR ${required_var} MATCHES "-NOTFOUND$")
    set(RODAS5_CUDSS_READY FALSE)
  endif()
endforeach()

if(RODAS5_CUDSS_READY)
  set(CUDSS_SOURCES native/rodas5_cudss_ffi.cc)
  set(RODAS5_HAS_CUDA_HELPER_SOURCE FALSE)
  if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/native/cudss_diagonal_kernel.cu")
    set(RODAS5_HAS_CUDA_HELPER_SOURCE TRUE)
    list(APPEND CUDSS_SOURCES native/cudss_diagonal_kernel.cu)
  endif()
  add_library(rodas5_cudss_ffi SHARED ${CUDSS_SOURCES})
  target_include_directories(
    rodas5_cudss_ffi
    PRIVATE
      ${JAX_FFI_INCLUDE_DIR}
      ${CUDSS_INCLUDE_DIR}
      ${CUDA_RUNTIME_INCLUDE_DIR}
      ${CUDA_NVCC_INCLUDE_DIR}
      ${CUSPARSE_INCLUDE_DIR}
  )
  if(RODAS5_HAS_CUDA_HELPER_SOURCE)
    target_compile_definitions(rodas5_cudss_ffi PRIVATE HAVE_CUDA_DIAGONAL_KERNEL=1)
    target_compile_options(
      rodas5_cudss_ffi
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math -U_GNU_SOURCE>
    )
  endif()
  target_link_directories(
    rodas5_cudss_ffi
    PRIVATE
      ${CUDSS_LIBRARY_DIR}
      ${CUDA_RUNTIME_LIBRARY_DIR}
      ${CUSPARSE_LIBRARY_DIR}
      ${CUBLAS_LIBRARY_DIR}
  )
  target_link_libraries(
    rodas5_cudss_ffi
    PRIVATE
      ${CUDSS_LIBRARY}
      ${CUDA_RUNTIME_LIBRARY}
      ${CUSPARSE_LIBRARY}
      ${CUBLAS_LIBRARY}
  )
  set_target_properties(
    rodas5_cudss_ffi
    PROPERTIES
      OUTPUT_NAME "rodas5_cudss_ffi"
      BUILD_RPATH "${CUDSS_LIBRARY_DIR};${CUDA_RUNTIME_LIBRARY_DIR};${CUSPARSE_LIBRARY_DIR};${CUBLAS_LIBRARY_DIR}"
      INSTALL_RPATH "${CUDSS_LIBRARY_DIR};${CUDA_RUNTIME_LIBRARY_DIR};${CUSPARSE_LIBRARY_DIR};${CUBLAS_LIBRARY_DIR}"
  )
else()
  message(STATUS "Skipping rodas5_cudss_ffi build: CUDA/cuDSS headers or libraries were not found")
endif()
