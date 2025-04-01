/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_MOSAIC_GPU_COMM_H_
#define JAXLIB_MOSAIC_GPU_COMM_H_

#include <dlfcn.h>
#include <mutex>
#include <cstdio>

#include "third_party/gpus/cuda/include/cuda.h"
#include "cuda_runtime_api.h"

#define NVSHMEM_SUCCESS 0
#define NVSHMEM_LIB_SONAME "libnvshmem_host.so.3"

namespace mosaic {
namespace gpu {

#define NVSHMEM_SET_FN(FnName)                                            \
  FnName = reinterpret_cast<decltype(FnName)>(dlsym(library, #FnName));   \
  if (!FnName) {                                                          \
    fprintf(stderr, #FnName " not available in this library.");           \
    abort();                                                              \
  }

class NvshmemApi {
 public:
  // Returns a default NvshmemApi for a current process.
  // NvshmemApi follows the Singleton design pattern
  static NvshmemApi& Default() {
    static NvshmemApi instance;
    return instance;
  }

  int cumodule_int(CUmodule module) {
    std::lock_guard<std::mutex> lock(mutex_);
    return nvshmemx_cumodule_init(module);
  }

  void barrier_all_on_stream(cudaStream_t stream) {
    nvshmemx_barrier_all_on_stream(stream);
  }

  NvshmemApi(NvshmemApi const&)     = delete;
  void operator=(NvshmemApi const&) = delete;

 private:
  NvshmemApi() {
    const char* env_value = getenv("NVSHMEM_LIBRARY_PATH");
    const char* libnvshmem_path =
      env_value && *env_value != 0 ? env_value : NVSHMEM_LIB_SONAME;
    void* library = dlopen(libnvshmem_path, RTLD_LAZY);
    if (library == nullptr) {
      fprintf(stderr, "Failed to open %s library: %s", libnvshmem_path, dlerror());
      abort();
    }

    // Initialize supported NVSHMEM host API
    NVSHMEM_SET_FN(nvshmemx_cumodule_init)
    NVSHMEM_SET_FN(nvshmemx_barrier_all_on_stream)
  }

  // Dlopened NVSHMEM API
  int (*nvshmemx_cumodule_init)(CUmodule);
  int (*nvshmemx_barrier_all_on_stream)(cudaStream_t);

  std::mutex mutex_;
};

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_COMM_H_
