/* Copyright 2024 The JAX Authors.

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

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>

#include "jaxlib/gpu/vendor.h"

#if defined(JAX_GPU_CUDA)
#include "jaxlib/mosaic/gpu/nvshmem.h"
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // defined(JAX_GPU_CUDA)

namespace {
template <typename... Args>
void abort_on_error(gpuDrvError_t result, const char *fmt, Args &&...args) {
  if (result != gpuDrvSuccess) {
    const char *ptr{gpuGetErrorString(result)};
    if (ptr == nullptr) ptr = "<unknown error>";
    fprintf(stderr, fmt, std::forward<Args>(args)..., ptr);
    abort();
  }
}
}  // namespace

extern "C" {

#if defined(JAX_GPU_CUDA)
void mosaic_gpu_init_tma_desc(CUtensorMap *tma_desc, void *base_addr,
                              int64_t elem_type, int64_t rank, int64_t *sizes,
                              int64_t *strides, int64_t swizzle_bytes,
                              int64_t *window_shape) {
  if (((uintptr_t)tma_desc) % 64 != 0) {
    fprintf(stderr,
            "TMA descriptor address must be 64 byte aligned, but got: %p\n",
            tma_desc);
    abort();
  }

  CUtensorMapDataType data_type;
  int64_t elem_bitwidth;
  // types are defined in: launch_context._tma_dma_type()
  if (elem_type == 8) {
    // this is for int2s
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    elem_bitwidth = 2;
  } else if (elem_type == 0) {
    // this is for int4s
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    elem_bitwidth = 4;
  } else if (elem_type == 1) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    elem_bitwidth = 8;
  } else if (elem_type == 2) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    elem_bitwidth = 16;
  } else if (elem_type == 3) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    elem_bitwidth = 32;
  } else if (elem_type == 4) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT64;
    elem_bitwidth = 64;
  } else if (elem_type == 5) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    elem_bitwidth = 16;
  } else if (elem_type == 6) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    elem_bitwidth = 32;
  } else if (elem_type == 7) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    elem_bitwidth = 16;
  } else if (elem_type == 9) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_INT32;
    elem_bitwidth = 32;
  } else if (elem_type == 10) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_INT64;
    elem_bitwidth = 64;
  } else {
    fprintf(stderr, "Unsupported element type: %ld \n", elem_type);
    abort();
  }

  // Pack sub byte types in 8 bit pairs.
  int64_t elem_bytewidth;
  if (elem_bitwidth < 8) {
    // Check that it's a power of 2.
    assert((elem_bitwidth & (elem_bitwidth - 1)) == 0);
    int packing = 8 / elem_bitwidth;
    assert(sizes[rank - 1] % packing == 0);
    assert(window_shape[rank - 1] % packing == 0);
    assert(strides[rank - 1] == 1);

    // TMA requires that the last dimension be the contiguous one so we pack the
    // elements under that assumption.
    sizes[rank - 1] /= packing;
    window_shape[rank - 1] /= packing;
    for (int i = 0; i < rank - 1; i++) {
      strides[i] /= packing;
    }
    elem_bytewidth = 1;
  } else {
    elem_bytewidth = elem_bitwidth / 8;
  }

  if (rank < 1 || rank > 5) {
    fprintf(stderr, "Rank must be in [1, 5], but got %ld\n", rank);
    abort();
  }
  cuuint64_t tma_sizes[5] = {1, 1, 1, 1, 1};
  for (int i = 0; i < rank; ++i) {
    cuuint64_t tma_size_i = static_cast<cuuint64_t>(sizes[rank - i - 1]);
    if (tma_size_i > static_cast<cuuint64_t>(1) << 32) {
      fprintf(stderr,
              "TMA size must be less than 2**32, but got %ld at index %ld\n",
              tma_size_i, rank - i - 1);
      abort();
    }
    tma_sizes[i] = tma_size_i;
  }
  cuuint64_t tma_strides[5] = {1, 1, 1, 1, 1};
  if (strides[rank - 1] != 1) {
    fprintf(stderr, "Minormost stride must be 1, but got %ld\n",
            strides[rank - 1]);
    abort();
  }
  for (int i = 0; i < rank - 1; ++i) {  // We skip the implicit minor stride.
    cuuint64_t tma_stride_i =
        static_cast<cuuint64_t>(strides[rank - i - 2] * elem_bytewidth);
    if (tma_stride_i % 16 != 0 || tma_stride_i >= static_cast<cuuint64_t>(1)
                                                      << 40) {
      fprintf(stderr,
              "Byte strides must be divisible by 16 and less than 2**40, but "
              "got %ld (item stride = %ld, item size = %ld) at index %ld\n",
              tma_stride_i, strides[rank - 1], elem_bytewidth, rank - i - 2);
      abort();
    }
    tma_strides[i] = tma_stride_i;
  }
  cuuint32_t tma_window_shape[5] = {1, 1, 1, 1, 1};
  for (int64_t i = 0; i < rank; ++i) {
    cuuint32_t tma_window_shape_i =
        static_cast<cuuint32_t>(window_shape[rank - i - 1]);
    if (tma_window_shape_i > 256) {
      fprintf(stderr,
              "Window shape must be in [0, 256], but got %d at index %ld\n",
              tma_window_shape_i, rank - i - 1);
      abort();
    }
    if (i == 0 && (tma_window_shape_i * elem_bytewidth) % 16 != 0) {
      fprintf(stderr,
              "The last dimension of window shape must have a bytewidth "
              "divisible by 16, but got %d*%ld at index %ld\n",
              tma_window_shape_i, elem_bytewidth, rank - i - 1);
      abort();
    }
    tma_window_shape[i] = tma_window_shape_i;
  }
  cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMapSwizzle swizzle;
  if (swizzle_bytes == 16) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if (swizzle_bytes == 32) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (swizzle_bytes == 64) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (swizzle_bytes == 128) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  } else {
    fprintf(stderr, "Unsupported swizzle: %ld\n", swizzle_bytes);
    abort();
  }
  abort_on_error(
      cuTensorMapEncodeTiled(tma_desc, data_type, rank, base_addr, tma_sizes,
                             tma_strides, tma_window_shape, element_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled failed: %s\n");
}
#endif  // defined(JAX_GPU_CUDA)

#if defined(JAX_GPU_HIP)
void mosaic_gpu_init_tma_desc(void *tma_desc, void *base_addr,
                              int64_t elem_type, int64_t rank, int64_t *sizes,
                              int64_t *strides, int64_t swizzle_bytes,
                              int64_t *window_shape) {
  // Current gen. of AMD cards doesn't have TMA analogues
}
#endif  // defined(JAX_GPU_HIP)

void *mosaic_gpu_module_load(void *data) {
  gpuModule_t module = nullptr;
  abort_on_error(gpuModuleLoadData(&module, data),
                 "gpuModuleLoadData failed: %s\n");

#if defined(JAX_GPU_CUDA)
  {  // Set the NVSHMEM state if it's used by the module.
    CUdeviceptr ptr = 0;
    size_t size = 0;
    if (cuModuleGetGlobal(&ptr, &size, module,
                          "nvshmemi_device_lib_version_d") == CUDA_SUCCESS) {
      if (mosaic::gpu::NvshmemApi::Default().cumodule_init(module) !=
          NVSHMEM_SUCCESS) {
        fprintf(stderr, "nvshmemx_cumodule_init failed.\n");
        abort();
      }
    }
  }
#endif  // defined(JAX_GPU_CUDA)
  // TODO(Arech) ROCSHMEM support here?

  return module;
}

// cluster_size can be -1 when it's not statically known.
void *mosaic_gpu_get_function(gpuModule_t module, const char *name,
                              int32_t smem_bytes, int32_t cluster_size) {
  gpuFunction_t function = nullptr;
  abort_on_error(gpuModuleGetFunction(&function, module, name),
                 "Failed to retrieve function pointer to kernel \"%s\", "
                 "gpuModuleGetFunction failed: %s\n",
                 name);
  if (smem_bytes) {
    abort_on_error(
#if defined(JAX_GPU_CUDA)
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           smem_bytes)
#elif defined(JAX_GPU_HIP)
        hipFuncSetAttribute(
            function, hipFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)
#endif  // defined(JAX_GPU_CUDA) || defined(JAX_GPU_HIP)
            ,
        "Failed to set maximum dynamic shared memory size for kernel \"%s\" "
        "to %d bytes, cuFuncSetAttribute failed: %s\n",
        name, smem_bytes);
  }
#if defined(JAX_GPU_CUDA)
  if (cluster_size > 8) {
    abort_on_error(
        cuFuncSetAttribute(
            function, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1),
        "Failed to set allowed cluster size for kernel \"%s\" to %d, "
        "cuFuncSetAttribute failed: %s\n",
        name, cluster_size);
  }
#elif defined(JAX_GPU_HIP)
  if (cluster_size > 1) {
    fprintf(stderr, "Unsupported cluster size for kernel \"%s\" to be %d\n",
            name, cluster_size);
    abort();
  }
#endif  // defined(JAX_GPU_CUDA) || defined(JAX_GPU_HIP)
  return function;
}

void mosaic_gpu_launch_kernel(gpuFunction_t function, uint32_t grid_x,
                              uint32_t grid_y, uint32_t grid_z,
                              uint32_t cluster_x, uint32_t cluster_y,
                              uint32_t cluster_z, uint32_t block_x,
                              uint32_t block_y, uint32_t block_z,
                              uint32_t smem_bytes, gpuStream_t stream,
                              void **params) {
  gpuLaunchConfig_t config{
      .gridDimX = grid_x,
      .gridDimY = grid_y,
      .gridDimZ = grid_z,
      .blockDimX = block_x,
      .blockDimY = block_y,
      .blockDimZ = block_z,
      .sharedMemBytes = smem_bytes,
      .hStream = stream,
      .attrs = nullptr,
      .numAttrs = 0,
  };
  gpuLaunchAttribute_t cluster_attr;
  if (cluster_x != 0) {
#if defined(JAX_GPU_CUDA)
    cluster_attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster_attr.value.clusterDim = {
        .x = cluster_x,
        .y = cluster_y,
        .z = cluster_z,
    };
    config.attrs = &cluster_attr;
    config.numAttrs = 1;
#elif defined(JAX_GPU_HIP)
    fprintf(stderr, "Unsupported cluster size for kernel to launch: %d,%d,%d\n",
            cluster_x, cluster_y, cluster_z);
    abort();
#endif  // defined(JAX_GPU_CUDA) || defined(JAX_GPU_HIP)
  } else {
    assert(cluster_x == 0 && cluster_y == 0 && cluster_z == 0);
  }
  gpuDrvError_t result = gpuLaunchKernelEx(&config, function, params, nullptr);
#if defined(JAX_GPU_CUDA)
  if (result == CUDA_ERROR_INVALID_CLUSTER_SIZE) {
    int max_cluster_size;
    abort_on_error(cuOccupancyMaxPotentialClusterSize(&max_cluster_size,
                                                      function, &config),
                   "cuOccupancyMaxPotentialClusterSize failed: %s\n");
    fprintf(stderr,
            "cuLaunchKernel failed with invalid cluster size (%d, %d, %d)"
            ": maximum is %d\n",
            cluster_x, cluster_y, cluster_z, max_cluster_size);
    abort();
  } else {
#elif defined(JAX_GPU_HIP)
  if (result != gpuDrvSuccess) {
#endif
    abort_on_error(result, "gpuLaunchKernelEx: %s\n");
  }
}
}
