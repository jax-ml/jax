// Copyright 2025 The JAX Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "aiter_mha_common_utils.h"
#include <chrono>
#include <cstring>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_AMD__
using hip_bf16_type = hip_bfloat16;
#else
using hip_bf16_type = __hip_bfloat16;
#endif

namespace jax_aiter {
namespace mha_utils {

template <typename T>
__global__ void mqa_gqa_reduce_kernel(const T *__restrict__ d_expanded,
                                      T *__restrict__ d_reduced,
                                      int64_t batch_size, int64_t seqlen,
                                      int64_t num_heads_q, int64_t num_heads_k,
                                      int64_t head_dim, int64_t num_groups) {

  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_elements = batch_size * seqlen * num_heads_k * head_dim;

  if (tid >= total_elements)
    return;

  int64_t b = tid / (seqlen * num_heads_k * head_dim);
  int64_t remainder = tid % (seqlen * num_heads_k * head_dim);
  int64_t s = remainder / (num_heads_k * head_dim);
  remainder = remainder % (num_heads_k * head_dim);
  int64_t hk = remainder / head_dim;
  int64_t d = remainder % head_dim;

  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int h = hk * num_groups + g;
    int expanded_idx = b * seqlen * num_heads_q * head_dim +
                       s * num_heads_q * head_dim + h * head_dim + d;
    sum += static_cast<float>(d_expanded[expanded_idx]);
  }

  d_reduced[tid] = static_cast<T>(sum);
}

template __global__ void
mqa_gqa_reduce_kernel<__half>(const __half *__restrict__, __half *__restrict__,
                              int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t);
template __global__ void mqa_gqa_reduce_kernel<hip_bf16_type>(
    const hip_bf16_type *__restrict__, hip_bf16_type *__restrict__, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t);

JAX_AITER_EXPORT
xla::ffi::Error launch_mqa_gqa_reduction(const void *src, void *dst,
                                         int64_t batch_size, int64_t seqlen_k,
                                         int64_t num_heads_q,
                                         int64_t num_heads_k,
                                         int64_t head_size, int64_t groups,
                                         xla::ffi::DataType dtype,
                                         hipStream_t stream) {

  int64_t total_elements = batch_size * seqlen_k * num_heads_k * head_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  if (dtype == xla::ffi::DataType::F16) {
    mqa_gqa_reduce_kernel<__half><<<blocks, threads, 0, stream>>>(
        static_cast<const __half *>(src), static_cast<__half *>(dst),
        batch_size, seqlen_k, num_heads_q, num_heads_k, head_size, groups);
  } else if (dtype == xla::ffi::DataType::BF16) {
    mqa_gqa_reduce_kernel<hip_bf16_type><<<blocks, threads, 0, stream>>>(
        static_cast<const hip_bf16_type *>(src),
        static_cast<hip_bf16_type *>(dst), batch_size, seqlen_k, num_heads_q,
        num_heads_k, head_size, groups);
  } else {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "launch_mqa_gqa_reduction: unsupported dtype " +
            std::to_string(static_cast<int>(dtype)) + " (expected F16 or BF16)");
  }

  return xla::ffi::Error::Success();
}

JAX_AITER_EXPORT
xla::ffi::Error
prepare_rng_state_for_fwd(hipStream_t stream, float dropout_p, int dev_idx,
                          int64_t batch_size, int64_t num_heads,
                          const std::optional<xla::ffi::AnyBuffer> &gen,
                          xla::ffi::Result<xla::ffi::AnyBuffer> &rng_state,
                          RngStatePointers &out_ptrs) {

  if (rng_state->size_bytes() < 2 * sizeof(uint64_t)) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "rng_state result buffer must have at least 2 uint64s (16 bytes)");
  }

  void *rng_base = rng_state->untyped_data();

  if (dropout_p > 0.0f) {
    if (gen.has_value() && gen->size_bytes() >= 2 * sizeof(uint64_t)) {
      // gen is a device buffer.  Issue an async D->D copy on the compute
      // stream; do NOT do a synchronous hipMemcpy on the hot path.
      if (VLOG_IS_ON(1)) {
        uint64_t host_gen[2] = {0, 0};
        hipError_t copy_err =
            hipMemcpy(host_gen, gen->untyped_data(), 2 * sizeof(uint64_t),
                      hipMemcpyDeviceToHost);
        if (copy_err == hipSuccess) {
          VLOG(1) << "[JAX_AITER_CPP] Using provided generator with seed: "
                  << host_gen[0] << ", offset: " << host_gen[1];
        } else {
          VLOG(1) << "[JAX_AITER_CPP] Using provided generator (failed to "
                     "stage seed/offset for logging: "
                  << hipGetErrorString(copy_err) << ")";
        }
      }

      hipError_t err = hipMemcpyAsync(
          rng_base, gen->untyped_data(), 2 * sizeof(uint64_t),
          hipMemcpyDeviceToDevice, stream);
      if (err != hipSuccess) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            std::string("Failed to copy RNG state to device: ") +
                hipGetErrorString(err));
      }
    } else {
      auto now = std::chrono::high_resolution_clock::now();
      auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                           now.time_since_epoch())
                           .count();

      uint64_t seed_value =
          static_cast<uint64_t>(timestamp) ^ static_cast<uint64_t>(dev_idx);
      uint64_t offset_value = static_cast<uint64_t>(
          batch_size * num_heads * ck_tile::get_warp_size());

      VLOG(1) << "Generated RNG with seed: " << seed_value
              << ", offset: " << offset_value << " (no gen provided)";

      uint64_t host_rng[2] = {seed_value, offset_value};
      hipError_t err =
          hipMemcpy(rng_base, host_rng, 2 * sizeof(uint64_t),
                         hipMemcpyHostToDevice);

      if (err != hipSuccess) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            std::string("Failed to copy generated RNG state to device: ") +
                hipGetErrorString(err));
      }
    }

    out_ptrs.seed = rng_base;
    out_ptrs.offset = static_cast<char *>(rng_base) + sizeof(uint64_t);
  } else {
    hipError_t err = hipMemsetAsync(rng_base, 0, 2 * sizeof(uint64_t), stream);
    if (err != hipSuccess) {
      VLOG(1) << "[JAX_AITER_CPP] Warning: Failed to zero RNG state: "
              << hipGetErrorString(err);
    }

    out_ptrs.seed = rng_base;
    out_ptrs.offset = static_cast<char *>(rng_base) + sizeof(uint64_t);
  }

  return xla::ffi::Error::Success();
}

} // namespace mha_utils
} // namespace jax_aiter