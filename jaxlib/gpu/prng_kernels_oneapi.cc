/* Copyright 2026 The JAX Authors.

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

#include "jaxlib/gpu/prng_kernels.h"

#include <algorithm>
#include <cstdint>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

// Kernel name type — must be in a named namespace for SYCL linkage.
class prng_threefry2x32_kernel;

namespace {

void ThreeFry2x32Kernel(::sycl::nd_item<1> item,
                        const std::uint32_t* key0,
                        const std::uint32_t* key1,
                        const std::uint32_t* data0,
                        const std::uint32_t* data1,
                        std::uint32_t* out0, std::uint32_t* out1,
                        std::int64_t n, const std::int64_t* n_ptr) {
  if (n < 0) {
    n = *n_ptr;
  }

  const std::int64_t grid_stride = item.get_global_range(0);
  for (std::int64_t idx = item.get_global_id(0); idx < n;
       idx += grid_stride) {
    std::uint32_t rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};
    std::uint32_t x[2];
    std::uint32_t ks[3];

    // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
    ks[2] = 0x1BD11BDA;

    ks[0] = key0[idx];
    x[0] = data0[idx];
    ks[2] = ks[2] ^ key0[idx];

    ks[1] = key1[idx];
    x[1] = data1[idx];
    ks[2] = ks[2] ^ key1[idx];

    auto rotate_left = [](std::uint32_t v, std::uint32_t distance) {
      return (v << distance) | (v >> (32 - distance));
    };

    auto round = [&](std::uint32_t* v, std::uint32_t rotation) {
      v[0] += v[1];
      v[1] = rotate_left(v[1], rotation);
      v[1] ^= v[0];
    };

    // 20 rounds of ThreeFry2x32 (5 iterations x 4 rounds).
    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1];
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 1u;
    for (int i = 4; i < 8; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[2];
    x[1] = x[1] + ks[0] + 2u;
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1] + 3u;
    for (int i = 4; i < 8; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 4u;
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    out0[idx] = x[0] + ks[2];
    out1[idx] = x[1] + ks[0] + 5u;
  }
}

}  // namespace

void LaunchThreeFry2x32KernelFfi(gpuStream_t stream,
                                 std::int64_t n,
                                 std::uint32_t* keys0,
                                 std::uint32_t* keys1,
                                 std::uint32_t* data0,
                                 std::uint32_t* data1,
                                 std::uint32_t* out0,
                                 std::uint32_t* out1) {
  // Unlike CUDA, SYCL throws sycl::exception(errc::invalid) when constructing a
  // zero-sized nd_range. Guard against empty inputs (n <= 0) to avoid it.
  if (n <= 0) {
    return;
  }
  const int local_range = 128;  // work-items per work-group
  const std::int64_t num_work_groups =
      std::min<std::int64_t>(1024, (n + local_range - 1) / local_range);

  absl::Status status = TryCatchToStatus([&] {
    stream->submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<prng_threefry2x32_kernel>(
          ::sycl::nd_range<1>(::sycl::range<1>(num_work_groups * local_range),
                              ::sycl::range<1>(local_range)),
          [=](::sycl::nd_item<1> item) {
            ThreeFry2x32Kernel(item, keys0, keys1, data0, data1, out0, out1, n,
                               nullptr);
          });
    });
  });

  if (!status.ok()) {
    LOG(ERROR) << "LaunchThreeFry2x32KernelFfi: " << status.message();
  }

}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
