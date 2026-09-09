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

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>
#include "jaxlib/mosaic/gpu/launch_config.h"

extern "C" void mosaic_gpu_build_kernel_spec(
    mosaic::gpu::MosaicKernelSpec* cfg, uint32_t grid_x, uint32_t grid_y,
    uint32_t grid_z, uint32_t cluster_x, uint32_t cluster_y, uint32_t cluster_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z, uint32_t smem_bytes,
    int32_t uses_pdl, int32_t num_args, void** arg_ptrs,
    const int32_t* arg_bytes);

namespace mosaic::gpu {
namespace {

TEST(RuntimeTest, BuildKernelSpecPacksDeviceAndHostArguments) {
  // A dummy host-side argument.
  struct HostArg {
    uint64_t data[4] = {0xAA, 0xBB, 0xCC, 0xDD};
  } host_arg;

  void* mock_dev_ptr = reinterpret_cast<void*>(0xDEADBEEF0000ULL);

  void* raw_slots[2] = {mock_dev_ptr, &host_arg};
  void* arg_ptrs[2] = {&raw_slots[0], raw_slots[1]};
  int32_t arg_bytes[2] = {0, sizeof(HostArg)};

  MosaicKernelSpec cfg;
  mosaic_gpu_build_kernel_spec(&cfg,       // out-param
                               2, 4, 1,    // grid
                               1, 2, 1,    // cluster
                               128, 1, 1,  // block
                               /*smem_bytes=*/2048,
                               /*uses_pdl=*/1,
                               /*num_args=*/2, arg_ptrs, arg_bytes);

  EXPECT_EQ(cfg.grid.x, 2);
  EXPECT_EQ(cfg.grid.y, 4);
  EXPECT_EQ(cfg.grid.z, 1);
  EXPECT_EQ(cfg.cluster.x, 1);
  EXPECT_EQ(cfg.cluster.y, 2);
  EXPECT_EQ(cfg.cluster.z, 1);
  EXPECT_EQ(cfg.block.x, 128);
  EXPECT_EQ(cfg.block.y, 1);
  EXPECT_EQ(cfg.block.z, 1);
  EXPECT_EQ(cfg.smem_bytes, 2048);
  EXPECT_TRUE(cfg.uses_pdl);

  ASSERT_EQ(cfg.args.size(), 2);

  // Device pointer arg
  EXPECT_FALSE(cfg.args[0].is_host);
  EXPECT_EQ(cfg.args[0].size, 0);
  EXPECT_EQ(cfg.args[0].value, mock_dev_ptr);

  // Host arg
  EXPECT_TRUE(cfg.args[1].is_host);
  EXPECT_EQ(cfg.args[1].size, sizeof(HostArg));
  EXPECT_NE(cfg.args[1].value, &host_arg);
  EXPECT_EQ(std::memcmp(cfg.args[1].value, &host_arg, sizeof(HostArg)), 0);

  // kernel_params() format for cuLaunchKernelEx
  auto params = cfg.kernel_params();
  ASSERT_EQ(params.size(), 2);
  EXPECT_EQ(*reinterpret_cast<void**>(params[0]), mock_dev_ptr);
  EXPECT_EQ(params[1], cfg.args[1].value);
}

}  // namespace
}  // namespace mosaic::gpu
