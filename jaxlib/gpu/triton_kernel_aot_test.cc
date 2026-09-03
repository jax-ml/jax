/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "jaxlib/gpu/triton_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/stream_executor/cuda/cuda_platform.h"  // IWYU pragma: keep
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"

namespace jax::JAX_GPU_NAMESPACE {
namespace {

class TritonKernelAotTest : public ::testing::Test {};

TEST_F(TritonKernelAotTest, TritonKernelCallFfiAotFromHloFile) {
  // Register FFI handler with AOT instantiate hook
  XLA_FFI_Handler_Bundle bundle = {
      /*instantiate=*/kTritonKernelCallFfiInstantiate,
      /*prepare=*/nullptr,
      /*initialize=*/kTritonKernelCallFfiInitialize,
      /*execute=*/kTritonKernelCallFfi,
  };
  if (auto* error = xla::ffi::Ffi::RegisterStaticHandler(
          xla::ffi::GetXlaFfiApi(), "triton_kernel_call_ffi", "CUDA", bundle)) {
    FAIL() << "RegisterStaticHandler failed";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                       xla::GetXlaPjrtGpuClient(/*options=*/{}));

  std::string hlo_path =
      tsl::GetDataDependencyFilepath("py/jax/jaxlib/gpu/triton_add_kernel.hlo");
  std::string hlo_string;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo_string));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(hlo_string));

  xla::CompileOptions compile_options;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtExecutable> executable,
      client->Compile(xla::XlaComputation(module->ToProto()), compile_options));

  ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                       executable->SerializeExecutable());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
      client->LoadSerializedExecutable(serialized_aot_result, compile_options,
                                       xla::LoadOptions()));

  // Create input literals and buffers (1.0 + 2.0 == 3.0)
  std::vector<float> data_a(1024, 1.0f);
  std::vector<float> data_b(1024, 2.0f);
  const xla::Literal literal_a = xla::LiteralUtil::CreateR1<float>(data_a);
  const xla::Literal literal_b = xla::LiteralUtil::CreateR1<float>(data_b);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer_a,
      client->BufferFromHostLiteral(literal_a, client->memory_spaces()[0]));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer_b,
      client->BufferFromHostLiteral(literal_b, client->memory_spaces()[0]));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> result_buffers,
      loaded_executable->Execute({{buffer_a.get(), buffer_b.get()}},
                                 /*options=*/{}));

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result,
                       result_buffers[0][0]->ToLiteral().Await());

  std::vector<float> expected(1024, 3.0f);
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      xla::LiteralUtil::CreateR1<float>(expected), *result));
}
}  // namespace
}  // namespace jax::JAX_GPU_NAMESPACE
