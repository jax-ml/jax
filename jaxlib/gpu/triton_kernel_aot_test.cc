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

#include <atomic>
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

class TritonKernelAotTest : public ::testing::Test {
 protected:
  inline static std::atomic<int> execute_count_{0};
  inline static std::atomic<int> record_count_{0};

  static XLA_FFI_Error* CountingExecute(XLA_FFI_CallFrame* frame) {
    ++execute_count_;
    return kTritonKernelCallFfi(frame);
  }
  static XLA_FFI_Error* CountingRecord(XLA_FFI_CallFrame* frame) {
    ++record_count_;
    return kTritonKernelCallFfiRecord(frame);
  }

  void SetUp() override {
    execute_count_ = 0;
    record_count_ = 0;
  }

  static void SetUpTestSuite() {
    XLA_FFI_Handler_Bundle bundle = {
        /*instantiate=*/kTritonKernelCallFfiInstantiate,
        /*prepare=*/nullptr,
        /*initialize=*/kTritonKernelCallFfiInitialize,
        /*execute=*/CountingExecute,
        /*record=*/CountingRecord,
    };
    if (auto* error = xla::ffi::Ffi::RegisterStaticHandler(
            xla::ffi::GetXlaFfiApi(), "triton_kernel_call_ffi", "CUDA",
            bundle)) {
      FAIL() << "RegisterStaticHandler failed";
    }
  }
};

TEST_F(TritonKernelAotTest, TritonKernelCallFfiAotFromHloFile) {
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

TEST_F(TritonKernelAotTest, TritonKernelCallFfiCommandBufferRecording) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                       xla::GetXlaPjrtGpuClient(/*options=*/{}));

  std::string hlo_path =
      tsl::GetDataDependencyFilepath("py/jax/jaxlib/gpu/triton_add_kernel.hlo");
  std::string hlo_string;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo_string));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(hlo_string));

  xla::CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_graph_min_graph_size(1);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
      client->CompileAndLoad(xla::XlaComputation(module->ToProto()),
                             compile_options));

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

  std::vector<float> expected_3(1024, 3.0f);
  // Keep the last result buffer alive so the subsequent execution allocates a
  // new output buffer address and exercises Record(kUpdate) for both inputs
  // and outputs.
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> held_result;
  // Warmup, Create, and Replay with the same input buffers.
  for (int i = 0; i < 3; ++i) {
    held_result.clear();
    ASSERT_OK_AND_ASSIGN(held_result, loaded_executable->Execute(
                                          {{buffer_a.get(), buffer_b.get()}},
                                          /*options=*/{}));
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result,
                         held_result[0][0]->ToLiteral().Await());
    EXPECT_TRUE(xla::LiteralTestUtil::Equal(
        xla::LiteralUtil::CreateR1<float>(expected_3), *result));
  }
  EXPECT_EQ(execute_count_, 1);
  EXPECT_EQ(record_count_, 1);

  // Execute with newly allocated input buffers to exercise Record(kUpdate).
  std::vector<float> data_c(1024, 4.0f);
  std::vector<float> data_d(1024, 5.0f);
  const xla::Literal literal_c = xla::LiteralUtil::CreateR1<float>(data_c);
  const xla::Literal literal_d = xla::LiteralUtil::CreateR1<float>(data_d);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer_c,
      client->BufferFromHostLiteral(literal_c, client->memory_spaces()[0]));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer_d,
      client->BufferFromHostLiteral(literal_d, client->memory_spaces()[0]));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> result_buffers,
      loaded_executable->Execute({{buffer_c.get(), buffer_d.get()}},
                                 /*options=*/{}));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result,
                       result_buffers[0][0]->ToLiteral().Await());

  std::vector<float> expected_9(1024, 9.0f);
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      xla::LiteralUtil::CreateR1<float>(expected_9), *result));
  EXPECT_EQ(execute_count_, 1);
  EXPECT_EQ(record_count_, 2);
}
}  // namespace
}  // namespace jax::JAX_GPU_NAMESPACE
