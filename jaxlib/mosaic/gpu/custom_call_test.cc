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

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/stream_executor/cuda/cuda_platform.h"  // IWYU pragma: keep
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"

namespace {

TEST(CustomCallTest, MosaicGpuUsesCommandBuffers) {
  constexpr absl::string_view kHloModule = R"(
HloModule mosaic_gpu_uses_command_buffers

ENTRY main {
  c0 = f32[] constant(0.0)
  // Use several custom calls to make sure that XLA decides to wrap them inside
  // a command buffer thunk. At the time of writing, the minimum number of
  // thunks necessary to trigger the behavior is 5.
  cc0 = f32[] custom-call(c0),
    custom_call_target="mosaic_gpu_v2", api_version=API_VERSION_TYPED_FFI
  cc1 = f32[] custom-call(cc0),
    custom_call_target="mosaic_gpu_v2", api_version=API_VERSION_TYPED_FFI
  cc2 = f32[] custom-call(cc1),
    custom_call_target="mosaic_gpu_v2", api_version=API_VERSION_TYPED_FFI
  cc3 = f32[] custom-call(cc2),
    custom_call_target="mosaic_gpu_v2", api_version=API_VERSION_TYPED_FFI
  ROOT cc4 = f32[] custom-call(cc3),
    custom_call_target="mosaic_gpu_v2", api_version=API_VERSION_TYPED_FFI
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseAndReturnUnverifiedModule(kHloModule));

  std::string tmp_path = testing::TempDir();
  tsl::setenv("XLA_FLAGS", absl::StrCat("--xla_dump_to=", tmp_path).c_str(),
              /*overwrite=*/true);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::GetXlaPjrtGpuClient(/*options=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      client->CompileAndLoad(xla::XlaComputation(module->ToProto()),
                             /*options=*/{}));

  // Ignore return value. Execution will fail because the custom calls don't
  // wrap any valid Mosaic code, but we only care that the chosen execution
  // plan uses a command buffer thunk.
  (void)executable->Execute(/*argument_handles=*/{}, /*options=*/{});

  // Matching the name exactly is vulnerable to renaming changes, and is not
  // ideal. With that said, this seems like the most reasonable thing to do, and
  // the naming scheme is relatively stable, so this is unlikely to produce
  // churn.
  constexpr absl::string_view kBeforeThunkPassesFilename =
      "module_0001.mosaic_gpu_uses_command_buffers.thunk_sequence.txt";
  constexpr absl::string_view kAfterThunkPassesFilename =
      "module_0001.mosaic_gpu_uses_command_buffers.thunk_sequence_after_thunk_"
      "passes.txt";

  // Ensure that before the thunk passes have run, the first thunk is a custom
  // call thunk as expected.
  std::string before_contents;
  TF_CHECK_OK(tsl::ReadFileToString(
      ::tsl::Env::Default(),
      absl::StrCat(tmp_path, "/", kBeforeThunkPassesFilename),
      &before_contents));
  EXPECT_THAT(before_contents, testing::StartsWith("001: kCustomCall"));

  // Ensure that after the thunk passes have run, the first thunk is a command
  // buffer thunk (which therefore wraps the custom call thunk identified in
  // the previous step).
  std::string after_contents;
  TF_CHECK_OK(tsl::ReadFileToString(
      ::tsl::Env::Default(),
      absl::StrCat(tmp_path, "/", kAfterThunkPassesFilename), &after_contents));

  // There should be only command buffer thunks.
  EXPECT_THAT(after_contents, testing::StartsWith("000: kCommandBuffer"));
}

}  // namespace
