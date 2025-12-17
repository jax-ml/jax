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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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

using ::absl_testing::IsOk;

absl::Status ExecuteSync(xla::PjRtLoadedExecutable* executable) {
  std::vector<xla::PjRtBuffer*> no_buffers;
  TF_ASSIGN_OR_RETURN(auto result,
                      executable->Execute({no_buffers}, /*options=*/{}));
  return result[0][0]->GetReadyFuture().Await();
}

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
  ExecuteSync(executable.get()).IgnoreError();

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

TEST(CustomCallTest, LegacyCustomCall) {
  absl::string_view hlo_string = R"hlo(
    HloModule test

    ENTRY main {
      ROOT result = s32[] custom-call(), custom_call_target="mosaic_gpu", api_version=API_VERSION_STATUS_RETURNING, backend_config="\220\307\037$\222=c\235\344\250\025\261Y\233.\002\264\260\013\026\305Ol\324\355\315dA-\311\3277\"builtin.module\"() <{sym_name = \"kernel\"}> ({\n  \"stable_mosaic_gpu.func.func\"() ({\n  }) {function_type = (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> (), sym_name = \"mosaic_gpu_init_tma_desc\", sym_visibility = \"private\"} : () -> ()\n  \"stable_mosaic_gpu.llvm.mlir.global\"() ({\n  }) {addr_space = 4 : i32, global_type = !llvm.array<0 x i8>, linkage = #llvm.linkage<external>, sym_name = \"global_scratch\", unnamed_addr = 0 : i64, visibility_ = 0 : i64} : () -> ()\n  \"stable_mosaic_gpu.func.func\"() ({\n  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):\n    %0 = \"stable_mosaic_gpu.arith.constant\"() {value = 42 : i32} : () -> i32\n    %1 = \"stable_mosaic_gpu.arith.constant\"() {value = 0 : i32} : () -> i32\n    %2 = \"stable_mosaic_gpu.arith.constant\"() {value = 128 : index} : () -> index\n    %3 = \"stable_mosaic_gpu.arith.constant\"() {value = 1 : index} : () -> index\n    %4 = \"stable_mosaic_gpu.llvm.mlir.constant\"() {value = 0 : i64} : () -> i64\n    %5 = \"stable_mosaic_gpu.llvm.mlir.undef\"() : () -> !llvm.struct<(ptr, ptr, i64)>\n    %6 = \"stable_mosaic_gpu.builtin.unrealized_conversion_cast\"(%arg0) : (!llvm.ptr) -> !gpu.async.token\n    %7 = \"stable_mosaic_gpu.llvm.load\"(%arg1) {ordering = 0 : i64} : (!llvm.ptr) -> !llvm.ptr\n    %8 = \"stable_mosaic_gpu.llvm.insertvalue\"(%5, %7) {position = array<i64: 0>} : (!llvm.struct<(ptr, ptr, i64)>, !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64)>\n    %9 = \"stable_mosaic_gpu.llvm.insertvalue\"(%8, %7) {position = array<i64: 1>} : (!llvm.struct<(ptr, ptr, i64)>, !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64)>\n    %10 = \"stable_mosaic_gpu.llvm.insertvalue\"(%9, %4) {position = array<i64: 2>} : (!llvm.struct<(ptr, ptr, i64)>, i64) -> !llvm.struct<(ptr, ptr, i64)>\n    %11 = \"stable_mosaic_gpu.builtin.unrealized_conversion_cast\"(%10) : (!llvm.struct<(ptr, ptr, i64)>) -> memref<i32>\n    %12 = \"stable_mosaic_gpu.gpu.launch\"(%6, %3, %3, %3, %2, %3, %3, %1) ({\n    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index):\n      %13 = \"stable_mosaic_gpu.nvvm.elect.sync\"() : () -> i1\n      %14 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim x>} : () -> index\n      %15 = \"stable_mosaic_gpu.arith.index_cast\"(%14) : (index) -> i32\n      %16 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim x>} : () -> index\n      %17 = \"stable_mosaic_gpu.arith.index_cast\"(%16) : (index) -> i32\n      %18 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim y>} : () -> index\n      %19 = \"stable_mosaic_gpu.arith.index_cast\"(%18) : (index) -> i32\n      %20 = \"stable_mosaic_gpu.arith.muli\"(%19, %17) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %21 = \"stable_mosaic_gpu.arith.addi\"(%15, %20) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %22 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim y>} : () -> index\n      %23 = \"stable_mosaic_gpu.arith.index_cast\"(%22) : (index) -> i32\n      %24 = \"stable_mosaic_gpu.arith.muli\"(%17, %23) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %25 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim z>} : () -> index\n      %26 = \"stable_mosaic_gpu.arith.index_cast\"(%25) : (index) -> i32\n      %27 = \"stable_mosaic_gpu.arith.muli\"(%26, %24) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %28 = \"stable_mosaic_gpu.arith.addi\"(%21, %27) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %29 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim z>} : () -> index\n      %30 = \"stable_mosaic_gpu.arith.index_cast\"(%29) : (index) -> i32\n      %31 = \"stable_mosaic_gpu.arith.muli\"(%24, %30) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %32 = \"stable_mosaic_gpu.arith.constant\"() {value = 5 : i32} : () -> i32\n      %33 = \"stable_mosaic_gpu.arith.shrui\"(%28, %32) : (i32, i32) -> i32\n      %34 = \"stable_mosaic_gpu.arith.constant\"() {value = -1 : i32} : () -> i32\n      %35 = \"stable_mosaic_gpu.arith.constant\"() {value = 0 : i32} : () -> i32\n      %36 = \"stable_mosaic_gpu.arith.constant\"() {value = 31 : i32} : () -> i32\n      %37 = \"stable_mosaic_gpu.nvvm.shfl.sync\"(%34, %33, %35, %36) {kind = #nvvm<shfl_kind idx>} : (i32, i32, i32, i32) -> i32\n      %38 = \"stable_mosaic_gpu.arith.constant\"() {value = 0 : i32} : () -> i32\n      %39 = \"stable_mosaic_gpu.arith.cmpi\"(%37, %38) {predicate = 0 : i64} : (i32, i32) -> i1\n      %40 = \"stable_mosaic_gpu.arith.andi\"(%39, %13) : (i1, i1) -> i1\n      %41 = \"stable_mosaic_gpu.nvvm.elect.sync\"() : () -> i1\n      %42 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim x>} : () -> index\n      %43 = \"stable_mosaic_gpu.arith.index_cast\"(%42) : (index) -> i32\n      %44 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim x>} : () -> index\n      %45 = \"stable_mosaic_gpu.arith.index_cast\"(%44) : (index) -> i32\n      %46 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim y>} : () -> index\n      %47 = \"stable_mosaic_gpu.arith.index_cast\"(%46) : (index) -> i32\n      %48 = \"stable_mosaic_gpu.arith.muli\"(%47, %45) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %49 = \"stable_mosaic_gpu.arith.addi\"(%43, %48) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %50 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim y>} : () -> index\n      %51 = \"stable_mosaic_gpu.arith.index_cast\"(%50) : (index) -> i32\n      %52 = \"stable_mosaic_gpu.arith.muli\"(%45, %51) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %53 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim z>} : () -> index\n      %54 = \"stable_mosaic_gpu.arith.index_cast\"(%53) : (index) -> i32\n      %55 = \"stable_mosaic_gpu.arith.muli\"(%54, %52) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %56 = \"stable_mosaic_gpu.arith.addi\"(%49, %55) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %57 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim z>} : () -> index\n      %58 = \"stable_mosaic_gpu.arith.index_cast\"(%57) : (index) -> i32\n      %59 = \"stable_mosaic_gpu.arith.muli\"(%52, %58) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %60 = \"stable_mosaic_gpu.arith.constant\"() {value = 5 : i32} : () -> i32\n      %61 = \"stable_mosaic_gpu.arith.shrui\"(%56, %60) : (i32, i32) -> i32\n      %62 = \"stable_mosaic_gpu.arith.constant\"() {value = -1 : i32} : () -> i32\n      %63 = \"stable_mosaic_gpu.arith.constant\"() {value = 0 : i32} : () -> i32\n      %64 = \"stable_mosaic_gpu.arith.constant\"() {value = 31 : i32} : () -> i32\n      %65 = \"stable_mosaic_gpu.nvvm.shfl.sync\"(%62, %61, %63, %64) {kind = #nvvm<shfl_kind idx>} : (i32, i32, i32, i32) -> i32\n      %66 = \"stable_mosaic_gpu.arith.constant\"() {value = 4 : i32} : () -> i32\n      %67 = \"stable_mosaic_gpu.arith.remui\"(%65, %66) : (i32, i32) -> i32\n      %68 = \"stable_mosaic_gpu.arith.constant\"() {value = 0 : i32} : () -> i32\n      %69 = \"stable_mosaic_gpu.arith.cmpi\"(%67, %68) {predicate = 0 : i64} : (i32, i32) -> i1\n      %70 = \"stable_mosaic_gpu.arith.andi\"(%69, %41) : (i1, i1) -> i1\n      %71 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim x>} : () -> index\n      %72 = \"stable_mosaic_gpu.arith.index_cast\"(%71) : (index) -> i32\n      %73 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim x>} : () -> index\n      %74 = \"stable_mosaic_gpu.arith.index_cast\"(%73) : (index) -> i32\n      %75 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim y>} : () -> index\n      %76 = \"stable_mosaic_gpu.arith.index_cast\"(%75) : (index) -> i32\n      %77 = \"stable_mosaic_gpu.arith.muli\"(%76, %74) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %78 = \"stable_mosaic_gpu.arith.addi\"(%72, %77) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %79 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim y>} : () -> index\n      %80 = \"stable_mosaic_gpu.arith.index_cast\"(%79) : (index) -> i32\n      %81 = \"stable_mosaic_gpu.arith.muli\"(%74, %80) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %82 = \"stable_mosaic_gpu.gpu.thread_id\"() {dimension = #gpu<dim z>} : () -> index\n      %83 = \"stable_mosaic_gpu.arith.index_cast\"(%82) : (index) -> i32\n      %84 = \"stable_mosaic_gpu.arith.muli\"(%83, %81) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %85 = \"stable_mosaic_gpu.arith.addi\"(%78, %84) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %86 = \"stable_mosaic_gpu.gpu.block_dim\"() {dimension = #gpu<dim z>} : () -> index\n      %87 = \"stable_mosaic_gpu.arith.index_cast\"(%86) : (index) -> i32\n      %88 = \"stable_mosaic_gpu.arith.muli\"(%81, %87) {overflowFlags = #arith.overflow<none>} : (i32, i32) -> i32\n      %89 = \"stable_mosaic_gpu.arith.constant\"() {value = 5 : i32} : () -> i32\n      %90 = \"stable_mosaic_gpu.arith.shrui\"(%85, %89) : (i32, i32) -> i32\n      %91 = \"stable_mosaic_gpu.arith.constant\"() {value = 0 : i32} : () -> i32\n      %92 = \"stable_mosaic_gpu.arith.cmpi\"(%90, %91) {predicate = 0 : i64} : (i32, i32) -> i1\n      %93 = \"stable_mosaic_gpu.gpu.dynamic_shared_memory\"() : () -> memref<?xi8, #gpu.address_space<workgroup>>\n      %94 = \"stable_mosaic_gpu.arith.index_cast\"(%1) : (i32) -> index\n      %95 = \"stable_mosaic_gpu.memref.view\"(%93, %94) : (memref<?xi8, #gpu.address_space<workgroup>>, index) -> memref<0xi8, #gpu.address_space<workgroup>>\n      %96 = \"stable_mosaic_gpu.builtin.unrealized_conversion_cast\"(%95) {transforms = []} : (memref<0xi8, #gpu.address_space<workgroup>>) -> memref<0xi8, #gpu.address_space<workgroup>>\n      \"stable_mosaic_gpu.nvvm.fence.mbarrier.init\"() : () -> ()\n      \"stable_mosaic_gpu.gpu.barrier\"() : () -> ()\n      \"stable_mosaic_gpu.memref.store\"(%0, %11) : (i32, memref<i32>) -> ()\n      \"stable_mosaic_gpu.gpu.terminator\"() : () -> ()\n    }) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1>, workgroup_attributions = 0 : i64} : (!gpu.async.token, index, index, index, index, index, index, i32) -> !gpu.async.token\n    \"stable_mosaic_gpu.func.return\"() : () -> ()\n  }) {function_type = (!llvm.ptr, !llvm.ptr) -> (), llvm.emit_c_interface, sym_name = \"kernel_mosaic_gpu\"} : () -> ()\n}) {stable_mosaic_gpu.version = 6 : i64} : () -> ()\n"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module,
                       xla::ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                       xla::GetXlaPjrtGpuClient(/*options=*/{}));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      client->CompileAndLoad(xla::XlaComputation(module->ToProto()),
                             /*options=*/{}));
  EXPECT_THAT(ExecuteSync(executable.get()), IsOk());
}

}  // namespace
