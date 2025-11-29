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
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/scoped_mock_log.h"
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
using ::testing::_;

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

TEST(CustomCallTest, UnloadGPUModule) {
  // Dumped from the following JAX program:
  //
  // ```
  // @functools.partial(
  //     pl.pallas_call,
  //     out_shape=jax.ShapeDtypeStruct((128,), jnp.int32)
  // )
  // def kernel(o_ref):
  //   o_ref[...] = jnp.zeros((128,), dtype=jnp.int32)
  // ```
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule test

    ENTRY main {
      ROOT %result = s32[128]{0} custom-call(), custom_call_target="mosaic_gpu_v2", operand_layout_constraints={}, api_version=API_VERSION_TYPED_FFI, metadata={op_name="jit(wrapped)/pallas_call" source_file="third_party/py/jax/tests/pallas/mosaic_gpu_test.py" source_line=317 source_end_line=317 source_column=4 source_end_column=12}, backend_config={kernel_hash = "\F8\B7\F7\CE\FD\91\AE\D8\A2\8009n4P\0516Z\B1\ED\A9k~\85a\B8\F700D\E9", module = "ML\EFR\01MLIR\00\01\7F\0D\01\03\05\07\09\0B\01\03\0D\03g\0F\11\13\15\17\19\1B\1D\1F!#%')+-/13579;=?ACEGIKMOQSUWY[]_acegikmoqs\03~\04\C6\03-\01\FD\0F\17\1B\17\17\0F\0B\17\13\17\0B\17\17\17\13\0F\13\17\0B\13\0B\17\0B\0B\17\17G\07#e\13\0B\13\1B\0B\0B\13\0B\17\17\0B\0B\0F3\13\13\13\1B\0B\0F\0B\17G\1B\17\17\17\1B\0B\0F\0B\0B\07#\0B\0B\0B;\0B\0B\0B\0B\0B\0B\0B#\0B\0B\0B\1B\0B\0B\1333\13\13S\0F\13S\0F\13\13\13\0B\0F\0B#\0B\13\0B\0B\0BU\1B\0B#\13\0F\13\0F\13\0B\0B\1B\C3\0B\13\13\13\13\13\17\13\13\05\03a\07\09y111\01\D1\13\17\17\17\0F\0B\17\0F\17\1F\17\0F\17\13\13\0B\13\0B\85\0B\13\0B\0B\13\0B\0B\13\0BaS\0BS\0BS\0BS\17\0B\17\0B\17\0B\17\0B\17\0F\0F\0B\0B\0B\0B\0B\0B\0B\13\0F\0B\13\0B\0B\13\0B\17\0F\0B\13\0B\17\0F\17\0B\13\0B\17\0B\13\0B\0B\13\0B\17\0B'\0B\0F\13\0F\17\13\17\0F'\0Fc\17\0F\13\0F\17\0F\0F\13\0F\17\05\03e\09\03Y\0B\09U}am\01\1D\0F\07\0F\0B#\0B;\13/\17\1B\13\1F'\05\0F)\F9-Y5Q\22\02\07\03E\02\0E\1A\1D{}\1D\B2\02\B6\02\03\03\22\02\B2\03\1D^\03b\03\1D\9E\03\A2\03\1D\BF\C1\05u\1DZ\02^\02\03\03\0D\B1\17?\F6\04\09\05w\03\03G\02\02\03\03G\06\02\03\03G\0A\02\03\03\0D\1F\11\05\01\03\03\0D\B7\1Df\02j\02\05y\03\03\A3\1F\05{\03\03\0D\1E\02\05}\05\7F\03\03\0D\C2\02\03\03\0D\C6\02\03\09\CA\02\CE\02\D2\02\D6\02\DA\027\DE\02\AE\03\0F#\01\03\09\00\00\00\00affine_map<(d0) -> (d0)>\00\03\03\0D\B9\05\81\03\03%a\03\05%a/9\05\83\05\85\03\03\0Dc\05\87\1DJ\02N\02\1DB\03F\03\05\89\05\8B\11\01\11#\05\03\11\00\00\00\00\00\00\00\00\03\03)\A7\03\03)\A9\03\03\0D\AB\03\05%\CF/9\0D\05\11\01\01\05\8D\1Dr\02v\02\03\09E\82\02\86\02\8A\02\8E\02\92\02\96\02\9A\02\03\03\E2\02\E6\02\1D\EE\02\F2\02\1D\FA\02\FE\02\1D\0E\03\12\03\03\05uw-y\05\8F\11\05\19\05\91\05\93\1F\03\07Q\81-S\83\85\0D\11\05\95\05\97\03\0D\89U\8B\8D\8F\FD-\91\93\1F\95\1F\05\99\05\9B\0D'\05\9D\05\9F\05\A1\05\A3\03\07Q\99\9B7-\9D\0D\13\05\A5\05\A7\03\05%\A1/9\0D\1D\05\A9\03\03)W#\05\03\11\01\00\00\00\00\00\00\00#\05\03\11\02\00\00\00\00\00\00\00\11\05\02\04\03\03)\AF#\05\05!\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\11\05\05\03\03)\B5#\05\05!\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\11\03\05\11\03\02\04\03\03\0D\BD\11\01\02 \05\AB\1D\C3\13\05\AD\03\07\C7\C9%\CB\CD7\05\AF\11\05\02\02\0D#\05\B1\0D\0Baffine_map<() -> ()>\00\03\05%\D5/\D7\0D\01#\01\03\09\00\00\00\80\03\03\0D\DB\11\05\0D\03\03\0D\DF\11\05A\03\03\E3\E5\05\B3\09S\03\05E\E9\EB\1F#\01\17Y\01\00\00\00\01\00\00\00\01\00\00\00\01\00\00\00\01\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\05\B5\17\15oQ\17\15o_\17\15om\17\15o\EB\17\15o\F9\17\15o\0E\02\17\15o\83\17\15o\9D#llvm.linkage<external>\00#gpu.address_space<workgroup>\00#gpu<dim x>\00#gpu<dim y>\00#gpu<dim z>\00\17\15o\BB\17\15o:\02\17\15oz\02\17\15o\BA\02\11\03\01\05\B7\03\03\0D*\02\11\01\15\03\03\0D2\02\11\01\D0\FF\FF\FF?\03\03\0D:\02\11\01}\03\03e\B6\03\03\03\0DU\03\03K\1F\05\B9\1DR\02\13\05\BBaffine_map<(d0, d1) -> (d0, d1)>\00\05\BD\1Db\02\13\05\BF\05\C1\1Dn\02\13\05\C3\05\C5\1Dz\02\13\05\C7strided<[1], offset: ?>\00#\01\09!\01\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\05\C9#\05\05!\00\00\00\00\00\00\00\80\00\00\00\00\00\00\00\00\05\CB#\05\05!\01\00\00\00\00\00\00\00\80\00\00\00\00\00\00\00\05\CD#\05\05!\01\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\1D\A2\02\A6\02\05\CF\1D\AA\02\AE\02\05\D1\17?\EE\04'\05\D3\1D\BA\02\BE\02\05\D5\17?\EE\04\0D\11\01\1D\11\01\05\05\D7\05\D9\05\DB\05\DD\05\DF\05\E1\05\E3\01\03\EA\02\01\03\1F\05\E5\1D\F6\02\13\05\E7\05\E9\1D\02\03\13\05\EB\03\03K\0A\03\11\05\15\05\ED\1D\16\03\13\05\EF\03\03K\1E\03\11\05\09\1D&\03*\03\05\F1\1D.\03\13\05\F3\1D6\03:\03\05\F5\1D>\03\13\05\F7\05\F9\1DJ\03\13\05\FB\03\03R\03W\05\FD\03\05e\BA\03Z\03\BE\03\05\FF\05\02\02\1Df\03\13\05\06\02\03\03\0Dn\03\11\01\02\10\03\03\0Dv\03\11\05\11\03\05~\03\C2\03E\82\03\05\0A\02#\01\0B)\01\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\01\00\00\00\1D\8A\03\8E\03\05\0E\02\1D\92\03\13\05\12\02\03\03\9A\03c\05\16\02\05\1A\02\1D\A6\03\13\05\1E\02\17\15\0A\03\0F#llvm.tailcallkind<none>\00#arith.overflow<none>\00#nvvm<shfl_kind idx>\00#nvvm.proxy_kind<async.shared>\00#nvvm.shared_space<cta>\00#nvvm.tma_store_mode<tile>\00\01\02\02\03\01\02\04\01\09\17\FF\03\02\04\01~\02\01A\17\FF\03\00\FF\FF\FF\FF\FF\FF\FF\FF\0B;'\03\05\01\05\11\1D\1D\05\05\1D\1D\05\1D\01\05\05\1D\1D\01\15\03\02\04\01;\15\01\01\D1\17\FF\03\02 \0B;\17\FF\05\05\02\04\01V\02!llvm.ptr\00!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\00!llvm.void\00!llvm.array<128 x i8>\00!llvm.ptr<3>\00!llvm.array<0 x i8>\00!llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>\00!gpu.async.token\00\04\B2 \05\01\11\01s\07\03\01\0D%\11\01\7F\07\01;\11\01\87\07\01%\11\01\97\07\03g\CF\05\1D\01\1D\01\15\06\01\03+\03\01\0F\07\01\9F\03\1D\03\03'\07\01'\03\1D\03\07=\02\01\03\1F\11\07\01\A5\03\1F\05\0B\09\11\07\01Y\03\1F\05\0D\09\1B\03\01\1D\03\05\11\07\01[\03\1F\05\0F\11\1B\03\01]\03\05\11\07\01\AD\03\1F\05\13\15\1B\03\01\11\03\05\11\07\01\B3\03\1F\05\17\19\15\06\01\03\15\03\1B\03\03\01!\03\03\03\03\01!\03\03\03\03\01!\03\03\03\03\01=\03\03\03\03\01!\03\03\03\03\01!\03\03\03\03\01\BB\03\01\03\03\0B\11\03\05\17\07\0B\C5\03\1D\03-\0F\07\01_\03\1D\03/?\06\01\09\17\03\03\03\03\1DA\06\01\03\03\03\1D\05\06\01\03\05\03;)\06\01\03\1D\03=\05\06\01\03\05\035\0F\07\01\D3\03\1D\05?A\03\03\01\D9\03\05\03\03\01\11\03\05\05\06\01\03\05\037\03\03\01\11\03\05\17\07\01A\03\1D\03K\0F\07\01C\03\1D\03M\1D\05\01'\05IO\05\06\01\03\05\039\03\03\01\11\03\05\17\07\01A\03\1D\03S\0F\07\01C\03\1D\03U\1D\05\01'\05QW\03\03\01\DD\03\05\03\03\01]\03\05\03\03\01\11\03\05\17\07\01A\03\1D\03]\0F\07\01C\03\1D\03_\1D\05\01'\05[aC\05\01\E1\111CEGMUY_'\07\0B'\03#\03/E\17\01\E7\03+\11\05\1F!#%')+\05\03N\03\86\06\19\03\ED\03\EF\03\F1\03\F3\03\F5\03\F7\03\F9\03\FB\03\0E\02\03\12\02\03\16\02\03\1A\02\15\06\0B\03\1D\03c\0F\07\0B_\03\1D\03\7F+\02\01\03\0D\03\03\01+\03\03-\06\01\03\19\05\83\85I\00\01K\00\01/\02\01\03\07\0B\03\01\17\03\03\05\06\01\03\01\03\8B\0D\03\01\17\03\03\05\06\01\03\01\03\8F\0B\03\01\19\03\03\05\06\01\03\01\03\93\07\07\01\05\03\01\05\95\91\09\07\01\05\03\01\05\8D\97\0D\03\01\19\03\03\05\06\01\03\01\03\9B\07\07\01\05\03\01\05\91\9D\0B\03\01\1B\03\03\05\06\01\03\01\03\A1\07\07\01\05\03\01\05\A3\9F\09\07\01\05\03\01\05\99\A5\0D\03\01\1B\03\03\05\06\01\03\01\03\A9\07\07\01\05\03\01\05\9F\AB\03\03\01&\02\03\01\13\06\01\03\01\05\A7\AF\03\03\01.\02\03\01\03\03\01I\03\01\03\03\016\02\03\01M\07\01>\02\03\01\09\B3\B1\B5\B7\03\03\01B\02\03\01\1F\06\01\03\01\05\B9\BB\03\03\01I\03\01!\07\01F\02\03\07\05\BD\BF1\06\01\03\07\05\C1\89/\02\01\03\07+\02M\03\0D\03\03M+\03\03-\06M\03\1B\05\C7\C9\03\03\0F\1D\03\05\03\03\0F\1D\03\05\03\03\0F\11\03\05\07\07\0F\05\03\05\05\CF\D1\09\07\0F\05\03\05\05\D3\CD\03\03\0F\1D\03\05\09\07\0F\05\03\05\05\D5\D7\03\03\0F\11\03\05\09\07\0F\05\03\05\05\CD\DB\03\03\0F\1D\03\05\03\03\0F\1D\03\05\03\03\0F\11\03\05\07\07\0F\05\03\05\05\E1\E3\09\07\0F\05\03\05\05\E5\DF\03\03\0F\1D\03\05\09\07\0F\05\03\05\05\E7\E9\03\03#\11\03\05#\06#\03\05\05\EB\ED\05\06g\03\03\03\EF3\07gi\03\09\05\CB\F1\03\03\9E\02I\03\01\0B\03\03\17\03\03\05\06\03\03\01\03\F7\0D\03\03\17\03\03\05\06\03\03\01\03\FB\0B\03\03\19\03\03\05\06\03\03\01\03\FF\07\07\03\05\03\01\05\02\02\FD\09\07\03\05\03\01\05\F9\06\02\0D\03\03\19\03\03\05\06\03\03\01\03\0E\02\07\07\03\05\03\01\05\FD\12\02\0B\03\03\1B\03\03\05\06\03\03\01\03\1A\02\07\07\03\05\03\01\05\1E\02\16\02\09\07\03\05\03\01\05\0A\02\22\02\0D\03\03\1B\03\03\05\06\03\03\01\03*\02\07\07\03\05\03\01\05\16\02.\02\03\03\031\03\01\13\06\03\03\01\05&\026\02\03\03\033\03\01\09\07\03\05\03\01\05:\02>\02\19\07\035\03!\03B\025\07\03k\03\09\03\F3\0B\03\03\17\03\03\03\03\03=\03\03\1F\06\03\03\03\05N\02R\02\03\03\03!\03\03\07\07\03\05\03\03\05V\02Z\02\03\03\03+\03\03\09\07\03\05\03\03\05^\02b\02O\06\03\03\0F\05J\02f\02Q\06\03\03\0F\03\F55\07\03k\03\09\03\F3\0B\03\03\17\03\03\03\03\03=\03\03\1F\06\03\03\03\05v\02z\02\03\03\03!\03\03\07\07\03\05\03\03\05~\02\82\02\03\03\03+\03\03\09\07\03\05\03\03\05\86\02\8A\02S\04\03\07n\02r\02\8E\02\0B\03\03\17\03\03\05\06\03\03\01\03\92\02\0D\03\03\17\03\03\05\06\03\03\01\03\9A\02\0B\03\03\19\03\03\05\06\03\03\01\03\A2\02\07\07\03\05\03\01\05\A6\02\9E\02\09\07\03\05\03\01\05\96\02\AA\02\0D\03\03\19\03\03\05\06\03\03\01\03\B2\02\07\07\03\05\03\01\05\9E\02\B6\02\0B\03\03\1B\03\03\05\06\03\03\01\03\BE\02\07\07\03\05\03\01\05\C2\02\BA\02\09\07\03\05\03\01\05\AE\02\C6\02\0D\03\03\1B\03\03\05\06\03\03\01\03\CE\02\07\07\03\05\03\01\05\BA\02\D2\02\03\03\031\03\01\13\06\03\03\01\05\CA\02\DA\02\03\03\033\03\01\09\07\03\05\03\01\05\DE\02\E2\02\19\07\035\03!\03\E6\02\03\03m\11\03\05\09\07m\05\03\05\05\EB\EE\02\03\03#\11\03\05#\06#\03\05\05\F2\02\F6\02\03\03o\1D\03\05!\07o\06\03\03\07\05\EB\FE\02\03\03q\11\03\05!\07q\1A\03\03\07\05\F2\02\06\031\06\22\03\03\07\05\02\03\0A\03U\062\03\03\01\03\0E\03\05\06O\03\03\03\12\03W\15ON\03\03\16\03\0B\03\01\059\00\AA\03\03\01\059\00O\03\03\0F\11\03\05\09\07\0F\05\03\05\05\DF\1A\03Y\01\07V\03\0B\03\07\17\03\03\05\06\07\03\01\03\22\03\0D\03\07\17\03\03\05\06\07\03\01\03*\03\0B\03\07\19\03\03\05\06\07\03\01\032\03\07\07\07\05\03\01\056\03.\03\09\07\07\05\03\01\05&\03:\03\0D\03\07\19\03\03\05\06\07\03\01\03B\03\07\07\07\05\03\01\05.\03F\03\0B\03\07\1B\03\03\05\06\07\03\01\03N\03\07\07\07\05\03\01\05R\03J\03\09\07\07\05\03\01\05>\03V\03\0D\03\07\1B\03\03\05\06\07\03\01\03^\03\07\07\07\05\03\01\05J\03b\03\03\03\071\03\01\13\06\07\03\01\05Z\03j\03\03\03\073\03\01\09\07\07\05\03\01\05n\03r\03\19\07\075\03!\03v\03\03\03#\1D\03\05\03\03#\11\03\05#\06#\03\05\05~\03\82\03\05\06\0B\03\03\03\86\033\07\0Bi\03\09\05\CB\8A\03\03\03\0B+\03\03\03\03\0Bj\03\03\01\05\06\0B\03\01\03\92\03\15\06\0B\03)\03\8E\037\07\0BY\03%\03\9E\037\07\0B[\03\05\03\9E\03\03\03\0Br\03\03\05[\06\0B\03\05\05\A6\03\AA\03]\06\0B\03\05\03\A2\03_\06\0B\03\05\05\B2\03\AE\03)\06\0B\03%\03\B6\03a\05\0Bz\03\09\81\BA\03\9A\03\C3c\00\86\03e\01\09\96\03\0B\03\09\17\03\03\05\06\09\03\01\03\BE\03\0D\03\09\17\03\03\05\06\09\03\01\03\C6\03\0B\03\09\19\03\03\05\06\09\03\01\03\CE\03\07\07\09\05\03\01\05\D2\03\CA\03\09\07\09\05\03\01\05\C2\03\D6\03\0D\03\09\19\03\03\05\06\09\03\01\03\DE\03\07\07\09\05\03\01\05\CA\03\E2\03\0B\03\09\1B\03\03\05\06\09\03\01\03\EA\03\07\07\09\05\03\01\05\EE\03\E6\03\09\07\09\05\03\01\05\DA\03\F2\03\0D\03\09\1B\03\03\05\06\09\03\01\03\FA\03\07\07\09\05\03\01\05\E6\03\FE\03\03\03\091\03\01\13\06\09\03\01\05\F6\03\06\04\03\03\093\03\01\09\07\09\05\03\01\05\0A\04\0E\04\19\07\095\03!\03\12\04g\00\01G\00\01\06\03\01\05\01\00\E6\1E\22\02%'\0D\1B\1D\0B\19\1B\0D\0D\0B\0D+-\09\0B\07\09\07\09\09\0B\1D\1F#\05\19%\17\0B\0D#%\1F\1B\1F\17\19\09\0B\0B\0D\17\19\1D/\0F-\15%'\13%-\19\1B\1F\11\19\17\11\1F\19\0F5\0B3\1D\15\15)g'\13\13\15\05\0D\1F=AY\13\1D\13##\19\1B#\19\1F\193\19\17\15QA!#\15%-\1F\17!\195\1D\15\15\19\17\19\17'!\19G\19#'\1D\1D\17\17#\1F\0F\0B\0D\09\0B%\11builtin\00stable_mosaic_gpu\00llvm\00gpu\00arith\00nvvm\00module\00arith.constant\00arith.index_cast\00arith.muli\00arith.addi\00gpu.thread_id\00gpu.block_dim\00llvm.getelementptr\00llvm.insertvalue\00arith.shrui\00builtin.unrealized_conversion_cast\00llvm.alloca\00llvm.inline_asm\00llvm.mlir.constant\00llvm.store\00arith.remui\00arith.cmpi\00arith.remsi\00func.func\00llvm.load\00llvm.inttoptr\00gpu.dynamic_shared_memory\00memref.view\00nvvm.elect.sync\00arith.andi\00memref.subview\00memref.collapse_shape\00llvm.extractvalue\00scf.yield\00llvm.mlir.global\00llvm.mlir.undef\00memref.extract_strided_metadata\00memref.extract_aligned_pointer_as_index\00func.call\00gpu.launch\00func.return\00nvvm.fence.mbarrier.init\00gpu.barrier\00nvvm.shfl.sync\00vector.load\00vector.broadcast\00vector.store\00arith.extui\00scf.index_switch\00nvvm.fence.proxy\00llvm.mul\00llvm.ptrtoint\00llvm.add\00nvvm.cp.async.bulk.tensor.global.shared.cta\00nvvm.cp.async.bulk.commit.group\00nvvm.cp.async.bulk.wait_group\00gpu.terminator\00value\00-\00elem_type\00position\00sym_name\00rawConstantIndices\00third_party/py/jax/tests/pallas/mosaic_gpu_test.py\00operandSegmentSizes\00dimension\00predicate\00function_type\00mosaic_gpu_init_tma_desc\00kind\00stable_mosaic_gpu.version\00kernel\00pallas_call\00sym_visibility\00private\00addr_space\00global_type\00linkage\00global_scratch\00unnamed_addr\00visibility_\00llvm.emit_c_interface\00kernel_mosaic_gpu\00ordering\00copy_smem_to_gmem:\00copy_smem_to_gmem\00alignment\00mosaic_gpu_smem_alloc\00callee\00workgroup_attributions\00overflowFlags\00run_scoped:\00run_scoped\00scan:\00scan\00rem:\00rem\00jaxpr_call:\00jaxpr_call\00static_offsets\00static_sizes\00static_strides\00broadcast_in_dim:\00broadcast_in_dim\00swap:\00swap\00asm_string\00bar.sync $0, 128;\00constraints\00r\00has_side_effects\00tail_call_kind\00reassociation\00add:\00add\00ge:\00ge\00lt:\00lt\00and:\00and\00convert_element_type:\00convert_element_type\00cond:\00cond\00cases\00space\00commit_smem:\00commit_smem\00mode\00commit_group:\00commit_group\00group\00wait_smem_to_gmem:\00wait_smem_to_gmem\00", use_custom_barrier = false}
    }
  )hlo";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseAndReturnUnverifiedModule(kHloModule));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::GetXlaPjrtGpuClient(/*options=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      client->CompileAndLoad(xla::XlaComputation(module->ToProto()),
                             /*options=*/{}));

  absl::SetVLogLevel("custom_call", 5);
  std::vector<xla::PjRtBuffer*> no_buffers;
  {
    absl::ScopedMockLog log;
    EXPECT_CALL(log,
                Log(absl::LogSeverity::kInfo, _,
                    "Successfully compiled and initialized Mosaic GPU kernel"))
        .Times(1);
    log.StartCapturingLogs();
    EXPECT_THAT(executable->Execute({no_buffers}, {}), IsOk());
  }

  {
    // The second execution the compilation should be cached.
    absl::ScopedMockLog log;
    EXPECT_CALL(log,
                Log(absl::LogSeverity::kInfo, _,
                    "Successfully compiled and initialized Mosaic GPU kernel"))
        .Times(0);
    log.StartCapturingLogs();
    EXPECT_THAT(executable->Execute({no_buffers}, {}), IsOk());
  }

  {
    // GPU module should be unloaded when the executable is destroyed.
    absl::ScopedMockLog log;
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                         "Successfully unloaded GPU module"))
        .Times(1);
    log.StartCapturingLogs();
    executable.reset();
  }
}

}  // namespace
