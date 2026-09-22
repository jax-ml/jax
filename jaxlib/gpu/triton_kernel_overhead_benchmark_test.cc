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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "jaxlib/gpu/triton.pb.h"
#include "jaxlib/gpu/triton_kernels.h"
#include "jaxlib/gpu/triton_utils.h"
#include "jaxlib/gpu/vendor.h"
#include "third_party/rust/libz_rs_sys/google/zlib.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/stream_executor/cuda/cuda_platform.h"  // IWYU pragma: keep
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"

namespace jax::JAX_GPU_NAMESPACE {
namespace {

absl::StatusOr<std::string> ZlibCompress(std::string_view uncompressed) {
  uLongf dest_len = compressBound(uncompressed.size());
  std::string data(dest_len, '\0');
  int ret = compress(reinterpret_cast<Bytef*>(data.data()), &dest_len,
                     reinterpret_cast<const Bytef*>(uncompressed.data()),
                     uncompressed.size());
  if (ret != Z_OK) {
    return absl::InternalError("Failed to compress data with zlib.");
  }
  data.resize(dest_len);
  return data;
}

// Unescapes string literals from MLIR/HLO format (where arbitrary bytes are
// formatted as \XX with 2 hex digits).
std::string UnescapeMlirString(std::string_view s) {
  std::string result;
  result.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 2 < s.size() &&
        absl::ascii_isxdigit(s[i + 1]) && absl::ascii_isxdigit(s[i + 2])) {
      int val;
      if (absl::SimpleHexAtoi(s.substr(i + 1, 2), &val)) {
        result.push_back(static_cast<char>(val));
        i += 2;
        continue;
      }
    }
    if (s[i] == '\\' && i + 1 < s.size()) {
      char c = s[i + 1];
      if (c == '\\') { result.push_back('\\'); ++i; continue; }
      if (c == '"') { result.push_back('"'); ++i; continue; }
      if (c == 'n') { result.push_back('\n'); ++i; continue; }
      if (c == 't') { result.push_back('\t'); ++i; continue; }
      if (c == 'r') { result.push_back('\r'); ++i; continue; }
    }
    result.push_back(s[i]);
  }
  return result;
}

// Escapes binary bytes as \XX hexadecimal escape sequences for MLIR/HLO parser.
std::string EscapeMlirString(std::string_view s) {
  std::string result;
  result.reserve(s.size() * 3);
  for (unsigned char c : s) {
    absl::StrAppendFormat(&result, "\\%02X", c);
  }
  return result;
}

void RegisterTritonHandlerOnce() {
  static absl::once_flag flag;
  absl::call_once(flag, [] {
    XLA_FFI_Handler_Bundle bundle = {
        /*instantiate=*/kTritonKernelCallFfiInstantiate,
        /*prepare=*/nullptr,
        /*initialize=*/kTritonKernelCallFfiInitialize,
        /*execute=*/kTritonKernelCallFfi,
    };
    CHECK(xla::ffi::Ffi::RegisterStaticHandler(
              xla::ffi::GetXlaFfiApi(), "triton_kernel_call_ffi", "CUDA",
              bundle) == nullptr)
        << "RegisterStaticHandler failed";
  });
}

class TritonKernelBenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterTritonHandlerOnce(); }
};

TEST_F(TritonKernelBenchmarkTest, Benchmark1000KernelsInitialization) {
  std::string hlo_path =
      tsl::GetDataDependencyFilepath("py/jax/jaxlib/gpu/triton_add_kernel.hlo");
  std::string hlo_string;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo_string));

  constexpr absl::string_view kOpaquePrefix = "opaque = \"";
  size_t start = hlo_string.find(kOpaquePrefix);
  ASSERT_NE(start, std::string::npos);
  start += kOpaquePrefix.size();
  size_t end = hlo_string.find("\"}", start);
  ASSERT_NE(end, std::string::npos);

  std::string unescaped_opaque =
      UnescapeMlirString(hlo_string.substr(start, end - start));

  ASSERT_OK_AND_ASSIGN(std::string serialized_base,
                       ZlibUncompress(unescaped_opaque));
  jax_triton::TritonAnyKernelCall base_proto;
  ASSERT_TRUE(base_proto.ParseFromString(serialized_base));

  // Generate 1000 kernel variations. Each kernel has a unique name and metadata
  // so that its opaque string is distinct, bypassing the cache in
  // GetOrCreateKernelCall.
  constexpr int kNumKernels = 1000;
  std::vector<std::string> escaped_opaques;
  escaped_opaques.reserve(kNumKernels);

  for (int i = 0; i < kNumKernels; ++i) {
    jax_triton::TritonAnyKernelCall proto = base_proto;
    proto.set_name(absl::StrFormat("add_kernel_%d", i));
    proto.set_metadata(absl::StrFormat("instance_%d", i));

    std::string proto_str = proto.SerializeAsString();
    ASSERT_OK_AND_ASSIGN(std::string compressed, ZlibCompress(proto_str));
    escaped_opaques.push_back(EscapeMlirString(compressed));
  }

  // Chain the kernels in an HLO module: c_i = custom-call(c_{i-1}, p1)
  std::string generated_hlo;
  absl::StrAppend(
      &generated_hlo,
      "HloModule benchmark_1000_kernels, "
      "entry_computation_layout={(f32[1024]{0}, f32[1024]{0})->f32[1024]{0}}\n"
      "ENTRY main {\n"
      "  p0 = f32[1024]{0} parameter(0)\n"
      "  p1 = f32[1024]{0} parameter(1)\n");

  std::string prev = "p0";
  for (int i = 0; i < kNumKernels; ++i) {
    std::string curr = absl::StrFormat("c_%d", i);
    std::string root_prefix = (i == kNumKernels - 1) ? "ROOT " : "";
    absl::StrAppendFormat(
        &generated_hlo,
        "  %s%s = f32[1024]{0} custom-call(%s, p1), "
        "custom_call_target=\"triton_kernel_call_ffi\", "
        "operand_layout_constraints={f32[1024]{0}, f32[1024]{0}}, "
        "api_version=API_VERSION_TYPED_FFI, "
        "backend_config={opaque = \"%s\"}\n",
        root_prefix, curr, prev, escaped_opaques[i]);
    prev = curr;
  }
  absl::StrAppend(&generated_hlo, "}\n");

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(generated_hlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                       xla::GetXlaPjrtGpuClient(/*options=*/{}));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      client->CompileAndLoad(xla::XlaComputation(module->ToProto()),
                             xla::CompileOptions()));

  // Input buffers.
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

  // Expected result
  std::vector<float> expected(1024, 1.0f + kNumKernels * 2.0f);
  const xla::Literal expected_literal =
      xla::LiteralUtil::CreateR1<float>(expected);

  auto run_module = [&]() -> absl::StatusOr<absl::Duration> {
    absl::Time start = absl::Now();
    ASSIGN_OR_RETURN(
        std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> result_buf,
        executable->Execute({{buffer_a.get(), buffer_b.get()}},
                            /*options=*/{}));
    ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> result,
                        result_buf[0][0]->ToLiteral().Await());
    absl::Duration duration = absl::Now() - start;
    EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, *result));
    return duration;
  };

  // Warm-up run of the entire HLO module -> 1000 kernels.
  ASSERT_OK_AND_ASSIGN(absl::Duration cold_duration, run_module());
  LOG(INFO) << "Warm-up run (cold cache) of " << kNumKernels
            << " kernels took: " << cold_duration << " ("
            << absl::ToDoubleMilliseconds(cold_duration) / kNumKernels
            << " ms/kernel)";

  // CUDA Graph capture run.
  // If CUDA graphs are enabled, this second run is also slow.
  ASSERT_OK_AND_ASSIGN(absl::Duration graph_capture_duration, run_module());
  LOG(INFO) << "CUDA graph (if enabled) capture run of " << kNumKernels
            << " kernels took: " << graph_capture_duration << " ("
            << absl::ToDoubleMilliseconds(graph_capture_duration) / kNumKernels
            << " ms/kernel)";

  // More warm executions of the entire HLO module -> Nx1000 kernels.
  constexpr int kWarmIterations = 10;
  absl::Duration total_warm_duration = absl::ZeroDuration();
  for (int iter = 0; iter < kWarmIterations; ++iter) {
    ASSERT_OK_AND_ASSIGN(absl::Duration duration, run_module());
    LOG(INFO) << "Run " << iter << ". Module took: " << duration;
    total_warm_duration += duration;
  }
  absl::Duration avg_warm_duration = total_warm_duration / kWarmIterations;
  LOG(INFO) << "Average steady-state execution (" << kWarmIterations
            << " iterations) of " << kNumKernels
            << " kernels took: " << avg_warm_duration << " ("
            << absl::ToDoubleMilliseconds(avg_warm_duration) / kNumKernels
            << " ms/kernel)";
}

}  // namespace
}  // namespace jax::JAX_GPU_NAMESPACE
