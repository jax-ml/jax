/* Copyright 2026 The JAX Authors

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

#include "jaxlib/pathways_profiler.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/struct.pb.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace jax {
namespace {

using ::google::protobuf::Struct;

Struct ParseJson(const std::string& json) {
  Struct parsed;
  CHECK_OK(tsl::protobuf::util::JsonStringToMessage(json, &parsed)) << json;
  return parsed;
}

void ExpectJsonEq(const std::string& actual, const std::string& expected) {
  EXPECT_TRUE(tsl::protobuf::util::MessageDifferencer::Equals(
      ParseJson(actual), ParseJson(expected)))
      << "actual: " << actual << "\nexpected: " << expected;
}

TEST(BuildPathwaysProfileRequestTest, ReturnsNulloptWithoutLogDir) {
  tensorflow::ProfileOptions options;
  options.set_session_id("my_session");

  absl::StatusOr<std::optional<std::string>> request =
      BuildPathwaysProfileRequest(options);

  ASSERT_TRUE(request.ok()) << request.status();
  EXPECT_EQ(*request, std::nullopt);
}

TEST(BuildPathwaysProfileRequestTest, BuildsRequestFromOptions) {
  tensorflow::ProfileOptions options;
  options.set_repository_path("gs://bucket/logs");
  options.set_session_id("my_session");
  options.set_host_tracer_level(0);
  options.set_python_tracer_level(0);
  options.set_duration_ms(2500);
  auto& config = *options.mutable_advanced_configuration();
  config["pathways_max_num_hosts"].set_int64_value(4);
  config["tpu_trace_mode"].set_string_value("TRACE_ONLY_HOST");
  config["enable_fw_throttle_event"].set_bool_value(true);
  config["num_buffers"].set_int64_value(2);

  absl::StatusOr<std::optional<std::string>> request =
      BuildPathwaysProfileRequest(options);

  ASSERT_TRUE(request.ok()) << request.status();
  ASSERT_TRUE(request->has_value());
  ExpectJsonEq(**request, R"json({
    "profileRequest": {
      "traceLocation": "gs://bucket/logs",
      "maxNumHosts": 4,
      "includeResourceManagers": true,
      "maxDurationSecs": 2.5,
      "xprofTraceOptions": {
        "traceDirectory": "gs://bucket/logs",
        "traceSessionName": "my_session",
        "hostTraceLevel": 0,
        "pwTraceOptions": {
          "enablePythonTracer": false,
          "advancedConfiguration": {
            "tpu_trace_mode": {"stringValue": "TRACE_ONLY_HOST"},
            "enable_fw_throttle_event": {"boolValue": true},
            "num_buffers": {"int64Value": 2}
          }
        }
      }
    }
  })json");
}

TEST(BuildPathwaysProfileRequestTest, TracesOneHostByDefault) {
  tensorflow::ProfileOptions options;
  options.set_repository_path("/logs");
  options.set_session_id("my_session");
  options.set_host_tracer_level(2);
  options.set_python_tracer_level(1);

  absl::StatusOr<std::optional<std::string>> request =
      BuildPathwaysProfileRequest(options);

  ASSERT_TRUE(request.ok()) << request.status();
  ASSERT_TRUE(request->has_value());
  ExpectJsonEq(**request, R"json({
    "profileRequest": {
      "traceLocation": "/logs",
      "maxNumHosts": 1,
      "includeResourceManagers": true,
      "maxDurationSecs": 0,
      "xprofTraceOptions": {
        "traceDirectory": "/logs",
        "traceSessionName": "my_session",
        "hostTraceLevel": 2,
        "pwTraceOptions": {
          "enablePythonTracer": true,
          "advancedConfiguration": {}
        }
      }
    }
  })json");
}

TEST(BuildPathwaysProfileRequestTest, ReturnsRequestFromOptions) {
  tensorflow::ProfileOptions options;
  options.set_repository_path("gs://bucket/logs");
  (*options.mutable_advanced_configuration())["pathways_profile_request"]
      .set_string_value(R"({"profileRequest": {}})");

  absl::StatusOr<std::optional<std::string>> request =
      BuildPathwaysProfileRequest(options);

  ASSERT_TRUE(request.ok()) << request.status();
  EXPECT_EQ(*request, R"({"profileRequest": {}})");
}

TEST(BuildPathwaysProfileRequestTest, RejectsRequestThatIsNotAString) {
  tensorflow::ProfileOptions options;
  options.set_repository_path("gs://bucket/logs");
  (*options.mutable_advanced_configuration())["pathways_profile_request"]
      .set_int64_value(1);

  EXPECT_EQ(BuildPathwaysProfileRequest(options).status().code(),
            absl::StatusCode::kInvalidArgument);
}

std::shared_ptr<xla::ifrt::Client> CreateCpuClient() {
  xla::CpuClientOptions options;
  options.cpu_device_count = 1;
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> pjrt_client =
      xla::GetXlaPjrtCpuClient(std::move(options));
  CHECK_OK(pjrt_client);
  xla::ifrt::PjRtClient::CreateOptions create_options;
  create_options.pjrt_client = *std::move(pjrt_client);
  absl::StatusOr<std::unique_ptr<xla::ifrt::PjRtClient>> client =
      xla::ifrt::PjRtClient::Create(std::move(create_options));
  CHECK_OK(client);
  return *std::move(client);
}

TEST(RegisterPathwaysProfilerTest, TracesClientWhileItIsAlive) {
  tensorflow::ProfileOptions options;
  options.set_repository_path("/logs");
  options.set_session_id("my_session");
  const size_t num_other_profilers =
      tsl::profiler::CreateProfilers(options).size();
  std::shared_ptr<xla::ifrt::Client> client = CreateCpuClient();

  RegisterPathwaysProfiler(client);
  std::vector<std::unique_ptr<tsl::profiler::ProfilerInterface>> profilers =
      tsl::profiler::CreateProfilers(options);

  ASSERT_EQ(profilers.size(), num_other_profilers + 1);
  // The profiler factory is registered last. CPU clients can't compile the
  // request, so the trace fails to start.
  EXPECT_FALSE(profilers.back()->Start().ok());
  profilers.clear();
  client.reset();
  EXPECT_EQ(tsl::profiler::CreateProfilers(options).size(),
            num_other_profilers);
}

}  // namespace
}  // namespace jax
