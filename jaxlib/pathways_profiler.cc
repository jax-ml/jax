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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "google/protobuf/struct.pb.h"
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/plugin_program.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace jax {
namespace {

using ::google::protobuf::Struct;
using ::google::protobuf::Value;
using AdvancedConfigValue = ::tensorflow::ProfileOptions::AdvancedConfigValue;

// Keys of `ProfileOptions.advanced_configuration` that configure how profiler
// sessions trace Pathways backends.
constexpr char kMaxNumHostsKey[] = "pathways_max_num_hosts";
constexpr char kProfileRequestKey[] = "pathways_profile_request";

Value StructValue(Struct fields) {
  Value value;
  *value.mutable_struct_value() = std::move(fields);
  return value;
}

// Returns `config` as a JSON string, boolean or number. JSON numbers are
// doubles, so integers beyond 2^53 lose precision.
absl::StatusOr<Value> ToJson(const AdvancedConfigValue& config) {
  Value value;
  switch (config.value_case()) {
    case AdvancedConfigValue::kStringValue:
      value.set_string_value(config.string_value());
      break;
    case AdvancedConfigValue::kBoolValue:
      value.set_bool_value(config.bool_value());
      break;
    case AdvancedConfigValue::kInt64Value:
      value.set_number_value(static_cast<double>(config.int64_value()));
      break;
    case AdvancedConfigValue::VALUE_NOT_SET:
      return absl::InvalidArgumentError("Unset advanced configuration value");
  }
  return value;
}

// Returns the JSON field of a Pathways `AdvancedConfigValue` that holds
// `config`, e.g. "int64Value". `config` must be set.
const char* PathwaysConfigValueField(const AdvancedConfigValue& config) {
  switch (config.value_case()) {
    case AdvancedConfigValue::kStringValue:
      return "stringValue";
    case AdvancedConfigValue::kBoolValue:
      return "boolValue";
    default:
      return "int64Value";
  }
}

// Traces the Pathways backend during a profiler session by executing a request
// as an `xla::ifrt::PluginProgram`: the first execution starts the trace and
// the second one stops it. The traced Pathways hosts write their traces into
// the profile directory of the session themselves.
class PathwaysProfiler : public tsl::profiler::ProfilerInterface {
 public:
  PathwaysProfiler(std::shared_ptr<xla::ifrt::Client> client,
                   tensorflow::ProfileOptions options)
      : client_(std::move(client)), options_(std::move(options)) {}

  absl::Status Start() override {
    absl::Status status = StartTrace();
    if (!status.ok()) {
      executable_ = nullptr;
      // tsl::profiler ignores the status unless `raise_error_on_start_failure`.
      LOG(WARNING) << "Failed to start tracing the Pathways backend: "
                   << status;
    }
    return status;
  }

  absl::Status Stop() override {
    if (executable_ == nullptr) {
      return absl::OkStatus();
    }
    absl::Status status = Execute();
    executable_ = nullptr;
    if (!status.ok()) {
      // tsl::profiler ignores the status.
      LOG(WARNING) << "Failed to stop tracing the Pathways backend: " << status;
    }
    return status;
  }

  absl::Status CollectData(tensorflow::profiler::XSpace* space) override {
    return absl::OkStatus();
  }

 private:
  absl::Status StartTrace() {
    ABSL_ASSIGN_OR_RETURN(std::optional<std::string> request,
                          BuildPathwaysProfileRequest(options_));
    if (!request.has_value()) {
      return absl::OkStatus();
    }
    auto program = std::make_unique<xla::ifrt::PluginProgram>();
    program->data = *std::move(request);
    ABSL_ASSIGN_OR_RETURN(
        executable_,
        client_->GetDefaultCompiler()
            ->CompileAndLoad(
                std::move(program),
                std::make_unique<xla::ifrt::PluginCompileOptions>())
            .Await());
    return Execute();
  }

  absl::Status Execute() {
    xla::ifrt::ExecuteOptions options;
    options.fill_status = true;
    ABSL_ASSIGN_OR_RETURN(
        xla::ifrt::LoadedExecutable::ExecuteResult result,
        executable_->Execute(/*args=*/{}, options, /*devices=*/std::nullopt));
    return result.status.Await();
  }

  const std::shared_ptr<xla::ifrt::Client> client_;
  const tensorflow::ProfileOptions options_;
  xla::ifrt::LoadedExecutableRef executable_;
};

ABSL_CONST_INIT absl::Mutex registered_client_mu(absl::kConstInit);

std::weak_ptr<xla::ifrt::Client>& RegisteredClient()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(registered_client_mu) {
  static absl::NoDestructor<std::weak_ptr<xla::ifrt::Client>> client;
  return *client;
}

// Runs while tsl::profiler holds the lock of its profiler factory registry, so
// it must not wait for the Pathways backend.
std::unique_ptr<tsl::profiler::ProfilerInterface> CreatePathwaysProfiler(
    const tensorflow::ProfileOptions& options) {
  absl::MutexLock lock(registered_client_mu);
  std::shared_ptr<xla::ifrt::Client> client = RegisteredClient().lock();
  if (client == nullptr) {
    return nullptr;
  }
  return std::make_unique<PathwaysProfiler>(std::move(client), options);
}

}  // namespace

absl::StatusOr<std::optional<std::string>> BuildPathwaysProfileRequest(
    const tensorflow::ProfileOptions& options) {
  if (options.repository_path().empty()) {
    return std::nullopt;
  }
  const auto& config = options.advanced_configuration();
  if (auto it = config.find(kProfileRequestKey); it != config.end()) {
    if (it->second.value_case() != AdvancedConfigValue::kStringValue) {
      return absl::InvalidArgumentError(
          absl::StrCat(kProfileRequestKey, " must be a string"));
    }
    return it->second.string_value();
  }

  Value max_num_hosts;
  max_num_hosts.set_number_value(1);
  Struct advanced_configuration;
  for (const auto& [key, config_value] : config) {
    ABSL_ASSIGN_OR_RETURN(Value value, ToJson(config_value));
    if (key == kMaxNumHostsKey) {
      max_num_hosts = std::move(value);
      continue;
    }
    Struct pathways_value;
    (*pathways_value.mutable_fields())[PathwaysConfigValueField(config_value)] =
        std::move(value);
    (*advanced_configuration.mutable_fields())[key] =
        StructValue(std::move(pathways_value));
  }

  Struct pw_trace_options;
  auto& pw_trace_fields = *pw_trace_options.mutable_fields();
  pw_trace_fields["enablePythonTracer"].set_bool_value(
      options.python_tracer_level() > 0);
  pw_trace_fields["advancedConfiguration"] =
      StructValue(std::move(advanced_configuration));

  Struct xprof_trace_options;
  auto& xprof_trace_fields = *xprof_trace_options.mutable_fields();
  xprof_trace_fields["traceDirectory"].set_string_value(
      options.repository_path());
  xprof_trace_fields["traceSessionName"].set_string_value(options.session_id());
  xprof_trace_fields["hostTraceLevel"].set_number_value(
      options.host_tracer_level());
  xprof_trace_fields["pwTraceOptions"] =
      StructValue(std::move(pw_trace_options));

  Struct profile_request;
  auto& profile_request_fields = *profile_request.mutable_fields();
  profile_request_fields["traceLocation"].set_string_value(
      options.repository_path());
  profile_request_fields["maxNumHosts"] = std::move(max_num_hosts);
  profile_request_fields["includeResourceManagers"].set_bool_value(true);
  profile_request_fields["maxDurationSecs"].set_number_value(
      static_cast<double>(options.duration_ms()) / 1000);
  profile_request_fields["xprofTraceOptions"] =
      StructValue(std::move(xprof_trace_options));

  Struct request;
  (*request.mutable_fields())["profileRequest"] =
      StructValue(std::move(profile_request));
  std::string json;
  ABSL_RETURN_IF_ERROR(
      tsl::protobuf::util::MessageToJsonString(request, &json));
  return json;
}

void RegisterPathwaysProfiler(std::shared_ptr<xla::ifrt::Client> client) {
  {
    absl::MutexLock lock(registered_client_mu);
    RegisteredClient() = client;
  }
  // tsl::profiler does not support unregistering profiler factories, so
  // register a single one that traces the most recently registered client.
  [[maybe_unused]] static const bool registered = [] {
    tsl::profiler::RegisterProfilerFactory(&CreatePathwaysProfiler);
    return true;
  }();
}

}  // namespace jax
