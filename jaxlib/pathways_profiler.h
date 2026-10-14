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

#ifndef JAXLIB_PATHWAYS_PROFILER_H_
#define JAXLIB_PATHWAYS_PROFILER_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "xla/python/ifrt/client.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace jax {

// Returns the JSON-encoded Pathways profile request that traces the Pathways
// backend during a profiler session with `options`, or nullopt if the session
// has no log directory. Each traced Pathways host writes its trace to
// `<log_dir>/plugins/profile/<session_id>`.
//
// `options.advanced_configuration` can set `pathways_max_num_hosts`, the
// maximum number of Pathways hosts to trace (defaults to 1), or
// `pathways_profile_request`, a JSON-encoded request (string) to return
// instead. These keys are not forwarded to the Pathways hosts.
absl::StatusOr<std::optional<std::string>> BuildPathwaysProfileRequest(
    const tensorflow::ProfileOptions& options);

// Makes every profiler session in this process also trace the Pathways backend
// of `client` while `client` is alive. Replaces the previously registered
// client, if any.
void RegisterPathwaysProfiler(std::shared_ptr<xla::ifrt::Client> client);

}  // namespace jax

#endif  // JAXLIB_PATHWAYS_PROFILER_H_
