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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_PY_MPMD_LOADED_EXECUTABLE_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_PY_MPMD_LOADED_EXECUTABLE_H_

#include <stdbool.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/jax_jit.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_client.h"
#include "jaxlib/pytree.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"  // IWYU pragma: keep
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/ir/program_memory_tracer.h"
#include "xla/python/ifrt/mpmd_executable.h"
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace jax {
namespace mpmd {

namespace nb = ::nanobind;

struct MpmdExecutableFastpathData {
  explicit MpmdExecutableFastpathData(jax::PyTreeRegistry* registry)
      : out_pytree_def(registry) {}

  jax::PyTreeDef out_pytree_def;
  std::vector<bool> kept_var_bitvec;
};

class MpmdExecutableFastpathCache {
 public:
  std::pair<std::shared_ptr<MpmdExecutableFastpathData>, bool>
  GetOrInsertIfAbsent(const jax::CallSignature& call_signature,
                      jax::PyTreeRegistry* pytree_registry);

 private:
  struct CacheEntry {
    jax::CallSignature call_signature;
    std::shared_ptr<MpmdExecutableFastpathData> data;
  };
  std::optional<CacheEntry> cache_ ABSL_GUARDED_BY(mutex_);
  absl::Mutex mutex_;
};

class PyMpmdLoadedExecutable {
 public:
  PyMpmdLoadedExecutable(
      jax::nb_class_ptr<jax::PyClient> backend,
      std::shared_ptr<xla::ifrt::MpmdLoadedExecutable> ifrt_loaded_executable,
      absl::Span<const nb::object> out_avals,
      absl::Span<const nb::object> out_shardings)
      : backend_(std::move(backend)),
        ifrt_loaded_executable_(std::move(ifrt_loaded_executable)),
        out_avals_(out_avals.begin(), out_avals.end()),
        out_shardings_(out_shardings.begin(), out_shardings.end()) {
    out_dtypes_.reserve(out_avals_.size());
    out_shapes_.reserve(out_avals_.size());
    for (const nb::object& aval : out_avals_) {
      out_dtypes_.push_back(aval.attr("dtype"));
      out_shapes_.push_back(nb::cast<std::vector<int64_t>>(aval.attr("shape")));
    }
  }

  absl::StatusOr<nb::list> Execute(nb::sequence args);

  absl::StatusOr<nb::object> ExecuteFastpath(
      nb::sequence args, nb::dict kwargs);

  // Returns a mapping between atom program name and compiled memory stats.
  absl::StatusOr<absl::flat_hash_map<std::string, xla::CompiledMemoryStats>>
  GetMpmdCompiledMemoryStats() const;

  absl::StatusOr<xla::ifrt::IfrtIrProgramMemoryStats>
  GetIfrtIrProgramMemoryStats() const;

  absl::StatusOr<std::string> GetIfrtIrProgramXprofUrl() const;

  // Returns a mapping between atom program name and map of cost properties.
  absl::StatusOr<absl::flat_hash_map<std::string, xla::ifrt::AttributeMap>>
  GetMpmdCostAnalysis();

  absl::StatusOr<std::vector<xla::OpSharding>> GetParameterShardings();

  absl::StatusOr<absl::flat_hash_map<
      std::string, std::vector<std::shared_ptr<xla::HloModule>>>>
  GetHloModules();

  absl::StatusOr<std::vector<xla::OpSharding>> GetOutputShardings();

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts();

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts();

  void ParseArguments(nb::sequence args, nb::dict kwargs,
                      jax::ArgumentSignature& arg_signature,
                      absl::InlinedVector<nb::object, 2>& flat_args);

  absl::Status ComputeCallSignature(
      absl::InlinedVector<nb::object, 2>& flat_args,
      jax::CallSignature& call_signature);

  void PopulateCache(MpmdExecutableFastpathData& cache_data,
                     nb::tuple out_and_fastpath_data);

  void SetupFastpath(nb::callable cache_miss, nb::object pytree_registry);

 protected:
  jax::nb_class_ptr<jax::PyClient> backend_;
  std::shared_ptr<xla::ifrt::MpmdLoadedExecutable> ifrt_loaded_executable_;
  std::vector<nb::object> out_avals_;
  std::vector<nb::object> out_dtypes_;
  std::vector<std::vector<int64_t>> out_shapes_;
  std::vector<nb::object> out_shardings_;
  MpmdExecutableFastpathCache cache_;
  nb::callable cache_miss_;
  jax::nb_class_ptr<jax::PyTreeRegistry> pytree_registry_;
};

}  // namespace mpmd
}  // namespace jax

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_PY_MPMD_LOADED_EXECUTABLE_H_
