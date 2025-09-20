/* Copyright 2025 The JAX Authors

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

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "mlir/CAPI/IR.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "nanobind/nanobind.h"
// IWYU pragma: begin_keep; Nanobind conversions for std types.
#include "nanobind/stl/map.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/vector.h"
// IWYU pragma: end_keep
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/integrations/python/jax/mpmd/jaxlib/mpmd_program.h"

namespace nb = nanobind;

namespace jax::mpmd {
namespace {

using ::mlir::Builder;
using ::mlir::ModuleOp;
using ::mlir::mpmd::FlatMesh;
using ::mlir::mpmd::FragmentInfo;
using ::mlir::mpmd::FragmentMergeRule;
using ::mlir::mpmd::FragmentMergeRules;
using ::mlir::mpmd::FragmentOrigin;
using ::mlir::mpmd::FunctionIOShardingSpecsAndMeshes;
using ::mlir::mpmd::MpmdProgram;
using ::mlir::mpmd::NamedSpmdShardingSpec;
using ::mlir::mpmd::PartitioningOptions;
using ::mlir::mpmd::PartitioningPhase;
using ::mlir::mpmd::PartitioningResult;
using ::mlir::mpmd::SplitFragmentType;
using ::mlir::mpmd::SpmdTensorPartitionSpec;
using ::mlir::mpmd::UserAssignmentMap;

// Wrapper of PartitioningResult, which stores MlirModules instead of ModuleOps.
struct PartitioningResultWrapper {
  MlirModule mpmd_module;
  mlir::mpmd::FunctionIOShardingSpecsAndMeshes
      module_io_sharding_specs_and_meshes;
};

// name -> [mesh | (mesh, stage)]
using PyUserAssignmentMap =
    std::map<std::string,
             std::variant<std::string, std::pair<std::string, int64_t>>>;

UserAssignmentMap GetCppUserAssignmentMap(const PyUserAssignmentMap& py_map) {
  UserAssignmentMap cpp_map;
  for (const auto& [name, py_value] : py_map) {
    if (const auto* mesh = std::get_if<std::string>(&py_value)) {
      cpp_map[name] = std::make_pair(*mesh, std::nullopt);
    } else if (const auto* mesh_stage =
                   std::get_if<std::pair<std::string, int64_t>>(&py_value)) {
      cpp_map[name] = *mesh_stage;
    }
  }
  return cpp_map;
}

NB_MODULE(sdy_mpmd, m) {
  nb::enum_<PartitioningPhase>(m, "PartitioningPhase")
      .value("NONE", PartitioningPhase::kNone)
      .value("IMPORT", PartitioningPhase::kImport)
      .value("PARTITION", PartitioningPhase::kPartition)
      .value("ALL", PartitioningPhase::kAll)
      .export_values();

  nb::enum_<SplitFragmentType>(m, "SplitFragmentType")
      .value("KEEP_TRANSFERRED", SplitFragmentType::kKeepTransferred)
      .value("DROP_TRANSFERRED", SplitFragmentType::kDropTransferred)
      .export_values();

  nb::class_<FragmentOrigin>(m, "FragmentOrigin")
      .def(nb::init<const std::string&, int>(), nb::arg("computation_name"),
           nb::arg("transpose_count"))
      .def_ro("computation_name", &FragmentOrigin::computation_name)
      .def_ro("transpose_count", &FragmentOrigin::transpose_count);

  nb::class_<FragmentInfo>(m, "FragmentInfo")
      .def(nb::init<const std::vector<FragmentOrigin>&, std::optional<int>,
                    std::optional<int>,
                    std::optional<mlir::mpmd::SplitFragmentType>>(),
           nb::arg("origins"), nb::arg("stage_id"), nb::arg("call_counter"),
           nb::arg("split_type"))
      .def_ro("origins", &FragmentInfo::origins)
      .def_ro("stage_id", &FragmentInfo::stage_id)
      .def_ro("call_counter", &FragmentInfo::call_counter)
      .def_ro("split_type", &FragmentInfo::split_type);

  nb::class_<FragmentMergeRule>(m, "FragmentMergeRule")
      .def(nb::init<const std::vector<FragmentInfo>&, FragmentInfo>(),
           nb::arg("sources"), nb::arg("target"))
      .def_ro("sources", &FragmentMergeRule::sources)
      .def_ro("target", &FragmentMergeRule::target);

  nb::class_<PartitioningResultWrapper>(m, "PartitioningResult")
      .def_ro("mpmd_module", &PartitioningResultWrapper::mpmd_module)
      .def_ro("module_io_sharding_specs_and_meshes",
              &PartitioningResultWrapper::module_io_sharding_specs_and_meshes);

  m.def(
      "apply_mpmd_partitioning",
      [](MlirModule c_module, std::string func_name,
         const std::vector<std::pair<std::string, FlatMesh>>& named_meshes,
         const mpmd::PyUserAssignmentMap& assignment,
         const std::vector<std::optional<std::string>>& input_meshes,
         const std::vector<std::optional<std::string>>& output_meshes,
         const std::vector<int64_t>& donate_argnums,
         const std::optional<
             std::map<std::string, std::variant<std::string, bool>>>&
             partitioning_options,
         const FragmentMergeRules& fragment_merge_rules,
         PartitioningPhase phases) -> PartitioningResultWrapper {
        PartitioningOptions options;
        if (partitioning_options) {
          options = mlir::mpmd::ParsePartitioningOptions(*partitioning_options);
        }
        MpmdProgram program{.module = unwrap(c_module),
                            .func_name = func_name,
                            .options = std::move(options),
                            .named_meshes = named_meshes,
                            .assignment = GetCppUserAssignmentMap(assignment),
                            .input_meshes = input_meshes,
                            .output_meshes = output_meshes,
                            .donate_argnums = donate_argnums,
                            .fragment_merge_rules = fragment_merge_rules};

        PartitioningResult partitioning_result =
            program.ApplyPartitioning(phases);

        return PartitioningResultWrapper{
            wrap(partitioning_result.mpmd_module),
            std::move(partitioning_result.module_io_sharding_specs_and_meshes),
        };
      },
      nb::arg("module"), nb::arg("func_name"), nb::arg("named_meshes"),
      nb::arg("assignment"), nb::arg("input_meshes"), nb::arg("output_meshes"),
      nb::arg("donate_argnums"),
      nb::arg("partitioning_options").none() = std::nullopt,
      nb::arg("fragment_merge_rules"), nb::arg("phases"));

  m.def("get_fragment_info",
        [](MlirModule c_module) -> std::vector<FragmentInfo> {
          std::vector<FragmentInfo> fragment_info;
          auto module = unwrap(c_module);
          // Walk module and get info for each fragment
          module.walk([&fragment_info](mlir::mpmd::FragmentOp fragment) {
            fragment_info.push_back(mlir::mpmd::GetFragmentInfo(fragment));
          });
          return fragment_info;
        });

  nb::class_<NamedSpmdShardingSpec>(m, "NamedSpmdShardingSpec")
      .def(nb::init<std::string const&, SpmdTensorPartitionSpec,
                    std::optional<std::string>>(),
           nb::arg("mesh_name"), nb::arg("tensor_spec"),
           nb::arg("memory_kind").none() = std::nullopt)
      .def_ro("mesh_name", &NamedSpmdShardingSpec::mesh_name)
      .def_ro("tensor_spec", &NamedSpmdShardingSpec::tensor_spec)
      .def_ro("memory_kind", &NamedSpmdShardingSpec::memory_kind);
  nb::class_<FunctionIOShardingSpecsAndMeshes>(
      m, "FunctionIOShardingSpecsAndMeshes")
      .def(nb::init<std::vector<NamedSpmdShardingSpec>,
                    std::vector<NamedSpmdShardingSpec>>())
      .def_ro("input_specs", &FunctionIOShardingSpecsAndMeshes::input_specs)
      .def_ro("output_specs", &FunctionIOShardingSpecsAndMeshes::output_specs);

  m.def(
      "clone_mlir_module",
      [](MlirModule c_module, const std::vector<std::string>& unit_attributes) {
        MlirOperation op = mlirModuleGetOperation(c_module);
        MlirModule module = mlirModuleFromOperation(mlirOperationClone(op));
        if (unit_attributes.empty()) {
          return module;
        }

        ModuleOp module_op = unwrap(module);
        for (const std::string& attr_name : unit_attributes) {
          module_op->setAttr(attr_name, Builder(module_op).getUnitAttr());
        }
        return wrap(module_op);
      },
      nb::arg("c_module"),
      nb::arg("unit_attributes") = std::vector<std::string>());
}

}  // namespace
}  // namespace jax::mpmd
