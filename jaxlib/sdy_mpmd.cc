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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "mlir/CAPI/IR.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/nanobind.h"
// IWYU pragma: begin_keep; Nanobind conversions for std types.
#include "nanobind/stl/map.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/vector.h"
// IWYU pragma: end_keep
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_assignment_map.h"
#include "shardy/integrations/python/jax/mpmd/jaxlib/mpmd_program.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device.h"
#include "jaxlib/py_executable.h"
#include "jaxlib/py_mpmd_loaded_executable.h"
#include "jaxlib/py_user_context.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"  // IWYU pragma: keep; Needed for ValueOrThrow
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/conversions/mpmd/lower_to_ifrt.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/program_memory_tracer.h"
#include "xla/python/ifrt/mpmd_executable.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

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
using ::mlir::mpmd::FragmentScheduleRule;
using ::mlir::mpmd::FunctionIOShardingSpecsAndMeshes;
using ::mlir::mpmd::MpmdProgram;
using ::mlir::mpmd::NamedSpmdShardingSpec;
using ::mlir::mpmd::PartitioningOptions;
using ::mlir::mpmd::PartitioningPhase;
using ::mlir::mpmd::PartitioningResult;
using ::mlir::mpmd::SplitFragmentType;
using ::mlir::mpmd::SpmdTensorPartitionSpec;
using ::mlir::mpmd::UserAssignmentMap;
using ::xla::ifrt::mpmd::EnvOptionsOverride;
using ::xla::ifrt::mpmd::GetCompileOptions;
using ::xla::ifrt::mpmd::LowerToIfrt;

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

// Helper function to create an xla::ifrt::DeviceListeRef from JAX devices.
absl::StatusOr<xla::ifrt::DeviceListRef> MakeDeviceListFromPyDevices(
    jax::nb_class_ptr<jax::PyClient> py_client,
    std::vector<jax::nb_class_ptr<jax::PyDevice>> py_devices) {
  absl::InlinedVector<xla::ifrt::Device*, 1> unwrapped_devices;
  unwrapped_devices.reserve(py_devices.size());
  for (const jax::nb_class_ptr<jax::PyDevice>& d : py_devices) {
    if (d->client().get() != py_client.get()) {
      if (d->client().get() == nullptr) {
        return xla::InvalidArgument("Unattached device '%s' (expected: '%s')",
                                    d->device()->ToString(),
                                    py_client->platform_name());
      }
      return xla::InvalidArgument(
          "Device '%s' is from client '%s' (expected: '%s')",
          d->device()->ToString(), d->client()->platform_name(),
          py_client->platform_name());
    }
    unwrapped_devices.push_back(d->device());
  }
  return py_client->ifrt_client()->MakeDeviceList(unwrapped_devices);
}

// Calls `Compiler::CompileAndLoad()` and returns
// `PyMpmdLoadedExecutable` for the compiled MPMD program.
//
// Requires GIL.
absl::StatusOr<std::unique_ptr<PyMpmdLoadedExecutable>>
ExperimentalCompileMpmd(
    nb::object backend_py, MlirModule c_module, nb::sequence devices_py,
    const std::vector<nb::object> out_avals,
    std::optional<const std::vector<nb::object>> out_shardings,
    const absl::flat_hash_map<std::string, nb::object>& xla_compile_options,
    const absl::flat_hash_map<std::string, nb::handle>&
        loaded_executable_bindings) {
  auto backend = nb::cast<jax::nb_class_ptr<jax::PyClient>>(backend_py);
  auto devices = nb::cast<std::vector<jax::nb_class_ptr<jax::PyDevice>>>(
      devices_py);

  // Get IFRT client and construct an IFRT device list.
  xla::ifrt::Client* client = backend->ifrt_client();
  TF_ASSIGN_OR_RETURN(const xla::ifrt::DeviceListRef device_list,
                      MakeDeviceListFromPyDevices(backend, std::move(devices)));

  xla::ifrt::UserContextScope user_context_scope(jax::PyUserContext::Create());
  xla::ifrt::LoadedExecutableRef loaded_executable;
  auto compile_options = std::make_shared<absl::flat_hash_map<
      std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>();
  for (const auto& [compile_key, atom_program_compile_options_py] :
       xla_compile_options) {
    const xla::CompileOptions* atom_program_compile_options =
        nb::cast<const xla::CompileOptions*>(atom_program_compile_options_py);
    compile_options->emplace(compile_key,
                             std::make_unique<xla::ifrt::XlaCompileOptions>(
                                 *atom_program_compile_options, device_list));
  }
  absl::flat_hash_map<std::string, xla::ifrt::LoadedExecutableRef>
      loaded_exec_bindings;
  for (const auto& [exec_symbol_name, py_exec] : loaded_executable_bindings) {
    jax::PyLoadedExecutable* executable =
        nb::cast<jax::PyLoadedExecutable*>(py_exec);
    loaded_exec_bindings.emplace(exec_symbol_name,
                                 executable->shared_ifrt_loaded_executable());
  }
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        loaded_executable,
        client->GetDefaultCompiler()
            ->CompileAndLoad(
                std::make_unique<xla::ifrt::IfrtIRProgram>(unwrap(c_module)),
                std::make_unique<xla::ifrt::IfrtIRCompileOptions>(
                    xla::ifrt::GetDeviceIds(device_list),
                    std::move(loaded_exec_bindings), compile_options))
            .Await());
  }
  if (!llvm::isa<xla::ifrt::MpmdLoadedExecutable>(loaded_executable.get())) {
    return absl::InternalError(
        "Loaded executable must be an `xla::ifrt::MpmdLoadedExecutable`.");
  };
  if (out_shardings.has_value()) {
    return std::make_unique<PyMpmdLoadedExecutable>(
        backend,
        std::static_pointer_cast<xla::ifrt::MpmdLoadedExecutable>(
            std::move(loaded_executable)),
        out_avals, *out_shardings);
  } else {
    auto exec_out_shardings = loaded_executable->GetOutputShardings();
    if (!exec_out_shardings.has_value()) {
      return absl::InternalError("Executable does not have output shardings.");
    }
    std::vector<nb::object> out_shardings_py;
    for (auto& op_sharding : *exec_out_shardings) {
      out_shardings_py.push_back(nb::cast(std::move(op_sharding)));
    }
    return std::make_unique<PyMpmdLoadedExecutable>(
        backend,
        std::static_pointer_cast<xla::ifrt::MpmdLoadedExecutable>(
            std::move(loaded_executable)),
        out_avals, out_shardings_py);
  }
}

NB_MODULE(_sdy_mpmd, m) {
  nb::enum_<PartitioningPhase>(m, "PartitioningPhase", nb::is_flag())
      .value("NONE", PartitioningPhase::kNone)
      .value("IMPORT", PartitioningPhase::kImport)
      .value("OPTIMIZE", PartitioningPhase::kOptimize)
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
                    std::optional<mlir::mpmd::SplitFragmentType>,
                    const std::string&>(),
           nb::arg("origins"), nb::arg("stage_id").none() = std::nullopt,
           nb::arg("call_counter").none() = std::nullopt,
           nb::arg("split_type").none() = std::nullopt, nb::arg("mesh_name"))
      .def_ro("origins", &FragmentInfo::origins)
      .def_ro("stage_id", &FragmentInfo::stage_id)
      .def_ro("call_counter", &FragmentInfo::call_counter)
      .def_ro("split_type", &FragmentInfo::split_type)
      .def_ro("mesh_name", &FragmentInfo::mesh_name);

  nb::class_<FragmentScheduleRule>(m, "FragmentScheduleRule")
      .def(nb::init<const std::vector<FragmentInfo>&>(),
           nb::arg("ordered_fragments"))
      .def_ro("ordered_fragments", &FragmentScheduleRule::ordered_fragments);

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

  m.def(
      "lower_to_ifrt",
      [](MlirModule module) -> void {
        return xla::ThrowIfError(LowerToIfrt(unwrap(module)));
      },
      nb::arg("module"));

  m.def("get_compile_options",
        [](MlirModule c_module,
           const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
               compile_options_overrides) -> nb::dict {
          auto module = unwrap(c_module);
          auto compile_options_map = ValueOrThrow(
              GetCompileOptions(module, compile_options_overrides));
          nb::dict out;
          for (const auto& [name, options] : compile_options_map) {
            out[nb::cast(name)] =
                nb::steal<nb::object>(nanobind::cast(options).release().ptr());
          }
          return out;
        });

  m.def("experimental_compile_mpmd",
        xla::ValueOrThrowWrapper(ExperimentalCompileMpmd),  //
        nb::arg("backend"),                                 //
        nb::arg("ifrt_mlir_module"),                        //
        nb::arg("devices"),                                 //
        nb::arg("out_avals"),                               //
        nb::arg("out_shardings"),                           //
        nb::arg("xla_compile_options"),                     //
        nb::arg("loaded_executable_bindings")               //
  );

  nb::class_<xla::ifrt::IfrtIrProgramMemoryStats>(m, "IfrtIrProgramMemoryStats")
      .def_ro(
          "argument_size_in_bytes",
          &xla::ifrt::IfrtIrProgramMemoryStats::argument_size_in_bytes)
      .def_ro("output_size_in_bytes",
                    &xla::ifrt::IfrtIrProgramMemoryStats::output_size_in_bytes)
      .def_ro(
          "device_to_peak_bytes_used",
          &xla::ifrt::IfrtIrProgramMemoryStats::device_to_peak_bytes_used)
      .def_ro("device_to_min_memory_bytes_available",
                    &xla::ifrt::IfrtIrProgramMemoryStats::
                        device_to_min_memory_bytes_available)
      .def_ro(
          "host_argument_size_in_bytes",
          &xla::ifrt::IfrtIrProgramMemoryStats::host_argument_size_in_bytes)
      .def_ro(
          "host_output_size_in_bytes",
          &xla::ifrt::IfrtIrProgramMemoryStats::host_output_size_in_bytes);

  auto mpmd_executable =
      nb::class_<PyMpmdLoadedExecutable>(m, "MpmdLoadedExecutable");
  mpmd_executable.def(
      "execute",
      xla::ValueOrThrowWrapper(&PyMpmdLoadedExecutable::Execute),
      nb::arg("args"));
  mpmd_executable.def(
      "execute_fastpath",
      xla::ValueOrThrowWrapper(&PyMpmdLoadedExecutable::ExecuteFastpath));
  mpmd_executable.def("setup_fastpath",
                      &PyMpmdLoadedExecutable::SetupFastpath);
  mpmd_executable.def("input_shardings",
                      xla::ValueOrThrowWrapper(
                          &PyMpmdLoadedExecutable::GetParameterShardings));
  mpmd_executable.def("output_shardings",
                      xla::ValueOrThrowWrapper(
                          &PyMpmdLoadedExecutable::GetOutputShardings));
  mpmd_executable.def("input_layouts",
                      xla::ValueOrThrowWrapper(
                          &PyMpmdLoadedExecutable::GetParameterLayouts));
  mpmd_executable.def("output_layouts",
                      xla::ValueOrThrowWrapper(
                          &PyMpmdLoadedExecutable::GetOutputLayouts));
  mpmd_executable.def(
      "get_compiled_memory_stats",
      [](PyMpmdLoadedExecutable& exec)
          -> absl::flat_hash_map<std::string, nb::object> {
        absl::flat_hash_map<std::string, xla::CompiledMemoryStats> stats;
        {
          nb::gil_scoped_release gil;
          stats = xla::ValueOrThrow(exec.GetMpmdCompiledMemoryStats());
        }
        absl::flat_hash_map<std::string, nb::object> py_stats;
        for (const auto& [atom_program_name, memory_stats] : stats) {
          py_stats[atom_program_name] = nb::steal<nb::object>(
              nb::cast(memory_stats).release().ptr());
        }
        return py_stats;
      });
  mpmd_executable.def(
      "get_ifrt_ir_program_memory_stats",
      xla::ValueOrThrowWrapper(
          &PyMpmdLoadedExecutable::GetIfrtIrProgramMemoryStats));
  mpmd_executable.def(
      "get_ifrt_ir_program_xprof_url",
      xla::ValueOrThrowWrapper(
          &PyMpmdLoadedExecutable::GetIfrtIrProgramXprofUrl));
  mpmd_executable.def(
      "hlo_modules",
      [](PyMpmdLoadedExecutable& exec)
          -> absl::StatusOr<
              absl::flat_hash_map<std::string, std::vector<nb::object>>> {
        absl::flat_hash_map<std::string,
                            std::vector<std::shared_ptr<xla::HloModule>>>
            hlo_modules;
        {
          nb::gil_scoped_release gil;
          TF_ASSIGN_OR_RETURN(hlo_modules, exec.GetHloModules());
        }
        absl::flat_hash_map<std::string, std::vector<nb::object>>
            py_hlo_modules;
        py_hlo_modules.reserve(hlo_modules.size());
        for (auto& [name, modules] : hlo_modules) {
          std::vector<nb::object> py_modules;
          py_modules.reserve(modules.size());
          for (auto& hlo_module : modules) {
            py_modules.push_back(nb::cast(std::move(hlo_module)));
          }
          py_hlo_modules.insert({name, std::move(py_modules)});
        }
        return py_hlo_modules;
      });
  mpmd_executable.def("cost_analysis", [](PyMpmdLoadedExecutable& exec) {
    auto attrs = xla::ValueOrThrow(exec.GetMpmdCostAnalysis());
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, xla::PjRtValueType>>
        map;
    map.reserve(attrs.size());
    for (auto& [key, value] : attrs) {
      map.insert({key, xla::ifrt::ToPjRtAttributeMap(std::move(value))});
    }
    return map;
  });
  mpmd_executable.def("get_raw_ptr", [](PyMpmdLoadedExecutable* self) {
    return reinterpret_cast<uintptr_t>(self);
  });
}

}  // namespace
}  // namespace jax::mpmd
