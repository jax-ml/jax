/* Copyright 2020 The JAX Authors

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

#include "jaxlib/xla_compiler.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/dlpack.h"
#include "jaxlib/hash_util.h"
#include "jaxlib/py_client.h"
#include "xla/array.h"
#include "xla/client/executable_build_options.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace nb = nanobind;

namespace xla {

void BuildXlaCompilerSubmodule(nb::module_& m) {
  m.def(
      "hlo_module_cost_analysis",
      xla::ValueOrThrowWrapper([](jax::PyClient* client,
                                  const HloModule& module)
                                   -> absl::StatusOr<nb::dict> {
        TF_ASSIGN_OR_RETURN(auto analysis,
                            client->pjrt_client()->GetHloCostAnalysis());
        TF_RETURN_IF_ERROR(module.entry_computation()->Accept(analysis.get()));

        // Convert from HloCostAnalysis::Properties to a standard map.
        nb::dict ret;
        analysis->properties().ForEach([&](std::string_view key, float val) {
          ret[nb::str(key.data(), key.size())] = nb::cast(val);
        });
        return ret;
      }));

  // Device assignments
  nb::class_<DeviceAssignment>(m, "DeviceAssignment")
      .def_static(
          "create",
          xla::ValueOrThrowWrapper([](nb::ndarray<int, nb::ndim<2>> array)
                                       -> absl::StatusOr<DeviceAssignment> {
            if (array.ndim() != 2) {
              return InvalidArgument(
                  "Argument to DeviceAssignment constructor must be a "
                  "2D array, received an %dD array.",
                  array.ndim());
            }
            DeviceAssignment result(array.shape(0), array.shape(1));
            for (int i = 0; i < array.shape(0); ++i) {
              for (int j = 0; j < array.shape(1); ++j) {
                result(i, j) = array(i, j);
              }
            }
            return result;
          }))
      .def("replica_count", &DeviceAssignment::replica_count)
      .def("computation_count", &DeviceAssignment::computation_count)
      .def("__repr__", &DeviceAssignment::ToString)
      .def("serialize",
           xla::ValueOrThrowWrapper(
               [](const DeviceAssignment& da) -> absl::StatusOr<nb::bytes> {
                 DeviceAssignmentProto proto;
                 da.Serialize(&proto);
                 std::string result;
                 if (!tsl::SerializeToStringDeterministic(proto, &result)) {
                   return Unknown(
                       "Failed to serialize the DeviceAssignmentProto.");
                 }
                 return nb::bytes(result.data(), result.size());
               }));

  nb::class_<CompileOptions> compile_options(m, "CompileOptions");
  compile_options
      .def("__init__",
           [](CompileOptions* self) {
             new (self) CompileOptions();
             DebugOptions* debug_options =
                 self->executable_build_options.mutable_debug_options();
             // Sets fast-math-disabling default options expected by JAX.
             debug_options->set_xla_cpu_enable_fast_min_max(false);
             debug_options->set_xla_gpu_enable_fast_min_max(false);
           })
      .def("__getstate__",
           [](const CompileOptions& self) -> nb::tuple {
             auto proto = ValueOrThrow(self.ToProto());
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               // throw converted by PyBind to a Python RuntimeError.
               throw XlaRuntimeError(
                   absl::StrCat("CompileOptions.py_pickle: ",
                                "SerializeToStringDeterministic failed"));
             }
             return nb::make_tuple(nb::bytes(result.data(), result.size()));
           })
      .def("__setstate__",
           [](CompileOptions* self, nb::tuple t) {
             CompileOptionsProto result;
             nb::bytes serialized = nb::cast<nb::bytes>(t[0]);
             result.ParseFromArray(serialized.c_str(), serialized.size());
             new (self) CompileOptions(
                 ValueOrThrow(CompileOptions::FromProto(result)));
           })
      .def("SerializeAsString",
           [](const CompileOptions& self) -> nb::bytes {
             auto proto = ValueOrThrow(self.ToProto());
             std::string result;
             if (!tsl::SerializeToStringDeterministic(proto, &result)) {
               // throw converted by PyBind to a Python RuntimeError.
               throw XlaRuntimeError(
                   absl::StrCat("CompileOptions.SerializeAsString: ",
                                "SerializeToStringDeterministic failed"));
             }
             return nb::bytes(result.data(), result.size());
           })
      .def_static("ParseFromString",
                  [](nb::bytes s) {
                    CompileOptionsProto result;
                    result.ParseFromArray(s.c_str(), s.size());
                    return ValueOrThrow(CompileOptions::FromProto(result));
                  })
      .def_rw("argument_layouts", &CompileOptions::argument_layouts)
      .def_rw("parameter_is_tupled_arguments",
              &CompileOptions::parameter_is_tupled_arguments)
      .def_rw("compile_portable_executable",
              &CompileOptions::compile_portable_executable)
      .def_ro("executable_build_options",
              &CompileOptions::executable_build_options)
      .def_rw("env_option_overrides", &CompileOptions::env_option_overrides)
      .def_prop_rw(
          "num_replicas",
          [](const CompileOptions& options) {
            return options.executable_build_options.num_replicas();
          },
          [](CompileOptions& options, int num_replicas) {
            options.executable_build_options.set_num_replicas(num_replicas);
          })
      .def_prop_rw(
          "num_partitions",
          [](const CompileOptions& options) {
            return options.executable_build_options.num_partitions();
          },
          [](CompileOptions& options, int num_partitions) {
            options.executable_build_options.set_num_partitions(num_partitions);
          })
      .def_prop_rw(
          "profile_version",
          [](const CompileOptions& options) { return options.profile_version; },
          [](CompileOptions& options, int64_t profile_version) {
            options.profile_version = profile_version;
          })
      .def_prop_rw(
          "device_assignment",
          [](const CompileOptions& options) -> std::optional<DeviceAssignment> {
            return options.executable_build_options.has_device_assignment()
                       ? std::optional<DeviceAssignment>(
                             options.executable_build_options
                                 .device_assignment())
                       : std::nullopt;
          },
          [](CompileOptions& options,
             const DeviceAssignment& device_assignment) {
            options.executable_build_options.set_device_assignment(
                device_assignment);
          });

  nb::enum_<DebugOptions::AutotuneCacheMode>(m, "AutotuneCacheMode")
      .value("UNSPECIFIED", DebugOptions::AUTOTUNE_CACHE_MODE_UNSPECIFIED)
      .value("UPDATE", DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE)
      .value("READ", DebugOptions::AUTOTUNE_CACHE_MODE_READ);

  nb::class_<DebugOptions>(m, "DebugOptions")
      .def("__repr__", &DebugOptions::DebugString)
      .def_prop_rw("xla_backend_optimization_level",
                   &DebugOptions::xla_backend_optimization_level,
                   &DebugOptions::set_xla_backend_optimization_level)
      .def_prop_rw("xla_cpu_enable_fast_math",
                   &DebugOptions::xla_cpu_enable_fast_math,
                   &DebugOptions::set_xla_cpu_enable_fast_math)
      .def_prop_rw("xla_cpu_enable_xprof_traceme",
                   &DebugOptions::xla_cpu_enable_xprof_traceme,
                   &DebugOptions::set_xla_cpu_enable_xprof_traceme)
      .def_prop_rw("xla_cpu_fast_math_honor_infs",
                   &DebugOptions::xla_cpu_fast_math_honor_infs,
                   &DebugOptions::set_xla_cpu_fast_math_honor_infs)
      .def_prop_rw("xla_cpu_fast_math_honor_nans",
                   &DebugOptions::xla_cpu_fast_math_honor_nans,
                   &DebugOptions::set_xla_cpu_fast_math_honor_nans)
      .def_prop_rw("xla_cpu_fast_math_honor_division",
                   &DebugOptions::xla_cpu_fast_math_honor_division,
                   &DebugOptions::set_xla_cpu_fast_math_honor_division)
      .def_prop_rw("xla_cpu_fast_math_honor_functions",
                   &DebugOptions::xla_cpu_fast_math_honor_functions,
                   &DebugOptions::set_xla_cpu_fast_math_honor_functions)
      .def_prop_rw("xla_detailed_logging", &DebugOptions::xla_detailed_logging,
                   &DebugOptions::set_xla_detailed_logging)
      .def_prop_rw("xla_enable_dumping", &DebugOptions::xla_enable_dumping,
                   &DebugOptions::set_xla_enable_dumping)
      .def_prop_rw("xla_gpu_enable_fast_min_max",
                   &DebugOptions::xla_gpu_enable_fast_min_max,
                   &DebugOptions::set_xla_gpu_enable_fast_min_max)
      .def_prop_rw("xla_gpu_dump_autotune_results_to",
                   &DebugOptions::xla_gpu_dump_autotune_results_to,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_dump_autotune_results_to(value);
                   })
      .def_prop_rw("xla_gpu_load_autotune_results_from",
                   &DebugOptions::xla_gpu_load_autotune_results_from,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_load_autotune_results_from(value);
                   })
      .def_prop_rw("xla_gpu_cuda_data_dir",
                   &DebugOptions::xla_gpu_cuda_data_dir,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_cuda_data_dir(value);
                   })
      .def_prop_rw("xla_llvm_disable_expensive_passes",
                   &DebugOptions::xla_llvm_disable_expensive_passes,
                   &DebugOptions::set_xla_llvm_disable_expensive_passes)
      .def_prop_rw(
          "xla_disable_hlo_passes",
          [](DebugOptions* self) {
            return absl::StrJoin(self->xla_disable_hlo_passes(), ",");
          },
          [](DebugOptions* self, std::string value) {
            self->clear_xla_disable_hlo_passes();
            for (const auto& passname :
                 std::vector<std::string>(absl::StrSplit(value, ','))) {
              self->add_xla_disable_hlo_passes(passname);
            }
          })
      .def_prop_rw(
          "xla_enable_hlo_passes_only",
          [](DebugOptions* self) {
            return absl::StrJoin(self->xla_enable_hlo_passes_only(), ",");
          },
          [](DebugOptions* self, std::string value) {
            self->clear_xla_enable_hlo_passes_only();
            for (const auto& passname :
                 std::vector<std::string>(absl::StrSplit(value, ','))) {
              self->add_xla_enable_hlo_passes_only(passname);
            }
          })
      .def_prop_rw("xla_test_all_input_layouts",
                   &DebugOptions::xla_test_all_input_layouts,
                   &DebugOptions::set_xla_test_all_input_layouts)
      .def_prop_rw("xla_force_host_platform_device_count",
                   &DebugOptions::xla_force_host_platform_device_count,
                   &DebugOptions::set_xla_force_host_platform_device_count)
      .def_prop_rw("xla_dump_to", &DebugOptions::xla_dump_to,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_to(value);
                   })
      .def_prop_rw("xla_dump_hlo_module_re",
                   &DebugOptions::xla_dump_hlo_module_re,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_hlo_module_re(value);
                   })
      .def_prop_rw("xla_dump_hlo_pass_re", &DebugOptions::xla_dump_hlo_pass_re,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_hlo_pass_re(value);
                   })
      .def_prop_rw("xla_dump_hlo_as_text", &DebugOptions::xla_dump_hlo_as_text,
                   &DebugOptions::set_xla_dump_hlo_as_text)
      .def_prop_rw("xla_dump_hlo_as_proto",
                   &DebugOptions::xla_dump_hlo_as_proto,
                   &DebugOptions::set_xla_dump_hlo_as_proto)
      .def_prop_rw("xla_dump_hlo_as_dot", &DebugOptions::xla_dump_hlo_as_dot,
                   &DebugOptions::set_xla_dump_hlo_as_dot)
      .def_prop_rw("xla_dump_hlo_as_url", &DebugOptions::xla_dump_hlo_as_url,
                   &DebugOptions::set_xla_dump_hlo_as_url)
      .def_prop_rw("xla_dump_hlo_as_html", &DebugOptions::xla_dump_hlo_as_html,
                   &DebugOptions::set_xla_dump_hlo_as_html)
      .def_prop_rw("xla_dump_fusion_visualization",
                   &DebugOptions::xla_dump_fusion_visualization,
                   &DebugOptions::set_xla_dump_fusion_visualization)
      .def_prop_rw("xla_dump_hlo_snapshots",
                   &DebugOptions::xla_dump_hlo_snapshots,
                   &DebugOptions::set_xla_dump_hlo_snapshots)
      .def_prop_rw("xla_dump_max_hlo_modules",
                   &DebugOptions::xla_dump_max_hlo_modules,
                   &DebugOptions::set_xla_dump_max_hlo_modules)
      .def_prop_rw("xla_dump_module_metadata",
                   &DebugOptions::xla_dump_module_metadata,
                   &DebugOptions::set_xla_dump_module_metadata)
      .def_prop_rw("xla_dump_compress_protos",
                   &DebugOptions::xla_dump_compress_protos,
                   &DebugOptions::set_xla_dump_compress_protos)
      .def_prop_rw("xla_dump_hlo_as_long_text",
                   &DebugOptions::xla_dump_hlo_as_long_text,
                   &DebugOptions::set_xla_dump_hlo_as_long_text)
      .def_prop_rw("xla_dump_disable_metadata",
                   &DebugOptions::xla_dump_disable_metadata,
                   &DebugOptions::set_xla_dump_disable_metadata)
      .def_prop_rw("xla_dump_hlo_pipeline_re",
                   &DebugOptions::xla_dump_hlo_pipeline_re,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_dump_hlo_pipeline_re(value);
                   })
      .def_prop_rw("xla_gpu_dump_autotune_logs_to",
                   &DebugOptions::xla_gpu_dump_autotune_logs_to,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_dump_autotune_logs_to(value);
                   })
      .def_prop_rw("xla_gpu_kernel_cache_file",
                   &DebugOptions::xla_gpu_kernel_cache_file,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_kernel_cache_file(value);
                   })
      .def_prop_rw("xla_gpu_per_fusion_autotune_cache_dir",
                   &DebugOptions::xla_gpu_per_fusion_autotune_cache_dir,
                   [](DebugOptions* self, std::string value) {
                     self->set_xla_gpu_per_fusion_autotune_cache_dir(value);
                   })
      .def_prop_rw("xla_gpu_experimental_autotune_cache_mode",
                   &DebugOptions::xla_gpu_experimental_autotune_cache_mode,
                   &DebugOptions::set_xla_gpu_experimental_autotune_cache_mode);

  nb::class_<ExecutableBuildOptions>(m, "ExecutableBuildOptions")
      .def(nb::init<>())
      .def("__repr__", &ExecutableBuildOptions::ToString)
      .def_prop_rw(
          "fdo_profile",
          [](const ExecutableBuildOptions& options) {
            return nb::bytes(options.fdo_profile().data(),
                             options.fdo_profile().size());
          },
          [](ExecutableBuildOptions& options, nb::bytes fdo_profile) {
            options.set_fdo_profile(
                std::string(fdo_profile.c_str(), fdo_profile.size()));
          })
      .def_prop_rw(
          "result_layout",
          [](const ExecutableBuildOptions& options) -> std::optional<Shape> {
            return options.result_layout()
                       ? std::optional<Shape>(*options.result_layout())
                       : std::nullopt;
          },
          &ExecutableBuildOptions::set_result_layout)
      .def_prop_rw("num_replicas", &ExecutableBuildOptions::num_replicas,
                   &ExecutableBuildOptions::set_num_replicas)
      .def_prop_rw("num_partitions", &ExecutableBuildOptions::num_partitions,
                   &ExecutableBuildOptions::set_num_partitions)
      .def_prop_ro("debug_options",
                   &ExecutableBuildOptions::mutable_debug_options,
                   nb::rv_policy::reference, nb::keep_alive<1, 0>())
      .def_prop_rw(
          "device_assignment",
          [](const ExecutableBuildOptions& options)
              -> std::optional<DeviceAssignment> {
            return options.has_device_assignment()
                       ? std::optional<DeviceAssignment>(
                             options.device_assignment())
                       : std::nullopt;
          },
          &ExecutableBuildOptions::set_device_assignment)
      .def("compilation_environments_from_serialized_proto",
           [](ExecutableBuildOptions& options,
              const nb::bytes& serialized_proto) {
             xla::CompilationEnvironmentsProto env_proto;
             env_proto.ParseFromArray(serialized_proto.c_str(),
                                      serialized_proto.size());
             auto comp_envs = xla::ValueOrThrow(
                 xla::CompilationEnvironments::CreateFromProto(env_proto));
             *options.mutable_comp_envs() = std::move(*comp_envs);
           })
      .def_prop_rw("exec_time_optimization_effort",
                   &ExecutableBuildOptions::exec_time_optimization_effort,
                   &ExecutableBuildOptions::set_exec_time_optimization_effort)
      .def_prop_rw("memory_fitting_effort",
                   &ExecutableBuildOptions::memory_fitting_effort,
                   &ExecutableBuildOptions::set_memory_fitting_effort)
      .def_prop_rw(
          "optimization_level",
          [](ExecutableBuildOptions& options) {
            return static_cast<int>(options.optimization_level());
          },
          [](ExecutableBuildOptions& options, int value) {
            options.set_optimization_level(
                static_cast<xla::ExecutionOptions::EffortLevel>(value));
          })
      .def_prop_rw(
          "memory_fitting_level",
          [](ExecutableBuildOptions& options) {
            return static_cast<int>(options.memory_fitting_level());
          },
          [](ExecutableBuildOptions& options, int value) {
            options.set_memory_fitting_level(
                static_cast<xla::ExecutionOptions::EffortLevel>(value));
          })
      .def_prop_rw("use_spmd_partitioning",
                   &ExecutableBuildOptions::use_spmd_partitioning,
                   &ExecutableBuildOptions::set_use_spmd_partitioning)
      .def_prop_rw("use_auto_spmd_partitioning",
                   &ExecutableBuildOptions::use_auto_spmd_partitioning,
                   &ExecutableBuildOptions::set_use_auto_spmd_partitioning)
      .def_prop_rw(
          "auto_spmd_partitioning_mesh_shape",
          &ExecutableBuildOptions::auto_spmd_partitioning_mesh_shape,
          &ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_shape)
      .def_prop_rw("auto_spmd_partitioning_mesh_ids",
                   &ExecutableBuildOptions::auto_spmd_partitioning_mesh_ids,
                   &ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_ids)
      .def_prop_rw(
          "allow_spmd_sharding_propagation_to_parameters",
          [](const ExecutableBuildOptions& options) -> std::vector<bool> {
            return std::vector<bool>(
                options.allow_spmd_sharding_propagation_to_parameters().begin(),
                options.allow_spmd_sharding_propagation_to_parameters().end());
          },
          [](ExecutableBuildOptions& options, std::vector<bool> values) {
            absl::InlinedVector<bool, 1> v(values.begin(), values.end());
            options.set_allow_spmd_sharding_propagation_to_parameters(v);
          })
      .def_prop_rw(
          "allow_spmd_sharding_propagation_to_output",
          [](const ExecutableBuildOptions& options) -> std::vector<bool> {
            return std::vector<bool>(
                options.allow_spmd_sharding_propagation_to_output().begin(),
                options.allow_spmd_sharding_propagation_to_output().end());
          },
          [](ExecutableBuildOptions& options, std::vector<bool> values) {
            absl::InlinedVector<bool, 1> v(values.begin(), values.end());
            options.set_allow_spmd_sharding_propagation_to_output(v);
          })
      .def_prop_rw("use_shardy_partitioner",
                   &ExecutableBuildOptions::use_shardy_partitioner,
                   &ExecutableBuildOptions::set_use_shardy_partitioner);
}

}  // namespace xla
