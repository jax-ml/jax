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

#include "jaxlib/py_mpmd_loaded_executable.h"

#include <stdbool.h>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/jax_jit.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_user_context.h"
#include "jaxlib/py_values.h"
#include "jaxlib/pytree.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"  // IWYU pragma: keep
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable.h"
#include "xla/python/ifrt/ir/program_memory_tracer.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep
#include "xla/python/nb_numpy.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace jax {

namespace mpmd {

namespace {

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> UnwrapArrays(
    nb::sequence args) {
  std::vector<xla::ifrt::ArrayRef> result;
  result.reserve(nb::len(args));
  for (nb::handle arg : args) {
    jax::PyArray array = nb::cast<jax::PyArray>(arg);
    xla::ifrt::Array* ifrt_array = array.ifrt_array();
    if (ifrt_array == nullptr) {
      return xla::InvalidArgument("Array deleted or donated");
    }
    result.push_back(tsl::FormRef(ifrt_array));
  }
  return result;
}
}  // namespace

std::pair<std::shared_ptr<MpmdExecutableFastpathData>, bool>
MpmdExecutableFastpathCache::GetOrInsertIfAbsent(
    const jax::CallSignature& call_signature,
    jax::PyTreeRegistry* pytree_registry) {
  absl::MutexLock lock(mutex_);
  if (cache_.has_value()) {
    if (cache_->call_signature == call_signature) {
      return {cache_->data, false};
    }
    return {nullptr, false};
  }

  std::shared_ptr<MpmdExecutableFastpathData> data =
      std::make_shared<MpmdExecutableFastpathData>(pytree_registry);
  cache_.emplace(CacheEntry{call_signature, data});
  return {data, true};
}

absl::StatusOr<nb::list> PyMpmdLoadedExecutable::Execute(
    nb::sequence args) {
  TF_ASSIGN_OR_RETURN(std::vector<xla::ifrt::ArrayRef> ifrt_args,
                      UnwrapArrays(args));
  xla::ifrt::ExecuteOptions execute_options;
  execute_options.execution_stream_id = GetExecutionStreamId();

  xla::ifrt::UserContextScope user_context_scope(jax::PyUserContext::Create());
  xla::ifrt::LoadedExecutable::ExecuteResult result;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        result, ifrt_loaded_executable_->Execute(absl::MakeSpan(ifrt_args),
                                                 std::move(execute_options),
                                                 /*devices=*/std::nullopt));
  }
  if (result.outputs.size() != out_avals_.size()) {
    return xla::Internal(
        "IFRT LoadedExecutable returned a different number of outputs: %d "
        "vs. %d",
        result.outputs.size(), out_avals_.size());
  }
  nb::list results;
  for (int i = 0; i < result.outputs.size(); ++i) {
    nb::object out = jax::PyArray(
        nb::borrow<nb::object>(out_avals_[i].ptr()), /*weak_type=*/false,
        nb::borrow<xla::nb_dtype>(out_dtypes_[i].ptr()), out_shapes_[i],
        nb::borrow<nb::object>(out_shardings_[i].ptr()), backend_,
        std::move(result.outputs[i]),
        /*committed=*/true, /*skip_checks=*/true);
    results.append(std::move(out));
  }
  return results;
}

absl::StatusOr<nb::object> PyMpmdLoadedExecutable::ExecuteFastpath(
    nb::sequence args, nb::dict kwargs) {
  jax::CallSignature call_signature;
  absl::InlinedVector<nb::object, 2> flat_args;
  ParseArguments(args, kwargs, call_signature.arg_signature, flat_args);

  // Check that all arguments are of type `jax::PyArray`.
  for (int i = 0; i < flat_args.size(); ++i) {
    nb::object arg = flat_args[i];
    if (arg.type().ptr() != jax::PyArray::type().ptr()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Got input ", i, " of type ",
                       nb::cast<std::string>(nb::inst_name(arg)),
                       " instead of `jax::PyArray`"));
    }
  }

  TF_RETURN_IF_ERROR(ComputeCallSignature(flat_args, call_signature));

  std::pair<std::shared_ptr<MpmdExecutableFastpathData>, bool>
      cache_data_and_inserted =
          cache_.GetOrInsertIfAbsent(call_signature, pytree_registry_.get());
  std::shared_ptr<MpmdExecutableFastpathData> cache_data =
      cache_data_and_inserted.first;
  bool inserted = cache_data_and_inserted.second;
  if (!cache_data) {
    return absl::InvalidArgumentError(
        "Input signature does not match the signature of the compiled "
        "function.");
  }

  if (inserted) {
    // Cache miss. Execute the cache miss function and populate the cache.
    nb::object out_and_fastpath_data;
    nb::tuple out_tuple;
    try {
      out_and_fastpath_data = cache_miss_(*args, **kwargs);
      if (!out_and_fastpath_data.ptr()) {
        return absl::InternalError("Failure in cache miss callback.");
      }

      out_tuple = nb::cast<nb::tuple>(out_and_fastpath_data);
      PopulateCache(*cache_data, nb::cast<nb::tuple>(out_tuple));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to execute cache miss callback: " << e.what();
      throw;
    }
    return nb::borrow<nb::object>(out_tuple[0].ptr());
  }

  // Cache hit. Execute the function and return the unflattened output.
  // Filter out the arguments that are not required by the function.
  std::vector<bool> kept_var_bitvec = cache_data->kept_var_bitvec;
  nb::list kept_args;
  for (int i = 0; i < flat_args.size(); ++i) {
    if (kept_var_bitvec[i]) {
      kept_args.append(std::move(flat_args[i]));
    }
  }

  TF_ASSIGN_OR_RETURN(
      nb::list result, Execute(nb::cast<nb::sequence>(kept_args)));
  std::vector<nb::object> result_list;
  result_list.reserve(result.size());
  for (nb::handle result_item : result) {
    result_list.push_back(nb::cast(std::move(result_item)));
  }
  jax::PyTreeDef out_pytree_def = cache_data->out_pytree_def;
  return out_pytree_def.Unflatten(result_list);
}

// Returns a mapping between atom program name and compiled memory stats.
absl::StatusOr<absl::flat_hash_map<std::string, xla::CompiledMemoryStats>>
PyMpmdLoadedExecutable::GetMpmdCompiledMemoryStats() const {
  return ifrt_loaded_executable_->GetMpmdCompiledMemoryStats();
}

absl::StatusOr<xla::ifrt::IfrtIrProgramMemoryStats>
PyMpmdLoadedExecutable::GetIfrtIrProgramMemoryStats() const {
  if (auto* exec_ptr = llvm::dyn_cast<xla::ifrt::IfrtIrLoadedExecutable>(
          ifrt_loaded_executable_.get())) {
    return exec_ptr->GetIfrtIrProgramMemoryStats();
  } else {
    return absl::UnimplementedError(
        "`GetIfrtIrProgramMemoryStats` only supported on "
        "`MpmdLoadedExecutable`.");
  }
}

absl::StatusOr<std::string>
PyMpmdLoadedExecutable::GetIfrtIrProgramXprofUrl() const {
  if (auto* exec_ptr = llvm::dyn_cast<xla::ifrt::IfrtIrLoadedExecutable>(
          ifrt_loaded_executable_.get())) {
    return exec_ptr->GetIfrtIrProgramXprofUrl();
  } else {
    return absl::UnimplementedError(
        "`GetIfrtIrProgramXprofUrl` only supported on "
        "`MpmdLoadedExecutable`.");
  }
}

// Returns a mapping between atom program name and map of cost properties.
absl::StatusOr<absl::flat_hash_map<std::string, xla::ifrt::AttributeMap>>
PyMpmdLoadedExecutable::GetMpmdCostAnalysis() {
  return ifrt_loaded_executable_->GetMpmdCostAnalysis();
}

absl::StatusOr<std::vector<xla::OpSharding>>
PyMpmdLoadedExecutable::GetParameterShardings() {
  auto* exec_ptr = llvm::dyn_cast<xla::ifrt::IfrtIrLoadedExecutable>(
      ifrt_loaded_executable_.get());
  std::optional<std::vector<xla::OpSharding>> param_shardings;
  if (exec_ptr != nullptr) {
    param_shardings = exec_ptr->GetParameterShardings();
  } else {
    param_shardings = ifrt_loaded_executable_->GetParameterShardings();
  }

  if (param_shardings.has_value()) {
    return param_shardings.value();
  } else {
    return absl::NotFoundError(absl::StrCat(
        "No parameter shardings found for MPMD executable: ",
        exec_ptr ? exec_ptr->name() : ifrt_loaded_executable_->name()));
  }
}

absl::StatusOr<absl::flat_hash_map<
    std::string, std::vector<std::shared_ptr<xla::HloModule>>>>
PyMpmdLoadedExecutable::GetHloModules() {
  return ifrt_loaded_executable_->GetMpmdHloModules();
}

absl::StatusOr<std::vector<xla::OpSharding>>
PyMpmdLoadedExecutable::GetOutputShardings() {
  auto* exec_ptr = llvm::dyn_cast<xla::ifrt::IfrtIrLoadedExecutable>(
      ifrt_loaded_executable_.get());
  std::optional<std::vector<xla::OpSharding>> output_shardings;
  if (exec_ptr != nullptr) {
    output_shardings = exec_ptr->GetOutputShardings();
  } else {
    output_shardings = ifrt_loaded_executable_->GetOutputShardings();
  }
  if (output_shardings.has_value()) {
    return output_shardings.value();
  } else {
    return absl::NotFoundError(absl::StrCat(
        "No output shardings found for MPMD executable: ",
        exec_ptr ? exec_ptr->name() : ifrt_loaded_executable_->name()));
  }
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
PyMpmdLoadedExecutable::GetParameterLayouts() {
  if (auto* exec_ptr = llvm::dyn_cast<xla::ifrt::IfrtIrLoadedExecutable>(
          ifrt_loaded_executable_.get())) {
    return exec_ptr->GetParameterLayouts();
  } else {
    return ifrt_loaded_executable_->GetParameterLayouts();
  }
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
PyMpmdLoadedExecutable::GetOutputLayouts() {
  if (auto* exec_ptr = llvm::dyn_cast<xla::ifrt::IfrtIrLoadedExecutable>(
          ifrt_loaded_executable_.get())) {
    return exec_ptr->GetOutputLayouts();
  } else {
    return ifrt_loaded_executable_->GetOutputLayouts();
  }
}

void PyMpmdLoadedExecutable::ParseArguments(
    nb::sequence args, nb::dict kwargs, jax::ArgumentSignature& arg_signature,
    absl::InlinedVector<nb::object, 2>& flat_args) {
  size_t num_positional_args = nb::len(args);
  size_t num_keyword_args = kwargs ? kwargs.size() : 0;
  flat_args.reserve(num_positional_args + num_keyword_args);

  // Positional arguments.
  for (size_t i = 0; i < num_positional_args; ++i) {
    arg_signature.dynamic_arg_treedefs.emplace_back(pytree_registry_);
    jax::PyTreeDef& pytree_def = arg_signature.dynamic_arg_treedefs.back();
    pytree_def.Flatten(nb::handle(args[i].ptr()), flat_args);
  }
  // Keyword arguments.
  for (const auto& [key, value] : kwargs) {
    arg_signature.dynamic_arg_treedefs.emplace_back(pytree_registry_);
    jax::PyTreeDef& pytree_def = arg_signature.dynamic_arg_treedefs.back();
    pytree_def.Flatten(nb::handle(value.ptr()), flat_args);
  }
}

absl::Status PyMpmdLoadedExecutable::ComputeCallSignature(
    absl::InlinedVector<nb::object, 2>& flat_args,
    jax::CallSignature& call_signature) {
  bool jax_enable_x64 = jax::GetEnableX64();

  auto& dynamic_arg_signatures = call_signature.dynamic_arg_signatures;
  dynamic_arg_signatures.reserve(flat_args.size());
  auto& dynamic_arg_shardings = call_signature.dynamic_arg_shardings;
  dynamic_arg_shardings.reserve(flat_args.size());
  auto& dynamic_arg_layouts = call_signature.dynamic_arg_layouts;
  dynamic_arg_layouts.reserve(flat_args.size());

  for (nb::handle arg : flat_args) {
    TF_ASSIGN_OR_RETURN(jax::PyArgSignature arg_signature,
                        jax::PyArgSignatureOfValue(arg, jax_enable_x64));
    call_signature.dynamic_arg_signatures.push_back(std::move(arg_signature));

    jax::PyArray py_array = nb::borrow<jax::PyArray>(arg);
    call_signature.dynamic_arg_shardings.push_back(py_array.sharding());
    absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> layout =
        py_array.layout();
    if (absl::IsUnimplemented(layout.status())) {
      call_signature.dynamic_arg_layouts.push_back(nullptr);
    } else {
      call_signature.dynamic_arg_layouts.push_back(*std::move(layout));
    }
    call_signature.committed_args.push_back(py_array.committed());
  }
  call_signature.cached_hash = absl::HashOf(call_signature);

  return absl::OkStatus();
}

void PyMpmdLoadedExecutable::PopulateCache(
    MpmdExecutableFastpathData& cache_data, nb::tuple out_and_fastpath_data) {
  DCHECK_GE(out_and_fastpath_data.size(), 2);

  nb::tuple fastpath_data = nb::cast<nb::tuple>(out_and_fastpath_data[1]);
  cache_data.out_pytree_def = nb::cast<jax::PyTreeDef>(
      nb::handle(fastpath_data.attr("out_pytree_def").ptr()));

  nb::list kept_var_bitvec = fastpath_data.attr("kept_var_bitvec");
  cache_data.kept_var_bitvec.reserve(nb::len(kept_var_bitvec));
  for (nb::handle k : kept_var_bitvec) {
    cache_data.kept_var_bitvec.push_back(nb::cast<bool>(k));
  }
}

void PyMpmdLoadedExecutable::SetupFastpath(nb::callable cache_miss,
                                                nb::object pytree_registry) {
  cache_miss_ = std::move(cache_miss);
  pytree_registry_ = nb::cast<jax::nb_class_ptr<jax::PyTreeRegistry>>(
      nb::handle(pytree_registry.ptr()));
}

}  // namespace mpmd
}  // namespace jax
