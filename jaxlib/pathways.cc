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

#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_user_context.h"
#include "jaxlib/to_ifrt_sharding.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/types.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace jax {

namespace {

// Returns strides for the given `axis_sizes`.
absl::StatusOr<std::vector<int>> GetStrides(absl::Span<const int> axis_sizes) {
  if (axis_sizes.empty()) {
    return absl::InvalidArgumentError("`axis_sizes` must not be empty");
  }
  std::vector<int> strides;
  strides.reserve(axis_sizes.size());
  strides.push_back(1);
  for (int i = axis_sizes.size() - 1; i > 0; --i) {
    strides.push_back(axis_sizes[i] * strides.back());
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

// Populates `offsets` with the offsets to use to create continuous intervals
// for `RemapPlan`.
// `axis_sizes` represents the mesh axis sizes up to the mesh axis on which
// the arrays are split/concatenated.
// `current_entry` iterates over the mesh axis sizes to generate the offsets.
// `strides` are the strides of the mesh axes.
absl::Status PopulateSubmeshOffsets(absl::Span<const int> axis_sizes,
                                    absl::Span<int> current_entry,
                                    absl::Span<const int> strides,
                                    std::vector<int>& offsets) {
  int offset = 0;
  for (int idx = 0; idx < axis_sizes.size(); ++idx) {
    offset += strides[idx] * current_entry[idx];
  }
  offsets.push_back(offset);
  current_entry[current_entry.size() - 1] += 1;
  for (int idx = current_entry.size() - 1;
       idx > 0 && current_entry[idx] >= axis_sizes[idx]; --idx) {
    current_entry[idx] = 0;
    current_entry[idx - 1] += 1;
  }
  if (current_entry[0] < axis_sizes[0]) {
    return PopulateSubmeshOffsets(axis_sizes, current_entry, strides, offsets);
  } else {
    return absl::OkStatus();
  }
}

// If `backend` is nullptr, sets it to `array.py_client()`; otherwise checks
// that `backend` equals `array.py_client()`.
absl::Status PyClientFromPyArray(const PyArray& array,
                                 nb_class_ptr<PyClient>& backend) {
  if (array.py_client().get() == nullptr) {
    return absl::InternalError("Unexpected array with py_client as nullptr.");
  }
  if (backend.get() == nullptr) {
    backend = array.py_client();
  } else if (backend.get() != array.py_client().get()) {
    std::string old_description =
        absl::StrFormat("%p/%s/%s/%s/%s", backend.get(),
                        backend->platform_name(), backend->platform_version(),
                        backend->runtime_type(), backend->raw_platform_name());
    std::string new_description =
        absl::StrFormat("%p/%s/%s/%s/%s", array.py_client().get(),
                        array.py_client()->platform_name(),
                        array.py_client()->platform_version(),
                        array.py_client()->runtime_type(),
                        array.py_client()->raw_platform_name());
    return absl::InvalidArgumentError(absl::StrCat(
        "py_client mismatch: ", old_description, " vs ", new_description));
  }
  return absl::OkStatus();
}

namespace nb = ::nanobind;

// Runs `xla::ifrt::Client::ReshardArrays`.
absl::StatusOr<nb::list> ExperimentalReshardArrays(nb::sequence py_arrays,
                                                   nb::sequence out_shardings,
                                                   bool donate_input) {
  const int num_arrays = nb::len(py_arrays);

  if (nb::len(out_shardings) != num_arrays) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of out_shardings must match number of arrays: ",
                     nb::len(out_shardings), num_arrays));
  }

  if (num_arrays == 0) {
    return nb::list();
  }

  PyUserContextScope user_context_scope;
  nb_class_ptr<PyClient> backend;
  std::vector<xla::ifrt::ArrayRef> ifrt_arrays;
  std::vector<xla::ifrt::ArraySpec> ifrt_specs;
  ifrt_arrays.reserve(num_arrays);
  ifrt_specs.reserve(num_arrays);

  for (int i = 0; i < num_arrays; ++i) {
    PyArray array = nb::cast<PyArray>(py_arrays[i]);
    if (array.ifrt_array() == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input array ", i, " has been donated or deleted."));
    }
    TF_RETURN_IF_ERROR(PyClientFromPyArray(array, backend));
    ifrt_arrays.push_back(tsl::FormRef(array.ifrt_array()));

    TF_ASSIGN_OR_RETURN(xla::ifrt::DType ifrt_dtype,
                        xla::DtypeToIfRtDType(array.dtype()));
    xla::ifrt::Shape ifrt_shape(array.shape());
    TF_ASSIGN_OR_RETURN(xla::ifrt::ShardingRef ifrt_sharding,
                        GetIfrtHloSharding(out_shardings[i], ifrt_shape));
    ifrt_specs.push_back(xla::ifrt::ArraySpec{
        /*dtype=*/std::move(ifrt_dtype),
        /*shape=*/std::move(ifrt_shape),
        /*sharding=*/std::move(ifrt_sharding),
    });
  }

  const xla::ifrt::ArrayCopySemantics copy_semantics =
      donate_input ? xla::ifrt::ArrayCopySemantics::kDonateInput
                   : xla::ifrt::ArrayCopySemantics::kAlwaysCopy;

  std::vector<xla::ifrt::ArrayRef> outputs;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        outputs, backend->ifrt_client()->ReshardArrays(
                     absl::MakeSpan(ifrt_arrays), ifrt_specs, copy_semantics));
  }

  nb::list result;
  for (int i = 0; i < num_arrays; ++i) {
    PyArray new_py_array = PyArray::MakeFromIfrtArrayAndSharding(
        backend, std::move(outputs[i]), out_shardings[i],
        /*weak_type=*/false,
        /*committed=*/true,
        /*skip_checks=*/true);
    result.append(std::move(new_py_array));
  }

  return result;
}

absl::StatusOr<std::vector<std::vector<nb::object>>>
ExperimentalSplitByMeshAxis(
    nb::object py_arrays_py, absl::Span<const int> sharded_dim_idxs,
    absl::Span<const int> mesh_axis_sizes, int mesh_axis_idx,
    absl::Span<const int> mesh_axis_sections,
    absl::Span<const std::vector<nb::object>> submesh_shardings, bool donate) {
  // Using `nb_class_ptr<PyClient>` requires GIL.
  DCHECK(PyGILState_Check());

  auto py_arrays = nb::cast<std::vector<PyArray>>(py_arrays_py);
  if (py_arrays.empty()) {
    return std::vector<std::vector<nb::object>>();
  }

  if (sharded_dim_idxs.size() != py_arrays.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of sharded_dim_idxs must match number of arrays: ",
                     sharded_dim_idxs.size(), " vs ", py_arrays.size()));
  }
  if (submesh_shardings.size() != py_arrays.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of submesh_shardings must match number of arrays: ",
        submesh_shardings.size(), " vs ", py_arrays.size()));
  }

  int num_submeshes = submesh_shardings[0].size();
  if (mesh_axis_sections.size() != num_submeshes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of mesh_axis_sections must match number of submeshes: ",
        mesh_axis_sections.size(), " vs ", num_submeshes));
  }

  PyUserContextScope user_context_scope;
  // All input arrays are expected to use the same mesh.
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      GetIfrtDeviceList(py_arrays[0].sharding()));
  int num_devices = device_list->size();
  // The last entry in `mexh_axis_sections` contains the mesh axis size.
  int mesh_axis_size = mesh_axis_sections.back();
  if (num_devices % mesh_axis_size != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of devices must be divisible by the mesh axis size: ",
        num_devices, " vs ", mesh_axis_size));
  }

  xla::ifrt::RemapPlan remap_plan;
  remap_plan.mappings =
      std::make_shared<std::vector<xla::ifrt::RemapPlan::Mapping>>();
  auto& mappings = *remap_plan.mappings;

  TF_ASSIGN_OR_RETURN(std::vector<int> strides, GetStrides(mesh_axis_sizes));
  std::vector<int> submesh_offsets;
  if (mesh_axis_idx == 0) {
    submesh_offsets.push_back(0);
  } else {
    std::vector<int> current_entry(mesh_axis_idx, 0);
    TF_RETURN_IF_ERROR(PopulateSubmeshOffsets(
        mesh_axis_sizes.subspan(0, mesh_axis_idx),
        absl::MakeSpan(current_entry), strides, submesh_offsets));
  }

  nb_class_ptr<PyClient> backend;
  std::vector<xla::ifrt::ArrayRef> input_ifrt_arrays;
  input_ifrt_arrays.reserve(py_arrays.size());
  for (int array_idx = 0; array_idx < py_arrays.size(); ++array_idx) {
    TF_RETURN_IF_ERROR(PyClientFromPyArray(py_arrays[array_idx], backend));
    xla::ifrt::Array* array = py_arrays[array_idx].ifrt_array();
    if (array == nullptr) {
      return xla::InvalidArgument("Input array #%d has been donated or deleted",
                                  array_idx);
    }

    remap_plan.input_specs.push_back(
        xla::ifrt::ArraySpec{/*dtype=*/array->dtype(),
                             /*shape=*/array->shape(),
                             /*sharding=*/array->shared_ptr_sharding()});

    for (int submesh_idx = 0; submesh_idx < num_submeshes; ++submesh_idx) {
      auto& mapping = mappings.emplace_back();
      mapping.in_array = array_idx;
      mapping.out_array = remap_plan.output_specs.size();
      int submesh_axis_size = mesh_axis_sections[submesh_idx];
      int submesh_axis_start = 0;
      if (submesh_idx > 0) {
        submesh_axis_size -= mesh_axis_sections[submesh_idx - 1];
        submesh_axis_start = mesh_axis_sections[submesh_idx - 1];
      }
      int offset_to_array = 0;
      for (const auto& submesh_offset : submesh_offsets) {
        int num_contiguous_shards = submesh_axis_size * strides[mesh_axis_idx];
        int offset_from_array =
            submesh_offset + submesh_axis_start * strides[mesh_axis_idx];
        mapping.from.push_back(xla::ifrt::RemapPlan::Interval{
            offset_from_array, offset_from_array + num_contiguous_shards, 1});
        mapping.to.push_back(xla::ifrt::RemapPlan::Interval{
            offset_to_array, offset_to_array + num_contiguous_shards, 1});
        offset_to_array += num_contiguous_shards;
      }
      if (sharded_dim_idxs[array_idx] >= 0) {
        std::vector<int64_t> dims(array->shape().dims().begin(),
                                  array->shape().dims().end());
        dims[sharded_dim_idxs[array_idx]] = dims[sharded_dim_idxs[array_idx]] /
                                            mesh_axis_size * submesh_axis_size;
        xla::ifrt::Shape subshape = xla::ifrt::Shape(dims);
        TF_ASSIGN_OR_RETURN(
            auto ifrt_submesh_sharding,
            GetIfrtHloSharding(submesh_shardings[array_idx][submesh_idx],
                               subshape));
        remap_plan.output_specs.push_back(xla::ifrt::ArraySpec{
            /*dtype=*/array->dtype(),
            /*shape=*/std::move(subshape),
            /*sharding=*/std::move(ifrt_submesh_sharding)});
      } else {
        // The arrays is replicated, so its shape does not change.
        TF_ASSIGN_OR_RETURN(
            auto ifrt_submesh_sharding,
            GetIfrtHloSharding(submesh_shardings[array_idx][submesh_idx],
                               array->shape()));
        remap_plan.output_specs.push_back(xla::ifrt::ArraySpec{
            /*dtype=*/array->dtype(),
            /*shape=*/array->shape(),
            /*sharding=*/std::move(ifrt_submesh_sharding)});
      }
    }

    input_ifrt_arrays.push_back(FormRef(array));
  }

  DCHECK_OK(remap_plan.Validate());

  std::vector<xla::ifrt::ArrayRef> result_ifrt_arrays;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        result_ifrt_arrays,
        backend->ifrt_client()->RemapArrays(
            remap_plan, absl::MakeSpan(input_ifrt_arrays),
            donate ? xla::ifrt::ArrayCopySemantics::kDonateInput
                   : xla::ifrt::ArrayCopySemantics::kReuseInput));
  }

  DCHECK_EQ(result_ifrt_arrays.size(), py_arrays.size() * num_submeshes);

  // Wrap IFRT arrays as JAX arrays.
  std::vector<std::vector<nb::object>> py_results;
  int offset_in_results = 0;
  for (int array_idx = 0; array_idx < py_arrays.size(); ++array_idx) {
    auto& py_submesh_results = py_results.emplace_back();
    for (int submesh_idx = 0; submesh_idx < num_submeshes; ++submesh_idx) {
      PyArray new_py_array = PyArray::MakeFromIfrtArrayAndSharding(
          backend,
          std::move(result_ifrt_arrays[offset_in_results + submesh_idx]),
          submesh_shardings[array_idx][submesh_idx],
          py_arrays[array_idx].weak_type(),
          /*committed=*/true,
          /*skip_checks=*/true);
      py_submesh_results.push_back(new_py_array);
    }
    offset_in_results += num_submeshes;
  }

  return py_results;
}

}  // namespace

NB_MODULE(_pathways, m) {
  m.def("_transfer_to_shardings",
        xla::ValueOrThrowWrapper(ExperimentalReshardArrays), nb::arg("arrays"),
        nb::arg("out_shardings"), nb::arg("donate") = false);
  m.def("_split_by_mesh_axis",
        xla::ValueOrThrowWrapper(ExperimentalSplitByMeshAxis),
        nb::arg("arrays"), nb::arg("sharded_dim_idxs"),
        nb::arg("mesh_axis_sizes"), nb::arg("mesh_axis_idx"),
        nb::arg("mesh_axis_sections"), nb::arg("submesh_shardings"),
        nb::arg("donate"));
}

}  // namespace jax
