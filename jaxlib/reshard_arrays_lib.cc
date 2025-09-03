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

#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/to_ifrt_sharding.h"
#include "jaxlib/traceback.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/types.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace jax {

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

  nb_class_ptr<PyClient> backend;
  std::vector<xla::ifrt::ArrayRef> ifrt_arrays;
  std::vector<xla::ifrt::ArraySpec> ifrt_specs;
  ifrt_arrays.reserve(num_arrays);
  ifrt_specs.reserve(num_arrays);

  for (int i = 0; i < num_arrays; ++i) {
    jax::PyArray array = nb::cast<jax::PyArray>(py_arrays[i]);
    if (array.ifrt_array() == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input array ", i, " has been donated or deleted."));
    }
    if (array.py_client().get() == nullptr) {
      return absl::InternalError("Unexpected array with py_client as nullptr.");
    }
    if (backend.get() == nullptr) {
      backend = array.py_client();
    } else if (backend.get() != array.py_client().get()) {
      return absl::InvalidArgumentError(
          absl::StrCat("All arrays must come from the same client. Array ", i,
                       " has a different client than the first array."));
    }
    ifrt_arrays.push_back(tsl::FormRef(array.ifrt_array()));

    TF_ASSIGN_OR_RETURN(xla::ifrt::DType ifrt_dtype,
                        xla::DtypeToIfRtDType(array.dtype()));
    xla::ifrt::Shape ifrt_shape(array.shape());
    TF_ASSIGN_OR_RETURN(xla::ifrt::ShardingRef ifrt_sharding,
                        jax::GetIfrtHloSharding(out_shardings[i], ifrt_shape));
    ifrt_specs.push_back(xla::ifrt::ArraySpec{
        .dtype = std::move(ifrt_dtype),
        .shape = std::move(ifrt_shape),
        .sharding = std::move(ifrt_sharding),
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

  auto traceback = jax::Traceback::Get();
  nb::list result;
  for (int i = 0; i < num_arrays; ++i) {
    jax::PyArray new_py_array = jax::PyArray::MakeFromIfrtArrayAndSharding(
        backend, traceback, std::move(outputs[i]), out_shardings[i],
        /*weak_type=*/false,
        /*committed=*/true,
        /*skip_checks=*/true);
    result.append(std::move(new_py_array));
  }

  return result;
}

}  // namespace jax
