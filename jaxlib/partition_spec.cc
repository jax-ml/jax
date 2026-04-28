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

#include "jaxlib/partition_spec.h"

#include <cstddef>
#include <string>

#include "absl/strings/str_format.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace jax {

namespace {

nb::object& pspec_unconstrained = *new nb::object();

bool IsTrue(nb::handle x) {
  int ret = PyObject_IsTrue(x.ptr());
  if (ret == -1) {
    throw nb::python_error();
  }
  return static_cast<bool>(ret);
}

nb::object CanonicalizePartition(nb::object partition) {
  if (!IsTrue(partition)) {
    return nb::none();
  }
  if (partition.is(pspec_unconstrained)) {
    return pspec_unconstrained;
  }
  bool is_tuple = nb::isinstance<nb::tuple>(partition);
  if (is_tuple || nb::isinstance<nb::list>(partition)) {
    for (nb::handle p : partition) {
      if (nb::isinstance<nb::tuple>(p) || nb::isinstance<nb::list>(p)) {
        throw nb::value_error(
            absl::StrFormat(
                "A tuple inside PartitionSpec cannot contain a "
                "nested tuple. Got partition: %s and the nested tuple: %s",
                nb::cast<std::string>(nb::str(partition)),
                nb::cast<std::string>(nb::str(p)))
                .c_str());
      }
    }
    if (nb::len(partition) == 1) {
      return partition[0];
    }
    if (!is_tuple) {
      return nb::tuple(partition);
    }
    return partition;
  }
  return partition;
}

void SetPspecUnconstrained(nb::object t) { pspec_unconstrained = t; }

}  // namespace

void RegisterPartitionSpec(nb::module_& m) {
  m.def("set_pspec_unconstrained", &SetPspecUnconstrained);

  m.def(
      "canonicalize_partition",
      [](nb::object partition) { return CanonicalizePartition(partition); },
      nb::arg("partition").none());

  m.def(
      "canonicalize_partitions",
      [](nb::tuple partitions_arg) {
        nb::tuple partitions =
            nb::steal<nb::tuple>(PyTuple_New(partitions_arg.size()));
        for (size_t i = 0; i < partitions_arg.size(); ++i) {
          PyTuple_SET_ITEM(
              partitions.ptr(), i,
              CanonicalizePartition(partitions_arg[i]).release().ptr());
        }
        return partitions;
      },
      nb::arg("partitions"));
}

}  // namespace jax
