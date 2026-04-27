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

#include <string>

#include "absl/strings/str_format.h"
#include "jaxlib/partition_spec.h"
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

void CheckPartitionSpec(nb::object partitions, nb::frozenset unreduced,
                        nb::frozenset reduced) {
  if (unreduced.contains(nb::none())) {
    throw nb::value_error(
        "unreduced cannot contain None. All elements in unreduced should "
        "refer to the mesh axes.");
  }
  if (reduced.contains(nb::none())) {
    throw nb::value_error(
        "reduced cannot contain None. All elements in reduced should "
        "refer to the mesh axes.");
  }
  if (nb::len((unreduced & reduced)) != 0) {
    throw nb::value_error(
        absl::StrFormat("`unreduced` and `reduced` argument to PartitionSpec "
                        "cannot overlap. "
                        "Got unreduced: %s and reduced: %s",
                        nb::cast<std::string>(nb::str(unreduced)),
                        nb::cast<std::string>(nb::str(reduced)))
            .c_str());
  }
  auto check_overlap = [&](nb::handle partition) {
    if (unreduced.contains(partition)) {
      throw nb::value_error(
          absl::StrFormat(
              "partitions cannot overlap with unreduced axes passed to "
              "PartitionSpec. Got partitions: %s and unreduced axes: %s",
              nb::cast<std::string>(nb::str(partitions)),
              nb::cast<std::string>(nb::str(unreduced)))
              .c_str());
    }
    if (reduced.contains(partition)) {
      throw nb::value_error(
          absl::StrFormat(
              "partitions cannot overlap with reduced axes passed to "
              "PartitionSpec. Got partitions: %s and reduced axes: %s",
              nb::cast<std::string>(nb::str(partitions)),
              nb::cast<std::string>(nb::str(reduced)))
              .c_str());
    }
  };
  for (nb::handle partition : partitions) {
    if (nb::isinstance<nb::tuple>(partition)) {
      for (nb::handle p : partition) {
        check_overlap(p);
      }
    } else {
      check_overlap(partition);
    }
  }
}

nb::tuple PspecInit(nb::args partition_args, nb::object unreduced_arg,
                    nb::object reduced_arg) {
  nb::tuple partitions =
      nb::steal<nb::tuple>(PyTuple_New(partition_args.size()));
  for (size_t i = 0; i < partition_args.size(); ++i) {
    PyTuple_SET_ITEM(partitions.ptr(), i,
                     CanonicalizePartition(partition_args[i]).release().ptr());
  }
  nb::frozenset unreduced;
  nb::frozenset reduced;
  if (!PyAnySet_Check(unreduced_arg.ptr())) {
    throw nb::type_error(
        absl::StrFormat("unreduced argument of PartitionSpec should "
                        "of type `frozenset` or `set`. Got type %s",
                        nb::cast<std::string>(nb::repr(unreduced_arg.type())))
            .c_str());
  }
  if (!PyAnySet_Check(reduced_arg.ptr())) {
    throw nb::type_error(
        absl::StrFormat("reduced argument of PartitionSpec should "
                        "of type `frozenset` or `set`. Got type %s",
                        nb::cast<std::string>(nb::repr(reduced_arg.type())))
            .c_str());
  }
  unreduced = nb::frozenset(unreduced_arg);
  reduced = nb::frozenset(reduced_arg);
  CheckPartitionSpec(partitions, unreduced, reduced);
  return nb::make_tuple(partitions, unreduced, reduced);
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
      "pspec_init",
      [](nb::args partitions, nb::object unreduced, nb::object reduced) {
        return PspecInit(partitions, unreduced, reduced);
      },
      nb::arg("partitions"), nb::arg("unreduced"), nb::arg("reduced"));
}

}  // namespace jax
