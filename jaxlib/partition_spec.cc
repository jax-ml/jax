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
#include <utility>

#include "absl/base/casts.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace jax {

namespace {

bool IsTrue(nb::handle x) {
  int ret = PyObject_IsTrue(x.ptr());
  if (ret == -1) {
    throw nb::python_error();
  }
  return static_cast<bool>(ret);
}

nb::object CanonicalizePartition(nb::object unconstrained_singleton,
                                 nb::object partition) {
  if (!IsTrue(partition)) {
    return nb::none();
  }
  if (partition.is(unconstrained_singleton)) {
    return unconstrained_singleton;
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

void CheckPartitionSpec(nb::tuple partitions, nb::frozenset unreduced,
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
  if (nb::len((unreduced & reduced)) != 0) {
    throw nb::value_error(
        absl::StrFormat("`unreduced` and `reduced` argument to PartitionSpec "
                        "cannot overlap. "
                        "Got unreduced: %s and reduced: %s",
                        nb::cast<std::string>(nb::str(unreduced)),
                        nb::cast<std::string>(nb::str(reduced)))
            .c_str());
  }
}

}  // namespace

PartitionSpec::PartitionSpec(nb::tuple partitions, nb::frozenset unreduced,
                             nb::frozenset reduced)
    : partitions_(std::move(partitions)),
      unreduced_(std::move(unreduced)),
      reduced_(std::move(reduced)) {}

Py_hash_t PartitionSpec::Hash() const {
  size_t h = absl::HashOf(nb::hash(partitions_), nb::hash(unreduced_),
                          nb::hash(reduced_));
  Py_hash_t s = absl::bit_cast<Py_hash_t>(h);  // Python hashes are signed.
  return s == -1 ? -2 : s;  // -1 must not be used as a Python hash value.
}

bool PartitionSpec::operator==(const PartitionSpec& other) const {
  return partitions().equal(other.partitions()) &&
         unreduced().equal(other.unreduced()) &&
         reduced().equal(other.reduced());
}

bool PartitionSpec::Eq(const nb::object& other) const {
  if (!other.ptr() || other.is_none()) {
    return false;
  }
  PartitionSpec* other_spec;
  if (nb::try_cast<PartitionSpec*>(other, other_spec)) {
    return *this == *other_spec;
  }
  nb::tuple other_tuple;
  if (nb::try_cast<nb::tuple>(other, other_tuple)) {
    if (unreduced().size() > 0 || reduced().size() > 0 ||
        partitions().size() != other_tuple.size()) {
      return false;
    }
    for (size_t i = 0; i < partitions().size(); ++i) {
      if (!partitions()[i].equal(CanonicalizePartition(
              *unconstrained_singleton_, other_tuple[i]))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

nb::object* PartitionSpec::unconstrained_singleton_ = nullptr;

void PartitionSpec::Register(nb::module_& m) {
  nb::class_<UnconstrainedSingleton>(m, "UnconstrainedSingleton")
      .def("__repr__", [](nb::handle self) { return nb::str("UNCONSTRAINED"); })
      .def("__reduce__",
           [](nb::handle self) { return nb::str("UNCONSTRAINED_PARTITION"); });

  unconstrained_singleton_ = new nb::object(nb::cast(UnconstrainedSingleton()));
  m.attr("UNCONSTRAINED_PARTITION") = *unconstrained_singleton_;

  m.def("canonicalize_partition", [](nb::object partition) {
    return CanonicalizePartition(*unconstrained_singleton_, partition);
  });

  auto partition_spec =
      nb::class_<PartitionSpec>(m, "PartitionSpec", nb::is_final(),
                                nb::sig("class PartitionSpec(typing.Any)"))
          .def(
              "__init__",
              [](PartitionSpec* self, nb::args partition_args,
                 nb::object unreduced_arg, nb::object reduced_arg) {
                nb::tuple partitions =
                    nb::steal<nb::tuple>(PyTuple_New(partition_args.size()));
                for (size_t i = 0; i < partition_args.size(); ++i) {
                  PyTuple_SET_ITEM(partitions.ptr(), i,
                                   CanonicalizePartition(
                                       *PartitionSpec::unconstrained_singleton_,
                                       partition_args[i])
                                       .release()
                                       .ptr());
                }
                nb::frozenset unreduced;
                nb::frozenset reduced;
                if (!PyAnySet_Check(unreduced_arg.ptr())) {
                  throw nb::type_error(
                      absl::StrFormat(
                          "unreduced argument of PartitionSpec should "
                          "of type `frozenset` or `set`. Got type %s",
                          nb::cast<std::string>(nb::repr(unreduced_arg.type())))
                          .c_str());
                }
                if (!PyAnySet_Check(reduced_arg.ptr())) {
                  throw nb::type_error(
                      absl::StrFormat(
                          "reduced argument of PartitionSpec should "
                          "of type `frozenset` or `set`. Got type %s",
                          nb::cast<std::string>(nb::repr(reduced_arg.type())))
                          .c_str());
                }
                unreduced = nb::frozenset(unreduced_arg);
                reduced = nb::frozenset(reduced_arg);
                CheckPartitionSpec(partitions, unreduced, reduced);
                new (self)
                    PartitionSpec(std::move(partitions), std::move(unreduced),
                                  std::move(reduced));
              },
              nb::arg("partitions"), nb::arg("unreduced") = nb::frozenset(),
              nb::arg("reduced") = nb::frozenset())
          .def_prop_ro("_partitions", &PartitionSpec::partitions)
          .def_prop_ro("unreduced", &PartitionSpec::unreduced)
          .def_prop_ro("reduced", &PartitionSpec::reduced)
          .def("__eq__", &PartitionSpec::Eq, nb::arg(), nb::is_operator())
          .def("__hash__", &PartitionSpec::Hash);
}

}  // namespace jax
