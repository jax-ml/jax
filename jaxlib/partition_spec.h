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

#ifndef JAX_JAXLIB_PARTITION_SPEC_H_
#define JAX_JAXLIB_PARTITION_SPEC_H_

#include <vector>

#include "nanobind/nanobind.h"

namespace jax {

struct UnconstrainedSingleton {};

class nb_frozenset : public nanobind::object {
  NB_OBJECT(nb_frozenset, object, "frozenset", PyFrozenSet_Check)
  nb_frozenset()
      : object(PyFrozenSet_New(nullptr), nanobind::detail::steal_t()) {}
  explicit nb_frozenset(handle h)
      : object(nb_frozenset_from_obj(h.ptr()), nanobind::detail::steal_t{}) {}
  size_t size() const { return (size_t)NB_SET_GET_SIZE(m_ptr); }
  template <typename T>
  bool contains(T&& key) const;

 private:
  static PyObject* nb_frozenset_from_obj(PyObject* o);
};

class PartitionSpec {
 public:
  PartitionSpec(nanobind::tuple partitions, nb_frozenset unreduced);

  nanobind::tuple partitions() const { return partitions_; }
  nb_frozenset unreduced() const { return unreduced_; }

  bool Eq(const nanobind::object& other) const;
  Py_ssize_t Hash() const;

  static void Register(nanobind::module_& m);

 private:
  nanobind::tuple partitions_;
  nb_frozenset unreduced_;

  static nanobind::object* unconstrained_singleton_;
};

}  // namespace jax

#endif  // JAX_JAXLIB_PARTITION_SPEC_H_
