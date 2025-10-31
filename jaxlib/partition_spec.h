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

class PartitionSpec {
 public:
  PartitionSpec(nanobind::tuple partitions, nanobind::frozenset unreduced,
                nanobind::frozenset reduced);

  nanobind::tuple partitions() const { return partitions_; }
  nanobind::frozenset unreduced() const { return unreduced_; }
  nanobind::frozenset reduced() const { return reduced_; }

  bool operator==(const PartitionSpec& other) const;

  bool Eq(const nanobind::object& other) const;  // Python __eq__
  Py_hash_t Hash() const;  // Python __hash__

  static void Register(nanobind::module_& m);

 private:
  nanobind::tuple partitions_;
  nanobind::frozenset unreduced_;
  nanobind::frozenset reduced_;

  static nanobind::object* unconstrained_singleton_;
};

}  // namespace jax

#endif  // JAX_JAXLIB_PARTITION_SPEC_H_
