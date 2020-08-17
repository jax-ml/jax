/* Copyright 2019 Google LLC

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

#include "jaxlib/pytree.h"
#include "include/pybind11/pybind11.h"

namespace jax {

namespace py = pybind11;

PYBIND11_MODULE(pytree, m) {
  m.def("flatten", &PyTreeDef::Flatten);
  m.def("tuple", &PyTreeDef::Tuple);
  m.def("all_leaves", &PyTreeDef::AllLeaves);

  py::class_<PyTreeDef>(m, "PyTreeDef")
      .def("unflatten", &PyTreeDef::Unflatten)
      .def("flatten_up_to", &PyTreeDef::FlattenUpTo)
      .def("compose", &PyTreeDef::Compose)
      .def("walk", &PyTreeDef::Walk)
      .def("from_iterable_tree", &PyTreeDef::FromIterableTree)
      .def("children", &PyTreeDef::Children)
      .def_property_readonly("num_leaves", &PyTreeDef::num_leaves)
      .def_property_readonly("num_nodes", &PyTreeDef::num_nodes)
      .def("__repr__", &PyTreeDef::ToString)
      .def("__eq__",
           [](const PyTreeDef& a, const PyTreeDef& b) { return a == b; })
      .def("__ne__",
           [](const PyTreeDef& a, const PyTreeDef& b) { return a != b; })
      .def("__hash__",
           [](const PyTreeDef& t) { return absl::Hash<PyTreeDef>()(t); });

  m.def("register_node", [](py::object type, py::function to_iterable,
                            py::function from_iterable) {
    return CustomNodeRegistry::Register(type, to_iterable, from_iterable);
  });
}

}  // namespace jax
