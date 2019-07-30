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

// Caution: this code uses exceptions. The exceptions use is local to the
// binding code and the idiomatic way to emit Python exceptions.

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"


namespace jax {

namespace py = pybind11;

// Registry of custom node types.
class CustomNodeRegistry {
 public:
  struct Registration {
    // The Python type object, used to identify the type.
    py::object type;
    // A function with signature: object -> (iterable, aux_data)
    py::function to_iterable;
    // A function with signature: (aux_data, iterable) -> object
    py::function from_iterable;
  };

  // Registers a new custom type. Objects of `type` will be treated as container
  // node types in PyTrees.
  static void Register(py::object type, py::function to_iterable,
                       py::function from_iterable);

  // Finds the custom type registration for `type`. Returns nullptr if none
  // exists.
  static const Registration* Lookup(py::handle type);

 private:
  static CustomNodeRegistry* Singleton();

  struct TypeHash {
    size_t operator()(const py::object& t) const { return py::hash(t); }
  };
  struct TypeEq {
    bool operator()(const py::object& a, const py::object& b) const {
      return a.equal(b);
    }
  };
  absl::Mutex mu_;
  absl::flat_hash_map<py::object, std::unique_ptr<Registration>, TypeHash,
                      TypeEq>
      registrations_ GUARDED_BY(mu_);
};

/*static*/ CustomNodeRegistry* CustomNodeRegistry::Singleton() {
  static auto* registry = new CustomNodeRegistry;
  return registry;
}

/*static*/ void CustomNodeRegistry::Register(py::object type,
                                             py::function to_iterable,
                                             py::function from_iterable) {
  CustomNodeRegistry* registry = Singleton();
  absl::MutexLock lock(&registry->mu_);
  auto registration = absl::make_unique<Registration>();
  registration->type = type;
  registration->to_iterable = std::move(to_iterable);
  registration->from_iterable = std::move(from_iterable);
  auto it = registry->registrations_.emplace(type, std::move(registration));
  if (!it.second) {
    throw std::logic_error("Duplicate custom PyTreeDef type registration.");
  }
}

/*static*/ const CustomNodeRegistry::Registration* CustomNodeRegistry::Lookup(
    py::handle type) {
  CustomNodeRegistry* registry = Singleton();
  absl::MutexLock lock(&registry->mu_);
  auto it =
      registry->registrations_.find(py::reinterpret_borrow<py::object>(type));
  return it == registry->registrations_.end() ? nullptr : it->second.get();
}

// A PyTreeDef describes the tree structure of a PyTree.
class PyTreeDef {
 public:
  PyTreeDef() = default;

  // Flattens a Pytree into a list of leaves and a PyTreeDef.
  static std::pair<py::list, std::unique_ptr<PyTreeDef>> Flatten(py::handle x);

  // Returns an unflattened PyTree given an iterable of leaves and a PyTreeDef.
  py::object Unflatten(py::iterable leaves) const;

  // Composes two PyTreeDefs, replacing the leaves of this tree with copies of
  // `inner`.
  std::unique_ptr<PyTreeDef> Compose(const PyTreeDef& inner) const;

  // Makes a Tuple PyTreeDef out of a vector of PyTreeDefs.
  static std::unique_ptr<PyTreeDef> Tuple(const std::vector<PyTreeDef>& defs);

  std::vector<std::unique_ptr<PyTreeDef>> Children() const;

  // Maps a function over a PyTree structure, applying f_leaf to each leaf, and
  // f_node to each container node.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  py::object Walk(const py::function& f_node, py::handle f_leaf,
                  py::iterable leaves) const;

  // Given a tree of iterables with the same node/leaf structure as this PyTree,
  // build the corresponding PyTree.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  py::object FromIterableTree(py::handle xs) const;

  int num_leaves() const {
    if (traversal_.empty()) {
      return 0;
    }
    return traversal_.back().num_leaves;
  }

  int num_nodes() const { return traversal_.size(); }

  size_t Hash() const;

  bool operator==(const PyTreeDef& other) const;
  bool operator!=(const PyTreeDef& other) const { return !(*this == other); }

  std::string ToString() const;

 private:
  enum class Kind {
    kLeaf,        // An opaque leaf node
    kNone,        // None.
    kTuple,       // A tuple
    kNamedTuple,  // A collections.namedtuple
    kList,        // A list
    kDict,        // A dict
    kCustom,      // A custom type.
  };

  struct Node {
    Kind kind = Kind::kLeaf;

    // Arity for non-kLeaf types.
    int arity = 0;

    // Kind-specific auxiliary data. For a kNamedTuple, contains the tuple type
    // object. For a kDict, contains a sorted list of keys. For a kCustom type,
    // contains the auxiliary data returned by the `to_iterable` function.
    py::object node_data;

    const CustomNodeRegistry::Registration* custom = nullptr;

    // Number of leaf nodes in the subtree rooted at this node.
    int num_leaves = 0;

    // Number of leaf and interior nodes in the subtree rooted at this node.
    int num_nodes = 0;
  };
  template <typename H>
  friend H AbslHashValue(H h, const Node& n);

  template <typename H>
  friend H AbslHashValue(H h, const PyTreeDef& t);

  // Helper that manufactures an instance of a node given its children.
  static py::object MakeNode(const Node& node, absl::Span<py::object> children);

  // Recursive helper used to implement Flatten()
  static void FlattenHelper(py::handle handle, py::list* leaves,
                            PyTreeDef* tree);

  // Recursive helper used to implement FromIterableTree()
  py::object FromIterableTreeHelper(
      py::handle xs,
      std::vector<PyTreeDef::Node>::const_reverse_iterator* it) const;

  // Nodes, in a post-order traversal. We use an ordered traversal to minimize
  // allocations, and post-order corresponds to the order we need to rebuild the
  // tree structure.
  std::vector<Node> traversal_;
};

template <typename H>
H AbslHashValue(H h, const PyTreeDef::Node& n) {
  h = H::combine(std::move(h), n.kind, n.arity, n.custom);
  if (n.node_data) {
    h = H::combine(std::move(h), py::hash(n.node_data));
  }
  return h;
}

template <typename H>
H AbslHashValue(H h, const PyTreeDef& t) {
  return H::combine_contiguous(std::move(h), t.traversal_.data(),
                               t.traversal_.size());
}

bool PyTreeDef::operator==(const PyTreeDef& other) const {
  if (traversal_.size() != other.traversal_.size()) {
    return false;
  }
  for (size_t i = 0; i < traversal_.size(); ++i) {
    const Node& a = traversal_[i];
    const Node& b = other.traversal_[i];
    if (a.kind != b.kind || a.arity != b.arity ||
        (a.node_data.ptr() == nullptr) != (b.node_data.ptr() == nullptr) ||
        a.custom != b.custom) {
      return false;
    }
    if (a.node_data && a.node_data.not_equal(b.node_data)) {
      return false;
    }
    // We don't need to test equality of num_leaves and num_nodes since they
    // are derivable from the other node data.
  }
  return true;
}

void PyTreeDef::FlattenHelper(py::handle handle, py::list* leaves,
                              PyTreeDef* tree) {
  Node node;
  int start_num_nodes = tree->traversal_.size();
  int start_num_leaves = leaves->size();
  if (py::isinstance<py::none>(handle)) {
    node.kind = Kind::kNone;
  } else if (PyTuple_CheckExact(handle.ptr())) {
    py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
    node.kind = Kind::kTuple;
    node.arity = tuple.size();
    for (py::handle entry : tuple) {
      FlattenHelper(entry, leaves, tree);
    }
  } else if (PyList_CheckExact(handle.ptr())) {
    py::list list = py::reinterpret_borrow<py::list>(handle);
    node.kind = Kind::kList;
    node.arity = list.size();
    for (py::handle entry : list) {
      FlattenHelper(entry, leaves, tree);
    }
  } else if (PyDict_CheckExact(handle.ptr())) {
    py::dict dict = py::reinterpret_borrow<py::dict>(handle);
    py::list keys = py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
    if (PyList_Sort(keys.ptr())) {
      throw std::runtime_error("Dictionary key sort failed.");
    }
    for (py::handle key : keys) {
      FlattenHelper(dict[key], leaves, tree);
    }
    node.kind = Kind::kDict;
    node.arity = dict.size();
    node.node_data = std::move(keys);
  } else if ((node.custom = CustomNodeRegistry::Lookup(handle.get_type()))) {
    node.kind = Kind::kCustom;
    py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
    if (out.size() != 2) {
      throw std::runtime_error(
          "PyTree custom to_iterable function should return a pair");
    }
    node.node_data = out[1];
    node.arity = 0;
    for (py::handle entry : py::cast<py::iterable>(out[0])) {
      ++node.arity;
      FlattenHelper(entry, leaves, tree);
    }
  } else if (py::isinstance<py::tuple>(handle) &&
             py::hasattr(handle, "_fields")) {
    // We can only identify namedtuples heuristically, here by the presence of
    // a _fields attribute.
    py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
    node.kind = Kind::kNamedTuple;
    node.arity = tuple.size();
    node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
    for (py::handle entry : tuple) {
      FlattenHelper(entry, leaves, tree);
    }
  } else {
    node.kind = Kind::kLeaf;
    leaves->append(py::reinterpret_borrow<py::object>(handle));
  }
  node.num_nodes = tree->traversal_.size() - start_num_nodes + 1;
  node.num_leaves = leaves->size() - start_num_leaves;
  tree->traversal_.push_back(std::move(node));
}

/*static*/ std::pair<py::list, std::unique_ptr<PyTreeDef>> PyTreeDef::Flatten(
    py::handle x) {
  py::list leaves;
  auto tree = absl::make_unique<PyTreeDef>();
  FlattenHelper(x, &leaves, tree.get());
  return std::make_pair(std::move(leaves), std::move(tree));
}

py::object PyTreeDef::Unflatten(py::iterable leaves) const {
  std::vector<py::object> agenda;
  auto it = leaves.begin();
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for TreeDef node.");
    }
    switch (node.kind) {
      case Kind::kLeaf:
        if (it == leaves.end()) {
          throw std::invalid_argument("Too few leaves for PyTreeDef");
        }
        agenda.push_back(py::reinterpret_borrow<py::object>(*it));
        ++it;
        break;

      case Kind::kNone:
      case Kind::kTuple:
      case Kind::kNamedTuple:
      case Kind::kList:
      case Kind::kDict:
      case Kind::kCustom: {
        int size = agenda.size();
        py::object o = MakeNode(
            node,
            absl::Span<py::object>(&agenda[size - node.arity], node.arity));
        agenda.resize(agenda.size() - node.arity);
        agenda.push_back(o);
        break;
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument("Too many leaves for PyTreeDef");
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

/*static*/ py::object PyTreeDef::MakeNode(const PyTreeDef::Node& node,
                                          absl::Span<py::object> children) {
  if (children.size() != node.arity) {
    throw std::logic_error("Node arity mismatch.");
  }
  switch (node.kind) {
    case Kind::kLeaf:
      return std::move(children.front());

    case Kind::kNone:
      return py::none();

    case Kind::kTuple:
    case Kind::kNamedTuple: {
      py::tuple tuple(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        tuple[i] = std::move(children[i]);
      }
      if (node.kind == Kind::kNamedTuple) {
        return node.node_data(*tuple);
      } else {
        return std::move(tuple);
      }
    }

    case Kind::kList: {
      py::list list(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        list[i] = std::move(children[i]);
      }
      return std::move(list);
    }

    case Kind::kDict: {
      py::dict dict;
      py::list keys = py::reinterpret_borrow<py::list>(node.node_data);
      for (int i = 0; i < node.arity; ++i) {
        dict[keys[i]] = std::move(children[i]);
      }
      return std::move(dict);
      break;
    }
    case Kind::kCustom: {
      py::tuple tuple(node.arity);
      for (int i = 0; i < node.arity; ++i) {
        tuple[i] = std::move(children[i]);
      }
      return node.custom->from_iterable(node.node_data, tuple);
    }
  }
}

py::object PyTreeDef::Walk(const py::function& f_node, py::handle f_leaf,
                           py::iterable leaves) const {
  std::vector<py::object> agenda;
  auto it = leaves.begin();
  for (const Node& node : traversal_) {
    switch (node.kind) {
      case Kind::kLeaf: {
        if (it == leaves.end()) {
          throw std::invalid_argument("Too few leaves for PyTreeDef");
        }

        py::object leaf = py::reinterpret_borrow<py::object>(*it);
        agenda.push_back(f_leaf.is_none() ? std::move(leaf)
                                          : f_leaf(std::move(leaf)));
        ++it;
        break;
      }

      case Kind::kNone:
      case Kind::kTuple:
      case Kind::kNamedTuple:
      case Kind::kList:
      case Kind::kDict:
      case Kind::kCustom: {
        if (agenda.size() < node.arity) {
          throw std::logic_error("Too few elements for custom type.");
        }
        py::tuple tuple(node.arity);
        for (int i = node.arity - 1; i >= 0; --i) {
          tuple[i] = agenda.back();
          agenda.pop_back();
        }
        agenda.push_back(f_node(tuple));
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument("Too many leaves for PyTreeDef");
  }
  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

py::object PyTreeDef::FromIterableTreeHelper(
    py::handle xs,
    std::vector<PyTreeDef::Node>::const_reverse_iterator* it) const {
  if (*it == traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  const Node& node = **it;
  ++*it;
  if (node.kind == Kind::kLeaf) {
    return py::reinterpret_borrow<py::object>(xs);
  }
  py::iterable iterable = py::reinterpret_borrow<py::iterable>(xs);
  std::vector<py::object> ys;
  ys.reserve(node.arity);
  for (py::handle x : iterable) {
    ys.push_back(py::reinterpret_borrow<py::object>(x));
  }
  if (ys.size() != node.arity) {
    throw std::invalid_argument("Arity mismatch between trees");
  }
  for (int j = node.arity - 1; j >= 0; --j) {
    ys[j] = FromIterableTreeHelper(ys[j], it);
  }

  return MakeNode(node, absl::MakeSpan(ys));
}

py::object PyTreeDef::FromIterableTree(py::handle xs) const {
  auto it = traversal_.rbegin();
  py::object out = FromIterableTreeHelper(xs, &it);
  if (it != traversal_.rend()) {
    throw std::invalid_argument("Tree structures did not match.");
  }
  return out;
}

std::unique_ptr<PyTreeDef> PyTreeDef::Compose(const PyTreeDef& inner) const {
  auto out = absl::make_unique<PyTreeDef>();
  for (const Node& n : traversal_) {
    if (n.kind == Kind::kLeaf) {
      absl::c_copy(inner.traversal_, std::back_inserter(out->traversal_));
    } else {
      out->traversal_.push_back(n);
    }
  }
  return out;
}

/*static*/ std::unique_ptr<PyTreeDef> PyTreeDef::Tuple(
    const std::vector<PyTreeDef>& defs) {
  auto out = absl::make_unique<PyTreeDef>();
  for (const PyTreeDef& def : defs) {
    absl::c_copy(def.traversal_, std::back_inserter(out->traversal_));
  }
  Node node;
  node.kind = Kind::kTuple;
  node.arity = defs.size();
  out->traversal_.push_back(node);
  return out;
}

std::vector<std::unique_ptr<PyTreeDef>> PyTreeDef::Children() const {
  std::vector<std::unique_ptr<PyTreeDef>> children;
  if (traversal_.empty()) {
    return children;
  }
  Node const& root = traversal_.back();
  children.resize(root.arity);
  int pos = traversal_.size() - 1;
  for (int i = root.arity - 1; i >= 0; --i) {
    children[i] = absl::make_unique<PyTreeDef>();
    const Node& node = traversal_.at(pos - 1);
    if (pos < node.num_nodes) {
      throw std::logic_error("children() walked off start of array");
    }
    std::copy(traversal_.begin() + pos - node.num_nodes,
              traversal_.begin() + pos,
              std::back_inserter(children[i]->traversal_));
    pos -= node.num_nodes;
  }
  return children;
}

std::string PyTreeDef::ToString() const {
  std::vector<std::string> agenda;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for container.");
    }

    std::string kind;
    switch (node.kind) {
      case Kind::kLeaf:
        agenda.push_back("*");
        continue;
      case Kind::kNone:
        kind = "None";
        break;
      case Kind::kNamedTuple:
        kind = "namedtuple";
        break;
      case Kind::kTuple:
        kind = "tuple";
        break;
      case Kind::kList:
        kind = "list";
        break;
      case Kind::kDict:
        kind = "dict";
        break;
      case Kind::kCustom:
        kind = static_cast<std::string>(py::str(node.custom->type));
        break;
    }

    std::string children =
        absl::StrJoin(agenda.end() - node.arity, agenda.end(), ",");
    agenda.erase(agenda.end() - node.arity, agenda.end());

    std::string data;
    if (node.node_data) {
      data = static_cast<std::string>(py::str(node.node_data));
    }

    agenda.push_back(
        absl::StrFormat("PyTreeDef(%s%s, [%s])", kind, data, children));
  }

  if (agenda.size() != 1) {
    throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
  }
  return std::move(agenda.back());
}

PYBIND11_MODULE(pytree, m) {
  m.def("flatten", &PyTreeDef::Flatten);
  m.def("tuple", &PyTreeDef::Tuple);

  py::class_<PyTreeDef>(m, "PyTreeDef")
      .def("unflatten", &PyTreeDef::Unflatten)
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
