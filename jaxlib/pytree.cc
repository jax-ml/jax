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

// Caution: this code uses exceptions. The exception use is local to the
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
  absl::flat_hash_map<py::object, std::unique_ptr<Registration>, TypeHash,
                      TypeEq>
      registrations_;
};

/*static*/ CustomNodeRegistry* CustomNodeRegistry::Singleton() {
  static auto* registry = new CustomNodeRegistry;
  return registry;
}

/*static*/ void CustomNodeRegistry::Register(py::object type,
                                             py::function to_iterable,
                                             py::function from_iterable) {
  CustomNodeRegistry* registry = Singleton();
  auto registration = absl::make_unique<Registration>();
  registration->type = type;
  registration->to_iterable = std::move(to_iterable);
  registration->from_iterable = std::move(from_iterable);
  auto it = registry->registrations_.emplace(type, std::move(registration));
  if (!it.second) {
    throw std::invalid_argument(
        absl::StrFormat("Duplicate custom PyTreeDef type registration for %s.",
                        py::repr(type)));
  }
}

/*static*/ const CustomNodeRegistry::Registration* CustomNodeRegistry::Lookup(
    py::handle type) {
  CustomNodeRegistry* registry = Singleton();
  auto it =
      registry->registrations_.find(py::reinterpret_borrow<py::object>(type));
  return it == registry->registrations_.end() ? nullptr : it->second.get();
}

// A PyTreeDef describes the tree structure of a PyTree. A PyTree is a tree of
// Python values, where the interior nodes are tuples, lists, dictionaries, or
// user-defined containers, and the leaves are other objects.
class PyTreeDef {
 public:
  PyTreeDef() = default;

  // Flattens a Pytree into a list of leaves and a PyTreeDef.
  static std::pair<py::list, std::unique_ptr<PyTreeDef>> Flatten(py::handle x);

  // Tests whether the given list is a flat list of leaves.
  static bool AllLeaves(const py::iterable& x);

  // Flattens a Pytree up to this PyTreeDef. 'this' must be a tree prefix of
  // the tree-structure of 'x'. For example, if we flatten a value
  // [(1, (2, 3)), {"foo": 4}] with a treedef [(*, *), *], the result is the
  // list of leaves [1, (2, 3), {"foo": 4}].
  py::list FlattenUpTo(py::handle x) const;

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
  
  py::tuple GetTraversal() {return py::make_tuple(traversal_); }
  void SetTraversal(py::tuple t) { traversal_ = t.cast<std::vector<Node>>(); }

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

  // Computes the node kind of a given Python object.
  static Kind GetKind(const py::handle& obj,
                      CustomNodeRegistry::Registration const** custom);

  // Nodes, in a post-order traversal. We use an ordered traversal to minimize
  // allocations, and post-order corresponds to the order we need to rebuild the
  // tree structure.
  std::vector<Node> traversal_;
};

template <typename H>
H AbslHashValue(H h, const PyTreeDef::Node& n) {
  h = H::combine(std::move(h), n.kind, n.arity, n.custom);
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

/*static*/ PyTreeDef::Kind PyTreeDef::GetKind(
    const py::handle& obj,
    CustomNodeRegistry::Registration const** custom) {
  const PyObject* ptr = obj.ptr();
  if (PyTuple_CheckExact(ptr)) return Kind::kTuple;
  if (PyList_CheckExact(ptr)) return Kind::kList;
  if (PyDict_CheckExact(ptr)) return Kind::kDict;
  if ((*custom = CustomNodeRegistry::Lookup(obj.get_type()))) {
    return Kind::kCustom;
  } else if (py::isinstance<py::none>(obj)) {
    return Kind::kNone;
  } else if (py::isinstance<py::tuple>(obj) && py::hasattr(obj, "_fields")) {
    // We can only identify namedtuples heuristically, here by the presence of
    // a _fields attribute.
    return Kind::kNamedTuple;
  } else {
    return Kind::kLeaf;
  }
}

void PyTreeDef::FlattenHelper(py::handle handle, py::list* leaves,
                              PyTreeDef* tree) {
  Node node;
  int start_num_nodes = tree->traversal_.size();
  int start_num_leaves = leaves->size();
  node.kind = GetKind(handle, &node.custom);
  if (node.kind == Kind::kNone) {
    // Nothing to do.
  } else if (node.kind == Kind::kTuple) {
    py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
    node.arity = tuple.size();
    for (py::handle entry : tuple) {
      FlattenHelper(entry, leaves, tree);
    }
  } else if (node.kind == Kind::kList) {
    py::list list = py::reinterpret_borrow<py::list>(handle);
    node.arity = list.size();
    for (py::handle entry : list) {
      FlattenHelper(entry, leaves, tree);
    }
  } else if (node.kind == Kind::kDict) {
    py::dict dict = py::reinterpret_borrow<py::dict>(handle);
    py::list keys = py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
    if (PyList_Sort(keys.ptr())) {
      throw std::runtime_error("Dictionary key sort failed.");
    }
    for (py::handle key : keys) {
      FlattenHelper(dict[key], leaves, tree);
    }
    node.arity = dict.size();
    node.node_data = std::move(keys);
  } else if (node.kind == Kind::kCustom) {
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
  } else if (node.kind == Kind::kNamedTuple) {
    py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
    node.arity = tuple.size();
    node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
    for (py::handle entry : tuple) {
      FlattenHelper(entry, leaves, tree);
    }
  } else {
    assert(node.kind == Kind::kLeaf);
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

/*static*/ bool PyTreeDef::AllLeaves(const py::iterable& x) {
  const CustomNodeRegistry::Registration* custom;
  for (const py::handle& h : x) {
    if (GetKind(h, &custom) != Kind::kLeaf) return false;
  }
  return true;
}

py::object PyTreeDef::Unflatten(py::iterable leaves) const {
  std::vector<py::object> agenda;
  auto it = leaves.begin();
  int leaf_count = 0;
  for (const Node& node : traversal_) {
    if (agenda.size() < node.arity) {
      throw std::logic_error("Too few elements for TreeDef node.");
    }
    switch (node.kind) {
      case Kind::kLeaf:
        if (it == leaves.end()) {
          throw std::invalid_argument(absl::StrFormat(
              "Too few leaves for PyTreeDef; expected %d, got %d", num_leaves(),
              leaf_count));
        }
        agenda.push_back(py::reinterpret_borrow<py::object>(*it));
        ++it;
        ++leaf_count;
        break;

      case Kind::kNone:
      case Kind::kTuple:
      case Kind::kNamedTuple:
      case Kind::kList:
      case Kind::kDict:
      case Kind::kCustom: {
        const int size = agenda.size();
        absl::Span<py::object> span;
        if (node.arity > 0) {
          span = absl::Span<py::object>(&agenda[size - node.arity], node.arity);
        }
        py::object o = MakeNode(node, span);
        agenda.resize(size - node.arity);
        agenda.push_back(o);
        break;
      }
    }
  }
  if (it != leaves.end()) {
    throw std::invalid_argument(absl::StrFormat(
        "Too many leaves for PyTreeDef; expected %d.", num_leaves()));
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
      throw std::logic_error("MakeNode not implemented for leaves.");

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

py::list PyTreeDef::FlattenUpTo(py::handle xs) const {
  py::list leaves(num_leaves());
  std::vector<py::object> agenda;
  agenda.push_back(py::reinterpret_borrow<py::object>(xs));
  auto it = traversal_.rbegin();
  int leaf = num_leaves() - 1;
  while (!agenda.empty()) {
    if (it == traversal_.rend()) {
      throw std::invalid_argument(
          absl::StrFormat("Tree structures did not match: %s vs %s",
                          py::repr(xs), ToString()));
    }
    const Node& node = *it;
    py::object object = agenda.back();
    agenda.pop_back();
    ++it;

    switch (node.kind) {
      case Kind::kLeaf:
        if (leaf < 0) {
          throw std::logic_error("Leaf count mismatch.");
        }
        leaves[leaf] = py::reinterpret_borrow<py::object>(object);
        --leaf;
        break;

      case Kind::kNone:
        break;

      case Kind::kTuple: {
        if (!PyTuple_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected tuple, got %s.", py::repr(object)));
        }
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(
              absl::StrFormat("Tuple arity mismatch: %d != %d; tuple: %s.",
                              tuple.size(), node.arity, py::repr(object)));
        }
        for (py::handle entry : tuple) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case Kind::kList: {
        if (!PyList_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected list, got %s.", py::repr(object)));
        }
        py::list list = py::reinterpret_borrow<py::list>(object);
        if (list.size() != node.arity) {
          throw std::invalid_argument(
              absl::StrFormat("List arity mismatch: %d != %d; list: %s.",
                              list.size(), node.arity, py::repr(object)));
        }
        for (py::handle entry : list) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case Kind::kDict: {
        if (!PyDict_CheckExact(object.ptr())) {
          throw std::invalid_argument(
              absl::StrFormat("Expected dict, got %s.", py::repr(object)));
        }
        py::dict dict = py::reinterpret_borrow<py::dict>(object);
        py::list keys =
            py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
        if (PyList_Sort(keys.ptr())) {
          throw std::runtime_error("Dictionary key sort failed.");
        }
        if (keys.not_equal(node.node_data)) {
          throw std::invalid_argument(
              absl::StrFormat("Dict key mismatch; expected keys: %s; dict: %s.",
                              py::repr(node.node_data), py::repr(object)));
        }
        for (py::handle key : keys) {
          agenda.push_back(dict[key]);
        }
        break;
      }

      case Kind::kNamedTuple: {
        if (!py::isinstance<py::tuple>(object) ||
            !py::hasattr(object, "_fields")) {
          throw std::invalid_argument(absl::StrFormat(
              "Expected named tuple, got %s.", py::repr(object)));
        }
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
        if (tuple.size() != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple arity mismatch: %d != %d; tuple: %s.", tuple.size(),
              node.arity, py::repr(object)));
        }
        if (tuple.get_type().not_equal(node.node_data)) {
          throw std::invalid_argument(absl::StrFormat(
              "Named tuple type mismatch: expected type: %s, tuple: %s.",
              py::repr(node.node_data), py::repr(object)));
        }
        for (py::handle entry : tuple) {
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        break;
      }

      case Kind::kCustom: {
        auto* registration = CustomNodeRegistry::Lookup(object.get_type());
        if (registration != node.custom) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom node type mismatch: expected type: %s, value: %s.",
              py::repr(node.custom->type), py::repr(object)));
        }
        py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(object));
        if (out.size() != 2) {
          throw std::runtime_error(
              "PyTree custom to_iterable function should return a pair");
        }
        if (node.node_data.not_equal(out[1])) {
          throw std::invalid_argument(absl::StrFormat(
              "Mismatch custom node data: %s != %s; value: %s.",
              py::repr(node.node_data), py::repr(out[1]), py::repr(object)));
        }
        int arity = 0;
        for (py::handle entry : py::cast<py::iterable>(out[0])) {
          ++arity;
          agenda.push_back(py::reinterpret_borrow<py::object>(entry));
        }
        if (arity != node.arity) {
          throw std::invalid_argument(absl::StrFormat(
              "Custom type arity mismatch: %d != %d; value: %s.", arity,
              node.arity, py::repr(object)));
        }
        break;
      }
    }
  }
  if (it != traversal_.rend() || leaf != -1) {
    throw std::invalid_argument(
        absl::StrFormat("Tree structures did not match: %s vs %s",
                        py::repr(xs), ToString()));
  }
  return leaves;
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
  if (pos != 0) {
    throw std::logic_error("pos != 0 at end of PyTreeDef::Children");
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
      data = absl::StrFormat("[%s]", py::str(node.node_data));
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
           [](const PyTreeDef& t) { return absl::Hash<PyTreeDef>()(t); })
      // make PyTreeDef pickleable to enable use with multiprocessing
      // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
      .def(py::pickle(
               // function 1: __getstate__ - return a tuple with traversal_ vector to encode state
               [](const PyTreeDef &p) { return p.GetTraversal(); },
               // function 2: __setstate__ - build a new PyTreeDef from a tuple with a traversal_
               [](py::tuple t) {
                   PyTreeDef p();
                   p.SetTraversal(t);
                   return p;
               });)
  m.def("register_node", [](py::object type, py::function to_iterable,
                            py::function from_iterable) {
    return CustomNodeRegistry::Register(type, to_iterable, from_iterable);
  });
}

}  // namespace jax
