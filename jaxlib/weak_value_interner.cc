/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "jaxlib/weak_value_interner.h"

#include <Python.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"

namespace nb = nanobind;

namespace jax {

template <typename H>
H AbslHashValue(H h, const WeakValueInterner::KeyView& key) {
  for (const auto& name : key.kwnames) {
    h = H::combine(std::move(h), name.ptr());
  }
  for (const auto& arg : key.args) {
    h = H::combine(std::move(h), nanobind::hash(arg));
  }
  return h;
}

template <typename H>
H AbslHashValue(H h, const WeakValueInterner::Key& key) {
  for (const auto& name : key.kwnames) {
    h = H::combine(std::move(h), name.ptr());
  }
  for (const auto& arg : key.args) {
    h = H::combine(std::move(h), nanobind::hash(arg));
  }
  return h;
}

bool WeakValueInterner::KeyEqual::operator()(Key a, Key b) const {
  if (a.kwnames.size() != b.kwnames.size()) return false;
  for (size_t i = 0; i < a.kwnames.size(); ++i) {
    if (a.kwnames[i].ptr() != b.kwnames[i].ptr()) return false;
  }
  if (a.args.size() != b.args.size()) return false;
  for (size_t i = 0; i < a.args.size(); ++i) {
    if (!a.args[i].equal(b.args[i])) return false;
  }
  return true;
}

bool WeakValueInterner::KeyEqual::operator()(Key a, KeyView b) const {
  if (a.kwnames.size() != b.kwnames.size()) return false;
  for (size_t i = 0; i < a.kwnames.size(); ++i) {
    if (a.kwnames[i].ptr() != b.kwnames[i].ptr()) return false;
  }
  if (a.args.size() != b.args.size()) return false;
  for (size_t i = 0; i < a.args.size(); ++i) {
    if (!a.args[i].equal(b.args[i])) return false;
  }
  return true;
}

bool WeakValueInterner::KeyEqual::operator()(Key a, PointerKey b) const {
  if (a.kwnames.size() != b.kwnames.size()) return false;
  for (size_t i = 0; i < a.kwnames.size(); ++i) {
    if (a.kwnames[i].ptr() != b.kwnames[i].ptr()) return false;
  }
  if (a.args.size() != b.args.size()) return false;
  for (size_t i = 0; i < a.args.size(); ++i) {
    if (a.args[i].ptr() != b.args[i].ptr()) return false;
  }
  return true;
}

nb_class_ptr<WeakValueInterner> WeakValueInterner::Create(nb::callable fn) {
  auto self = make_nb_class<WeakValueInterner>(std::move(fn));
  self->weakref_callback_ = nb::cast<nb::callable>(nb::cpp_function(
      [this_weak = nb::weakref(self)](nb::handle dying_weakref) {
        nb::object py_cache = this_weak();
        if (py_cache.is_none()) return;
        nb::ft_object_guard lock(py_cache);
        nb::cast<WeakValueInterner*>(py_cache)->OnWeakrefDestroyed(
            dying_weakref.ptr());
      }));
  return self;
}

PyObject* WeakValueInterner::VectorCall(PyObject* self_obj,
                                        PyObject* const* args,
                                        Py_ssize_t nargsf, PyObject* kwnames) {
  try {
    WeakValueInterner* self = nb::inst_ptr<WeakValueInterner>(self_obj);

    Py_ssize_t num_pos_args = PyVectorcall_NARGS(nargsf);
    Py_ssize_t num_kwargs = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    size_t num_args = num_pos_args + num_kwargs;

    absl::InlinedVector<nb::object, 2> sorted_kwnames;
    absl::InlinedVector<nb::object, 4> strong_args;

    strong_args.reserve(num_args);
    for (Py_ssize_t i = 0; i < num_pos_args; ++i) {
      strong_args.push_back(nb::borrow<nb::object>(args[i]));
    }

    if (num_kwargs > 0) {
      absl::InlinedVector<std::pair<PyObject*, PyObject*>, 4> sorted;
      sorted.reserve(num_kwargs);
      for (Py_ssize_t i = 0; i < num_kwargs; ++i) {
        PyObject* p = PyTuple_GET_ITEM(kwnames, i);
        Py_INCREF(p);
        PyUnicode_InternInPlace(&p);
        sorted.push_back({p, args[num_pos_args + i]});
      }
      absl::c_sort(sorted, [](const std::pair<PyObject*, PyObject*>& a,
                              const std::pair<PyObject*, PyObject*>& b) {
        return a.first < b.first;
      });

      sorted_kwnames.reserve(num_kwargs);
      for (auto& info : sorted) {
        sorted_kwnames.push_back(nb::steal<nb::object>(info.first));
        strong_args.push_back(nb::borrow<nb::object>(info.second));
      }
    }

    KeyView ptr_key{sorted_kwnames, strong_args, 0};
    size_t hash = absl::HashOf(ptr_key);
    ptr_key.cached_hash = hash;

    {
      nb::ft_object_guard lock(self_obj);
      auto it = self->entries_.find(ptr_key);
      if (it != self->entries_.end()) {
        nb::object ans = it->second->value_weakref();
        if (!ans.is_none()) {
          return ans.release().ptr();
        }
      }
    }

    // Miss. Call the function without holding the lock.
    nb::object result = nb::steal<nb::object>(
        PyObject_Vectorcall(self->fn_.ptr(), args, nargsf, kwnames));
    if (!result) return nullptr;

    nb::weakref value_weakref(result, self->weakref_callback_);

    PyObject* weakref_ptr = value_weakref.ptr();
    auto entry_ptr = std::make_shared<Entry>(
        Key{std::move(sorted_kwnames), std::move(strong_args), hash},
        std::move(value_weakref));

    {
      nb::ft_object_guard lock(self_obj);
      auto [inserted_ptr, inserted] =
          self->entries_.insert(entry_ptr->key, entry_ptr);
      if (inserted) {
        self->reverse_index_[weakref_ptr] = entry_ptr.get();
      } else {
        nb::object ans = inserted_ptr->second->value_weakref();
        if (!ans.is_none()) {
          return ans.release().ptr();
        }
        // Entry exists but value is dead. Update it.
        PyObject* old_weakref_ptr = inserted_ptr->second->value_weakref.ptr();
        self->reverse_index_.erase(old_weakref_ptr);

        inserted_ptr->second->value_weakref =
            std::move(entry_ptr->value_weakref);
        self->reverse_index_[weakref_ptr] = inserted_ptr->second.get();
      }
    }

    return result.release().ptr();
  } catch (nb::python_error& e) {
    e.restore();
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

void WeakValueInterner::OnWeakrefDestroyed(PyObject* weakref_obj) {
  auto it = reverse_index_.find(weakref_obj);
  if (it == reverse_index_.end()) return;

  Entry* entry = it->second;
  reverse_index_.erase(it);

  PointerKey ptr_key{entry->key.kwnames, entry->key.args,
                     entry->key.cached_hash};
  auto map_it = entries_.find(ptr_key);
  if (map_it != entries_.end()) {
    entries_.erase(map_it);
  }
}

int WeakValueInterner::tp_traverse(PyObject* self_obj, visitproc visit,
                                   void* arg) {
  Py_VISIT(Py_TYPE(self_obj));
  if (!nb::inst_ready(self_obj)) return 0;
  WeakValueInterner* self = nb::inst_ptr<WeakValueInterner>(self_obj);
  Py_VISIT(self->fn_.ptr());
  Py_VISIT(self->weakref_callback_.ptr());
  for (const auto& [key, entry] : self->entries_) {
    for (const auto& kwname : key.kwnames) {
      Py_VISIT(kwname.ptr());
    }
    for (const auto& a : key.args) {
      Py_VISIT(a.ptr());
    }
    Py_VISIT(entry->value_weakref.ptr());
  }
  return 0;
}

int WeakValueInterner::tp_clear(PyObject* self_obj) {
  WeakValueInterner* self = nb::inst_ptr<WeakValueInterner>(self_obj);
  self->fn_.reset();
  self->weakref_callback_.reset();
  self->entries_.clear();
  self->reverse_index_.clear();
  return 0;
}

/*static*/ PyType_Slot WeakValueInterner::slots_[] = {
    {Py_tp_traverse, (void*)WeakValueInterner::tp_traverse},
    {Py_tp_clear, (void*)WeakValueInterner::tp_clear},
    {0, nullptr},
};

void WeakValueInterner::Register(nb::module_& m) {
  static PyMethodDef call_def = {
      "__call__", reinterpret_cast<PyCFunction>(WeakValueInterner::VectorCall),
      METH_FASTCALL | METH_KEYWORDS, "Calls the interner."};

  nb::class_<WeakValueInterner> weak_value_interner(
      m, "WeakValueInterner", nb::is_weak_referenceable(),
      nb::type_slots(WeakValueInterner::slots_));

  weak_value_interner.attr("__call__") =
      nb::steal<nb::object>(PyDescr_NewMethod(
          reinterpret_cast<PyTypeObject*>(weak_value_interner.ptr()),
          &call_def));

  m.def(
      "weak_value_interner",
      [](nb::callable fn) { return WeakValueInterner::Create(std::move(fn)); },
      nb::arg("fn"));
}

}  // namespace jax
