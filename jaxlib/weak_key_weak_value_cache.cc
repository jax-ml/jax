/* Copyright 2026 The JAX Authors

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

#include "jaxlib/weak_key_weak_value_cache.h"

#include <Python.h>

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"

namespace nb = nanobind;

namespace jax {

void WeakKeyWeakValueCache::OnWeakrefDestroyed(PyObject* weakref_obj) {
  auto it = weakref_to_key_.find(weakref_obj);
  if (it != weakref_to_key_.end()) {
    PyObject* x_id = it->second;
    auto entry_it = entries_.find(x_id);
    CHECK(entry_it != entries_.end());
    PyObject* first = entry_it->second.first.ptr();
    PyObject* second = entry_it->second.second.ptr();
    weakref_to_key_.erase(it);
    weakref_to_key_.erase(first == weakref_obj ? second : first);
    entries_.erase(entry_it);
  }
}

nb_class_ptr<WeakKeyWeakValueCache> WeakKeyWeakValueCache::Create(
    nb::callable fn) {
  auto self = make_nb_class<WeakKeyWeakValueCache>(std::move(fn));
  self->weakref_callback_ = nb::cast<nb::callable>(nb::cpp_function(
      [this_weak = nb::weakref(self)](nb::handle dying_weakref) {
        nb::object py_cache = this_weak();
        if (py_cache.is_none()) return;
        nb::ft_object_guard lock(py_cache);
        nb::cast<WeakKeyWeakValueCache*>(py_cache)->OnWeakrefDestroyed(
            dying_weakref.ptr());
      }));
  return self;
}

PyObject* WeakKeyWeakValueCache::VectorCall(PyObject* self_obj,
                                            PyObject* const* args,
                                            Py_ssize_t nargs) {
  try {
    WeakKeyWeakValueCache* self = nb::inst_ptr<WeakKeyWeakValueCache>(self_obj);

    if (nargs != 1) {
      PyErr_SetString(
          PyExc_TypeError,
          "WeakKeyWeakValueCache expects exactly one positional argument");
      return nullptr;
    }

    PyObject* x = args[0];

    {
      nb::ft_object_guard lock(self_obj);
      auto it = self->entries_.find(x);
      if (it != self->entries_.end()) {
        nb::object x_ref = it->second.first();
        nb::object ans = it->second.second();
        // Another thread might have freed one of both of the weakrefs, so
        // check they are valid.
        if (!ans.is_none() && x_ref.ptr() == x) {
          return ans.release().ptr();
        }
      }
    }

    nb::object result = nb::steal<nb::object>(
        PyObject_Vectorcall(self->fn_.ptr(), args, 1, nullptr));
    if (!result) return nullptr;

    nb::weakref xref(nb::borrow<nb::object>(x), self->weakref_callback_);
    nb::weakref ansref(result, self->weakref_callback_);
    {
      nb::ft_object_guard lock(self_obj);
      auto [it, inserted] =
          self->entries_.try_emplace(x, std::move(xref), std::move(ansref));
      if (inserted) {
        self->weakref_to_key_[it->second.first.ptr()] = x;
        self->weakref_to_key_[it->second.second.ptr()] = x;
      } else {
        // If !inserted, another thread populated the cache. Because we hold a
        // strong reference to `x`, we know the entry refers to our object.
        result = it->second.second();
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

/*static*/ int WeakKeyWeakValueCache::tp_traverse(PyObject* self_obj,
                                                  visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self_obj));
  if (!nb::inst_ready(self_obj)) {
    return 0;
  }
  WeakKeyWeakValueCache* self = nb::inst_ptr<WeakKeyWeakValueCache>(self_obj);
  Py_VISIT(self->fn_.ptr());
  Py_VISIT(self->weakref_callback_.ptr());
  for (const auto& kv : self->entries_) {
    Py_VISIT(kv.second.first.ptr());
    Py_VISIT(kv.second.second.ptr());
  }
  return 0;
}

/*static*/ int WeakKeyWeakValueCache::tp_clear(PyObject* self_obj) {
  if (!nb::inst_ready(self_obj)) {
    return 0;
  }
  WeakKeyWeakValueCache* self = nb::inst_ptr<WeakKeyWeakValueCache>(self_obj);
  self->fn_.reset();
  self->weakref_callback_.reset();
  self->entries_.clear();
  self->weakref_to_key_.clear();
  return 0;
}

/*static*/ PyType_Slot WeakKeyWeakValueCache::slots_[] = {
    {Py_tp_traverse, (void*)WeakKeyWeakValueCache::tp_traverse},
    {Py_tp_clear, (void*)WeakKeyWeakValueCache::tp_clear},
    {0, nullptr},
};

void WeakKeyWeakValueCache::Register(nb::module_& m) {
  static PyMethodDef call_def = {
      "__call__",
      reinterpret_cast<PyCFunction>(WeakKeyWeakValueCache::VectorCall),
      METH_FASTCALL, "Calls the weak key weak value cache."};

  nb::class_<WeakKeyWeakValueCache> weak_key_weak_value_cache(
      m, "WeakKeyWeakValueCache", nb::is_weak_referenceable(),
      nb::type_slots(WeakKeyWeakValueCache::slots_));

  weak_key_weak_value_cache.attr("__call__") =
      nb::steal<nb::object>(PyDescr_NewMethod(
          reinterpret_cast<PyTypeObject*>(weak_key_weak_value_cache.ptr()),
          &call_def));

  m.def(
      "weak_key_weak_value_cache",
      [](nb::callable fn) {
        return WeakKeyWeakValueCache::Create(std::move(fn));
      },
      nb::arg("fn"));
}

}  // namespace jax
