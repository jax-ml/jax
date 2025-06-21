/* Copyright 2023 The JAX Authors.

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

#include <Python.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace {

// A variant of map(...) that:
// a) returns a list instead of an iterator, and
// b) checks that the input iterables are of equal length.
PyObject* SafeMap(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs < 2) {
    PyErr_SetString(PyExc_TypeError, "safe_map requires at least 2 arguments");
    return nullptr;
  }
  PyObject* fn = args[0];
  absl::InlinedVector<nb::object, 4> iterators;
  iterators.reserve(nargs - 1);
  for (Py_ssize_t i = 1; i < nargs; ++i) {
    PyObject* it = PyObject_GetIter(args[i]);
    if (!it) return nullptr;
    iterators.push_back(nb::steal<nb::object>(it));
  }

  // Try to use a length hint to estimate how large a list to allocate.
  Py_ssize_t length_hint = PyObject_LengthHint(args[1], 2);
  if (PyErr_Occurred()) {
    PyErr_Clear();
  }
  if (length_hint < 0) {
    length_hint = 2;
  }

  nb::list list = nb::steal<nb::list>(PyList_New(length_hint));
  int n = 0;  // Current true size of the list

  // The arguments we will pass to fn. We allocate space for one more argument
  // than we need at the start of the argument list so we can use
  // PY_VECTORCALL_ARGUMENTS_OFFSET which may speed up the callee.
  absl::InlinedVector<PyObject*, 4> values(nargs, nullptr);
  while (true) {
    absl::Cleanup values_cleanup = [&values]() {
      for (PyObject* v : values) {
        Py_XDECREF(v);
        v = nullptr;
      }
    };
    values[1] = PyIter_Next(iterators[0].ptr());
    if (PyErr_Occurred()) return nullptr;

    if (values[1]) {
      for (size_t i = 1; i < iterators.size(); ++i) {
        values[i + 1] = PyIter_Next(iterators[i].ptr());
        if (PyErr_Occurred()) return nullptr;
        if (!values[i + 1]) {
          PyErr_Format(PyExc_ValueError,
                       "safe_map() argument %u is shorter than argument 1",
                       i + 1);
          return nullptr;
        }
      }
    } else {
      // No more elements should be left. Checks the other iterators are
      // exhausted.
      for (size_t i = 1; i < iterators.size(); ++i) {
        values[i + 1] = PyIter_Next(iterators[i].ptr());
        if (PyErr_Occurred()) return nullptr;
        if (values[i + 1]) {
          PyErr_Format(PyExc_ValueError,
                       "safe_map() argument %u is longer than argument 1",
                       i + 1);
          return nullptr;
        }
      }

      // If the length hint was too large, truncate the list to the true size.
      if (n < length_hint) {
        if (PyList_SetSlice(list.ptr(), n, length_hint, nullptr) < 0) {
          return nullptr;
        }
      }
      return list.release().ptr();
    }

    nb::object out = nb::steal<nb::object>(PyObject_Vectorcall(
        fn, &values[1], (nargs - 1) | PY_VECTORCALL_ARGUMENTS_OFFSET,
        /*kwnames=*/nullptr));
    if (PyErr_Occurred()) {
      return nullptr;
    }

    if (n < length_hint) {
      PyList_SET_ITEM(list.ptr(), n, out.release().ptr());
    } else {
      if (PyList_Append(list.ptr(), out.ptr()) < 0) {
        return nullptr;
      }
    }
    ++n;
  }
}

PyMethodDef safe_map_def = {
    "safe_map",
    reinterpret_cast<PyCFunction>(SafeMap),
    METH_FASTCALL,
};

// Similar to SafeMap, but ignores the return values of the function and returns
// None.
PyObject* ForEach(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs < 2) {
    PyErr_SetString(PyExc_TypeError, "foreach() requires at least 2 arguments");
    return nullptr;
  }
  PyObject* fn = args[0];
  absl::InlinedVector<nb::object, 4> iterators;
  iterators.reserve(nargs - 1);
  for (Py_ssize_t i = 1; i < nargs; ++i) {
    PyObject* it = PyObject_GetIter(args[i]);
    if (!it) return nullptr;
    iterators.push_back(nb::steal<nb::object>(it));
  }

  // The arguments we will pass to fn. We allocate space for one more argument
  // than we need at the start of the argument list so we can use
  // PY_VECTORCALL_ARGUMENTS_OFFSET which may speed up the callee.
  absl::InlinedVector<PyObject*, 4> values(nargs, nullptr);
  while (true) {
    absl::Cleanup values_cleanup = [&values]() {
      for (PyObject* v : values) {
        Py_XDECREF(v);
        v = nullptr;
      }
    };
    values[1] = PyIter_Next(iterators[0].ptr());
    if (PyErr_Occurred()) return nullptr;

    if (values[1]) {
      for (size_t i = 1; i < iterators.size(); ++i) {
        values[i + 1] = PyIter_Next(iterators[i].ptr());
        if (PyErr_Occurred()) return nullptr;
        if (!values[i + 1]) {
          PyErr_Format(PyExc_ValueError,
                       "foreach() argument %u is shorter than argument 1",
                       i + 1);
          return nullptr;
        }
      }
    } else {
      // No more elements should be left. Checks the other iterators are
      // exhausted.
      for (size_t i = 1; i < iterators.size(); ++i) {
        values[i + 1] = PyIter_Next(iterators[i].ptr());
        if (PyErr_Occurred()) return nullptr;
        if (values[i + 1]) {
          PyErr_Format(PyExc_ValueError,
                       "foreach() argument %u is longer than argument 1",
                       i + 1);
          return nullptr;
        }
      }
      Py_INCREF(Py_None);
      return Py_None;
    }

    nb::object out = nb::steal<nb::object>(PyObject_Vectorcall(
        fn, &values[1], (nargs - 1) | PY_VECTORCALL_ARGUMENTS_OFFSET,
        /*kwnames=*/nullptr));
    if (PyErr_Occurred()) {
      return nullptr;
    }
  }
}

PyMethodDef foreach_def = {
    "foreach", reinterpret_cast<PyCFunction>(ForEach), METH_FASTCALL,
    "foreach() applies a function elementwise to one or more iterables, "
    "ignoring the return values and returns None. The iterables must all have "
    "the same lengths."};

// A variant of zip(...) that:
// a) returns a list instead of an iterator, and
// b) checks that the input iterables are of equal length.
// TODO(phawkins): consider replacing this function with
// list(zip(..., strict=True)) once TensorFlow 2.13 is released, which should
// resolve an incompatibility with strict=True and jax2tf.
PyObject* SafeZip(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs < 1) {
    PyErr_SetString(PyExc_TypeError, "safe_zip requires at least 1 argument");
    return nullptr;
  }
  absl::InlinedVector<nb::object, 4> iterators;
  iterators.reserve(nargs);
  for (Py_ssize_t i = 0; i < nargs; ++i) {
    PyObject* it = PyObject_GetIter(args[i]);
    if (!it) return nullptr;
    iterators.push_back(nb::steal<nb::object>(it));
  }

  // Try to use a length hint to estimate how large a list to allocate.
  Py_ssize_t length_hint = PyObject_LengthHint(args[0], 2);
  if (PyErr_Occurred()) {
    PyErr_Clear();
  }
  if (length_hint < 0) {
    length_hint = 2;
  }

  nb::list list = nb::steal<nb::list>(PyList_New(length_hint));
  int n = 0;  // Current true size of the list

  while (true) {
    nb::object tuple;
    nb::object v = nb::steal<nb::object>(PyIter_Next(iterators[0].ptr()));
    if (PyErr_Occurred()) return nullptr;

    if (v.ptr()) {
      tuple = nb::steal<nb::object>(PyTuple_New(nargs));
      if (!tuple.ptr()) return nullptr;

      PyTuple_SET_ITEM(tuple.ptr(), 0, v.release().ptr());
      for (size_t i = 1; i < iterators.size(); ++i) {
        v = nb::steal<nb::object>(PyIter_Next(iterators[i].ptr()));
        if (PyErr_Occurred()) return nullptr;
        if (!v.ptr()) {
          PyErr_Format(PyExc_ValueError,
                       "safe_zip() argument %u is shorter than argument 1",
                       i + 1);
          return nullptr;
        }
        PyTuple_SET_ITEM(tuple.ptr(), i, v.release().ptr());
      }
    } else {
      // No more elements should be left. Checks the other iterators are
      // exhausted.
      for (size_t i = 1; i < iterators.size(); ++i) {
        v = nb::steal<nb::object>(PyIter_Next(iterators[i].ptr()));
        if (PyErr_Occurred()) return nullptr;
        if (v.ptr()) {
          PyErr_Format(PyExc_ValueError,
                       "safe_zip() argument %u is longer than argument 1",
                       i + 1);
          return nullptr;
        }
      }

      // If the length hint was too large, truncate the list to the true size.
      if (n < length_hint) {
        if (PyList_SetSlice(list.ptr(), n, length_hint, nullptr) < 0) {
          return nullptr;
        }
      }
      return list.release().ptr();
    }

    if (n < length_hint) {
      PyList_SET_ITEM(list.ptr(), n, tuple.release().ptr());
    } else {
      if (PyList_Append(list.ptr(), tuple.ptr()) < 0) {
        return nullptr;
      }
      tuple = nb::object();
    }
    ++n;
  }
}

PyMethodDef safe_zip_def = {
    "safe_zip",
    reinterpret_cast<PyCFunction>(SafeZip),
    METH_FASTCALL,
};

nb::list TopologicalSort(nb::str parents_attr,
                         nb::iterable end_nodes_iterable) {
  // This is a direct conversion of the original Python implementation.
  // More efficient implementations of a topological sort are possible (and
  // indeed, easier to write), but changing the choice of topological order
  // would break existing tests.
  std::vector<nb::object> end_nodes;
  absl::flat_hash_set<PyObject*> seen;
  for (nb::handle n : end_nodes_iterable) {
    nb::object node = nb::borrow(n);
    if (seen.insert(node.ptr()).second) {
      end_nodes.push_back(node);
    }
  }

  nb::list sorted_nodes;
  if (end_nodes.empty()) {
    return sorted_nodes;
  }

  std::vector<nb::object> stack = end_nodes;
  absl::flat_hash_map<PyObject*, int> child_counts;
  while (!stack.empty()) {
    nb::object node = std::move(stack.back());
    stack.pop_back();
    auto& count = child_counts[node.ptr()];
    if (count == 0) {
      for (nb::handle parent : node.attr(parents_attr)) {
        stack.push_back(nb::borrow(parent));
      }
    }
    ++count;
  }

  for (nb::handle n : end_nodes) {
    child_counts[n.ptr()] -= 1;
  }

  std::vector<nb::object> childless_nodes;
  childless_nodes.reserve(end_nodes.size());
  for (nb::handle n : end_nodes) {
    if (child_counts[n.ptr()] == 0) {
      childless_nodes.push_back(nb::borrow(n));
    }
  }

  while (!childless_nodes.empty()) {
    nb::object node = std::move(childless_nodes.back());
    childless_nodes.pop_back();
    sorted_nodes.append(node);
    for (nb::handle parent : node.attr(parents_attr)) {
      auto& count = child_counts[parent.ptr()];
      if (count == 1) {
        childless_nodes.push_back(nb::borrow(parent));
      } else {
        --count;
      }
    }
  }
  sorted_nodes.reverse();
  return sorted_nodes;
}

}  // namespace

NB_MODULE(utils, m) {
  nb::object module_name = m.attr("__name__");
  m.attr("safe_map") = nb::steal<nb::object>(
      PyCFunction_NewEx(&safe_map_def, /*self=*/nullptr, module_name.ptr()));
  m.attr("foreach") = nb::steal<nb::object>(
      PyCFunction_NewEx(&foreach_def, /*self=*/nullptr, module_name.ptr()));
  m.attr("safe_zip") = nb::steal<nb::object>(
      PyCFunction_NewEx(&safe_zip_def, /*self=*/nullptr, module_name.ptr()));

  m.def("topological_sort", &TopologicalSort, nb::arg("parents_attr"),
        nb::arg("end_nodes"),
        "Computes a topological sort of a graph of objects. parents_attr is "
        "the name of the attribute on each object that contains the list of "
        "parent objects. end_nodes is an iterable of objects from which we "
        "should start a backwards search.");

  // Python has no reader-writer lock in its standard library, so we expose
  // bindings around absl::Mutex.
  nb::class_<absl::Mutex>(m, "Mutex")
      .def(nb::init<>())
      .def("lock", &absl::Mutex::Lock, nb::call_guard<nb::gil_scoped_release>())
      .def("unlock", &absl::Mutex::Unlock)
      .def("assert_held", &absl::Mutex::AssertHeld)
      .def("reader_lock", &absl::Mutex::ReaderLock,
           nb::call_guard<nb::gil_scoped_release>())
      .def("reader_unlock", &absl::Mutex::ReaderUnlock)
      .def("assert_reader_held", &absl::Mutex::AssertReaderHeld)
      .def("writer_lock", &absl::Mutex::WriterLock,
           nb::call_guard<nb::gil_scoped_release>())
      .def("writer_unlock", &absl::Mutex::WriterUnlock);
}