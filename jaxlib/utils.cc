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

#include "pybind11/pybind11.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"

namespace py = pybind11;

PyObject* SafeMap(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs < 2) {
    PyErr_SetString(PyExc_TypeError, "safe_map requires at least 2 arguments");
    return nullptr;
  }
  PyObject* fn = args[0];
  absl::InlinedVector<py::object, 4> iterators;
  iterators.reserve(nargs - 1);
  for (Py_ssize_t i = 1; i < nargs; ++i) {
    PyObject* it = PyObject_GetIter(args[i]);
    if (!it) return nullptr;
    iterators.push_back(py::reinterpret_steal<py::object>(it));
  }

  // Try to use a length hint to estimate how large a list to allocate.
  Py_ssize_t length_hint = PyObject_LengthHint(args[1], 2);
  if (PyErr_Occurred()) {
    PyErr_Clear();
  }
  if (length_hint < 0) {
    length_hint = 2;
  }

  py::list list(length_hint);
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
          PyErr_SetString(PyExc_ValueError,
                          "Length mismatch for arguments to safe_map");
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
          PyErr_SetString(PyExc_ValueError,
                          "Length mismatch for arguments to safe_map");
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

    // TODO(phawkins): use PyObject_Vectorcall after dropping Python 3.8 support
    py::object out = py::reinterpret_steal<py::object>(_PyObject_Vectorcall(
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

PYBIND11_MODULE(utils, m) {
  m.attr("safe_map") = py::reinterpret_steal<py::object>(
      PyCFunction_NewEx(&safe_map_def, nullptr, nullptr));
}