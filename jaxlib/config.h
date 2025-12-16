/* Copyright 2024 The JAX Authors

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

#ifndef JAXLIB_CONFIG_H_
#define JAXLIB_CONFIG_H_

#include <string>
#include <vector>

// placeholder for index annotation headers
#include "nanobind/nanobind.h"

namespace jax {

// A Config object represents a configurable object with both global and
// thread-local state. This class is wrapped using nanobind and exposed to
// Python.
class Config {
 public:
  Config(std::string name, nanobind::object value, bool include_in_jit_key,
         bool include_in_trace_context);

  // Returns the name of the config.
  const std::string& Name();

  // Returns the thread-local value if it is set, otherwise the global value.
  nanobind::object Get();

  // Returns the global value.
  nanobind::object GetGlobal();

  // Sets the global value.
  void SetGlobal(nanobind::object value);

  // Returns the thread-local value.
  nanobind::object GetLocal();

  // Sets the thread-local value. May be `unset`.
  void SetLocal(nanobind::object value);

  // Swaps the thread-local value with `value`. Returns the previous value.
  // Either may be `unset`.
  nanobind::object SwapLocal(nanobind::object value);

  // This class doesn't actually hold any data, but it's the only type
  // known to Python. We pretend that this object owns both the global and any
  // thread-local values corresponding to this key.
  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
  static PyType_Slot slots_[];

  static const nanobind::object& UnsetObject();

 private:
  int key_;
};

// Returns the set of configuration values that should be included in the JIT
// cache key.
std::vector<nanobind::object> JitConfigs();

// The corresponding config names, for debugging.
std::vector<std::string> JitConfigNames();

nanobind::tuple TraceContext();

void BuildConfigSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_CONFIG_H_
