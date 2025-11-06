/* Copyright 2020 The JAX Authors

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

#ifndef JAXLIB_TRACEBACK_H_
#define JAXLIB_TRACEBACK_H_

#include <Python.h>

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

// placeholder for index annotation headers
#include "absl/types/span.h"
#include "nanobind/nanobind.h"

namespace jax {

// Entry in a traceback. Must be POD.
struct TracebackEntry {
  TracebackEntry() = default;
  TracebackEntry(PyCodeObject* code, int lasti) : code(code), lasti(lasti) {}
  PyCodeObject* code;
  int lasti;

  bool operator==(const TracebackEntry& other) const {
    return code == other.code && lasti == other.lasti;
  }
  bool operator!=(const TracebackEntry& other) const {
    return !operator==(other);
  }
};
static_assert(std::is_trivial_v<TracebackEntry> == true);

template <typename H>
H AbslHashValue(H h, const TracebackEntry& entry) {
  h = H::combine(std::move(h), entry.code, entry.lasti);
  return h;
}

class Traceback : public nanobind::object {
 public:
  NB_OBJECT(Traceback, nanobind::object, "Traceback", Traceback::Check);

  // Returns a traceback if it is enabled, otherwise returns nullopt.
  static std::optional<Traceback> Get();

  // Returns true if traceback collection is enabled.
  static bool IsEnabled();

  // Returns a string representation of the traceback.
  std::string ToString() const;

  // Returns a list of (code, lasti) pairs for each frame in the traceback.
  // Frames are from innermost to outermost.
  absl::Span<TracebackEntry const> RawFrames() const;

  struct Frame {
    nanobind::str file_name;
    nanobind::str function_name;
    int function_start_line;
    int line_num;

    std::string ToString() const;
  };
  // Returns a list of Frames for the traceback.
  std::vector<Frame> Frames() const;

  static void RegisterType(nanobind::module_& m);

 private:
  static bool Check(PyObject* o);
};

}  // namespace jax

#endif  // JAXLIB_TRACEBACK_H_
