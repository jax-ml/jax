/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MLIR_MLIR_LIBS_TRACEBACK_TO_LOCATION_H_
#define JAXLIB_MLIR_MLIR_LIBS_TRACEBACK_TO_LOCATION_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "nanobind/nanobind.h"
#include "jaxlib/traceback.h"

namespace jax {

class TracebackToLocationCache {
 public:
  // code_to_filename is a user provided callable that maps a code object to
  // its canonicalized filename that should appeared in the MLIR location.
  // Returns None if the filename should be omitted in tracebacks.
  TracebackToLocationCache(nanobind::callable code_to_filename, int frame_limit,
                           mlir::MLIRContext* context);

  // Returns an MLIR location for the given traceback.
  // If the traceback is empty, returns an unknown location.
  nanobind::object Get(const Traceback& traceback);

 private:
  nanobind::callable code_to_filename_;
  int frame_limit_;
  mlir::MLIRContext* context_;

  // Cached results of code_to_filename_.
  absl::flat_hash_map<PyCodeObject*, std::optional<std::string>>
      code_to_filename_cache_;

  // Cached mapping from individual frames to MLIR locations.
  absl::flat_hash_map<TracebackEntry, std::optional<mlir::Location>>
      frame_cache_;

  // Cached mapping from tracebacks to MLIR locations.
  struct TracebackHash {
    size_t operator()(const Traceback& traceback) const noexcept {
      // We know the hash of a traceback will not throw.
      return absl::bit_cast<size_t>(nanobind::hash(traceback));
    }
  };
  struct TracebackEqual {
    bool operator()(const Traceback& a, const Traceback& b) const noexcept {
      // We know equality of tracebacks will not throw.
      return a.equal(b);
    }
  };
  absl::flat_hash_map<Traceback, nanobind::object, TracebackHash,
                      TracebackEqual>
      traceback_to_location_cache_;
};

}  // namespace jax

#endif  // JAXLIB_MLIR_MLIR_LIBS_TRACEBACK_TO_LOCATION_H_
