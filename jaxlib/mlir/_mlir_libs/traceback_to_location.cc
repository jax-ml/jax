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

#include "jaxlib/mlir/_mlir_libs/traceback_to_location.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "jaxlib/traceback.h"

namespace nb = ::nanobind;

namespace jax {

TracebackToLocationCache::TracebackToLocationCache(
    nanobind::callable code_to_filename, int frame_limit,
    mlir::MLIRContext* context)
    : code_to_filename_(std::move(code_to_filename)),
      frame_limit_(frame_limit),
      context_(context) {}

nb::object TracebackToLocationCache::Get(const Traceback& traceback) {
  auto& traceback_cache_entry = traceback_to_location_cache_[traceback];
  if (!traceback_cache_entry.ptr()) {
    absl::Span<const TracebackEntry> frames = traceback.RawFrames();
    std::vector<mlir::Location> frame_locs_vector;
    frame_locs_vector.reserve(frames.size());
    for (const TracebackEntry& frame : frames) {
      auto& frame_cache_entry = frame_cache_[frame];
      if (!frame_cache_entry.has_value()) {
        // Canonicalize the filename, and skip it if it's not to be shown.
        auto [filename_cache_it, inserted] =
            code_to_filename_cache_.insert({frame.code, std::nullopt});
        auto& filename_cache_entry = filename_cache_it->second;
        if (inserted) {
          nb::object out = code_to_filename_(
              nb::borrow<nb::object>(reinterpret_cast<PyObject*>(frame.code)));
          if (out.is_none()) {
            filename_cache_entry = std::nullopt;
          } else {
            filename_cache_entry = nb::cast<std::string>(out);
          }
        }
        if (!filename_cache_entry.has_value()) {
          continue;
        }
        const std::string& filename = *filename_cache_entry;

        int start_line, start_column, end_line, end_column;
        if (!PyCode_Addr2Location(frame.code, frame.lasti, &start_line,
                                  &start_column, &end_line, &end_column)) {
          throw nb::python_error();
        }
        std::string_view function_name = nb::cast<std::string_view>(
            nb::borrow<nb::str>(frame.code->co_qualname));
        frame_cache_entry = mlir::NameLoc::get(
            mlir::StringAttr::get(context_, function_name),
            mlir::FileLineColRange::get(
                mlir::StringAttr::get(context_, filename), start_line,
                start_column, end_line, end_column));
      }
      frame_locs_vector.push_back(*frame_cache_entry);
    }
    absl::Span<mlir::Location const> frame_locs_span = frame_locs_vector;
    frame_locs_span = frame_locs_span.first(
        std::min<size_t>(frame_locs_span.size(), frame_limit_));
    std::optional<mlir::Location> loc;
    for (auto it = frame_locs_span.rbegin(); it != frame_locs_span.rend();
         ++it) {
      if (loc.has_value()) {
        loc = mlir::CallSiteLoc::get(*it, *loc);
      } else {
        loc = *it;
      }
    }
    traceback_cache_entry = nb::cast(
        wrap(loc.has_value() ? *loc : mlir::UnknownLoc::get(context_)));
  }
  return traceback_cache_entry;
}

}  // namespace jax
