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
==============================================================================*/

#include "jaxlib/execution_options.h"

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/executable.h"

namespace nb = nanobind;

namespace jax {

using xla::ifrt::AttributeMap;

namespace {

thread_local ExecutionOptions execution_options_thread_local_;

AttributeMap::Value ValueFromPyObject(const std::string& key,
                                      nb::handle value) {
  // `bool` is a subclass of `int` in Python, so it must be checked first.
  if (nb::isinstance<nb::bool_>(value)) {
    return AttributeMap::BoolValue(nb::cast<bool>(value));
  }
  if (nb::isinstance<nb::int_>(value)) {
    return AttributeMap::Int64Value(nb::cast<int64_t>(value));
  }
  if (nb::isinstance<nb::float_>(value)) {
    return AttributeMap::FloatValue(nb::cast<float>(value));
  }
  if (nb::isinstance<nb::str>(value)) {
    return AttributeMap::StringValue(nb::cast<std::string>(value));
  }
  if (nb::isinstance<nb::bytes>(value)) {
    nb::bytes bytes = nb::cast<nb::bytes>(value);
    return AttributeMap::StringValue(std::string(bytes.c_str(), bytes.size()));
  }
  if (nb::isinstance<nb::sequence>(value)) {
    std::vector<int64_t> values;
    for (nb::handle item : nb::cast<nb::sequence>(value)) {
      if (!nb::isinstance<nb::int_>(item) || nb::isinstance<nb::bool_>(item)) {
        throw nb::type_error(
            absl::StrCat("Unsupported custom option ", key,
                         ": sequences must contain only integers")
                .c_str());
      }
      values.push_back(nb::cast<int64_t>(item));
    }
    return AttributeMap::Int64ListValue(std::move(values));
  }
  throw nb::type_error(
      absl::StrCat("Unsupported custom option ", key,
                   ": expected bool, int, float, str, bytes or a sequence of "
                   "ints, got ",
                   nb::cast<std::string>(nb::str(value.type())))
          .c_str());
}

}  // namespace

AttributeMap AttributeMapFromPyDict(nb::dict dict) {
  AttributeMap::Map map;
  for (auto [key, value] : dict) {
    if (!nb::isinstance<nb::str>(key)) {
      throw nb::type_error("Custom option keys must be strings");
    }
    std::string name = nb::cast<std::string>(key);
    AttributeMap::Value attr = ValueFromPyObject(name, value);
    map.insert_or_assign(std::move(name), std::move(attr));
  }
  return AttributeMap(std::move(map));
}

nb::dict AttributeMapToPyDict(const AttributeMap& map) {
  nb::dict dict;
  map.ForEach([&](const std::string& key, const AttributeMap::Value& value) {
    std::visit(
        [&](const auto& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, AttributeMap::StringValue>) {
            dict[nb::str(key.c_str())] =
                nb::str(v.value.c_str(), v.value.size());
          } else if constexpr (std::is_same_v<T,
                                              AttributeMap::Int64ListValue>) {
            nb::list list;
            for (int64_t item : v.value) list.append(item);
            dict[nb::str(key.c_str())] = list;
          } else {
            dict[nb::str(key.c_str())] = nb::cast(v.value);
          }
        },
        value);
  });
  return dict;
}

void SetExecutionOptionsThreadLocal(ExecutionOptions options) {
  execution_options_thread_local_ = std::move(options);
}

const ExecutionOptions& GetExecutionOptionsThreadLocal() {
  return execution_options_thread_local_;
}

void PopulateExecutionOptions(xla::ifrt::ExecuteOptions& options) {
  const std::optional<AttributeMap>& custom_options =
      execution_options_thread_local_.custom_options;
  if (!custom_options.has_value()) {  // Default case
    return;
  }
  if (!options.custom_options.has_value()) {
    options.custom_options = *custom_options;
    return;
  }
  custom_options->ForEach(
      [&](const std::string& key, const AttributeMap::Value& value) {
        std::visit(
            [&](const auto& v) {
              CHECK_OK(options.custom_options->Set(key, v.value));
            },
            value);
      });
}

void BuildExecutionOptionsSubmodule(nb::module_& m) {
  nb::class_<ExecutionOptions>(m, "ExecutionOptions")
      .def_prop_ro(
          "custom_options",
          [](const ExecutionOptions& options) -> std::optional<nb::dict> {
            if (!options.custom_options.has_value()) {
              return std::nullopt;
            }
            return AttributeMapToPyDict(*options.custom_options);
          });
  m.def(
      "set_execution_options_thread_local",
      [](std::optional<nb::dict> custom_options) {
        ExecutionOptions options;
        if (custom_options.has_value()) {
          options.custom_options =
              AttributeMapFromPyDict(*std::move(custom_options));
        }
        SetExecutionOptionsThreadLocal(std::move(options));
      },
      nb::kw_only(), nb::arg("custom_options").none() = nb::none(),
      "Sets execution options attached to every execution dispatched from the "
      "current thread.");
  m.def("get_execution_options_thread_local", GetExecutionOptionsThreadLocal,
        "Returns execution options in effect on the current thread.");
}

}  // namespace jax
