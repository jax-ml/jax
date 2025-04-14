// Copyright 2023 The JAX Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "jaxlib/xla/ifrt_proxy.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"  // IWYU pragma: keep
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/unordered_map.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "jaxlib/xla/nb_class_ptr.h"
#include "jaxlib/xla/py_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"

namespace nb = ::nanobind;

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

struct PyClientConnectionOptions {
  std::optional<std::function<void(std::string)>> on_disconnect;
  std::optional<std::function<void(std::string)>> on_connection_update;
  std::optional<int64_t> connection_timeout_in_seconds;
  std::optional<
      std::unordered_map<std::string, std::variant<nb::bytes, bool, int64_t>>>
      initialization_data;
};

absl::StatusOr<nb_class_ptr<PyClient>> GetClient(
    std::string proxy_server_address,
    const PyClientConnectionOptions& py_options) {
  DCHECK(PyGILState_Check());
  std::unique_ptr<xla::ifrt::Client> client;

  ClientConnectionOptions options;
  if (py_options.on_disconnect) {
    // While it is possible to pass around `py_options.on_disconnect` without
    // wrapping it via a shared_ptr, copying the `py_options.on_disconnect`
    // object can internally attempt to acquire the GIL [1], and can thus block
    // or even deadlock. A unique_ptr or `absl::AnyInvocable` is not sufficient
    // because downstream code can make copies. Reference:
    // https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
    auto py_on_disconnect = std::make_shared<std::function<void(std::string)>>(
        std::move(*py_options.on_disconnect));

    options.on_disconnect =
        [on_disconnect = std::move(py_on_disconnect)](absl::Status s) mutable {
          LOG(WARNING) << "Connection to server failed, calling supplied "
                       << "`on_disconnect` function: " << s;
          tsl::Env::Default()->SchedClosure([s, on_disconnect]() mutable {
            nb::gil_scoped_acquire gil_acquire;
            (*on_disconnect)(s.ToString());
            on_disconnect = nullptr;
          });
        };
  }

  if (py_options.on_connection_update) {
    auto fn = std::make_shared<std::function<void(std::string)>>(
        std::move(*py_options.on_connection_update));
    options.on_connection_update = [fn](absl::string_view log_line) -> void {
      tsl::Env::Default()->SchedClosure([fn, str = std::string(log_line)] {
        nb::gil_scoped_acquire gil_acquire;
        (*fn)(std::string(str));
      });
    };
  }

  if (py_options.connection_timeout_in_seconds.has_value()) {
    options.connection_timeout =
        absl::Seconds(*py_options.connection_timeout_in_seconds);
  }

  if (py_options.initialization_data.has_value()) {
    AttributeMap::Map attribute_map;
    for (const auto& [key, py_value] : *py_options.initialization_data) {
      if (std::holds_alternative<nb::bytes>(py_value)) {
        nb::bytes value = std::get<nb::bytes>(py_value);
        attribute_map.insert({key, AttributeMap::StringValue(std::string(
                                       value.c_str(), value.size()))});
      } else if (std::holds_alternative<bool>(py_value)) {
        attribute_map.insert(
            {key, AttributeMap::BoolValue(std::get<bool>(py_value))});
      } else {
        CHECK(std::holds_alternative<int64_t>(py_value));
        attribute_map.insert(
            {key, AttributeMap::Int64Value(std::get<int64_t>(py_value))});
      }
    }
    options.initialization_data = AttributeMap(std::move(attribute_map));
  }

  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(client, CreateClient(proxy_server_address, options));
  }

  // Constructing `xla::PyClient` requires GIL as it may dec-ref Python objects.
  return xla::PyClient::Make(std::move(client));
}

}  // namespace

void BuildIfrtProxySubmodule(nb::module_& m) {
  nb::module_ sub_module = m.def_submodule("ifrt_proxy", "IFRT proxy");

  nb::class_<PyClientConnectionOptions>(sub_module, "ClientConnectionOptions")
      .def(nb::init<>())
      .def_rw("on_disconnect", &PyClientConnectionOptions::on_disconnect,
              nb::arg().none())
      .def_rw("on_connection_update",
              &PyClientConnectionOptions::on_connection_update,
              nb::arg().none())
      .def_rw("connection_timeout_in_seconds",
              &PyClientConnectionOptions::connection_timeout_in_seconds,
              nb::arg().none())
      .def_rw("initialization_data",
              &PyClientConnectionOptions::initialization_data,
              nb::arg().none());

  sub_module.def("get_client", xla::ValueOrThrowWrapper(GetClient),
                 nb::arg("proxy_server_address"), nb::arg("options"));
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
