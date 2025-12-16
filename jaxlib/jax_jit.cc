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

// This files implements the `jax.jit` dispatch and just-in-time feature.
//
// In a nutshell, `Jit(f)` returns a callable that will dispatch (i.e. forward
// based on passed arguments dtypes/shapes/identity) the execution to a
// just-in-time compiled XLA Executable. All of that is done in C++ for
// performance reasons.
//
// This file contains the utilities to:
// (a) inspect arguments and describe their structure, dtype/shapes, etc.
// (b) keep a mapping from function signatures to compiled XLA Executables.

#include "jaxlib/jax_jit.h"

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/config.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_values.h"
#include "jaxlib/pytree.h"
#include "jaxlib/sharding.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_inlined_vector.h"  // IWYU pragma: keep
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/types.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax {

namespace nb = nanobind;

// TODO(phawkins): Add support for Tracers.
// TODO(jblespiau): Use absl absl::Status.

namespace {

nb_class_ptr<Config>& disable_jit_state = *new nb_class_ptr<Config>();
nb_class_ptr<Config>& enable_x64_state = *new nb_class_ptr<Config>();
nb_class_ptr<Config>& post_hook_state = *new nb_class_ptr<Config>();

// Callback called the first time the C++ jit accesses thread-local state.
nb::object& initialize_local_state = *new nb::object();

}  // namespace

void InitializeThreadLocalState() {
  thread_local bool initialized = false;
  if (!initialized) {
    initialized = true;
    // Set the flag first to avoid reentrant calls to the initialization
    // function.
    initialize_local_state();
  }
}

bool GetDisableJit() {
  if (!disable_jit_state.ptr()) {
    throw std::runtime_error("disable_jit_state is not set");
  }
  return nb::cast<bool>(disable_jit_state->Get());
}

bool GetEnableX64() {
  if (!enable_x64_state.ptr()) {
    throw std::runtime_error("enable_x64_state is not set");
  }
  bool out = nb::cast<bool>(enable_x64_state->Get());
  return out;
}

std::optional<nb::callable> GetPostHook() {
  if (!post_hook_state.ptr()) {
    throw std::runtime_error("post_hook_state is not set");
  }
  return nb::cast<std::optional<nb::callable>>(post_hook_state->Get());
}

std::string ArgumentSignature::DebugString() const {
  auto py_object_formatter = [](std::string* out, const nb::object& o) {
    out->append(nb::cast<std::string_view>(nb::str(o)));
  };
  auto treedef_formatter = [](std::string* out, const PyTreeDef& d) {
    out->append(d.ToString());
  };
  return absl::StrFormat(
      "static args (positional + keyword): [%s], "
      "static arg keyword names: [%s], "
      "dynamic arg signatures (positional + keyword): [%s], "
      "dynamic arg shardings: [%s]",
      absl::StrJoin(static_args, ",", py_object_formatter),
      absl::StrJoin(static_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_treedefs, "| ", treedef_formatter));
}

bool ArgumentSignature::operator==(const ArgumentSignature& other) const {
  if (dynamic_arg_treedefs != other.dynamic_arg_treedefs) {
    return false;
  }
  auto object_ptr_equality = [](nb::handle a, nb::handle b) {
    return a.ptr() == b.ptr();
  };
  if (!absl::c_equal(dynamic_arg_names, other.dynamic_arg_names,
                     object_ptr_equality)) {
    return false;
  }
  if (!absl::c_equal(static_arg_names, other.static_arg_names,
                     object_ptr_equality)) {
    return false;
  }
  return absl::c_equal(
      static_args, other.static_args,
      [](const nb::object& a, const nb::object& b) {
        try {
          return a.type().ptr() == b.type().ptr() && a.equal(b);
        } catch (const nb::python_error& e) {
          throw std::invalid_argument(absl::StrCat(
              "static arguments should be comparable using __eq__."
              "The following error was raised when comparing two objects of "
              "types ",
              nb::cast<std::string_view>(nb::str(a.type())), " and ",
              nb::cast<std::string_view>(nb::str(b.type())),
              ". The error was:\n", e.what()));
        }
      });
}

std::string CallSignature::DebugString() const {
  auto py_object_formatter = [](std::string* out, const nb::object& o) {
    out->append(nb::cast<std::string_view>(nb::str(o)));
  };
  auto signature_formatter = [](std::string* out, const PyArgSignature& s) {
    out->append(s.DebugString());
  };
  auto layout_formatter = [](std::string* out,
                             const std::shared_ptr<const xla::PjRtLayout>& l) {
    if (l != nullptr) {
      out->append(l->ToString());
    } else {
      out->append("None");
    }
  };
  auto bool_formatter = [](std::string* out, bool o) {
    out->append(o ? "true" : "false");
  };
  std::vector<std::string> config_names = JitConfigNames();
  std::vector<std::string> config_strs;
  config_strs.reserve(configs.size());
  for (int i = 0; i < configs.size(); ++i) {
    config_strs.push_back(absl::StrFormat(
        "%s: %s", i < config_names.size() ? config_names[i] : "unknown",
        nb::cast<std::string_view>(nb::str(configs[i]))));
  }
  return absl::StrFormat(
      "arg signature: %s\n"
      "dynamic arg signatures (positional + keyword): %s\n"
      "dynamic arg shardings: %s\n"
      "dynamic arg layouts: %s\n"
      "committed args: %s\n"
      "device: %s\n"
      "configs: %s\n",
      arg_signature.DebugString(),
      absl::StrJoin(dynamic_arg_signatures, ", ", signature_formatter),
      absl::StrJoin(dynamic_arg_shardings, ", ", py_object_formatter),
      absl::StrJoin(dynamic_arg_layouts, ", ", layout_formatter),
      absl::StrJoin(committed_args, ",", bool_formatter),
      device != nullptr ? device->DebugString() : "nullptr",
      absl::StrJoin(config_strs, ", "));
}

size_t HashShardingForJit(nb::handle sharding) {
  auto type = sharding.type();

  if (type.is(NamedSharding::type())) {
    const auto* named_sharding = nb::inst_ptr<NamedSharding>(sharding);
    return absl::Hash<void*>()(named_sharding->mesh().ptr());
  }

  if (type.is(GSPMDSharding::type())) {
    auto* gspmd_sharding = nb::inst_ptr<GSPMDSharding>(sharding);
    return gspmd_sharding->Hash();
  }

  if (type.is(SingleDeviceSharding::type())) {
    auto* single_device_sharding = nb::inst_ptr<SingleDeviceSharding>(sharding);
    return absl::Hash<void*>()(single_device_sharding->device().ptr());
  }

  try {
    return nb::hash(sharding);
  } catch (const nb::python_error& e) {
    // Gracefully handle non-hashable sharding. We cannot let a C++ exception
    // escape because this hash function may have been called from a code that
    // disables C++ exception support.
    return 0;
  }
}

bool EqualShardingsForJit(nb::handle a, nb::handle b) {
  if (a.ptr() == b.ptr()) {
    return true;
  }

  auto a_type = a.type();
  auto b_type = b.type();

  if (!a_type.is(b_type)) {
    return false;
  }

  if (a_type.is(NamedSharding::type())) {
    auto* a_named_sharding = nb::inst_ptr<const NamedSharding>(a);
    auto* b_named_sharding = nb::inst_ptr<const NamedSharding>(b);
    return a_named_sharding->mesh().ptr() == b_named_sharding->mesh().ptr() &&
           *a_named_sharding->spec() == *b_named_sharding->spec() &&
           a_named_sharding->memory_kind().equal(
               b_named_sharding->memory_kind()) &&
           a_named_sharding->logical_device_ids().equal(
               b_named_sharding->logical_device_ids());
  }

  if (a_type.is(GSPMDSharding::type())) {
    auto* a_gspmd_sharding = nb::inst_ptr<const GSPMDSharding>(a);
    auto* b_gspmd_sharding = nb::inst_ptr<const GSPMDSharding>(b);
    return *a_gspmd_sharding == *b_gspmd_sharding;
  }

  if (a_type.is(SingleDeviceSharding::type())) {
    auto* a_single_device_sharding =
        nb::inst_ptr<const SingleDeviceSharding>(a);
    auto* b_single_device_sharding =
        nb::inst_ptr<const SingleDeviceSharding>(b);
    return a_single_device_sharding->device().ptr() ==
               b_single_device_sharding->device().ptr() &&
           a_single_device_sharding->memory_kind().equal(
               b_single_device_sharding->memory_kind());
  }

  return a.equal(b);
}

bool CallSignature::operator==(const CallSignature& other) const {
  if (arg_signature != other.arg_signature) {
    return false;
  }
  if (dynamic_arg_signatures != other.dynamic_arg_signatures) {
    return false;
  }
  if (device != other.device) {
    return false;
  }
  if (committed_args != other.committed_args) {
    return false;
  }
  return
      // `==` on py:objects is the Python `is`. We need equal.
      absl::c_equal(dynamic_arg_shardings, other.dynamic_arg_shardings,
                    EqualShardingsForJit) &&
      absl::c_equal(dynamic_arg_layouts, other.dynamic_arg_layouts,
                    [](const std::shared_ptr<const xla::PjRtLayout>& a,
                       const std::shared_ptr<const xla::PjRtLayout>& b) {
                      return (a && b) ? *a == *b : a == b;
                    }) &&
      configs.size() == other.configs.size() &&
      absl::c_equal(
          configs, other.configs,
          [](const nb::object& a, const nb::object& b) { return a.equal(b); });
}

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
absl::Status ParseArguments(
    absl::Span<PyObject* const> positional_args,
    absl::Span<PyObject* const> keyword_args, nb::handle kwnames,
    absl::Span<int const> static_argnums,
    absl::Span<nb::str const> static_argnames, PyTreeRegistry* pytree_registry,
    ArgumentSignature& signature,
    absl::InlinedVector<nanobind::object, 2>& flat_dynamic_args) {
  tsl::profiler::TraceMe traceme("ParseArguments");

  DCHECK(absl::c_all_of(static_argnames, [](const nb::str& name) {
    return PyUnicode_CHECK_INTERNED(name.ptr());
  }));

  flat_dynamic_args.reserve(positional_args.size() + keyword_args.size());
  if (static_argnums.empty()) {
    signature.dynamic_arg_treedefs.reserve(positional_args.size());

    // Positional arguments.
    for (int i = 0; i < positional_args.size(); ++i) {
      signature.dynamic_arg_treedefs.emplace_back(pytree_registry);
      PyTreeDef& pytree_def = signature.dynamic_arg_treedefs.back();
      pytree_def.Flatten(nb::handle(positional_args[i]), flat_dynamic_args);
    }
  } else {
    signature.dynamic_arg_treedefs.reserve(positional_args.size());

    // Positional arguments.
    int num_positional_args = positional_args.size();
    for (int i = 0; i < positional_args.size(); ++i) {
      if (std::find_if(static_argnums.begin(), static_argnums.end(),
                       [i, num_positional_args](int t) {
                         return t >= 0 ? i == t : i == t + num_positional_args;
                       }) == static_argnums.end()) {
        signature.dynamic_arg_treedefs.emplace_back(pytree_registry);
        PyTreeDef& pytree_def = signature.dynamic_arg_treedefs.back();
        pytree_def.Flatten(positional_args[i], flat_dynamic_args);
      } else {
        signature.static_args.emplace_back(
            nb::borrow<nb::object>(positional_args[i]));
      }
    }
  }

  // Keyword arguments.
  if (!keyword_args.empty()) {
    std::vector<std::pair<nb::handle, nb::handle>> kwargs(keyword_args.size());
    // We first intern the keys, then sort them (by name, as in the Python path)
    // (see also PyTreeDef::Flatten) and then create the signatures.
    // TODO(jblespiau): We should be able to sort the keys by interned-key
    // pointers, but this requires the Python compilation to do the same.
    for (int i = 0; i < keyword_args.size(); ++i) {
      // Intern the key if not already interned.
      PyObject* key = PyTuple_GET_ITEM(kwnames.ptr(), i);
      Py_INCREF(key);
      if (!PyUnicode_CHECK_INTERNED(key)) {
        PyUnicode_InternInPlace(&key);
      }
      kwargs[i].first = key;
      kwargs[i].second = keyword_args[i];
    }

    std::sort(kwargs.begin(), kwargs.end(),
              [](const std::pair<nb::handle, nb::handle>& a,
                 const std::pair<nb::handle, nb::handle>& b) {
                return a.first < b.first;
              });
    auto kwarg_is_static = [&](nb::handle name) {
      for (const auto& kw : static_argnames) {
        if (kw.ptr() == name.ptr()) return true;
      }
      return false;
    };

    signature.dynamic_arg_names.reserve(keyword_args.size());
    for (int i = 0; i < keyword_args.size(); ++i) {
      if (kwarg_is_static(kwargs[i].first)) {
        signature.static_arg_names.push_back(
            nb::steal<nb::str>(kwargs[i].first));
        signature.static_args.push_back(
            nb::borrow<nb::object>(kwargs[i].second));
      } else {
        signature.dynamic_arg_names.push_back(
            nb::steal<nb::str>(kwargs[i].first));
        signature.dynamic_arg_treedefs.emplace_back(pytree_registry);
        PyTreeDef& pytree_def = signature.dynamic_arg_treedefs.back();
        pytree_def.Flatten(nb::handle(kwargs[i].second.ptr()),
                           flat_dynamic_args);
      }
    }
  }
  return absl::OkStatus();
}

void BuildJaxjitSubmodule(nb::module_& m) {
  nb::module_ jitlib = m.def_submodule("jax_jit", "Jax C++ jit library");

  jitlib.attr("_Config") = m.attr("config").attr("Config");
  jitlib.attr("_PyTreeDef") = m.attr("pytree").attr("PyTreeDef");
  jitlib.attr("_PyTreeRegistry") = m.attr("pytree").attr("PyTreeRegistry");

  jitlib.def(
      "set_disable_jit_state",
      [](nb_class_ptr<Config> config) { disable_jit_state = config; },
      nb::sig("def set_disable_jit_state(config: _Config) -> None"));
  jitlib.def(
      "set_enable_x64_state",
      [](nb_class_ptr<Config> config) { enable_x64_state = config; },
      nb::sig("def set_enable_x64_state(config: _Config) -> None"));
  jitlib.def(
      "set_post_hook_state",
      [](nb_class_ptr<Config> config) { post_hook_state = config; },
      nb::sig("def set_post_hook_state(config: _Config) -> None"));

  jitlib.def(
      "set_thread_local_state_initialization_callback",
      [](nb::object f) { initialize_local_state = f; },
      nb::sig("def set_thread_local_state_initialization_callback("
              "f: typing.Callable[[], None]) -> None"));

  nb::class_<PyArgSignature> arg_signature(jitlib, "PyArgSignature");
  arg_signature
      .def_prop_ro(
          "dtype",
          [](const PyArgSignature& sig) {
            return xla::ValueOrThrow(xla::PrimitiveTypeToNbDtype(sig.dtype));
          })
      .def_prop_ro("shape",
                   [](const PyArgSignature& sig) {
                     return xla::SpanToNbTuple(absl::MakeConstSpan(sig.shape));
                   })
      .def_ro("weak_type", &PyArgSignature::weak_type);
  jitlib.def("_ArgSignatureOfValue",
             xla::ValueOrThrowWrapper(PyArgSignatureOfValue));

  nb::class_<ArgumentSignature> argument_signature(jitlib, "ArgumentSignature");
  argument_signature.def_ro("static_args", &ArgumentSignature::static_args)
      .def_ro("static_arg_names", &ArgumentSignature::static_arg_names)
      .def_ro("dynamic_arg_names", &ArgumentSignature::dynamic_arg_names)
      .def_ro(
          "dynamic_arg_treedefs", &ArgumentSignature::dynamic_arg_treedefs,
          nb::sig(
              "def dynamic_arg_treedefs(self) -> typing.Sequence[_PyTreeDef]"))
      .def("__repr__", &ArgumentSignature::DebugString)
      .def("__str__", &ArgumentSignature::DebugString)
      .def("__hash__",
           [](const ArgumentSignature& s) { return absl::HashOf(s); })
      .def(
          "__eq__",
          [](const ArgumentSignature& a, nb::object b) {
            return nb::isinstance<ArgumentSignature>(b) &&
                   a == nb::cast<ArgumentSignature>(b);
          },
          nb::is_operator())
      .def(
          "__ne__",
          [](const ArgumentSignature& a, nb::object b) {
            return !nb::isinstance<ArgumentSignature>(b) ||
                   a != nb::cast<ArgumentSignature>(b);
          },
          nb::is_operator());

  jitlib.def(
      "parse_arguments",
      [](nb::sequence positional_args, nb::sequence keyword_args,
         nb::typed<nb::tuple, nb::str, nb::ellipsis> kwnames,
         absl::Span<int const> static_argnums,
         absl::Span<nb::str const> static_argnames,
         PyTreeRegistry* pytree_registry) {
        ArgumentSignature signature;
        absl::InlinedVector<nanobind::object, 2> flat_dynamic_args;
        nb::object positional_args_seq = nb::steal(PySequence_Fast(
            positional_args.ptr(), "positional_args must be a list or tuple"));
        if (!positional_args_seq.ptr()) {
          throw nb::python_error();
        }
        nb::object keyword_args_seq = nb::steal(PySequence_Fast(
            keyword_args.ptr(), "keyword_args must be a list or tuple"));
        if (!keyword_args_seq.ptr()) {
          throw nb::python_error();
        }
        absl::Span<PyObject* const> positional_args_span =
            absl::MakeSpan(PySequence_Fast_ITEMS(positional_args_seq.ptr()),
                           PySequence_Fast_GET_SIZE(positional_args_seq.ptr()));
        absl::Span<PyObject* const> keyword_args_span =
            absl::MakeSpan(PySequence_Fast_ITEMS(keyword_args_seq.ptr()),
                           PySequence_Fast_GET_SIZE(keyword_args_seq.ptr()));

        // Intern the static argument names.
        std::vector<nb::str> static_argnames_interned;
        static_argnames_interned.reserve(static_argnames.size());
        for (const nb::str& name : static_argnames) {
          PyObject* s = name.inc_ref().ptr();
          PyUnicode_InternInPlace(&s);
          static_argnames_interned.push_back(nb::steal<nb::str>(s));
        }

        xla::ThrowIfError(
            ParseArguments(positional_args_span, keyword_args_span, kwnames,
                           static_argnums, static_argnames_interned,
                           pytree_registry, signature, flat_dynamic_args));
        return std::make_pair(std::move(signature),
                              std::move(flat_dynamic_args));
      },
      nb::arg("positional_args"), nb::arg("keyword_args"), nb::arg("kwnames"),
      nb::arg("static_argnums"), nb::arg("static_argnames"),
      nb::arg("pytree_registry"),
      nb::sig(
          // clang-format off
        "def parse_arguments("
        "positional_args: Sequence[object], "
        "keyword_args: Sequence[object], "
        "kwnames: tuple[str, ...], "
        "static_argnums: Sequence[int], "
        "static_argnames: Sequence[str], "
        "pytree_registry: _PyTreeRegistry"
        ") -> tuple[ArgumentSignature, list[object]]"
          // clang-format on
          ),
      R"doc(Parses the arguments to a function as jax.jit would.

Returns a ArgumentSignature and the flattened dynamic arguments.

Args:
  positional_args: The positional arguments.
  keyword_args: The keyword arguments.
  kwnames: The keyword names.
  static_argnums: The static argument numbers.
  static_argnames: The static argument names.
  pytree_registry: The pytree registry.
)doc");
}

}  // namespace jax
