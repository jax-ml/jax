/* Copyright 2022 The JAX Authors

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

#include "jaxlib/sharding.h"

#include <Python.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/casts.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/partition_spec.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device.h"  // IWYU pragma: keep
#include "jaxlib/py_device_list.h"
#include "jaxlib/sharded_device_array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/safe_static_init.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace jax {

namespace nb = nanobind;

// Gets `PyDeviceList` from a JAX Sharding.
absl::StatusOr<nb_class_ptr<PyDeviceList>> GetPyDeviceList(
    nb::handle sharding) {
  if (sharding.type().is(NamedSharding::type())) {
    TF_ASSIGN_OR_RETURN(
        auto ns_device_list,
        nb::cast<const NamedSharding*>(sharding)->internal_device_list());
    return ns_device_list;
  } else if (sharding.type().is(SingleDeviceSharding::type())) {
    return nb::cast<const SingleDeviceSharding*>(sharding)
        ->internal_device_list();
  } else if (sharding.type().is(PmapSharding::type())) {
    return nb::cast<const PmapSharding*>(sharding)->internal_device_list();
  } else if (sharding.type().is(GSPMDSharding::type())) {
    return nb::cast<const GSPMDSharding*>(sharding)->internal_device_list();
  } else {
    return nb::cast<nb_class_ptr<PyDeviceList>>(
        sharding.attr("_internal_device_list"));
  }
}

nb::object CheckAndCanonicalizeMemoryKind(
    nb::object memory_kind, const nb_class_ptr<PyDeviceList>& device_list) {
  if (!memory_kind.is_none()) {
    // If memory kind is not None, check if it's supported by the devices
    // mentioned in the Sharding.
    auto supported_memory_kinds = PyDeviceList::MemoryKinds(device_list);
    if (absl::IsUnimplemented(supported_memory_kinds.status())) {
      // TODO(b/473586037): Implement
      // PjRtDeviceDescription::default_memory_space() for all backends so this
      // fallback isn't necessary.
      return nb::none();
    }
    if (!supported_memory_kinds.ok()) {
      supported_memory_kinds = nb::tuple();
    }
    for (nb::handle supported_memory_kind : *supported_memory_kinds) {
      if (supported_memory_kind.equal(memory_kind)) {
        return memory_kind;
      }
    }
    auto addressable_device_list =
        PyDeviceList::AddressableDeviceList(device_list);
    if (addressable_device_list->Len() == 0) {
      // If the device list is not addressable, we can't check if the memory
      // kind is supported, so we assume it is.
      return memory_kind;
    }
    nb::object device_kind =
        addressable_device_list->GetItem(0).attr("device_kind");
    std::string_view device_kind_str = nb::cast<std::string_view>(device_kind);
    auto py_str_formatter = [](std::string* out, nb::handle h) {
      *out += nb::cast<std::string_view>(nb::str(h));
    };
    throw nb::value_error(
        absl::StrCat(
            "Could not find memory addressable by device ", device_kind_str,
            ". Device ", device_kind_str,
            " can address the following memory kinds: ",
            absl::StrJoin(*supported_memory_kinds, ", ", py_str_formatter),
            ". Got memory kind: ", nb::cast<std::string_view>(memory_kind))
            .c_str());
  }
  // If memory kind is None, canonicalize to default memory.
  absl::StatusOr<nb::object> default_memory_kind =
      PyDeviceList::DefaultMemoryKind(device_list);
  if (!default_memory_kind.ok()) {
    return nb::none();
  }
  return *std::move(default_memory_kind);
}

// This list is to check for valid memory kinds when an AbstractMesh is passed
// to NamedSharding.
static const std::array<std::string_view, 3> valid_memory_kinds = {
    "device",
    "pinned_host",
    "unpinned_host",
};

NamedSharding::NamedSharding(nb::object mesh, nb_class_ptr<PartitionSpec> spec,
                             nb::object memory_kind,
                             nb::object logical_device_ids)
    : Sharding(/*num_devices=*/[&mesh]() {
        return nb::cast<int>(mesh.attr("size"));
      }()),
      mesh_(std::move(mesh)),
      spec_(std::move(spec)),
      memory_kind_(std::move(memory_kind)),
      logical_device_ids_(std::move(logical_device_ids)) {
  nb::object idl = nb::object(mesh_.attr("_internal_device_list"));
  if (idl.is_none()) {
    internal_device_list_ = std::nullopt;
  } else {
    internal_device_list_ = nb::cast<nb_class_ptr<PyDeviceList>>(idl);
  }
  if (internal_device_list_) {
    memory_kind_ =
        CheckAndCanonicalizeMemoryKind(memory_kind_, *internal_device_list_);
  } else {
    if (!memory_kind_.is_none() &&
        (std::find(valid_memory_kinds.begin(), valid_memory_kinds.end(),
                   nb::cast<std::string_view>(memory_kind_)) ==
         valid_memory_kinds.end())) {
      throw nb::value_error(
          absl::StrCat("Got invalid memory kind: ",
                       nb::cast<std::string_view>(memory_kind_),
                       ". Valid memory kinds are: ",
                       absl::StrJoin(valid_memory_kinds, ", "))
              .c_str());
    }
  }

  // TODO(phawkins): this leaks a reference to the check_pspec function.
  // A better way to fix this would be to move PartitionSpec and this check into
  // C++.
  auto init_fn = []() {
    nb::module_ si = nb::module_::import_("jax._src.named_sharding");
    return std::make_unique<nb::object>(si.attr("check_pspec"));
  };
  nb::object& check_pspec = xla::SafeStaticInit<nb::object>(init_fn);
  check_pspec(mesh_, spec_);
}

/*static*/ PyObject* NamedSharding::type_ = nullptr;

/*static*/ void NamedSharding::InitializeType() {
  // Intentionally leaks a reference.
  type_ = nanobind::type<NamedSharding>().inc_ref().ptr();
}

bool NamedSharding::operator==(const NamedSharding& other) const {
  // Caution: you may need to update EqualShardingsForJit in jax_jit.cc as well.
  return mesh().equal(other.mesh()) && *spec() == *other.spec() &&
         memory_kind().equal(other.memory_kind()) &&
         logical_device_ids().equal(other.logical_device_ids());
}

bool NamedSharding::Eq(const nanobind::object& other) const {
  if (!other.ptr() || other.is_none()) {
    return false;
  }
  const NamedSharding* other_sharding;
  if (!nb::try_cast<const NamedSharding*>(other, other_sharding)) {
    return false;
  }
  return this == other_sharding || *this == *other_sharding;
}

nb::int_ NamedSharding::Hash() const {
  // Caution: you may need to update HashShardingForJit in jax_jit.cc as well.
  return nb::cast<nb::int_>(hash_.Get([&]() {
    size_t h =
        absl::HashOf(nb::hash(mesh_), spec_->Hash(), nb::hash(memory_kind_),
                     nb::hash(logical_device_ids_));
    Py_hash_t s = absl::bit_cast<Py_hash_t>(h);  // Python hashes are signed.
    return nb::cast(
        s == -1 ? -2 : s);  // -1 must not be used as a Python hash value.
  }));
}

SingleDeviceSharding::SingleDeviceSharding(nb::object device,
                                           nb::object memory_kind)
    : Sharding(/*num_devices=*/1),
      device_(device),
      memory_kind_(std::move(memory_kind)),
      internal_device_list_(
          make_nb_class<PyDeviceList>(nb::make_tuple(std::move(device)))) {
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

/*static*/ PyObject* SingleDeviceSharding::type_ = nullptr;

/*static*/ void SingleDeviceSharding::InitializeType() {
  // Intentionally leaks a reference.
  type_ = nanobind::type<SingleDeviceSharding>().inc_ref().ptr();
}

SingleDeviceSharding::SingleDeviceSharding(nb_class_ptr<PyClient> client,
                                           xla::ifrt::DeviceListRef device_list,
                                           nb::object memory_kind)
    : Sharding(/*num_devices=*/1),
      device_(client->GetPyDevice(device_list->devices().front())),
      memory_kind_(std::move(memory_kind)),
      internal_device_list_(make_nb_class<PyDeviceList>(
          std::move(client), std::move(device_list))) {
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

PmapSharding::PmapSharding(xla::nb_numpy_ndarray devices,
                           ShardingSpec sharding_spec)
    : Sharding(/*num_devices=*/devices.size()),
      devices_(std::move(devices)),
      sharding_spec_(std::move(sharding_spec)) {
  nb::object flat_devices = devices_.attr("flat");
  internal_device_list_ = make_nb_class<PyDeviceList>(nb::tuple(flat_devices));
}

/*static*/ PyObject* PmapSharding::type_ = nullptr;

// /*static*/ nanobind::handle PmapSharding::type() { return type_; }

/*static*/ void PmapSharding::InitializeType() {
  // Intentionally leaks a reference.
  type_ = nanobind::type<PmapSharding>().inc_ref().ptr();
}

GSPMDSharding::GSPMDSharding(nb_class_ptr<PyDeviceList> devices,
                             xla::HloSharding op_sharding,
                             nb::object memory_kind)
    : Sharding(/*num_devices=*/nb::len(devices.ptr())),
      devices_(std::move(devices)),
      hlo_sharding_(std::move(op_sharding)),
      memory_kind_(std::move(memory_kind)) {
  internal_device_list_ = devices_;
  // This checks in python if the memory kind is correct for the given
  // devices. Currently in python this check is optimized but we want to
  // move that check to C++ after which we can remove this call.
  CHECK(devices_->Len() != 0)
      << "Devices given to GSPMDSharding must not be empty";
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

/*static*/ PyObject* GSPMDSharding::type_ = nullptr;

/*static*/ void GSPMDSharding::InitializeType() {
  // Intentionally leaks a reference.
  type_ = nanobind::type<GSPMDSharding>().inc_ref().ptr();
}

void RegisterSharding(nb::module_& m) {
  nb::class_<Sharding>(m, "Sharding").def(nb::init<>());

  nb::class_<NamedSharding, Sharding>(m, "NamedSharding", nb::dynamic_attr())
      .def(nb::init<nb::object, nb_class_ptr<PartitionSpec>, nb::object,
                    nb::object>(),
           nb::arg("mesh"), nb::arg("spec"),
           nb::arg("memory_kind").none() = nb::none(),
           nb::arg("_logical_device_ids").none() = nb::none())
      .def_prop_ro("mesh", &NamedSharding::mesh)
      .def_prop_ro("spec", &NamedSharding::spec)
      .def_prop_ro("_memory_kind", &NamedSharding::memory_kind)
      .def_prop_ro("_logical_device_ids", &NamedSharding::logical_device_ids)
      .def_prop_ro("_internal_device_list",
                   [](const NamedSharding& s) {
                     return xla::ValueOrThrow(s.internal_device_list());
                   })
      .def("__eq__", &NamedSharding::Eq, nb::arg(), nb::is_operator())
      .def("__hash__", &NamedSharding::Hash);
  NamedSharding::InitializeType();

  nb::class_<SingleDeviceSharding, Sharding>(m, "SingleDeviceSharding",
                                             nb::dynamic_attr())
      .def(nb::init<nb::object, nb::object>(), nb::arg("device"),
           nb::arg("memory_kind").none() = nb::none())
      .def_prop_ro("_device", &SingleDeviceSharding::device)
      .def_prop_ro("_memory_kind", &SingleDeviceSharding::memory_kind)
      .def_prop_ro("_internal_device_list",
                   &SingleDeviceSharding::internal_device_list);
  SingleDeviceSharding::InitializeType();

  nb::class_<PmapSharding, Sharding>(m, "PmapSharding", nb::dynamic_attr())
      .def(
          "__init__",
          [](PmapSharding* self, nb::object devices,
             ShardingSpec sharding_spec) {
            new (self) PmapSharding(xla::nb_numpy_ndarray::ensure(devices),
                                    std::move(sharding_spec));
          },
          nb::arg("devices"), nb::arg("sharding_spec"))
      .def_prop_ro("devices", &PmapSharding::devices)
      .def_prop_ro("sharding_spec", &PmapSharding::sharding_spec)
      .def_prop_ro("_internal_device_list",
                   &PmapSharding::internal_device_list);
  PmapSharding::InitializeType();

  nb::class_<GSPMDSharding, Sharding>(m, "GSPMDSharding", nb::dynamic_attr())
      // NOTE: We explicitly list the two PyDeviceList ctors first since they
      // are the fast path and PyDeviceList conforms to `nb::sequence` so we
      // can silently fall back to the slow sequence ctor(s).
      .def(nb::init<nb_class_ptr<PyDeviceList>, xla::OpSharding, nb::object>(),
           nb::arg("devices"), nb::arg("op_sharding"),
           nb::arg("memory_kind").none() = nb::none())
      .def(nb::init<nb_class_ptr<PyDeviceList>, xla::HloSharding, nb::object>(),
           nb::arg("devices"), nb::arg("op_sharding"),
           nb::arg("memory_kind").none() = nb::none())
      .def(nb::init<nb::typed<nb::sequence, PyDevice>, xla::OpSharding,
                    nb::object>(),
           nb::arg("devices"), nb::arg("op_sharding"),
           nb::arg("memory_kind").none() = nb::none())
      .def(nb::init<nb::typed<nb::sequence, PyDevice>, xla::HloSharding,
                    nb::object>(),
           nb::arg("devices"), nb::arg("op_sharding"),
           nb::arg("memory_kind").none() = nb::none())
      .def_prop_ro("_devices", &GSPMDSharding::devices)
      .def_prop_ro("_hlo_sharding", &GSPMDSharding::hlo_sharding)
      .def_prop_ro("_memory_kind", &GSPMDSharding::memory_kind)
      .def_prop_ro("_internal_device_list",
                   &GSPMDSharding::internal_device_list);
  GSPMDSharding::InitializeType();
}

}  // namespace jax
