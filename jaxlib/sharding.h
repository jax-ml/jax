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

#ifndef JAXLIB_SHARDING_H_
#define JAXLIB_SHARDING_H_

#include <Python.h>

#include <cstddef>
#include <optional>
#include <utility>

// placeholder for index annotation headers
#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "jaxlib/cached_py_object.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/partition_spec.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device_list.h"
#include "jaxlib/sharded_device_array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/nb_numpy.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace jax {

class Sharding {
 public:
  Sharding() = default;

  // This constructor is used in the fast path to retrieve the number of devices
  // without falling back to python. This is only used in the cpp path.
  explicit Sharding(int num_devices) : num_devices_(num_devices) {}

  virtual ~Sharding() = default;

  int num_devices() const { return num_devices_; }

 private:
  int num_devices_;
};

// Gets `PyDeviceList` from a JAX Sharding.
absl::StatusOr<nb_class_ptr<PyDeviceList>> GetPyDeviceList(
    nanobind::handle sharding);

// Checks if the memory kind is valid, and canonicalizes the
// memory kind to default memory on backends that support memories.
nanobind::object CheckAndCanonicalizeMemoryKind(
    nanobind::object memory_kind,
    const nb_class_ptr<PyDeviceList>& device_list);

class NamedSharding : public Sharding {
 public:
  NamedSharding(nanobind::object mesh, nb_class_ptr<PartitionSpec> spec,
                nanobind::object memory_kind,
                nanobind::object logical_device_ids);

  const nanobind::object& mesh() const { return mesh_; }
  const nb_class_ptr<PartitionSpec>& spec() const { return spec_; }
  const nanobind::object& memory_kind() const { return memory_kind_; }
  const nanobind::object& logical_device_ids() const {
    return logical_device_ids_;
  }

  static nanobind::handle type() { return type_; }
  static void InitializeType();

  absl::StatusOr<nb_class_ptr<PyDeviceList>> internal_device_list() const {
    if (internal_device_list_) {
      return *internal_device_list_;
    }
    return xla::InvalidArgument(
        "internal_device_list is not implemented for "
        "`jax.sharding.AbstractMesh`");
  }

  bool operator==(const NamedSharding& other) const;

  bool Eq(const nanobind::object& other) const;  // Python __eq__
  nanobind::int_ Hash() const;                   // Python __hash__

 private:
  nanobind::object mesh_;
  nb_class_ptr<PartitionSpec> spec_;
  nanobind::object memory_kind_;
  nanobind::object logical_device_ids_;
  std::optional<nb_class_ptr<PyDeviceList>> internal_device_list_;
  mutable CachedPyObject hash_;
  static PyObject* type_;
};

class SingleDeviceSharding : public Sharding {
 public:
  explicit SingleDeviceSharding(
      nanobind::object device, nanobind::object memory_kind = nanobind::none());

  // Used only in C++ to accelerate `PyArray::MakeFromSingleDeviceArray()`.
  SingleDeviceSharding(nb_class_ptr<PyClient> client,
                       xla::ifrt::DeviceListRef device_list,
                       nanobind::object memory_kind);

  const nanobind::object& device() const { return device_; }
  const nanobind::object& memory_kind() const { return memory_kind_; }

  static nanobind::handle type() { return type_; }
  static void InitializeType();

  nb_class_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  nanobind::object device_;
  nanobind::object memory_kind_;
  nb_class_ptr<PyDeviceList> internal_device_list_;

  static PyObject* type_;
};

// The C++ implementation of jax.PmapSharding in python. It contains a few key
// data members and methods that are performance-critical.
class PmapSharding : public Sharding {
 public:
  PmapSharding(xla::nb_numpy_ndarray devices, ShardingSpec sharding_spec);

  ~PmapSharding() override = default;

  xla::nb_numpy_ndarray devices() const { return devices_; }

  const ShardingSpec& sharding_spec() const { return sharding_spec_; }

  static nanobind::handle type() { return type_; }
  static void InitializeType();

  nb_class_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  xla::nb_numpy_ndarray devices_;
  ShardingSpec sharding_spec_;
  nb_class_ptr<PyDeviceList> internal_device_list_;
  static PyObject* type_;
};

class GSPMDSharding : public Sharding {
 public:
  GSPMDSharding(nanobind::sequence devices, xla::OpSharding op_sharding,
                nanobind::object memory_kind)
      : GSPMDSharding(
            make_nb_class<PyDeviceList>(nanobind::tuple(devices)),
            xla::ValueOrThrow(xla::HloSharding::FromProto(op_sharding)),
            std::move(memory_kind)) {}

  GSPMDSharding(nanobind::sequence devices, xla::HloSharding op_sharding,
                nanobind::object memory_kind)
      : GSPMDSharding(make_nb_class<PyDeviceList>(nanobind::tuple(devices)),
                      std::move(op_sharding), std::move(memory_kind)) {}

  GSPMDSharding(nb_class_ptr<PyDeviceList> devices, xla::OpSharding op_sharding,
                nanobind::object memory_kind)
      : GSPMDSharding(
            std::move(devices),
            xla::ValueOrThrow(xla::HloSharding::FromProto(op_sharding)),
            std::move(memory_kind)) {}

  GSPMDSharding(nb_class_ptr<PyDeviceList> devices,
                xla::HloSharding op_sharding, nanobind::object memory_kind);

  nb_class_ptr<PyDeviceList> devices() const { return devices_; }
  const nanobind::object& memory_kind() const { return memory_kind_; }

  size_t Hash() {
    if (!hash_.has_value()) {
      hash_ = CalculateHash();
    }
    return *hash_;
  }

  static nanobind::handle type() { return type_; }
  static void InitializeType();

  const xla::HloSharding& hlo_sharding() const { return hlo_sharding_; }

  bool operator==(const GSPMDSharding& other) const {
    return AreOpShardingsEqual(*this, other) &&
           this->devices().equal(other.devices()) &&
           this->memory_kind().equal(other.memory_kind());
  }

  nb_class_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  size_t CalculateHash() const {
    // We only hash `hlo_sharding_` here for performance.
    return absl::Hash<xla::HloSharding>()(hlo_sharding_);
  }

  static bool AreOpShardingsEqual(const GSPMDSharding& a,
                                  const GSPMDSharding& b) {
    // If the OpSharding object is the same, return true
    if (&a.hlo_sharding() == &b.hlo_sharding()) {
      return true;
    }
    // If both OpShardings are replicated, return true
    if (a.IsOpShardingReplicated() && b.IsOpShardingReplicated()) {
      return true;
    }
    return a.hlo_sharding() == b.hlo_sharding();
  }

  bool IsOpShardingReplicated() const {
    // For JAX, shardings with 1 device are considered as replicated in its
    // semantics so that downstream things continue to work.
    if (hlo_sharding_.tile_assignment().num_elements() == 1) {
      return true;
    }
    return hlo_sharding().IsReplicated();
  }

  nb_class_ptr<PyDeviceList> devices_;
  xla::HloSharding hlo_sharding_;
  nanobind::object memory_kind_;
  std::optional<size_t> hash_;
  nb_class_ptr<PyDeviceList> internal_device_list_;

  static PyObject* type_;
};

void RegisterSharding(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_SHARDING_H_
