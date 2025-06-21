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

#include "jaxlib/py_values.h"

#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/complex.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "jaxlib/py_array.h"
#include "jaxlib/python_ref_manager.h"
#include "jaxlib/sharding.h"
#include "jaxlib/to_ifrt_sharding.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/safe_static_init.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/profiler/lib/traceme.h"

namespace nb = nanobind;

namespace xla {

namespace {

// Gets the thread-local instance.
static DevicePutInfo& GetDevicePutInfo() {
  thread_local DevicePutInfo device_put_info;
  return device_put_info;
}

// Prepared data for creating a single shard of an array. Holds a single-device
// IFRT array or a host buffer.
struct Shard {
  explicit Shard(ifrt::ArrayRef ifrt_array, bool weak_type)
      : ifrt_array_or_host_buffer(std::move(ifrt_array)),
        weak_type(weak_type),
        // host_buffer_semantics is not meaningful when
        // `ifrt_array_or_host_buffer` is an IFRT Array.
        host_buffer_semantics(
            ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall) {}

  Shard(ifrt::Client::HostBuffer ifrt_host_buffer, bool weak_type,
        ifrt::Client::HostBufferSemantics host_buffer_semantics)
      : ifrt_array_or_host_buffer(std::move(ifrt_host_buffer)),
        weak_type(weak_type),
        host_buffer_semantics(host_buffer_semantics) {}

  Shard(const Shard&) = delete;
  Shard& operator=(const Shard&) = delete;
  Shard(Shard&&) noexcept = default;
  Shard& operator=(Shard&&) noexcept = default;

  bool is_ifrt_array() const {
    return std::holds_alternative<ifrt::ArrayRef>(ifrt_array_or_host_buffer);
  }
  ifrt::DType ifrt_dtype() const;
  const ifrt::Shape& ifrt_shape() const;

  // Points to the on-device array or on-host buffer.
  std::variant<ifrt::ArrayRef, ifrt::Client::HostBuffer>
      ifrt_array_or_host_buffer;
  bool weak_type;
  ifrt::Client::HostBufferSemantics host_buffer_semantics;
};

// A function that creates a `Shard` from a Python object when called.
using ShardFn = absl::AnyInvocable<absl::StatusOr<Shard>() &&>;

absl::StatusOr<std::vector<absl::Cord>> StringDTypeArrayToCords(
    PyArrayObject* py_array_obj) {
  if (PyArray_SIZE(py_array_obj) == 0) {
    return absl::InvalidArgumentError("empty numpy array");
  }

  std::vector<absl::Cord> cords;
  cords.reserve(PyArray_SIZE(py_array_obj));

  auto iter =
      nb::steal(PyArray_IterNew(reinterpret_cast<PyObject*>(py_array_obj)));
  while (PyArray_ITER_NOTDONE(iter.ptr())) {
    auto* iter_data = PyArray_ITER_DATA(iter.ptr());
    auto* item = PyArray_GETITEM(py_array_obj, static_cast<char*>(iter_data));
    if (!item) {
      return absl::InternalError(
          "Failed to get elements out of the ndarray iter.");
    }
    Py_ssize_t len;
    auto str = PyUnicode_AsUTF8AndSize(item, &len);
    cords.push_back(absl::Cord(absl::string_view(str, len)));
    PyArray_ITER_NEXT(iter.ptr());
  }
  return cords;
}

// Handler that creates a `Shard` from a Python object.
using DevicePutHandler = std::function<absl::StatusOr<ShardFn>(
    nb::handle obj, ifrt::Client* client, ifrt::Device* to_device,
    ifrt::MemoryKind to_memory_kind, const DevicePutOptions& options)>;

// Shared logic that makes an IFRT array (either single-device or multi-device)
// from a fully-replicated `shard` that is created from a host buffer (not from
// an existing IFRT array). `shard` will be consumed.
//
// `user_context` will be used for a new IFRT array created.
//
// Expected to be called without holding GIL.
absl::StatusOr<tsl::RCReference<ifrt::Array>>
MakeIfrtArrayFromFullyReplicatedShard(
    ifrt::Client* ifrt_client, ifrt::ShardingRef ifrt_sharding, Shard& shard,
    tsl::RCReference<ifrt::UserContext> user_context) {
  auto host_buffer_shard = std::get<ifrt::Client::HostBuffer>(
      std::move(shard.ifrt_array_or_host_buffer));
  return ifrt_client->MakeArrayFromHostBuffer(
      host_buffer_shard.data, host_buffer_shard.dtype,
      std::move(host_buffer_shard.shape),
      std::move(host_buffer_shard.byte_strides), std::move(ifrt_sharding),
      shard.host_buffer_semantics, std::move(host_buffer_shard.on_done),
      std::move(user_context));
}

// Shared logic that makes a single-device IFRT array from a `shard`. `shard`
// will be consumed.
//
// `user_context` will be used for a new IFRT array created from the host
// buffer, and be not applied when reusing an existing IFRT array.
//
// Expected to be called without holding GIL.
absl::StatusOr<ifrt::ArrayRef> MakeSingleDeviceIfrtArrayFromShard(
    xla::ifrt::Client* ifrt_client, xla::ifrt::Device* ifrt_device,
    xla::ifrt::MemoryKind ifrt_memory_kind, Shard& shard,
    tsl::RCReference<ifrt::UserContext> user_context) {
  if (auto* ifrt_array =
          std::get_if<ifrt::ArrayRef>(&shard.ifrt_array_or_host_buffer)) {
    return std::move(*ifrt_array);
  }
  ifrt::ShardingRef ifrt_sharding =
      ifrt::SingleDeviceSharding::Create(ifrt_device, ifrt_memory_kind);
  return MakeIfrtArrayFromFullyReplicatedShard(
      ifrt_client, std::move(ifrt_sharding), shard, std::move(user_context));
}

// Makes an IFRT Array from `shards` using a batched array creation API (fast
// path). `shards` will be consumed.
//
// Expected to be called without holding GIL.
absl::StatusOr<ifrt::ArrayRef> MakeIfrtArrayFromShardsInBatch(
    ifrt::Client* ifrt_client, ifrt::DType ifrt_dtype, ifrt::Shape ifrt_shape,
    ifrt::ShardingRef ifrt_sharding, absl::Span<Shard> shards,
    tsl::RCReference<ifrt::UserContext> user_context) {
  absl::InlinedVector<
      std::pair<absl::InlinedVector<int64_t, 1>, ifrt::Client::HostBuffer>, 1>
      host_buffers;
  host_buffers.reserve(shards.size());
  ifrt::Client::HostBufferSemantics safe_host_semantics =
      ifrt::Client::HostBufferSemantics::kImmutableZeroCopy;
  // TODO(hyeontaek): Deduplicate shards here or early on to create a unique
  // HostBuffer for each set of replicated shards.
  for (int64_t i = 0; i < shards.size(); ++i) {
    host_buffers.push_back({{i},
                            std::get<ifrt::Client::HostBuffer>(std::move(
                                shards[i].ifrt_array_or_host_buffer))});
    // The minimum host buffer semantics is a safe semantics that can be used
    // for all shards when they are created in a single batch.
    safe_host_semantics =
        std::min(safe_host_semantics, shards[i].host_buffer_semantics);
  }

  std::vector<ifrt::Client::MakeArraysFromHostBufferShardsSpec> specs;
  specs.push_back(ifrt::Client::MakeArraysFromHostBufferShardsSpec{
      std::move(host_buffers),
      ifrt::ArraySpec{/*dtype=*/ifrt_dtype,
                      /*shape=*/std::move(ifrt_shape),
                      /*sharding=*/std::move(ifrt_sharding),
                      /*layout=*/nullptr}});
  TF_ASSIGN_OR_RETURN(
      auto arrays,
      ifrt_client->MakeArraysFromHostBufferShards(
          absl::MakeSpan(specs), safe_host_semantics, std::move(user_context)));
  return std::move(arrays.front());
}

// Makes an IFRT Array from `shards` using an array assembly API (slow path).
// `shards` will be consumed.
//
// Expected to be called without holding GIL.
absl::StatusOr<ifrt::ArrayRef> MakeIfrtArrayFromShardsWithAssembly(
    ifrt::Client* ifrt_client, ifrt::DType ifrt_dtype, ifrt::Shape ifrt_shape,
    ifrt::ShardingRef ifrt_sharding,
    ifrt::DeviceList* ifrt_addressable_device_list,
    ifrt::MemoryKind ifrt_memory_kind, absl::Span<Shard> shards,
    tsl::RCReference<ifrt::UserContext> user_context) {
  absl::Span<ifrt::Device* const> ifrt_addressable_devices =
      ifrt_addressable_device_list->devices();
  std::vector<ifrt::ArrayRef> ifrt_array_shards;
  ifrt_array_shards.reserve(shards.size());
  for (int64_t i = 0; i < shards.size(); ++i) {
    TF_ASSIGN_OR_RETURN(ifrt::ArrayRef ifrt_array_shard,
                        MakeSingleDeviceIfrtArrayFromShard(
                            ifrt_client, ifrt_addressable_devices[i],
                            ifrt_memory_kind, shards[i], user_context));
    ifrt_array_shards.push_back(std::move(ifrt_array_shard));
  }
  return ifrt_client->AssembleArrayFromSingleDeviceArrays(
      ifrt_dtype, std::move(ifrt_shape), std::move(ifrt_sharding),
      absl::MakeSpan(ifrt_array_shards), ifrt::ArrayCopySemantics::kReuseInput,
      ifrt::SingleDeviceShardSemantics::kAddressableShards);
}

template <typename T, typename SquashedT>
absl::StatusOr<ShardFn> HandlePythonScalar(nb::handle obj, ifrt::Client* client,
                                           ifrt::Device* to_device,
                                           ifrt::MemoryKind to_memory_kind,
                                           const DevicePutOptions& options) {
  T value;
  try {
    value = nb::cast<T>(obj);
  } catch (const std::exception& e) {
    return InvalidArgument(
        "Unable to convert Python scalar to %s. This most likely means the "
        "value (%s) overflows the range of the type.",
        PrimitiveType_Name(primitive_util::NativeToPrimitiveType<T>()),
        nb::cast<absl::string_view>(nb::repr(obj)));
  }

  std::variant<T, SquashedT> data;
  Shape shape;
  PrimitiveType type;
  if (std::is_same<T, SquashedT>() || !options.squash_64bit_types) {
    data.template emplace<0>(value);
    type = primitive_util::NativeToPrimitiveType<T>();
  } else {
    // TODO(phawkins): we should check for overflow here, e.g., because of bugs
    // like https://github.com/google/jax/issues/2006
    data.template emplace<1>(static_cast<SquashedT>(value));
    type = primitive_util::NativeToPrimitiveType<SquashedT>();
  }
  TF_ASSIGN_OR_RETURN(ifrt::DType ifrt_dtype, ifrt::ToDType(type));

  return [data, ifrt_dtype]() -> absl::StatusOr<Shard> {
    const void* ptr = std::visit(
        [](const auto& v) { return static_cast<const void*>(&v); }, data);
    ifrt::Client::HostBuffer ifrt_host_buffer{
        ptr, ifrt_dtype, ifrt::Shape({}),
        /*byte_strides=*/std::nullopt,
        /*on_done_with_host_buffer=*/nullptr};
    return Shard(std::move(ifrt_host_buffer), /*weak_type=*/true,
                 ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall);
  };
}

absl::StatusOr<ShardFn> HandlePythonInt(nb::handle obj, ifrt::Client* client,
                                        ifrt::Device* to_device,
                                        ifrt::MemoryKind to_memory_kind,
                                        const DevicePutOptions& options) {
  PrimitiveType type;
  std::variant<int64_t, int32_t> data;

  if (options.squash_64bit_types) {
    try {
      data.emplace<1>(nb::cast<int32_t>(obj));
    } catch (const std::exception& e) {
      return InvalidArgument(
          "Unable to convert Python scalar to %s. This most likely means the "
          "value (%s) overflows the range of the type.",
          PrimitiveType_Name(primitive_util::NativeToPrimitiveType<int32_t>()),
          nb::cast<absl::string_view>(nb::repr(obj)));
    }
    type = S32;
  } else {
    try {
      data.emplace<0>(nb::cast<int64_t>(obj));
    } catch (const std::exception& e) {
      return InvalidArgument(
          "Unable to convert Python scalar to %s. This most likely means the "
          "value (%s) overflows the range of the type.",
          PrimitiveType_Name(primitive_util::NativeToPrimitiveType<int64_t>()),
          nb::cast<absl::string_view>(nb::repr(obj)));
    }
    type = S64;
  }
  TF_ASSIGN_OR_RETURN(ifrt::DType ifrt_dtype, ifrt::ToDType(type));
  return [data, ifrt_dtype]() -> absl::StatusOr<Shard> {
    const void* ptr = std::visit(
        [](const auto& v) { return static_cast<const void*>(&v); }, data);
    ifrt::Client::HostBuffer ifrt_host_buffer{
        ptr, ifrt_dtype, ifrt::Shape({}),
        /*byte_strides=*/std::nullopt,
        /*on_done_with_host_buffer=*/nullptr};
    return Shard(std::move(ifrt_host_buffer), /*weak_type=*/true,
                 ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall);
  };
}

template <typename T, typename SquashedT = T>
absl::StatusOr<ShardFn> HandleNumpyScalar(nb::handle h, ifrt::Client* client,
                                          ifrt::Device* to_device,
                                          ifrt::MemoryKind to_memory_kind,
                                          const DevicePutOptions& options) {
  std::variant<T, SquashedT, void*> data;
  PrimitiveType type;
  // For extension types, ScalarAsCtype returns a pointer to the data.
  if (std::is_same<T, xla::s2>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = S2;
  } else if (std::is_same<T, xla::s4>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = S4;
  } else if (std::is_same<T, xla::u2>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = U2;
  } else if (std::is_same<T, xla::u4>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = U4;
  } else if (std::is_same<T, bfloat16>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = BF16;
  } else if (std::is_same<T, tsl::float4_e2m1fn>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F4E2M1FN;
  } else if (std::is_same<T, tsl::float8_e3m4>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E3M4;
  } else if (std::is_same<T, tsl::float8_e4m3>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3;
  } else if (std::is_same<T, tsl::float8_e4m3fn>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3FN;
  } else if (std::is_same<T, tsl::float8_e4m3b11fnuz>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3B11FNUZ;
  } else if (std::is_same<T, tsl::float8_e5m2>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E5M2;
  } else if (std::is_same<T, tsl::float8_e4m3fnuz>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3FNUZ;
  } else if (std::is_same<T, tsl::float8_e5m2fnuz>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E5M2FNUZ;
  } else if (std::is_same<T, tsl::float8_e8m0fnu>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E8M0FNU;
  } else if (std::is_same<T, SquashedT>() || !options.squash_64bit_types) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<0>());
    type = primitive_util::NativeToPrimitiveType<T>();
  } else {
    T value;
    PyArray_ScalarAsCtype(h.ptr(), &value);
    data.template emplace<1>(static_cast<SquashedT>(value));
    type = primitive_util::NativeToPrimitiveType<SquashedT>();
  }
  std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref;
  if (data.index() == 2) {
    py_buffer_ref =
        GlobalPyRefManager()->ManageReference(nb::cast<nb::object>(h));
  }
  TF_ASSIGN_OR_RETURN(ifrt::DType ifrt_dtype, ifrt::ToDType(type));
  return [data, py_buffer_ref = std::move(py_buffer_ref),
          ifrt_dtype]() mutable -> absl::StatusOr<Shard> {
    const void* ptr = std::visit(
        [](const auto& v) -> const void* {
          if constexpr (std::is_same_v<std::decay_t<decltype(v)>, void*>) {
            return v;
          } else {
            return static_cast<const void*>(&v);
          }
        },
        data);
    ifrt::Client::HostBuffer ifrt_host_buffer{
        ptr, ifrt_dtype, ifrt::Shape({}),
        /*byte_strides=*/std::nullopt,
        /*on_done_with_host_buffer=*/
        [py_buffer_ref =
             std::move(py_buffer_ref)]() { /* keeps py_buffer_ref alive */ }};
    return Shard(std::move(ifrt_host_buffer), /*weak_type=*/false,
                 ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall);
  };
}

absl::StatusOr<ShardFn> HandleStringNumpyArray(
    nb::handle h, ifrt::Client* client, ifrt::Device* to_device,
    ifrt::MemoryKind to_memory_kind, const DevicePutOptions& options) {
  xla::nb_numpy_ndarray array = nb::cast<xla::nb_numpy_ndarray>(h);
  auto py_array_obj = reinterpret_cast<PyArrayObject*>(array.ptr());
  TF_ASSIGN_OR_RETURN(auto cords, StringDTypeArrayToCords(py_array_obj));

  // Assemble all the parameters of MakeArrayFromHostBuffer
  const void* data = cords.data();

  // Make an explicit copy of the shape elements so we won't run into complex
  // endianness and precision issues that might arise if we reinterpret-casted
  // from npy_intp, that can be just 32 bits-wide in some environments
  // such as macos_arm64 to const int64_t* that must be 64 bits-wide.
  ifrt::Shape::Dimensions dims;
  dims.reserve(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    dims.push_back(array.shape(i));
  }
  ifrt::Shape shape(std::move(dims));

  auto on_done_with_host_buffer = [cords = std::move(cords)] {};

  return [data, shape = std::move(shape),
          on_done_with_host_buffer = std::move(
              on_done_with_host_buffer)]() mutable -> absl::StatusOr<Shard> {
    ifrt::Client::HostBuffer ifrt_host_buffer{
        data, ifrt::DType(ifrt::DType::kString), std::move(shape),
        /*byte_strides=*/std::nullopt, std::move(on_done_with_host_buffer)};
    return Shard(
        std::move(ifrt_host_buffer), /*weak_type=*/false,
        ifrt::Client::HostBufferSemantics::kImmutableUntilTransferCompletes);
  };
}

absl::StatusOr<ShardFn> HandleNumpyArray(nb::handle h, ifrt::Client* client,
                                         ifrt::Device* to_device,
                                         ifrt::MemoryKind to_memory_kind,
                                         const DevicePutOptions& options) {
  xla::nb_numpy_ndarray array = nb::cast<xla::nb_numpy_ndarray>(h);

  // String numpy arrays require substantially different processing.
  if (array.dtype().char_() == (int)'T' || array.dtype().kind() == 'T') {
    return HandleStringNumpyArray(h, client, to_device, to_memory_kind,
                                  options);
  }

  TF_ASSIGN_OR_RETURN(PrimitiveType type, DtypeToPrimitiveType(array.dtype()));

  PrimitiveType squashed_type;
  if (options.squash_64bit_types) {
    squashed_type = Squash64BitTypes(type);
    if (squashed_type != type) {
      TF_ASSIGN_OR_RETURN(xla::nb_dtype squashed_dtype,
                          PrimitiveTypeToNbDtype(squashed_type));
      array = nb::steal<xla::nb_numpy_ndarray>(PyArray_CastToType(
          reinterpret_cast<PyArrayObject*>(array.ptr()),
          reinterpret_cast<PyArray_Descr*>(squashed_dtype.release().ptr()),
          /*fortran=*/0));
    }
  } else {
    squashed_type = type;
  }

  absl::InlinedVector<int64_t, 4> dims(array.ndim());
  ifrt::Client::HostBuffer::ByteStrides byte_strides(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    dims[i] = array.shape(i);
    byte_strides[i] = array.strides(i);
  }
  const void* data = array.data();
  std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref =
      GlobalPyRefManager()->ManageReference(std::move(array));
  TF_ASSIGN_OR_RETURN(ifrt::DType ifrt_dtype, ifrt::ToDType(squashed_type));
  return [data, ifrt_dtype, dims = std::move(dims),
          byte_strides = std::move(byte_strides),
          py_buffer_ref = std::move(py_buffer_ref),
          allow_zero_copy =
              options.allow_zero_copy]() mutable -> absl::StatusOr<Shard> {
    ifrt::Client::HostBufferSemantics host_buffer_semantics =
        ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall;
    std::function<void()> on_done_with_host_buffer;
    if (allow_zero_copy) {
      on_done_with_host_buffer =
          [py_buffer_ref{
              std::move(py_buffer_ref)}]() { /* keeps py_buffer_ref alive */ };
      host_buffer_semantics =
          ifrt::Client::HostBufferSemantics::kImmutableZeroCopy;
    }

    ifrt::Client::HostBuffer ifrt_host_buffer{
        data, ifrt_dtype, ifrt::Shape(dims), std::move(byte_strides),
        std::move(on_done_with_host_buffer)};
    return Shard(std::move(ifrt_host_buffer), /*weak_type=*/false,
                 host_buffer_semantics);
  };
}

absl::StatusOr<ShardFn> HandlePyArray(nb::handle obj, ifrt::Client* client,
                                      ifrt::Device* to_device,
                                      ifrt::MemoryKind to_memory_kind,
                                      const DevicePutOptions& options) {
  auto py_array = nb::borrow<PyArray>(obj);

  // We only allow single device case for PyArray in device put.
  if (py_array.num_shards() != 1) {
    return InvalidArgument(
        "device_put expects an array with exactly one shard, got an array with "
        "with %d shards.",
        py_array.num_shards());
  }

  ifrt::Array* ifrt_array = py_array.ifrt_array();
  if (ifrt_array == nullptr) {
    return InvalidArgument("Array has been deleted.");
  }

  // Fallback to python for non-matching clients or pmap sharding.
  if (py_array.sharding().type().ptr() == jax::PmapSharding::type().ptr() ||
      ifrt_array->sharding().devices()->devices().front()->client() !=
          to_device->client()) {
    return HandleNumpyArray(obj.attr("_value"), client, to_device,
                            to_memory_kind, options);
  }

  if (ifrt_array->sharding().devices()->devices().front() == to_device &&
      options.allow_zero_copy &&
      (!to_memory_kind.memory_kind().has_value() ||
       !ifrt_array->sharding().memory_kind().memory_kind().has_value() ||
       ifrt_array->sharding().memory_kind() == to_memory_kind)) {
    Shard result(tsl::FormRef(ifrt_array), py_array.weak_type());
    return [result = std::move(result)]() mutable { return std::move(result); };
  } else {
    return [ifrt_array = tsl::FormRef(ifrt_array), to_device, to_memory_kind,
            weak_type = py_array.weak_type(),
            allow_zero_copy =
                options.allow_zero_copy]() mutable -> absl::StatusOr<Shard> {
      auto* ifrt_client = ifrt_array->client();
      TF_ASSIGN_OR_RETURN(
          auto copied_ifrt_arrays,
          ifrt_client->CopyArrays(
              absl::MakeSpan(&ifrt_array, 1),
              ifrt_client->MakeDeviceList({to_device}), to_memory_kind,
              allow_zero_copy ? ifrt::ArrayCopySemantics::kReuseInput
                              : ifrt::ArrayCopySemantics::kAlwaysCopy));
      return Shard(std::move(copied_ifrt_arrays.front()), weak_type);
    };
  }
}

ifrt::DType Shard::ifrt_dtype() const {
  if (is_ifrt_array()) {
    return std::get<ifrt::ArrayRef>(ifrt_array_or_host_buffer)->dtype();
  } else {
    return std::get<ifrt::Client::HostBuffer>(ifrt_array_or_host_buffer).dtype;
  }
}

const ifrt::Shape& Shard::ifrt_shape() const {
  if (is_ifrt_array()) {
    return std::get<ifrt::ArrayRef>(ifrt_array_or_host_buffer)->shape();
  } else {
    return std::get<ifrt::Client::HostBuffer>(ifrt_array_or_host_buffer).shape;
  }
}

// Creates a `ShardFn` that copies `arg` to `to_device` and `to_memory_kind`.
//
// Requires GIL. The returned `ShardFn` should be called without GIL held.
absl::StatusOr<ShardFn> MakeShardFn(nb::handle arg, ifrt::Client* client,
                                    ifrt::Device* to_device,
                                    ifrt::MemoryKind to_memory_kind,
                                    const DevicePutOptions& options) {
  using PyObjectDeviceHandlerMap =
      absl::flat_hash_map<PyObject*, DevicePutHandler>;

  auto init_fn = []() {
    std::unique_ptr<PyObjectDeviceHandlerMap> p =
        std::make_unique<PyObjectDeviceHandlerMap>();

    const NumpyScalarTypes& dtypes = GetNumpyScalarTypes();
    // Python scalar types.
    static_assert(sizeof(bool) == 1, "Conversion code assumes bool is 1 byte");
    (*p)[reinterpret_cast<PyObject*>(&PyBool_Type)] =
        HandlePythonScalar<bool, bool>;
    (*p)[reinterpret_cast<PyObject*>(&PyLong_Type)] = HandlePythonInt;
    (*p)[reinterpret_cast<PyObject*>(&PyFloat_Type)] =
        HandlePythonScalar<double, float>;
    (*p)[reinterpret_cast<PyObject*>(&PyComplex_Type)] =
        HandlePythonScalar<complex128, complex64>;

    (*p)[reinterpret_cast<PyObject*>(&PyArray_Type)] = HandleNumpyArray;

    // Numpy scalar types. For some of them, we share the handler with
    // Python types (np_int64, np_float64, np_complex128).
    (*p)[dtypes.np_bool.ptr()] = HandleNumpyScalar<bool>;
    (*p)[dtypes.np_int4.ptr()] = HandleNumpyScalar<xla::s4>;
    if (dtypes.np_int2.has_value()) {
      (*p)[dtypes.np_int2->ptr()] = HandleNumpyScalar<xla::s2>;
    }
    (*p)[dtypes.np_int8.ptr()] = HandleNumpyScalar<int8_t>;
    (*p)[dtypes.np_int16.ptr()] = HandleNumpyScalar<int16_t>;
    (*p)[dtypes.np_int32.ptr()] = HandleNumpyScalar<int32_t>;
    (*p)[dtypes.np_int64.ptr()] = HandleNumpyScalar<int64_t, int32_t>;
    if (dtypes.np_uint2.has_value()) {
      (*p)[dtypes.np_uint2->ptr()] = HandleNumpyScalar<xla::u2>;
    }
    (*p)[dtypes.np_uint4.ptr()] = HandleNumpyScalar<xla::u4>;
    (*p)[dtypes.np_uint8.ptr()] = HandleNumpyScalar<uint8_t>;
    (*p)[dtypes.np_uint16.ptr()] = HandleNumpyScalar<uint16_t>;
    (*p)[dtypes.np_uint32.ptr()] = HandleNumpyScalar<uint32_t>;
    (*p)[dtypes.np_uint64.ptr()] = HandleNumpyScalar<uint64_t, uint32_t>;
    if (dtypes.np_float4_e2m1fn.has_value()) {
      (*p)[dtypes.np_float4_e2m1fn->ptr()] =
          HandleNumpyScalar<tsl::float4_e2m1fn>;
    }
    if (dtypes.np_float8_e3m4.has_value()) {
      (*p)[dtypes.np_float8_e3m4->ptr()] = HandleNumpyScalar<tsl::float8_e3m4>;
    }
    if (dtypes.np_float8_e4m3.has_value()) {
      (*p)[dtypes.np_float8_e4m3->ptr()] = HandleNumpyScalar<tsl::float8_e4m3>;
    }
    (*p)[dtypes.np_float8_e4m3fn.ptr()] = HandleNumpyScalar<tsl::float8_e4m3fn>;
    (*p)[dtypes.np_float8_e4m3b11fnuz.ptr()] =
        HandleNumpyScalar<tsl::float8_e4m3b11fnuz>;
    (*p)[dtypes.np_float8_e5m2.ptr()] = HandleNumpyScalar<tsl::float8_e5m2>;
    (*p)[dtypes.np_float8_e4m3fnuz.ptr()] =
        HandleNumpyScalar<tsl::float8_e4m3fnuz>;
    (*p)[dtypes.np_float8_e5m2fnuz.ptr()] =
        HandleNumpyScalar<tsl::float8_e5m2fnuz>;
    if (dtypes.np_float8_e8m0fnu.has_value()) {
      (*p)[dtypes.np_float8_e8m0fnu->ptr()] =
          HandleNumpyScalar<tsl::float8_e8m0fnu>;
    }
    (*p)[dtypes.np_bfloat16.ptr()] = HandleNumpyScalar<bfloat16>;
    (*p)[dtypes.np_float16.ptr()] = HandleNumpyScalar<half>;
    (*p)[dtypes.np_float32.ptr()] = HandleNumpyScalar<float>;
    (*p)[dtypes.np_float64.ptr()] = HandleNumpyScalar<double, float>;
    (*p)[dtypes.np_complex64.ptr()] = HandleNumpyScalar<complex64>;
    (*p)[dtypes.np_complex128.ptr()] = HandleNumpyScalar<complex128, complex64>;
    static_assert(sizeof(long long) == sizeof(int64_t),  // NOLINT
                  "long long must be the same size as int64_t");
    (*p)[dtypes.np_longlong.ptr()] = HandleNumpyScalar<int64_t, int32_t>;
    static_assert(sizeof(int) == sizeof(int32_t),
                  "int must be the same size as int32_t");
    (*p)[dtypes.np_intc.ptr()] = HandleNumpyScalar<int32_t>;
    return p;
  };
  const PyObjectDeviceHandlerMap& handlers =
      xla::SafeStaticInit<PyObjectDeviceHandlerMap>(init_fn);

  if (arg.type().ptr() == PyArray::type().ptr()) {
    auto array = nb::borrow<PyArray>(arg);
    return HandlePyArray(arg, client, to_device, to_memory_kind, options);
  }

  auto res = handlers.find(arg.type().ptr());
  if (res == handlers.end()) {
    for (auto base_class : arg.type().attr("__mro__")) {
      res = handlers.find(base_class.ptr());
      if (res != handlers.end()) {
        return res->second(arg, client, to_device, to_memory_kind, options);
      }
    }
    return InvalidArgument(
        "%s", absl::StrCat(
                  "Not supported: The C++ jax jit execution path, only accepts "
                  "DeviceArray, Numpy arrays scalars of supported types "
                  "(see implementation), or Python scalars. Got type ",
                  nb::cast<absl::string_view>(nb::str(arg.type()))));
  }
  return res->second(arg, client, to_device, to_memory_kind, options);
}

}  // namespace

bool IsFloat0(xla::nb_numpy_ndarray arg) {
  const nb::object& float0_dtype = SafeStaticInit<nb::object>([] {
    nb::module_ dtypes_module = nb::module_::import_("jax.dtypes");
    nb::object float0_dtype = dtypes_module.attr("float0");
    return std::make_unique<nb::object>(float0_dtype);
  });
  return float0_dtype.is(arg.attr("dtype"));
}

std::string PyArgSignature::DebugString() const {
  std::string result = "";
  if (weak_type) {
    absl::StrAppend(&result, "weak_");
  }
  absl::StrAppend(&result, xla::PrimitiveType_Name(dtype));
  absl::StrAppend(&result, "[", absl::StrJoin(shape, ","), "]");
  return result;
}

using ToPyArgSignatureHandler =
    std::function<absl::StatusOr<PyArgSignature>(nb::handle, bool)>;

absl::StatusOr<PyArgSignature> PyArgSignatureOfValue(nb::handle arg,
                                                     bool jax_enable_x64) {
  const absl::flat_hash_map<PyObject*, ToPyArgSignatureHandler>& handlers =
      SafeStaticInit<
          absl::flat_hash_map<PyObject*, ToPyArgSignatureHandler>>([] {
        auto p = std::make_unique<
            absl::flat_hash_map<PyObject*, ToPyArgSignatureHandler>>();

        const NumpyScalarTypes& dtypes = GetNumpyScalarTypes();

        // The 4 Python native types.
        ToPyArgSignatureHandler bool_handler =
            [](nb::handle, bool) -> absl::StatusOr<PyArgSignature> {
          return PyArgSignature(PrimitiveType::PRED, {}, true);
        };
        ToPyArgSignatureHandler int_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // TODO(phawkins): we should consider checking for integer overflow.
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::S64, {}, true);
          } else {
            return PyArgSignature(PrimitiveType::S32, {}, true);
          }
        };
        ToPyArgSignatureHandler float_handler =
            [&dtypes](nb::handle h,
                      bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // Only Python native types has a True weak_type.
          bool weak_type = !nb::isinstance(h, dtypes.np_float64);
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::F64, {}, weak_type);
          } else {
            return PyArgSignature(PrimitiveType::F32, {}, weak_type);
          }
        };
        ToPyArgSignatureHandler complex_handler =
            [&dtypes](nb::handle h,
                      bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // Note that this branch is also taken  for np.complex128:
          // isinstance(np.complex128(3), complex) returns True
          // isinstance(np.complex64(3), complex) returns False
          bool weak_type = !nb::isinstance(h, dtypes.np_complex128);
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::C128, {}, weak_type);
          } else {
            return PyArgSignature(PrimitiveType::C64, {}, weak_type);
          }
        };

        (*p)[reinterpret_cast<PyObject*>(&PyBool_Type)] = bool_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyLong_Type)] = int_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyFloat_Type)] = float_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyComplex_Type)] = complex_handler;

        ToPyArgSignatureHandler numpy_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          xla::nb_numpy_ndarray numpy_array =
              nb::cast<xla::nb_numpy_ndarray>(h);
          TF_ASSIGN_OR_RETURN(PrimitiveType dtype,
                              DtypeToPrimitiveType(numpy_array.dtype()));
          if (!jax_enable_x64) {
            dtype = Squash64BitTypes(dtype);
          }
          // We use reinterpret_cast<> to defend against environments where
          // ssize_t may not be precisely the same type as int64_t, even if it
          // is the same size (long vs long long).
          static_assert(sizeof(int64_t) == sizeof(ssize_t),
                        "Code assumes ssize_t is the same as int64_t");
          return PyArgSignature(
              dtype,
              absl::MakeConstSpan(
                  reinterpret_cast<const int64_t*>(numpy_array.shape()),
                  numpy_array.ndim()),
              /*weak_type=*/false);
        };
        (*p)[reinterpret_cast<PyObject*>(&PyArray_Type)] = numpy_handler;

        ToPyArgSignatureHandler np_uint64_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::U64, {}, /*weak_type=*/false);
          } else {
            return PyArgSignature(PrimitiveType::U32, {}, /*weak_type=*/false);
          }
        };
        ToPyArgSignatureHandler np_int_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::S64, {}, /*weak_type=*/false);
          } else {
            return PyArgSignature(PrimitiveType::S32, {}, /*weak_type=*/false);
          }
        };
        ToPyArgSignatureHandler numpy_array_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // This block deals with all numpy scalar types, except for int64_dt,
          // float64_dt and complex128_dt which are taken care of in previous if
          // blocks.
          TF_ASSIGN_OR_RETURN(auto dtype,
                              DtypeToPrimitiveType(h.attr("dtype")));
          return PyArgSignature(dtype, {}, /*weak_type=*/false);
        };

        // This block deals with all numpy scalar types, except for int64_dt,
        // float64_dt and complex128_dt which are taken care of in previous if
        // blocks.
        (*p)[dtypes.np_bool.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int4.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int8.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int32.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int64.ptr()] = np_int_handler;
        (*p)[dtypes.np_uint4.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_uint8.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_uint16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_uint32.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_uint64.ptr()] = np_uint64_handler;
        // TODO(upwind): Explore if we can remove std::optional for these types
        // in xla/python/types.h and xla/python/types.cc
        if (dtypes.np_float4_e2m1fn.has_value()) {
          (*p)[dtypes.np_float4_e2m1fn->ptr()] = numpy_array_handler;
        }
        if (dtypes.np_float8_e3m4.has_value()) {
          (*p)[dtypes.np_float8_e3m4->ptr()] = numpy_array_handler;
        }
        if (dtypes.np_float8_e4m3.has_value()) {
          (*p)[dtypes.np_float8_e4m3->ptr()] = numpy_array_handler;
        }
        (*p)[dtypes.np_float8_e4m3fn.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e4m3b11fnuz.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e4m3fnuz.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e5m2.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e5m2fnuz.ptr()] = numpy_array_handler;
        if (dtypes.np_float8_e8m0fnu.has_value()) {
          (*p)[dtypes.np_float8_e8m0fnu->ptr()] = numpy_array_handler;
        }
        (*p)[dtypes.np_float16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_bfloat16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float32.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float64.ptr()] = float_handler;
        (*p)[dtypes.np_complex64.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_complex128.ptr()] = complex_handler;
        (*p)[dtypes.np_longlong.ptr()] = np_int_handler;
        (*p)[dtypes.np_intc.ptr()] = numpy_array_handler;

        return p;
      });

  if (arg.type().ptr() == PyArray::type().ptr()) {
    auto array = nb::borrow<PyArray>(arg);
    ifrt::Array* ifrt_array = array.ifrt_array();
    if (ifrt_array == nullptr) {
      return xla::InvalidArgument("Array has been deleted.");
    }
    TF_ASSIGN_OR_RETURN(auto primitive_type,
                        ifrt::ToPrimitiveType(ifrt_array->dtype()));
    return PyArgSignature(primitive_type, array.shape(), array.weak_type());
  }

  auto res = handlers.find(arg.type().ptr());
  if (res == handlers.end()) {
    // We attempt to look at the MRO classes
    for (auto base_class : arg.type().attr("__mro__")) {
      res = handlers.find(base_class.ptr());
      if (res != handlers.end()) {
        return res->second(arg, jax_enable_x64);
      }
    }
    return InvalidArgument(
        "%s",
        absl::StrCat("Not supported: The C++ ToPyArgSignature only accepts "
                     "Buffer/DeviceArray, Numpy "
                     "arrays scalars of supported types "
                     "(see implementation), or Python scalars. Got type ",
                     nb::cast<absl::string_view>(nb::str(arg.type()))));
  }
  return res->second(arg, jax_enable_x64);
}

absl::StatusOr<DevicePutResult> DevicePutWithDevice(
    nanobind::handle addressable_shard, ifrt::Client* ifrt_client,
    ifrt::Device* ifrt_device, ifrt::MemoryKind ifrt_memory_kind,
    const DevicePutOptions& options) {
  tsl::profiler::TraceMe traceme("DevicePut");
  ++GetDevicePutInfo().device_put_with_device;

  if (!ifrt_device->IsAddressable()) {
    return InvalidArgument("Cannot copy array to non-addressable device: %s",
                           ifrt_device->DebugString());
  }

  TF_ASSIGN_OR_RETURN(ShardFn shard_fn,
                      MakeShardFn(addressable_shard, ifrt_client, ifrt_device,
                                  ifrt_memory_kind, options));

  tsl::RCReference<ifrt::UserContext> ifrt_user_context =
      ifrt_client->CreateUserContext();

  nb::gil_scoped_release gil_release;

  TF_ASSIGN_OR_RETURN(Shard shard, std::move(shard_fn)());
  TF_ASSIGN_OR_RETURN(ifrt::ArrayRef ifrt_array,
                      MakeSingleDeviceIfrtArrayFromShard(
                          ifrt_client, ifrt_device, ifrt_memory_kind, shard,
                          std::move(ifrt_user_context)));
  return DevicePutResult(std::move(ifrt_array), shard.weak_type);
}

absl::StatusOr<DevicePutResult> DevicePutWithSharding(
    absl::Span<const nanobind::handle> addressable_shards,
    ifrt::Client* ifrt_client, const nb_dtype& dtype,
    absl::Span<const int64_t> shape, nanobind::handle sharding,
    const DevicePutOptions& options) {
  tsl::profiler::TraceMe traceme("DevicePutWithSharding");
  ++GetDevicePutInfo().device_put_with_sharding;

  TF_ASSIGN_OR_RETURN(ifrt::DeviceListRef ifrt_device_list,
                      GetIfrtDeviceList(sharding));
  ifrt::DeviceList* ifrt_addressable_device_list =
      ifrt_device_list->AddressableDeviceList();
  absl::Span<ifrt::Device* const> ifrt_addressable_devices =
      ifrt_addressable_device_list->devices();
  // Pmap sharding requires special handling because it needs a shard shape
  // upfront.
  const bool is_pmap_sharding = sharding.type().is(jax::PmapSharding::type());

  if (addressable_shards.size() != ifrt_addressable_devices.size()) {
    // Try to generate a friendly error message if the user attempted to copy to
    // a non-addressable device.
    if (addressable_shards.size() > ifrt_addressable_devices.size()) {
      for (ifrt::Device* device : ifrt_device_list->devices()) {
        if (!device->IsAddressable()) {
          return InvalidArgument(
              "Cannot copy array to non-addressable device: %s",
              device->DebugString());
        }
      }
    }
    // Otherwise, generate a generic error message.
    return InvalidArgument(
        "Number of addressable shard data does not match the number "
        "of addressable devices in the sharding: %d vs. %d",
        addressable_shards.size(), ifrt_addressable_devices.size());
  }
  if (is_pmap_sharding && addressable_shards.empty()) {
    return InvalidArgument(
        "Pmap sharding requires at least one addressable shard.");
  }

  TF_ASSIGN_OR_RETURN(ifrt::DType ifrt_dtype, DtypeToIfRtDType(dtype));
  ifrt::Shape ifrt_shape(shape);
  ifrt::MemoryKind ifrt_memory_kind = GetMemoryKind(sharding);

  std::vector<ShardFn> shard_fns;
  shard_fns.reserve(addressable_shards.size());
  for (int i = 0; i < addressable_shards.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        ShardFn shard,
        MakeShardFn(addressable_shards[i], ifrt_client,
                    ifrt_addressable_devices[i], ifrt_memory_kind, options));
    shard_fns.push_back(std::move(shard));
  }

  ifrt::ShardingRef ifrt_sharding;
  bool is_fully_replicated;
  if (is_pmap_sharding) {
    CHECK(!shard_fns.empty());
    // IFRT Sharding will be determined once we discover the shard shape.
    is_fully_replicated = false;
  } else {
    TF_ASSIGN_OR_RETURN(ifrt_sharding,
                        GetIfrtHloSharding(sharding, ifrt_shape));
    // Fully-replicated shardings enable additional optimizations of using a
    // single host buffer.
    // TODO(hyeontaek): Enable a similar optimization for partially replicated
    // cases to reduce the number of host buffers to obtain.
    is_fully_replicated = ifrt_sharding->IsFullyReplicated();
  }
  tsl::RCReference<ifrt::UserContext> ifrt_user_context =
      ifrt_client->CreateUserContext();

  nb::gil_scoped_release gil_release;

  // Whether to build an IFRT array from host buffers as a single batch. We do
  // not batch any shard is already an IFRT array.
  bool should_batch = true;

  std::vector<Shard> shards;
  shards.reserve(shard_fns.size());
  for (int64_t i = 0; i < shard_fns.size(); ++i) {
    TF_ASSIGN_OR_RETURN(Shard shard, std::move(shard_fns[i])());
    if (shard.is_ifrt_array()) {
      // If any shard is an IFRT array, we should assemble shards.
      should_batch = false;
    }
    shards.push_back(std::move(shard));
    if (should_batch && is_fully_replicated) {
      // We need only one host buffer for a fully-replicated array.
      break;
    }
  }
  // While we have finished calling `shard_fns`, we cannot destroy them until we
  // make a call to IFRT array creation. Destroying `shard_fns` would release
  // host buffers prematurely and can cause the array creation API to see
  // garbage data.

  // TODO(emilyaf): Remove the following and just use ifrt_dtype when tokens are
  // supported.
  if (!shards.empty()) {
    ifrt_dtype = shards.front().ifrt_dtype();
  }
  if (is_pmap_sharding) {
    ifrt_sharding = ifrt::ConcreteEvenSharding::Create(
        ifrt::DeviceListRef(tsl::FormRef(ifrt_addressable_device_list)),
        ifrt_memory_kind, ifrt_shape,
        /*shard_shape=*/shards.front().ifrt_shape(),
        /*is_fully_replicated=*/false);
  }

  ifrt::ArrayRef ifrt_array;
  if (should_batch) {
    if (is_fully_replicated && shards.size() == 1) {
      ++GetDevicePutInfo().device_put_fully_replicated;
      TF_ASSIGN_OR_RETURN(
          ifrt_array, MakeIfrtArrayFromFullyReplicatedShard(
                          ifrt_client, std::move(ifrt_sharding), shards.front(),
                          std::move(ifrt_user_context)));
    } else {
      ++GetDevicePutInfo().device_put_batched;
      TF_ASSIGN_OR_RETURN(ifrt_array,
                          MakeIfrtArrayFromShardsInBatch(
                              ifrt_client, ifrt_dtype, std::move(ifrt_shape),
                              std::move(ifrt_sharding), absl::MakeSpan(shards),
                              std::move(ifrt_user_context)));
    }
  } else {
    ++GetDevicePutInfo().device_put_assembled;
    TF_ASSIGN_OR_RETURN(
        ifrt_array, MakeIfrtArrayFromShardsWithAssembly(
                        ifrt_client, ifrt_dtype, std::move(ifrt_shape),
                        std::move(ifrt_sharding), ifrt_addressable_device_list,
                        ifrt_memory_kind, absl::MakeSpan(shards),
                        std::move(ifrt_user_context)));
  }
  const bool weak_type = shards.empty() ? false : shards.front().weak_type;
  return DevicePutResult(std::move(ifrt_array), weak_type);
}

std::unordered_map<std::string, int64_t> DevicePutInfo::GetInfo() {
  const DevicePutInfo& info = GetDevicePutInfo();
  return std::unordered_map<std::string, int64_t>({
      {"device_put_with_device", info.device_put_with_device},
      {"device_put_with_sharding", info.device_put_with_sharding},
      {"device_put_fully_replicated", info.device_put_fully_replicated},
      {"device_put_batched", info.device_put_batched},
      {"device_put_assembled", info.device_put_assembled},
  });
}

}  // namespace xla
