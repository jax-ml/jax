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

#include "jaxlib/py_array.h"

#include <Python.h>
#include <structmember.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/free_threading.h"
#include "jaxlib/guard_lib.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/numpy.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device.h"
#include "jaxlib/py_device_list.h"
#include "jaxlib/py_user_context.h"
#include "jaxlib/py_values.h"
#include "jaxlib/python_ref_manager.h"
#include "jaxlib/sharding.h"
#include "jaxlib/to_ifrt_sharding.h"
#include "jaxlib/traceback.h"
#include "jaxlib/util.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/lru_cache.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/status_casters.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_helpers.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/safe_static_init.h"
#include "xla/python/strides.h"
#include "xla/python/types.h"
#include "xla/python/version.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/python/lib/core/numpy.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace ifrt = ::xla::ifrt;
namespace nb = nanobind;

namespace jax {
namespace {

nb::object& tracer_class = *new nb::object();

xla::PjRtBuffer* GetPjrtBuffer(ifrt::Array* ifrt_array) {
  auto* arr =
      xla::ifrt::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr == nullptr) {
    throw xla::XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  return arr->pjrt_buffers().front().get();
}

ifrt::ArrayRef CreateIfRtArrayFromSingleDeviceShardedPyArrays(
    xla::nb_dtype dtype, absl::Span<const int64_t> shape,
    absl::Span<const PyArray> py_arrays, const nb::object& sharding) {
  const ifrt::MemoryKind dst_memory_kind = GetMemoryKind(sharding);

  std::vector<ifrt::ArrayRef> ifrt_arrays;
  ifrt_arrays.reserve(py_arrays.size());
  absl::InlinedVector<ifrt::Device*, 1> devices;
  devices.reserve(py_arrays.size());
  absl::flat_hash_set<ifrt::Device*> device_set;
  device_set.reserve(py_arrays.size());
  std::vector<ifrt::Shape> shapes;
  shapes.reserve(py_arrays.size());

  auto sharding_device_list = GetIfrtDeviceList(sharding);
  if (!sharding_device_list.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(sharding_device_list.status().ToString().c_str());
  }
  ifrt::Device* device = sharding_device_list.value()->devices().front();

#if JAX_IFRT_VERSION_NUMBER >= 64
  const ifrt::MemoryKind& canonical_dst_memory_kind = dst_memory_kind;
#else
  // TODO(hyeontaek): Canonicalize every `ifrt::MemoryKind` at creation time to
  // skip canonicalization here once JAX begins to do it for JAX shardings.
  const ifrt::MemoryKind canonical_dst_memory_kind =
      ifrt::CanonicalizeMemoryKind(dst_memory_kind, device);
#endif
  for (const auto& py_array : py_arrays) {
    if (py_array.num_shards() != 1) {
      throw nb::value_error(
          absl::StrFormat(
              "When making an array from single-device arrays the input arrays "
              "must have one shard each. An argument array had %d shard(s).",
              py_array.num_shards())
              .c_str());
    }
    xla::ifrt::ArrayRef ifrt_arr = xla::ValueOrThrow(
        py_array.GetIfrtArray("make_array_from_single_device_arrays"));
    ifrt_arrays.push_back(ifrt_arr);
    ifrt::Device* const device =
        ifrt_arrays.back()->sharding().devices()->devices().front();
    devices.push_back(device);
    device_set.insert(device);
    shapes.push_back(ifrt_arrays.back()->shape());
    if (canonical_dst_memory_kind !=
        ifrt::CanonicalizeMemoryKind(
            ifrt_arrays.back()->sharding().memory_kind(), device)) {
      throw nb::value_error(
          absl::StrFormat(
              "Memory kind mismatch with xla::PjRtBuffers. Got sharding with "
              "memory kind '%v' and a buffer with memory_kind '%v'",
              dst_memory_kind, ifrt_arrays.back()->sharding().memory_kind())
              .c_str());
    }
  }
  ifrt::DeviceListRef device_list =
      xla::ValueOrThrow(device->client()->MakeDeviceList(devices));
  if (device_set.size() != device_list->size()) {
    throw nb::value_error(
        absl::StrFormat(
            "When making an array from single-device arrays, the input arrays "
            "must be from distinct devices, but got %v",
            *device_list)
            .c_str());
  }

  auto ifrt_dtype = DtypeToIfRtDType(dtype);
  if (!ifrt_dtype.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(ifrt_dtype.status().ToString().c_str());
  }

  absl::StatusOr<ifrt::ShardingRef> ifrt_sharding =
      GetIfrtHloSharding(sharding, ifrt::Shape(shape));
  if (!ifrt_sharding.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(ifrt_sharding.status().ToString().c_str());
  }
  // TODO(emilyaf): Always use `ifrt_dtype` once tokens are handled correctly.
  ifrt::DType array_dtype =
      ifrt_arrays.empty() ? ifrt_dtype.value() : ifrt_arrays[0]->dtype();
  absl::StatusOr<ifrt::ArrayRef> ifrt_array =
      device->client()->AssembleArrayFromSingleDeviceArrays(
          array_dtype, ifrt::Shape(shape), *std::move(ifrt_sharding),
          absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput,
          ifrt::SingleDeviceShardSemantics::kAddressableShards);
  if (!ifrt_array.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(ifrt_array.status().ToString().c_str());
  }
  return *std::move(ifrt_array);
}

// Creates a shallow clone of `ifrt_array` that uses HloSharding.
absl::StatusOr<xla::ifrt::ArrayRef> RemapArrayWithHloSharding(
    xla::ifrt::Client* client, const nanobind::object& py_sharding,
    xla::ifrt::ArrayRef& ifrt_array) {
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::ShardingRef hlo_sharding,
                        GetIfrtHloSharding(py_sharding, ifrt_array->shape()));
  xla::ifrt::UserContextScope user_context_scope(ifrt_array->user_context());
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<const xla::PjRtLayout> pjrt_layout,
                        ifrt_array->pjrt_layout());
  ABSL_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::ArrayRef> remapped,
      client->RemapArrays(
          xla::ifrt::RemapPlan(
              /*input_specs=*/{xla::ifrt::ArraySpec{
                  /*dtype=*/ifrt_array->dtype(),
                  /*shape=*/ifrt_array->shape(),
                  /*sharding=*/ifrt_array->shared_ptr_sharding(),
                  /*layout=*/pjrt_layout}},
              /*output_specs=*/
              {xla::ifrt::ArraySpec{/*dtype=*/ifrt_array->dtype(),
                                    /*shape=*/ifrt_array->shape(),
                                    /*sharding=*/std::move(hlo_sharding),
                                    /*layout=*/std::move(pjrt_layout)}},
              /*input_devices_for_output_map=*/
              {{0,
                {xla::ifrt::RemapPlan::InputDeviceRange{
                    .in_array = 0,
                    .input_devices =
                        ifrt_array->shared_ptr_sharding()->devices()}}}}),
          absl::MakeSpan(&ifrt_array, 1),
          xla::ifrt::ArrayCopySemantics::kReuseInput));
  return std::move(remapped.front());
}

struct PyBaseArrayObject {
  PyObject_HEAD;
};

extern "C" void PyBaseArray_tp_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  PyTypeObject* tp = Py_TYPE(self);
  tp->tp_free(self);
  Py_DECREF(tp);
}

extern "C" int PyBaseArray_tp_traverse(PyObject* self, visitproc visit,
                                       void* arg) {
  Py_VISIT(Py_TYPE(self));
  return 0;
}

struct PyArrayObject {
  PyObject_HEAD;
  bool initialized;
  alignas(PyArray::Storage) char array_storage[sizeof(PyArray::Storage)];
};
static_assert(std::is_standard_layout<PyArrayObject>::value);

PyArray::Storage* GetPyArrayStorageFromObject(PyArrayObject* py_array_object) {
  return std::launder(
      reinterpret_cast<PyArray::Storage*>(py_array_object->array_storage));
}

extern "C" PyObject* PyArray_tp_new(PyTypeObject* type, PyObject*, PyObject*) {
  PyObject* self = type->tp_alloc(type, 0);
#ifdef Py_GIL_DISABLED
  PyUnstable_EnableTryIncRef(self);
#endif
  auto* obj = reinterpret_cast<PyArrayObject*>(self);
  obj->initialized = false;
  return self;
}

extern "C" void PyArray_tp_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  PyTypeObject* tp = Py_TYPE(self);
  auto* obj = reinterpret_cast<PyArrayObject*>(self);

  if (obj->initialized) {
    GetPyArrayStorageFromObject(obj)->~PyArray_Storage();
  }

  PyObject_ClearWeakRefs(self);
#if PY_VERSION_HEX < 0x030D0000
  _PyObject_ClearManagedDict(self);
#else
  PyObject_ClearManagedDict(self);
#endif

  tp->tp_free(self);
  Py_DECREF(tp);
}

// dynamic_attr: Allow the garbage collector to traverse the internal instance
// `__dict__`.
extern "C" int PyArray_tp_traverse(PyObject* self, visitproc visit, void* arg) {
#if PY_VERSION_HEX < 0x030D0000
  _PyObject_VisitManagedDict(self, visit, arg);
#else
  PyObject_VisitManagedDict(self, visit, arg);
#endif
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
  Py_VISIT(Py_TYPE(self));
  return 0;
}

extern "C" void PyArray_tp_finalize(PyObject* self) {
  // This method assumes that `PyObject_CallFinalizerFromDealloc` is not called
  // from `PyArray_tp_dealloc`. If this assumption is violated, then the garbage
  // collector guard would trigger for an array deallocated via reference
  // counting.
  switch (auto guard_level = GetGarbageCollectArrayGuard(); guard_level) {
    case GarbageCollectionGuardLevel::kAllow:
      break;
    case GarbageCollectionGuardLevel::kLog:
    case GarbageCollectionGuardLevel::kFatal: {
      auto* obj = reinterpret_cast<PyArrayObject*>(self);
      std::string traceback_str;
      if (obj->initialized) {
        auto* storage = GetPyArrayStorageFromObject(obj);
        xla::ifrt::ArrayRef ifrt_array;
        {
          ft_lock_guard lock(storage->mu);
          ifrt_array = storage->ifrt_array;
        }
        if (ifrt_array != nullptr) {
          std::optional<Traceback> traceback =
              GetTraceback(ifrt_array->user_context().get());
          if (traceback.has_value()) {
            traceback_str = traceback->ToString();
          }
        }
      }
      auto error_msg = absl::StrCat(
          "`jax.Array` was deleted by the Python garbage collector "
          "instead of reference counting. Break the reference cycle "
          "that delays the deletion of this `jax.Array` to avoid hogging "
          "memory. Traceback: \n",
          traceback_str.empty() ? "not available" : traceback_str);
      if (guard_level == GarbageCollectionGuardLevel::kFatal) {
        Py_FatalError(error_msg.c_str());
      } else {
        PyObject* exc = PyErr_GetRaisedException();
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        PyErr_Print();
        PyErr_Clear();
        PyErr_SetRaisedException(exc);
      }
      break;
    }
  }
}

// dynamic_attr: Allow the GC to clear the dictionary.
extern "C" int PyArray_tp_clear(PyObject* self) {
#if PY_VERSION_HEX < 0x030D0000
  _PyObject_ClearManagedDict(self);
#else
  PyObject_ClearManagedDict(self);
#endif
  return 0;
}

template <typename... Args>
PyArray::Storage* Construct(PyArrayObject* self, Args&&... args) {
  if (self->initialized) {
    throw nb::value_error("PyArray has already been initialized.");
  }
  PyArray::Storage* out =
      new (self->array_storage) PyArray::Storage(std::forward<Args>(args)...);
  self->initialized = true;
  return out;
}

struct ShapedArrayCacheKey {
  std::vector<int64_t> dims;
  ifrt::DType dtype{ifrt::DType::kInvalid};
  bool weak_type;

  template <typename H>
  friend H AbslHashValue(H h, const ShapedArrayCacheKey& value) {
    return H::combine(std::move(h), value.dims, value.dtype, value.weak_type);
  }
  bool operator==(const ShapedArrayCacheKey& other) const {
    return dims == other.dims && dtype == other.dtype &&
           weak_type == other.weak_type;
  }
};

// Constructing ShapedArrays has gotten slow. Cache it.
nb::object MakeShapedArrayCached(const ShapedArrayCacheKey& key) {
  using CacheT = xla::LRUCache<ShapedArrayCacheKey,
                               std::shared_ptr<std::optional<nb::object>>>;
  static ft_mutex mu;
  static auto* lru_list = new CacheT::LRUList(4096);
  static auto* cache ABSL_GUARDED_BY(mu) = new CacheT(lru_list);

  static xla::SafeStatic<nb::object> shaped_array_init;
  const nb::object& shaped_array = shaped_array_init.Get([]() {
    nb::object jax_core;
    try {
      jax_core = nb::module_::import_("jax.core");
    } catch (nb::python_error& e) {
      return std::make_unique<nb::object>();
    }
    return std::make_unique<nb::object>(jax_core.attr("ShapedArray"));
  });
  if (!shaped_array.ptr()) {
    return nb::none();
  }

  std::shared_ptr<std::optional<nb::object>> value;
  {
    ft_lock_guard lock(mu);
    value = cache->GetOrCreateIfAbsent(key, [](const ShapedArrayCacheKey& key) {
      return std::make_shared<std::optional<nb::object>>();
    });
    if (value->has_value()) {
      return **value;
    }
  }

  xla::nb_dtype dtype =
      xla::IfrtDtypeToDtypeWithTokenCanonicalization(key.dtype).value();
  nb::object aval = shaped_array(
      xla::SpanToNbTuple(absl::Span<const int64_t>(
          key.dtype.kind() == ifrt::DType::kToken ? std::vector<int64_t>{0}
                                                  : key.dims)),
      dtype, key.weak_type);

  ft_lock_guard lock(mu);
  if (!value->has_value()) {
    *value = aval;
  }
  return **value;
}

// Grouping key used by BatchedCopyToDeviceWithSharding.
// Defined outside of the function as required by templatized function
// `AbslHashValue`.
struct BatchedCopyToDeviceWithShardingKey {
  ifrt::DeviceListRef src_devices;
  ifrt::MemoryKind src_memory_kind;
  ifrt::DeviceListRef dst_devices;
  ifrt::MemoryKind dst_memory_kind;
  ifrt::ArrayCopySemantics array_copy_semantics;

  bool operator==(const BatchedCopyToDeviceWithShardingKey& other) const {
    return *src_devices == *other.src_devices &&
           src_memory_kind == other.src_memory_kind &&
           *dst_devices == *other.dst_devices &&
           dst_memory_kind == other.dst_memory_kind &&
           array_copy_semantics == other.array_copy_semantics;
  }

  template <typename H>
  friend H AbslHashValue(H h, const BatchedCopyToDeviceWithShardingKey& key) {
    return H::combine(std::move(h), key.src_devices, key.src_memory_kind,
                      key.dst_devices, key.dst_memory_kind,
                      key.array_copy_semantics);
  }
};

}  // namespace

PyArray_Storage::PyArray_Storage(nb::object aval, bool weak_type,
                                 xla::nb_dtype dtype,
                                 std::vector<int64_t> shape,
                                 nb::object sharding, bool committed,
                                 nb_class_ptr<PyClient> py_client,
                                 ifrt::ArrayRef ifrt_array,
                                 xla::Future<> result_status)
    : py_client(std::move(py_client)),
      thread_id_bucket(
          std::hash<std::thread::id>()(std::this_thread::get_id()) %
          PyClient::kNumArraysShards),
      committed(committed),
      weak_type(weak_type),
      aval(std::move(aval)),
      dtype(std::move(dtype)),
      shape(std::move(shape)),
      sharding(std::move(sharding)),
      ifrt_array(std::move(ifrt_array)),
      result_status(std::move(result_status)) {
  static_assert(PyClient::kNumArraysShards <
                std::numeric_limits<uint8_t>::max());

  PyClient::ArraysShard& shard = this->py_client->arrays_[thread_id_bucket];
  ft_lock_guard lock(shard.mutex);
  next = shard.arrays;
  shard.arrays = this;
  if (next) {
    next->prev = this;
  }
  prev = nullptr;
}

void PyInit_helper(PyArray self, nb::object aval, nb::object sharding,
                   absl::Span<const PyArray> py_arrays, bool committed) {
  auto dtype = nb::cast<xla::nb_dtype>(aval.attr("dtype"));
  auto shape = nb::cast<std::vector<int64_t>>(aval.attr("shape"));
  auto py_device_list =
      nb::cast<const PyDeviceList*>(sharding.attr("_internal_device_list"));
  nb_class_ptr<PyClient> py_client = py_device_list->py_client();
  auto ifrt_array = CreateIfRtArrayFromSingleDeviceShardedPyArrays(
      dtype, shape, py_arrays, sharding);
  Construct(reinterpret_cast<PyArrayObject*>(self.ptr()), aval,
            nb::cast<bool>(aval.attr("weak_type")), std::move(dtype),
            std::move(shape), std::move(sharding), committed, py_client,
            std::move(ifrt_array), xla::Future<>());
}

void PyArray::PyInit(PyArray self, nb::object aval, nb::object sharding,
                     absl::Span<const PyArray> py_arrays, bool committed,
                     bool skip_checks) {
  PyUserContextScope user_context_scope;
  if (skip_checks) {
    PyInit_helper(self, aval, sharding, py_arrays, committed);
  } else {
    nb::object rearranged_arrays = CheckAndRearrange(py_arrays, sharding, aval);
    auto rearranged_py_arrays =
        nb::cast<std::vector<PyArray>>(rearranged_arrays);
    PyInit_helper(self, aval, sharding, rearranged_py_arrays, committed);
  }
}

PyArray PyArray::MakeFromSingleDeviceArray(nb_class_ptr<PyClient> py_client,
                                           ifrt::ArrayRef ifrt_array,
                                           bool weak_type, bool committed,
                                           xla::Future<> result_status) {
  if (!xla::ifrt::isa<ifrt::SingleDeviceSharding>(ifrt_array->sharding())) {
    throw xla::XlaRuntimeError(xla::InvalidArgument(
        "Constructing single device jax.Array from non-single "
        "device ifrt array."));
  }
  auto shape_span = ifrt_array->shape().dims();
  ShapedArrayCacheKey key;
  key.dtype = ifrt_array->dtype();
  key.dims = key.dtype.kind() == ifrt::DType::kToken
                 ? std::vector<int64_t>{0}
                 : std::vector<int64_t>(shape_span.begin(), shape_span.end());
  key.weak_type = weak_type;
  auto aval = MakeShapedArrayCached(key);
  auto dtype =
      xla::IfrtDtypeToDtypeWithTokenCanonicalization(key.dtype).value();
  const ifrt::MemoryKind memory_kind = ifrt_array->sharding().memory_kind();
  nb::object py_memory_kind =
#if JAX_IFRT_VERSION_NUMBER >= 64
      nb::str(memory_kind.value().data(), memory_kind.value().size());
#else
      (memory_kind.memory_kind().has_value())
          ? nb::object(nb::str(memory_kind.memory_kind()->data(),
                               memory_kind.memory_kind()->size()))
          : nb::none();
#endif
  nb::object sharding = MakeSingleDeviceSharding(
      py_client->GetPyDevice(
          ifrt_array->sharding().devices()->devices().front()),
      std::move(py_memory_kind));
  return PyArray(std::move(aval), weak_type, dtype, std::move(key.dims),
                 std::move(sharding), std::move(py_client),
                 std::move(ifrt_array), committed, std::move(result_status));
}

PyArray PyArray::MakeFromIfrtArrayAndSharding(nb_class_ptr<PyClient> py_client,
                                              ifrt::ArrayRef ifrt_array,
                                              nb::object sharding,
                                              bool weak_type, bool committed,
                                              bool skip_checks) {
  auto shape_span = ifrt_array->shape().dims();
  ShapedArrayCacheKey key;
  key.dtype = ifrt_array->dtype();
  key.dims = key.dtype.kind() == ifrt::DType::kToken
                 ? std::vector<int64_t>{0}
                 : std::vector<int64_t>(shape_span.begin(), shape_span.end());
  key.weak_type = weak_type;
  auto aval = MakeShapedArrayCached(key);
  auto dtype =
      xla::IfrtDtypeToDtypeWithTokenCanonicalization(key.dtype).value();
  if (!skip_checks) {
    if (ifrt_array->IsDeleted()) {
      ifrt_array = ifrt::ArrayRef();
    } else {
      auto expected_sharding =
          xla::ValueOrThrow(GetIfrtHloSharding(sharding, ifrt_array->shape()));
      if (!(*expected_sharding == ifrt_array->sharding())) {
        xla::ifrt::UserContextScope user_context_scope(
            ifrt_array->user_context());
        std::vector<PyArray> py_arrays;
        const xla::ifrt::Sharding& ifrt_sharding = ifrt_array->sharding();
        if (xla::ifrt::isa<ifrt::SingleDeviceSharding>(&ifrt_sharding) &&
            ifrt_sharding.devices()->devices().front()->IsAddressable()) {
          py_arrays.push_back(PyArray::MakeFromSingleDeviceArray(
              py_client, ifrt_array, weak_type, committed));
        } else {
          auto single_device_arrays =
              xla::ValueOrThrow(ifrt_array->DisassembleIntoSingleDeviceArrays(
                  xla::ifrt::ArrayCopySemantics::kReuseInput,
                  xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));
          py_arrays.reserve(single_device_arrays.size());
          for (auto& arr : single_device_arrays) {
            py_arrays.push_back(PyArray::MakeFromSingleDeviceArray(
                py_client, std::move(arr), weak_type, committed));
          }
        }
        nb::object rearranged_arrays =
            CheckAndRearrange(py_arrays, sharding, aval);
        auto rearranged_py_arrays =
            nb::cast<std::vector<PyArray>>(rearranged_arrays);
        std::vector<ifrt::ArrayRef> ifrt_arrays;
        ifrt_arrays.reserve(rearranged_py_arrays.size());
        for (const auto& arr : rearranged_py_arrays) {
          ifrt_arrays.push_back(xla::ValueOrThrow(arr.GetIfrtArray()));
        }
        ifrt_array = xla::ValueOrThrow(
            py_client->ifrt_client()->AssembleArrayFromSingleDeviceArrays(
                ifrt_array->dtype(), ifrt_array->shape(),
                std::move(expected_sharding), absl::MakeSpan(ifrt_arrays),
                xla::ifrt::ArrayCopySemantics::kReuseInput,
                xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));
      }
    }
  }
  return PyArray(std::move(aval), weak_type, dtype, std::move(key.dims),
                 std::move(sharding), std::move(py_client),
                 std::move(ifrt_array), committed);
}

PyArrayResultHandler::PyArrayResultHandler(
    nb::object aval, nb::object sharding, bool committed,
    std::vector<nanobind::callable> wrappers)
    : aval_(std::move(aval)),
      sharding_(std::move(sharding)),
      committed_(committed),
      wrappers_(std::move(wrappers)) {
  weak_type_ = nb::cast<bool>(aval_.attr("weak_type"));
  dtype_ = nb::cast<xla::nb_dtype>(aval_.attr("dtype"));
  shape_ = nb::cast<std::vector<int64_t>>(aval_.attr("shape"));
}

nanobind::object PyArrayResultHandler::Call(
    absl::Span<const PyArray> py_arrays) const {
  auto py_device_list = GetPyDeviceList(sharding_);
  if (!py_device_list.ok()) {
    throw nb::value_error(
        absl::StrCat("Failed to get py device list from sharding: ",
                     py_device_list.status().ToString())
            .c_str());
  }
  PyUserContextScope user_context_scope;
  return Call(py_device_list.value()->py_client(),
              CreateIfRtArrayFromSingleDeviceShardedPyArrays(
                  dtype_, shape_, py_arrays, sharding_),
              xla::Future<>());
}

nanobind::object PyArrayResultHandler::Call(nb_class_ptr<PyClient> py_client,
                                            ifrt::ArrayRef ifrt_array,
                                            xla::Future<> result_status) const {
  nanobind::object result = PyArray(
      aval_, weak_type_, dtype_, shape_, sharding_, std::move(py_client),
      std::move(ifrt_array), committed_, std::move(result_status));
  for (auto& cb : wrappers_) {
    result = cb(std::move(result));
  }
  return result;
}

nanobind::object PyArrayResultHandler::Call(PyArray py_array) const {
  return Call(py_array.py_client(), xla::ValueOrThrow(py_array.GetIfrtArray()),
              xla::Future<>());
}

PyArray::PyArray(nb::object aval, bool weak_type, xla::nb_dtype dtype,
                 std::vector<int64_t> shape, nb::object sharding,
                 nb_class_ptr<PyClient> py_client, ifrt::ArrayRef ifrt_array,
                 bool committed, xla::Future<> result_status) {
  if (ifrt_array->user_context() == nullptr && Traceback::IsEnabled()) {
    throw nb::value_error(
        "Expecting an IFRT `Array` to have a user context, but got a null "
        "user context. Use `jax::PyUserContextScope` to set a user context for "
        "operations producing IFRT `Array`s.");
  }
  auto* self =
      PyArray_tp_new(reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr);
  m_ptr = self;
  Construct(reinterpret_cast<PyArrayObject*>(self), std::move(aval), weak_type,
            std::move(dtype), std::move(shape), std::move(sharding), committed,
            std::move(py_client), std::move(ifrt_array),
            std::move(result_status));
}

PyArray::Storage& PyArray::GetStorage() {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

const PyArray::Storage& PyArray::GetStorage() const {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

nb::object PyArray::CheckAndRearrange(const absl::Span<const PyArray> py_arrays,
                                      const nb::object sharding,
                                      const nb::object aval) {
  return PyArray::type().attr("_check_and_rearrange")(py_arrays, sharding,
                                                      aval);
}

nb::object PyArray::py_arrays_cached() {
  Storage& storage = GetStorage();
  xla::ifrt::ArrayRef ifrt_array;
  xla::Future<> res_status;
  {
    ft_lock_guard lock(storage.mu);
    if (!storage.py_arrays.is_none()) {
      return storage.py_arrays;
    }
    ifrt_array = storage.ifrt_array;
    res_status = storage.result_status;
  }

  if (ifrt_array == nullptr) {
    return nb::none();
  }

  // Use the user context of this array.
  xla::ifrt::UserContextScope user_context_scope(ifrt_array->user_context());
  auto ifrt_arrays = ifrt_array->DisassembleIntoSingleDeviceArrays(
      ifrt::ArrayCopySemantics::kReuseInput,
      ifrt::SingleDeviceShardSemantics::kAddressableShards);
  if (!ifrt_arrays.ok()) {
    throw nb::value_error(
        absl::StrCat("Failed to disassemble into single-device arrays: ",
                     ifrt_arrays.status().ToString())
            .c_str());
  }
  std::vector<PyArray> py_arrays;
  py_arrays.reserve(ifrt_arrays->size());
  for (auto& single_ifrt_array : *ifrt_arrays) {
    py_arrays.push_back(PyArray::MakeFromSingleDeviceArray(
        py_client(), std::move(single_ifrt_array), weak_type(), committed(),
        res_status));
  }
  nb::object py_arrays_obj = nb::cast(py_arrays);

  ft_lock_guard lock(storage.mu);
  if (storage.ifrt_array == nullptr) {
    return nb::none();
  }
  if (storage.ifrt_array != ifrt_array) {
    throw nb::value_error("Array was mutated concurrently.");
  }
  if (storage.py_arrays.is_none()) {
    storage.py_arrays = std::move(py_arrays_obj);
  }
  return storage.py_arrays;
}

nb::object PyArray::arrays() {
  // For performance, we only keep pjrt buffers by default. But on python side
  // "_arrays" returns PyArrays instead, and subsequent calls to "_arrays"
  // should return the same PyArrays (to avoid duplicate device to host
  // transfers). So we create PyArrays the first time it is called and reuse
  // them later.
  xla::ifrt::ArrayRef ifrt_arr = ifrt_array_ref();
  if (ifrt_arr == nullptr || ifrt_arr->IsDeleted()) {
    return nb::none();
  }

  const xla::ifrt::Sharding& sharding = ifrt_arr->sharding();
  if (xla::ifrt::isa<ifrt::SingleDeviceSharding>(&sharding) &&
      sharding.devices()->devices().front()->IsAddressable()) {
    std::vector<PyArray> py_arrays;
    py_arrays.push_back(*this);
    return nb::cast(py_arrays);
  }

  return py_arrays_cached();
}

absl::StatusOr<PyArray> PyArray::FullyReplicatedShard() {
  Storage& storage = GetStorage();
  xla::ifrt::ArrayRef ifrt_array;
  xla::Future<> res_status;
  {
    ft_lock_guard lock(storage.mu);
    if (!storage.fully_replicated_array.is_none()) {
      return nb::cast<PyArray>(storage.fully_replicated_array);
    }
    ifrt_array = storage.ifrt_array;
    res_status = storage.result_status;
  }

  if (ifrt_array == nullptr) {
    return xla::InvalidArgument(
        "FullyReplicatedShard() called on deleted or donated buffer");
  }

  // Use the user context of this array.
  xla::ifrt::UserContextScope user_context_scope(ifrt_array->user_context());
  ABSL_ASSIGN_OR_RETURN(
      auto fully_replicated_ifrt_shard,
      ifrt_array->FullyReplicatedShard(ifrt::ArrayCopySemantics::kReuseInput));
  auto array = MakeFromSingleDeviceArray(py_client(),
                                         std::move(fully_replicated_ifrt_shard),
                                         weak_type(), committed(), res_status);
  {
    ft_lock_guard lock(storage.mu);
    if (storage.ifrt_array == nullptr) {
      return xla::InvalidArgument(
          "FullyReplicatedShard() called on deleted or donated buffer");
    }
    if (storage.ifrt_array != ifrt_array) {
      return xla::InvalidArgument("Array was mutated concurrently.");
    }
    if (storage.fully_replicated_array.is_none()) {
      storage.fully_replicated_array = array;
    }
    return nb::cast<PyArray>(storage.fully_replicated_array);
  }
}

absl::Status PyArray::BlockUntilReady() const {
  PyUserContextScope user_context_scope;
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_array,
                        GetIfrtArray("BlockHostUntilReady()"));
  absl::Status status;
  {
    nb::gil_scoped_release gil_release;
    xla::ifrt::Array* ifrt_array_ptr = ifrt_array.get();
    status = AwaitBuffersReady(absl::MakeConstSpan(&ifrt_array_ptr, 1));
  }
  // The array ready future can reference an asynchronously propagated
  // `ifrt::UserContext` representing the context of an error. We expand this
  // future result right before returning it to Python (outside of
  // `nb::gil_scoped_release`) so that any attached user context is appended to
  // the status message.
  return xla::ifrt::ExpandUserContexts(std::move(status));
}

absl::StatusOr<size_t> PyArray::GetOnDeviceSizeInBytes() {
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_array,
                        GetIfrtArray("GetOnDeviceSizeInBytes()"));
  // TODO(emilyaf): Support this method for non-addressable arrays by calling
  // py_client()->pjrt_client()->GetOnDeviceBytesCount once all clients
  // implement it.
  if (ifrt_array->sharding().devices()->AddressableDeviceList()->empty()) {
    return xla::Unimplemented(
        "GetOnDeviceSizeInBytes() is not yet supported for arrays with no "
        "addressable devices");
  }
  ABSL_ASSIGN_OR_RETURN(std::optional<int64_t> byte_size,
                        ifrt_array->ByteSize());
  if (!byte_size.has_value()) {
    return xla::Unimplemented(
        "GetOnDeviceSizeInBytes() is not supported for arrays with no defined "
        "byte size");
  }
  return static_cast<size_t>(*byte_size *
                             ifrt_array->sharding().devices()->size());
}

absl::Status PyArray::BlockUntilResultStatusIsReady() {
  xla::Future<> result_status = this->result_status();
  // If the result_status future is not valid, this result did not come directly
  // from a computation that returns tokens, so we don't wait for the status.
  if (!result_status.IsValid()) {
    return absl::OkStatus();
  }
  absl::Status status;
  if (!result_status.IsReady()) {
    // Only release the gil if we need to Await().
    nb::gil_scoped_release release_gil;
    BlockUntilReadyWithCancel(result_status);
    status = result_status.Await();
  } else {
    status = result_status.Await();
  }
  // `result_status` originates from `ifrt::ExecuteResult::status`, which can
  // reference an asynchronously propagated `ifrt::UserContext` representing the
  // context of an error. We expand this future result right before returning it
  // to Python (outside of `nb::gil_scoped_release`) so that any attached user
  // context is appended to the status message.
  return xla::ifrt::ExpandUserContexts(std::move(status));
}

absl::StatusOr<nb::object> PyArray::SingleDeviceArrayToNumpyArray() {
  ABSL_ASSIGN_OR_RETURN(auto result, SingleDeviceArrayToNumpyArrayDidCopy());
  return result.first;
}

absl::StatusOr<std::vector<int64_t>> PyArray::dynamic_shape() const {
  Storage& storage = const_cast<PyArray*>(this)->GetStorage();
  {
    ft_lock_guard lock(storage.mu);
    if (storage.dynamic_shape.has_value()) {
      return *storage.dynamic_shape;
    }
  }

  ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_array, GetIfrtArray());
  auto* arr =
      xla::ifrt::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array.get());
  if (arr == nullptr || !ifrt_array->sharding().IsFullyReplicated()) {
    // Skip querying the dynamic shape for a non-PjRt or sharded Array.
    std::vector<int64_t> dims(ifrt_array->shape().dims().begin(),
                              ifrt_array->shape().dims().end());
    ft_lock_guard lock(storage.mu);
    if (storage.ifrt_array == nullptr || storage.ifrt_array->IsDeleted()) {
      return xla::InvalidArgument("Array has been deleted.");
    }
    if (storage.ifrt_array != ifrt_array) {
      return xla::InvalidArgument("Array was mutated concurrently.");
    }
    if (!storage.dynamic_shape.has_value()) {
      storage.dynamic_shape = dims;
    }
    return dims;
  }

  auto* pjrt_buffer = arr->pjrt_buffers().front().get();
  std::vector<int64_t> dims;
  if (pjrt_buffer->has_dynamic_dimensions()) {
    nb::gil_scoped_release gil_release;
    ABSL_ASSIGN_OR_RETURN(dims, pjrt_buffer->logical_dimensions());
  } else {
    dims.assign(pjrt_buffer->dimensions().begin(),
                pjrt_buffer->dimensions().end());
  }

  ft_lock_guard lock(storage.mu);
  if (storage.ifrt_array == nullptr || storage.ifrt_array->IsDeleted()) {
    return xla::InvalidArgument("Array has been deleted.");
  }
  if (storage.ifrt_array != ifrt_array) {
    return xla::InvalidArgument("Array was mutated concurrently.");
  }
  if (!storage.dynamic_shape.has_value()) {
    storage.dynamic_shape = dims;
  }
  return *storage.dynamic_shape;
}

absl::StatusOr<PyArray> PyArray::AssertUnsharded(std::string_view api) {
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_arr, GetIfrtArray(api));

  if (xla::ifrt::isa<ifrt::SingleDeviceSharding>(&ifrt_arr->sharding())) {
    return *this;
  }

  nb::object py_arrays = py_arrays_cached();
  if (py_arrays.is_none()) {
    return xla::InvalidArgument("%s called on deleted or donated buffer", api);
  }
  if (nb::len(py_arrays) != 1) {
    return xla::InvalidArgument("%s() is supported only for unsharded arrays.",
                                api);
  }
  return nb::cast<PyArray>(py_arrays[0]);
}

absl::StatusOr<std::uintptr_t> PyArray::UnsafeBufferPointer() {
  ABSL_ASSIGN_OR_RETURN(auto arr, AssertUnsharded("UnsafeBufferPointer"));
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_array,
                        arr.GetIfrtArray("UnsafeBufferPointer"));
  return py_client()->pjrt_client()->UnsafeBufferPointer(
      GetPjrtBuffer(ifrt_array.get()));
}

nb::dict PyArray::CudaArrayInterface() {
  auto arr_or_error = AssertUnsharded("__cuda_array_interface__");
  if (!arr_or_error.ok()) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only supported for unsharded arrays.");
  }
  auto arr = *arr_or_error;

  xla::ifrt::ArrayRef ifrt_array =
      xla::ValueOrThrow(arr.GetIfrtArray("__cuda_array_interface__"));
  auto* pjrt_buffer = GetPjrtBuffer(ifrt_array.get());
  if (pjrt_buffer->client()->platform_id() != xla::CudaId() &&
      pjrt_buffer->client()->platform_id() != xla::RocmId()) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only defined for GPU buffers.");
  }
  if (pjrt_buffer->IsTuple()) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only defined for array buffers.");
  }

  switch (pjrt_buffer->element_type()) {
    case xla::PrimitiveType::PRED:
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U8:
    case xla::PrimitiveType::U16:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::U64:
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::F32:
    case xla::PrimitiveType::F64:
    case xla::PrimitiveType::C64:
    case xla::PrimitiveType::C128:
      break;

    default:
      throw nb::attribute_error(
          absl::StrFormat(
              "__cuda_array_interface__ is not supported for %s buffers.",
              PrimitiveType_Name(pjrt_buffer->element_type()))
              .c_str());
  }

  nb::str typestr = xla::ValueOrThrow(
      TypeDescriptorForPrimitiveType(pjrt_buffer->element_type()));

  // TODO(b/327524065): use xla::PjRtLayout directly instead of xla::Layout
  xla::Layout xla_layout = pjrt_buffer->layout()->xla_layout();
  if (!xla::LayoutUtil::IsMonotonicWithDim0Major(xla_layout)) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only currently supported for "
        "buffers in row-major order.");
  }

  nb::dict result;
  std::vector<int64_t> dynamic_shape = xla::ValueOrThrow(arr.dynamic_shape());
  result["shape"] = xla::SpanToNbTuple(absl::MakeConstSpan(dynamic_shape));
  result["typestr"] = std::move(typestr);
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference_hold =
      xla::ValueOrThrow(pjrt_buffer->AcquireExternalReference());
  const void* root_ptr =
      external_reference_hold->OpaqueDeviceMemoryDataPointer();
  nb::tuple data =
      nb::make_tuple(nb::int_(absl::bit_cast<std::uintptr_t>(root_ptr)),
                     nb::bool_(true) /* read-only */
      );
  result["data"] = std::move(data);
  result["version"] = nb::int_(2);
  return result;
}

absl::StatusOr<nb::object> CudaArrayInterfaceToBuffer(
    const nb::dict& cai, nb_class_ptr<PyClient> client,
    std::optional<int> device_id) {
  if (!cai.contains("data")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `data`");
  }
  if (!cai.contains("shape")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `shape`");
  }
  if (!cai.contains("typestr")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `typestr`");
  }
  if (!cai.contains("version")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `version`");
  }
  auto version = nb::cast<int>(cai["version"]);
  if (version < 2 || version > 3) {
    LOG(WARNING) << "CUDA Array Interface version " << version
                 << " support is undefined";
  }
  auto data = nb::cast<nb::tuple>(cai["data"]);
  auto data_value = nb::cast<std::intptr_t>(data[0]);
  void* data_ptr = reinterpret_cast<void*>(data_value);
  auto dimensions = nb::cast<std::vector<int64_t>>(cai["shape"]);
  if (data_value == 0 && absl::c_find(dimensions, 0) == dimensions.end()) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface `data`(=NULL) and `shape`(no zero-valued "
        "dimensions) are inconsistent");
  }
  auto ndim = dimensions.size();
  ABSL_ASSIGN_OR_RETURN(
      xla::PrimitiveType element_type,
      DtypeToPrimitiveType(xla::nb_dtype::from_args(cai["typestr"])));

  if (!device_id.has_value()) {
    throw xla::XlaRuntimeError(
        "This operation requires CUDA support from jaxlib or jax cuda plugin.");
  }
  ABSL_ASSIGN_OR_RETURN(auto device,
                        client->DeviceFromLocalHardwareId(*device_id));
  bool is_default_stream =
      data_value == 0 || version == 2 ||
      (version == 3 && (!cai.contains("stream") || cai["stream"].is_none()));
  ABSL_ASSIGN_OR_RETURN(
      std::intptr_t stream,
      ([is_default_stream, cai, device]() -> absl::StatusOr<std::intptr_t> {
        if (is_default_stream) {
          return device->GetStreamForExternalReadyEvents();
        } else {
          auto stream_ = nb::cast<std::intptr_t>(cai["stream"]);
          if (stream_ == 0) {
            return absl::InvalidArgumentError(
                "CUDA Array Interface does not allow zero stream value");
          }
          return stream_;
        }
      }()));

  bool has_custom_layout;
  std::vector<int64_t> minor_to_major(ndim);
  if (cai.contains("strides") && !cai["strides"].is_none() && data_value != 0) {
    std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
    auto strides = nb::cast<std::vector<int64_t>>(cai["strides"]);
    if (strides.size() != ndim) {
      return absl::InvalidArgumentError(
          "CUDA Array Interface `shape` and `strides` dimensionalities are "
          "inconsistent");
    }
    has_custom_layout = true;
    absl::c_sort(minor_to_major, [&](int a, int b) {
      // If two dimensions have the same stride, prefer the major-to-minor
      // interpretation of the ordering, since that's what JAX wants.
      return (strides[a] == strides[b] ? b < a : strides[a] < strides[b]);
    });
    int64_t stride = xla::ShapeUtil::ByteSizeOfPrimitiveType(element_type);
    for (int64_t d : minor_to_major) {
      if (dimensions[d] > 1 && strides[d] != stride) {
        return absl::UnimplementedError(absl::StrCat(
            "Only arrays with trivial (compact) striding are supported; "
            "i.e., arrays whose striding represents a transposition of the "
            "underlying buffer but not broadcasting. Dimensions were: [%s], "
            "strides were [%s].",
            absl::StrJoin(dimensions, ","), absl::StrJoin(strides, ",")));
      }
      stride *= dimensions[d];
    }
  } else {
    has_custom_layout = false;
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      element_type, dimensions, minor_to_major);
  std::function<void()> on_delete_callback = []() {};
  auto* pjrt_device =
      xla::ifrt::dyn_cast_or_null<ifrt::PjRtDevice>(device->device());
  if (pjrt_device == nullptr) {
    return xla::InvalidArgument(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  TF_RET_CHECK(pjrt_device->IsAddressable());
  ABSL_ASSIGN_OR_RETURN(
      auto pjrt_buffer,
      device->client()->pjrt_client()->CreateViewOfDeviceBuffer(
          static_cast<char*>(data_ptr), shape,
          *pjrt_device->pjrt_device()->default_memory_space(),
          on_delete_callback,
          stream <= 2 ? std::nullopt : std::make_optional(stream)));
  auto* ifrt_client = xla::ifrt::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(
      client->ifrt_client());
  if (ifrt_client == nullptr) {
    throw xla::XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  PyUserContextScope user_context_scope;
  ABSL_ASSIGN_OR_RETURN(
      auto ifrt_array,
      ifrt_client->CreatePjRtArray(std::move(pjrt_buffer), has_custom_layout));
  return PyArray::MakeFromSingleDeviceArray(std::move(client),
                                            std::move(ifrt_array), false, true);
}

absl::StatusOr<xla::ifrt::ArrayRef> PyArray::GetIfrtArray(
    std::string_view api) const {
  xla::ifrt::ArrayRef ifrt_array_ptr = ifrt_array_ref();
  if (ifrt_array_ptr == nullptr || ifrt_array_ptr->IsDeleted()) {
    if (api.empty()) {
      return xla::InvalidArgument("Array has been deleted.");
    }
    return xla::InvalidArgument("%s called on deleted or donated buffer", api);
  }
  return ifrt_array_ptr;
}

absl::Status PyArray::Delete() {
  nanobind::object py_arrays;
  nanobind::object old_npy_value;
  nanobind::object old_fully_replicated_array;
  PyHostValue old_host_value;
  xla::ifrt::ArrayRef ifrt_arr;
  {
    Storage& storage = GetStorage();
    ft_lock_guard lock(storage.mu);
    py_arrays = std::move(storage.py_arrays);
    storage.py_arrays = nb::none();
    old_npy_value = std::move(storage.npy_value);
    storage.npy_value = nb::none();
    old_fully_replicated_array = std::move(storage.fully_replicated_array);
    storage.fully_replicated_array = nb::none();
    old_host_value = std::exchange(storage.host_value, PyHostValue());
    ifrt_arr = std::move(storage.ifrt_array);
  }
  if (!py_arrays.is_none()) {
    for (nb::handle item : py_arrays) {
      ABSL_RETURN_IF_ERROR(nb::cast<PyArray>(item).Delete());
    }
  }
  if (ifrt_arr != nullptr) {
    // We do not wait for the deletion to complete here.
    //
    // (1) Skipping blocking does not affect the correctness of deletion as long
    // as the runtime preserves dispatch ordering of deletion w.r.t. other
    // operations.
    //
    // (2) Synchronously waiting for the deletion to complete is very expensive
    // when the deletion can return a status only after the underlying physical
    // buffer has been deleted or a request must be processed via RPC,
    // especially as this deletion is done per array.
    ifrt_arr->Delete();
    nb::gil_scoped_release gil_release;
    ifrt_arr.reset();
  }
  return absl::OkStatus();
}

bool PyArray::IsDeleted() const {
  xla::ifrt::ArrayRef ifrt_arr = ifrt_array_ref();
  if (ifrt_arr == nullptr) {
    return true;
  }

  return ifrt_arr->IsDeleted();
}

PyArray PyArray::Clone() const {
  xla::ifrt::ArrayRef array = xla::ValueOrThrow(GetIfrtArray("Clone()"));
  auto* ifrt_client = py_client()->ifrt_client();
  // Use the user context of this array.
  xla::ifrt::UserContextScope user_context_scope(array->user_context());
  ifrt::ArrayRef out =
      ifrt_client
          ->CopyArrays(absl::MakeSpan(&array, 1), /*devices=*/std::nullopt,
                       /*memory_kind=*/std::nullopt,
                       ifrt::ArrayCopySemantics::kReuseInput)
          .value()
          .front();
  return PyArray(aval(), weak_type(), dtype(),
                 std::vector<int64_t>(shape().begin(), shape().end()),
                 sharding(), py_client(), std::move(out), committed(),
                 result_status());
}

nb::handle PyArray::Storage::AsHandle() {
  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(PyArrayObject, array_storage));
}

PyArray::Storage::~PyArray_Storage() {
  CHECK(PyGILState_Check());
  if (py_client) {
    PyClient::ArraysShard& shard = py_client->arrays_[thread_id_bucket];
    ft_lock_guard lock(shard.mutex);
    if (shard.arrays == this) {
      shard.arrays = next;
    }
    if (prev) {
      prev->next = next;
    }
    if (next) {
      next->prev = prev;
    }
  }
  // Release GIL and then explicitly destroy `ifrt_array` to prevent deadlock on
  // CPU backend caused by interactions between argument donations and host
  // callbacks.
  nb::gil_scoped_release gil_release;
  ifrt_array.reset();
}

absl::StatusOr<std::vector<PyArray>> PyArray::BatchedCopyToDeviceWithSharding(
    absl::Span<const PyArray> py_arrays,
    absl::Span<const ifrt::DeviceListRef> dst_device_lists,
    absl::Span<const nb::object> dst_shardings,
    absl::Span<const ifrt::ArrayCopySemantics> array_copy_semantics) {
  if (py_arrays.empty()) {
    return std::vector<PyArray>();
  }

  TF_RET_CHECK(py_arrays.size() == dst_device_lists.size());
  TF_RET_CHECK(py_arrays.size() == dst_shardings.size());

  ABSL_ASSIGN_OR_RETURN(
      xla::ifrt::ArrayRef front_ifrt_array,
      py_arrays.front().GetIfrtArray("BatchedCopyToDeviceWithSharding()"));
  ifrt::Client* const client = front_ifrt_array->client();
  std::vector<PyArray> results(py_arrays.size());

  // Arrays to be copied, grouped by source/destination devices and memory
  // kinds. The grouping is enforced by `ifrt::Client::CopyArrays()`.
  struct Batch {
    std::vector<int> indexes;
    std::vector<ifrt::ArrayRef> ifrt_arrays;
  };
  absl::flat_hash_map<BatchedCopyToDeviceWithShardingKey, Batch> batches;

  PyUserContextScope user_context_scope;
  {
    tsl::profiler::TraceMe results_traceme(
        "BatchedCopyToDeviceWithSharding create batch");
    for (int i = 0; i < py_arrays.size(); ++i) {
      const auto& py_array = py_arrays[i];
      const auto& dst_sharding = dst_shardings[i];
      const auto& array_cs = array_copy_semantics[i];

      ABSL_ASSIGN_OR_RETURN(
          xla::ifrt::ArrayRef ifrt_array_ptr,
          py_array.GetIfrtArray("BatchedCopyToDeviceWithSharding()"));
      const ifrt::DeviceListRef& src_devices =
          ifrt_array_ptr->sharding().devices();
      const ifrt::DeviceListRef& dst_devices = dst_device_lists[i];

#if JAX_IFRT_VERSION_NUMBER >= 64
      const ifrt::MemoryKind& src_memory_kind =
          ifrt_array_ptr->sharding().memory_kind();
      ifrt::MemoryKind dst_memory_kind = GetMemoryKind(dst_sharding);
#else
      ifrt::MemoryKind src_memory_kind =
          ifrt::CanonicalizeMemoryKind(ifrt_array_ptr->sharding().memory_kind(),
                                       src_devices->devices().front());
      ifrt::MemoryKind dst_memory_kind = ifrt::CanonicalizeMemoryKind(
          GetMemoryKind(dst_sharding), dst_devices->devices().front());
#endif

      if (*src_devices == *dst_devices && src_memory_kind == dst_memory_kind &&
          array_cs == ifrt::ArrayCopySemantics::kReuseInput) {
        if (py_array.sharding().equal(dst_sharding)) {
          results[i] = py_arrays[i];
        } else {
          absl::Span<const int64_t> shape_span = py_array.shape();
          // We can reuse the input array despite the sharding being different.
          // This is because this code expects no resharding is necessary, which
          // has been verified by the code invoking this method.
          results[i] = PyArray(
              py_array.aval(), py_array.weak_type(), py_array.dtype(),
              std::vector<int64_t>(shape_span.begin(), shape_span.end()),
              dst_sharding, py_array.py_client(), ifrt_array_ptr,
              py_array.committed(), py_array.result_status());
        }
        continue;
      }

      auto transfer_guard_formatter = [&py_array, &dst_sharding] {
        return absl::StrCat(
            "aval=", nb::cast<std::string_view>(nb::repr(py_array.aval())),
            ", sharding=",
            nb::cast<std::string_view>(nb::repr(py_array.sharding())),
            ", dst_sharding=",
            nb::cast<std::string_view>(nb::repr(dst_sharding)));
      };
      ABSL_RETURN_IF_ERROR(
          ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));

      Batch& batch = batches[BatchedCopyToDeviceWithShardingKey{
          src_devices, src_memory_kind, dst_devices, dst_memory_kind,
          array_cs}];
      batch.indexes.push_back(i);
      batch.ifrt_arrays.push_back(ifrt_array_ptr);
    }
  }

  std::vector<std::pair<int, ifrt::ArrayRef>> ifrt_arrays;
  {
    GlobalPyRefManager()->CollectGarbage();
    nb::gil_scoped_release gil_release;

    tsl::profiler::TraceMe copy_traceme(
        "BatchedCopyToDeviceWithSharding: dispatch");
    for (auto& [key, batch] : batches) {
      ABSL_ASSIGN_OR_RETURN(
          auto copied,
          client->CopyArrays(
              absl::MakeSpan(batch.ifrt_arrays),
              // All arrays in `batch` have the same `key.dst_devices` and
              // `key.dst_memory_kind` due to the grouping above.
              key.dst_devices, key.dst_memory_kind, key.array_copy_semantics));
      for (int i = 0; i < batch.indexes.size(); ++i) {
        ifrt_arrays.push_back(
            std::make_pair(batch.indexes[i], std::move(copied[i])));
      }
    }
  }

  tsl::profiler::TraceMe results_traceme(
      "BatchedCopyToDeviceWithSharding create results");
  for (auto& [i, ifrt_array] : ifrt_arrays) {
    ABSL_ASSIGN_OR_RETURN(nb_class_ptr<PyDeviceList> dst_device_list,
                          GetPyDeviceList(dst_shardings[i]));
    nb_class_ptr<PyClient> py_client = dst_device_list->py_client();
    const auto& py_array = py_arrays[i];
    absl::Span<const int64_t> shape_span = py_array.shape();
    results[i] =
        PyArray(py_array.aval(), py_array.weak_type(), py_array.dtype(),
                std::vector<int64_t>(shape_span.begin(), shape_span.end()),
                dst_shardings[i], py_client, std::move(ifrt_array),
                py_array.committed(), py_array.result_status());
  }
  return results;
}

absl::StatusOr<PyArray> PyArray::BatchedDevicePut(
    nb::object aval, nb::object sharding, std::vector<nb::object> xs,
    absl::Span<const PyDevice* const> dst_devices, bool committed,
    bool force_copy, xla::PjRtClient::HostBufferSemantics host_buffer_semantics,
    bool jax_enable_x64) {
  if (dst_devices.size() != xs.size()) {
    throw nb::value_error(
        absl::StrCat("Argument sizes (xs and devices) must match %zu vs %zu",
                     dst_devices.size(), xs.size())
            .c_str());
  }
  for (const PyDevice* device : dst_devices) {
    if (device == nullptr) {
      return xla::InvalidArgument("Device cannot be None.");
    }
    if (device->client().get() == nullptr) {
      return xla::InvalidArgument("Cannot copy to unattached devices.");
    }
  }
  auto transfer_guard_formatter = [&aval, &sharding] {
    return absl::StrCat(
        "aval=", nb::cast<std::string_view>(nb::repr(aval)),
        ", dst_sharding=", nb::cast<std::string_view>(nb::repr(sharding)));
  };

  GlobalPyRefManager()->CollectGarbage();

  PyUserContextScope user_context_scope;

  DevicePutOptions options;
  options.squash_64bit_types = !jax_enable_x64;
  options.allow_zero_copy =
      (!force_copy && (host_buffer_semantics ==
                       ifrt::Client::HostBufferSemantics::kImmutableZeroCopy));

  std::vector<nb::handle> args;
  args.reserve(xs.size());
  for (const nb::object& x : xs) {
    if (PyArray::IsPyArray(x)) {
      ABSL_RETURN_IF_ERROR(
          ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));
    } else {
      ABSL_RETURN_IF_ERROR(
          ApplyTransferGuardToHostToDevice(transfer_guard_formatter));
    }
    args.push_back(x);
  }
  auto weak_type = nb::cast<bool>(aval.attr("weak_type"));
  auto dtype = aval.attr("dtype");
  auto shape = nb::cast<std::vector<int64_t>>(aval.attr("shape"));
  ABSL_ASSIGN_OR_RETURN(nb_class_ptr<PyDeviceList> py_device_list,
                        GetPyDeviceList(sharding));

  std::vector<xla::ifrt::Device*> ifrt_devices;
  ifrt_devices.reserve(dst_devices.size());
  for (const PyDevice* device : dst_devices) {
    ifrt_devices.push_back(device->device());
  }
  ifrt::Client* ifrt_client = py_device_list->py_client()->ifrt_client();
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef ifrt_device_list,
                        ifrt_client->MakeDeviceList(ifrt_devices));
  ABSL_ASSIGN_OR_RETURN(
      DevicePutResult device_put_result,
      DevicePutWithSharding(args, ifrt_device_list, ifrt_client, dtype, shape,
                            sharding, options));

  return PyArray(aval, weak_type, dtype, std::move(shape), std::move(sharding),
                 py_device_list->py_client(),
                 std::move(device_put_result.ifrt_array), committed);
}

absl::StatusOr<PyArray> PyArray::ReorderShards(
    PyArray x, nanobind::object dst_sharding,
    ifrt::ArrayCopySemantics array_copy_semantics) {
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_array_ptr,
                        x.GetIfrtArray("Reorder()"));

  ifrt::Client* const client = ifrt_array_ptr->client();

  const auto& device_list = ifrt_array_ptr->sharding().devices();
  ABSL_ASSIGN_OR_RETURN(auto dst_device_list, GetIfrtDeviceList(dst_sharding));
  if (device_list->AddressableDeviceList()->size() !=
      dst_device_list->AddressableDeviceList()->size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Array is expected to have ",
        dst_device_list->AddressableDeviceList()->size(),
        " addressable shards, but has ",
        device_list->AddressableDeviceList()->size(), " addressable shards"));
  }

  ABSL_ASSIGN_OR_RETURN(
      xla::ifrt::ShardingRef dst_ifrt_sharding,
      GetIfrtConcreteEvenSharding(dst_sharding, ifrt_array_ptr->dtype(),
                                  ifrt_array_ptr->shape()));

  // Use the user context of this array.
  xla::ifrt::UserContextScope user_context_scope(
      ifrt_array_ptr->user_context());

  xla::ifrt::ArrayRef new_ifrt_array;
  {
    nb::gil_scoped_release gil_release;

    const absl::Span<xla::ifrt::Device* const> addressable_devices =
        device_list->AddressableDeviceList()->devices();
    const absl::Span<xla::ifrt::Device* const> dst_addressable_devices =
        dst_device_list->AddressableDeviceList()->devices();

    absl::flat_hash_map<int, int> device_id_to_array_shard_index;
    device_id_to_array_shard_index.reserve(dst_addressable_devices.size());
    for (int i = 0; i < dst_addressable_devices.size(); ++i) {
      const int device_id = dst_addressable_devices[i]->Id().value();
      const bool inserted =
          device_id_to_array_shard_index.insert({device_id, i}).second;
      if (!inserted) {
        return absl::InvalidArgumentError(
            absl::StrCat("Sharding contains duplicate device id=", device_id));
      }
    }

    for (int i = 0; i < dst_addressable_devices.size(); ++i) {
      const int shard_device_id = addressable_devices[i]->Id().value();
      const auto it = device_id_to_array_shard_index.find(shard_device_id);
      if (it == device_id_to_array_shard_index.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Array shard ", i, " is on device id=", shard_device_id,
            ", but sharding does not have a shard on that device."));
      }
    }

    absl::flat_hash_map<int,
                        std::vector<xla::ifrt::RemapPlan::InputDeviceRange>>
        input_devices_for_output_map;
    input_devices_for_output_map[0].push_back(
        xla::ifrt::RemapPlan::InputDeviceRange{
            .in_array = 0,
            .input_devices = dst_ifrt_sharding->devices(),
        });

    xla::ifrt::RemapPlan plan(
        /*input_specs=*/{xla::ifrt::ArraySpec{
            /*dtype=*/ifrt_array_ptr->dtype(),
            /*shape=*/ifrt_array_ptr->shape(),
            /*sharding=*/ifrt_array_ptr->shared_ptr_sharding()}},
        {xla::ifrt::ArraySpec{/*dtype=*/ifrt_array_ptr->dtype(),
                              /*shape=*/ifrt_array_ptr->shape(),
                              /*sharding=*/std::move(dst_ifrt_sharding)}},
        std::move(input_devices_for_output_map));
    DCHECK_OK(plan.Validate());
    std::vector<xla::ifrt::ArrayRef> input;
    input.push_back(ifrt_array_ptr);
    ABSL_ASSIGN_OR_RETURN(
        auto remapped,
        client->RemapArrays(plan, absl::MakeSpan(input), array_copy_semantics));

    TF_RET_CHECK(remapped.size() == 1);
    new_ifrt_array = std::move(remapped.front());
  }

  return PyArray(nb::borrow<nb::object>(x.aval().ptr()), x.weak_type(),
                 nb::borrow<xla::nb_dtype>(x.dtype().ptr()),
                 std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                 std::move(dst_sharding), x.py_client(),
                 std::move(new_ifrt_array),
                 /*committed=*/true);
}

absl::Status PyArray::BatchedBlockUntilReady(std::vector<nb::object> objs) {
  // Create ready futures for all arrays before blocking on their readiness.
  // This helps reduce the latency in some backend implementations where
  // querying readiness of an array is not free.

  std::vector<xla::ifrt::ArrayRef> ifrt_array_refs;
  ifrt_array_refs.reserve(objs.size());
  std::vector<ifrt::Array*> ifrt_arrays;
  ifrt_arrays.reserve(objs.size());
  for (nb::handle obj : objs) {
    if (obj.type().is(PyArray::type())) {
      auto py_array = nb::borrow<PyArray>(obj);
      ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_array,
                            py_array.GetIfrtArray("BlockHostUntilReady()"));
      ifrt_arrays.push_back(ifrt_array.get());
      ifrt_array_refs.push_back(std::move(ifrt_array));
    } else {
      return absl::InvalidArgumentError(
          "PyArray::BatchedBlockUntilReady can take PyArray only");
    }
  }

  GlobalPyRefManager()->CollectGarbage();
  PyUserContextScope user_context_scope;
  absl::Status status;
  {
    nb::gil_scoped_release gil_release;
    status = AwaitBuffersReady(absl::MakeConstSpan(ifrt_arrays));
    ifrt_array_refs.clear();
  }
  // `status` can reference an asynchronously propagated `ifrt::UserContext`
  // representing the context of an error. We expand this future result right
  // before returning it to Python (outside of `nb::gil_scoped_release`) so that
  // any attached user context is appended to the status message.
  return xla::ifrt::ExpandUserContexts(std::move(status));
}

absl::Status PyArray::ReplaceWithAlias(PyArray o)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto& storage = GetStorage();
  auto& o_storage = o.GetStorage();
  if (&storage == &o_storage) {
    return absl::InvalidArgumentError(
        "Unable to replace an Array with itself.");
  }
  if (storage.py_client.get() != o_storage.py_client.get()) {
    return absl::InvalidArgumentError(
        "Unable to replace an Array with an Array from a different client.");
  }
  if (!storage.dtype.equal(o_storage.dtype)) {
    return absl::InvalidArgumentError(
        "Unable to replace an Array with an Array of different dtype.");
  }
  if (storage.shape != o_storage.shape) {
    return absl::InvalidArgumentError(
        "Unable to replace an Array with an Array of different shape.");
  }
  if (!storage.sharding.equal(o_storage.sharding)) {
    return absl::InvalidArgumentError(
        "Unable to replace an Array with an Array of different sharding.");
  }
  if (storage.committed != o_storage.committed) {
    return absl::InvalidArgumentError(
        "Unable to replace an Array with an Array of different committed.");
  }
  if (storage.weak_type != o_storage.weak_type) {
    return absl::InvalidArgumentError(
        "Unable to replace an Array with an Array of different weak_type.");
  }
  auto* mu1 = &storage.mu;
  auto* mu2 = &o_storage.mu;
  if (mu1 > mu2) {
    std::swap(mu1, mu2);
  }
  nanobind::object old_npy_value;
  xla::ifrt::ArrayRef old_ifrt_array;
  nanobind::object old_fully_replicated_array;
  nanobind::object old_py_arrays;
  PyHostValue old_host_value;
  {
    ft_lock_guard lock1(*mu1);
    ft_lock_guard lock2(*mu2);

    if (storage.ifrt_array == nullptr || o_storage.ifrt_array == nullptr) {
      return absl::InvalidArgumentError(
          "Unable to replace an Array when one or both arrays have been "
          "deleted.");
    }

    old_npy_value = std::move(storage.npy_value);
    storage.npy_value = o_storage.npy_value;
    old_ifrt_array = std::move(storage.ifrt_array);
    storage.ifrt_array = o_storage.ifrt_array;
    old_fully_replicated_array = std::move(storage.fully_replicated_array);
    storage.fully_replicated_array = o_storage.fully_replicated_array;
    old_py_arrays = std::move(storage.py_arrays);
    storage.py_arrays = o_storage.py_arrays;
    old_host_value = std::exchange(storage.host_value, PyHostValue());
    storage.dynamic_shape = o_storage.dynamic_shape;
    storage.result_status = o_storage.result_status;
  }
  if (old_ifrt_array != nullptr) {
    nb::gil_scoped_release gil_release;
    old_ifrt_array.reset();
  }
  return absl::OkStatus();
}

std::vector<PyArray> PyClient::LiveArrays() const {
  std::vector<PyArray> result;
  for (auto& shard : arrays_) {
    ft_lock_guard lock(shard.mutex);
    for (PyArray::Storage* array = shard.arrays; array; array = array->next) {
      PyArray py_array = TryBorrow<PyArray>(array->AsHandle());
      if (!py_array.is_valid()) {
        continue;
      }
      xla::ifrt::ArrayRef ifrt_array = py_array.ifrt_array_ref();
      bool all_deleted = (ifrt_array == nullptr || ifrt_array->IsDeleted());
      if (!all_deleted) {
        result.push_back(std::move(py_array));
      }
    }
  }
  return result;
}

// PEP 3118 buffer protocol implementation.

namespace {

// Extra data to be kept alive by the consumer of the buffer protocol.
struct ExtraBufferInfo {
  explicit ExtraBufferInfo(std::shared_ptr<xla::PjRtBuffer> buffer,
                           std::unique_ptr<xla::PjRtBuffer::ExternalReference>
                               external_reference_hold)
      : buffer(std::move(buffer)),
        external_reference_hold(std::move(external_reference_hold)) {}

  std::vector<int64_t> strides;
  // We keep an external reference hold to the xla::PjRtBuffer. This prevents a
  // use-after-free in the event that Delete() is called on a buffer with an
  // live buffer protocol view. It does however mean that Delete() sometimes
  // won't actually delete immediately.
  std::shared_ptr<xla::PjRtBuffer> buffer;
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference_hold;
};

// The default layout of a non-tuple array should have major-to-minor layout
// and no tiles.
bool HasDefaultLayout(const xla::Layout& layout) {
  return xla::LayoutUtil::IsMonotonicWithDim0Major(layout) &&
         layout.tiles().empty();
}

int PyArray_bf_getbuffer(PyObject* exporter, Py_buffer* view, int flags) {
  absl::Status status = [&]() -> absl::Status {
    PyArray py_array = nb::borrow<PyArray>(exporter);
    ABSL_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef ifrt_array,
                          py_array.GetIfrtArray());
    if (!xla::ifrt::isa<ifrt::PjRtCompatibleArray>(ifrt_array.get())) {
      return xla::InvalidArgument("Only local arrays are supported, got %s",
                                  ifrt_array->DebugString());
    }
    auto* array = static_cast<ifrt::PjRtCompatibleArray*>(ifrt_array.get());
    absl::Span<const std::shared_ptr<xla::PjRtBuffer>> buffers =
        array->pjrt_buffers();

    if (buffers.empty()) {
      return xla::InvalidArgument("Array has no buffers.");
    }
    xla::PjRtBuffer& buffer = *buffers.front();
    if (!buffer.IsOnCpu()) {
      return xla::InvalidArgument(
          "Python buffer protocol is only defined for CPU buffers.");
    }

    if (buffers.size() != 1) {
      return xla::InvalidArgument(
          "Python buffer protocol is only defined for buffers with a single "
          "shard.");
    }

    if (!array->sharding().IsFullyReplicated()) {
      return xla::InvalidArgument(
          "Python buffer protocol is only defined for single-device sharded "
          "buffers.");
    }

    const char* format =
        PEP3118FormatDescriptorForPrimitiveType(buffer.element_type());
    // It isn't an option for us to export unknown types as, say, bytes. When
    // converting an object to an ndarray, NumPy tries the buffer protocol
    // first. We very much want NumPy to fail and fall back to using
    // __array__, which allows us to handle custom dtypes correctly.
    if (!format) {
      return xla::InvalidArgument(
          "Buffers of type %s are not supported by the Python buffer protocol.",
          PrimitiveType_Name(buffer.element_type()));
    }

    std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference_hold;
    {
      // We call BlockHostUntilReady() below, which may block.
      nb::gil_scoped_release gil_release;

      if (buffer.IsTuple()) {
        return xla::InvalidArgument(
            "Python buffer protocol is only defined for array buffers.");
      }
      if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
        return xla::InvalidArgument("XLA buffers are read-only.");
      }
      ABSL_ASSIGN_OR_RETURN(external_reference_hold,
                            buffer.AcquireExternalReference());
      if (buffer.IsDeleted()) {
        return xla::InvalidArgument("Deleted buffer used in buffer protocol.");
      }

      // TODO(b/327524065): use xla::PjRtLayout directly instead of xla::Layout
      xla::Layout xla_layout = buffer.layout()->xla_layout();

      if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS ||
           (flags & PyBUF_STRIDES) == PyBUF_ND) &&
          !xla::LayoutUtil::IsMonotonicWithDim0Major(xla_layout)) {
        return xla::InvalidArgument("Buffer is not in C-contiguous layout.");
      } else if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
                 !xla::LayoutUtil::IsMonotonicWithDim0Minor(xla_layout)) {
        return xla::InvalidArgument("Buffer is not in F-contiguous layout.");
      } else if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
                 !xla::LayoutUtil::IsMonotonicWithDim0Major(xla_layout) &&
                 !xla::LayoutUtil::IsMonotonicWithDim0Minor(xla_layout)) {
        return xla::InvalidArgument("Buffer is not in contiguous layout.");
      } else if (!HasDefaultLayout(xla_layout)) {
        // Fail and fall back to using __array__ if the CPU buffer has a device
        // specific layout. For instance, this happens for host buffers in
        // pinned memories of the TPU device.
        return xla::InvalidArgument(
            "Buffer is potentially a device buffer with non default layout.");
      }
      ABSL_RETURN_IF_ERROR(buffer.GetReadyFuture().Await());
    }

    // We must hold the GIL (or at least prevent Python GC) while writing to the
    // view object, see https://github.com/python/cpython/issues/130409.
    std::memset(view, 0, sizeof(Py_buffer));
    const void* root_ptr =
        external_reference_hold->OpaqueDeviceMemoryDataPointer();
    view->buf = const_cast<void*>(root_ptr);
    auto extra = std::make_unique<ExtraBufferInfo>(
        buffers.front(), std::move(external_reference_hold));
    view->itemsize =
        xla::ShapeUtil::ByteSizeOfPrimitiveType(buffer.element_type());
    ABSL_ASSIGN_OR_RETURN(view->len, buffer.GetOnDeviceSizeInBytes());
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
      view->format = const_cast<char*>(format);
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
      view->ndim = buffer.dimensions().size();
      static_assert(sizeof(int64_t) == sizeof(Py_ssize_t),
                    "Py_ssize_t must be 64 bits");
      if (view->ndim != 0) {
        view->shape = reinterpret_cast<Py_ssize_t*>(
            const_cast<int64_t*>(buffer.dimensions().data()));
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          extra->strides =
              ByteStridesForShape(buffer.element_type(), buffer.dimensions(),
                                  buffer.layout()->xla_layout());
          view->strides = reinterpret_cast<Py_ssize_t*>(
              const_cast<int64_t*>(extra->strides.data()));
        }
      }
    }
    view->internal = extra.release();
    return absl::OkStatus();
  }();
  if (!status.ok()) {
    // numpy.asarray(...) eats the PyExc_BufferError. Adding a log here helps
    // debugging when the error really occurs.
    VLOG(1) << "Buffer Protocol Error: " << status;
    PyErr_SetString(PyExc_BufferError, status.ToString().c_str());
    return -1;
  }
  view->obj = exporter;
  Py_INCREF(view->obj);
  return 0;
}

void PyArray_bf_releasebuffer(PyObject*, Py_buffer* buffer) {
  auto extra = static_cast<ExtraBufferInfo*>(buffer->internal);
  delete extra;
}

// Returns if shape has a major-to-minor layout.
bool HasMajorToMinorLayout(const xla::Shape& shape) {
  if (shape.has_layout()) {
    for (int i = 0; i < shape.layout().minor_to_major().size(); ++i) {
      if (shape.layout().minor_to_major(i) !=
          shape.layout().minor_to_major().size() - 1 - i) {
        return false;
      }
    }
  }
  return true;
}

// Returns byte_strides if shape has a non-major-to-minor layout.
std::optional<std::vector<int64_t>> ByteStridesOrDefaultForShapeInt64(
    const xla::Shape& shape) {
  if (!shape.has_layout() || HasMajorToMinorLayout(shape)) {
    return std::nullopt;
  }
  return ByteStridesForShape(shape);
}

bool IsZeroCopyableCpuBuffer(const xla::PjRtBuffer* buf) {
  // For CPU buffers with device-specific layouts, we must delinearize
  // to unpack the array. This could happen for the host buffer
  // pre-mapped to the TPU device, a.k.a., pinned host buffers for the
  // device.
  bool has_default_layout =
      buf->layout() == nullptr || HasDefaultLayout(buf->layout()->xla_layout());
  // On CPU for values >= 8 bits, we can return the value in a zero-copy way.
  // For sub-byte values, we must copy in order to unpack the array.
  return buf->IsOnCpu() &&
         !xla::primitive_util::IsSubByteNonPredType(buf->element_type()) &&
         has_default_layout;
}

// Copies a single dense slice into a sub-region of a destination NumPy array
// using NumPy multidimensional slice assignment, which handles arbitrary
// multi-dimensional strided copies.
//
// REQUIRES: Python GIL is held.
absl::Status SetSlice(nanobind::handle dst_array, xla::nb_dtype dtype,
                      const xla::ifrt::IndexDomain& index_domain,
                      std::optional<absl::Span<const int64_t>> byte_strides,
                      const void* src_data) {
  const absl::Span<const int64_t>& dims = index_domain.shape().dims();
  const int ndim = dims.size();
  nanobind::tuple index_tuple =
      nanobind::steal<nanobind::tuple>(PyTuple_New(ndim));
  for (int d = 0; d < ndim; ++d) {
    const int64_t start = index_domain.origin().elements()[d];
    const int64_t stop = start + dims[d];
    PyTuple_SET_ITEM(index_tuple.ptr(), d,
                     nanobind::slice(start, stop).release().ptr());
  }
  xla::nb_numpy_ndarray shard_array(dtype, dims, byte_strides, src_data);
  if (PyObject_SetItem(dst_array.ptr(), index_tuple.ptr(), shard_array.ptr()) <
      0) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    std::string err_msg = "Failed to copy shard slice to host array";
    if (pvalue != nullptr) {
      nanobind::str err_str =
          nanobind::steal<nanobind::str>(PyObject_Str(pvalue));
      absl::StrAppend(&err_msg, ": ", err_str.c_str());
    }
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    return absl::InternalError(err_msg);
  }
  return absl::OkStatus();
}

// Copies a single dense slice of string cords into a sub-region of the
// destination string array contents buffer.
void SetStringSlice(const xla::ifrt::IndexDomain& index_domain,
                    absl::Span<const int64_t> global_byte_strides,
                    std::vector<absl::Cord>& src_shard_cords,
                    std::vector<absl::Cord>& dst_array_contents) {
  const auto& shard_dims = index_domain.shape().dims();
  const auto& shard_origin = index_domain.origin().elements();
  const int ndim = shard_dims.size();
  std::vector<int64_t> coord(ndim, 0);
  for (size_t i = 0; i < src_shard_cords.size(); ++i) {
    int64_t global_idx = 0;
    for (int d = 0; d < ndim; ++d) {
      global_idx += (shard_origin[d] + coord[d]) *
                    (global_byte_strides[d] / sizeof(absl::Cord));
    }
    DCHECK_LT(global_idx, dst_array_contents.size());
    dst_array_contents[global_idx] = std::move(src_shard_cords[i]);
    for (int d = ndim - 1; d >= 0; --d) {
      ++coord[d];
      if (coord[d] < shard_dims[d] || d == 0) {
        break;
      }
      coord[d] = 0;
    }
  }
}

absl::StatusOr<xla::Shape> HostShapeForArray(
    ifrt::Array* ifrt_array, absl::Span<const int64_t> dynamic_shape) {
  auto* arr_pjrt =
      xla::ifrt::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr_pjrt != nullptr) {
    auto* pjrt_buffer = arr_pjrt->pjrt_buffers().front().get();
    xla::Shape shape =
        xla::ShapeUtil::MakeShape(pjrt_buffer->element_type(), dynamic_shape);
    *shape.mutable_layout() = pjrt_buffer->layout()->xla_layout();
    return shape;
  }
  ABSL_ASSIGN_OR_RETURN(xla::PrimitiveType type,
                        ifrt::ToPrimitiveType(ifrt_array->dtype()));
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dynamic_shape);
}

// Determines the host shape and layout of the array.
absl::StatusOr<xla::Shape> GetHostShape(
    ifrt::Array* ifrt_array, absl::Span<const int64_t> dynamic_shape) {
  if (ifrt_array->dtype().kind() == ifrt::DType::kString) {
    return xla::ShapeUtil::MakeShape(xla::U8, dynamic_shape);
  }
  ABSL_ASSIGN_OR_RETURN(xla::Shape shape,
                        HostShapeForArray(ifrt_array, dynamic_shape));
  return xla::ShapeUtil::DeviceShapeToHostShape(std::move(shape));
}

// Returns the element byte size for the given dtype.
absl::StatusOr<int64_t> GetElementSize(ifrt::DType dtype) {
  if (dtype.kind() == ifrt::DType::kString) {
    return sizeof(absl::Cord);
  }
  ABSL_ASSIGN_OR_RETURN(xla::PrimitiveType type,
                        xla::ifrt::ToPrimitiveType(dtype));
  return xla::ShapeUtil::ByteSizeOfPrimitiveType(type);
}

// Computes global byte strides for string arrays.
std::vector<int64_t> GetGlobalByteStridesForStringType(
    const ifrt::Shape& shape) {
  const auto& dims = shape.dims();
  std::vector<int64_t> global_byte_strides(dims.size());
  int64_t stride = sizeof(absl::Cord);
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    global_byte_strides[d] = stride;
    stride *= dims[d];
  }
  return global_byte_strides;
}

// Computes the per-shard XLA shape matching the host layout.
absl::StatusOr<xla::Shape> GetXlaShardShape(const xla::ifrt::Sharding& sharding,
                                            const xla::ifrt::Shape& shape,
                                            const xla::Shape& host_shape) {
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::Shape ifrt_shard_shape,
                        sharding.GetShardShape(shape));
  const auto& shard_dims = ifrt_shard_shape.dims();
  if (host_shape.has_layout()) {
    return xla::ShapeUtil::MakeShapeWithDenseLayout(
        host_shape.element_type(), shard_dims,
        host_shape.layout().minor_to_major());
  }
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(
      host_shape.element_type(), shard_dims);
}

// Computes the per-shard byte strides for the given sharding, shape, and host
// shape.
absl::StatusOr<std::optional<std::vector<int64_t>>> GetShardByteStrides(
    ifrt::DType dtype, const xla::ifrt::Sharding& sharding,
    const xla::ifrt::Shape& shape, const xla::Shape& host_shape) {
  if (dtype.kind() == ifrt::DType::kString) {
    return std::nullopt;
  }
  ABSL_ASSIGN_OR_RETURN(xla::Shape xla_shard_shape,
                        GetXlaShardShape(sharding, shape, host_shape));
  return ByteStridesOrDefaultForShapeInt64(xla_shard_shape);
}

// Returns whether shard dimensions and strides form contiguous regions given
// the global byte strides.
// Returns whether shard dimensions and strides form contiguous regions given
// the global byte strides.
absl::StatusOr<bool> IsContiguous(
    ifrt::DType dtype, const xla::ifrt::Sharding& sharding,
    const xla::ifrt::Shape& shape, const xla::Shape& host_shape,
    absl::Span<const int64_t> global_byte_strides) {
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::Shape ifrt_shard_shape,
                        sharding.GetShardShape(shape));
  const auto& shard_dims = ifrt_shard_shape.dims();
  std::vector<int64_t> compact_strides;
  if (dtype.kind() == ifrt::DType::kString) {
    compact_strides = GetGlobalByteStridesForStringType(ifrt_shard_shape);
  } else {
    ABSL_ASSIGN_OR_RETURN(xla::Shape xla_shard_shape,
                          GetXlaShardShape(sharding, shape, host_shape));
    compact_strides = ByteStridesForShape(xla_shard_shape);
  }
  CHECK_EQ(shard_dims.size(), compact_strides.size());
  CHECK_EQ(shard_dims.size(), global_byte_strides.size());
  for (size_t d = 0; d < shard_dims.size(); ++d) {
    if (shard_dims[d] > 1 && global_byte_strides[d] != compact_strides[d]) {
      return false;
    }
  }
  return true;
}

}  // namespace

namespace {

absl::Status FillStringNumpyArray(std::vector<absl::Cord>& contents,
                                  xla::nb_numpy_ndarray& value) {
  auto* dst_py_array_obj = reinterpret_cast<::PyArrayObject*>(value.ptr());
  auto* descr = reinterpret_cast<PyArray_StringDTypeObject*>(
      PyArray_DESCR(dst_py_array_obj));

  npy_string_allocator* allocator = NpyString_acquire_allocator(descr);
  if (allocator == nullptr) {
    return absl::InternalError("NpyString_acquire_allocator returned null");
  }
  absl::Cleanup release_allocator = [allocator] {
    NpyString_release_allocator(allocator);
  };

  char* dst = PyArray_BYTES(dst_py_array_obj);
  const npy_intp itemsize = PyArray_ITEMSIZE(dst_py_array_obj);

  for (auto& cord : contents) {
    std::string_view input_str_view = cord.Flatten();
    auto* packed_entry = reinterpret_cast<npy_packed_static_string*>(dst);
    if (NpyString_pack(allocator, packed_entry, input_str_view.data(),
                       input_str_view.size()) < 0) {
      return absl::InternalError("NpyString_pack failed");
    }
    dst += itemsize;
  }
  return absl::OkStatus();
}

}  // namespace

// Information needed to copy an array to the host.
struct PyArray::CopyToHostState {
  struct DataSlice {
    // The logical sub-region (shape and offset) in the global array.
    xla::ifrt::IndexDomain index_domain;
    // In case the slice is non-contiguous in the destination buffer, temporary
    // buffers to copy the shard data into; nullptr otherwise.
    std::unique_ptr<std::vector<char>> temp_buffer;
    std::unique_ptr<std::vector<absl::Cord>> temp_string_buffer;
  };

  struct CopyData {
    // The IFRT array to copy.
    xla::ifrt::ArrayRef ifrt_array;
    // Host shape of the array.
    xla::Shape host_shape;
    // Size of an element in bytes.
    int64_t elem_size;
    // Byte strides of the destination NumPy array.
    std::vector<int64_t> byte_strides;
    // Byte strides of an individual shard, if strided.
    std::optional<std::vector<int64_t>> shard_byte_strides;
    // Whether shards occupy contiguous regions in destination NumPy array.
    bool is_contiguous;
    // Pointer to the start of the destination NumPy array.
    char* dst_ptr;
    // Data slices belonging to this array.
    std::vector<DataSlice> data_slices;
    // Promise to be set when the copy is complete.
    tsl::Promise<> result_promise;
    // Optional field, only used for arrays of type kString.
    std::shared_ptr<std::vector<absl::Cord>> string_array_contents;
  };

  CopyToHostState(xla::nb_numpy_ndarray result, tsl::Future<> result_future,
                  std::optional<CopyData> copy_data)
      : result(std::move(result)),
        result_future(std::move(result_future)),
        copy_data(std::move(copy_data)) {}
  ~CopyToHostState() {
    if (result.ptr() == nullptr) {  // std::moved
      return;
    }
    GlobalPyRefManager()->AddGarbage(std::move(result));
  }

  // Allow moves.
  CopyToHostState(CopyToHostState&&) = default;
  CopyToHostState& operator=(CopyToHostState&&) = default;
  // Disallow copies.
  CopyToHostState(const CopyToHostState&) = delete;
  CopyToHostState& operator=(const CopyToHostState&) = delete;

  // The numpy array on the host that will be populated with the result.
  xla::nb_numpy_ndarray result;
  // The future that will be set when `result` is ready.
  tsl::Future<> result_future;
  // If array should be copied to the host: stores all the information required
  // to perform the copy. Otherwise, std::nullopt.
  std::optional<CopyData> copy_data;
};

absl::StatusOr<tsl::Future<>> PyArray::CopyToHostAsync() {
  PyUserContextScope user_context_scope;
  xla::ifrt::Client* client = py_client()->ifrt_client();
  ABSL_ASSIGN_OR_RETURN(std::optional<CopyToHostState> state,
                        GetCopyToHostState(client));
  if (!state.has_value()) {  // No addressable devices.
    return tsl::Future<>();
  }
  tsl::Future<> result_future = state->result_future;
  if (state->copy_data.has_value()) {
    std::vector<CopyToHostState> states;
    states.push_back(*std::move(state));
    ABSL_RETURN_IF_ERROR(
        PyArray::BatchedCopyToHostAsyncHelper(client, std::move(states)));
  }
  return result_future;
}

absl::Status PyArray::CopySingleDeviceArrayToHostAsync() {
  ABSL_ASSIGN_OR_RETURN(auto arr, FullyReplicatedShard());
  return arr.CopyToHostAsync().status();
}

absl::StatusOr<std::pair<nb::object, bool>>
PyArray::SingleDeviceArrayToNumpyArrayDidCopy() {
  ABSL_ASSIGN_OR_RETURN(auto arr, FullyReplicatedShard());
  return arr.ToNumpyArrayDidCopy();
}

absl::StatusOr<std::pair<nb::object, bool>> PyArray::ToNumpyArrayDidCopy() {
  PyUserContextScope user_context_scope;
  xla::ifrt::Client* client = py_client()->ifrt_client();
  ABSL_ASSIGN_OR_RETURN(std::optional<CopyToHostState> state,
                        GetCopyToHostState(client));
  if (!state.has_value()) {
    return xla::InvalidArgument("Array has no addressable devices.");
  }
  xla::nb_numpy_ndarray result = state->result;
  tsl::Future<> result_future = state->result_future;
  bool did_copy = false;
  if (state->copy_data.has_value()) {
    // No previous copy issued: issue one now.
    did_copy = true;
    std::vector<CopyToHostState> states;
    states.push_back(*std::move(state));
    ABSL_RETURN_IF_ERROR(
        PyArray::BatchedCopyToHostAsyncHelper(client, std::move(states)));
  }

  // Wait for the copy to complete.
  if (!result_future.IsReady()) {
    nb::gil_scoped_release gil;
    BlockUntilReadyWithCancel(result_future);
  }
  absl::Status status = result_future.Await();
  if (!status.ok()) {
    return xla::ifrt::ExpandUserContexts(std::move(status));
  }
  ABSL_RETURN_IF_ERROR(BlockUntilResultStatusIsReady());
  return std::make_pair(std::move(result), did_copy);
}

absl::StatusOr<std::optional<PyArray::CopyToHostState>>
PyArray::GetCopyToHostState(xla::ifrt::Client* client) {
  xla::ifrt::ArrayRef ifrt_array = ifrt_array_ref();
  if (ifrt_array == nullptr || ifrt_array->IsDeleted()) {
    return xla::InvalidArgument("Array has been deleted.");
  }
  if (ifrt_array->dtype().kind() == ifrt::DType::kToken) {
    return xla::InvalidArgument(
        "Cannot convert a token-shape buffer to a numpy array.");
  }
  auto* addressable_devices =
      ifrt_array->sharding().devices()->AddressableDeviceList();
  if (addressable_devices->empty()) {  // No addressable devices.
    return std::nullopt;
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<int64_t> dynamic_shape, dynamic_shape());
  ABSL_ASSIGN_OR_RETURN(xla::Shape host_shape,
                        GetHostShape(ifrt_array.get(), dynamic_shape));

  Storage& storage = GetStorage();
  nanobind::object py_sharding;
  xla::nb_numpy_ndarray result;
  tsl::Future<> result_future;
  tsl::Promise<> result_promise;
  std::shared_ptr<std::vector<absl::Cord>> string_array_contents;
  {
    ft_lock_guard l(storage.mu);
    ifrt_array = storage.ifrt_array;
    py_sharding = storage.sharding;
    if (ifrt_array == nullptr || ifrt_array->IsDeleted()) {
      return xla::InvalidArgument("Array has been deleted.");
    }
    if (storage.host_value.ready.IsValid()) {  // Copy has already been issued.
      return std::make_optional<CopyToHostState>(storage.host_value.value,
                                                 storage.host_value.ready,
                                                 /*copy_data=*/std::nullopt);
    }

    // See if the array is zero-copyable to the host. If so perform the
    // zero-copy synchronously and return.
    auto* arr_pjrt =
        xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleArray>(
            ifrt_array.get());
    if (arr_pjrt != nullptr && ifrt_array->sharding().IsFullyReplicated()) {
      auto* pjrt_buffer = arr_pjrt->pjrt_buffers().front().get();
      TF_RET_CHECK(!pjrt_buffer->IsTuple());
      if (IsZeroCopyableCpuBuffer(pjrt_buffer)) {
        ABSL_ASSIGN_OR_RETURN(
            xla::nb_dtype dtype,
            PrimitiveTypeToNbDtype(host_shape.element_type()));
        ABSL_ASSIGN_OR_RETURN(auto external_ref,
                              pjrt_buffer->AcquireExternalReference());
        void* data = external_ref->OpaqueDeviceMemoryDataPointer();
        struct Hold {
          xla::ifrt::ArrayRef buffer;
          std::unique_ptr<xla::PjRtBuffer::ExternalReference>
              external_reference_hold;
        };
        auto hold = std::make_unique<Hold>();
        hold->buffer = ifrt_array;
        hold->external_reference_hold = std::move(external_ref);
        nb::capsule hold_capsule(hold.release(), [](void* h) noexcept {
          delete static_cast<Hold*>(h);
        });
        storage.host_value.value = xla::nb_numpy_ndarray(
            dtype, host_shape.dimensions(), ByteStridesForShape(host_shape),
            data, hold_capsule);
        storage.host_value.value.attr("flags").attr("writeable") =
            nb::bool_(false);
        storage.host_value.ready = ifrt_array->GetReadyFuture();
        return std::make_optional<CopyToHostState>(storage.host_value.value,
                                                   storage.host_value.ready,
                                                   /*copy_data=*/std::nullopt);
      }
    }
    auto transfer_guard_formatter = [ifrt_array] {
      return absl::StrCat(
          "shape=(", absl::StrJoin(ifrt_array->shape().dims(), ","),
          "), dtype=", ifrt_array->dtype(), ", device=",
          ifrt_array->sharding().devices()->devices().front()->DebugString());
    };
    ABSL_RETURN_IF_ERROR(
        ApplyTransferGuardToDeviceToHost(transfer_guard_formatter));

    // Prepare the host value for the copy.
    auto [promise, future] = tsl::MakePromise<>();
    storage.host_value.ready = std::move(future);
    if (ifrt_array->dtype().kind() == ifrt::DType::kString) {
      storage.host_value.string_array_contents =
          std::make_unique<std::vector<absl::Cord>>(
              ifrt_array->shape().num_elements());
      storage.host_value.value = xla::nb_numpy_ndarray(
          NumpyTypes::Get().string_dtype, ifrt_array->shape().dims(),
          /*strides=*/std::nullopt);
    } else {
      std::optional<std::vector<int64_t>> strides =
          ByteStridesOrDefaultForShapeInt64(host_shape);
      ABSL_ASSIGN_OR_RETURN(xla::nb_dtype dtype,
                            PrimitiveTypeToNbDtype(host_shape.element_type()));
      // TODO(hyeontaek): Several PjRt runtimes assume that the host buffer uses
      // the same transposition as the device buffer. This is different from
      // xla::PjRtBuffer::ToLiteral()'s semantics that the runtime respects the
      // layout of the host buffer literal. On the other hand, the runtime often
      // knows better about an efficient layout for the host buffer. It will be
      // useful to revisit the semantics of xla::PjRtBuffer::ToLiteral() to see
      // if it is desirable for the runtime to choose the layout.
      storage.host_value.value =
          xla::nb_numpy_ndarray(dtype, host_shape.dimensions(), strides);
    }
    result = storage.host_value.value;
    result_future = storage.host_value.ready;
    result_promise = std::move(promise);
    string_array_contents = storage.host_value.string_array_contents;
  }

  auto state = [&]() -> absl::StatusOr<CopyToHostState> {
    // Retrieve the unique index domains for the array, using the PyArray's
    // sharding if necessary.
    using UniqueIndexDomains =
        absl::InlinedVector<xla::ifrt::Sharding::IndexDomainAndShardIndices, 1>;
    absl::StatusOr<UniqueIndexDomains> unique_index_domains =
        ifrt_array->sharding().UniqueIndexDomains(ifrt_array->shape());
    if (!unique_index_domains.ok()) {
      // Remap the array with HloSharding, which has index domains. We
      // expect this path to be taken extremely rarely.
      ABSL_ASSIGN_OR_RETURN(ifrt_array, RemapArrayWithHloSharding(
                                            client, py_sharding, ifrt_array));
      unique_index_domains =
          ifrt_array->sharding().UniqueIndexDomains(ifrt_array->shape());
      ABSL_RETURN_IF_ERROR(unique_index_domains.status());
    }

    // Compute the strides for the host shape.
    ABSL_ASSIGN_OR_RETURN(int64_t elem_size,
                          GetElementSize(ifrt_array->dtype()));
    std::vector<int64_t> global_byte_strides;
    if (ifrt_array->dtype().kind() == ifrt::DType::kString) {
      global_byte_strides =
          GetGlobalByteStridesForStringType(ifrt_array->shape());
    } else {
      global_byte_strides = ByteStridesForShape(host_shape);
    }
    ABSL_ASSIGN_OR_RETURN(
        std::optional<std::vector<int64_t>> shard_byte_strides,
        GetShardByteStrides(ifrt_array->dtype(), ifrt_array->sharding(),
                            ifrt_array->shape(), host_shape));

    // Determine if shards occupy contiguous regions in destination NumPy
    // buffer. A shard is non-contiguous if its rows/elements have gaps in
    // the global buffer (e.g. when partitioning along columns or multiple
    // mesh axes).
    //
    // Contiguous shards' data can be copied directly into the destination
    // NumPy buffer. Non-contiguous shards must be copied into a temporary
    // dense buffer first, and then re-mapped to the destination NumPy
    // buffer.
    ABSL_ASSIGN_OR_RETURN(
        bool is_contiguous,
        IsContiguous(ifrt_array->dtype(), ifrt_array->sharding(),
                     ifrt_array->shape(), host_shape, global_byte_strides));

    // Determine the pointer to the start of the destination NumPy array.
    char* dst_ptr = nullptr;
    if (ifrt_array->dtype().kind() == ifrt::DType::kString) {
      CHECK(string_array_contents != nullptr);
      dst_ptr = reinterpret_cast<char*>(string_array_contents->data());
    } else {
      dst_ptr = reinterpret_cast<char*>(result.mutable_data());
    }

    // Partition the array data into logical data slices.
    std::vector<CopyToHostState::DataSlice> data_slices;
    data_slices.reserve(unique_index_domains->size());
    for (xla::ifrt::Sharding::IndexDomainAndShardIndices& unique_index_domain :
         *unique_index_domains) {
      data_slices.push_back(CopyToHostState::DataSlice{
          .index_domain = std::move(unique_index_domain.index_domain),
      });
    }

    return CopyToHostState(
        std::move(result), std::move(result_future),
        CopyToHostState::CopyData{
            .ifrt_array = std::move(ifrt_array),
            .host_shape = std::move(host_shape),
            .elem_size = elem_size,
            .byte_strides = std::move(global_byte_strides),
            .shard_byte_strides = std::move(shard_byte_strides),
            .is_contiguous = is_contiguous,
            .dst_ptr = dst_ptr,
            .data_slices = std::move(data_slices),
            .result_promise = std::move(result_promise),
            .string_array_contents = std::move(string_array_contents),
        });
  }();
  if (!state.ok()) {
    result_promise.Set(state.status());
    return state.status();
  }
  return std::optional<CopyToHostState>(*std::move(state));
}

absl::Status PyArray::BatchedCopyToHostAsyncHelper(
    xla::ifrt::Client* client,
    std::vector<PyArray::CopyToHostState> array_states) {
  CHECK(client != nullptr);
  if (array_states.empty()) {
    return absl::OkStatus();
  }

  // Create copy specs, one per array.
  std::vector<xla::ifrt::Client::CopyArraysToHostBufferShardsSpec> specs;
  specs.reserve(array_states.size());
  for (PyArray::CopyToHostState& array_state : array_states) {
    CHECK(array_state.copy_data.has_value());
    CopyToHostState::CopyData& copy_data = *array_state.copy_data;
    xla::ifrt::ArrayRef& ifrt_array = copy_data.ifrt_array;
    xla::ifrt::Client::CopyArraysToHostBufferShardsSpec::Buffers buffers;
    buffers.reserve(copy_data.data_slices.size());
    for (PyArray::CopyToHostState::DataSlice& slice : copy_data.data_slices) {
      void* data_ptr = nullptr;
      if (copy_data.is_contiguous) {
        // Shard data is contiguous in destination NumPy buffer: copy directly
        // into the destination buffer.
        int64_t offset = 0;
        for (int d = 0; d < copy_data.host_shape.dimensions().size(); ++d) {
          offset += slice.index_domain.origin().elements()[d] *
                    copy_data.byte_strides[d];
        }
        data_ptr = copy_data.dst_ptr + offset;
      } else {
        // Shard data is non-contiguous in destination NumPy buffer: copy
        // into a temporary buffer.
        int64_t shard_elements = slice.index_domain.shape().num_elements();
        if (ifrt_array->dtype().kind() == ifrt::DType::kString) {
          slice.temp_string_buffer =
              std::make_unique<std::vector<absl::Cord>>(shard_elements);
          data_ptr = slice.temp_string_buffer->data();
        } else {
          slice.temp_buffer = std::make_unique<std::vector<char>>(
              shard_elements * copy_data.elem_size);
          data_ptr = slice.temp_buffer->data();
        }
      }
      buffers.push_back(xla::ifrt::Client::MutableHostBuffer{
          /*data=*/data_ptr,
          /*dtype=*/ifrt_array->dtype(),
          /*shape=*/slice.index_domain.shape(),
          /*byte_strides=*/copy_data.shard_byte_strides});
    }
    specs.push_back(xla::ifrt::Client::CopyArraysToHostBufferShardsSpec{
        .array = ifrt_array,
        .buffers = std::move(buffers),
    });
  }

  if (specs.empty()) {
    // Nothing to copy.
    return absl::OkStatus();
  }

  // Issue the copy.
  absl::StatusOr<std::vector<tsl::Future<>>> futures =
      client->CopyArraysToHostBufferShards(
          absl::MakeSpan(specs), xla::ifrt::ArrayCopySemantics::kAlwaysCopy);
  if (!futures.ok()) {
    for (CopyToHostState& array_state : array_states) {
      array_state.copy_data->result_promise.Set(futures.status());
    }
    return futures.status();
  }

  // Prepare the futures to run as each array copy completes.
  CHECK_EQ(futures->size(), array_states.size());
  for (int i = 0; i < array_states.size(); ++i) {
    tsl::Future<>& future = (*futures)[i];
    CopyToHostState& array_state = array_states[i];
    const xla::ifrt::ArrayRef& ifrt_array = array_state.copy_data->ifrt_array;
    if (ifrt_array->dtype().kind() == ifrt::DType::kString) {
      future.OnReady([array_state =
                          std::move(array_state)](absl::Status status) mutable {
        if (!status.ok()) {
          array_state.copy_data->result_promise.Set(std::move(status));
          return;
        }
        nanobind::gil_scoped_acquire gil;
        try {
          if (!array_state.copy_data->is_contiguous) {
            // Shard data is copied into temporary Cords. Unpack the temporary
            // buffers into the global row-major `string_array_contents`.
            for (auto& slice : array_state.copy_data->data_slices) {
              CHECK(slice.temp_string_buffer != nullptr);
              SetStringSlice(slice.index_domain,
                             array_state.copy_data->byte_strides,
                             *slice.temp_string_buffer,
                             *array_state.copy_data->string_array_contents);
            }
          }
          absl::Status s = FillStringNumpyArray(
              *array_state.copy_data->string_array_contents,
              array_state.result);
          if (!s.ok()) {
            array_state.copy_data->result_promise.Set(std::move(s));
            return;
          }
          array_state.result.attr("flags").attr("writeable") =
              nanobind::bool_(false);
          array_state.copy_data->result_promise.Set(absl::OkStatus());
        } catch (const nb::python_error& e) {
          array_state.copy_data->result_promise.Set(
              absl::InternalError(absl::StrCat(
                  "Unable to assign to a slice of NumPy Array: ", e.what())));
        } catch (...) {
          array_state.copy_data->result_promise.Set(
              absl::InternalError("Unknown exception while assigning to a "
                                  "slice of NumPy Array."));
        }
      });
    } else if (array_state.copy_data->is_contiguous) {
      // All shard data is copied directly into the destination NumPy buffer.
      array_state.result.attr("flags").attr("writeable") =
          nanobind::bool_(false);
      future.OnReady(
          [array_state = std::move(array_state)](absl::Status status) mutable {
            array_state.copy_data->result_promise.Set(std::move(status));
          });
    } else {
      // Shard data is copied into temporary dense buffers. Once the copy
      // completes, unpack the temporary buffers into the destination NumPy
      // array.
      future.OnReady([array_state =
                          std::move(array_state)](absl::Status status) mutable {
        if (!status.ok()) {
          array_state.copy_data->result_promise.Set(std::move(status));
          return;
        }
        nanobind::gil_scoped_acquire gil;
        try {
          xla::nb_dtype dtype =
              nanobind::cast<xla::nb_dtype>(array_state.result.attr("dtype"));
          for (const auto& slice : array_state.copy_data->data_slices) {
            CHECK(slice.temp_buffer != nullptr);
            std::optional<absl::Span<const int64_t>> shard_byte_strides;
            if (array_state.copy_data->shard_byte_strides.has_value()) {
              shard_byte_strides = absl::MakeConstSpan(
                  *array_state.copy_data->shard_byte_strides);
            }
            absl::Status copy_status =
                SetSlice(array_state.result, dtype, slice.index_domain,
                         shard_byte_strides, slice.temp_buffer->data());
            if (!copy_status.ok()) {
              array_state.copy_data->result_promise.Set(std::move(copy_status));
              return;
            }
          }
          array_state.result.attr("flags").attr("writeable") =
              nanobind::bool_(false);
          array_state.copy_data->result_promise.Set(absl::OkStatus());
        } catch (const nb::python_error& e) {
          array_state.copy_data->result_promise.Set(
              absl::InternalError(absl::StrCat(
                  "Unable to assign to a slice of NumPy Array: ", e.what())));
        } catch (...) {
          array_state.copy_data->result_promise.Set(
              absl::InternalError("Unknown exception while assigning to a "
                                  "slice of NumPy Array."));
        }
      });
    }
  }

  return absl::OkStatus();
}

absl::Status PyArray::BatchedCopyToHostAsync(nanobind::sequence py_arrays) {
  // Group the arrays by client and device list.
  struct GroupKey {
    xla::ifrt::Client* client;
    xla::ifrt::DeviceListRef devices;

    bool operator==(const GroupKey& other) const {
      return client == other.client && devices == other.devices;
    }

    struct Hash {
      size_t operator()(const GroupKey& key) const {
        return absl::HashOf(key.client, key.devices);
      }
    };
  };
  absl::flat_hash_map<GroupKey, std::vector<CopyToHostState>, GroupKey::Hash>
      groups;
  for (nanobind::handle item : py_arrays) {
    if (!PyArray::IsPyArray(item)) {
      return xla::InvalidArgument(
          "PyArray::BatchedCopyToHostAsync can take PyArray only, got %s",
          nb::cast<std::string_view>(nb::repr(item.type())));
    }
    PyArray py_array = nb::cast<PyArray>(item);
    xla::ifrt::Client* client = py_array.py_client()->ifrt_client();
    ABSL_ASSIGN_OR_RETURN(std::optional<CopyToHostState> state,
                          py_array.GetCopyToHostState(client));
    if (!state.has_value()) {  // Array has no addressable devices.
      continue;
    }
    if (!state->copy_data.has_value()) {  // Copy already issued.
      continue;
    }
    xla::ifrt::DeviceListRef devices =
        state->copy_data->ifrt_array->sharding().devices();
    groups[GroupKey{client, std::move(devices)}].push_back(*std::move(state));
  }

  // Copy each group of arrays separately.
  absl::Status status;
  for (auto& [key, grouped_arrays] : groups) {
    status.Update(
        BatchedCopyToHostAsyncHelper(key.client, std::move(grouped_arrays)));
  }
  return status;
}

namespace {

PyType_Slot array_meta_slots[] = {
    {Py_tp_base, &PyType_Type},
    {0, nullptr},
};

PyType_Slot array_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(PyBaseArray_tp_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void*>(PyBaseArray_tp_traverse)},
    {Py_tp_hash, reinterpret_cast<void*>(PyObject_HashNotImplemented)},
    {0, nullptr},
};

PyGetSetDef array_impl_tp_getset[] = {
    {"__dict__", PyObject_GenericGetDict, PyObject_GenericSetDict, nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyType_Slot array_impl_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyArray_tp_new)},
    {Py_tp_finalize, reinterpret_cast<void*>(PyArray_tp_finalize)},
    {Py_tp_dealloc, reinterpret_cast<void*>(PyArray_tp_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void*>(PyArray_tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(PyArray_tp_clear)},
    {Py_tp_getset, reinterpret_cast<void*>(array_impl_tp_getset)},
    {Py_bf_getbuffer, reinterpret_cast<void*>(PyArray_bf_getbuffer)},
    {Py_bf_releasebuffer, reinterpret_cast<void*>(PyArray_bf_releasebuffer)},
    {0, nullptr},
};

}  // namespace

absl::Status PyArray::Register(nb::module_& m) {
  std::string metaclass_name =
      absl::StrCat(nb::cast<std::string>(m.attr("__name__")), ".ArrayMeta");
  PyType_Spec array_meta_spec = {
      /*.name=*/metaclass_name.c_str(),
      /*.basicsize=*/0,
      /*.itemsize=*/0,
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
      /*.slots=*/array_meta_slots};
  nb::object array_meta_type =
      nb::steal<nb::object>(PyType_FromSpec(&array_meta_spec));
  if (!array_meta_type) {
    throw nb::python_error();
  }
  m.attr("ArrayMeta") = array_meta_type;

  // We are not using nanobind to avoid having a non-standard metaclass, which
  // would make Array incompatible with abc.ABCMeta.
  std::string base_name =
      absl::StrCat(nb::cast<std::string>(m.attr("__name__")), ".Array");
  PyType_Spec array_spec = {
      /*.name=*/base_name.c_str(),
      /*.basicsize=*/static_cast<int>(sizeof(PyBaseArrayObject)),
      /*.itemsize=*/0,
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
      /*.slots=*/array_slots};
  nb::object base_type = nb::steal<nb::object>(PyType_FromMetaclass(
      reinterpret_cast<PyTypeObject*>(array_meta_type.ptr()), m.ptr(),
      &array_spec, nullptr));
  if (!base_type) {
    throw nb::python_error();
  }
  m.attr("Array") = base_type;

  m.def("set_tracer_class", [](nb::object f) { tracer_class = f; });

  nb::object type_instancecheck =
      nb::borrow<nb::object>(reinterpret_cast<PyObject*>(&PyType_Type))
          .attr("__instancecheck__");
  array_meta_type.attr("__instancecheck__") = nb::cpp_function(
      [base_type, type_instancecheck](nb::object self, nb::object x) {
        // We are calling type's instancecheck method rather than
        // PyObject_TypeCheck to avoid breaking users who use wrapt.ObjectProxy,
        // such as TFP's NumpyVariable.
        if (nb::cast<bool>(type_instancecheck(self, x))) {
          return true;
        }
        // Instances of Tracer that have array avals are considered instances of
        // Array.
        if (tracer_class.ptr() && self.ptr() == base_type.ptr() &&
            PyObject_TypeCheck(x.ptr(), reinterpret_cast<PyTypeObject*>(
                                            tracer_class.ptr())) != 0) {
          auto is_traced_array_fn =
              nb::getattr(x, "_is_traced_array", nb::none());
          if (!is_traced_array_fn.is_none()) {
            return nb::cast<bool>(is_traced_array_fn());
          }
        }
        return false;
      },
      nb::is_method(), nb::arg("x").none());

  std::string name =
      absl::StrCat(nb::cast<std::string>(m.attr("__name__")), ".ArrayImpl");

  PyType_Spec array_impl_spec = {
      /*.name=*/name.c_str(),
      /*.basicsize=*/static_cast<int>(sizeof(PyArrayObject)),
      /*.itemsize=*/0,
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
          Py_TPFLAGS_MANAGED_DICT | Py_TPFLAGS_MANAGED_WEAKREF,
      /*.slots=*/array_impl_slots,
  };

  type_ = PyType_FromSpecWithBases(&array_impl_spec, base_type.ptr());
  if (!type_) {
    throw nb::python_error();
  }
  auto type = nb::borrow<nb::object>(type_);
  m.attr("ArrayImpl") = type;

  type.attr("__init__") = nb::cpp_function(
      [](PyArray self, nb::object aval, nb::object sharding, nb::list arrays,
         bool committed, bool skip_checks) {
        if (!(arrays.size() == 0 || arrays[0].type().is(PyArray::type()))) {
          throw nb::type_error(
              absl::StrCat(
                  "Unsupported type for elements in `arrays`: ",
                  nb::cast<std::string_view>(nb::str(arrays[0].type())))
                  .c_str());
        }
        auto py_arrays = nb::cast<std::vector<PyArray>>(arrays);
        PyArray::PyInit(self, std::move(aval), std::move(sharding), py_arrays,
                        committed, skip_checks);
      },
      nb::is_method(), nb::arg("aval"), nb::arg("sharding"), nb::arg("arrays"),
      nb::arg("committed"), nb::arg("_skip_checks") = false);
  type.attr("delete") = nb::cpp_function(
      [](PyArray& self) { xla::ThrowIfError(self.Delete()); }, nb::is_method());
  type.attr("_rewrap_with_aval_and_sharding") = nb::cpp_function(
      // NOTE(dsuo): Zero-copy metadata rewrapping. Returns a new PyArray with
      // new aval and sharding metadata with a new underlying ifrt
      // array that reuses the same underlying buffers, avoiding memory copies
      // when only the logical view changes.
      [](PyArray self, nb::object aval, nb::object sharding) -> PyArray {
        xla::ifrt::ArrayRef ifrt_array_ptr = xla::ValueOrThrow(
            self.GetIfrtArray("_rewrap_with_aval_and_sharding()"));
        // Create a new PyArray that shares the same ifrt array.
        bool weak_type = nb::cast<bool>(aval.attr("weak_type"));
        xla::nb_dtype dtype = nb::cast<xla::nb_dtype>(aval.attr("dtype"));
        std::vector<int64_t> shape =
            nb::cast<std::vector<int64_t>>(aval.attr("shape"));
        auto ifrt_sharding = xla::ValueOrThrow(
            jax::GetIfrtHloSharding(sharding, xla::ifrt::Shape(shape)));
        PyUserContextScope user_context;

        auto single_device_arrays =
            xla::ValueOrThrow(ifrt_array_ptr->DisassembleIntoSingleDeviceArrays(
                xla::ifrt::ArrayCopySemantics::kReuseInput,
                xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

        auto new_ifrt_array = xla::ValueOrThrow(
            self.py_client()
                ->ifrt_client()
                ->AssembleArrayFromSingleDeviceArrays(
                    ifrt_array_ptr->dtype(), xla::ifrt::Shape(shape),
                    std::move(ifrt_sharding),
                    absl::MakeSpan(single_device_arrays),
                    xla::ifrt::ArrayCopySemantics::kReuseInput,
                    xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

        return PyArray(std::move(aval), weak_type, dtype, std::move(shape),
                       std::move(sharding), self.py_client(),
                       std::move(new_ifrt_array), /*committed=*/true);
      },
      nb::is_method(), nb::arg("aval"), nb::arg("sharding"));
  type.attr("_sharding") = xla::nb_property_readonly(&PyArray::sharding);
  type.attr("aval") = xla::nb_property_readonly(&PyArray::aval);
  type.attr("_arrays") = xla::nb_property_readonly(&PyArray::arrays);
  type.attr("_fully_replicated_shard") = nb::cpp_function(
      [](PyArray self) {
        return xla::ValueOrThrow(self.FullyReplicatedShard());
      },
      nb::is_method());
  type.attr("_npy_value") =
      xla::nb_property(&PyArray::npy_value, &PyArray::set_npy_value);
  type.attr("_committed") = xla::nb_property_readonly(&PyArray::committed);
  nb::class_<xla::PjRtRawBufferRef>(m, "RawBuffer")
      .def_prop_ro("ptr",
                   [](const xla::PjRtRawBufferRef& self) {
                     return reinterpret_cast<std::uintptr_t>(self.get());
                   })
      .def("__repr__", [](const xla::PjRtRawBufferRef& self) {
        return absl::StrFormat("<RawBuffer 0x%x>",
                               reinterpret_cast<std::uintptr_t>(self.get()));
      });
  // TODO(parkers): consider replacing with unsafe_raw_buffer.
  type.attr("unsafe_buffer_pointer") = nb::cpp_function(
      [](PyArray self) {
        return xla::ValueOrThrow(self.UnsafeBufferPointer());
      },
      nb::is_method());
  type.attr("unsafe_raw_buffer") = nb::cpp_function(
      [](PyArray self) {
        auto arr = xla::ValueOrThrow(self.AssertUnsharded("unsafe_raw_buffer"));
        auto ifrt_array =
            xla::ValueOrThrow(arr.GetIfrtArray("unsafe_raw_buffer"));
        return xla::ValueOrThrow(xla::PjRtRawBuffer::CreateRawAliasOfBuffer(
            GetPjrtBuffer(ifrt_array.get())));
      },
      nb::is_method());
  type.attr("__cuda_array_interface__") = xla::nb_property_readonly(
      [](PyArray self) { return self.CudaArrayInterface(); });
  type.attr("_pjrt_layout") =
      xla::nb_property_readonly(xla::ValueOrThrowWrapper(&PyArray::layout));
  type.attr("on_device_size_in_bytes") = nb::cpp_function(
      xla::ValueOrThrowWrapper(&PyArray::GetOnDeviceSizeInBytes),
      nb::is_method());
  type.attr("_single_device_array_to_np_array_did_copy") = nb::cpp_function(
      xla::ValueOrThrowWrapper(&PyArray::SingleDeviceArrayToNumpyArrayDidCopy),
      nb::is_method());
  type.attr("_to_np_array_did_copy") = nb::cpp_function(
      xla::ValueOrThrowWrapper(&PyArray::ToNumpyArrayDidCopy), nb::is_method());
  type.attr("_copy_single_device_array_to_host_async") = nb::cpp_function(
      [](PyArray& self) {
        xla::ThrowIfError(self.CopySingleDeviceArrayToHostAsync());
      },
      nb::is_method());
  type.attr("_replace_with") = nb::cpp_function(
      [](PyArray& self, PyArray& o) {
        xla::ThrowIfError(self.ReplaceWithAlias(o));
      },
      nb::is_method());
  type.attr("block_until_ready") = nb::cpp_function(
      [](PyArray self) -> nb::object {
        xla::ThrowIfError(self.BlockUntilReady());
        return self;
      },
      nb::is_method());
  type.attr("platform") = nb::cpp_function(
      [](PyArray self) {
        xla::ifrt::ArrayRef ifrt_array =
            xla::ValueOrThrow(self.GetIfrtArray("platform()"));
        const xla::ifrt::DeviceListRef& devices =
            ifrt_array->sharding().devices();
        absl::string_view platform_name =
            devices->devices().front()->PlatformName();
        if (platform_name == "cuda" || platform_name == "rocm" ||
            platform_name == "oneapi") {
          return std::string_view("gpu");
        } else {
          return platform_name;
        }
      },
      nb::is_method());
  type.attr("is_ready") = nb::cpp_function(
      [](PyArray self) { return xla::ValueOrThrow(self.IsReady()); },
      nb::is_method());
  type.attr("is_deleted") =
      nb::cpp_function(&PyArray::IsDeleted, nb::is_method());
  type.attr("traceback") = xla::nb_property_readonly(&PyArray::traceback);
  type.attr("clone") = nb::cpp_function(&PyArray::Clone, nb::is_method());
  type.attr("__module__") = m.attr("__name__");

  m.attr("batched_copy_array_to_devices_with_sharding") = nb::cpp_function(
      [](absl::Span<const PyArray> arrays,
         absl::Span<const nb_class_ptr<PyDeviceList>> dst_device_lists,
         absl::Span<const nb::object> shardings,
         absl::Span<const ifrt::ArrayCopySemantics> array_copy_semantics) {
        if (arrays.empty()) {
          return std::vector<PyArray>();
        }
        tsl::profiler::TraceMe traceme(
            "batched_copy_array_to_devices_with_sharding");
        std::vector<ifrt::DeviceListRef> device_lists;
        {
          tsl::profiler::TraceMe device_list_traceme(
              "batched_copy_array_to_devices_with_sharding: assemble device "
              "lists");
          device_lists.reserve(dst_device_lists.size());
          for (const auto& dst_devices : dst_device_lists) {
            device_lists.push_back(
                xla::ValueOrThrow(dst_devices->ifrt_device_list()));
          }
        }
        return xla::ValueOrThrow(PyArray::BatchedCopyToDeviceWithSharding(
            arrays, device_lists, shardings, array_copy_semantics));
      });
  m.attr("array_result_handler") = nb::cpp_function(
      [](nb::object aval, nb::object sharding,
         bool committed) -> nb_class_ptr<PyArrayResultHandler> {
        return make_nb_class<PyArrayResultHandler>(
            std::move(aval), std::move(sharding), committed);
      },
      nb::arg("aval"), nb::arg("sharding"), nb::arg("committed"));

  nb::class_<PyArrayResultHandler>(m, "ResultHandler")
      .def(
          "__call__",
          [](const PyArrayResultHandler& self, nb::object arg) {
            if (PyArray py_array; nb::try_cast<PyArray>(arg, py_array)) {
              return self.Call(py_array);
            }
            if (std::vector<PyArray> py_arrays;
                nb::try_cast<std::vector<PyArray>>(arg, py_arrays)) {
              return self.Call(py_arrays);
            }
            throw nb::type_error(
                absl::StrCat(
                    "Expected a single PyArray or a sequence of PyArrays, got ",
                    nb::cast<std::string_view>(nb::str(arg.type())))
                    .c_str());
          },
          nb::sig(
              "def __call__(self, arg: Array | Sequence[Array], /) -> Array"))
      .def("wrap",
           [](const PyArrayResultHandler& self, nb::callable wrapper) {
             auto wrappers = self.wrappers();
             wrappers.push_back(std::move(wrapper));
             return make_nb_class<PyArrayResultHandler>(
                 self.aval(), self.sharding(), self.committed(),
                 std::move(wrappers));
           })
      .def("pre_wrap",
           [](const PyArrayResultHandler& self, nb::callable wrapper) {
             auto wrappers = self.wrappers();
             wrappers.insert(wrappers.begin(), std::move(wrapper));
             return make_nb_class<PyArrayResultHandler>(
                 self.aval(), self.sharding(), self.committed(),
                 std::move(wrappers));
           });

  return absl::OkStatus();
}

}  // namespace jax
