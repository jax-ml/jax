/* Copyright 2025 The JAX Authors

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
#include "jaxlib/py_socket_transfer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/array.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/to_ifrt_sharding.h"
#include "jaxlib/traceback.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/transfer_server_interface.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/pjrt_transfer_server.h"
#include "xla/python/transfer/socket-server.h"
#include "xla/python/transfer/socket_bulk_transport.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/streaming_ifrt.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/python/types.h"
#include "xla/python/version.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace aux {

namespace nb = nanobind;

absl::StatusOr<xla::PjRtMemorySpace*> MemorySpaceFromSharding(
    const xla::ifrt::Sharding& sharding) {
  if (sharding.devices()->devices().size() != 1) {
    return xla::InvalidArgument(
        "Can only convert SingleDeviceSharding to MemorySpace not %s",
        sharding.DebugString());
  }
  auto* device = sharding.devices()->devices()[0];
  if (sharding.memory_kind().memory_kind().has_value()) {
    // Find `PjRtMemorySpace` that is associated with the sharding's device
    // and matches the sharding's memory_kind.
    xla::ifrt::Memory* memory = nullptr;
    for (xla::ifrt::Memory* ms : device->Memories()) {
      if (ms->Kind() == sharding.memory_kind()) {
        memory = ms;
        break;
      }
    }
    if (memory == nullptr) {
      return xla::InvalidArgument(
          "Invalid memory kind: %s; available memory kinds: %s",
          *sharding.memory_kind().memory_kind(),
          absl::StrJoin(sharding.devices()->devices().front()->Memories(), ", ",
                        [](std::string* out, xla::ifrt::Memory* ms) {
                          absl::StrAppend(out, *ms->Kind().memory_kind());
                        }));
    }
    return tensorflow::down_cast<xla::ifrt::PjRtMemory*>(memory)->pjrt_memory();
  } else {
    if (!device->IsAddressable()) {
      return xla::InvalidArgument(
          "Cannot copy array to non-addressable device %s",
          device->DebugString());
    }
    return tensorflow::down_cast<xla::ifrt::PjRtDevice*>(device)
        ->pjrt_device()
        ->default_memory_space();
  }
}

absl::StatusOr<tsl::RCReference<PullTable::Entry>> CreatePullEntry(
    const std::vector<xla::ifrt::ArrayRef>& arrs,
    std::shared_ptr<PremappedCopierState> state, size_t xfer_size,
    bool use_raw_buffers) {
  if (use_raw_buffers) {
    std::vector<RawBufferEntry::BufferRef> refs;
    for (auto& arr : arrs) {
      auto* pjrt_arr = llvm::dyn_cast_or_null<xla::ifrt::PjRtArray>(arr.get());
      if (pjrt_arr == nullptr) {
        return absl::InvalidArgumentError(
            "Cannot remote transfer non-pjrt arrays.");
      }
      for (auto& pjrt_buf : pjrt_arr->pjrt_buffers()) {
        TF_ASSIGN_OR_RETURN(size_t buf_size,
                            pjrt_buf->GetOnDeviceSizeInBytes());
        TF_ASSIGN_OR_RETURN(
            auto raw_buffer,
            xla::PjRtRawBuffer::CreateRawAliasOfBuffer(pjrt_buf.get()));
        refs.push_back(
            {pjrt_buf->GetReadyFuture(), std::move(raw_buffer), buf_size});
      }
    }
    return tsl::MakeRef<RawBufferEntry>(std::move(refs), state, xfer_size);
  }

  std::vector<PjRtBufferEntry::BufferRef> refs;
  for (auto& arr : arrs) {
    auto* pjrt_arr = llvm::dyn_cast_or_null<xla::ifrt::PjRtArray>(arr.get());
    if (pjrt_arr == nullptr) {
      return absl::InvalidArgumentError(
          "Cannot remote transfer non-pjrt arrays.");
    }
    for (auto& pjrt_buf : pjrt_arr->pjrt_buffers()) {
      TF_ASSIGN_OR_RETURN(size_t buf_size, pjrt_buf->GetOnDeviceSizeInBytes());
      refs.push_back({pjrt_buf, buf_size});
    }
  }
  return tsl::MakeRef<PjRtBufferEntry>(std::move(refs), state, xfer_size);
}

class PyTransferServerConnection {
 public:
  explicit PyTransferServerConnection(
      tsl::RCReference<SocketServer::Connection> conn)
      : conn_(std::move(conn)) {}

  void Pull(uint64_t uuid, std::vector<int> buffer_ids,
            std::vector<tsl::RCReference<ChunkDestination>> pull_dests) {
    for (size_t i = 0; i < buffer_ids.size(); ++i) {
      conn_->Pull(uuid, buffer_ids[i], std::move(pull_dests[i]));
    }
  }

  SocketServer::Connection& conn() { return *conn_; }

 private:
  tsl::RCReference<SocketServer::Connection> conn_;
};

class PyTransferServer {
 public:
  PyTransferServer() = default;
  absl::Status Start(xla::ifrt::Client* client, size_t max_num_parallel_copies,
                     size_t xfer_size, const SocketAddress& addr,
                     const std::vector<SocketAddress>& transport_addresses,
                     bool supports_pinned_allocator, bool use_raw_buffers) {
    use_raw_buffers_ = use_raw_buffers;
    std::shared_ptr<BulkTransportFactory> factory;
    std::shared_ptr<xla::PjRtClient> pjrt_client =
        tensorflow::down_cast<xla::ifrt::PjRtClient*>(client)
            ->shared_ptr_pjrt_client();
    if (transport_addresses.empty()) {
      factory = BulkTransportFactory::CreateLocal();
    } else {
      auto tmp = xla::ValueOrThrow(
          AllocateAlignedMemory(xfer_size * max_num_parallel_copies));
#if JAX_IFRT_VERSION_NUMBER >= 11
      SlabAllocator uallocator(xla::ValueOrThrow(MapPjrtMemory(
                                   pjrt_client, tmp->data(), tmp->size(), tmp)),
                               xfer_size);
#else
      SlabAllocator uallocator(xla::ValueOrThrow(MapPjrtMemory(
                                   client, tmp->data(), tmp->size(), tmp)),
                               xfer_size);
#endif
      std::optional<SlabAllocator> pinned_allocator;
      if (supports_pinned_allocator) {
        auto tmp = xla::ValueOrThrow(
            AllocateNetworkPinnedMemory(xfer_size * max_num_parallel_copies));
#if JAX_IFRT_VERSION_NUMBER >= 11
        pinned_allocator.emplace(
            xla::ValueOrThrow(
                MapPjrtMemory(pjrt_client, tmp->data(), tmp->size(), tmp)),
            xfer_size);
#else
        pinned_allocator.emplace(
            xla::ValueOrThrow(
                MapPjrtMemory(client, tmp->data(), tmp->size(), tmp)),
            xfer_size);
#endif
      }
      factory = xla::ValueOrThrow(CreateSocketBulkTransportFactory(
          transport_addresses, pinned_allocator, uallocator));
    }

    server_ = std::make_shared<SocketServer>();

#if JAX_IFRT_VERSION_NUMBER >= 11
    TF_ASSIGN_OR_RETURN(
        auto mem, AllocateAndMapPjrtMemory(
                      pjrt_client, max_num_parallel_copies * xfer_size * 2));
#else
    TF_ASSIGN_OR_RETURN(
        auto mem, AllocateAndMapPjrtMemory(
                      client, max_num_parallel_copies * xfer_size * 2));
#endif
    premapped_copier_ = std::make_shared<PremappedCopierState>(
        mem, max_num_parallel_copies, xfer_size);
    xfer_size_ = xfer_size;
    return server_->Start(addr, factory);
  }
  std::string address() { return server_->addr().ToString(); }

  PyTransferServerConnection Connect(const std::string& saddr) {
    return PyTransferServerConnection(
        server_->Connect(xla::ValueOrThrow(SocketAddress::Parse(saddr))));
  }

  void AwaitPull(uint64_t uuid, const std::vector<xla::ifrt::ArrayRef>& arrs) {
    server_->AwaitPull(
        uuid, xla::ValueOrThrow(CreatePullEntry(arrs, premapped_copier_,
                                                xfer_size_, use_raw_buffers_)));
  }

  size_t xfer_size() { return xfer_size_; }

  std::shared_ptr<PremappedCopierState> premapped_copier() {
    return premapped_copier_;
  }

 private:
  std::shared_ptr<SocketServer> server_;
  std::shared_ptr<PremappedCopierState> premapped_copier_;
  size_t xfer_size_;
  bool use_raw_buffers_ = false;
};

absl::StatusOr<xla::ifrt::ArraySpec> ArraySpecFromShapeDtypeStruct(
    nb::handle aval) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType dtype,
                      xla::DtypeToIfRtDType(
                          nb::borrow<xla::nb_dtype>(aval.attr("dtype").ptr())));
  auto shape_dims = nb::cast<std::vector<int64_t>>(aval.attr("shape"));
  auto shape = xla::ifrt::Shape(
      xla::ifrt::Shape::Dimensions(shape_dims.begin(), shape_dims.end()));
  TF_ASSIGN_OR_RETURN(auto sharding,
                      xla::GetIfrtHloSharding(aval.attr("sharding"), shape));
  return xla::ifrt::ArraySpec{dtype, std::move(shape), std::move(sharding)};
}

struct BufferSource {
  xla::ifrt::ArrayRef arr;
  xla::PjRtBuffer* buffer;
};

struct CopyDests {
  std::vector<xla::PjRtClient::ShapeSpec> shape_specs;
  xla::PjRtMemorySpace* memory_space;
};

void RegisterTransferServerTypes(nanobind::module_& m) {
  nb::class_<PyTransferServerConnection>(m, "TransferConnection")
#if JAX_IFRT_VERSION_NUMBER > 9
      .def(
          "_testonly_inject_failure",
          [](PyTransferServerConnection& self) { self.conn().InjectFailure(); })
#endif
      .def("_pull_flat", [](PyTransferServerConnection& self, uint64_t uuid,
                            xla::nb_class_ptr<xla::PyClient> py_client,
                            std::vector<nb::object> py_avals) {
        auto* ifrt_client = llvm::dyn_cast_or_null<xla::ifrt::PjRtClient>(
            py_client->ifrt_client());
        if (ifrt_client == nullptr) {
          xla::ThrowIfError(absl::InvalidArgumentError(
              "_pull_flat only supported on pjrt-ifrt clients."));
        }

        std::vector<xla::ifrt::ArraySpec> avals;
        std::vector<nb::object> shardings;
        shardings.reserve(py_avals.size());
        avals.reserve(py_avals.size());
        for (const auto& py_aval : py_avals) {
          avals.push_back(
              xla::ValueOrThrow(ArraySpecFromShapeDtypeStruct(py_aval)));
          shardings.push_back(py_aval.attr("sharding"));
        }

        std::vector<CopyDests> dests;
        std::vector<std::pair<int, int>> fetch_idxs;
        absl::flat_hash_map<xla::PjRtMemorySpace*, int> mapping;
        std::vector<std::vector<std::pair<int, int>>> buffer_list;

        for (auto& aval : avals) {
          std::vector<std::pair<int, int>> buf_list;
          auto prim_type =
              xla::ValueOrThrow(xla::ifrt::ToPrimitiveType(aval.dtype));
          auto shards = xla::ValueOrThrow(aval.sharding->Disassemble(
              aval.shape,
              xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));
          buf_list.reserve(shards.size());
          for (auto& shard : shards) {
            auto* mem_space =
                xla::ValueOrThrow(MemorySpaceFromSharding(*shard.second));
            int dest_idx =
                mapping.emplace(mem_space, static_cast<int>(dests.size()))
                    .first->second;
            if (dest_idx == dests.size()) {
              dests.emplace_back();
              dests.back().memory_space = mem_space;
            }
            fetch_idxs.push_back(
                {dest_idx,
                 static_cast<int>(dests[dest_idx].shape_specs.size())});
            buf_list.push_back(fetch_idxs.back());
            dests[dest_idx].shape_specs.push_back(
                {prim_type, xla::DimensionVector(shard.first.dims().begin(),
                                                 shard.first.dims().end())});
          }
          buffer_list.push_back(std::move(buf_list));
        }

        std::vector<
            std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager>>
            atms;
        atms.reserve(dests.size());

        for (auto& dest : dests) {
          atms.push_back(xla::ValueOrThrow(
              py_client->pjrt_client()->CreateBuffersForAsyncHostToDevice(
                  dest.shape_specs, std::nullopt, dest.memory_space)));
        }

        std::vector<tsl::RCReference<ChunkDestination>> pull_dests;
        std::vector<int> buffer_ids;
        pull_dests.reserve(fetch_idxs.size());
        buffer_ids.reserve(fetch_idxs.size());
        for (auto& fetch_idx : fetch_idxs) {
          auto& atm = atms[fetch_idx.first];
          pull_dests.push_back(MakeDmaDestination(
              atm, fetch_idx.second, atm->buffer_size(fetch_idx.second)));
          buffer_ids.push_back(static_cast<int>(buffer_ids.size()));
        }

        self.Pull(uuid, buffer_ids, std::move(pull_dests));

        std::vector<xla::PyArray> out;
        auto traceback = xla::Traceback::Get();
        for (size_t i = 0; i < buffer_list.size(); ++i) {
          xla::ifrt::PjRtArray::PjRtBuffers buffers;
          buffers.reserve(buffer_list[i].size());
          for (auto& v : buffer_list[i]) {
            buffers.push_back(atms[v.first]->RetrieveBuffer(v.second));
          }
          auto arr = xla::ValueOrThrow(xla::ifrt::PjRtArray::Create(
              ifrt_client, avals[i].dtype, avals[i].shape, avals[i].sharding,
              std::move(buffers), avals[i].layout));
          out.push_back(xla::PyArray::MakeFromIfrtArrayAndSharding(
              py_client, traceback, std::move(arr), shardings[i], false, true,
              /*skip_checks=*/false));
        }

        return out;
      });

  nb::class_<PyTransferServer>(m, "TransferServer")
      .def("address", [](PyTransferServer& self) { return self.address(); })
      .def("_await_pull_flat",
           [](PyTransferServer& self, uint64_t uuid,
              std::vector<xla::PyArray> inputs) {
             std::vector<xla::ifrt::ArrayRef> arrs;
             arrs.reserve(inputs.size());
             for (const xla::PyArray& input : inputs) {
               arrs.push_back(tsl::FormRef(input.ifrt_array()));
             }
             self.AwaitPull(uuid, arrs);
           })
      .def("connect", [](PyTransferServer& self, const std::string& address) {
        return self.Connect(address);
      });

  m.def(
      "start_transfer_server",
      [](xla::nb_class_ptr<xla::PyClient> py_client, std::string address,
         std::vector<std::string> transport_addresses_str,
         size_t max_num_parallel_copies, size_t transfer_size,
         bool supports_pinned_allocator,
         bool use_raw_buffers) -> PyTransferServer {
        PyTransferServer result;
        std::vector<SocketAddress> transport_addresses;
        transport_addresses.reserve(transport_addresses_str.size());
        for (const std::string& addr : transport_addresses_str) {
          transport_addresses.push_back(
              xla::ValueOrThrow(SocketAddress::Parse(addr)));
        }
        xla::ThrowIfError(result.Start(
            py_client->ifrt_client(), max_num_parallel_copies, transfer_size,
            xla::ValueOrThrow(SocketAddress::Parse(address)),
            transport_addresses, supports_pinned_allocator, use_raw_buffers));
        return result;
      },
      nb::arg("client"), nb::arg("address") = SocketAddress().ToString(),
      nb::arg("transport_addresses") = std::vector<std::string>(),
      nb::arg("max_num_parallel_copies") = 8,
      nb::arg("transfer_size") = 256 * 1024 * 1024,
      // Dual pinning not confirmed to be supported.
      nb::arg("supports_pinned_allocator") = false,
      // Technically unsafe (because a future donation won't wait for the
      // transfer to complete).
      nb::arg("use_raw_buffers") = false);
#if JAX_IFRT_VERSION_NUMBER >= 15
  m.def(
      "make_transfer_server_interface_factory",
      [](size_t transfer_size, int cross_host_transfer_timeout_seconds,
         std::shared_ptr<xla::DistributedRuntimeClient> distributed_client,
         const std::string& socket_address,
         const std::vector<std::string>& transport_addresses)
          -> xla::ifrt::TransferServerInterfaceFactory {
        std::shared_ptr<xla::KeyValueStoreInterface> kv_store =
            xla::GetDistributedKeyValueStore(distributed_client,
                                             "transfer_server:");
        auto factory_fn = xla::ValueOrThrow(
            xla::ifrt::PjRtTransferServer::MakePjRtTransferServerFactory(
                transfer_size,
                absl::Seconds(cross_host_transfer_timeout_seconds), kv_store,
                socket_address, transport_addresses));
        return xla::ifrt::TransferServerInterfaceFactory{std::move(factory_fn)};
      },
      nb::arg("transfer_size") = 256 * 1024 * 1024,
      nb::arg("cross_host_transfer_timeout_seconds") = 60,
      nb::arg("distributed_client").none() = nullptr,
      nb::arg("socket_address") = SocketAddress().ToString(),
      nb::arg("transport_addresses") = std::vector<std::string>());
#endif
}

}  // namespace aux
