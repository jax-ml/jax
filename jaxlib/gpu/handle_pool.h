/* Copyright 2019 The JAX Authors.

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

#ifndef JAXLIB_GPU_HANDLE_POOL_H_
#define JAXLIB_GPU_HANDLE_POOL_H_

#include <map>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace jax {

// To avoid creating cublas/cusolver contexts in the middle of execution, we
// maintain a pool of them.
template <typename HandleType, typename StreamType>
class HandlePool {
 public:
  HandlePool() = default;

  // RAII class representing a cublas/cusolver handle borrowed from the pool.
  // Returns the handle to the pool on destruction.
  class Handle {
   public:
    Handle() = default;
    ~Handle() {
      if (pool_) {
        pool_->Return(handle_, stream_);
      }
    }

    Handle(Handle const&) = delete;
    Handle(Handle&& other) {
      pool_ = other.pool_;
      handle_ = other.handle_;
      stream_ = other.stream_;
      other.pool_ = nullptr;
      other.handle_ = nullptr;
      other.stream_ = nullptr;
    }
    Handle& operator=(Handle const&) = delete;
    Handle& operator=(Handle&& other) {
      pool_ = other.pool_;
      handle_ = other.handle_;
      stream_ = other.stream_;
      other.pool_ = nullptr;
      other.handle_ = nullptr;
      other.stream_ = nullptr;
      return *this;
    }

    HandleType get() { return handle_; }

   private:
    friend class HandlePool<HandleType, StreamType>;
    Handle(HandlePool<HandleType, StreamType>* pool, HandleType handle,
           StreamType stream)
        : pool_(pool), handle_(handle), stream_(stream) {}
    HandlePool<HandleType, StreamType>* pool_ = nullptr;
    HandleType handle_ = nullptr;
    StreamType stream_ = nullptr;
  };

  // Borrows a handle from the pool. If 'stream' is non-null, sets the stream
  // associated with the handle.
  static absl::StatusOr<Handle> Borrow(StreamType stream);

 private:
  static HandlePool<HandleType, StreamType>* Instance();

  void Return(HandleType handle, StreamType stream);

  absl::Mutex mu_;
  std::map<StreamType, std::vector<HandleType>> handles_ ABSL_GUARDED_BY(mu_);
};

template <typename HandleType, typename StreamType>
/*static*/ HandlePool<HandleType, StreamType>*
HandlePool<HandleType, StreamType>::Instance() {
  static auto* pool = new HandlePool<HandleType, StreamType>;
  return pool;
}

template <typename HandleType, typename StreamType>
void HandlePool<HandleType, StreamType>::Return(HandleType handle,
                                                StreamType stream) {
  absl::MutexLock lock(&mu_);
  handles_[stream].push_back(handle);
}

// template <typename HandleType, typename StreamType>
// HandlePool<HandleType, StreamType>::Borrow(StreamType stream)

}  // namespace jax

#endif  // JAXLIB_GPU_HANDLE_POOL_H_
