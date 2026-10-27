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

#ifndef JAXLIB_CPU_THREAD_POOL_H_
#define JAXLIB_CPU_THREAD_POOL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "Eigen/ThreadPool"
#include "xla/backends/cpu/runtime/work_queue.h"
#include "xla/ffi/api/ffi.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/math/math_util.h"

namespace jax {

// Bridges `xla::ffi::ThreadPool` to `Eigen::ThreadPoolInterface` so XLA:CPU's
// work-stealing `xla::cpu::Worker::Parallelize` can schedule chunks onto the
// intra-op pool.
class FfiThreadPoolAdapter final : public Eigen::ThreadPoolInterface {
 public:
  explicit FfiThreadPoolAdapter(xla::ffi::ThreadPool thread_pool)
      : thread_pool_(thread_pool) {}

  void Schedule(std::function<void()> fn) override {
    {
      absl::MutexLock lock(&mu_);
      if (shutdown_) {
        return;
      }
      ++active_schedules_;
    }
    struct ActiveScheduleGuard {
      FfiThreadPoolAdapter* self;
      ~ActiveScheduleGuard() {
        absl::MutexLock lock(&self->mu_);
        --self->active_schedules_;
      }
    } guard{this};
    thread_pool_.Schedule(std::move(fn));
  }

  void Shutdown() {
    absl::MutexLock lock(&mu_);
    shutdown_ = true;
    mu_.Await(absl::Condition(
        +[](int* active) { return *active == 0; }, &active_schedules_));
  }

  int NumThreads() const override {
    return static_cast<int>(thread_pool_.num_threads());
  }

  int CurrentThreadId() const override { return -1; }

 private:
  absl::Mutex mu_;
  bool shutdown_ ABSL_GUARDED_BY(mu_) = false;
  int active_schedules_ ABSL_GUARDED_BY(mu_) = 0;
  xla::ffi::ThreadPool thread_pool_;
};

// Divides `count` items into chunks of at most `chunk_size` and invokes
// `fn(start, chunk_count)` on each chunk using XLA:CPU's work-stealing
// `Worker::Parallelize`.
inline xla::ffi::Error ParallelFor(
    xla::ffi::ThreadPool thread_pool, int64_t count, int64_t chunk_size,
    absl::FunctionRef<void(int64_t, int64_t)> fn) {
  if (count <= 0) {
    return xla::ffi::Error::Success();
  }
  chunk_size = std::max<int64_t>(1, chunk_size);
  if (chunk_size >= count || thread_pool.num_threads() <= 1) {
    fn(0, count);
    return xla::ffi::Error::Success();
  }
  const size_t num_chunks =
      static_cast<size_t>(tsl::MathUtil::CeilOfRatio(count, chunk_size));
  const size_t num_workers = std::min<size_t>(
      num_chunks, static_cast<size_t>(thread_pool.num_threads()));
  auto pool = std::make_shared<FfiThreadPoolAdapter>(thread_pool);
  tsl::BlockUntilReady(xla::cpu::Worker::Parallelize(
      pool.get(), num_workers, num_chunks,
      [pool, count, chunk_size, fn](size_t chunk_idx) {
        const int64_t start = static_cast<int64_t>(chunk_idx) * chunk_size;
        const int64_t chunk_count =
            std::min<int64_t>(chunk_size, count - start);
        fn(start, chunk_count);
      }));
  pool->Shutdown();
  return xla::ffi::Error::Success();
}

}  // namespace jax

#endif  // JAXLIB_CPU_THREAD_POOL_H_
