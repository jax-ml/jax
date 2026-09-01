/* Copyright 2026 The JAX Authors

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

#ifndef JAXLIB_FT_MUTEX_H_
#define JAXLIB_FT_MUTEX_H_

#include "absl/base/thread_annotations.h"
#include "nanobind/nanobind.h"

namespace jax {

// Free-threaded mutex that delegates to nanobind::ft_mutex (which maps to
// PyMutex in free-threaded Python builds and a no-op under the GIL), annotated
// with Abseil thread-safety attributes.
class ABSL_LOCKABLE ft_mutex {
 public:
  ft_mutex() = default;
  ft_mutex(const ft_mutex&) = delete;
  ft_mutex& operator=(const ft_mutex&) = delete;

  void lock() ABSL_EXCLUSIVE_LOCK_FUNCTION() { mutex_.lock(); }
  void unlock() ABSL_UNLOCK_FUNCTION() { mutex_.unlock(); }

  nanobind::ft_mutex& mutex() { return mutex_; }
  const nanobind::ft_mutex& mutex() const { return mutex_; }

 private:
  friend class ft_lock_guard;
  nanobind::ft_mutex mutex_;
};

// Scoped lock guard for jax::ft_mutex that delegates to
// nanobind::ft_lock_guard, annotated with Abseil thread-safety attributes.
class ABSL_SCOPED_LOCKABLE ft_lock_guard {
 public:
  explicit ft_lock_guard(ft_mutex& m) ABSL_EXCLUSIVE_LOCK_FUNCTION(m)
      : guard_(m.mutex_) {}
  ~ft_lock_guard() ABSL_UNLOCK_FUNCTION() = default;

  ft_lock_guard(const ft_lock_guard&) = delete;
  ft_lock_guard& operator=(const ft_lock_guard&) = delete;

 private:
  nanobind::ft_lock_guard guard_;
};

}  // namespace jax

#endif  // JAXLIB_FT_MUTEX_H_
