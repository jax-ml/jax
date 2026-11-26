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

#ifndef JAXLIB_DONATION_RECOVERY_H_
#define JAXLIB_DONATION_RECOVERY_H_

#include <memory>
#include <vector>

#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/tsl/concurrency/future.h"

namespace jax {

class RecoveryFuture;

// Thread-local RAII scope tracking JIT buffer donation recovery.
class DonationRecoveryScope {
 public:
  DonationRecoveryScope() = default;
  ~DonationRecoveryScope();

  DonationRecoveryScope(const DonationRecoveryScope&) = delete;
  DonationRecoveryScope& operator=(const DonationRecoveryScope&) = delete;

  // Returns the active scope for the current thread, or nullptr if none.
  static DonationRecoveryScope* Current();

  void Enter();
  void Exit();
  bool has_executed() const { return recovery_future_ != nullptr; }

  // Called during JIT execution when DonationRecoveryScope is active.
  void RecordExecution(
      tsl::Future<std::vector<xla::ifrt::ArrayRef>> recovery_future,
      nb_class_ptr<PyClient> client,
      std::vector<const xla::ifrt::Array*> input_ifrt_arrays,
      nanobind::object py_args);

  // Updates the recorded Python (args, kwargs) tuple when an outer C++ pjit
  // frame wraps a Python cache-miss execution.
  void SetPyArgs(nanobind::object py_args);

  std::shared_ptr<RecoveryFuture> future();

 private:
  std::shared_ptr<RecoveryFuture> recovery_future_;
};

class RecoveryFuture {
 public:
  RecoveryFuture(tsl::Future<std::vector<xla::ifrt::ArrayRef>> recovery_future,
                 nb_class_ptr<PyClient> client,
                 std::vector<const xla::ifrt::Array*> input_ifrt_arrays,
                 nanobind::object py_args);

  bool is_ready() const;
  nanobind::object get(nanobind::sequence leaves);
  nanobind::object py_args() const { return py_args_; }
  void set_py_args(nanobind::object py_args) { py_args_ = std::move(py_args); }

 private:
  tsl::Future<std::vector<xla::ifrt::ArrayRef>> recovery_future_;
  nb_class_ptr<PyClient> client_;
  std::vector<const xla::ifrt::Array*> input_ifrt_arrays_;
  nanobind::object py_args_;
};

void BuildDonationRecoverySubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_DONATION_RECOVERY_H_
