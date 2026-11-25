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

#include "jaxlib/donation_recovery.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/version.h"
#include "xla/tsl/concurrency/future.h"

namespace jax {

namespace nb = nanobind;

namespace {

constinit thread_local DonationRecoveryScope* current_scope = nullptr;

}  // namespace

DonationRecoveryScope::~DonationRecoveryScope() { Exit(); }

DonationRecoveryScope* DonationRecoveryScope::Current() {
  return current_scope;
}

void DonationRecoveryScope::Enter() {
#if !defined(JAX_XLA_VERSION_NUMBER) || JAX_XLA_VERSION_NUMBER < 91
  xla::ThrowIfError(absl::UnimplementedError(
      "DonationRecovery requires JAX_XLA_VERSION_NUMBER >= 91."));
#else
  if (current_scope != nullptr) {
    throw nb::value_error(
        "Nested DonationRecovery scopes are not supported on the same thread.");
  }
  current_scope = this;
#endif
}

void DonationRecoveryScope::Exit() {
  if (current_scope == this) {
    current_scope = nullptr;
  }
}

void DonationRecoveryScope::RecordExecution(
    tsl::Future<std::vector<xla::ifrt::ArrayRef>> recovery_future,
    nb_class_ptr<PyClient> client,
    std::vector<const xla::ifrt::Array*> input_ifrt_arrays,
    nanobind::object py_args) {
  recovery_future_ = std::make_shared<RecoveryFuture>(
      std::move(recovery_future), std::move(client),
      std::move(input_ifrt_arrays), std::move(py_args));
}

void DonationRecoveryScope::SetPyArgs(nanobind::object py_args) {
  if (recovery_future_ != nullptr) {
    recovery_future_->set_py_args(std::move(py_args));
  }
}

std::shared_ptr<RecoveryFuture> DonationRecoveryScope::future() {
  if (recovery_future_ == nullptr) {
    throw std::runtime_error(
        "No JIT execution occurred within the DonationRecovery scope.");
  }
  return recovery_future_;
}

RecoveryFuture::RecoveryFuture(
    tsl::Future<std::vector<xla::ifrt::ArrayRef>> recovery_future,
    nb_class_ptr<PyClient> client,
    std::vector<const xla::ifrt::Array*> input_ifrt_arrays,
    nanobind::object py_args)
    : recovery_future_(std::move(recovery_future)),
      client_(std::move(client)),
      input_ifrt_arrays_(std::move(input_ifrt_arrays)),
      py_args_(std::move(py_args)) {
  if (!recovery_future_.IsValid()) {
    recovery_future_ =
        tsl::Future<std::vector<xla::ifrt::ArrayRef>>(absl::UnimplementedError(
            "Donation recovery is not supported by this backend."));
  }
}

bool RecoveryFuture::is_ready() const { return recovery_future_.IsReady(); }

nanobind::object RecoveryFuture::get(nanobind::sequence leaves) {
  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> recovered_arrays;
  {
    nb::gil_scoped_release release;
    recovered_arrays = recovery_future_.Await();
  }
  if (!recovered_arrays.ok()) {
    xla::ThrowIfError(recovered_arrays.status());
  }
  if (recovered_arrays->empty()) {
    return nb::none();
  }

  const size_t num_inputs =
      std::min(recovered_arrays->size(), input_ifrt_arrays_.size());
  absl::flat_hash_map<const xla::ifrt::Array*, size_t>
      recovered_index_by_ifrt_array;
  recovered_index_by_ifrt_array.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    if ((*recovered_arrays)[i] != nullptr && input_ifrt_arrays_[i] != nullptr) {
      recovered_index_by_ifrt_array.emplace(input_ifrt_arrays_[i], i);
    }
  }

  absl::flat_hash_map<const xla::ifrt::Array*, PyArray>
      materialized_by_ifrt_array;
  nb::list result;
  for (nb::handle leaf : leaves) {
    if (!PyArray::IsPyArray(leaf)) {
      result.append(leaf);
      continue;
    }
    PyArray orig = nb::borrow<PyArray>(leaf);
    const xla::ifrt::Array* orig_ifrt = orig.ifrt_array_ref().get();
    if (auto mat_it = materialized_by_ifrt_array.find(orig_ifrt);
        mat_it != materialized_by_ifrt_array.end()) {
      result.append(mat_it->second);
      continue;
    }
    auto it = recovered_index_by_ifrt_array.find(orig_ifrt);
    if (it == recovered_index_by_ifrt_array.end()) {
      result.append(leaf);
      continue;
    }
    absl::Span<const int64_t> shape = orig.shape();
    PyArray recovered_py_array(
        orig.aval(), orig.weak_type(), orig.dtype(),
        std::vector<int64_t>(shape.begin(), shape.end()), orig.sharding(),
        client_, std::move((*recovered_arrays)[it->second]), orig.committed());
    materialized_by_ifrt_array.emplace(orig_ifrt, recovered_py_array);
    result.append(std::move(recovered_py_array));
  }
  return result;
}

void BuildDonationRecoverySubmodule(nanobind::module_& m) {
  nb::module_ dr = m.def_submodule("donation_recovery",
                                   "JAX donation recovery support library");

  nb::class_<DonationRecoveryScope>(dr, "DonationRecoveryScope")
      .def(nb::init<>())
      .def("enter", &DonationRecoveryScope::Enter)
      .def("exit", &DonationRecoveryScope::Exit)
      .def("future", &DonationRecoveryScope::future);

  nb::class_<RecoveryFuture>(dr, "RecoveryFuture")
      .def("is_ready", &RecoveryFuture::is_ready)
      .def("get", &RecoveryFuture::get)
      .def_prop_ro("py_args", &RecoveryFuture::py_args);
}

}  // namespace jax
