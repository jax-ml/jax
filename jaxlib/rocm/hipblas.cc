/* Copyright 2021 Google LLC

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

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "jaxlib/rocm/hipblas_kernels.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/hipblas.h"

namespace jax {
namespace {

namespace py = pybind11;

// Converts a NumPy dtype to a Type.
HipblasType DtypeToHipblasType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, HipblasType>({
          {{'f', 4}, HipblasType::F32},
          {{'f', 8}, HipblasType::F64},
          {{'c', 8}, HipblasType::C64},
          {{'c', 16}, HipblasType::C128},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", py::repr(np_type)));
  }
  return it->second;
}

// Returns the descriptor for a TrsmBatched operation.
std::pair<size_t, py::bytes>
BuildTrsmBatchedDescriptor(const py::dtype& dtype, int batch, int m, int n,
                           bool left_side, bool lower, bool trans_a,
                           bool conj_a, bool unit_diagonal) {
  size_t size = batch * sizeof(void*);
  TrsmBatchedDescriptor desc;
  desc.type = DtypeToHipblasType(dtype);
  desc.batch = batch;
  desc.m = m;
  desc.n = n;
  desc.side = left_side ? HIPBLAS_SIDE_LEFT : HIPBLAS_SIDE_RIGHT;
  desc.uplo = lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER;
  desc.trans = trans_a ? (conj_a ? HIPBLAS_OP_C : HIPBLAS_OP_T) : HIPBLAS_OP_N;
  desc.diag = unit_diagonal ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT;
  return {size, PackDescriptor(desc)};
}

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, py::bytes> BuildGetrfBatchedDescriptor(const py::dtype& dtype,
                                                         int b, int n) {
  HipblasType type = DtypeToHipblasType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GetrfBatchedDescriptor{type, b, n})};
}

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, py::bytes> BuildGeqrfBatchedDescriptor(const py::dtype& dtype,
                                                         int b, int m, int n) {
  HipblasType type = DtypeToHipblasType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GeqrfBatchedDescriptor{type, b, m, n})};
}

py::dict Registrations() {
  py::dict dict;
  dict["hipblas_trsm_batched"] = EncapsulateFunction(TrsmBatched);
  dict["hipblas_getrf_batched"] = EncapsulateFunction(GetrfBatched);
  dict["hipblas_geqrf_batched"] = EncapsulateFunction(GeqrfBatched);
  return dict;
}

PYBIND11_MODULE(_hipblas, m) {
  m.def("registrations", &Registrations);
  m.def("build_trsm_batched_descriptor", &BuildTrsmBatchedDescriptor);
  m.def("build_getrf_batched_descriptor", &BuildGetrfBatchedDescriptor);
  m.def("build_geqrf_batched_descriptor", &BuildGeqrfBatchedDescriptor);
}

}  // namespace
}  // namespace jax
