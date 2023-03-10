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

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "jaxlib/gpu/blas_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_pybind11_helpers.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace py = pybind11;

// Converts a NumPy dtype to a Type.
BlasType DtypeToBlasType(const py::dtype& np_type) {
  static auto* types = new absl::flat_hash_map<std::pair<char, int>, BlasType>({
      {{'f', 4}, BlasType::F32},
      {{'f', 8}, BlasType::F64},
      {{'c', 8}, BlasType::C64},
      {{'c', 16}, BlasType::C128},
  });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", py::repr(np_type)));
  }
  return it->second;
}

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, py::bytes> BuildGetrfBatchedDescriptor(const py::dtype& dtype,
                                                         int b, int n) {
  BlasType type = DtypeToBlasType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GetrfBatchedDescriptor{type, b, n})};
}

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, py::bytes> BuildGeqrfBatchedDescriptor(const py::dtype& dtype,
                                                         int b, int m, int n) {
  BlasType type = DtypeToBlasType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GeqrfBatchedDescriptor{type, b, m, n})};
}

py::dict Registrations() {
  py::dict dict;
  dict[JAX_GPU_PREFIX "blas_getrf_batched"] = EncapsulateFunction(GetrfBatched);
  dict[JAX_GPU_PREFIX "blas_geqrf_batched"] = EncapsulateFunction(GeqrfBatched);
  return dict;
}

PYBIND11_MODULE(_blas, m) {
  m.def("registrations", &Registrations);
  m.def("build_getrf_batched_descriptor", &BuildGetrfBatchedDescriptor);
  m.def("build_geqrf_batched_descriptor", &BuildGeqrfBatchedDescriptor);
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
