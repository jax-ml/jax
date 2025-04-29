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

#include <stdexcept>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/solver_handle_pool.h"
#include "jaxlib/gpu/solver_kernels.h"
#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/tsl/python/lib/core/numpy.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;

// Converts a NumPy dtype to a Type.
SolverType DtypeToSolverType(const dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, SolverType>({
          {{'f', 4}, SolverType::F32},
          {{'f', 8}, SolverType::F64},
          {{'c', 8}, SolverType::C64},
          {{'c', 16}, SolverType::C128},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    nb::str repr = nb::repr(np_type);
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", repr.c_str()));
  }
  return it->second;
}

#ifdef JAX_GPU_CUDA

// csrlsvqr: Linear system solve via Sparse QR

// Returns a descriptor for a csrlsvqr operation.
nb::bytes BuildCsrlsvqrDescriptor(const dtype& dtype, int n, int nnzA,
                                  int reorder, double tol) {
  SolverType type = DtypeToSolverType(dtype);
  return PackDescriptor(CsrlsvqrDescriptor{type, n, nnzA, reorder, tol});
}

#endif  // JAX_GPU_CUDA

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, nb::bytes> BuildSytrdDescriptor(const dtype& dtype, bool lower,
                                               int b, int n) {
  SolverType type = DtypeToSolverType(dtype);
  auto h = SolverHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  gpusolverFillMode_t uplo =
      lower ? GPUSOLVER_FILL_MODE_LOWER : GPUSOLVER_FILL_MODE_UPPER;
  switch (type) {
    case SolverType::F32:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusolverDnSsytrd_bufferSize(
          handle.get(), uplo, n, /*A=*/nullptr, /*lda=*/n, /*D=*/nullptr,
          /*E=*/nullptr, /*tau=*/nullptr, &lwork)));
      break;
    case SolverType::F64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusolverDnDsytrd_bufferSize(
          handle.get(), uplo, n, /*A=*/nullptr, /*lda=*/n, /*D=*/nullptr,
          /*E=*/nullptr, /*tau=*/nullptr, &lwork)));
      break;
    case SolverType::C64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusolverDnChetrd_bufferSize(
          handle.get(), uplo, n, /*A=*/nullptr, /*lda=*/n, /*D=*/nullptr,
          /*E=*/nullptr, /*tau=*/nullptr, &lwork)));
      break;
    case SolverType::C128:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusolverDnZhetrd_bufferSize(
          handle.get(), uplo, n, /*A=*/nullptr, /*lda=*/n, /*D=*/nullptr,
          /*E=*/nullptr, /*tau=*/nullptr, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(SytrdDescriptor{type, uplo, b, n, n, lwork})};
}

nb::dict Registrations() {
  nb::dict dict;
  dict[JAX_GPU_PREFIX "solver_sytrd"] = EncapsulateFunction(Sytrd);

#ifdef JAX_GPU_CUDA
  dict["cusolver_csrlsvqr"] = EncapsulateFunction(Csrlsvqr);
#endif  // JAX_GPU_CUDA

  dict[JAX_GPU_PREFIX "solver_getrf_ffi"] = EncapsulateFfiHandler(GetrfFfi);
  dict[JAX_GPU_PREFIX "solver_geqrf_ffi"] = EncapsulateFfiHandler(GeqrfFfi);
  dict[JAX_GPU_PREFIX "solver_orgqr_ffi"] = EncapsulateFfiHandler(OrgqrFfi);
  dict[JAX_GPU_PREFIX "solver_syevd_ffi"] = EncapsulateFfiHandler(SyevdFfi);
  dict[JAX_GPU_PREFIX "solver_syrk_ffi"] = EncapsulateFfiHandler(SyrkFfi);
  dict[JAX_GPU_PREFIX "solver_gesvd_ffi"] = EncapsulateFfiHandler(GesvdFfi);
  dict[JAX_GPU_PREFIX "solver_sytrd_ffi"] = EncapsulateFfiHandler(SytrdFfi);

#ifdef JAX_GPU_CUDA
  dict[JAX_GPU_PREFIX "solver_gesvdj_ffi"] = EncapsulateFfiHandler(GesvdjFfi);
  dict[JAX_GPU_PREFIX "solver_csrlsvqr_ffi"] =
      EncapsulateFfiHandler(CsrlsvqrFfi);
#endif  // JAX_GPU_CUDA

  return dict;
}

NB_MODULE(_solver, m) {
  tsl::ImportNumpy();
  m.def("registrations", &Registrations);
  m.def("build_sytrd_descriptor", &BuildSytrdDescriptor);
#ifdef JAX_GPU_CUDA
  m.def("build_csrlsvqr_descriptor", &BuildCsrlsvqrDescriptor);
#endif  // JAX_GPU_CUDA
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
