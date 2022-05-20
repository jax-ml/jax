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

// This file is not used by JAX itself, but exists to assist with running
// JAX-generated HLO code from outside of JAX.

#include "jaxlib/hip/hipblas_kernels.h"
#include "jaxlib/hip/hip_lu_pivot_kernels.h"
#include "jaxlib/hip/hip_prng_kernels.h"
#include "jaxlib/hip/hipsolver_kernels.h"
#include "jaxlib/hip/hipsparse_kernels.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

namespace jax {
namespace {

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipblas_trsm_batched", TrsmBatched,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipblas_getrf_batched", GetrfBatched,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hip_lu_pivots_to_permutation",
                                         HipLuPivotsToPermutation, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hip_threefry2x32", HipThreeFry2x32,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_potrf", Potrf, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_getrf", Getrf, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_geqrf", Geqrf, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_orgqr", Orgqr, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_syevd", Syevd, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_syevj", Syevj, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_gesvd", Gesvd, "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsolver_gesvdj", Gesvdj, "ROCM");

XLT_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_csr_todense", CsrToDense,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_csr_fromdense", CsrFromDense,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_csr_matvec", CsrMatvec,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_csr_matmat", CsrMatmat,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_coo_todense", CooToDense,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_coo_fromdense", CooFromDense,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_coo_matvec", CooMatvec,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_coo_matmat", CooMatmat,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_gtsv2_f32", gtsv2_f32,
                                         "ROCM");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("hipsparse_gtsv2_f64", gtsv2_f64,
                                         "ROCM");

}  // namespace
}  // namespace jax
