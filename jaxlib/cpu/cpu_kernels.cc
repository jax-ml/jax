/* Copyright 2021 The JAX Authors.

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

#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/cpu/sparse_kernels.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#define JAX_CPU_REGISTER_HANDLER(name) \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

namespace jax {
namespace {

JAX_CPU_REGISTER_HANDLER(lapack_strsm_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dtrsm_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_ctrsm_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_ztrsm_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgeqp3_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgeqp3_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgeqp3_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgeqp3_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sorgqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dorgqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cungqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zungqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_spotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dpotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cpotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zpotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgesvd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgesvd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgesvd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgesvd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_ssyevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dsyevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cheevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zheevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_ssytrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dsytrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_chetrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zhetrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgees_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgees_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgees_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgees_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgehrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgehrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgehrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgehrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgtsv_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgtsv_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgtsv_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgtsv_ffi);

JAX_CPU_REGISTER_HANDLER(cpu_csr_sparse_dense_ffi);

#undef JAX_CPU_REGISTER_HANDLER

}  // namespace
}  // namespace jax
