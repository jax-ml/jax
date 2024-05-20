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

#include "jaxlib/cpu/cpu_kernels.h"
#include "xla/ffi/api/c_api.h"

#define JAX_CPU_REGISTER_HANDLER(name) \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

namespace jax {
namespace {

JAX_CPU_REGISTER_HANDLER(blas_strsm);
JAX_CPU_REGISTER_HANDLER(blas_dtrsm);
JAX_CPU_REGISTER_HANDLER(blas_ctrsm);
JAX_CPU_REGISTER_HANDLER(blas_ztrsm);
JAX_CPU_REGISTER_HANDLER(lapack_sgetrf);
JAX_CPU_REGISTER_HANDLER(lapack_dgetrf);
JAX_CPU_REGISTER_HANDLER(lapack_cgetrf);
JAX_CPU_REGISTER_HANDLER(lapack_zgetrf);
JAX_CPU_REGISTER_HANDLER(lapack_sgeqrf);
JAX_CPU_REGISTER_HANDLER(lapack_dgeqrf);
JAX_CPU_REGISTER_HANDLER(lapack_cgeqrf);
JAX_CPU_REGISTER_HANDLER(lapack_zgeqrf);
JAX_CPU_REGISTER_HANDLER(lapack_sorgqr);
JAX_CPU_REGISTER_HANDLER(lapack_dorgqr);
JAX_CPU_REGISTER_HANDLER(lapack_cungqr);
JAX_CPU_REGISTER_HANDLER(lapack_zungqr);
JAX_CPU_REGISTER_HANDLER(lapack_spotrf);
JAX_CPU_REGISTER_HANDLER(lapack_dpotrf);
JAX_CPU_REGISTER_HANDLER(lapack_cpotrf);
JAX_CPU_REGISTER_HANDLER(lapack_zpotrf);
JAX_CPU_REGISTER_HANDLER(lapack_sgesdd);
JAX_CPU_REGISTER_HANDLER(lapack_dgesdd);
JAX_CPU_REGISTER_HANDLER(lapack_cgesdd);
JAX_CPU_REGISTER_HANDLER(lapack_zgesdd);
JAX_CPU_REGISTER_HANDLER(lapack_ssyevd);
JAX_CPU_REGISTER_HANDLER(lapack_dsyevd);
JAX_CPU_REGISTER_HANDLER(lapack_cheevd);
JAX_CPU_REGISTER_HANDLER(lapack_zheevd);
JAX_CPU_REGISTER_HANDLER(lapack_sgeev);
JAX_CPU_REGISTER_HANDLER(lapack_dgeev);
JAX_CPU_REGISTER_HANDLER(lapack_cgeev);
JAX_CPU_REGISTER_HANDLER(lapack_zgeev);
JAX_CPU_REGISTER_HANDLER(lapack_sgees);
JAX_CPU_REGISTER_HANDLER(lapack_dgees);
JAX_CPU_REGISTER_HANDLER(lapack_cgees);
JAX_CPU_REGISTER_HANDLER(lapack_zgees);
JAX_CPU_REGISTER_HANDLER(lapack_sgehrd);
JAX_CPU_REGISTER_HANDLER(lapack_dgehrd);
JAX_CPU_REGISTER_HANDLER(lapack_cgehrd);
JAX_CPU_REGISTER_HANDLER(lapack_zgehrd);
JAX_CPU_REGISTER_HANDLER(lapack_ssytrd);
JAX_CPU_REGISTER_HANDLER(lapack_dsytrd);
JAX_CPU_REGISTER_HANDLER(lapack_chetrd);
JAX_CPU_REGISTER_HANDLER(lapack_zhetrd);

#undef JAX_CPU_REGISTER_HANDLER

}  // namespace
}  // namespace jax
