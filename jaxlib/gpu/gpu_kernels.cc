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

#include "jaxlib/gpu/linalg_kernels.h"
#include "jaxlib/gpu/prng_kernels.h"
#include "jaxlib/gpu/rnn_kernels.h"
#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/sparse_kernels.h"
#include "jaxlib/gpu/triton_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_target_registry.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cudnn_rnn", "CUDA", RNNForwardFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cudnn_rnn_bwd", "CUDA",
                         RNNBackwardFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_getrf_ffi", "CUDA",
                         GetrfFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_syrk_ffi", "CUDA",
                         SyrkFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_geqrf_ffi", "CUDA",
                         GeqrfFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_csrlsvqr_ffi", "CUDA",
                         CsrlsvqrFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_orgqr_ffi", "CUDA",
                         OrgqrFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_syevd_ffi", "CUDA",
                         SyevdFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_sytrd_ffi", "CUDA",
                         SytrdFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_gesvd_ffi", "CUDA",
                         GesvdFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_gesvdj_ffi", "CUDA",
                         GesvdjFfi);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cu_cholesky_update_ffi", "CUDA",
                         CholeskyUpdateFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cu_lu_pivots_to_permutation",
                         "CUDA", LuPivotsToPermutation);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cu_threefry2x32_ffi", "CUDA",
                         ThreeFry2x32Ffi);

#if JAX_GPU_HAVE_SPARSE
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_csr_todense_ffi", "CUDA",
                         CsrToDenseFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_csr_fromdense_ffi", "CUDA",
                         CsrFromDenseFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_csr_matvec_ffi", "CUDA",
                         CsrMatvecFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_csr_matmat_ffi", "CUDA",
                         CsrMatmatFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_coo_todense_ffi", "CUDA",
                         CooToDenseFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_coo_fromdense_ffi", "CUDA",
                         CooFromDenseFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_coo_matvec_ffi", "CUDA",
                         CooMatvecFfi);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_coo_matmat_ffi", "CUDA",
                         CooMatmatFfi);
#endif
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusparse_gtsv2_ffi", "CUDA",
                         kGtsv2);

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("triton_kernel_call", TritonKernelCall,
                                         "CUDA");

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
