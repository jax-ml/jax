/* Copyright 2026 The JAX Authors.

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

#include "nanobind/nanobind.h"
#include "jaxlib/gpu/digamma_kernels.h"
#include "jaxlib/gpu/lgamma_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;

nb::dict Registrations() {
  nb::dict dict;
  dict[JAX_GPU_PREFIX "_digamma_ffi"] = EncapsulateFfiHandler(DigammaFfi);
  dict[JAX_GPU_PREFIX "_lgamma_ffi"] = EncapsulateFfiHandler(LgammaFfi);
  return dict;
}

NB_MODULE(_math, m) { m.def("registrations", &Registrations); }

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
