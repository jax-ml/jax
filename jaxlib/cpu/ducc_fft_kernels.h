/* Copyright 2020 The JAX Authors.

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

#ifndef JAXLIB_CPU_DUCC_FFT_KERNELS_H_
#define JAXLIB_CPU_DUCC_FFT_KERNELS_H_

#include "xla/service/custom_call_status.h"

namespace jax {


// TODO(b/311175955): this must be kept until May 2024 for backwards
// of serialized functions using fft.
void DynamicDuccFft(void* out, void** in, XlaCustomCallStatus*);

}  // namespace jax

#endif  // JAXLIB_CPU_DUCC_FFT_KERNELS_H_
