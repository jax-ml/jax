/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_NUMPY_H_
#define JAXLIB_NUMPY_H_

#include "nanobind/nanobind.h"
#include "xla/python/nb_numpy.h"

namespace jax {

struct NumpyTypes {
  static const NumpyTypes& Get();

  xla::nb_dtype int32_dtype;
  xla::nb_dtype int64_dtype;
  xla::nb_dtype uint32_dtype;
  xla::nb_dtype float32_dtype;
  xla::nb_dtype float64_dtype;
  xla::nb_dtype complex64_dtype;
  xla::nb_dtype complex128_dtype;
  xla::nb_dtype string_dtype;
  nanobind::object numpy_generic;
};

}  // namespace jax

#endif  // JAXLIB_NUMPY_H_
