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

#include "jaxlib/numpy.h"

#include <Python.h>

#include <memory>

#include "nanobind/nanobind.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/safe_static_init.h"
#include "xla/tsl/python/lib/core/numpy.h"

namespace jax {

namespace nb = nanobind;

const NumpyTypes& NumpyTypes::Get() {
  static xla::SafeStatic<NumpyTypes> dtypes_init;
  return dtypes_init.Get([] {
    auto d = std::make_unique<NumpyTypes>();
    auto descr_from_type = [](int type) {
      return nb::steal<xla::nb_dtype>(
          reinterpret_cast<PyObject*>(PyArray_DescrFromType(type)));
    };
    d->int32_dtype = descr_from_type(NPY_INT32);
    d->int64_dtype = descr_from_type(NPY_INT64);
    d->uint32_dtype = descr_from_type(NPY_UINT32);
    d->float32_dtype = descr_from_type(NPY_FLOAT32);
    d->float64_dtype = descr_from_type(NPY_FLOAT64);
    d->complex64_dtype = descr_from_type(NPY_COMPLEX64);
    d->complex128_dtype = descr_from_type(NPY_COMPLEX128);
    d->string_dtype = descr_from_type(NPY_VSTRING);
    d->numpy_generic = nb::borrow(&PyGenericArrType_Type);
    return d;
  });
}

}  // namespace jax
