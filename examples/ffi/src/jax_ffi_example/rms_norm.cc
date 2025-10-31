/* Copyright 2024 The JAX Authors.

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

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// This is the example "library function" that we want to expose to JAX. This
// isn't meant to be a particularly good implementation, it's just here as a
// placeholder for the purposes of this tutorial.
template <typename T>
T ComputeRmsNorm(float eps, int64_t size, const T *x, T *y) {
  T sm = static_cast<T>(0);
  for (int64_t n = 0; n < size; ++n) {
    sm += x[n] * x[n];
  }
  T scale = static_cast<T>(1) /
            std::sqrt(sm / static_cast<T>(size) + static_cast<T>(eps));
  for (int64_t n = 0; n < size; ++n) {
    y[n] = x[n] * scale;
  }
  return scale;
}

// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.
std::pair<int64_t, int64_t> GetDims(const ffi::AnyBuffer buffer) {
  const ffi::AnyBuffer::Dimensions dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

#define ELEMENT_TYPE_DISPATCH(element_type, fn, ...)                      \
  switch (element_type) {                                                 \
    case ffi::F32:                                                        \
      return fn<float>(__VA_ARGS__);                                      \
    case ffi::F64:                                                        \
      return fn<double>(__VA_ARGS__);                                     \
    case ffi::C64:                                                        \
      return fn<std::complex<float>>(__VA_ARGS__);                        \
    case ffi::C128:                                                       \
      return fn<std::complex<double>>(__VA_ARGS__);                       \
    default:                                                              \
      return ffi::Error::InvalidArgument("Unsupported input data type."); \
  }

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRmsNorm` above. This function handles the batch
// dimensions by calling `ComputeRmsNorm` within a loop.
template <typename T>
ffi::Error RmsNormImpl(int64_t totalSize, int64_t lastDim, float eps,
                       ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y) {
  T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeRmsNorm(eps, lastDim, x_data + n, y_data + n);
  }
  return ffi::Error::Success();
}

ffi::Error RmsNormDispatch(float eps, ffi::AnyBuffer x,
                           ffi::Result<ffi::AnyBuffer> y) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNorm input must be an array");
  }
  ELEMENT_TYPE_DISPATCH(x.element_type(), RmsNormImpl, totalSize, lastDim, eps,
                        x, y);
}

// Wrap `RmsNormImpl` and specify the interface to XLA. If you need to declare
// this handler in a header, you can use the `XLA_FFI_DECLARE_HANDLER_SYMBOL`
// macro: `XLA_FFI_DECLARE_HANDLER_SYMBOL(RmsNorm)`.
XLA_FFI_DEFINE_HANDLER_SYMBOL(RmsNorm, RmsNormDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<float>("eps")
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Ret<ffi::AnyBuffer>()  // y
);

template <typename T>
ffi::Error RmsNormFwdImpl(int64_t totalSize, int64_t lastDim, float eps,
                          ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y,
                          ffi::Result<ffi::AnyBuffer> res) {
  T *x_data = x.typed_data<T>();
  T *y_data = y->typed_data<T>();
  T *res_data = res->typed_data<T>();
  for (int64_t n = 0, idx = 0; n < totalSize; n += lastDim, ++idx) {
    res_data[idx] = ComputeRmsNorm(eps, lastDim, x_data + n, y_data + n);
  }
  return ffi::Error::Success();
}

ffi::Error RmsNormFwdDispatch(float eps, ffi::AnyBuffer x,
                              ffi::Result<ffi::AnyBuffer> y,
                              ffi::Result<ffi::AnyBuffer> res) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNormFwd input must be an array");
  }
  ELEMENT_TYPE_DISPATCH(x.element_type(), RmsNormFwdImpl, totalSize, lastDim,
                        eps, x, y, res);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RmsNormFwd, RmsNormFwdDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<float>("eps")
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Ret<ffi::AnyBuffer>()  // y
                                  .Ret<ffi::AnyBuffer>()  // res
);

template <typename T>
void ComputeRmsNormBwd(int64_t size, T res, const T *x, const T *ct_y,
                       T *ct_x) {
  T ct_res = static_cast<T>(0);
  for (int64_t n = 0; n < size; ++n) {
    ct_res += x[n] * ct_y[n];
  }
  T factor = ct_res * res * res * res / static_cast<T>(size);
  for (int64_t n = 0; n < size; ++n) {
    ct_x[n] = res * ct_y[n] - factor * x[n];
  }
}

template <typename T>
ffi::Error RmsNormBwdImpl(int64_t totalSize, int64_t lastDim,
                          ffi::AnyBuffer res, ffi::AnyBuffer x,
                          ffi::AnyBuffer ct_y,
                          ffi::Result<ffi::AnyBuffer> ct_x) {
  T *res_data = res.typed_data<T>();
  T *x_data = x.typed_data<T>();
  T *ct_y_data = ct_y.typed_data<T>();
  T *ct_x_data = ct_x->typed_data<T>();
  for (int64_t n = 0, idx = 0; n < totalSize; n += lastDim, ++idx) {
    ComputeRmsNormBwd(lastDim, res_data[idx], x_data + n, ct_y_data + n,
                      ct_x_data + n);
  }
  return ffi::Error::Success();
}

ffi::Error RmsNormBwdDispatch(ffi::AnyBuffer res, ffi::AnyBuffer x,
                              ffi::AnyBuffer ct_y,
                              ffi::Result<ffi::AnyBuffer> ct_x) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNormBwd inputs must be arrays");
  }
  ELEMENT_TYPE_DISPATCH(x.element_type(), RmsNormBwdImpl, totalSize, lastDim,
                        res, x, ct_y, ct_x);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RmsNormBwd, RmsNormBwdDispatch,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()  // res
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Arg<ffi::AnyBuffer>()  // ct_y
                                  .Ret<ffi::AnyBuffer>()  // ct_x
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_rms_norm, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["rms_norm"] = EncapsulateFfiHandler(RmsNorm);
    registrations["rms_norm_fwd"] = EncapsulateFfiHandler(RmsNormFwd);
    registrations["rms_norm_bwd"] = EncapsulateFfiHandler(RmsNormBwd);
    return registrations;
  });
}
