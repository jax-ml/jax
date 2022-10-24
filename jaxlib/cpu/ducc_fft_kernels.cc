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

#include <complex>

#include "flatbuffers/flatbuffers.h"
#include "jaxlib/cpu/ducc_fft_generated.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "ducc/src/ducc0/fft/fft.h"

namespace jax {

using shape_t = ducc0::fmav_info::shape_t;
using stride_t = ducc0::fmav_info::stride_t;

void DuccFft(void *out, void **in, XlaCustomCallStatus *) {
  const DuccFftDescriptor *descriptor = GetDuccFftDescriptor(in[0]);
  shape_t shape(descriptor->shape()->begin(), descriptor->shape()->end());
  stride_t stride_in(descriptor->strides_in()->begin(),
                     descriptor->strides_in()->end());
  stride_t stride_out(descriptor->strides_out()->begin(),
                      descriptor->strides_out()->end());
  shape_t axes(descriptor->axes()->begin(), descriptor->axes()->end());

  switch (descriptor->fft_type()) {
  case DuccFftType_C2C:
    if (descriptor->dtype() == DuccFftDtype_COMPLEX64) {
      ducc0::cfmav<std::complex<float>> m_in(
          reinterpret_cast<std::complex<float> *>(in[1]), shape, stride_in);
      ducc0::vfmav<std::complex<float>> m_out(
          reinterpret_cast<std::complex<float> *>(out), shape, stride_out);
      ducc0::c2c(m_in, m_out, axes, descriptor->forward(),
                 static_cast<float>(descriptor->scale()));
    } else {
      ducc0::cfmav<std::complex<double>> m_in(
          reinterpret_cast<std::complex<double> *>(in[1]), shape, stride_in);
      ducc0::vfmav<std::complex<double>> m_out(
          reinterpret_cast<std::complex<double> *>(out), shape, stride_out);
      ducc0::c2c(m_in, m_out, axes, descriptor->forward(),
                 static_cast<double>(descriptor->scale()));
    }
    break;
  case DuccFftType_C2R:
    if (descriptor->dtype() == DuccFftDtype_COMPLEX64) {
      auto shape_in = shape;
      shape_in[axes.back()] = shape_in[axes.back()] / 2 + 1;
      ducc0::cfmav<std::complex<float>> m_in(
          reinterpret_cast<std::complex<float> *>(in[1]), shape_in, stride_in);
      ducc0::vfmav<float> m_out(reinterpret_cast<float *>(out), shape,
                                stride_out);
      ducc0::c2r(m_in, m_out, axes, descriptor->forward(),
                 static_cast<float>(descriptor->scale()));
    } else {
      auto shape_in = shape;
      shape_in[axes.back()] = shape_in[axes.back()] / 2 + 1;
      ducc0::cfmav<std::complex<double>> m_in(
          reinterpret_cast<std::complex<double> *>(in[1]), shape_in, stride_in);
      ducc0::vfmav<double> m_out(reinterpret_cast<double *>(out), shape,
                                 stride_out);
      ducc0::c2r(m_in, m_out, axes, descriptor->forward(),
                 static_cast<double>(descriptor->scale()));
    }
    break;
  case DuccFftType_R2C:
    if (descriptor->dtype() == DuccFftDtype_COMPLEX64) {
      auto shape_out = shape;
      shape_out[axes.back()] = shape_out[axes.back()] / 2 + 1;
      ducc0::cfmav<float> m_in(reinterpret_cast<float *>(in[1]), shape,
                               stride_in);
      ducc0::vfmav<std::complex<float>> m_out(
          reinterpret_cast<std::complex<float> *>(out), shape_out, stride_out);
      ducc0::r2c(m_in, m_out, axes, descriptor->forward(),
                 static_cast<float>(descriptor->scale()));
    } else {
      auto shape_out = shape;
      shape_out[axes.back()] = shape_out[axes.back()] / 2 + 1;
      ducc0::cfmav<double> m_in(reinterpret_cast<double *>(in[1]), shape,
                                stride_in);
      ducc0::vfmav<std::complex<double>> m_out(
          reinterpret_cast<std::complex<double> *>(out), shape_out, stride_out);
      ducc0::r2c(m_in, m_out, axes, descriptor->forward(),
                 static_cast<double>(descriptor->scale()));
    }
    break;
  }
}

} // namespace jax
