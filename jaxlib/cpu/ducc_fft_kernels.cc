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

#include "ducc/src/ducc0/fft/fft.h"
#include "ducc/src/ducc0/fft/fft1d_impl.h"  // NOLINT: required for fft definitions.
#include "ducc/src/ducc0/fft/fftnd_impl.h"  // NOLINT: required for fft definitions.
#include "flatbuffers/flatbuffers.h"
#include "jaxlib/cpu/ducc_fft_generated.h"
#include "xla/service/custom_call_status.h"

namespace jax {

using shape_t = ducc0::fmav_info::shape_t;
using stride_t = ducc0::fmav_info::stride_t;

namespace  {

void DuccFftImpl(void *out, void *operand, jax::DuccFftDtype dtype,
    jax::DuccFftType fft_type,
    shape_t shape, stride_t strides_in, stride_t strides_out, shape_t axes,
    bool forward, double scale) {

  switch (fft_type) {
  case DuccFftType_C2C:
    if (dtype == DuccFftDtype_COMPLEX64) {
      ducc0::cfmav<std::complex<float>> m_in(
          reinterpret_cast<std::complex<float> *>(operand), shape, strides_in);
      ducc0::vfmav<std::complex<float>> m_out(
          reinterpret_cast<std::complex<float> *>(out), shape, strides_out);
      ducc0::c2c(m_in, m_out, axes, forward, static_cast<float>(scale));
    } else {
      ducc0::cfmav<std::complex<double>> m_in(
          reinterpret_cast<std::complex<double> *>(operand), shape, strides_in);
      ducc0::vfmav<std::complex<double>> m_out(
          reinterpret_cast<std::complex<double> *>(out), shape, strides_out);
      ducc0::c2c(m_in, m_out, axes, forward, scale);
    }
    break;
  case DuccFftType_C2R:
    if (dtype == DuccFftDtype_COMPLEX64) {
      auto shape_in = shape;
      shape_in[axes.back()] = shape_in[axes.back()] / 2 + 1;
      ducc0::cfmav<std::complex<float>> m_in(
          reinterpret_cast<std::complex<float> *>(operand),
          shape_in, strides_in);
      ducc0::vfmav<float> m_out(reinterpret_cast<float *>(out), shape,
                                strides_out);
      ducc0::c2r(m_in, m_out, axes, forward, static_cast<float>(scale));
    } else {
      auto shape_in = shape;
      shape_in[axes.back()] = shape_in[axes.back()] / 2 + 1;
      ducc0::cfmav<std::complex<double>> m_in(
          reinterpret_cast<std::complex<double> *>(operand),
          shape_in, strides_in);
      ducc0::vfmav<double> m_out(reinterpret_cast<double *>(out), shape,
                                 strides_out);
      ducc0::c2r(m_in, m_out, axes, forward, scale);
    }
    break;
  case DuccFftType_R2C:
    if (dtype == DuccFftDtype_COMPLEX64) {
      auto shape_out = shape;
      shape_out[axes.back()] = shape_out[axes.back()] / 2 + 1;
      ducc0::cfmav<float> m_in(reinterpret_cast<float *>(operand), shape,
                               strides_in);
      ducc0::vfmav<std::complex<float>> m_out(
          reinterpret_cast<std::complex<float> *>(out),
          shape_out, strides_out);
      ducc0::r2c(m_in, m_out, axes, forward, static_cast<float>(scale));
    } else {
      auto shape_out = shape;
      shape_out[axes.back()] = shape_out[axes.back()] / 2 + 1;
      ducc0::cfmav<double> m_in(reinterpret_cast<double *>(operand), shape,
                                strides_in);
      ducc0::vfmav<std::complex<double>> m_out(
          reinterpret_cast<std::complex<double> *>(out),
          shape_out, strides_out);
      ducc0::r2c(m_in, m_out, axes, forward, scale);
    }
    break;
  }
}

}  // namespace


// TODO(b/287702203): this must be kept until EOY 2023 for backwards
// of serialized functions using fft.
void DuccFft(void *out, void **in, XlaCustomCallStatus *) {
  const DuccFftDescriptor *descriptor = GetDuccFftDescriptor(in[0]);
  shape_t shape(descriptor->shape()->begin(), descriptor->shape()->end());
  stride_t strides_in(descriptor->strides_in()->begin(),
                      descriptor->strides_in()->end());
  stride_t strides_out(descriptor->strides_out()->begin(),
                       descriptor->strides_out()->end());
  shape_t axes(descriptor->axes()->begin(), descriptor->axes()->end());

  DuccFftImpl(out, in[1], descriptor->dtype(), descriptor->fft_type(),
              shape, strides_in, strides_out, axes,
              descriptor->forward(), descriptor->scale());
}


void DynamicDuccFft(void *out, void **in, XlaCustomCallStatus *) {
  // in[0]=descriptor, in[1]=operand,
  // in[2]=shape, in[3]=strides_in, in[4]=strides_out, in[5]=scale.
  const DynamicDuccFftDescriptor *descriptor =
      flatbuffers::GetRoot<DynamicDuccFftDescriptor>(in[0]);
  const std::uint32_t *dynamic_shape =
      reinterpret_cast<const std::uint32_t*>(in[2]);
  shape_t shape(dynamic_shape, dynamic_shape + descriptor->ndims());
  const std::uint32_t *dynamic_strides_in =
      reinterpret_cast<const std::uint32_t*>(in[3]);
  stride_t strides_in(dynamic_strides_in,
      dynamic_strides_in + descriptor->ndims());
  const std::uint32_t *dynamic_strides_out =
      reinterpret_cast<const std::uint32_t*>(in[4]);
  stride_t strides_out(dynamic_strides_out,
      dynamic_strides_out + descriptor->ndims());
  shape_t axes(descriptor->axes()->begin(), descriptor->axes()->end());
  const double *dynamic_scale = reinterpret_cast<const double*>(in[5]);

  DuccFftImpl(out, in[1], descriptor->dtype(), descriptor->fft_type(),
              shape, strides_in, strides_out, axes,
              descriptor->forward(), *dynamic_scale);
}

}  // namespace jax
