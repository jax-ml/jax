/* Copyright 2020 Google LLC

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
#include "pocketfft/pocketfft_hdronly.h"
#include "jaxlib/pocketfft_generated.h"

namespace jax {

void PocketFft(void* out, void** in) {
  const PocketFftDescriptor* descriptor = GetPocketFftDescriptor(in[0]);
  pocketfft::shape_t shape(descriptor->shape()->begin(),
                           descriptor->shape()->end());
  pocketfft::stride_t stride_in(descriptor->strides_in()->begin(),
                                descriptor->strides_in()->end());
  pocketfft::stride_t stride_out(descriptor->strides_out()->begin(),
                                 descriptor->strides_out()->end());
  pocketfft::shape_t axes(descriptor->axes()->begin(),
                          descriptor->axes()->end());

  switch (descriptor->fft_type()) {
    case PocketFftType_C2C:
      if (descriptor->dtype() == PocketFftDtype_COMPLEX64) {
        std::complex<float>* a_in =
            reinterpret_cast<std::complex<float>*>(in[1]);
        std::complex<float>* a_out =
            reinterpret_cast<std::complex<float>*>(out);
        pocketfft::c2c(shape, stride_in, stride_out, axes,
                       descriptor->forward(), a_in, a_out,
                       static_cast<float>(descriptor->scale()));
      } else {
        std::complex<double>* a_in =
            reinterpret_cast<std::complex<double>*>(in[1]);
        std::complex<double>* a_out =
            reinterpret_cast<std::complex<double>*>(out);
        pocketfft::c2c(shape, stride_in, stride_out, axes,
                       descriptor->forward(), a_in, a_out, descriptor->scale());
      }
      break;
    case PocketFftType_C2R:
      if (descriptor->dtype() == PocketFftDtype_COMPLEX64) {
        std::complex<float>* a_in =
            reinterpret_cast<std::complex<float>*>(in[1]);
        float* a_out = reinterpret_cast<float*>(out);
        pocketfft::c2r(shape, stride_in, stride_out, axes,
                       descriptor->forward(), a_in, a_out,
                       static_cast<float>(descriptor->scale()));
      } else {
        std::complex<double>* a_in =
            reinterpret_cast<std::complex<double>*>(in[1]);
        double* a_out = reinterpret_cast<double*>(out);
        pocketfft::c2r(shape, stride_in, stride_out, axes,
                       descriptor->forward(), a_in, a_out, descriptor->scale());
      }
      break;
    case PocketFftType_R2C:
      if (descriptor->dtype() == PocketFftDtype_COMPLEX64) {
        float* a_in = reinterpret_cast<float*>(in[1]);
        std::complex<float>* a_out =
            reinterpret_cast<std::complex<float>*>(out);
        pocketfft::r2c(shape, stride_in, stride_out, axes,
                       descriptor->forward(), a_in, a_out,
                       static_cast<float>(descriptor->scale()));
      } else {
        double* a_in = reinterpret_cast<double*>(in[1]);
        std::complex<double>* a_out =
            reinterpret_cast<std::complex<double>*>(out);
        pocketfft::r2c(shape, stride_in, stride_out, axes,
                       descriptor->forward(), a_in, a_out, descriptor->scale());
      }
      break;
  }
}

}  // namespace jax
