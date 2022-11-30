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
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "jaxlib/cpu/ducc_fft_generated.h"
#include "jaxlib/cpu/ducc_fft_kernels.h"
#include "jaxlib/kernel_pybind11_helpers.h"

namespace py = pybind11;

namespace jax {
namespace {

py::bytes BuildDuccFftDescriptor(const std::vector<uint64_t> &shape,
                                 bool is_double, int fft_type,
                                 const std::vector<uint64_t> &fft_lengths,
                                 const std::vector<uint64_t> &strides_in,
                                 const std::vector<uint64_t> &strides_out,
                                 const std::vector<uint32_t> &axes,
                                 bool forward, double scale) {
  DuccFftDescriptorT descriptor;
  descriptor.shape = shape;
  descriptor.fft_type = static_cast<DuccFftType>(fft_type);
  descriptor.dtype =
      is_double ? DuccFftDtype_COMPLEX128 : DuccFftDtype_COMPLEX64;
  descriptor.strides_in = strides_in;
  descriptor.strides_out = strides_out;
  descriptor.axes = axes;
  descriptor.forward = forward;
  descriptor.scale = scale;
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(DuccFftDescriptor::Pack(fbb, &descriptor));
  return py::bytes(reinterpret_cast<char *>(fbb.GetBufferPointer()),
                   fbb.GetSize());
}

py::dict Registrations() {
  pybind11::dict dict;
  dict["ducc_fft"] = EncapsulateFunction(DuccFft);
  return dict;
}

PYBIND11_MODULE(_ducc_fft, m) {
  m.def("registrations", &Registrations);
  m.def("ducc_fft_descriptor", &BuildDuccFftDescriptor, py::arg("shape"),
        py::arg("is_double"), py::arg("fft_type"), py::arg("fft_lengths"),
        py::arg("strides_in"), py::arg("strides_out"), py::arg("axes"),
        py::arg("forward"), py::arg("scale"));
}

} // namespace
} // namespace jax
