/* Copyright 2021 Google LLC

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

// Implementation of host callback using CustomCall
// See callback_custom_call.py module documentation for design comments.

#ifndef CALLBACK_CUSTOM_CALL_H_
#define CALLBACK_CUSTOM_CALL_H_

#include <cstddef>
#include <vector>

#include "jaxlib/kernel_helpers.h"
#include "jaxlib/callback_custom_call_generated.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/functional.h"
#include "third_party/tensorflow/compiler/xla/shape.h"
#include "third_party/tensorflow/core/platform/logging.h"

namespace jax {

namespace callback_custom_call_fb {

// A Descriptor communicates metadata to the CustomCall. A Descriptor is
// encoded as a byte array and is passed as one of the arguments of the
// CustomCall.
struct Descriptor {
  uint32_t callback_id;
  std::vector<xla::Shape> operands;
  bool ignore_results;
  std::vector<xla::Shape> results;
};


pybind11::bytes EncodeDescriptor(uint32_t callback_id,
                                 std::vector<xla::Shape> operand_shapes,
                                 bool ignore_results,
                                 std::vector<xla::Shape> result_shapes);

Descriptor DecodeDescriptor(const void*);

}  // namespace callback_custom_call_fb

// Invokes the Python trampoline for the callback described by the
// descriptor, passing the given arrays as arguments.
//
// Args:
//  descriptor: describes the callback metadata.
//  arrays: the pointers to the data (on the host) to be passed to the callback
//
// Returns:
//  an array of pointers to the result arrays (on the host) returned by the
//   callback.
std::vector<const void*> RunHostCallback(
    callback_custom_call_fb::Descriptor &descriptor,
    std::vector<const void*> arrays);

// A callback trampoline to Python takes: callback id, one literal argument.
using CallbackTrampolineToPython =
      std::function<pybind11::object(uint32_t, pybind11::object)>;

// The Python code uses this to set the pointer to the trampoline.
void SetCallbackTrampoline(CallbackTrampolineToPython trampoline);


}  // namespace jax

#endif  // CALLBACK_CUSTOM_CALL_H_
