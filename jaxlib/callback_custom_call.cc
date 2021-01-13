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

#include "jaxlib/callback_custom_call.h"

#include <cstdint>
#import <iostream>
#include <sstream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "flatbuffers/flatbuffers.h"
#include "jaxlib/callback_custom_call_generated.h"
#include "jaxlib/kernel_helpers.h"
#include "include/pybind11/detail/common.h"
#include "include/pybind11/pytypes.h"
#include "third_party/tensorflow/compiler/xla/layout_util.h"
#include "third_party/tensorflow/compiler/xla/literal.h"
#include "third_party/tensorflow/compiler/xla/python/types.h"
#include "third_party/tensorflow/compiler/xla/shape.h"
#include "third_party/tensorflow/compiler/xla/shape_util.h"
#include "third_party/tensorflow/compiler/xla/status_macros.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/platform/logging.h"

namespace jax {

namespace py = pybind11;

namespace callback_custom_call_fb {

flatbuffers::Offset<flatbuffers::String> EncodeShape(
    flatbuffers::FlatBufferBuilder &builder, xla::Shape shape) {
  return builder.CreateString(shape.SerializeAsString());
}

xla::Shape DecodeShape(const flatbuffers::String *shape_fb) {
  xla::ShapeProto shape_proto;
  shape_proto.ParseFromString(flatbuffers::GetString(shape_fb));
  xla::Shape shape(shape_proto);
  return shape;
}

py::bytes EncodeDescriptor(uint32_t callback_id,
                           std::vector<xla::Shape> operand_shapes,
                           bool ignore_results,
                           std::vector<xla::Shape> result_shapes) {
  flatbuffers::FlatBufferBuilder builder(1024);
  // Serialize in reverse order
  std::vector<flatbuffers::Offset<flatbuffers::String>> operand_offset_vector;
  operand_offset_vector.resize(operand_shapes.size());
  for (int i = operand_shapes.size() - 1; i >= 0; --i) {
    operand_offset_vector[i] = EncodeShape(builder, operand_shapes[i]);
  }
  auto operands_encoded = builder.CreateVector(operand_offset_vector);

  std::vector<flatbuffers::Offset<flatbuffers::String>> result_offset_vector;
  result_offset_vector.resize(result_shapes.size());
  for (int i = result_shapes.size() - 1; i >= 0; --i) {
    result_offset_vector[i] = EncodeShape(builder, result_shapes[i]);
  }
  auto results_encoded = builder.CreateVector(result_offset_vector);
  auto descriptor = CreateCallbackDescriptor(
      builder, callback_id, operands_encoded, ignore_results, results_encoded);
  builder.Finish(descriptor);

  uint8_t *buffer = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  return py::bytes(reinterpret_cast<char *>(buffer), size);
}

Descriptor DecodeDescriptor(const void *buffer) {
  const CallbackDescriptor *descr_fb = GetCallbackDescriptor(buffer);
  std::vector<xla::Shape> operands;
  for (auto op : *(descr_fb->operands())) {
    operands.push_back(DecodeShape(op));
  }
  std::vector<xla::Shape> results;
  for (auto res : *(descr_fb->results())) {
    results.push_back(DecodeShape(res));
  }
  return Descriptor{descr_fb->callback_id(), operands,
                    descr_fb->ignore_results(), results};
}

}  // namespace callback_custom_call_fb

// Store here the pointer to the Python function to call. The Python code should
// call SetCallbackTrampoline to set this to None before exiting.
static CallbackTrampolineToPython the_trampoline_ = nullptr;

void SetCallbackTrampoline(CallbackTrampolineToPython trampoline) {
  the_trampoline_ = trampoline;
}

std::vector<const void *> RunHostCallback(
    callback_custom_call_fb::Descriptor &descriptor,
    std::vector<const void *> arrays) {
  // descriptor.operands describes: token, arrays
  VLOG(2) << "Calling RunHostCallback with " << arrays.size() << " operands";
  py::gil_scoped_acquire gil_acquire;
  int nr_ops = descriptor.operands.size() - 1;
  CHECK_EQ(nr_ops, arrays.size());

  py::list ops_list;
  for (int i = 0; i < nr_ops; ++i) {
    xla::Shape arg_shape = descriptor.operands[1 + i];
    py::dtype dtype =
        xla::PrimitiveTypeToDtype(arg_shape.element_type()).ValueOrDie();
    VLOG(2) << "Preparing operand [" << i << "] of shape " << arg_shape.ToString();
    py::array arg_array(dtype, arg_shape.dimensions(),
                        xla::ByteStridesForShape(arg_shape),
                        const_cast<void *>(arrays[i]));
    ops_list.insert(i, arg_array);
  }
  py::object result = the_trampoline_(descriptor.callback_id, ops_list);

  int nr_results = descriptor.results.size() - 1;

  py::sequence result_sequence = py::cast<py::sequence>(result);
  if (! descriptor.ignore_results) {
    CHECK_EQ(nr_results, result_sequence.size());
  }
  std::vector<const void *> outputs;
  for (int i = 0; i < nr_results; ++i) {
    py::array result_array = result_sequence[i];
    outputs.push_back(result_array.data());
  }
  VLOG(2) << "Returning from RunHostCallback with " << outputs.size() << " results";
  return outputs;
}



}  // namespace jax
