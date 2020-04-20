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

// A library of functions needed for the CPU and GPU implementation of
// host_callback_cpu_py and host_callback_cuda_py, and also for unit testing.

#include "jaxlib/host_callback.h"

#include <cstdint>
#import <iostream>
#include <vector>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "jaxlib/kernel_helpers.h"
#include "llvm/BinaryFormat/MsgPackReader.h"

namespace jax {

namespace {

namespace msgpack = llvm::msgpack;

// The Python code will query the version of the metadata, and
// it can abort or encode specially for backwards compatibility.
// Bump this version always when you change the metadata format,
// even if it is backwards compatible.
// As long as we use msgpack, the client can add new objects at
// the end while being compatible with the library.
int constexpr kPrintMetadataVersion = 1;

// Template functions for parsing from descriptors.
// "what" is used for error messages.
template <class T>
T ParseFromDescriptor(msgpack::Reader &reader, const std::string &what);

// Parses size_t.
template <>
size_t ParseFromDescriptor<size_t>(msgpack::Reader &reader,
                                   const std::string &what) {
  msgpack::Object obj;
  if (!reader.read(obj) ||
      (obj.Kind != msgpack::Type::UInt && obj.Kind != msgpack::Type::Int)) {
    throw std::invalid_argument(
        absl::StrFormat("Cannot find integer %s", what));
  }
  size_t res = obj.Kind == msgpack::Type::Int ? obj.Int : obj.UInt;
  return res;
}

// Parses int.
template <>
int ParseFromDescriptor<int>(msgpack::Reader &reader,
                                const std::string &what) {
  return static_cast<int>(ParseFromDescriptor<size_t>(reader, what));
}

// Parses std::string.
template <>
std::string ParseFromDescriptor(msgpack::Reader &reader,
                                const std::string &what) {
  msgpack::Object obj;
  if (!reader.read(obj) || obj.Kind != msgpack::Type::String) {
    throw std::invalid_argument(absl::StrFormat("Cannot find string %s", what));
  }
  auto res = std::string(reinterpret_cast<const char *>(obj.Raw.data()),
                         obj.Raw.size());
  return res;
}

// Parses the start of a vector.
size_t ParseArrayLength(msgpack::Reader &reader, const std::string &what) {
  msgpack::Object obj;
  if (!reader.read(obj) || obj.Kind != msgpack::Type::Array) {
    throw std::invalid_argument(absl::StrFormat("Cannot find array %s", what));
  }
  return obj.Length;
}


// Parses a std::vector<T>
template <class T>
std::vector<T> ParseVectorFromDescriptor(msgpack::Reader &reader,
                                         const std::string &what) {
  int vector_size = ParseArrayLength(reader, what);
  std::vector<T> res(vector_size);
  for (int j = 0; j < vector_size; ++j) {
    res[j] = ParseFromDescriptor<T>(reader, what);
  }
  return res;
}

// Parses a Shape.
template <>
Shape ParseFromDescriptor(msgpack::Reader &reader, const std::string &what) {
  int tuple_size = ParseArrayLength(reader, "tuple");
  if (tuple_size != 2) {
    throw std::invalid_argument("Expected tuple of size 2");
  }
  std::string type_descriptor = ParseFromDescriptor<std::string>(
      reader, "type_descriptor");
  std::vector<int> dimensions = ParseVectorFromDescriptor<int>(reader,
                                                              "dimensions");
  static auto *types =
      new absl::flat_hash_map<std::string,
                              std::tuple<ElementType, size_t>>({
          {"f2", {ElementType::F16, 2}},
          {"f4", {ElementType::F32, 4}},
          {"f8", {ElementType::F64, 8}},
          {"i1", {ElementType::I8, 1}},
          {"i2", {ElementType::I16, 2}},
          {"i4", {ElementType::I32, 4}},
          {"i8", {ElementType::I8, 8}},
          {"u1", {ElementType::U8, 1}},
          {"u2", {ElementType::U16, 2}},
          {"u4", {ElementType::U32, 4}},
          {"u8", {ElementType::U64, 8}},
          {"V2", {ElementType::BF16, 2}},
      });
  auto it = types->find(type_descriptor);
  if (it == types->end()) {
    throw std::invalid_argument(absl::StrFormat(
        "Unsupported type descriptor %s", type_descriptor));
  }
  return Shape{std::get<0>(it->second),
               std::get<1>(it->second), dimensions};
}


// TODO(necula): add parameters for these
int constexpr kSideElements = 3;
int constexpr kSummarizeThreshold = 100;
int constexpr kPrecision = 2;  // Decimal digits


class Printer {
 public:
  Printer(std::ostringstream &output, const Shape &shape, const uint8_t *data)
      : output_{output},
        shape_(shape),
        current_ptr_{data} {
    ndims_ = shape_.dimensions.size();
    current_index_.reserve(ndims_);
    skip_values_.reserve(ndims_);
    int current_skip = 1;
    for (int i = ndims_ - 1; i >= 0; --i) {
      current_index_[i] = 0;
      skip_values_[i] = current_skip;
      current_skip *= shape_.dimensions[i];
    }
    total_size_ = shape_.ByteSize();
    element_size_ = shape_.element_size;
  }

  void EmitArray();

 private:
  std::ostringstream &output_;
  Shape shape_;
  const uint8_t *current_ptr_;

  size_t element_size_;
  int ndims_;
  size_t total_size_;

  // The current index to be emitted: [i0, i1, ..., in-1].
  Dimensions current_index_;
  // For each dimension, how many elements to skip to get
  // to the next value in the same dimension.
  Dimensions skip_values_;

  void EmitInnermostDimension();
  void EmitCurrentElement();
};

// Emits the element at current_ptr.
void Printer::EmitCurrentElement() {
  switch (shape_.element_type) {
    case I8:
      output_ << *reinterpret_cast<const int8_t *>(current_ptr_);
      break;
    case I16:
      output_ << *reinterpret_cast<const int16_t *>(current_ptr_);
      break;
    case I32:
      output_ << *reinterpret_cast<const int32_t *>(current_ptr_);
      break;
    case I64:
      output_ << *reinterpret_cast<const int64_t *>(current_ptr_);
      break;
    case U8:
      output_ << *reinterpret_cast<const uint8_t *>(current_ptr_);
      break;
    case U16:
      output_ << *reinterpret_cast<const uint16_t *>(current_ptr_);
      break;
    case U32:
      output_ << *reinterpret_cast<const uint32_t *>(current_ptr_);
      break;
    case U64:
      output_ << *reinterpret_cast<const uint64_t *>(current_ptr_);
      break;
    case F16:
    case BF16:  // TODO(float16)
      output_ << *reinterpret_cast<const uint16_t *>(current_ptr_);
      break;
    case F32:
      output_ << *reinterpret_cast<const float *>(current_ptr_);
      break;
    case F64:
      output_ << *reinterpret_cast<const double *>(current_ptr_);
      break;
  }
}

// Emits spaces and [, then the elements in the current
// innermost dimension, then ].
// Assumes current_index[ndims - 1] = 0, current_ptr points to first
// element in the dimension to be printed.
void Printer::EmitInnermostDimension() {
  // Emit ndim spaces and [, as many [ as trailing 0s in current_index.
  assert(current_index_[ndims_ - 1] == 0);
  int count_start_spaces = ndims_ - 1;
  while (count_start_spaces >= 1 &&
         current_index_[count_start_spaces - 1] == 0) {
    --count_start_spaces;
  }
  for (int i = 0; i < ndims_; ++i) {
    output_ << (i < count_start_spaces ? ' ' : '[');
  }

  // Now emit the elements
  const size_t innermost_size = shape_.dimensions[ndims_ - 1];
  const size_t element_size = shape_.element_size;
  for (int idx = 0; idx < innermost_size; ++idx, current_ptr_ += element_size) {
    EmitCurrentElement();
    if (idx < innermost_size - 1) output_ << ' ';
    if (total_size_ > kSummarizeThreshold &&
        innermost_size > 2 * kSideElements && idx == kSideElements - 1) {
      int skip_indices = innermost_size - kSideElements - 1 - idx;
      current_ptr_ += element_size * skip_indices;
      idx += skip_indices;
      output_ << "... ";
    }
  }
  // Update the index to last one emitted.
  current_index_[ndims_ - 1] = innermost_size - 1;
  int count_stop_brackets = 0;
  // Emit as many ] as how many inner dimensions have reached the end
  for (int i = ndims_ - 1;
      i >= 0 && current_index_[i] == shape_.dimensions[i] - 1; --i) {
    ++count_stop_brackets;
    output_ << ']';
  }
  if (count_stop_brackets > 1) {
    output_ << std::endl;
  }
  output_ << std::endl;
}

// Emits a string representation of the array to output,
// in the style of numpy.array2string.
void Printer::EmitArray() {
  output_.precision(kPrecision);
  output_.setf(std::ios::fixed);
  if (ndims_ == 0) {
    EmitCurrentElement();
    return;
  }
  while (true) {
    EmitInnermostDimension();
    assert(current_index_[ndims_ - 1] == shape_.dimensions[ndims_ - 1] - 1);

    // Advance to the next innermost dimension
    int dim_to_advance = ndims_ - 1;
    for (; dim_to_advance >= 0; --dim_to_advance) {
      ++current_index_[dim_to_advance];
      assert (current_index_[dim_to_advance] <=
              shape_.dimensions[dim_to_advance]);
      if (current_index_[dim_to_advance] == shape_.dimensions[dim_to_advance]) {
        current_index_[dim_to_advance] = 0;
        continue;
      } else {
        // Have not reached the end of the dim_to_advance.
        if (total_size_ > kSummarizeThreshold &&
            current_index_[dim_to_advance] == kSideElements &&
            shape_.dimensions[dim_to_advance] > 2 * kSideElements) {
          int skip_indices = shape_.dimensions[dim_to_advance] - kSideElements -
                             current_index_[dim_to_advance];
          current_ptr_ +=
              shape_.element_size * skip_values_[dim_to_advance] * skip_indices;
          current_index_[dim_to_advance] += skip_indices;
          for (int j = 0; j <= dim_to_advance; ++j) {
            output_ << ' ';
          }
          output_ << "..." << std::endl;
        }
        break;
      }
    }
    if (dim_to_advance < 0) {
      return;
    }
  }
}

}  // namespace

int GetPrintMetadataVersion() { return kPrintMetadataVersion; }

PrintMetadata ParsePrintMetadata(std::string bytes) {
  PrintMetadata meta;

  const char *buffer = bytes.data();
  const size_t len = bytes.size();
  llvm::StringRef buffer_s(buffer, len);
  msgpack::Reader mp_reader(buffer_s);

  meta.arg_shapes = ParseVectorFromDescriptor<Shape>(mp_reader, "arg_shapes");
  meta.preamble = ParseFromDescriptor<std::string>(mp_reader, "preamble");
  meta.separator = ParseFromDescriptor<std::string>(mp_reader, "separator");
  return meta;
}

void EmitOneArray(std::ostringstream &output, const PrintMetadata &meta,
                  int arg_idx, const void *data) {
  Shape arg_shape = meta.arg_shapes[arg_idx];
  Printer printer(output, arg_shape, static_cast<const uint8_t*>(data));
  output << (arg_idx == 0 ? meta.preamble : meta.separator) << "\n";
  output << absl::StreamFormat("arg[%d] ", arg_idx);
  output << " shape = (";
  for (const int &dim : arg_shape.dimensions) {
    output << dim << ", ";
  }
  output << ")\n";
  printer.EmitArray();
}


void PrintCPU(void *out, const void **args) {
  static constexpr int kReservedArgs = 2;
  const int *opaque_len = static_cast<const int *>(args[0]);
  const char *opaque = static_cast<const char *>(args[1]);
  const PrintMetadata &meta =
      ParsePrintMetadata(std::string(opaque, *opaque_len));

  std::ostringstream output;
  for (int i = 0; i < meta.arg_shapes.size(); i++) {
    EmitOneArray(output, meta, i, args[kReservedArgs + i]);
  }
  std::cout << output.str();

  bool *resultPtr = static_cast<bool *>(out);
  *resultPtr = true;
}

}  // namespace jax
