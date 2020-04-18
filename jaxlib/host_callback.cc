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
// host_callback_cpu_py and hostcallback_gpu_py, and also for unit testing.

#include "jaxlib/host_callback.h"

#include <cstdint>
#import <iostream>
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

size_t ParseInteger(msgpack::Reader &reader, const std::string &what) {
  msgpack::Object obj;
  if (!reader.read(obj) ||
      (obj.Kind != msgpack::Type::UInt && obj.Kind != msgpack::Type::Int)) {
    throw std::invalid_argument(
        absl::StrFormat("Cannot find integer %s", what));
  }
  size_t res = obj.Kind == msgpack::Type::Int ? obj.Int : obj.UInt;
  return res;
}

std::string ParseString(msgpack::Reader &reader, const std::string &what) {
  msgpack::Object obj;
  if (!reader.read(obj) || obj.Kind != msgpack::Type::String) {
    throw std::invalid_argument(absl::StrFormat("Cannot find string %s", what));
  }
  auto res = std::string(reinterpret_cast<const char *>(obj.Raw.data()),
                         obj.Raw.size());
  return res;
}

size_t ParseArrayLength(msgpack::Reader &reader, const std::string &what) {
  msgpack::Object obj;
  if (!reader.read(obj) || obj.Kind != msgpack::Type::Array) {
    throw std::invalid_argument(absl::StrFormat("Cannot find array %s", what));
  }
  return obj.Length;
}

// Converts a type descriptor and shape to TypeAndShape.
TypeAndShape ParseTypeDescriptor(msgpack::Reader &reader) {
  static auto *types =
      new absl::flat_hash_map<std::pair<char, int>, ElementType>({
          {{'f', 2}, ElementType::F16},
          {{'f', 4}, ElementType::F32},
          {{'f', 8}, ElementType::F64},
          {{'i', 1}, ElementType::I8},
          {{'i', 2}, ElementType::I16},
          {{'i', 4}, ElementType::I32},
          {{'i', 8}, ElementType::I64},
          {{'u', 1}, ElementType::U8},
          {{'u', 2}, ElementType::U16},
          {{'u', 4}, ElementType::U32},
          {{'u', 8}, ElementType::U64},
          {{'V', 2}, ElementType::BF16},
      });

  int tuple_size = ParseArrayLength(reader, "tuple");
  if (tuple_size != 2) {
    throw std::invalid_argument("Expected tuple of size 2");
  }
  std::string type_descriptor = ParseString(reader, "type_descriptor");
  int shape_length = ParseArrayLength(reader, "shape");
  std::vector<int> shape(shape_length);
  for (int j = 0; j < shape_length; ++j) {
    shape[j] = static_cast<int>(ParseInteger(reader, "shape_dim"));
  }

  size_t element_size;
  if (!absl::SimpleAtoi(type_descriptor.substr(1), &element_size)) {
    throw std::invalid_argument(absl::StrFormat(
        "Unsupported type descriptor %s (no size found)", type_descriptor));
  }

  auto it = types->find({type_descriptor.at(0), element_size});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported type descriptor %s", type_descriptor));
  }
  return TypeAndShape{it->second, element_size, shape};
}


// TODO(necula): add parameters for these
int constexpr kSideElements = 3;
int constexpr kSummarizeThreshold = 100;
int constexpr kPrecision = 2;  // Decimal digits


class Printer {
 public:
  Printer(std::ostringstream &output,
          const TypeAndShape &type_and_shape, const uint8_t *data)
      : output_{output},
        type_and_shape_(type_and_shape),
        current_ptr_{data},
        shape_{type_and_shape.shape},
        element_size_{type_and_shape.element_size} {
    ndims_ = type_and_shape.shape.size();
    current_index_.reserve(ndims_);
    skip_values_.reserve(ndims_);
    int current_skip = 1;
    for (int i = ndims_ - 1; i >= 0; --i) {
      current_index_[i] = 0;
      skip_values_[i] = current_skip;
      current_skip *= shape_[i];
    }
    total_size_ = type_and_shape.ByteSize();
  }

  void EmitArray();

 private:
  std::ostringstream &output_;
  TypeAndShape type_and_shape_;
  const uint8_t *current_ptr_;

  Shape shape_;  // TODO(add accessors for these?)

  size_t element_size_;
  int ndims_;
  size_t total_size_;

  // The current index to be emitted: [i0, i1, ..., in-1].
  Shape current_index_;
  // For each dimension, how many elements to skip to get
  // to the next value in the same dimension.
  Shape skip_values_;

  void EmitInnermostDimension();
  void EmitCurrentElement();
};

// Emits the element at current_ptr.
void Printer::EmitCurrentElement() {
  switch (type_and_shape_.element_type) {
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
  for (int idx = 0; idx < shape_[ndims_ - 1];
       ++idx, current_ptr_ += element_size_) {
    EmitCurrentElement();
    if (idx < shape_[ndims_ - 1] - 1) output_ << ' ';
    if (total_size_ > kSummarizeThreshold &&
        shape_[ndims_ - 1] > 2 * kSideElements && idx == kSideElements - 1) {
      int skip_indices = shape_[ndims_ - 1] - kSideElements - 1 - idx;
      current_ptr_ += element_size_ * skip_indices;
      idx += skip_indices;
      output_ << "... ";
    }
  }
  // Update the index to last one emitted.
  current_index_[ndims_ - 1] = shape_[ndims_ - 1] - 1;
  int count_stop_brackets = 0;
  // Emit as many ] as how many inner dimensions have reached the end
  for (int i = ndims_ - 1; i >= 0 && current_index_[i] == shape_[i] - 1; --i) {
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
    assert(current_index_[ndims_ - 1] == shape_[ndims_ - 1] - 1);

    // Advance to the next innermost dimension
    int dim_to_advance = ndims_ - 1;
    for (; dim_to_advance >= 0; --dim_to_advance) {
      ++current_index_[dim_to_advance];
      if (current_index_[dim_to_advance] >= shape_[dim_to_advance]) {
        current_index_[dim_to_advance] = 0;
        continue;
      } else {
        // Have not reached the end of the dim_to_advance.
        if (total_size_ > kSummarizeThreshold &&
            current_index_[dim_to_advance] == kSideElements &&
            shape_[dim_to_advance] > 2 * kSideElements) {
          int skip_indices = shape_[dim_to_advance] - kSideElements -
                             current_index_[dim_to_advance];
          current_ptr_ +=
              element_size_ * skip_values_[dim_to_advance] * skip_indices;
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

  meta.preamble = ParseString(mp_reader, "preamble");
  meta.separator = ParseString(mp_reader, "separator");
  int nr_args = ParseArrayLength(mp_reader, "args");
  for (int i = 0; i < nr_args; ++i) {
    meta.args_type_and_shape.push_back(ParseTypeDescriptor(mp_reader));
  }
  return meta;
}

void EmitOneArray(std::ostringstream &output, const PrintMetadata &meta,
                  int arg_idx, const void *data) {
  TypeAndShape arg_ts = meta.args_type_and_shape[arg_idx];
  Printer printer(output, arg_ts, static_cast<const uint8_t*>(data));
  output << (arg_idx == 0 ? meta.preamble : meta.separator) << "\n";
  output << absl::StreamFormat("arg[%d] ", arg_idx);
  output << " shape = (";
  for (const int &dim : arg_ts.shape) {
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
  for (int i = 0; i < meta.args_type_and_shape.size(); i++) {
    EmitOneArray(output, meta, i, args[kReservedArgs + i]);
  }
  std::cout << output.str();

  bool *resultPtr = static_cast<bool *>(out);
  *resultPtr = true;
}

}  // namespace jax
