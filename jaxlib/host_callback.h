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
#ifndef JAXLIB_HOST_CALLBACK_H_
#define JAXLIB_HOST_CALLBACK_H_

#include <vector>

#include "jaxlib/kernel_helpers.h"

namespace jax {

// Metadata for id_print runtime functions.
typedef std::vector<int> Shape;
enum ElementType {
  I8, I16, I32, I64,
  U8, U16, U32, U64,
  F16, F32, F64,
  BF16
};

struct TypeAndShape {
  ElementType element_type;
  size_t element_size;
  Shape shape;

  // The size of the array, in bytes.
  size_t ByteSize() const {
    size_t result = element_size;
    for (int const &i : shape) {
      result *= i;
    }
    return result;
  }
};
struct PrintMetadata {
  // The preamble to be printed before the arguments.
  std::string preamble;
  // The separator to be printed between the arguments.
  std::string separator;
  // Types and shapes for the arguments to be printed.
  std::vector<TypeAndShape> args_type_and_shape;

  // The maximum size in bytes of all the arrays.
  size_t MaximumByteSize() const {
    size_t res = 0;
    for (TypeAndShape const &ts : args_type_and_shape) {
      res = std::max(res, ts.ByteSize());
    }
    return res;
  }
};

// Retrieves the current implemented print metadata version.
int GetPrintMetadataVersion();

// Parses PrintMetadata msgpack-encoded by Python.
// The metadata has the following format:
//     (preamble: str,    # to be printed before the first argument
//      separator: str,   # to be printed between arguments
//      [ (type_descriptor: str,
//         shape: Tuple[int, ...]) ]
//
PrintMetadata ParsePrintMetadata(std::string bytes);

// Emits one array to the output stream.
// If this is the first argument, emits the preamble first, otherwise
// emits the separator. Then emits a newline.
// Before each arguments emits "arg[idx]  shape=(...)\n".
// Arguments:
//   output: the output stream.
//   meta: the PrintMetadata giving parameters for the printing
//   arg_idx: the argument index in meta
//   data: the starting address of the data.
void EmitOneArray(std::ostringstream &output,
                  const PrintMetadata &meta,
                  int arg_idx,
                  const void *data);

// Prints the arguments and returns True. This is the CustomCall for CPU.
void PrintCPU(void* out, const void** args);

}  // namespace jax

#endif  // JAXLIB_HOST_CALLBACK_H_
