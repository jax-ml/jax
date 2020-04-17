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
};
struct PrintMetadata {
  // The preamble to be printed before the arguments.
  std::string preamble;
  // The separator to be printed between the arguments.
  std::string separator;
  // Types and shapes for the arguments to be printed.
  std::vector<TypeAndShape> args_type_and_shape;
};

// Retrieves the current implemented print metadata version.
int GetPrintMetadataVersion();

// Emits one array to the output stream.
// Arguments:
//   output: the output stream.
//   meta: the PrintMetadata giving parameters for the printing
//   arg_idx: the argument index in meta
//   data: the starting address of the data.
void EmitOneArray(std::ostringstream &output,
                  const PrintMetadata &meta,
                  int arg_idx,
                  const uint8_t *data);

// Emits multiple arrays to the output stream.
// Arguments:
//   output: the output stream.
//   meta: the PrintMetadata giving parameters for the printing
//   data: the vector with the starting addresses of the data.
void EmitArrays(std::ostringstream &output,
                const PrintMetadata &meta,
                const std::vector<const uint8_t *> &arrays);

// Prints the arguments and returns True.
void PrintCPU(void* out, const void** args);

}  // namespace jax

#endif  // JAXLIB_HOST_CALLBACK_H_
