/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_GPU_DUMP_H_
#define JAXLIB_MOSAIC_GPU_DUMP_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"

namespace mosaic {
namespace gpu {

struct DumpOptions {
  // Whether to dump the MLIR module before and after each pass.
  bool mlir_passes = false;
  // Whether to dump the PTX resulting from the compilation.
  bool ptx = false;
  // Whether to run ptxas in verbose mode.
  bool ptxas = false;
  // Whether to dump the SASS resulting from the compilation. If both `sass`
  // and `sass_ctrl` are true, a single dump containing both will be
  // generated.
  bool sass = false;
  // Whether to dump the SASS control codes following NervanaSystems/maxas. If
  // both `sass` and `sass_ctrl` are true, a single dump containing both will be
  // generated.
  bool sass_ctrl = false;
  // Where to dump the output files. If empty, dump to stdout.
  std::string dump_path = "";
  // The basename to use when dumping files.
  std::string module_basename;
};

// Extracts the dump options for the given module from environment variables.
//
// This function takes in a module in order to ensure that subsequent
// compilations of modules that share the same name will result in distinct
// dumps.
DumpOptions GetDumpOptionsForModule(mlir::ModuleOp module);

// Dumps `content` to `path`/`name` if `path` is non-empty, otherwise to
// stdout.
void DumpToFileOrStdout(absl::string_view content, absl::string_view name,
                        absl::string_view path);

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_DUMP_H_
