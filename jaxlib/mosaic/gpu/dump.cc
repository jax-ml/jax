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

#include "jaxlib/mosaic/gpu/dump.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <system_error>  // NOLINT

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "tsl/platform/path.h"

namespace mosaic {
namespace gpu {

namespace {

// The name of the attribute wrapping the module basename for dumping. See
// `GetDumpOptionsForModule` for more details.
constexpr absl::string_view kDumpBasenameAttr = "mosaic_gpu.dump_basename";

}  // namespace

DumpOptions GetOrSetDumpOptionsForModule(mlir::ModuleOp module) {
  // Use a static variable in order to ensure that subsequent compilations of
  // modules that share the same name will result in distinct dumps.
  static std::atomic<int> dumped_module_count = 0;
  DumpOptions opts;
  // In order to make sure that we use a consistent module basename for the same
  // module even if we end up calling this function multiple times, we set an
  // attribute on the module that records its basename whenever we first
  // generate it. Subsequent calls will just return the value from the
  // attribute.
  if (auto attr = module->getAttrOfType<mlir::StringAttr>(kDumpBasenameAttr)) {
    opts.module_basename = attr.getValue().str();
  } else {
    int current_count = dumped_module_count.fetch_add(1);
    if (std::optional<llvm::StringRef> name = module.getName();
        name.has_value()) {
      opts.module_basename = absl::StrCat(name->str(), "_", current_count);
    } else {
      opts.module_basename = absl::StrCat("mosaic_gpu_module_", current_count);
    }
    module->setAttr(
        kDumpBasenameAttr,
        mlir::StringAttr::get(module.getContext(), opts.module_basename));
  }

  if (char* dump_to = getenv("MOSAIC_GPU_DUMP_TO"); dump_to != nullptr) {
    // "sponge" is a special value, which if set, will result in the files being
    // dumped to a directory path specified in the `TEST_UNDECLARED_OUTPUTS_DIR`
    // environment variable.
    if (absl::string_view(dump_to) == "sponge") {
      if (char* dump_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
          dump_dir != nullptr) {
        opts.dump_path = dump_dir;
      } else {
        LOG(WARNING) << "\"sponge\" specified as dump directory but "
                        "TEST_UNDECLARED_OUTPUTS_DIR is not set! "
                        "Will dump to stdout instead.";
      }
    } else if (absl::string_view(dump_to) == "-") {
      // Dump to stdout.
      opts.dump_path = "";
    } else {
      opts.dump_path = dump_to;
    }

    opts.mlir_passes = true;
    opts.ptx = true;
    opts.ptxas = true;
    opts.sass = true;
    opts.sass_ctrl = true;
    return opts;
  }

  opts.mlir_passes = getenv("MOSAIC_GPU_DUMP_MLIR_PASSES") != nullptr;
  opts.ptx = getenv("MOSAIC_GPU_DUMP_PTX") != nullptr;
  opts.ptxas = getenv("MOSAIC_GPU_DUMP_PTXAS") != nullptr;
  opts.sass_ctrl = getenv("MOSAIC_GPU_DUMP_SASS_CTRL") != nullptr;
  opts.sass = getenv("MOSAIC_GPU_DUMP_SASS") != nullptr || opts.sass_ctrl;
  return opts;
}

void DumpToFileOrStdout(absl::string_view content, absl::string_view name,
                        absl::string_view path) {
  if (path.empty()) {
    std::cout << content << std::endl;
    return;
  }
  std::error_code error;
  llvm::raw_fd_ostream out_file(tsl::io::JoinPath(path, name), error,
                                llvm::sys::fs::OF_None);
  if (error) {
    LOG(ERROR) << error.message();
    LOG(ERROR) << "Output will be written to stdout instead.";
    std::cout << content << std::endl;
    return;
  }
  out_file << content << "\n";
}

}  // namespace gpu
}  // namespace mosaic
