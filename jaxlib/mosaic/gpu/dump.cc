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

#if defined(__APPLE__)
// This is the fix recommended by
// https://www.gnu.org/software/gnulib/manual/html_node/environ.html to make
// sure accessing `environ` works on Apple platforms.
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#endif
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "jaxlib/mosaic/gpu/library_paths.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "tsl/platform/path.h"

namespace mosaic {
namespace gpu {

namespace {

class TemporaryDirectory {
 private:
  TemporaryDirectory(std::string path) : path(std::move(path)) {}
  // TODO(apaszke): Unlink in destructor.

 public:
  static absl::StatusOr<TemporaryDirectory> Create() {
    std::string pattern = "/tmp/mosaic-gpu-XXXXXX";
    if (mkdtemp(pattern.data()) == NULL) {
      return absl::InternalError("Failed to create temporary directory");
    }
    return TemporaryDirectory(std::move(pattern));
  }

  std::string_view GetPath() { return path; }

 private:
  std::string path;
};

absl::StatusOr<std::string> RunCUDATool(const char* tool,
                                        const std::vector<const char*>& args,
                                        bool stderr_to_stdout = true) {
  CHECK(!args.empty() && args.back() == nullptr);
  const char* cuda_path_ptr = mosaic::gpu::GetCUDARoot();
  if (!cuda_path_ptr)
    return absl::InternalError("Failed to get the CUDA toolkit path");
  std::string tool_path(cuda_path_ptr);
  tool_path += "/bin/";
  tool_path += tool;
  int stdout_pipe[2] = {-1, -1};
  pid_t child_pid;
  posix_spawn_file_actions_t file_actions;
  if (posix_spawn_file_actions_init(&file_actions)) {
    return absl::InternalError("Failed to initialize spawn file actions");
  }
  absl::Cleanup file_actions_destroyer = [&file_actions] {
    posix_spawn_file_actions_destroy(&file_actions);
  };
  if (pipe(stdout_pipe) == -1) {
    return absl::InternalError("Failed to set up pipe");
  }
  absl::Cleanup pipe_closer = [&stdout_pipe] {
    if (stdout_pipe[0] != -1) close(stdout_pipe[0]);
    if (stdout_pipe[1] != -1) close(stdout_pipe[1]);
  };
  // close read end in child
  if (posix_spawn_file_actions_addclose(&file_actions, stdout_pipe[0])) {
    return absl::InternalError("Failed to close read end of the pipe in child");
  }
  if (posix_spawn_file_actions_adddup2(&file_actions, stdout_pipe[1],
                                       STDOUT_FILENO)) {
    return absl::InternalError("Failed to redirect stdout to pipe");
  }
  if (stderr_to_stdout && posix_spawn_file_actions_adddup2(
                              &file_actions, STDOUT_FILENO, STDERR_FILENO)) {
    return absl::InternalError("Failed to redirect stderr to stdout");
  }
  // execv is guaranteed by POSIX to not modify the args (other than
  // replacing the whole process image), so the const_cast is valid.
  if (int status =
          posix_spawn(&child_pid, tool_path.c_str(), &file_actions, nullptr,
                      const_cast<char* const*>(args.data()), environ)) {
    return absl::InternalError(
        absl::StrCat("Process spawn failed: ", strerror(status)));
  }
  // Proactively close write end in parent. If we don't do this, read
  // will block since the pipe will have an open write end in the
  // parent process.
  if (close(stdout_pipe[1]) == -1) {
    return absl::InternalError(
        absl::StrCat("Failed to close write end of pipe in parent process: ",
                     strerror(errno)));
  }
  // Mark the write end as successfully closed, so it doesn't get
  // closed a second time by the deferred pipe_closer.
  stdout_pipe[1] = -1;
  std::string stdout;
  char buf[1024];
  while (int bytes_read = read(stdout_pipe[0], buf, sizeof buf)) {
    if (bytes_read == -1) {
      return absl::InternalError(
          absl::StrCat("Failed to read from pipe: ", strerror(errno)));
    }
    stdout.append(buf, bytes_read);
  }
  int status;
  if (waitpid(child_pid, &status, 0) == -1) {
    return absl::InternalError("Failed to wait for CUDA tool invocation");
  }
  if (status != 0) {
    std::string error_message = "CUDA tool failed";
    if (!stdout.empty()) {
      error_message += ": ";
      error_message += stdout;
    }
    return absl::InternalError(error_message);
  }
  return stdout;
}

// Parse the SASS and reformat control codes following NervanaSystems/maxas.
std::string FormatSassCtrl(const std::string& sass) {
  std::string result;
  result.reserve(sass.size());
  std::vector<std::string> lines = absl::StrSplit(sass, '\n');
  for (int i = 0; i < lines.size(); ++i) {
    std::string_view line = lines[i];
    if (i + 1 < lines.size()) {
      const std::string& next_line = lines[i + 1];
      size_t first_hex_start = line.rfind("/* 0x");
      size_t first_instr_end = line.rfind(';');
      size_t second_hex_start = next_line.rfind("/* 0x");
      bool second_line_empty = true;
      if (second_hex_start != std::string::npos) {
        for (size_t i = 0; i < second_hex_start; ++i) {
          second_line_empty &= next_line[i] == ' ';
        }
      }
      if (first_hex_start != std::string::npos &&
          first_instr_end != std::string::npos &&
          second_hex_start != std::string::npos && second_line_empty) {
        line = line.substr(0, first_instr_end);
        std::string hex_str = next_line.substr(second_hex_start + 5, 16);
        uint64_t ctrl;
        if (absl::SimpleHexAtoi(hex_str, &ctrl)) {
          uint64_t stall = (ctrl >> 41) & 0xf;
          uint64_t yield = (ctrl >> 45) & 0x1;
          uint64_t write_barrier = (ctrl >> 46) & 0x7;
          uint64_t read_barrier = (ctrl >> 49) & 0x7;
          uint64_t wait_barrier = (ctrl >> 52) & 0x3f;
          std::string wait_barrier_str;
          if (wait_barrier == 0) {
            result += "   -";
          } else if (absl::has_single_bit(wait_barrier)) {
            absl::StrAppendFormat(&result, "   %d",
                                  absl::countr_zero(wait_barrier));
          } else {
            int first_set = absl::countr_zero(wait_barrier);
            uint64_t without_first_set = wait_barrier ^ (1 << first_set);
            if (absl::has_single_bit(without_first_set)) {
              absl::StrAppendFormat(&result, " %d&%d",
                                    absl::countr_zero(without_first_set),
                                    first_set);
            } else {
              absl::StrAppendFormat(&result, "0x%02x", wait_barrier);
            }
          }
          absl::StrAppendFormat(
              &result, ":%c:%c:%c:%02llu",
              read_barrier == 7 ? '-' : ('0' + read_barrier),
              write_barrier == 7 ? '-' : ('0' + write_barrier),
              yield ? 'Y' : '-', stall);
        }
        i++;  // Skip the hex line.
      }
    }
    result += line;
    result.append("\n");
  }
  return result;
}

// The name of the attribute wrapping the module basename for dumping. See
// `GetDumpOptionsForModule` for more details.
constexpr std::string_view kDumpBasenameAttr = "mosaic_gpu.dump_basename";

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
    if (std::string_view(dump_to) == "sponge") {
      if (char* dump_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
          dump_dir != nullptr) {
        opts.dump_path = dump_dir;
      } else {
        LOG(WARNING) << "\"sponge\" specified as dump directory but "
                        "TEST_UNDECLARED_OUTPUTS_DIR is not set! "
                        "Will dump to stdout instead.";
      }
    } else if (std::string_view(dump_to) == "-") {
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

void DumpToFileOrStdout(std::string_view content, std::string_view name,
                        std::string_view path) {
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

void DumpSass(mlir::gpu::BinaryOp binary, std::string_view path,
              std::string_view basename, bool include_sass_ctrl) {
  auto objects = binary.getObjects();
  if (objects.size() != 1) {
    std::cerr << "Multiple objects per gpu.binary unsupported" << std::endl;
    return;
  }
  auto object = mlir::cast<mlir::gpu::ObjectAttr>(*objects.begin());
  if (object.getFormat() != mlir::gpu::CompilationTarget::Binary) {
    std::cerr << "gpu.binary object is not in binary format" << std::endl;
    return;
  }
  std::string elf = object.getObject().getValue().str();
  auto tmpdir = TemporaryDirectory::Create();
  if (!tmpdir.ok()) {
    std::cerr << "Failed to create a temporary directory" << std::endl;
    return;
  }
  std::string elf_path = std::string(tmpdir->GetPath()) + "/kernel.bin";
  // Dump ELF into a file.
  std::ofstream elf_out(elf_path.c_str());
  if (!elf_out) {
    std::cerr << "Failed to write binary to a file" << std::endl;
    return;
  }
  elf_out << elf << std::endl;
  // Call nvdisasm to pretty-print SASS.
  std::vector<const char*> nvdisasm_args = {"nvdisasm", "-ndf", "-c",
                                            elf_path.c_str()};
  if (include_sass_ctrl) {
    nvdisasm_args.push_back("-hex");
  }
  nvdisasm_args.push_back(nullptr);
  auto result = RunCUDATool("nvdisasm", nvdisasm_args);
  if (!result.ok()) {
    std::cerr << "nvdisasm invocation failed: " << result.status() << std::endl;
    return;
  }

  if (include_sass_ctrl) {
    mosaic::gpu::DumpToFileOrStdout(FormatSassCtrl(*result),
                                    absl::StrCat(basename, ".sass_ctrl"), path);
  } else {
    // Dump SASS.
    mosaic::gpu::DumpToFileOrStdout(*result, absl::StrCat(basename, ".sass"),
                                    path);
  }
}

}  // namespace gpu
}  // namespace mosaic
