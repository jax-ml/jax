#!/usr/bin/python
#
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Helper script for building JAX's libjax easily.

import argparse
import logging
import os
import platform
import textwrap

from tools import utils


logger = logging.getLogger(__name__)

def write_bazelrc(*, remote_build,
                  cuda_version, cudnn_version, rocm_toolkit_path,
                  cpu, cuda_compute_capabilities,
                  rocm_amdgpu_targets, target_cpu_features,
                  wheel_cpu, enable_mkl_dnn, use_clang, clang_path,
                  clang_major_version, python_version,
                  enable_cuda, enable_nccl, enable_rocm,
                  use_cuda_nvcc):

  with open("../.jax_configure.bazelrc", "w") as f:
    if not remote_build:
      f.write(textwrap.dedent("""\
        build --strategy=Genrule=standalone
        """))

    if use_clang:
      f.write(f'build --action_env CLANG_COMPILER_PATH="{clang_path}"\n')
      f.write(f'build --repo_env CC="{clang_path}"\n')
      f.write(f'build --repo_env BAZEL_COMPILER="{clang_path}"\n')
      f.write('build --copt=-Wno-error=unused-command-line-argument\n')
      if clang_major_version in (16, 17, 18):
        # Necessary due to XLA's old version of upb. See:
        # https://github.com/openxla/xla/blob/c4277a076e249f5b97c8e45c8cb9d1f554089d76/.bazelrc#L505
        f.write("build --copt=-Wno-gnu-offsetof-extensions\n")

    if rocm_toolkit_path:
      f.write("build --action_env ROCM_PATH=\"{rocm_toolkit_path}\"\n"
              .format(rocm_toolkit_path=rocm_toolkit_path))
    if rocm_amdgpu_targets:
      f.write(
        f'build:rocm --action_env TF_ROCM_AMDGPU_TARGETS="{rocm_amdgpu_targets}"\n')
    if cpu is not None:
      f.write(f"build --cpu={cpu}\n")

    if target_cpu_features == "release":
      if wheel_cpu == "x86_64":
        f.write("build --config=avx_windows\n" if utils.is_windows()
                else "build --config=avx_posix\n")
    elif target_cpu_features == "native":
      if utils.is_windows():
        print("--target_cpu_features=native is not supported on Windows; ignoring.")
      else:
        f.write("build --config=native_arch_posix\n")

    if enable_mkl_dnn:
      f.write("build --config=mkl_open_source_only\n")
    if enable_cuda:
      f.write("build --config=cuda\n")
      if use_cuda_nvcc:
        f.write("build --config=build_cuda_with_nvcc\n")
      else:
        f.write("build --config=build_cuda_with_clang\n")
      f.write(f"build --action_env=CLANG_CUDA_COMPILER_PATH={clang_path}\n")
      if not enable_nccl:
        f.write("build --config=nonccl\n")
      if cuda_version:
        f.write("build --repo_env HERMETIC_CUDA_VERSION=\"{cuda_version}\"\n"
                .format(cuda_version=cuda_version))
      if cudnn_version:
        f.write("build --repo_env HERMETIC_CUDNN_VERSION=\"{cudnn_version}\"\n"
                .format(cudnn_version=cudnn_version))
      if cuda_compute_capabilities:
        f.write(
          f'build:cuda --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES="{cuda_compute_capabilities}"\n')
    if enable_rocm:
      f.write("build --config=rocm_base\n")
      if not enable_nccl:
        f.write("build --config=nonccl\n")
      if use_clang:
        f.write("build --config=rocm\n")
        f.write(f"build --action_env=CLANG_COMPILER_PATH={clang_path}\n")
    if python_version:
      f.write(
        "build --repo_env HERMETIC_PYTHON_VERSION=\"{python_version}\"".format(
            python_version=python_version))
BANNER = r"""
     _   _  __  __
    | | / \ \ \/ /
 _  | |/ _ \ \  /
| |_| / ___ \/  \
 \___/_/   \/_/\_\

"""

EPILOG = """

From the 'build' directory in the JAX repository, run
    python build.py
or
    python3 build.py
to download and build JAX's XLA (jaxlib) dependency.
"""


def _parse_string_as_bool(s):
  """Parses a string as a boolean argument."""
  lower = s.lower()
  if lower == "true":
    return True
  elif lower == "false":
    return False
  else:
    raise ValueError(f"Expected either 'true' or 'false'; got {s}")


def add_boolean_argument(parser, name, default=False, help_str=None):
  """Creates a boolean flag."""
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--" + name,
      nargs="?",
      default=default,
      const=True,
      type=_parse_string_as_bool,
      help=help_str)
  group.add_argument("--no" + name, dest=name, action="store_false")


def _get_editable_output_paths(output_path):
  """Returns the paths to the editable wheels."""
  return (
      os.path.join(output_path, "jaxlib"),
      os.path.join(output_path, "jax_gpu_pjrt"),
      os.path.join(output_path, "jax_gpu_plugin"),
  )


def main():
  cwd = os.getcwd()
  parser = argparse.ArgumentParser(
      description="Builds jaxlib from source.", epilog=EPILOG)
  add_boolean_argument(
      parser,
      "verbose",
      default=False,
      help_str="Should we produce verbose debugging output?")
  parser.add_argument(
      "--bazel_path",
      help="Path to the Bazel binary to use. The default is to find bazel via "
      "the PATH; if none is found, downloads a fresh copy of bazel from "
      "GitHub.")
  parser.add_argument(
      "--python_bin_path",
      help="Path to Python binary whose version to match while building with "
      "hermetic python. The default is the Python interpreter used to run the "
      "build script. DEPRECATED: use --python_version instead.")
  parser.add_argument(
      "--target_cpu_features",
      choices=["release", "native", "default"],
      default="release",
      help="What CPU features should we target? 'release' enables CPU "
           "features that should be enabled for a release build, which on "
           "x86-64 architectures enables AVX. 'native' enables "
           "-march=native, which generates code targeted to use all "
           "features of the current machine. 'default' means don't opt-in "
           "to any architectural features and use whatever the C compiler "
           "generates by default.")
  add_boolean_argument(
      parser,
      "use_clang",
      default = "true",
      help_str=(
          "DEPRECATED: This flag is redundant because clang is "
          "always used as default compiler."
      ),
  )
  parser.add_argument(
      "--clang_path",
      help=(
          "Path to clang binary to use. The default is "
          "to find clang via the PATH."
      ),
  )
  add_boolean_argument(
      parser,
      "enable_mkl_dnn",
      default=True,
      help_str="Should we build with MKL-DNN enabled?",
  )
  add_boolean_argument(
      parser,
      "enable_cuda",
      help_str="Should we build with CUDA enabled? Requires CUDA and CuDNN."
  )
  add_boolean_argument(
      parser,
      "use_cuda_nvcc",
      default=True,
      help_str=(
          "Should we build CUDA code using NVCC compiler driver? The default value "
          "is true. If --nouse_cuda_nvcc flag is used then CUDA code is built "
          "by clang compiler."
      ),
  )
  add_boolean_argument(
      parser,
      "build_gpu_plugin",
      default=False,
      help_str=(
          "Are we building the gpu plugin in addition to jaxlib? The GPU "
          "plugin is still experimental and is not ready for use yet."
      ),
  )
  parser.add_argument(
      "--build_gpu_kernel_plugin",
      choices=["cuda", "rocm"],
      default="",
      help=(
          "Specify 'cuda' or 'rocm' to build the respective kernel plugin."
          " When this flag is set, jaxlib will not be built."
      ),
  )
  add_boolean_argument(
      parser,
      "build_gpu_pjrt_plugin",
      default=False,
      help_str=(
          "Are we building the cuda/rocm pjrt plugin? jaxlib will not be built "
          "when this flag is True."
      ),
  )
  parser.add_argument(
      "--gpu_plugin_cuda_version",
      choices=["12"],
      default="12",
      help="Which CUDA major version the gpu plugin is for.")
  parser.add_argument(
      "--gpu_plugin_rocm_version",
      choices=["60"],
      default="60",
      help="Which ROCM major version the gpu plugin is for.")
  add_boolean_argument(
      parser,
      "enable_rocm",
      help_str="Should we build with ROCm enabled?")
  add_boolean_argument(
      parser,
      "enable_nccl",
      default=True,
      help_str="Should we build with NCCL enabled? Has no effect for non-CUDA "
               "builds.")
  add_boolean_argument(
      parser,
      "remote_build",
      default=False,
      help_str="Should we build with RBE (Remote Build Environment)?")
  parser.add_argument(
      "--cuda_version",
      default=None,
      help="CUDA toolkit version, e.g., 12.3.2")
  parser.add_argument(
      "--cudnn_version",
      default=None,
      help="CUDNN version, e.g., 8.9.7.29")
  # Caution: if changing the default list of CUDA capabilities, you should also
  # update the list in .bazelrc, which is used for wheel builds.
  parser.add_argument(
      "--cuda_compute_capabilities",
      default=None,
      help="A comma-separated list of CUDA compute capabilities to support.")
  parser.add_argument(
      "--rocm_amdgpu_targets",
      default="gfx900,gfx906,gfx908,gfx90a,gfx940,gfx941,gfx942,gfx1030,gfx1100",
      help="A comma-separated list of ROCm amdgpu targets to support.")
  parser.add_argument(
      "--rocm_path",
      default=None,
      help="Path to the ROCm toolkit.")
  parser.add_argument(
      "--bazel_startup_options",
      action="append", default=[],
      help="Additional startup options to pass to bazel.")
  parser.add_argument(
      "--bazel_options",
      action="append", default=[],
      help="Additional options to pass to the main Bazel command to be "
           "executed, e.g. `run`.")
  parser.add_argument(
      "--output_path",
      default=os.path.join(cwd, "dist"),
      help="Directory to which the jaxlib wheel should be written")
  parser.add_argument(
      "--target_cpu",
      default=None,
      help="CPU platform to target. Default is the same as the host machine. "
           "Currently supported values are 'darwin_arm64' and 'darwin_x86_64'.")
  parser.add_argument(
      "--editable",
      action="store_true",
      help="Create an 'editable' jaxlib build instead of a wheel.")
  parser.add_argument(
      "--python_version",
      default=None,
      help="hermetic python version, e.g., 3.10")
  add_boolean_argument(
      parser,
      "configure_only",
      default=False,
      help_str="If true, writes a .bazelrc file but does not build jaxlib.")
  add_boolean_argument(
      parser,
      "requirements_update",
      default=False,
      help_str="If true, writes a .bazelrc and updates requirements_lock.txt "
               "for a corresponding version of Python but does not build "
               "jaxlib.")
  add_boolean_argument(
      parser,
      "requirements_nightly_update",
      default=False,
      help_str="Same as update_requirements, but will consider dev, nightly "
               "and pre-release versions of packages.")

  args = parser.parse_args()

  logging.basicConfig()
  if args.verbose:
    logger.setLevel(logging.DEBUG)

  if args.enable_cuda and args.enable_rocm:
    parser.error("--enable_cuda and --enable_rocm cannot be enabled at the same time.")

  print(BANNER)

  output_path = os.path.abspath(args.output_path)
  os.chdir(os.path.dirname(__file__ or args.prog) or '.')

  host_cpu = platform.machine()
  wheel_cpus = {
      "darwin_arm64": "arm64",
      "darwin_x86_64": "x86_64",
      "ppc": "ppc64le",
      "aarch64": "aarch64",
  }
  # TODO(phawkins): support other bazel cpu overrides.
  wheel_cpu = (wheel_cpus[args.target_cpu] if args.target_cpu is not None
               else host_cpu)

  # Find a working Bazel.
  bazel_path, bazel_version = utils.get_bazel_path(args.bazel_path)
  print(f"Bazel binary path: {bazel_path}")
  print(f"Bazel version: {bazel_version}")

  if args.python_version:
    python_version = args.python_version
  else:
    python_bin_path = utils.get_python_bin_path(args.python_bin_path)
    print(f"Python binary path: {python_bin_path}")
    python_version = utils.get_python_version(python_bin_path)
    print("Python version: {}".format(".".join(map(str, python_version))))
    utils.check_python_version(python_version)
    python_version = ".".join(map(str, python_version))

  print("Use clang: {}".format("yes" if args.use_clang else "no"))
  clang_path = args.clang_path
  clang_major_version = None
  if args.use_clang:
    if not clang_path:
      clang_path = utils.get_clang_path_or_exit()
    print(f"clang path: {clang_path}")
    clang_major_version = utils.get_clang_major_version(clang_path)

  print("MKL-DNN enabled: {}".format("yes" if args.enable_mkl_dnn else "no"))
  print(f"Target CPU: {wheel_cpu}")
  print(f"Target CPU features: {args.target_cpu_features}")

  rocm_toolkit_path = args.rocm_path
  print("CUDA enabled: {}".format("yes" if args.enable_cuda else "no"))
  if args.enable_cuda:
    if args.cuda_compute_capabilities is not None:
      print(f"CUDA compute capabilities: {args.cuda_compute_capabilities}")
    if args.cuda_version:
      print(f"CUDA version: {args.cuda_version}")
    if args.cudnn_version:
      print(f"CUDNN version: {args.cudnn_version}")
    print("NCCL enabled: {}".format("yes" if args.enable_nccl else "no"))

  print("ROCm enabled: {}".format("yes" if args.enable_rocm else "no"))
  if args.enable_rocm:
    if rocm_toolkit_path:
      print(f"ROCm toolkit path: {rocm_toolkit_path}")
    print(f"ROCm amdgpu targets: {args.rocm_amdgpu_targets}")

  write_bazelrc(
      remote_build=args.remote_build,
      cuda_version=args.cuda_version,
      cudnn_version=args.cudnn_version,
      rocm_toolkit_path=rocm_toolkit_path,
      cpu=args.target_cpu,
      cuda_compute_capabilities=args.cuda_compute_capabilities,
      rocm_amdgpu_targets=args.rocm_amdgpu_targets,
      target_cpu_features=args.target_cpu_features,
      wheel_cpu=wheel_cpu,
      enable_mkl_dnn=args.enable_mkl_dnn,
      use_clang=args.use_clang,
      clang_path=clang_path,
      clang_major_version=clang_major_version,
      python_version=python_version,
      enable_cuda=args.enable_cuda,
      enable_nccl=args.enable_nccl,
      enable_rocm=args.enable_rocm,
      use_cuda_nvcc=args.use_cuda_nvcc,
  )

  if args.requirements_update or args.requirements_nightly_update:
    if args.requirements_update:
      task = "//build:requirements.update"
    else:  # args.requirements_nightly_update
      task = "//build:requirements_nightly.update"
    update_command = ([bazel_path] + args.bazel_startup_options +
      ["run", "--verbose_failures=true", task, *args.bazel_options])
    print(" ".join(update_command))
    utils.shell(update_command)
    return

  if args.configure_only:
    return

  print("\nBuilding XLA and installing it in the jaxlib source tree...")

  command_base = (
    bazel_path,
    *args.bazel_startup_options,
    "run",
    "--verbose_failures=true",
    *args.bazel_options,
  )

  if args.build_gpu_plugin and args.editable:
    output_path_jaxlib, output_path_jax_pjrt, output_path_jax_kernel = (
        _get_editable_output_paths(output_path)
    )
  else:
    output_path_jaxlib = output_path
    output_path_jax_pjrt = output_path
    output_path_jax_kernel = output_path

  if args.build_gpu_kernel_plugin == "" and not args.build_gpu_pjrt_plugin:
    build_cpu_wheel_command = [
        *command_base,
        "//jaxlib/tools:build_wheel",
        "--",
        f"--output_path={output_path_jaxlib}",
        f"--jaxlib_git_hash={utils.get_githash()}",
        f"--cpu={wheel_cpu}",
    ]
    if args.build_gpu_plugin:
      build_cpu_wheel_command.append("--skip_gpu_kernels")
    if args.editable:
      build_cpu_wheel_command.append("--editable")
    print(" ".join(build_cpu_wheel_command))
    utils.shell(build_cpu_wheel_command)

  if args.build_gpu_plugin or (args.build_gpu_kernel_plugin == "cuda") or \
      (args.build_gpu_kernel_plugin == "rocm"):
    build_gpu_kernels_command = [
        *command_base,
        "//jaxlib/tools:build_gpu_kernels_wheel",
        "--",
        f"--output_path={output_path_jax_kernel}",
        f"--jaxlib_git_hash={utils.get_githash()}",
        f"--cpu={wheel_cpu}",
    ]
    if args.enable_cuda:
      build_gpu_kernels_command.append(f"--enable-cuda={args.enable_cuda}")
      build_gpu_kernels_command.append(f"--platform_version={args.gpu_plugin_cuda_version}")
    elif args.enable_rocm:
      build_gpu_kernels_command.append(f"--enable-rocm={args.enable_rocm}")
      build_gpu_kernels_command.append(f"--platform_version={args.gpu_plugin_rocm_version}")
    else:
      raise ValueError("Unsupported GPU plugin backend. Choose either 'cuda' or 'rocm'.")
    if args.editable:
      build_gpu_kernels_command.append("--editable")
    print(" ".join(build_gpu_kernels_command))
    utils.shell(build_gpu_kernels_command)

  if args.build_gpu_plugin or args.build_gpu_pjrt_plugin:
    build_pjrt_plugin_command = [
        *command_base,
        "//jaxlib/tools:build_gpu_plugin_wheel",
        "--",
        f"--output_path={output_path_jax_pjrt}",
        f"--jaxlib_git_hash={utils.get_githash()}",
        f"--cpu={wheel_cpu}",
    ]
    if args.enable_cuda:
      build_pjrt_plugin_command.append(f"--enable-cuda={args.enable_cuda}")
      build_pjrt_plugin_command.append(f"--platform_version={args.gpu_plugin_cuda_version}")
    elif args.enable_rocm:
      build_pjrt_plugin_command.append(f"--enable-rocm={args.enable_rocm}")
      build_pjrt_plugin_command.append(f"--platform_version={args.gpu_plugin_rocm_version}")
    else:
      raise ValueError("Unsupported GPU plugin backend. Choose either 'cuda' or 'rocm'.")
    if args.editable:
      build_pjrt_plugin_command.append("--editable")
    print(" ".join(build_pjrt_plugin_command))
    utils.shell(build_pjrt_plugin_command)

  utils.shell([bazel_path] + args.bazel_startup_options + ["shutdown"])


if __name__ == "__main__":
  main()
