#!/usr/bin/env python3
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
# CLI for building JAX wheel packages from source and for updating the
# requirements_lock.txt files

import argparse
import asyncio
import logging
import os
import platform
import sys
import copy

from tools import command, utils


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BANNER = r"""
     _   _  __  __
    | | / \ \ \/ /
 _  | |/ _ \ \  /
| |_| / ___ \/  \
 \___/_/   \/_/\_\

"""

EPILOG = """
From the root directory of the JAX repository, run
  `python build/build.py build --wheels=<list of JAX wheels>` to build JAX
  artifacts.

  Multiple wheels can be built with a single invocation of the CLI.
  E.g. python build/build.py build --wheels=jaxlib,jax-cuda-plugin

  To update the requirements_lock.txt files, run
  `python build/build.py requirements_update`
"""

# Define the build target for each wheel.
WHEEL_BUILD_TARGET_DICT = {
    "jaxlib": "//jaxlib/tools:build_wheel",
    "jax-cuda-plugin": "//jaxlib/tools:build_gpu_kernels_wheel",
    "jax-cuda-pjrt": "//jaxlib/tools:build_gpu_plugin_wheel",
    "jax-rocm-plugin": "//jaxlib/tools:build_gpu_kernels_wheel",
    "jax-rocm-pjrt": "//jaxlib/tools:build_gpu_plugin_wheel",
}

# Dictionary with the new wheel build rule. Note that when JAX migrates to the
# new wheel build rule fully, the build CLI will switch to the new wheel build
# rule as the default.
WHEEL_BUILD_TARGET_DICT_NEW = {
    "jax": "//:jax_wheel",
    "jax_editable": "//:jax_wheel_editable",
    "jax_source_package": "//:jax_source_package",
    "jaxlib": "//jaxlib/tools:jaxlib_wheel",
    "jaxlib_editable": "//jaxlib/tools:jaxlib_wheel_editable",
    "jax-cuda-plugin": "//jaxlib/tools:jax_cuda_plugin_wheel",
    "jax-cuda-plugin_editable": "//jaxlib/tools:jax_cuda_plugin_wheel_editable",
    "jax-cuda-pjrt": "//jaxlib/tools:jax_cuda_pjrt_wheel",
    "jax-cuda-pjrt_editable": "//jaxlib/tools:jax_cuda_pjrt_wheel_editable",
    "jax-rocm-plugin": "//jaxlib/tools:jax_rocm_plugin_wheel",
    "jax-rocm-pjrt": "//jaxlib/tools:jax_rocm_pjrt_wheel",
}

_JAX_CUDA_VERSION = "12"

def add_global_arguments(parser: argparse.ArgumentParser):
  """Adds all the global arguments that applies to all the CLI subcommands."""
  parser.add_argument(
      "--python_version",
      type=str,
      default=f"{sys.version_info.major}.{sys.version_info.minor}",
      help=
        """
        Hermetic Python version to use. Default is to use the version of the
        Python binary that executed the CLI.
        """,
  )

  bazel_group = parser.add_argument_group('Bazel Options')
  bazel_group.add_argument(
      "--bazel_path",
      type=str,
      default="",
      help="""
        Path to the Bazel binary to use. The default is to find bazel via the
        PATH; if none is found, downloads a fresh copy of Bazel from GitHub.
        """,
  )

  bazel_group.add_argument(
      "--bazel_startup_options",
      action="append",
      default=[],
      help="""
        Additional startup options to pass to Bazel, can be specified multiple
        times to pass multiple options.
        E.g. --bazel_startup_options='--nobatch'
        """,
  )

  bazel_group.add_argument(
      "--bazel_options",
      action="append",
      default=[],
      help="""
        Additional build options to pass to Bazel, can be specified multiple
        times to pass multiple options.
        E.g. --bazel_options='--local_resources=HOST_CPUS'
        """,
  )

  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Prints the Bazel command that is going to be executed.",
  )

  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Produce verbose output for debugging.",
  )

  parser.add_argument(
      "--detailed_timestamped_log",
      action="store_true",
      help="""
        Enable detailed logging of the Bazel command with timestamps. The logs
        will be stored and can be accessed as artifacts.
        """,
  )


def add_artifact_subcommand_arguments(parser: argparse.ArgumentParser):
  """Adds all the arguments that applies to the artifact subcommands."""
  parser.add_argument(
      "--wheels",
      type=str,
      default="jaxlib",
      help=
        """
        A comma separated list of JAX wheels to build. E.g: --wheels="jaxlib",
        --wheels="jaxlib,jax-cuda-plugin", etc.
        Valid options are: jaxlib, jax-cuda-plugin or cuda-plugin, jax-cuda-pjrt or cuda-pjrt,
        jax-rocm-plugin or rocm-plugin, jax-rocm-pjrt or rocm-pjrt
        """,
  )

  parser.add_argument(
      "--use_new_wheel_build_rule",
      action="store_true",
      help=
        """
        Whether to use the new wheel build rule. Temporary flag and will be
        removed once JAX migrates to the new wheel build rule fully.
        """,
  )

  parser.add_argument(
      "--editable",
      action="store_true",
      help="Create an 'editable' build instead of a wheel.",
  )

  parser.add_argument(
      "--output_path",
      type=str,
      default=os.path.join(os.getcwd(), "dist"),
      help="Directory to which the JAX wheel packages should be written.",
  )

  parser.add_argument(
    "--configure_only",
    action="store_true",
    help="""
      If true, writes the Bazel options to the .jax_configure.bazelrc file but
      does not build the artifacts.
      """,
  )

  # CUDA Options
  cuda_group = parser.add_argument_group('CUDA Options')
  cuda_group.add_argument(
      "--cuda_version",
      type=str,
      help=
        """
        Hermetic CUDA version to use. Default is to use the version specified
        in the .bazelrc.
        """,
  )

  cuda_group.add_argument(
      "--cuda_major_version",
      type=str,
      default="12",
      help=
        """
        Which CUDA major version should the wheel be tagged as? Auto-detected if
        --cuda_version is set. When --cuda_version is not set, the default is to
        set the major version to 12 to match the default in .bazelrc.
        """,
  )

  cuda_group.add_argument(
      "--cudnn_version",
      type=str,
      help=
        """
        Hermetic cuDNN version to use. Default is to use the version specified
        in the .bazelrc.
        """,
  )

  cuda_group.add_argument(
      "--disable_nccl",
      action="store_true",
      help="Should NCCL be disabled?",
  )

  cuda_group.add_argument(
      "--cuda_compute_capabilities",
      type=str,
      default=None,
      help=
        """
        A comma-separated list of CUDA compute capabilities to support. Default
        is to use the values specified in the .bazelrc.
        """,
  )

  cuda_group.add_argument(
      "--build_cuda_with_clang",
      action="store_true",
      help="""
        Should CUDA code be compiled using Clang? The default behavior is to
        compile CUDA with NVCC.
        """,
  )

  # ROCm Options
  rocm_group = parser.add_argument_group('ROCm Options')
  rocm_group.add_argument(
      "--rocm_version",
      type=str,
      default="60",
      help="ROCm version to use",
  )

  rocm_group.add_argument(
      "--rocm_amdgpu_targets",
      type=str,
      default="gfx900,gfx906,gfx908,gfx90a,gfx940,gfx941,gfx942,gfx1030,gfx1100,gfx1200,gfx1201",
      help="A comma-separated list of ROCm amdgpu targets to support.",
  )

  rocm_group.add_argument(
      "--rocm_path",
      type=str,
      default="",
      help="Path to the ROCm toolkit.",
  )

  # Compile Options
  compile_group = parser.add_argument_group('Compile Options')

  compile_group.add_argument(
      "--use_clang",
      type=utils._parse_string_as_bool,
      default="true",
      const=True,
      nargs="?",
      help="""
        Whether to use Clang as the compiler. Not recommended to set this to
        False as JAX uses Clang as the default compiler.
        """,
  )

  compile_group.add_argument(
      "--clang_path",
      type=str,
      default="",
      help="""
        Path to the Clang binary to use.
        """,
  )

  compile_group.add_argument(
      "--gcc_path",
      type=str,
      default="",
      help="""
        Path to the GCC binary to use.
        """,
  )

  compile_group.add_argument(
      "--disable_mkl_dnn",
      action="store_true",
      help="""
        Disables MKL-DNN.
        """,
  )

  compile_group.add_argument(
      "--target_cpu_features",
      choices=["release", "native", "default"],
      default="release",
      help="""
        What CPU features should we target? Release enables CPU features that
        should be enabled for a release build, which on x86-64 architectures
        enables AVX. Native enables -march=native, which generates code targeted
        to use all features of the current machine. Default means don't opt-in
        to any architectural features and use whatever the C compiler generates
        by default.
        """,
  )

  compile_group.add_argument(
      "--target_cpu",
      default=None,
      help="CPU platform to target. Default is the same as the host machine.",
  )

  compile_group.add_argument(
      "--local_xla_path",
      type=str,
      default=os.environ.get("JAXCI_XLA_GIT_DIR", ""),
      help="""
        Path to local XLA repository to use. If not set, Bazel uses the XLA at
        the pinned version in workspace.bzl.
        """,
  )

async def main():
  parser = argparse.ArgumentParser(
      description=r"""
        CLI for building JAX wheel packages from source and for updating the
        requirements_lock.txt files
        """,
      epilog=EPILOG,
      formatter_class=argparse.RawDescriptionHelpFormatter
  )

  # Create subparsers for build and requirements_update
  subparsers = parser.add_subparsers(dest="command", required=True)

  # requirements_update subcommand
  requirements_update_parser = subparsers.add_parser(
      "requirements_update", help="Updates the requirements_lock.txt files"
  )
  requirements_update_parser.add_argument(
    "--nightly_update",
    action="store_true",
    help="""
      If true, updates requirements_lock.txt for a corresponding version of
      Python and will consider dev, nightly and pre-release versions of
      packages.
      """,
  )
  add_global_arguments(requirements_update_parser)

  # Artifact build subcommand
  build_artifact_parser = subparsers.add_parser(
      "build", help="Builds the jaxlib, plugin, and pjrt artifact"
  )
  add_artifact_subcommand_arguments(build_artifact_parser)
  add_global_arguments(build_artifact_parser)

  arch = platform.machine()
  os_name = platform.system().lower()

  custom_wheel_version_suffix = ""
  wheel_build_date = ""
  wheel_git_hash = ""
  wheel_type = "snapshot"

  args = parser.parse_args()

  logger.info("%s", BANNER)

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Verbose logging enabled")

  bazel_path, bazel_version = utils.get_bazel_path(args.bazel_path)

  logging.debug("Bazel path: %s", bazel_path)
  logging.debug("Bazel version: %s", bazel_version)

  executor = command.SubprocessExecutor()

  # Start constructing the Bazel command
  bazel_command_base = command.CommandBuilder(bazel_path)

  if args.bazel_startup_options:
    logging.debug(
        "Additional Bazel startup options: %s", args.bazel_startup_options
    )
    for option in args.bazel_startup_options:
      bazel_command_base.append(option)

  if args.command == "requirements_update" or not args.use_new_wheel_build_rule:
    bazel_command_base.append("run")
  else:
    bazel_command_base.append("build")

  freethreaded = False
  if args.python_version:
    # Do not add --repo_env=HERMETIC_PYTHON_VERSION with default args.python_version
    # if bazel_options override it
    python_version_opt = "--repo_env=HERMETIC_PYTHON_VERSION="
    if any([python_version_opt in opt for opt in args.bazel_options]):
      raise RuntimeError(
        "Please use python_version to set hermetic python version instead of "
        "setting --repo_env=HERMETIC_PYTHON_VERSION=<python version> bazel option"
      )
    logging.debug("Hermetic Python version: %s", args.python_version)
    bazel_command_base.append(
        f"--repo_env=HERMETIC_PYTHON_VERSION={args.python_version}"
    )
    # Let's interpret X.YY-ft version as free-threading python and set rules_python config flag:
    if args.python_version.endswith("-ft"):
      freethreaded = True
      bazel_command_base.append(
        "--@rules_python//python/config_settings:py_freethreaded='yes'"
      )

  # Enable verbose failures.
  bazel_command_base.append("--verbose_failures=true")

  # Requirements update subcommand execution
  if args.command == "requirements_update":
    requirements_command = copy.deepcopy(bazel_command_base)
    if args.bazel_options:
      logging.debug(
          "Using additional build options: %s", args.bazel_options
      )
      for option in args.bazel_options:
        requirements_command.append(option)

    ft_suffix = "_ft" if freethreaded else ""
    if args.nightly_update:
      logging.info(
          "--nightly_update is set. Bazel will run"
          " //build:requirements_nightly.update"
      )
      requirements_command.append(f"//build:requirements{ft_suffix}_nightly.update")
    else:
      requirements_command.append(f"//build:requirements{ft_suffix}.update")

    result = await executor.run(requirements_command.get_command_as_string(), args.dry_run, args.detailed_timestamped_log)
    if result.return_code != 0:
      raise RuntimeError(f"Command failed with return code {result.return_code}")
    else:
      sys.exit(0)

  wheel_build_command_base = copy.deepcopy(bazel_command_base)

  wheel_cpus = {
      "darwin_arm64": "arm64",
      "darwin_x86_64": "x86_64",
      "ppc": "ppc64le",
      "aarch64": "aarch64",
  }
  target_cpu = (
      wheel_cpus[args.target_cpu] if args.target_cpu is not None else arch
  )

  if args.local_xla_path:
    logging.debug("Local XLA path: %s", args.local_xla_path)
    wheel_build_command_base.append(f"--override_repository=xla=\"{args.local_xla_path}\"")

  if args.target_cpu:
    logging.debug("Target CPU: %s", args.target_cpu)
    wheel_build_command_base.append(f"--cpu={args.target_cpu}")

  if args.disable_nccl:
    logging.debug("Disabling NCCL")
    wheel_build_command_base.append("--config=nonccl")

  git_hash = utils.get_githash()

  clang_path = ""
  if args.use_clang:
    clang_path = args.clang_path or utils.get_clang_path_or_exit()
    clang_major_version = utils.get_clang_major_version(clang_path)
    clangpp_path = utils.get_clangpp_path(clang_path)
    logging.debug(
        "Using Clang as the compiler, clang path: %s, clang version: %s",
        clang_path,
        clang_major_version,
    )

    # Use double quotes around clang path to avoid path issues on Windows.
    wheel_build_command_base.append(f"--action_env=CLANG_COMPILER_PATH=\"{clang_path}\"")
    wheel_build_command_base.append(f"--repo_env=CC=\"{clang_path}\"")
    wheel_build_command_base.append(f"--repo_env=CXX=\"{clangpp_path}\"")
    wheel_build_command_base.append(f"--repo_env=BAZEL_COMPILER=\"{clang_path}\"")

    if clang_major_version >= 16:
      # Enable clang settings that are needed for the build to work with newer
      # versions of Clang.
      wheel_build_command_base.append("--config=clang")
    if clang_major_version < 19:
      wheel_build_command_base.append("--define=xnn_enable_avxvnniint8=false")

  else:
    gcc_path = args.gcc_path or utils.get_gcc_path_or_exit()
    logging.debug(
        "Using GCC as the compiler, gcc path: %s",
        gcc_path,
    )
    wheel_build_command_base.append(f"--repo_env=CC=\"{gcc_path}\"")
    wheel_build_command_base.append(f"--repo_env=BAZEL_COMPILER=\"{gcc_path}\"")

    gcc_major_version = utils.get_gcc_major_version(gcc_path)
    if gcc_major_version < 13:
      wheel_build_command_base.append("--define=xnn_enable_avxvnniint8=false")

  if not args.disable_mkl_dnn:
    logging.debug("Enabling MKL DNN")
    if target_cpu == "aarch64":
      wheel_build_command_base.append("--config=mkl_aarch64_threadpool")
    else:
      wheel_build_command_base.append("--config=mkl_open_source_only")

  if args.target_cpu_features == "release":
    if arch in ["x86_64", "AMD64"]:
      logging.debug(
          "Using release cpu features: --config=avx_%s",
          "windows" if os_name == "windows" else "posix",
      )
      wheel_build_command_base.append(
          "--config=avx_windows"
          if os_name == "windows"
          else "--config=avx_posix"
      )
  elif args.target_cpu_features == "native":
    if os_name == "windows":
      logger.warning(
          "--target_cpu_features=native is not supported on Windows;"
          " ignoring."
      )
    else:
      logging.debug("Using native cpu features: --config=native_arch_posix")
      wheel_build_command_base.append("--config=native_arch_posix")
  else:
    logging.debug("Using default cpu features")

  if "cuda" in args.wheels and "rocm" in args.wheels:
    logging.error("CUDA and ROCm cannot be enabled at the same time.")
    sys.exit(1)

  if "cuda" in args.wheels:
    wheel_build_command_base.append("--config=cuda")
    if args.use_clang:
      wheel_build_command_base.append(
          f"--action_env=CLANG_CUDA_COMPILER_PATH=\"{clang_path}\""
      )
      if args.build_cuda_with_clang:
        logging.debug("Building CUDA with Clang")
        wheel_build_command_base.append("--config=build_cuda_with_clang")
      else:
        logging.debug("Building CUDA with NVCC")
        wheel_build_command_base.append("--config=build_cuda_with_nvcc")
    else:
      logging.debug("Building CUDA with NVCC")
      wheel_build_command_base.append("--config=build_cuda_with_nvcc")

    if args.cuda_version:
      logging.debug("Hermetic CUDA version: %s", args.cuda_version)
      wheel_build_command_base.append(
          f"--repo_env=HERMETIC_CUDA_VERSION={args.cuda_version}"
      )
    if args.cudnn_version:
      logging.debug("Hermetic cuDNN version: %s", args.cudnn_version)
      wheel_build_command_base.append(
          f"--repo_env=HERMETIC_CUDNN_VERSION={args.cudnn_version}"
      )
    if args.cuda_compute_capabilities:
      logging.debug(
          "Hermetic CUDA compute capabilities: %s",
          args.cuda_compute_capabilities,
      )
      wheel_build_command_base.append(
          f"--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES={args.cuda_compute_capabilities}"
      )

  if "rocm" in args.wheels:
    wheel_build_command_base.append("--config=rocm_base")
    if args.use_clang:
      wheel_build_command_base.append("--config=rocm")
      wheel_build_command_base.append(f"--action_env=CLANG_COMPILER_PATH=\"{clang_path}\"")
    if args.rocm_path:
      logging.debug("ROCm toolkit path: %s", args.rocm_path)
      wheel_build_command_base.append(f"--action_env=ROCM_PATH=\"{args.rocm_path}\"")
    if args.rocm_amdgpu_targets:
      logging.debug("ROCm AMD GPU targets: %s", args.rocm_amdgpu_targets)
      wheel_build_command_base.append(
          f"--action_env=TF_ROCM_AMDGPU_TARGETS={args.rocm_amdgpu_targets}"
      )

  # Append additional build options at the end to override any options set in
  # .bazelrc or above.
  if args.bazel_options:
    logging.debug(
        "Additional Bazel build options: %s", args.bazel_options
    )
    for option in args.bazel_options:
      wheel_build_command_base.append(option)

      # Parse the build options for the wheel version suffix.
      if "ML_WHEEL_TYPE" in option:
        wheel_type = option.split("=")[-1]
      if "ML_WHEEL_VERSION_SUFFIX" in option:
        custom_wheel_version_suffix = option.split("=")[-1].replace("-", "")
      if "ML_WHEEL_BUILD_DATE" in option:
        wheel_build_date = option.split("=")[-1].replace("-", "")
      if "ML_WHEEL_GIT_HASH" in option:
        # Strip leading zeros as they end up being stripped by setuptools,
        # which leads to a mismatch between expected and actual wheel names
        # https://peps.python.org/pep-0440/
        wheel_git_hash = option.split("=")[-1].lstrip('0')[:9]

  with open(".jax_configure.bazelrc", "w") as f:
    jax_configure_options = utils.get_jax_configure_bazel_options(wheel_build_command_base.get_command_as_list(), args.use_new_wheel_build_rule)
    if not jax_configure_options:
      logging.error("Error retrieving the Bazel options to be written to .jax_configure.bazelrc, exiting.")
      sys.exit(1)
    f.write(jax_configure_options)
    logging.info("Bazel options written to .jax_configure.bazelrc")

  if args.use_new_wheel_build_rule:
    logging.info("Using new wheel build rule")
    wheel_build_targets = WHEEL_BUILD_TARGET_DICT_NEW
  else:
    wheel_build_targets = WHEEL_BUILD_TARGET_DICT

  if args.configure_only:
    logging.info("--configure_only is set so not running any Bazel commands.")
  else:
    # Wheel build command execution
    for wheel in args.wheels.split(","):
      output_path = args.output_path
      logger.debug("Artifacts output directory: %s", output_path)

      # Allow CUDA/ROCm wheels without the "jax-" prefix.
      if ("plugin" in wheel or "pjrt" in wheel) and "jax" not in wheel:
        wheel = "jax-" + wheel

      if wheel not in wheel_build_targets.keys():
        logging.error(
            "Incorrect wheel name provided, valid choices are jaxlib,"
            " jax-cuda-plugin or cuda-plugin, jax-cuda-pjrt or cuda-pjrt,"
            " jax-rocm-plugin or rocm-plugin, jax-rocm-pjrt or rocm-pjrt"
        )
        sys.exit(1)

      wheel_build_command = copy.deepcopy(bazel_command_base)
      if "cuda" in args.wheels:
        wheel_build_command.append("--config=cuda_libraries_from_stubs")
      print("\n")
      logger.info(
        "Building %s for %s %s...",
        wheel,
        os_name,
        arch,
      )

      # Append the build target to the Bazel command.
      if args.use_new_wheel_build_rule and args.editable:
        build_target = wheel_build_targets[wheel + "_editable"]
      else:
        build_target = wheel_build_targets[wheel]
      wheel_build_command.append(build_target)
      if args.use_new_wheel_build_rule and wheel == "jax" and not args.editable:
        wheel_build_command.append(wheel_build_targets["jax_source_package"])

      if not args.use_new_wheel_build_rule:
        wheel_build_command.append("--")

        if args.editable:
          logger.info("Building an editable build")
          output_path = os.path.join(output_path, wheel)
          wheel_build_command.append("--editable")

        wheel_build_command.append(f'--output_path="{output_path}"')
        wheel_build_command.append(f"--cpu={target_cpu}")

        if "cuda" in wheel:
          wheel_build_command.append("--enable-cuda=True")
          if args.cuda_version:
            cuda_major_version = args.cuda_version.split(".")[0]
          else:
            cuda_major_version = args.cuda_major_version
          wheel_build_command.append(f"--platform_version={cuda_major_version}")

        if "rocm" in wheel:
          wheel_build_command.append("--enable-rocm=True")
          wheel_build_command.append(f"--platform_version={args.rocm_version}")

        wheel_build_command.append(f"--jaxlib_git_hash={git_hash}")

      result = await executor.run(wheel_build_command.get_command_as_string(), args.dry_run, args.detailed_timestamped_log)
      # Exit with error if any wheel build fails.
      if result.return_code != 0:
        raise RuntimeError(f"Command failed with return code {result.return_code}")

  if args.use_new_wheel_build_rule:
    output_path = args.output_path
    jax_bazel_dir = os.path.join("bazel-bin", "dist")
    jaxlib_and_plugins_bazel_dir = os.path.join(
        "bazel-bin", "jaxlib", "tools", "dist"
    )
    for wheel in args.wheels.split(","):
      if wheel == "jax":
        bazel_dir = jax_bazel_dir
      else:
        bazel_dir = jaxlib_and_plugins_bazel_dir
      if "cuda" in wheel:
        wheel_dir = wheel.replace("cuda", f"cuda{_JAX_CUDA_VERSION}").replace(
            "-", "_"
        )
      else:
        wheel_dir = wheel

      if args.editable:
        src_dir = os.path.join(bazel_dir, wheel_dir)
        dst_dir = os.path.join(output_path, wheel_dir)
        utils.copy_dir_recursively(src_dir, dst_dir)
      else:
        wheel_version_suffix = "dev0+selfbuilt"
        if wheel_type == "release":
          wheel_version_suffix = custom_wheel_version_suffix
        elif wheel_type in ["nightly", "custom"]:
          wheel_version_suffix = f".dev{wheel_build_date}"
          if wheel_type == "custom":
            wheel_version_suffix += (
                f"+{wheel_git_hash}{custom_wheel_version_suffix}"
            )
        if wheel in ["jax", "jax-cuda-pjrt"]:
          python_tag = "py"
        else:
          python_tag = "cp"
        utils.copy_individual_files(
            bazel_dir,
            output_path,
            f"{wheel_dir}*{wheel_version_suffix}-{python_tag}*.whl",
        )
        if wheel == "jax":
          utils.copy_individual_files(
              bazel_dir,
              output_path,
              f"{wheel_dir}*{wheel_version_suffix}.tar.gz",
          )

  # Exit with success if all wheels in the list were built successfully.
  sys.exit(0)


if __name__ == "__main__":
  asyncio.run(main())
