#!/usr/bin/python
#
# Copyright 2018 Google LLC
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
import collections
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib

# pylint: disable=g-import-not-at-top
if hasattr(urllib, "urlretrieve"):
  urlretrieve = urllib.urlretrieve
else:
  import urllib.request
  urlretrieve = urllib.request.urlretrieve

if hasattr(shutil, "which"):
  which = shutil.which
else:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


def is_windows():
    return sys.platform.startswith("win32")


def shell(cmd):
  output = subprocess.check_output(cmd)
  return output.decode("UTF-8").strip()


# Python

def get_python_bin_path(python_bin_path_flag):
  """Returns the path to the Python interpreter to use."""
  path = python_bin_path_flag or sys.executable
  return path.replace(os.sep, "/")


def get_python_version(python_bin_path):
  version_output = shell(
    [python_bin_path, "-c",
     "import sys; print(\"{}.{}\".format(sys.version_info[0], "
     "sys.version_info[1]))"])
  major, minor = map(int, version_output.split("."))
  return major, minor

def check_python_version(python_version):
  if python_version < (3, 6):
    print("JAX requires Python 3.6 or newer.")
    sys.exit(-1)


# Bazel

BAZEL_BASE_URI = "https://github.com/bazelbuild/bazel/releases/download/3.7.2/"
BazelPackage = collections.namedtuple("BazelPackage", ["file", "sha256"])
bazel_packages = {
    "Linux":
        BazelPackage(
            file="bazel-3.7.2-linux-x86_64",
            sha256=
            "70dc0bee198a4c3d332925a32d464d9036a831977501f66d4996854ad4e4fc0d"),
    "Darwin":
        BazelPackage(
            file="bazel-3.7.2-darwin-x86_64",
            sha256=
            "80c82e93a12ba30021692b11c78007807e82383a673be1602573b944beb359ab"),
    "Windows":
        BazelPackage(
            file="bazel-3.7.2-windows-x86_64.exe",
            sha256=
            "ecb696b1b9c9da6728d92fbfe8410bafb4b3a65c358980e49742233f33f74d10"),
}


def download_and_verify_bazel():
  """Downloads a bazel binary from Github, verifying its SHA256 hash."""
  package = bazel_packages.get(platform.system())
  if package is None:
    return None

  if not os.access(package.file, os.X_OK):
    uri = BAZEL_BASE_URI + package.file
    sys.stdout.write("Downloading bazel from: {}\n".format(uri))

    def progress(block_count, block_size, total_size):
      if total_size <= 0:
        total_size = 170**6
      progress = (block_count * block_size) / total_size
      num_chars = 40
      progress_chars = int(num_chars * progress)
      sys.stdout.write("{} [{}{}] {}%\r".format(
          package.file, "#" * progress_chars,
          "." * (num_chars - progress_chars), int(progress * 100.0)))

    tmp_path, _ = urlretrieve(uri, None,
                              progress if sys.stdout.isatty() else None)
    sys.stdout.write("\n")

    # Verify that the downloaded Bazel binary has the expected SHA256.
    with open(tmp_path, "rb") as downloaded_file:
      contents = downloaded_file.read()

    digest = hashlib.sha256(contents).hexdigest()
    if digest != package.sha256:
      print(
          "Checksum mismatch for downloaded bazel binary (expected {}; got {})."
          .format(package.sha256, digest))
      sys.exit(-1)

    # Write the file as the bazel file name.
    with open(package.file, "wb") as out_file:
      out_file.write(contents)

    # Mark the file as executable.
    st = os.stat(package.file)
    os.chmod(package.file,
             st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  return os.path.join(".", package.file)


def get_bazel_paths(bazel_path_flag):
  """Yields a sequence of guesses about bazel path. Some of sequence elements
  can be None. The resulting iterator is lazy and potentially has a side
  effects."""
  yield bazel_path_flag
  yield which("bazel")
  yield download_and_verify_bazel()


def get_bazel_path(bazel_path_flag):
  """Returns the path to a Bazel binary, downloading Bazel if not found. Also,
  it checks Bazel's version at lease newer than 2.0.0.

  NOTE Manual version check is reasonably only for bazel < 2.0.0. Newer bazel
  releases performs version check against .bazelversion (see for details
  https://blog.bazel.build/2019/12/19/bazel-2.0.html#other-important-changes).
  """
  for path in filter(None, get_bazel_paths(bazel_path_flag)):
    if check_bazel_version(path):
      return path

  print("Cannot find or download bazel. Please install bazel.")
  sys.exit(-1)


def check_bazel_version(bazel_path):
  try:
    version_output = shell([bazel_path, "--bazelrc=/dev/null", "version"])
  except subprocess.CalledProcessError:
    return False
  match = re.search("Build label: *([0-9\\.]+)[^0-9\\.]", version_output)
  if match is None:
    return False
  actual_ints = [int(x) for x in match.group(1).split(".")]
  return actual_ints >= (2, 0, 0)


BAZELRC_TEMPLATE = """
# Flag to enable remote config
common --experimental_repo_remote_exec

build --repo_env PYTHON_BIN_PATH="{python_bin_path}"
build --action_env=PYENV_ROOT
build --python_path="{python_bin_path}"
build --repo_env TF_NEED_CUDA="{tf_need_cuda}"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="{cuda_compute_capabilities}"
build --repo_env TF_NEED_ROCM="{tf_need_rocm}"
build --action_env TF_ROCM_AMDGPU_TARGETS="{rocm_amdgpu_targets}"
build --distinct_host_configuration=false
build:linux --copt=-Wno-sign-compare
build:macos --copt=-Wno-sign-compare
build -c opt
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:mkl_open_source_only --define=tensorflow_mkldnn_contraction_kernel=1

# Sets the default Apple platform to macOS.
build --apple_platform_type=macos
build --macos_minimum_os=10.9

# Make Bazel print out all options from rc files.
build --announce_rc

build --define open_source_build=true

# Disable enabled-by-default TensorFlow features that we don't care about.
build:linux --define=no_aws_support=true
build:macos --define=no_aws_support=true
build:linux --define=no_gcp_support=true
build:macos --define=no_gcp_support=true
build:linux --define=no_hdfs_support=true
build:macos --define=no_hdfs_support=true
build --define=no_kafka_support=true
build --define=no_ignite_support=true
build --define=grpc_no_ares=true

build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true

build:rocm --crosstool_top=@local_config_rocm//crosstool:toolchain
build:rocm --define=using_rocm=true --define=using_rocm_hipcc=true
build:nonccl --define=no_nccl_support=true

build --spawn_strategy=standalone
build --strategy=Genrule=standalone

build --enable_platform_specific_config

# Tensorflow uses M_* math constants that only get defined by MSVC headers if
# _USE_MATH_DEFINES is defined.
build:windows --copt=/D_USE_MATH_DEFINES
build:windows --host_copt=/D_USE_MATH_DEFINES

# Make sure to include as little of windows.h as possible
build:windows --copt=-DWIN32_LEAN_AND_MEAN
build:windows --host_copt=-DWIN32_LEAN_AND_MEAN
build:windows --copt=-DNOGDI
build:windows --host_copt=-DNOGDI

# https://devblogs.microsoft.com/cppblog/announcing-full-support-for-a-c-c-conformant-preprocessor-in-msvc/
# otherwise, there will be some compiling error due to preprocessing.
build:windows --copt=/Zc:preprocessor

build:linux --cxxopt=-std=c++14
build:linux --host_cxxopt=-std=c++14

build:macos --cxxopt=-std=c++14
build:macos --host_cxxopt=-std=c++14

build:windows --cxxopt=/std:c++14
build:windows --host_cxxopt=/std:c++14

# Generate PDB files, to generate useful PDBs, in opt compilation_mode
# --copt /Z7 is needed.
build:windows --linkopt=/DEBUG
build:windows --host_linkopt=/DEBUG
build:windows --linkopt=/OPT:REF
build:windows --host_linkopt=/OPT:REF
build:windows --linkopt=/OPT:ICF
build:windows --host_linkopt=/OPT:ICF
build:windows --experimental_strict_action_env=true

# Suppress all warning messages.
build:short_logs --output_filter=DONT_MATCH_ANYTHING

# Workaround for gcc 10+ warnings related to upb.
# See https://github.com/tensorflow/tensorflow/issues/39467
build:linux --copt=-Wno-stringop-truncation
"""



def write_bazelrc(cuda_toolkit_path=None, cudnn_install_path=None,
                  cuda_version=None, cudnn_version=None, rocm_toolkit_path=None, **kwargs):
  with open("../.bazelrc", "w") as f:
    f.write(BAZELRC_TEMPLATE.format(**kwargs))
    if cuda_toolkit_path:
      f.write("build --action_env CUDA_TOOLKIT_PATH=\"{cuda_toolkit_path}\"\n"
              .format(cuda_toolkit_path=cuda_toolkit_path))
    if cudnn_install_path:
      f.write("build --action_env CUDNN_INSTALL_PATH=\"{cudnn_install_path}\"\n"
              .format(cudnn_install_path=cudnn_install_path))
    if cuda_version:
      f.write("build --action_env TF_CUDA_VERSION=\"{cuda_version}\"\n"
              .format(cuda_version=cuda_version))
    if cudnn_version:
      f.write("build --action_env TF_CUDNN_VERSION=\"{cudnn_version}\"\n"
              .format(cudnn_version=cudnn_version))
    if rocm_toolkit_path:
      f.write("build --action_env ROCM_PATH=\"{rocm_toolkit_path}\"\n"
              .format(rocm_toolkit_path=rocm_toolkit_path))

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
    raise ValueError("Expected either 'true' or 'false'; got {}".format(s))


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


def main():
  cwd = os.getcwd()
  parser = argparse.ArgumentParser(
      description="Builds jaxlib from source.", epilog=EPILOG)
  parser.add_argument(
      "--bazel_path",
      help="Path to the Bazel binary to use. The default is to find bazel via "
      "the PATH; if none is found, downloads a fresh copy of bazel from "
      "GitHub.")
  parser.add_argument(
      "--python_bin_path",
      help="Path to Python binary to use. The default is the Python "
      "interpreter used to run the build script.")
  add_boolean_argument(
      parser,
      "enable_march_native",
      default=False,
      help_str="Generate code targeted to the current machine? This may "
          "increase performance, but may generate code that does not run on "
          "older machines.")
  add_boolean_argument(
      parser,
      "enable_mkl_dnn",
      default=True,
      help_str="Should we build with MKL-DNN enabled?")
  add_boolean_argument(
      parser,
      "enable_cuda",
      help_str="Should we build with CUDA enabled? Requires CUDA and CuDNN.")
  add_boolean_argument(
      parser,
      "enable_rocm",
      help_str="Should we build with ROCm enabled?")
  parser.add_argument(
      "--cuda_path",
      default=None,
      help="Path to the CUDA toolkit.")
  parser.add_argument(
      "--cudnn_path",
      default=None,
      help="Path to CUDNN libraries.")
  parser.add_argument(
      "--cuda_version",
      default=None,
      help="CUDA toolkit version, e.g., 11.1")
  parser.add_argument(
      "--cudnn_version",
      default=None,
      help="CUDNN version, e.g., 8")
  parser.add_argument(
      "--cuda_compute_capabilities",
      default="3.5,5.2,6.0,6.1,7.0",
      help="A comma-separated list of CUDA compute capabilities to support.")
  parser.add_argument(
      "--rocm_path",
      default=None,
      help="Path to the ROCm toolkit.")
  parser.add_argument(
      "--rocm_amdgpu_targets",
      default="gfx803,gfx900,gfx906,gfx1010",
      help="A comma-separated list of ROCm amdgpu targets to support.")
  parser.add_argument(
      "--bazel_startup_options",
      action="append", default=[],
      help="Additional startup options to pass to bazel.")
  parser.add_argument(
      "--bazel_options",
      action="append", default=[],
      help="Additional options to pass to bazel.")
  parser.add_argument(
      "--output_path",
      default=os.path.join(cwd, "dist"),
      help="Directory to which the jaxlib wheel should be written")
  args = parser.parse_args()

  if is_windows() and args.enable_cuda:
    if args.cuda_version is None:
      parser.error("--cuda_version is needed for Windows CUDA build.")
    if args.cudnn_version is None:
      parser.error("--cudnn_version is needed for Windows CUDA build.")

  if args.enable_cuda and args.enable_rocm:
    parser.error("--enable_cuda and --enable_rocm cannot be enabled at the same time.")

  print(BANNER)

  output_path = os.path.abspath(args.output_path)
  os.chdir(os.path.dirname(__file__ or args.prog) or '.')

  # Find a working Bazel.
  bazel_path = get_bazel_path(args.bazel_path)
  print("Bazel binary path: {}".format(bazel_path))

  python_bin_path = get_python_bin_path(args.python_bin_path)
  print("Python binary path: {}".format(python_bin_path))
  python_version = get_python_version(python_bin_path)
  print("Python version: {}".format(".".join(map(str, python_version))))
  check_python_version(python_version)

  print("MKL-DNN enabled: {}".format("yes" if args.enable_mkl_dnn else "no"))
  print("-march=native: {}".format("yes" if args.enable_march_native else "no"))

  cuda_toolkit_path = args.cuda_path
  cudnn_install_path = args.cudnn_path
  rocm_toolkit_path = args.rocm_path
  print("CUDA enabled: {}".format("yes" if args.enable_cuda else "no"))
  if args.enable_cuda:
    if cuda_toolkit_path:
      print("CUDA toolkit path: {}".format(cuda_toolkit_path))
    if cudnn_install_path:
      print("CUDNN library path: {}".format(cudnn_install_path))
    print("CUDA compute capabilities: {}".format(args.cuda_compute_capabilities))
    if args.cuda_version:
      print("CUDA version: {}".format(args.cuda_version))
    if args.cudnn_version:
      print("CUDNN version: {}".format(args.cudnn_version))

  print("ROCm enabled: {}".format("yes" if args.enable_rocm else "no"))
  if args.enable_rocm:
    if rocm_toolkit_path:
      print("ROCm toolkit path: {}".format(rocm_toolkit_path))
      print("ROCm amdgpu targets: {}".format(args.rocm_amdgpu_targets))

  write_bazelrc(
      python_bin_path=python_bin_path,
      tf_need_cuda=1 if args.enable_cuda else 0,
      tf_need_rocm=1 if args.enable_rocm else 0,
      cuda_toolkit_path=cuda_toolkit_path,
      cudnn_install_path=cudnn_install_path,
      cuda_compute_capabilities=args.cuda_compute_capabilities,
      cuda_version=args.cuda_version,
      cudnn_version=args.cudnn_version,
      rocm_toolkit_path=rocm_toolkit_path,
      rocm_amdgpu_targets=args.rocm_amdgpu_targets,
)

  print("\nBuilding XLA and installing it in the jaxlib source tree...")
  config_args = args.bazel_options
  config_args += ["--config=short_logs"]
  if args.enable_march_native:
    config_args += ["--config=opt"]
  if args.enable_mkl_dnn:
    config_args += ["--config=mkl_open_source_only"]
  if args.enable_cuda:
    config_args += ["--config=cuda"]
    config_args += ["--define=xla_python_enable_gpu=true"]
  if args.enable_rocm:
    config_args += ["--config=rocm"]
    config_args += ["--config=nonccl"]
    config_args += ["--define=xla_python_enable_gpu=true"]
  command = ([bazel_path] + args.bazel_startup_options +
    ["run", "--verbose_failures=true"] + config_args +
    [":build_wheel", "--",
    f"--output_path={output_path}"])
  print(" ".join(command))
  shell(command)
  shell([bazel_path, "shutdown"])


if __name__ == "__main__":
  main()
