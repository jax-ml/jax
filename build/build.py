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
import textwrap
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
  try:
    output = subprocess.check_output(cmd)
  except subprocess.CalledProcessError as e:
    print(e.output)
    raise
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
  if python_version < (3, 7):
    print("ERROR: JAX requires Python 3.7 or newer, found ", python_version)
    sys.exit(-1)


def check_numpy_version(python_bin_path):
  version = shell(
      [python_bin_path, "-c", "import numpy as np; print(np.__version__)"])
  numpy_version = tuple(map(int, version.split(".")[:2]))
  if numpy_version < (1, 18):
    print("ERROR: JAX requires NumPy 1.18 or newer, found " + version + ".")
    sys.exit(-1)
  return version

def check_scipy_version(python_bin_path):
  version = shell(
      [python_bin_path, "-c", "import scipy as sp; print(sp.__version__)"])
  scipy_version = tuple(map(int, version.split(".")[:2]))
  if scipy_version < (1, 0):
    print("ERROR: JAX requires SciPy 1.0 or newer, found " + version + ".")
    sys.exit(-1)
  return version

# Bazel

BAZEL_BASE_URI = "https://github.com/bazelbuild/bazel/releases/download/4.1.0/"
BazelPackage = collections.namedtuple("BazelPackage",
                                      ["base_uri", "file", "sha256"])
bazel_packages = {
    ("Linux", "x86_64"):
        BazelPackage(
            base_uri=None,
            file="bazel-4.1.0-linux-x86_64",
            sha256=
            "0eb2e378d2782e7810753e2162245ad1179c1bb12f848c692b4a595b4edf779b"),
    ("Linux", "aarch64"):
        BazelPackage(
            base_uri=None,
            file="bazel-4.1.0-linux-arm64",
            sha256=
            "b3834742166379e52b880319dec4699082cb26fa96cbb783087deedc5fbb5f2b"),
    ("Darwin", "x86_64"):
        BazelPackage(
            base_uri=None,
            file="bazel-4.1.0-darwin-x86_64",
            sha256=
            "2eecc3abb0ff653ed0bffdb9fbfda7b08548c2868f13da4a995f01528db200a9"),
    ("Darwin", "arm64"):
        BazelPackage(
            base_uri=None,
            file="bazel-4.1.0-darwin-arm64",
            sha256=
            "c372d39ab9dac96f7fdfc2dd649e88b05ee4c94ce3d6cf2313438ef0ca6d5ac1"),
    ("Windows", "AMD64"):
        BazelPackage(
            base_uri=None,
            file="bazel-4.1.0-windows-x86_64.exe",
            sha256=
            "7b2077af7055b421fe31822f83c3c3c15e36ff39b69560ba2472dde92dd45b46"),
}


def download_and_verify_bazel():
  """Downloads a bazel binary from Github, verifying its SHA256 hash."""
  package = bazel_packages.get((platform.system(), platform.machine()))
  if package is None:
    return None

  if not os.access(package.file, os.X_OK):
    uri = (package.base_uri or BAZEL_BASE_URI) + package.file
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
    version_output = shell([bazel_path, "--version"])
  except subprocess.CalledProcessError:
    return False
  match = re.search(r"bazel *([0-9\\.]+)", version_output)
  if match is None:
    return False
  actual_ints = [int(x) for x in match.group(1).split(".")]
  return actual_ints >= [2, 0, 0]


def write_bazelrc(python_bin_path=None, remote_build=None,
                  cuda_toolkit_path=None, cudnn_install_path=None,
                  cuda_version=None, cudnn_version=None, rocm_toolkit_path=None,
                  cpu=None):
  tf_cuda_paths = []

  with open("../.jax_configure.bazelrc", "w") as f:
    if not remote_build and python_bin_path:
      f.write(textwrap.dedent("""\
        build --repo_env PYTHON_BIN_PATH="{python_bin_path}"
        build --action_env=PYENV_ROOT
        build --python_path="{python_bin_path}"
        """).format(python_bin_path=python_bin_path))

    if cuda_toolkit_path:
      tf_cuda_paths.append(cuda_toolkit_path)
      f.write("build --action_env CUDA_TOOLKIT_PATH=\"{cuda_toolkit_path}\"\n"
              .format(cuda_toolkit_path=cuda_toolkit_path))
    if cudnn_install_path:
      # see https://github.com/tensorflow/tensorflow/issues/51040
      if cudnn_install_path not in tf_cuda_paths:
        tf_cuda_paths.append(cudnn_install_path)
      f.write("build --action_env CUDNN_INSTALL_PATH=\"{cudnn_install_path}\"\n"
              .format(cudnn_install_path=cudnn_install_path))
    if len(tf_cuda_paths):
      f.write("build --action_env TF_CUDA_PATHS=\"{tf_cuda_paths}\"\n"
              .format(tf_cuda_paths=",".join(tf_cuda_paths)))
    if cuda_version:
      f.write("build --action_env TF_CUDA_VERSION=\"{cuda_version}\"\n"
              .format(cuda_version=cuda_version))
    if cudnn_version:
      f.write("build --action_env TF_CUDNN_VERSION=\"{cudnn_version}\"\n"
              .format(cudnn_version=cudnn_version))
    if rocm_toolkit_path:
      f.write("build --action_env ROCM_PATH=\"{rocm_toolkit_path}\"\n"
              .format(rocm_toolkit_path=rocm_toolkit_path))
    if cpu is not None:
      f.write("build --distinct_host_configuration=true\n")
      f.write(f"build --cpu={cpu}\n")
    else:
      f.write("build --distinct_host_configuration=false\n")


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
      "enable_mkl_dnn",
      default=True,
      help_str="Should we build with MKL-DNN enabled?")
  add_boolean_argument(
      parser,
      "enable_cuda",
      help_str="Should we build with CUDA enabled? Requires CUDA and CuDNN.")
  add_boolean_argument(
      parser,
      "enable_tpu",
      help_str="Should we build with Cloud TPU support enabled?")
  add_boolean_argument(
      parser,
      "enable_rocm",
      help_str="Should we build with ROCm enabled?")
  add_boolean_argument(
      parser,
      "enable_nccl",
      default=True,
      help_str="Should we build with NCCL enabled? Has non effect for non-CUDA "
               "builds.")
  add_boolean_argument(
      parser,
      "remote_build",
      default=False,
      help_str="Should we build with RBE.")
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
  parser.add_argument(
      "--target_cpu",
      default=None,
      help="CPU platform to target. Default is the same as the host machine. "
           "Currently supported values are 'darwin_arm64' and 'darwin_x86_64'.")
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

  host_cpu = platform.machine()
  wheel_cpus = {
      "darwin_arm64": "arm64",
      "darwin_x86_64": "x86_64",
      "ppc": "ppc64le",
  }
  # TODO(phawkins): support other bazel cpu overrides.
  wheel_cpu = (wheel_cpus[args.target_cpu] if args.target_cpu is not None
               else host_cpu)

  # Find a working Bazel.
  bazel_path = get_bazel_path(args.bazel_path)
  print("Bazel binary path: {}".format(bazel_path))

  python_bin_path = get_python_bin_path(args.python_bin_path)
  print("Python binary path: {}".format(python_bin_path))
  python_version = get_python_version(python_bin_path)
  print("Python version: {}".format(".".join(map(str, python_version))))
  check_python_version(python_version)

  numpy_version = check_numpy_version(python_bin_path)
  print("NumPy version: {}".format(numpy_version))
  scipy_version = check_scipy_version(python_bin_path)
  print("SciPy version: {}".format(scipy_version))

  print("MKL-DNN enabled: {}".format("yes" if args.enable_mkl_dnn else "no"))
  print("Target CPU: {}".format(wheel_cpu))
  print("Target CPU features: {}".format(args.target_cpu_features))

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
    print("NCCL enabled: {}".format("yes" if args.enable_nccl else "no"))

  print("TPU enabled: {}".format("yes" if args.enable_tpu else "no"))

  print("ROCm enabled: {}".format("yes" if args.enable_rocm else "no"))
  if args.enable_rocm:
    if rocm_toolkit_path:
      print("ROCm toolkit path: {}".format(rocm_toolkit_path))
      print("ROCm amdgpu targets: {}".format(args.rocm_amdgpu_targets))

  write_bazelrc(
      python_bin_path=python_bin_path,
      remote_build=args.remote_build,
      cuda_toolkit_path=cuda_toolkit_path,
      cudnn_install_path=cudnn_install_path,
      cuda_version=args.cuda_version,
      cudnn_version=args.cudnn_version,
      rocm_toolkit_path=rocm_toolkit_path,
      cpu=args.target_cpu,
  )

  print("\nBuilding XLA and installing it in the jaxlib source tree...")

  config_args = args.bazel_options
  if args.target_cpu_features == "release":
    if wheel_cpu == "x86_64":
      config_args += ["--config=avx_windows" if is_windows()
                      else "--config=avx_posix"]
  elif args.target_cpu_features == "native":
    if is_windows():
      print("--target_cpu_features=native is not supported on Windows; ignoring.")
    else:
      config_args += ["--config=native_arch_posix"]

  if args.enable_mkl_dnn:
    config_args += ["--config=mkl_open_source_only"]
  if args.enable_cuda:
    config_args += ["--config=cuda"]
    if not args.enable_nccl:
      config_args += ["--config=nonccl"]
  if args.enable_tpu:
    config_args += ["--config=tpu"]
  if args.enable_rocm:
    config_args += ["--config=rocm"]
    config_args += ["--config=nonccl"]

  command = ([bazel_path] + args.bazel_startup_options +
    ["run", "--verbose_failures=true"] + config_args +
    [":build_wheel", "--",
    f"--output_path={output_path}",
    f"--cpu={wheel_cpu}"])
  print(" ".join(command))
  shell(command)
  shell([bazel_path, "shutdown"])


if __name__ == "__main__":
  main()
