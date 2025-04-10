# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Helper script for tools/utilities used by the JAX build CLI.
import collections
import glob
import hashlib
import logging
import os
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib.request

logger = logging.getLogger(__name__)

BAZEL_BASE_URI = "https://github.com/bazelbuild/bazel/releases/download/7.4.1/"
BazelPackage = collections.namedtuple(
    "BazelPackage", ["base_uri", "file", "sha256"]
)
bazel_packages = {
    ("Linux", "x86_64"): BazelPackage(
        base_uri=None,
        file="bazel-7.4.1-linux-x86_64",
        sha256=(
            "c97f02133adce63f0c28678ac1f21d65fa8255c80429b588aeeba8a1fac6202b"
        ),
    ),
    ("Linux", "aarch64"): BazelPackage(
        base_uri=None,
        file="bazel-7.4.1-linux-arm64",
        sha256=(
            "d7aedc8565ed47b6231badb80b09f034e389c5f2b1c2ac2c55406f7c661d8b88"
        ),
    ),
    ("Darwin", "x86_64"): BazelPackage(
        base_uri=None,
        file="bazel-7.4.1-darwin-x86_64",
        sha256=(
            "52dd34c17cc97b3aa5bdfe3d45c4e3938226f23dd0bfb47beedd625a953f1f05"
        ),
    ),
    ("Darwin", "arm64"): BazelPackage(
        base_uri=None,
        file="bazel-7.4.1-darwin-arm64",
        sha256=(
            "02b117b97d0921ae4d4f4e11d27e2c0930381df416e373435d5d0419c6a26f24"
        ),
    ),
    ("Windows", "AMD64"): BazelPackage(
        base_uri=None,
        file="bazel-7.4.1-windows-x86_64.exe",
        sha256=(
            "4a76eddf6c5115e1d93355fd11db5ac2fc20e58f197f5d65d3f21da92aa0925b"
        ),
    ),
}

def download_and_verify_bazel():
  """Downloads a bazel binary from GitHub, verifying its SHA256 hash."""
  package = bazel_packages.get((platform.system(), platform.machine()))
  if package is None:
    return None

  if not os.access(package.file, os.X_OK):
    uri = (package.base_uri or BAZEL_BASE_URI) + package.file
    sys.stdout.write(f"Downloading bazel from: {uri}\n")

    def progress(block_count, block_size, total_size):
      if total_size <= 0:
        total_size = 170**6
      progress = (block_count * block_size) / total_size
      num_chars = 40
      progress_chars = int(num_chars * progress)
      sys.stdout.write(
          "{} [{}{}] {}%\r".format(
              package.file,
              "#" * progress_chars,
              "." * (num_chars - progress_chars),
              int(progress * 100.0),
          )
      )

    tmp_path, _ = urllib.request.urlretrieve(
        uri, None, progress if sys.stdout.isatty() else None
    )
    sys.stdout.write("\n")

    # Verify that the downloaded Bazel binary has the expected SHA256.
    with open(tmp_path, "rb") as downloaded_file:
      contents = downloaded_file.read()

    digest = hashlib.sha256(contents).hexdigest()
    if digest != package.sha256:
      print(
          "Checksum mismatch for downloaded bazel binary (expected {}; got {})."
          .format(package.sha256, digest)
      )
      sys.exit(-1)

    # Write the file as the bazel file name.
    with open(package.file, "wb") as out_file:
      out_file.write(contents)

    # Mark the file as executable.
    st = os.stat(package.file)
    os.chmod(
        package.file, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

  return os.path.join(".", package.file)

def get_bazel_paths(bazel_path_flag):
  """Yields a sequence of guesses about bazel path.

  Some of sequence elements can be None. The resulting iterator is lazy and
  potentially has a side effects.
  """
  yield bazel_path_flag
  yield shutil.which("bazel")
  yield download_and_verify_bazel()

def get_bazel_path(bazel_path_flag):
  """Returns the path to a Bazel binary, downloading Bazel if not found.

  Also, checks Bazel's version is at least newer than 7.4.1

  A manual version check is needed only for really old bazel versions.
  Newer bazel releases perform their own version check against .bazelversion
  (see for details
  https://blog.bazel.build/2019/12/19/bazel-2.0.html#other-important-changes).
  """
  for path in filter(None, get_bazel_paths(bazel_path_flag)):
    version = get_bazel_version(path)
    if version is not None and version >= (6, 5, 0):
      return path, ".".join(map(str, version))

  print(
      "Cannot find or download a suitable version of bazel."
      "Please install bazel >= 7.4.1."
  )
  sys.exit(-1)

def get_bazel_version(bazel_path):
  try:
    version_output = subprocess.run(
        [bazel_path, "--version"],
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout.strip()
  except (subprocess.CalledProcessError, OSError):
    return None
  match = re.search(r"bazel *([0-9\\.]+)", version_output)
  if match is None:
    return None
  return tuple(int(x) for x in match.group(1).split("."))

def get_compiler_path_or_exit(compiler_path_flag, compiler_name):
  which_compiler_output = shutil.which(compiler_name)
  if which_compiler_output:
    # If we've found a compiler on the path, need to get the fully resolved path
    # to ensure that system headers are found.
    return str(pathlib.Path(which_compiler_output).resolve())
  else:
    print(
        f"--{compiler_path_flag} is unset and {compiler_name} cannot be found"
        f" on the PATH. Please pass --{compiler_path_flag} to the build script."
    )
    sys.exit(-1)

def get_gcc_path_or_exit():
  return get_compiler_path_or_exit("gcc_path", "gcc")

def get_clang_path_or_exit():
  return get_compiler_path_or_exit("clang_path", "clang")

def get_clang_major_version(clang_path):
  clang_version_proc = subprocess.run(
      [clang_path, "-E", "-P", "-"],
      input="__clang_major__",
      check=True,
      capture_output=True,
      text=True,
  )
  major_version = int(clang_version_proc.stdout)

  return major_version

def get_clangpp_path(clang_path):
  clang_path = pathlib.Path(clang_path)
  clang_exec_name = clang_path.name
  clangpp_exec_name = clang_exec_name
  if "clang++" not in clang_exec_name:
    clangpp_exec_name = clang_exec_name.replace("clang", "clang++")
  clangpp_path = clang_path.parent / clangpp_exec_name
  if not clangpp_path.exists():
    raise FileNotFoundError(
      f"Failed to get clang++ path from clang path: '{clang_path!s}'. "
      f"Tried the path: '{clangpp_path!s}'."
    )
  return str(clangpp_path)

def get_gcc_major_version(gcc_path: str):
  gcc_version_proc = subprocess.run(
    [gcc_path, "-dumpversion"],
    check=True,
    capture_output=True,
    text=True,
  )
  major_version = int(gcc_version_proc.stdout.split(".")[0])

  return major_version


def get_jax_configure_bazel_options(bazel_command: list[str], use_new_wheel_build_rule: bool):
  """Returns the bazel options to be written to .jax_configure.bazelrc."""
  # Get the index of the "run" parameter. Build options will come after "run" so
  # we find the index of "run" and filter everything after it. If we are using
  # the new wheel build rule, we will find the index of "build" instead.
  if use_new_wheel_build_rule:
    start = bazel_command.index("build")
  else:
    start = bazel_command.index("run")
  jax_configure_bazel_options = ""
  try:
    for i in range(start + 1, len(bazel_command)):
      bazel_flag = bazel_command[i]
      # On Windows, replace all backslashes with double backslashes to avoid
      # unintended escape sequences.
      if platform.system() == "Windows":
        bazel_flag = bazel_flag.replace("\\", "\\\\")
      jax_configure_bazel_options += f"build {bazel_flag}\n"
    return jax_configure_bazel_options
  except ValueError:
    logging.error("Unable to find index for 'run' in the Bazel command")
    return ""

def get_githash():
  try:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout.strip()
  except (subprocess.CalledProcessError, OSError):
    return ""

def _parse_string_as_bool(s):
  """Parses a string as a boolean value."""
  lower = s.lower()
  if lower == "true":
    return True
  elif lower == "false":
    return False
  else:
    raise ValueError(f"Expected either 'true' or 'false'; got {s}")


def copy_dir_recursively(src, dst):
  if os.path.exists(dst):
    shutil.rmtree(dst)
  os.makedirs(dst, exist_ok=True)
  for root, dirs, files in os.walk(src):
    relative_path = os.path.relpath(root, src)
    dst_dir = os.path.join(dst, relative_path)
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
      src_file = os.path.join(root, f)
      dst_file = os.path.join(dst_dir, f)
      shutil.copy2(src_file, dst_file)
  logging.info("Editable wheel path: %s" % dst)


def copy_individual_files(src, dst, regex):
  os.makedirs(dst, exist_ok=True)
  for f in glob.glob(os.path.join(src, regex)):
    dst_file = os.path.join(dst, os.path.basename(f))
    if os.path.exists(dst_file):
      os.remove(dst_file)
    shutil.copy2(f, dst_file)
    logging.info("Distribution path: %s" % dst_file)
