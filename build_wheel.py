# Copyright 2025 The JAX Authors.
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

# Script that builds a JAX wheel, intended to be run via bazel run as part
# of the JAX build process.

import argparse
import os
import pathlib
import shutil
import tempfile

from jaxlib.tools import build_utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--sources_path",
    default=None,
    help=(
        "Path in which the wheel's sources should be prepared. Optional. If "
        "omitted, a temporary directory will be used."
    ),
)
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--jaxlib_git_hash",
    default="",
    required=True,
    help="Git hash. Empty if unknown. Optional.",
)
parser.add_argument(
    "--srcs", help="source files for the wheel", action="append"
)
args = parser.parse_args()


def copy_file(
    src_file: str,
    dst_dir: str,
) -> None:
  """Copy a file to the destination directory.

  Args:
    src_file: file to be copied
    dst_dir: destination directory
  """

  dest_dir_path = os.path.join(dst_dir, os.path.dirname(src_file))
  os.makedirs(dest_dir_path, exist_ok=True)
  shutil.copy(src_file, dest_dir_path)
  os.chmod(os.path.join(dst_dir, src_file), 0o644)


def prepare_srcs(deps: list[str], srcs_dir: str) -> None:
  """Filter the sources and copy them to the destination directory.

  Args:
    deps: a list of paths to files.
    srcs_dir: target directory where files are copied to.
  """

  for file in deps:
    if not (file.startswith("bazel-out") or file.startswith("external")):
      copy_file(file, srcs_dir)


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jax")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_srcs(args.srcs, pathlib.Path(sources_path))
  build_utils.build_wheel(
      sources_path,
      args.output_path,
      package_name="jax",
      git_hash=args.jaxlib_git_hash,
      build_wheel_only=False,
  )
finally:
  if tmpdir:
    tmpdir.cleanup()
