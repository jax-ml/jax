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
import functools
import os
import pathlib
import shutil
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from jaxlib.tools import build_utils

parser = argparse.ArgumentParser()
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
args = parser.parse_args()

r = runfiles.Create()


def prepare_wheel(sources_path: pathlib.Path):
  """Assembles a source tree for the wheel in `sources_path`."""
  copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

  copy_runfiles(
      dst_dir=sources_path,
      src_files=[
          "__main__/LICENSE",
          "__main__/setup.py",
          "__main__/AUTHORS",
          "__main__/README.md",
          "__main__/pyproject.toml",
      ],
  )
  version_runfiles_location = r.Rlocation("__main__/jax/version.py")
  jax_dir_runfiles_location = version_runfiles_location[
      : version_runfiles_location.rfind("/")
  ]
  shutil.copytree(
      jax_dir_runfiles_location,
      sources_path / "jax",
      dirs_exist_ok=True,
  )


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jax")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel(
      pathlib.Path(sources_path),
  )
  build_utils.build_wheel(
      sources_path,
      args.output_path,
      package_name="jax",
      git_hash=args.jaxlib_git_hash,
  )
finally:
  if tmpdir:
    tmpdir.cleanup()
