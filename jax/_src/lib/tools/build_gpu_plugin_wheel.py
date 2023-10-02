# Copyright 2023 The JAX Authors.
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

# Script that builds a jax cuda plugin wheel, intended to be run via bazel run
# as part of the jax cuda plugin build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from jax.tools import build_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sources_path",
    default=None,
    help="Path in which the wheel's sources should be prepared. Optional. If "
    "omitted, a temporary directory will be used.",
)
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--cpu", default=None, required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--cuda_version",
    default=None,
    required=True,
    help="Target CUDA version. Required.",
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' jax cuda plugin build instead of a wheel.",
)
args = parser.parse_args()

r = runfiles.Create()


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(sources_path / "setup.cfg", "w") as f:
    f.write(
        f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat-name={tag}
python-tag=py3
"""
    )


def update_setup(file_dir, cuda_version):
  src_file = file_dir / "setup.py"
  with open(src_file, "r") as f:
    content = f.read()
  content = content.replace(
      "cuda_version = 0  # placeholder", f"cuda_version = {cuda_version}"
  )
  with open(src_file, "w") as f:
    f.write(content)


def prepare_cuda_plugin_wheel(sources_path: pathlib.Path, *, cpu, cuda_version):
  """Assembles a source tree for the wheel in `sources_path`."""
  copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

  plugin_dir = sources_path / "jax_plugins" / f"xla_cuda_cu{cuda_version}"
  copy_runfiles(
      dst_dir=sources_path,
      src_files=[
          "__main__/plugins/cuda/pyproject.toml",
          "__main__/plugins/cuda/setup.py",
      ],
  )
  update_setup(sources_path, cuda_version)
  write_setup_cfg(sources_path, cpu)
  copy_runfiles(
      dst_dir=plugin_dir,
      src_files=[
          "__main__/plugins/cuda/__init__.py",
      ],
  )
  copy_runfiles(
      "xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so",
      dst_dir=plugin_dir,
      dst_filename="xla_cuda_plugin.so",
  )


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jaxcudaplugin")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_cuda_plugin_wheel(
      pathlib.Path(sources_path), cpu=args.cpu, cuda_version=args.cuda_version
  )
  package_name = "jax cuda plugin"
  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    build_utils.build_wheel(sources_path, args.output_path, package_name)
finally:
  if tmpdir:
    tmpdir.cleanup()
