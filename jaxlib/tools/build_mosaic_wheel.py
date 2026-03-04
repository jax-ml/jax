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

# Script that builds a jax cuda/rocm plugin wheel, intended to be run via bazel run
# as part of the jax cuda/rocm plugin build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import tempfile

from python.runfiles import runfiles
from jaxlib.tools import build_utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--sources_path",
    default=None,
    help="Path in which the wheel's sources should be prepared. Optional. If "
    "omitted, a temporary directory will be used.",
)
parser.add_argument(
    "--output_path",
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--jaxlib_git_hash",
    required=True,
    help="Git hash. Required.",
)
parser.add_argument(
    "--cpu", required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--platform_version",
    required=True,
    help="Target CUDA version. Required.",
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' mosaic build instead of a wheel.",
)
parser.add_argument(
    "--srcs", help="source files for the wheel", action="append"
)
parser.add_argument(
    "--nvidia_wheel_versions_data",
    default=None,
    required=True,
    help="NVIDIA wheel versions data",
)

# The jax_wheel target passes in some extra params, which we ignore
args, _ = parser.parse_known_args()

r = runfiles.Create()


def assemble_sources(
    wheel_sources_path: pathlib.Path,
    *,
    cpu,
    cuda_version,
    wheel_sources,
    nvidia_wheel_versions_data,
):
  """Assembles a source tree for the wheel in `wheel_sources_path`"""
  source_file_prefix = build_utils.get_source_file_prefix(wheel_sources)
  wheel_sources_map = build_utils.create_wheel_sources_map(
      wheel_sources, root_packages=["jaxlib"]
  )
  mgpudir = wheel_sources_path / "mosaic_gpu"

  copy_files = functools.partial(
      build_utils.copy_file,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )

  copy_files(
      dst_dir=wheel_sources_path,
      src_files=[
          f"{source_file_prefix}jaxlib/tools/LICENSE.txt",
          f"{source_file_prefix}jaxlib/mosaic/gpu/wheel/setup.py",
      ],
  )

  copy_files(
      dst_dir=mgpudir / f"mosaic_gpu_cuda{cuda_version}",
      src_files=[
          f"{source_file_prefix}jaxlib/mosaic/gpu/wheel/mosaic_gpu.so",
          f"{source_file_prefix}jaxlib/mosaic/gpu/wheel/__init__.py",
          f"{source_file_prefix}jaxlib/version.py",
      ],
  )

  # This sets the cuda version in setup.py
  build_utils.update_setup_with_cuda_and_nvidia_wheel_versions(
      wheel_sources_path, cuda_version, nvidia_wheel_versions_data
  )

  tag = build_utils.platform_tag(cpu)
  with open(wheel_sources_path / "setup.cfg", "w") as f:
    f.write(
        f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
python_tag=py3
"""
    )


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="mosaic_gpu")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  package_name = "mosaic_gpu"

  assemble_sources(
      pathlib.Path(sources_path),
      cpu=args.cpu,
      cuda_version=args.platform_version,
      wheel_sources=args.srcs,
      nvidia_wheel_versions_data=args.nvidia_wheel_versions_data,
  )
  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    build_utils.build_wheel(
        sources_path,
        args.output_path,
        package_name,
        git_hash=args.jaxlib_git_hash,
    )
finally:
  if tmpdir:
    tmpdir.cleanup()
