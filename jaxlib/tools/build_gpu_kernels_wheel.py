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

# Script that builds a jax-cuda12-plugin wheel for cuda kernels, intended to be
# run via bazel run as part of the jax cuda plugin build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from jaxlib.tools import build_utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
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
    "--cpu", default=None, required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--platform_version",
    default=None,
    required=True,
    help="Target CUDA/ROCM version. Required.",
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' jax cuda/rocm plugin build instead of a wheel.",
)
parser.add_argument(
    "--enable-cuda",
    default=False,
    help="Should we build with CUDA enabled? Requires CUDA and CuDNN.")
parser.add_argument(
    "--enable-rocm",
    default=False,
    help="Should we build with ROCM enabled?")
parser.add_argument(
    "--srcs", help="source files for the wheel", action="append"
)
args = parser.parse_args()

r = runfiles.Create()
pyext = "pyd" if build_utils.is_windows() else "so"


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(sources_path / "setup.cfg", "w") as f:
    f.write(f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
""")


def prepare_wheel_cuda(
    wheel_sources_path: pathlib.Path, *, cpu, cuda_version, wheel_sources
):
  """Assembles a source tree for the cuda kernel wheel in `wheel_sources_path`."""
  source_file_prefix = build_utils.get_source_file_prefix(wheel_sources)
  wheel_sources_map = build_utils.create_wheel_sources_map(
      wheel_sources,
      root_packages=[
          "jax_plugins",
          f"jax_cuda{cuda_version}_plugin",
          "jaxlib",
      ],
  )
  copy_files = functools.partial(
      build_utils.copy_file,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )

  copy_files(
      f"{source_file_prefix}jax_plugins/cuda/plugin_pyproject.toml",
      dst_dir=wheel_sources_path,
      dst_filename="pyproject.toml",
  )
  copy_files(
      f"{source_file_prefix}jax_plugins/cuda/plugin_setup.py",
      dst_dir=wheel_sources_path,
      dst_filename="setup.py",
  )
  build_utils.update_setup_with_cuda_version(wheel_sources_path, cuda_version)
  write_setup_cfg(wheel_sources_path, cpu)

  plugin_dir = wheel_sources_path / f"jax_cuda{cuda_version}_plugin"
  copy_files(
      dst_dir=plugin_dir,
      src_files=[
          f"{source_file_prefix}jaxlib/cuda/_solver.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/_linalg.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/_prng.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/_rnn.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/_sparse.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/_triton.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/_hybrid.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/_versions.{pyext}",
          f"{source_file_prefix}jaxlib/cuda/cuda_plugin_extension.{pyext}",
          f"{source_file_prefix}jaxlib/mosaic/gpu/_mosaic_gpu_ext.{pyext}",
          f"{source_file_prefix}jaxlib/mosaic/gpu/libmosaic_gpu_runtime.so",
          f"{source_file_prefix}jaxlib/version.py",
      ],
  )


def prepare_wheel_rocm(
    wheel_sources_path: pathlib.Path, *, cpu, rocm_version, wheel_sources
):
  """Assembles a source tree for the rocm kernel wheel in `wheel_sources_path`."""
  source_file_prefix = build_utils.get_source_file_prefix(wheel_sources)
  wheel_sources_map = build_utils.create_wheel_sources_map(
      wheel_sources,
      root_packages=[
          "jax_plugins",
          f"jax_rocm{rocm_version}_plugin",
          "jaxlib",
      ],
  )
  copy_files = functools.partial(
      build_utils.copy_file,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )

  copy_files(
      f"{source_file_prefix}jax_plugins/rocm/plugin_pyproject.toml",
      dst_dir=wheel_sources_path,
      dst_filename="pyproject.toml",
  )
  copy_files(
      f"{source_file_prefix}jax_plugins/rocm/plugin_setup.py",
      dst_dir=wheel_sources_path,
      dst_filename="setup.py",
  )
  build_utils.update_setup_with_rocm_version(wheel_sources_path, rocm_version)
  write_setup_cfg(wheel_sources_path, cpu)

  plugin_dir = wheel_sources_path / f"jax_rocm{rocm_version}_plugin"
  copy_files(
      dst_dir=plugin_dir,
      src_files=[
          f"{source_file_prefix}jaxlib/rocm/_linalg.{pyext}",
          f"{source_file_prefix}jaxlib/rocm/_prng.{pyext}",
          f"{source_file_prefix}jaxlib/rocm/_solver.{pyext}",
          f"{source_file_prefix}jaxlib/rocm/_sparse.{pyext}",
          f"{source_file_prefix}jaxlib/rocm/_hybrid.{pyext}",
          f"{source_file_prefix}jaxlib/rocm/_rnn.{pyext}",
          f"{source_file_prefix}jaxlib/rocm/_triton.{pyext}",
          f"{source_file_prefix}jaxlib/rocm/rocm_plugin_extension.{pyext}",
          f"{source_file_prefix}jaxlib/version.py",
      ],
  )


# Build wheel for cuda kernels
if args.enable_rocm:
  tmpdir = tempfile.TemporaryDirectory(prefix="jax_rocm_plugin")
else:
  tmpdir = tempfile.TemporaryDirectory(prefix="jax_cuda_plugin")
sources_path = tmpdir.name
try:
  os.makedirs(args.output_path, exist_ok=True)
  if args.enable_cuda:
    prepare_wheel_cuda(
        pathlib.Path(sources_path),
        cpu=args.cpu,
        cuda_version=args.platform_version,
        wheel_sources=args.srcs,
    )
    package_name = f"jax cuda{args.platform_version} plugin"
  elif args.enable_rocm:
    prepare_wheel_rocm(
        pathlib.Path(sources_path),
        cpu=args.cpu,
        rocm_version=args.platform_version,
        wheel_sources=args.srcs,
    )
    package_name = f"jax rocm{args.platform_version} plugin"
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
  tmpdir.cleanup()
