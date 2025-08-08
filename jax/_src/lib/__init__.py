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

# This module is largely a wrapper around `jaxlib` that performs version
# checking on import.

from __future__ import annotations

import importlib
import gc
import os
import pathlib
import re
from types import ModuleType
import typing

try:
  import jaxlib as jaxlib
except ModuleNotFoundError as err:
  raise ModuleNotFoundError(
    'jax requires jaxlib to be installed. See '
    'https://github.com/jax-ml/jax#installation for installation instructions.'
    ) from err

import jax.version
from jax.version import _minimum_jaxlib_version as _minimum_jaxlib_version_str
try:
  import jaxlib.version
except Exception as err:
  # jaxlib is too old to have version number.
  msg = f'This version of jax requires jaxlib version >= {_minimum_jaxlib_version_str}.'
  raise ImportError(msg) from err


# Checks the jaxlib version before importing anything else.
# Returns the jaxlib version string.
def check_jaxlib_version(jax_version: str, jaxlib_version: str,
                         minimum_jaxlib_version: str) -> tuple[int, ...]:
  # Regex to match a dotted version prefix 0.1.23.456.789 of a PEP440 version.
  # PEP440 allows a number of non-numeric suffixes, which we allow also.
  # We currently do not allow an epoch.
  version_regex = re.compile(r"[0-9]+(?:\.[0-9]+)*")
  def _parse_version(v: str) -> tuple[int, ...]:
    m = version_regex.match(v)
    if m is None:
      raise ValueError(f"Unable to parse jaxlib version '{v}'")
    return tuple(int(x) for x in m.group(0).split('.'))

  _jax_version = _parse_version(jax_version)
  _minimum_jaxlib_version = _parse_version(minimum_jaxlib_version)
  _jaxlib_version = _parse_version(jaxlib_version)

  if _jaxlib_version < _minimum_jaxlib_version:
    msg = (f'jaxlib is version {jaxlib_version}, but this version '
           f'of jax requires version >= {minimum_jaxlib_version}.')
    raise RuntimeError(msg)

  if _jaxlib_version > _jax_version:
    raise RuntimeError(
        f'jaxlib version {jaxlib_version} is newer than and '
        f'incompatible with jax version {jax_version}. Please '
        'update your jax and/or jaxlib packages.')
  return _jaxlib_version


version_str = jaxlib.version.__version__
version = check_jaxlib_version(
  jax_version=jax.version.__version__,
  jaxlib_version=jaxlib.version.__version__,
  minimum_jaxlib_version=jax.version._minimum_jaxlib_version)

# Before importing any C compiled modules, first import the CPU
# feature guard module to verify that jaxlib was compiled in a way that only
# uses instructions that are present on this machine.
import jaxlib.cpu_feature_guard as cpu_feature_guard
cpu_feature_guard.check_cpu_features()

import jaxlib.xla_client as xla_client  # noqa: F401

# Jaxlib code is split between the Jax and the XLA repositories.
# Only for the internal usage of the JAX developers, we expose a version
# number that can be used to perform changes without breaking the main
# branch on the Jax github.
jaxlib_extension_version: int = getattr(xla_client, '_version', 0)
ifrt_version: int = getattr(xla_client, '_ifrt_version', 0)

import jaxlib.lapack as lapack  # noqa: F401
import jaxlib.utils as utils  # noqa: F401
import jaxlib._jax as _jax  # noqa: F401
import jaxlib.mlir._mlir_libs._jax_mlir_ext as jax_mlir_ext  # noqa: F401
from jaxlib._jax import guard_lib as guard_lib  # noqa: F401
from jaxlib._jax import jax_jit as jax_jit  # noqa: F401
from jaxlib._jax import pmap_lib as pmap_lib  # noqa: F401
from jaxlib._jax import pytree as pytree  # noqa: F401
from jaxlib._jax import Device as Device  # noqa: F401
from jaxlib import _profiler as _profiler  # noqa: F401
try:
  from jaxlib import _profile_data as _profile_data  # noqa: F401
except (ImportError, ModuleNotFoundError):
  _profile_data = None

from jaxlib._jax import ffi as ffi  # noqa: F401
import jaxlib.cpu_sparse as cpu_sparse  # noqa: F401
has_cpu_sparse = True

import jaxlib.weakref_lru_cache as weakref_lru_cache  # noqa: F401
import jaxlib._pretty_printer as _pretty_printer  # noqa: F401

if jaxlib_extension_version >= 365 or typing.TYPE_CHECKING:
  import jaxlib._ifrt_proxy as ifrt_proxy  # noqa: F401
else:
  ifrt_proxy = _jax.ifrt_proxy


# XLA garbage collection: see https://github.com/jax-ml/jax/issues/14882
def _xla_gc_callback(*args):
  xla_client._xla.collect_garbage()
gc.callbacks.append(_xla_gc_callback)

cuda_versions: ModuleType | None
for pkg_name in ['jax_cuda13_plugin', 'jax_cuda12_plugin', 'jaxlib.cuda']:
  try:
    cuda_versions = importlib.import_module(
        f'{pkg_name}._versions'
    )
  except ImportError:
    cuda_versions = None
  else:
    break

import jaxlib.gpu_solver as gpu_solver  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_sparse as gpu_sparse  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_prng as gpu_prng  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_linalg as gpu_linalg  # pytype: disable=import-error  # noqa: F401

import jaxlib.gpu_rnn as gpu_rnn  # pytype: disable=import-error  # noqa: F401
import jaxlib.gpu_triton as gpu_triton # pytype: disable=import-error  # noqa: F401

import jaxlib.mosaic.python.mosaic_gpu as mosaic_gpu_dialect  # pytype: disable=import-error  # noqa: F401
import jaxlib.mosaic.python.tpu as tpu  # pytype: disable=import-error  # noqa: F401

# TODO(rocm): check if we need the same for rocm.

def _cuda_path() -> str | None:
  def _try_cuda_root_environment_variable() -> str | None:
    """Use `CUDA_ROOT` environment variable if set."""
    return os.environ.get('CUDA_ROOT', None)

  def _try_cuda_nvcc_import() -> str | None:
    """Try to import `cuda_nvcc` and get its path directly.

    If the pip package `nvidia-cuda-nvcc-cu11` is installed, it should have
    both of the things XLA looks for in the cuda path, namely `bin/ptxas` and
    `nvvm/libdevice/libdevice.10.bc`.
    """
    try:
      from nvidia import cuda_nvcc  # pytype: disable=import-error
    except ImportError:
      return None

    if hasattr(cuda_nvcc, '__file__') and cuda_nvcc.__file__ is not None:
      # `cuda_nvcc` is a regular package.
      cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__).parent
    elif hasattr(cuda_nvcc, '__path__') and cuda_nvcc.__path__ is not None:
      # `cuda_nvcc` is a namespace package, which might have multiple paths.
      cuda_nvcc_path = None
      for path in cuda_nvcc.__path__:
        if (pathlib.Path(path) / 'bin' / 'ptxas').exists():
          cuda_nvcc_path = pathlib.Path(path)
          break
    else:
      return None

    return str(cuda_nvcc_path)

  def _try_bazel_runfiles() -> str | None:
    """Try to get the path to the cuda installation in bazel runfiles."""
    python_runfiles = os.environ.get('PYTHON_RUNFILES')
    if not python_runfiles:
      return None
    cuda_nvcc_root = os.path.join(python_runfiles, 'cuda_nvcc')
    if os.path.exists(cuda_nvcc_root):
      return cuda_nvcc_root
    return None

  if (path := _try_cuda_root_environment_variable()) is not None:
    return path
  elif (path := _try_cuda_nvcc_import()) is not None:
    return path
  elif (path := _try_bazel_runfiles()) is not None:
    return path

  return None

cuda_path = _cuda_path()
