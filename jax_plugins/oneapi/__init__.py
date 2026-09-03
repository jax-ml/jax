# Copyright 2026 The JAX Authors.
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

import ctypes
import functools
import importlib
import importlib.util
import logging
import os
import pathlib
import sys

from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)

# oneapi_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
# preinstalled jax oneapi plugin packages.
def _load_oneapi_plugin_extension():
  for pkg_name in ['jax_oneapi_plugin', 'jax_oneapi_pjrt', 'jaxlib.oneapi']:
    try:
      return importlib.import_module(f'{pkg_name}.oneapi_plugin_extension')
    except ImportError:
      continue
  return None


def _try_load_library(path_or_name):
  try:
    if sys.platform == 'win32':
      ctypes.CDLL(str(path_or_name))
    else:
      mode = getattr(os, 'RTLD_GLOBAL', 0) | getattr(os, 'RTLD_LAZY', 0)
      ctypes.CDLL(str(path_or_name), mode=mode)
    logger.debug('Loaded library: %s', path_or_name)
    return True
  except OSError as e:
    logger.debug('Failed to load library %s: %s', path_or_name, e)
    return False


def _load_matching_libraries(lib_dir, patterns, loaded):
  import re
  def _version_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', path.name)]

  for pattern in patterns:
    matches = sorted(lib_dir.glob(pattern), key=_version_key, reverse=True)
    for path in matches:
      if str(path) in loaded:
        continue
      if _try_load_library(path):
        loaded.add(str(path))
        break


def _try_load_from_package(package_name, lib_patterns, loaded):
  pkg_paths = []
  try:
    spec = importlib.util.find_spec(package_name)
  except Exception:
    spec = None

  if spec is not None:
    if spec.submodule_search_locations:
      pkg_paths.extend(pathlib.Path(p) for p in spec.submodule_search_locations)
    elif spec.origin:
      pkg_paths.append(pathlib.Path(spec.origin).resolve().parent)

  for pkg_path in pkg_paths:
    for lib_dir in (pkg_path / 'lib', pkg_path):
      if lib_dir.exists():
        _load_matching_libraries(lib_dir, lib_patterns, loaded)


def _load_oneapi_libraries():
  patterns = [
      'libimf.so*',
      'libintlc.so*',
      'libirc.so*',
      'libsvml.so*',
      'libirng.so*',
      'libumf.so*',
      'libhwloc.so*',
      'libur_loader.so*',
      'libur_adapter_level_zero.so*',
      'libur_adapter_opencl.so*',
      'libOpenCL.so*',
      'libsycl.so*',
      'libmpi.so*',
      'libmkl_core.so*',
      'libmkl_sequential.so*',
      'libmkl_intel_ilp64.so*',
      'libmkl_intel_lp64.so*',
      'libmkl_intel_thread.so*',
      'libmkl_rt.so*',
      'libmkl_sycl_blas.so*',
      'libmkl_sycl_dft.so*',
      'libmkl_sycl_lapack.so*',
      'libmkl_sycl_rng.so*',
      'libmkl_sycl_sparse.so*',
      'libmkl_sycl_stats.so*',
  ]

  loaded = set()

  for package_name in [
      'jax_oneapi_plugin',
      'intel_sycl_rt',
      'intel_cmplr_lib_rt',
      'intel_cmplr_lic_rt',
      'intel_cmplr_lib_ur',
      'umf',
      'impi_rt',
      'mkl',
  ]:
    _try_load_from_package(package_name, patterns, loaded)

  lib_dirs = [
      pathlib.Path(sys.prefix) / 'lib',
      pathlib.Path(sys.prefix) / 'lib64',
      pathlib.Path('/usr/local/lib'),
  ]

  for lib_dir in lib_dirs:
    if lib_dir.exists():
      _load_matching_libraries(lib_dir, patterns, loaded)

  logger.debug('Loaded oneAPI libs via paths: %s', sorted(loaded))


def _get_library_path():
  base_path = pathlib.Path(__file__).resolve().parent
  installed_path = (
      base_path / 'xla_oneapi_plugin.so'
  )
  if installed_path.exists():
    return installed_path

  local_path = (
      base_path / 'pjrt_c_api_gpu_plugin.so'
  )
  if not local_path.exists():
    runfiles_dir = os.getenv('RUNFILES_DIR', None)
    if runfiles_dir:
      local_path = pathlib.Path(
          os.path.join(runfiles_dir, 'xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so')
      )

  if local_path.exists():
    logger.debug(
        'Native library %s does not exist. This most likely indicates an issue'
        ' with how %s was built or installed. Fallback to local test'
        ' library %s',
        installed_path,
        __package__,
        local_path,
    )
    return local_path

  logger.debug(
      'WARNING: Native library %s and local test library path %s do not'
      ' exist. This most likely indicates an issue with how %s was built or'
      ' installed or missing src files.',
      installed_path,
      local_path,
      __package__,
  )
  return None


def initialize():
  _load_oneapi_libraries()
  oneapi_plugin_extension = _load_oneapi_plugin_extension()

  path = _get_library_path()
  if path is None:
    return
  options = xla_client.generate_pjrt_gpu_plugin_options()
  options["platform_name"] = "SYCL"
  c_api = xb.register_plugin(
      'oneapi', priority=500, library_path=str(path), options=options
  )
  if oneapi_plugin_extension:
    xla_client.register_custom_type_handler(
        "ONEAPI",
        functools.partial(
            oneapi_plugin_extension.register_custom_type, c_api
        ),
    )
    xla_client.register_custom_call_handler(
        "ONEAPI",
        functools.partial(
            oneapi_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in oneapi_plugin_extension.ffi_types().items():
      xla_client.register_custom_type(
          _name, _value, platform='ONEAPI'
      )
    for _name, _value in oneapi_plugin_extension.ffi_handlers().items():
      xla_client.register_custom_call_target(
          _name, _value, platform='ONEAPI', api_version=1
      )
  else:
    logger.warning('oneapi_plugin_extension is not found.')
