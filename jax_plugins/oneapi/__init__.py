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

import functools
import importlib
import logging
import os
import pathlib

from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

# oneapi_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
# preinstalled jax oneapi plugin packages.
for pkg_name in ['jax_oneapi2025_1_plugin', 'jaxlib.oneapi']:
  try:
    oneapi_plugin_extension = importlib.import_module(
        f'{pkg_name}.oneapi_plugin_extension'
    )
  except ImportError:
    oneapi_plugin_extension = None
  else:
    break

logger = logging.getLogger(__name__)

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
