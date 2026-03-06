# Copyright 2024 The JAX Authors.
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
import pathlib

from jax._src import hardware_utils
from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

# rocm_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
# preinstalled jax rocm plugin packages.
for pkg_name in ['jax_rocm7_plugin', 'jax_rocm60_plugin', 'jaxlib.rocm']:
  try:
    rocm_plugin_extension = importlib.import_module(
        f'{pkg_name}.rocm_plugin_extension'
    )
  except ImportError:
    rocm_plugin_extension = None
  else:
    break

logger = logging.getLogger(__name__)


def _get_library_path():
  base_path = pathlib.Path(__file__).resolve().parent
  library_path = base_path / 'xla_rocm_plugin.so'
  if library_path.exists():
    return library_path
  logger.debug(
      'Native library %s does not exist. This most likely indicates an issue'
      ' with how %s was built or installed.',
      library_path,
      __package__,
  )
  return None


def initialize():
  logger.info("ROCm plugin initialize() called")
  path = _get_library_path()
  if path is None:
    logger.error("ROCm plugin is not detected")
    return
  logger.info("ROCm plugin library path: %s", path)

  # Count GPUs (stop at 2 since that's all we need to know)
  gpu_count = hardware_utils.count_amd_gpus(stop_at=2)
  logger.info("Detected %d AMD GPU(s)", gpu_count)
  if gpu_count == 0:
    logger.error("No AMD GPUs were found, skipping ROCm plugin initialization")
    return
  elif gpu_count > 1:
    shm_size_mb = hardware_utils.get_shm_size()
    if shm_size_mb and shm_size_mb <= 64:
      logger.warning(
          "Detected multiple GPUs but /dev/shm size is only %.1f MB. "
          "RCCL may exhaust shared memory during multi-GPU operations, "
          "causing runtime failures. Consider increasing /dev/shm size. "
          "For example in Docker, use: --shm-size=64g",
          shm_size_mb,
      )

  logger.info("Registering ROCm PJRT plugin")
  options = xla_client.generate_pjrt_gpu_plugin_options()
  options["platform_name"] = "ROCM"
  c_api = xb.register_plugin(
      'rocm', priority=500, library_path=str(path), options=options
  )
  logger.info("PJRT plugin registered, c_api=%s", c_api)

  if rocm_plugin_extension:
    logger.info("rocm_plugin_extension found, registering handlers")
    xla_client.register_custom_type_handler(
        "ROCM",
        functools.partial(
            rocm_plugin_extension.register_custom_type, c_api
        ),
    )
    xla_client.register_custom_call_handler(
        "ROCM",
        functools.partial(
            rocm_plugin_extension.register_custom_call_target, c_api
        ),
    )
    ffi_types = rocm_plugin_extension.ffi_types()
    logger.info("Registering %d FFI types: %s", len(ffi_types), list(ffi_types.keys()))
    for _name, _value in ffi_types.items():
      xla_client.register_custom_type(
          _name, _value, platform='ROCM'
      )
    ffi_handlers = rocm_plugin_extension.ffi_handlers()
    logger.info("Registering %d FFI handlers: %s", len(ffi_handlers), list(ffi_handlers.keys()))
    for _name, _value in ffi_handlers.items():
      xla_client.register_custom_call_target(
          _name, _value, platform='ROCM', api_version=1
      )
    logger.info("ROCm plugin initialization complete")
  else:
    logger.warning('rocm_plugin_extension is not found.')
