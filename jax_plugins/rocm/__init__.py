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

_MIN_SHM_SIZE_MB = 64
logger = logging.getLogger(__name__)

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
  path = _get_library_path()
  if path is None:
    return

  # Count GPUs (stop at 2 since that's all we need to know)
  gpu_count = hardware_utils.num_available_amd_gpus(stop_at=2)
  if gpu_count <= 0:
    raise ValueError("No AMD GPUs were found, skipping ROCm plugin initialization")
  elif gpu_count > 1:
    shm_size_mb = hardware_utils.get_shm_size_in_mb()
    if shm_size_mb <= _MIN_SHM_SIZE_MB:
      logger.warning(
          "Detected multiple GPUs but /dev/shm size is only %.1f MB. "
          "RCCL may exhaust shared memory during multi-GPU operations, "
          "causing runtime failures. Consider increasing /dev/shm size. "
          "For example in Docker, use: --shm-size=64g",
          shm_size_mb,
      )

  options = xla_client.generate_pjrt_gpu_plugin_options()
  options["platform_name"] = "ROCM"
  c_api = xb.register_plugin(
      'rocm', priority=500, library_path=str(path), options=options
  )
  if rocm_plugin_extension:
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
    for _name, _value in rocm_plugin_extension.ffi_types().items():
      xla_client.register_custom_type(
          _name, _value, platform='ROCM'
      )
    for _name, _value in rocm_plugin_extension.ffi_handlers().items():
      xla_client.register_custom_call_target(
          _name, _value, platform='ROCM', api_version=1
      )
  else:
    logger.warning('rocm_plugin_extension is not found.')
