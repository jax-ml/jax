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

"""JAX ROCm plugin initialization module."""

import functools
import importlib
import logging
import os
import os.path
import pathlib
import re

from jax._src.lib import triton
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
  installed_path = base_path / 'xla_rocm_plugin.so'
  if installed_path.exists():
    return installed_path

  local_path = base_path / 'pjrt_c_api_gpu_plugin.so'
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


def set_rocm_paths(path):
  """Set ROCm environment paths for bitcode and linker."""
  rocm_lib = None
  try:
    import rocm  # pylint: disable=import-outside-toplevel
    rocm_lib = os.path.join(rocm.__path__[0], "lib")
  except ImportError:
    sp = path.parent.parent.parent
    maybe_rocm_lib = os.path.join(sp, "rocm/lib")
    if os.path.exists(maybe_rocm_lib):
      rocm_lib = maybe_rocm_lib

  if not rocm_lib:
    logger.info("No ROCm wheel installation found")
    return

  logger.info("ROCm wheel install found at %r", rocm_lib)

  bitcode_path = ""
  lld_path = ""

  for root, _dirs, files in os.walk(os.path.join(rocm_lib, "llvm")):
    for f in files:
      if f == "ocml.bc":
        bitcode_path = root
      if f == "ld.lld":
        lld_path = root

    if bitcode_path and lld_path:
      break

  if not bitcode_path:
    logger.warning("jax_rocm_plugin couldn't locate amdgpu bitcode")
  else:
    logger.info("jax_rocm_plugin using bitcode found at %r", bitcode_path)

  if not lld_path:
    logger.warning("jax_rocm_plugin couldn't locate amdgpu ld.lld")
  else:
    logger.info("jax_rocm_plugin using ld.lld found at %r", lld_path)

  os.environ["JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH"] = bitcode_path
  os.environ["HIP_DEVICE_LIB_PATH"] = bitcode_path
  os.environ["JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH"] = lld_path


def count_amd_gpus(stop_at: int = None) -> int:
  """Count AMD GPUs available via KFD kernel driver.

  Checks for the presence of AMD GPUs by examining KFD kernel driver topology
  nodes as a proxy. We use a non-zero simd_count as the trait of a GPU,
  following the KFD implementation. We avoid initializing HIP here because
  doing so might spoil a proper initialization of the rocprofiler-sdk later
  during PJRT startup.

  Args:
      stop_at: If provided, stop counting once this many GPUs are found.

  Returns:
      The number of AMD GPUs detected (up to stop_at if provided).
  """
  try:
    kfd_nodes_path = "/sys/class/kfd/kfd/topology/nodes/"
    if not os.path.exists(kfd_nodes_path):
      return 0

    gpu_count = 0
    r_simd_count = re.compile(r"\bsimd_count\s+(\d+)\b", re.MULTILINE)

    for node in os.listdir(kfd_nodes_path):
      node_props_path = os.path.join(kfd_nodes_path, node, "properties")
      if not os.path.exists(node_props_path):
        continue

      try:
        file_size = os.path.getsize(node_props_path)
        if file_size <= 0 or file_size > 16 * 1024:
          continue

        with open(node_props_path, "r", encoding="ascii") as f:
          match = r_simd_count.search(f.read())
          if match:
            simd_count = int(match.group(1))
            if simd_count > 0:
              gpu_count += 1
              if stop_at is not None and gpu_count >= stop_at:
                return gpu_count
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug(
            "Failed to read KFD node file '%s': %s", node_props_path, e
        )
        continue

  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to count AMD GPUs: %s", e)
  return gpu_count


def check_shm_size(gpu_count: int = None):
  """Check /dev/shm size and warn if it's too small for multi-GPU setups."""
  try:
    if gpu_count is None:
      gpu_count = count_amd_gpus(stop_at=2)

    if gpu_count < 2:
      return

    shm_path = "/dev/shm"
    if not os.path.exists(shm_path):
      return

    stat = os.statvfs(shm_path)
    shm_size_bytes = stat.f_blocks * stat.f_frsize
    shm_size_mb = shm_size_bytes / (1024 * 1024)

    if shm_size_mb <= 64:
      logger.warning(
          "Detected multiple GPUs but /dev/shm size is only %.1f MB. "
          "RCCL may exhaust shared memory during multi-GPU operations, "
          "causing runtime failures. Consider increasing /dev/shm size. "
          "For example in Docker, use: --shm-size=64g",
          shm_size_mb,
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.debug("Failed to check /dev/shm size: %s", e)


def initialize():
  path = _get_library_path()
  if path is None:
    return

  set_rocm_paths(path)

  if rocm_plugin_extension is None:
    logger.warning('rocm_plugin_extension not found')
    return

  gpu_count = count_amd_gpus(stop_at=2)

  if gpu_count == 0:
    raise ValueError("No AMD GPUs were found, skipping ROCm plugin initialization")

  check_shm_size(gpu_count)

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
    if hasattr(rocm_plugin_extension, 'compile_triton_to_asm'):
      triton.register_compilation_handler(
          "ROCM",
          functools.partial(
              rocm_plugin_extension.compile_triton_to_asm, c_api
          ),
      )
  else:
    logger.warning('rocm_plugin_extension is not found.')
