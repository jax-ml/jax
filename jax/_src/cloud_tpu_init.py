# Copyright 2021 The JAX Authors.
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

import logging
import os
import re
import warnings

from jax._src import config
from jax._src import hardware_utils

logger = logging.getLogger(__name__)

running_in_cloud_tpu_vm: bool = False


def maybe_import_libtpu():
  try:
    import libtpu  # pyrefly: ignore[missing-import]
  except ImportError:
    return None
  else:
    return libtpu


def get_tpu_library_path() -> str | None:
  path_from_env = os.getenv('TPU_LIBRARY_PATH')
  if path_from_env is not None:
    if os.path.isfile(path_from_env):
      return path_from_env
    warning_message = (
        f'TPU_LIBRARY_PATH is set to a non-existent path: {path_from_env}.'
        ' Falling back to default libtpu path. Please unset TPU_LIBRARY_PATH'
        ' or set it to a valid path.'
    )
    warnings.warn(warning_message)

  libtpu_module = maybe_import_libtpu()
  if libtpu_module is not None:
    return libtpu_module.get_library_path()

  return None


def jax_force_tpu_init() -> bool:
  return 'JAX_FORCE_TPU_INIT' in os.environ


def cloud_tpu_init() -> None:
  """Automatically sets Cloud TPU topology and other env vars.

  **This must be called before the TPU runtime is loaded, which happens as soon
  as JAX's C++ backend is loaded! I.e. call this before xla_bridge or xla_client
  is imported.**

  Safe to call in non-Cloud TPU environments.

  Some of these environment variables are used to tell the TPU runtime what kind
  of mesh topology to use. It assumes a single-host topology by default, so we
  manually set them here to default to the full pod slice if applicable.

  This will not set any env vars if a single topology-related env var is already
  set.
  """
  global running_in_cloud_tpu_vm

  from jax import version

  # Exit early if we're not running on a Cloud TPU VM or libtpu isn't installed.
  libtpu_path = get_tpu_library_path()
  num_tpu_chips, tpu_id = hardware_utils.num_available_tpu_chips_and_device_id()
  if num_tpu_chips == 0:
    os.environ['TPU_SKIP_MDS_QUERY'] = '1'
  if (
      tpu_id is not None
      and tpu_id >= hardware_utils.TpuVersion.v5e
      and not hardware_utils.transparent_hugepages_enabled()
  ):
    warnings.warn(
        'Transparent hugepages are not enabled. TPU runtime startup and'
        ' shutdown time should be significantly improved on TPU v5e and newer.'
        ' If not already set, you may need to enable transparent hugepages in'
        ' your VM image (sudo sh -c "echo always >'
        ' /sys/kernel/mm/transparent_hugepage/enabled")'
    )
  if (libtpu_path is None or num_tpu_chips == 0) and not jax_force_tpu_init():
    return

  running_in_cloud_tpu_vm = True

  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('TPU_ML_PLATFORM', 'JAX')
  os.environ.setdefault('TPU_ML_PLATFORM_VERSION', version.__version__)
  os.environ.setdefault('ENABLE_RUNTIME_UPTIME_TELEMETRY', '1')
  if '--xla_tpu_use_enhanced_launch_barrier' not in os.environ.get(
      'LIBTPU_INIT_ARGS', ''
  ):
    os.environ['LIBTPU_INIT_ARGS'] = (
        os.environ.get('LIBTPU_INIT_ARGS', '')
        + ' --xla_tpu_use_enhanced_launch_barrier=true'
    )

  # this makes tensorstore serialization work better on TPU
  os.environ.setdefault('TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS', '60')
  os.environ.setdefault('TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES', '256')

  # If the JAX_PLATFORMS env variable isn't set, config.jax_platforms defaults
  # to None. In this case, we set it to 'tpu,cpu' to ensure that JAX uses the
  # TPU backend.
  if config.jax_platforms.value is None:
    config.update('jax_platforms', 'tpu,cpu')

  if config.jax_pjrt_client_create_options.value is None:
    config.update(
        'jax_pjrt_client_create_options',
        f'ml_framework_name:JAX;ml_framework_version:{version.__version__}',
    )


_version_regex = re.compile(r'([0-9]+(?:\.[0-9]+)*)(?:(rc|dev).*)?')


def _parse_version(v: str) -> tuple[int, ...]:
  m = _version_regex.match(v)
  if m is None:
    raise ValueError(f"Unable to parse version '{v}'")
  return tuple(int(x) for x in m.group(1).split('.'))


def is_libtpu_at_least(version_str: str) -> bool:
  """Returns True if not running on Cloud TPU.

  If running on Cloud TPU, returns True if the installed libtpu version
  is at least `version_str`.

  Note: This checks the version of the installed `libtpu` Python package.
  If `TPU_LIBRARY_PATH` is set to a different path than the installed
  package's default, a warning will be issued as the loaded library
  might not match the package version we are checking.
  """
  if not running_in_cloud_tpu_vm:
    return True

  tpu_library_path = get_tpu_library_path()
  libtpu = maybe_import_libtpu()
  if libtpu is None:
    if tpu_library_path:
      warnings.warn(
          (
              'libtpu Python package is not installed, but TPU_LIBRARY_PATH is'
              f' set to {tpu_library_path}. Cannot determine libtpu version.'
              f' Assuming it is newer than {version_str}.'
          ),
          stacklevel=2,
      )
    else:
      warnings.warn(
          (
              'libtpu Python package is not installed, but we appear to be on a'
              ' Cloud TPU VM. Cannot determine libtpu version. Assuming it is'
              f' newer than {version_str}.'
          ),
          stacklevel=2,
      )
    return True

  if tpu_library_path and tpu_library_path != libtpu.get_library_path():
    logger.info(
        'TPU_LIBRARY_PATH is set to %s, which differs from the installed'
        ' package default (%s). Using the custom path set by TPU_LIBRARY_PATH'
        ' and assuming the version of libtpu is head for version tests.',
        tpu_library_path,
        libtpu.get_library_path(),
    )
    return True

  actual_version = _parse_version(libtpu.__version__)
  required_version = _parse_version(version_str)

  return actual_version >= required_version
