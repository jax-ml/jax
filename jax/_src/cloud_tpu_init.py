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

import os
import warnings

running_in_cloud_tpu_vm: bool = False


def maybe_import_libtpu():
  try:
    # pylint: disable=import-outside-toplevel
    # pytype: disable=import-error
    import libtpu

    # pytype: enable=import-error
    # pylint: enable=import-outside-toplevel
  except ImportError:
    return None
  else:
    return libtpu


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

  # We assume we are in a correctly-configured Cloud TPU environment
  # if the following hold: a) libtpu is installed b) JAX_FORCE_TPU_INIT is set
  # Exit early if we're not running on Cloud TPU.
  libtpu_module = maybe_import_libtpu()
  if libtpu_module is not None:
    libtpu_module.configure_library_path()
  elif not jax_force_tpu_init():
    return

  running_in_cloud_tpu_vm = True

  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('JAX_PLATFORMS', 'tpu,cpu')
  os.environ['TPU_ML_PLATFORM'] = 'JAX'

  if 'JAX_USE_PJRT_C_API_ON_TPU' not in os.environ:
    os.environ['JAX_USE_PJRT_C_API_ON_TPU'] = 'true'

  use_pjrt_c_api = os.environ['JAX_USE_PJRT_C_API_ON_TPU']
  if use_pjrt_c_api in ("false", "0"):
    warnings.warn(
        f"JAX_USE_PJRT_C_API_ON_TPU={use_pjrt_c_api} will no longer be "
        "supported in an upcoming future release. Please file an issue at "
        "https://github.com/google/jax/issues if you need this setting.")
