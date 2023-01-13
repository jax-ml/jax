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

running_in_cloud_tpu_vm = False

def cloud_tpu_init():
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
  try:
    # pylint: disable=import-outside-toplevel
    # pytype: disable=import-error
    import libtpu
    # pytype: enable=import-error
    # pylint: enable=import-outside-toplevel
  except ImportError:
    # We assume libtpu is installed iff we're in a correctly-configured Cloud
    # TPU environment. Exit early if we're not running on Cloud TPU.
    return

  running_in_cloud_tpu_vm = True

  libtpu.configure_library_path()
  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('JAX_PLATFORMS', 'tpu,cpu')
  os.environ['TPU_ML_PLATFORM'] = 'JAX'
