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

  # If the user has set any topology-related env vars, don't set any
  # automatically.
  if any([
      os.environ.get('CLOUD_TPU_TASK_ID', None),
      os.environ.get('TPU_CHIPS_PER_HOST_BOUNDS', None),
      os.environ.get('TPU_HOST_BOUNDS', None),
      os.environ.get('TPU_MESH_CONTROLLER_ADDRESS', None),
      os.environ.get('TPU_MESH_CONTROLLER_PORT', None),
      os.environ.get('TPU_VISIBLE_DEVICES', None),
  ]):
    return

  worker_id = get_metadata('agent-worker-number')
  accelerator_type = get_metadata('accelerator-type')

  accelerator_type_to_host_bounds = {
      'v2-8': '1,1,1',
      'v2-32': '2,2,1',
      'v2-128': '4,4,1',
      'v2-256': '4,8,1',
      'v2-512': '8,8,1',
      'v3-8': '1,1,1',
      'v3-32': '2,2,1',
      'v3-64': '2,4,1',
      'v3-128': '4,4,1',
      'v3-256': '4,8,1',
      'v3-512': '8,8,1',
      'v3-1024': '8,16,1',
      'v3-2048': '16,16,1',
  }

  os.environ['CLOUD_TPU_TASK_ID'] = worker_id

  # If v4 TPU don't set any topology related flags, libtpu will set these values.
  if not accelerator_type.startswith('v4-'):
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
    os.environ['TPU_HOST_BOUNDS'] = accelerator_type_to_host_bounds[
        accelerator_type]


def get_metadata(key):
  import requests  # pytype: disable=import-error
  import time  # pytype: disable=import-error
  # Based on https://github.com/tensorflow/tensorflow/pull/40317
  gce_metadata_endpoint = 'http://' + os.environ.get(
      'GCE_METADATA_IP', 'metadata.google.internal')

  retry_count = 0
  retrySeconds = 0.500
  api_resp = None

  while retry_count < 6:
    api_resp = requests.get(
        f'{gce_metadata_endpoint}/computeMetadata/v1/instance/attributes/{key}',
        headers={'Metadata-Flavor': 'Google'})
    if api_resp.status_code == 200:
      break
    retry_count += 1
    time.sleep(retrySeconds)

  if api_resp is None:
    raise RuntimeError(f"Getting metadata['{key}'] failed for 6 tries")
  return api_resp.text
