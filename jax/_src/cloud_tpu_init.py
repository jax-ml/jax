# Copyright 2021 Google LLC
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

def cloud_tpu_init():
  """Automatically sets Cloud TPU topology env vars.

  **This must be called before the TPU runtime is loaded, which happens as soon
  as JAX's C++ backend is loaded! I.e. call this before xla_bridge or xla_client
  is imported.**

  These environment variables are used to tell the TPU runtime what kind of mesh
  topology to use. It assumes a single-host topology by default, so we manually
  set them here to default to the full pod slice if applicable.

  This will not set any env vars if a single topology-related env var is already
  set.
  """
  if not _running_in_cloud_tpu_vm():
    return

  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')

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

  # Don't assume non-Cloud TPU environments have requests installed
  # pylint: disable=import-outside-toplevel
  # pytype: disable=import-error
  import requests
  # pytype: enable=import-error
  # pylint: enable=import-outside-toplevel

  # Based on https://github.com/tensorflow/tensorflow/pull/40317
  gce_metadata_endpoint = 'http://' + os.environ.get('GCE_METADATA_IP',
                                                     'metadata.google.internal')
  def get_metadata(key):
    return requests.get(
        f'{gce_metadata_endpoint}/computeMetadata/v1/instance/attributes/{key}',
        headers={'Metadata-Flavor': 'Google'}).text

  worker_id = get_metadata('agent-worker-number')
  accelerator_type = get_metadata('accelerator-type')
  worker_network_endpoints = get_metadata('worker-network-endpoints')

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
  os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
  os.environ['TPU_HOST_BOUNDS'] = accelerator_type_to_host_bounds[
      accelerator_type]
  os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = worker_network_endpoints.split(
      ',')[0].split(':')[2] + ':8476'
  os.environ['TPU_MESH_CONTROLLER_PORT'] = '8476'


def _running_in_cloud_tpu_vm():
  return os.path.isfile('/lib/libtpu.so')
