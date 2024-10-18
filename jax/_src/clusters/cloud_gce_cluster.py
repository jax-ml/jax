# Copyright 2022 The JAX Authors.
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

from __future__ import annotations

import logging
import os
import re
from jax._src import clusters
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm

logger = logging.getLogger(__name__)

# We use an arbitrarily chosen port for the coordinator since we cannot
# rely on communication to choose one in real time.
coordinator_port = '8476'

metadata_response_code_success = 200

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
        headers={'Metadata-Flavor': 'Google'}, timeout=60)
    if api_resp.status_code == 200:
      break
    retry_count += 1
    time.sleep(retrySeconds)

  if api_resp is None:
    raise RuntimeError(f"Getting metadata['{key}'] failed for 6 tries")
  return api_resp.text, api_resp.status_code


class GceTpuCluster(clusters.BaseTpuCluster):

  name: str = "gcetpu"

  @classmethod
  def is_env_present(cls) -> bool:
    if not running_in_cloud_tpu_vm:
      logger.debug("Did not detect cloud TPU VM")
      return False
    metadata_response, metadata_code = get_metadata('agent-worker-number')
    if metadata_code == metadata_response_code_success:
      logger.debug("Gce Tpu Cluster detected for Jax Distributed System")
      return True
    else:
      logger.debug("Did not detect Gce Tpu Cluster since agent-worker-number is not set in metadata")
      logger.debug("Metadata code: %s", metadata_code)
      logger.debug("Metadata response: %s", metadata_response)
      return False

  @staticmethod
  def _get_process_id_in_slice() -> int:
    return int(get_metadata('agent-worker-number')[0])

  @staticmethod
  def _get_worker_list_in_slice() -> list[str]:
    workers = get_metadata('worker-network-endpoints')[0].split(',')
    return [worker.split(':')[2] for worker in workers]

  @staticmethod
  def _get_tpu_env_value(key):
    def get_tpu_env_value_from_metadata(key):
      tpu_env_data = get_metadata('tpu-env')[0]
      key_value_pairs = tpu_env_data.split('\n')
      for key_value_pair in key_value_pairs:
        # Typical line is MEGASCALE_NUM_SLICES: '2'
        if ':' in key_value_pair:
          row_key, value = re.split(':', key_value_pair, 1)
          row_key = row_key.strip()
          if row_key == key:
            return value.strip().strip("'")
      return None

    value = os.environ.get(key, None)
    return value if value is not None else get_tpu_env_value_from_metadata(key)
