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

from typing import Optional
from jax._src.clusters import ClusterEnv
from jax._src.lib import xla_bridge
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm, get_metadata

class TpuCluster(ClusterEnv):
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm

  @classmethod
  def get_coordinator_address(cls) -> str:
    return cls._get_worker_endpoints()[0].split(':')[2] + ':8476'

  @classmethod
  def get_process_count(cls) -> int:
    return xla_bridge.process_count()

  @classmethod
  def get_process_id(cls) -> int:
    if cls.get_process_count() != len(cls._get_worker_endpoints()):
      raise RuntimeError('Number of workers does not equal the number of '
                         'processes. Auto detecting process_id is not possible.'
                         'Please pass process_id to jax.distributed.initialize() manually.')
    return int(get_metadata('agent-worker-number'))

  @classmethod
  def get_local_process_id(cls) -> Optional[int]:
    return None

  @staticmethod
  def _get_worker_endpoints() -> str:
    return get_metadata('worker-network-endpoints').split(',')
