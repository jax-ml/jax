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
from jax._src import clusters
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm

logger = logging.getLogger(__name__)


class GkeTpuCluster(clusters.BaseTpuCluster):

  name: str = "gketpu"

  @classmethod
  def is_env_present(cls) -> bool:
    if running_in_cloud_tpu_vm and os.environ.get("TPU_WORKER_HOSTNAMES") is not None:
      logger.debug("Gke Tpu Cluster detected for Jax Distributed System")
      return True
    else:
      if not running_in_cloud_tpu_vm:
        logger.debug("Did not detect cloud TPU VM")
      else:
        logger.debug("Did not detect TPU GKE cluster since TPU_WORKER_HOSTNAMES is not set")
      return False

  @staticmethod
  def _get_process_id_in_slice() -> int:
    return int(str(os.environ.get('TPU_WORKER_ID')))

  @staticmethod
  def _get_worker_list_in_slice() -> list[str]:
    return str(os.environ.get('TPU_WORKER_HOSTNAMES', None)).split(',')

  @staticmethod
  def _get_tpu_env_value(key):
    return os.environ.get(key, None)
