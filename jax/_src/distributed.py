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
import functools

from typing import Optional

from absl import logging
from jax._src import cloud_tpu_init
from jax._src import config
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client
from jax._src.lib import xla_extension

jax_service = None
distributed_client = None


def initialize(coordinator_address: Optional[str] = None,
               num_processes: Optional[int] = None,
               process_id: Optional[int] = None):
  """Initialize distributed system for topology discovery.

  Currently, calling ``initialize`` sets up the multi-host GPU backend and Cloud
  TPU backend.

  If you are on GPU platform, you will have to provide the coordinator_address
  and other args to the `initialize` API.

  If you are on TPU platform, the coordinator_address and other args will be
  auto detected but you have the option to provide it too.

  Args:
    coordinator_address: IP address and port of the coordinator. The choice of
      port does not matter, so long as the port is available on the coordinator
      and all processes agree on the port.
      Can be None only for TPU platform. If coordinator_address is None on TPU,
      then it will be auto detected.
    num_processes: Number of processes. Can be None only for TPU platform and
      if None will be determined from the TPU slice metadata.
    process_id: Id of the current process. Can be None only for TPU platform and
      if None will default to the current TPU worker id determined via the TPU
      slice metadata.

  Raises:
    RuntimeError: If `distributed.initialize` is called more than once.

  Example:

  Suppose there are two GPU hosts, and host 0 is the designated coordinator
  with address ``10.0.0.1:1234``. To initialize the GPU cluster, run the
  following commands before anything else.

  On host 0:

  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 0)  # doctest: +SKIP

  On host 1:

  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 1)  # doctest: +SKIP
  """

  coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS',
                                       None) or coordinator_address

  if cloud_tpu_init.running_in_cloud_tpu_vm:
    worker_endpoints = cloud_tpu_init.get_metadata(
        'worker-network-endpoints').split(',')
    if coordinator_address is None:
      coordinator_address = worker_endpoints[0].split(':')[2] + ':8476'
    if num_processes is None:
      num_processes = xla_bridge.process_count()
    if process_id is None:
      process_id = int(cloud_tpu_init.get_metadata('agent-worker-number'))

    if num_processes != len(worker_endpoints):
      raise RuntimeError('Number of workers does not equal the number of '
                         'processes. Auto detecting process_id is not possible.'
                         'Please pass process_id manually.')

  if coordinator_address is None:
    raise ValueError('coordinator_address should be defined.')
  if num_processes is None:
    raise ValueError('Number of processes must be defined.')
  if process_id is None:
    raise ValueError('The process id of the current process must be defined.')

  if process_id == 0:
    global jax_service
    if jax_service is not None:
      raise RuntimeError('distributed.initialize should only be called once.')

  global distributed_client
  if distributed_client is not None:
    raise RuntimeError('distributed.initialize should only be called once.')

  logging.info('Starting JAX distributed service on %s', coordinator_address)
  if xla_client._version >= 72:
    jax_service = xla_extension.get_distributed_runtime_service(
        coordinator_address, num_processes, config.jax_coordination_service)
    distributed_client = xla_extension.get_distributed_runtime_client(
        coordinator_address, process_id, config.jax_coordination_service)
  else:
    jax_service = xla_extension.get_distributed_runtime_service(
        coordinator_address, num_processes)
    distributed_client = xla_extension.get_distributed_runtime_client(
        coordinator_address, process_id)
  logging.info('Connecting to JAX distributed service on %s', coordinator_address)
  distributed_client.connect()

  if xla_client._version >= 65:
    factory = functools.partial(
        xla_client.make_gpu_client,
        distributed_client,
        process_id,
        platform_name='cuda')
    xla_bridge.register_backend_factory('cuda', factory, priority=300)
    factory = functools.partial(
        xla_client.make_gpu_client,
        distributed_client,
        process_id,
        platform_name='rocm')
    xla_bridge.register_backend_factory('rocm', factory, priority=300)
  else:
    factory = functools.partial(
        xla_client.make_gpu_client,
        distributed_client,
        process_id)
    xla_bridge.register_backend_factory('gpu', factory, priority=300)
