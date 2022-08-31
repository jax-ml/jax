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

import atexit
import os
import functools

from typing import Any, Optional

from absl import logging
from jax._src import cloud_tpu_init
from jax._src.config import config
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client
from jax._src.lib import xla_extension

class State:
  process_id: int = 0
  service: Optional[Any] = None
  client: Optional[Any] = None
  preemption_sync_manager: Optional[Any] = None

  def initialize(self,
                 coordinator_address: Optional[str] = None,
                 num_processes: Optional[int] = None,
                 process_id: Optional[int] = None):
    coordinator_address = (coordinator_address or
                           os.environ.get('JAX_COORDINATOR_ADDRESS', None))

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

    self.process_id = process_id

    if process_id == 0:
      if self.service is not None:
        raise RuntimeError('distributed.initialize should only be called once.')
      logging.info('Starting JAX distributed service on %s', coordinator_address)
      self.service = xla_extension.get_distributed_runtime_service(
          coordinator_address, num_processes, config.jax_coordination_service)

    if self.client is not None:
      raise RuntimeError('distributed.initialize should only be called once.')

    # Set init_timeout to 5 min to leave time for all the processes to connect
    self.client = xla_extension.get_distributed_runtime_client(
        coordinator_address, process_id, config.jax_coordination_service,
        init_timeout=300)
    logging.info('Connecting to JAX distributed service on %s', coordinator_address)
    self.client.connect()

    if xla_client._version >= 77 and config.jax_coordination_service:
      self.initialize_preemption_sync_manager()

  def shutdown(self):
    if self.client:
      self.client.shutdown()
      self.client = None
    if self.service:
      self.service.shutdown()
      self.service = None
    if self.preemption_sync_manager:
      self.preemption_sync_manager = None

  def initialize_preemption_sync_manager(self):
    if self.preemption_sync_manager is not None:
      raise RuntimeError(
          'Preemption sync manager should only be initialized once.')
    self.preemption_sync_manager = (
        xla_extension.create_preemption_sync_manager())
    self.preemption_sync_manager.initialize(self.client)

global_state = State()


def initialize(coordinator_address: Optional[str] = None,
               num_processes: Optional[int] = None,
               process_id: Optional[int] = None):
  """Initializes the JAX distributed system.

  Calling :func:`~jax.distributed.initialize` prepares JAX for execution on
  multi-host GPU and Cloud TPU. :func:`~jax.distributed.initialize` must be
  called before performing any JAX computations.

  The JAX distributed system serves a number of roles:

    * it allows JAX processes to discover each other and share topology information,
    * it performs health checking, ensuring that all processes shut down if any process dies, and
    * it is used for distributed checkpointing.

  If you are using GPU, you must provide the ``coordinator_address``,
  ``num_processes``, and ``process_id`` arguments to :func:`~jax.distributed.initialize`.

  If you are using TPU, all arguments are optional: if omitted, they
  will be chosen automatically from the Cloud TPU metadata.

  Args:
    coordinator_address: the IP address of process `0` and a port on which that
      process should launch a coordinator service. The choice of
      port does not matter, so long as the port is available on the coordinator
      and all processes agree on the port.
      May be ``None`` only on TPU, in which case it will be chosen automatically.
    num_processes: Number of processes. May be ``None`` only on TPU, in
      which case it will be chosen automatically based on the TPU slice.
    process_id: The ID number of the current process. The ``process_id`` values across
      the cluster must be a dense range ``0``, ``1``, ..., ``num_processes - 1``.
      May be ``None`` only on TPU; if ``None`` it will be chosen from the TPU slice
      metadata.

  Raises:
    RuntimeError: If :func:`~jax.distributed.initialize` is called more than once.

  Example:

  Suppose there are two GPU processs, and process 0 is the designated coordinator
  with address ``10.0.0.1:1234``. To initialize the GPU cluster, run the
  following commands before anything else.

  On process 0:

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=0)  # doctest: +SKIP

  On process 1:

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=1)  # doctest: +SKIP
  """
  global_state.initialize(coordinator_address, num_processes, process_id)
  atexit.register(shutdown)


def shutdown():
  """Shuts down the distributed system.

  Does nothing if the distributed system is not running."""
  global_state.shutdown()
