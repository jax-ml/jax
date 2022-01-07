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

import functools

from absl import logging
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client
from jax._src.lib import xla_extension

_service = None
def initialize(coordinator_address: str, num_processes: int, process_id: int):
  """Initialize distributed system for topology discovery.

  Currently, calling ``initialize`` sets up the multi-host GPU backend, and
  is not required for CPU or TPU backends.

  Args:
    coordinator_address: IP address of the coordinator.
    num_processes: Number of processes.
    process_id: Id of the current processe.

  Example:

  Suppose there are two GPU hosts, and host 0 is the designated coordinator
  with address '10.0.0.1:1234', to initialize the GPU cluster, run the
  following commands before anything else.

  On host 0
  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 0)  # doctest: +SKIP

  On host 1
  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 1)  # doctest: +SKIP
  """
  if process_id == 0:
    global _service
    assert _service is None, 'initialize should be called once only'
    logging.info('Starting JAX distributed service on %s', coordinator_address)
    _service = xla_extension.get_distributed_runtime_service(coordinator_address,
                                                             num_processes)

  client = xla_extension.get_distributed_runtime_client(coordinator_address,
                                                        process_id)
  logging.info('Connecting to JAX distributed service on %s', coordinator_address)
  client.connect()

  factory = functools.partial(xla_client.make_gpu_client, client, process_id)
  xla_bridge.register_backend_factory('gpu', factory, priority=300)
