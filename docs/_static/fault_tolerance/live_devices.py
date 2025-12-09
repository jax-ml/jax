# Copyright 2025 The JAX Authors.
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
os.environ['XLA_FLAGS'] = ' '.join([
  '--xla_gpu_nccl_terminate_on_error=false',
  '--xla_gpu_nccl_async_execution=true',
  '--xla_gpu_nccl_blocking_communicators=false',
])
os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'

from absl import app
from absl import flags
from collections.abc import Sequence
from jax.experimental.multihost_utils import live_devices
import jax
import jax.numpy as jnp
import time

_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")


def main(_: Sequence[str]) -> None:
  jax.config.update("jax_enable_recoverability", True)
  jax.distributed.initialize(
      coordinator_address="localhost:9000",
      num_processes=_NUM_PROCESSES.value,
      process_id=_PROCESS_ID.value,
      local_device_ids=[_PROCESS_ID.value],
      heartbeat_timeout_seconds=10,
  )
  print(f'{jax.devices()=}')
  print(f'{jax.local_devices()=}')

  while True:
    try:
      with live_devices(jax.devices()) as devices:
        print(f'{devices=}')
        n = len(devices)
        jax.set_mesh(jax.make_mesh((n,), ("i",), devices=devices))
        x = jax.device_put(jnp.arange(n), jax.P("i"))
        print(jnp.sum(x))
    except Exception as e:
      print('FAIL:', e)
    else:
      print('PASS')
    time.sleep(1)


if __name__ == "__main__":
  app.run(main)
