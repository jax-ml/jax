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
        mesh = jax.make_mesh((n,), ("i",), devices=devices)
        spec = jax.sharding.PartitionSpec("i")
        sharding = jax.sharding.NamedSharding(mesh, spec)
        x = jax.device_put(jnp.arange(n), sharding)
        print(jnp.sum(x))
        time.sleep(1)
    except Exception as e:
      print('FAIL:', e)
    else:
      print('PASS')


if __name__ == "__main__":
  app.run(main)
