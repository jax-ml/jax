import os
os.environ['XLA_FLAGS'] = '--xla_gpu_nccl_terminate_on_error=false'

from absl import app
from absl import flags
from collections.abc import Sequence
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
    mesh = jax.make_mesh((4,), ("i",))
    spec = jax.sharding.PartitionSpec("i")
    sharding = jax.sharding.NamedSharding(mesh, spec)
    x = jax.device_put(jnp.array([1., 2., 3., 4.]), sharding)
    print(jnp.sum(x))
    time.sleep(1)


if __name__ == "__main__":
  app.run(main)
