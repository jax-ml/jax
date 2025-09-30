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
from jax.experimental import shard_map
import jax
import jax.numpy as jnp
import numpy as np
import sys
import time

_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")

def replicated(x: jax.Array, devices: list[jax.Device]):
  """Return x replicated across the provided devices.

  Note that replicated(x) doesn't actually move any data. It simply creates a
  logically replicated array with x as the local replica.
  """
  n = len(devices)
  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
  spec = jax.sharding.PartitionSpec(None)
  sharding = jax.sharding.NamedSharding(mesh, spec)
  shards = [
      jax.device_put(x.addressable_shards[0].data, d) for d in devices
      if d.process_index == jax.process_index()
  ]
  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)


def sharded(x: jax.Array, devices: list[jax.Device]):
  """Return x sharded across the provided devices.

  Note that sharded(x) doesn't actually move any data. It simply creates a
  logically sharded array. x should have the same shape as the global array.
  """
  n = len(devices)
  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
  spec = jax.sharding.PartitionSpec("i")
  sharding = jax.sharding.NamedSharding(mesh, spec)
  m = sharding.addressable_devices_indices_map(x.shape)
  shards = [jax.device_put(x[m[d]], d) for d in jax.local_devices()]
  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)


def send(x: jax.Array, from_device: jax.Device, to_device: jax.Device):
  """Sends x from one device to another."""
  assert isinstance(x, jax.Array)
  devices = [from_device, to_device]
  psum = lambda x: jax.lax.psum(x, "i")
  mesh = jax.make_mesh((2, ), ("i", ), devices=devices)
  spec = jax.sharding.PartitionSpec(None)
  x = replicated(x, [from_device, to_device])
  shard_map.shard_map(psum, mesh=mesh, in_specs=spec, out_specs=spec)(x)


def recv(x: jax.Array, from_device: jax.Device, to_device: jax.Device):
  """Receives x from a matching send."""
  assert isinstance(x, jax.Array)
  to_device = jax.local_devices()[0]
  devices = [from_device, to_device]
  psum = lambda x: jax.lax.psum(x, "i")
  mesh = jax.make_mesh((2, ), ("i", ), devices=devices)
  spec = jax.sharding.PartitionSpec(None)
  x = jnp.zeros_like(x)
  x = replicated(x, [from_device, to_device])
  return shard_map.shard_map(psum, mesh=mesh, in_specs=spec, out_specs=spec)(x)


def allgather(x: float, devices: list[jax.Device]) -> list[float]:
  """Performs an AllGather across the provided devices."""
  n = len(devices)
  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
  spec = jax.sharding.PartitionSpec('i')
  p = lambda x: jax.lax.all_gather(x, "i", tiled=True)
  f = jax.shard_map(p, mesh=mesh, in_specs=spec, out_specs=spec)
  return jax.block_until_ready(f(np.array([x] * len(devices)))).addressable_shards[0].data


def main(_: Sequence[str]) -> None:
  # Parse command line arguments and initialize multi-controller JAX.
  jax.config.update("jax_enable_recoverability", True)
  jax.distributed.initialize(coordinator_address="localhost:8000",
                             process_id=_PROCESS_ID.value,
                             num_processes=_NUM_PROCESSES.value,
                             local_device_ids=[process_id],
                             heartbeat_timeout_seconds=10)
  print(f'{jax.devices()=}')
  print(f'{jax.local_devices()=}')

  # Initialize the model's weights.
  keys = iter(jax.random.split(jax.random.key(seed=42), num=3))
  weights = jax.random.normal(next(keys), shape=(1, ))

  # We'll learn a trivial linear model: a*x.
  def predict(weights, X):
    return weights * X

  # We'll use mean squared error loss.
  def loss(weights, X, Y):
    return jnp.mean((predict(weights, X) - Y)**2)

  # Initialize the (noisy) training data with a=10.
  X = jax.random.permutation(next(keys), jnp.arange(-300., 300.))
  Y = 10 * X + jax.random.normal(next(keys), X.shape)

  # Hyperparameters.
  loss_and_grad = jax.jit(jax.value_and_grad(loss))
  learning_rate = 1e-6
  device_batch_size = 10

  step = 0
  while True:
    try:
      with live_devices(jax.devices()) as devices:
        print(f'=== Running step {step} with live devices = {devices} ===')

        # Handle recovering devices. A device is recovering if its step doesn't
        # match process 0's step. We assume process 0 never fails.
        print('all gathering steps...')
        steps = allgather(step, devices)
        print(f'{steps=}')
        recovering = [d for d, s in zip(devices, steps) if s != steps[0]]
        for d in recovering:
          # Process 0 sends weights and step to the recovering devices.
          if jax.process_index() == 0:
            print('sending...')
            send(weights, jax.devices()[0], d)
            send(jnp.array([step]), jax.devices()[0], d)
          elif d.process_index == jax.process_index():
            print('receiving...')
            weights = recv(weights, jax.devices()[0], d)
            step = recv(jnp.array([step]), jax.devices()[0], d)[0]

        # Replicate the model weights.
        weights = replicated(weights, devices)

        # Shard the batch.
        batch_size = device_batch_size * len(devices)
        start = (step * batch_size) % len(X)
        stop = start + batch_size
        X_batch = sharded(X[start:stop], devices)
        Y_batch = sharded(Y[start:stop], devices)

        # Compute gradients and update weights.
        l, grad = loss_and_grad(weights, X_batch, Y_batch)
        new_weights = jax.block_until_ready(weights - learning_rate * grad)
    except Exception as e:
      print(f'Step {step} failed: {e}')
    else:
      print(f'Step {step} succeeded: loss = {l}')
      step += 1
      weights = new_weights

    time.sleep(1)


if __name__ == "__main__":
  app.run(main)
