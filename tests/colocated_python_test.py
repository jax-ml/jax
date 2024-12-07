# Copyright 2024 The JAX Authors.
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

import contextlib
import threading
import time
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import xla_extension_version  # pylint: disable=g-importing-member
from jax.experimental import colocated_python
from jax.experimental.colocated_python import func as colocated_python_func
from jax.experimental.colocated_python import serialization
from jax.extend.ifrt_programs import ifrt_programs
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()


def _colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> Sequence[jax.Device]:
  """Returns CPU devices colocated with the given devices."""
  try:
    return colocated_python.colocated_cpu_devices(devices)
  except (ValueError, AttributeError):
    # PjRt-IFRT prepares CPU devices by its own.
    # TODO(hyeontaek): Remove this fallback path once PjRt-IFRT prepares CPU
    # devices by its own.
    cpu_backend_devices = jax.local_devices(backend="cpu")
    device_index_map = {device.id: i for i, device in enumerate(jax.devices())}

    available_devices = devices[: min(len(cpu_backend_devices), len(devices))]
    return [
        cpu_backend_devices[device_index_map[d.id]] for d in available_devices
    ]


@contextlib.contextmanager
def _count_colocated_python_specialization_cache_miss() -> list[int]:
  """Counts the number of cache misses for colocated_python specialization."""
  original_get_specialized_func = colocated_python_func._get_specialized_func
  count = [0]

  @jax.util.cache(max_size=None)
  def get_specialized_func(*args, **kwargs):
    count[0] += 1
    return original_get_specialized_func(*args, **kwargs)

  colocated_python_func._get_specialized_func = get_specialized_func
  try:
    yield count
  finally:
    colocated_python_func._get_specialized_func = original_get_specialized_func


_exit_stack = contextlib.ExitStack()


def setUpModule():
  # TODO(hyeontaek): Remove provisioning "cpu" backend devices once PjRt-IFRT
  # prepares CPU devices by its own.
  _exit_stack.enter_context(jtu.set_host_platform_device_count(8))


def tearDownModule():
  _exit_stack.close()


class ColocatedPythonTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if xla_extension_version < 300:
      self.skipTest("Requires xla_extension_version >= 300")

  def testMakeColocatedPythonProgram(self):
    def add_one(x):
      return x + 1

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    sharding = jax.sharding.SingleDeviceSharding(cpu_devices[0])
    sds = jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding)

    pickled_function = serialization._serialize(add_one)
    program = ifrt_programs.make_colocated_python_program(
        "add_one", pickled_function, [cpu_devices[0]], [sds], [sds]
    )
    del program

  def testSimpleFunction(self):
    @colocated_python.colocated_python
    def add_one(x):
      return x + 1

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])

    with _count_colocated_python_specialization_cache_miss() as count:
      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count[0], 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count[0], 1)

  def testSimpleFunctioWithTree(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = [np.array(1), (np.array(2), {"v": np.array(3)})]
    x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))

    with _count_colocated_python_specialization_cache_miss() as count:
      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count[0], 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count[0], 1)

  def testEmptyInputFailsWithoutSpecialization(self):
    @colocated_python.colocated_python
    def make_zero():
      return jnp.array(0)

    with self.assertRaisesRegex(
        ValueError,
        "No devices found. colocated_python function without input arguments"
        " must be first specialized with devices.",
    ):
      _ = make_zero()

  def testEmptyInputWithDevicesSpecialization(self):
    @colocated_python.colocated_python
    def make_zero():
      return jnp.array(0)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())

    with _count_colocated_python_specialization_cache_miss() as count:
      make_zero = make_zero.specialize(devices=cpu_devices[:1])
      out = make_zero()
      out = jax.device_get(out)
      self.assertEqual(out, np.array(0))
      self.assertEqual(count[0], 1)

      out = make_zero()
      out = jax.device_get(out)
      self.assertEqual(out, np.array(0))
      self.assertEqual(count[0], 1)

  def testInputPolymorphismWithoutOutSpecsFn(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])

    with _count_colocated_python_specialization_cache_miss() as count:
      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count[0], 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count[0], 1)

      # Different input tree structure and dtype/shape.
      x = [np.array(1), (np.array(2), {"v": np.array(3)})]
      x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count[0], 2)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count[0], 2)

  def testInputPolymorphismAllowedWithOutSpecsFn(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])

    with _count_colocated_python_specialization_cache_miss() as count:
      add_one = add_one.specialize(out_specs_fn=lambda x: x)
      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count[0], 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count[0], 1)

      # Different input tree structure and dtype/shape.
      x = [np.array(1), (np.array(2), {"v": np.array(3)})]
      x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count[0], 2)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count[0], 2)

  @parameterized.named_parameters(
      ("on_main_thread", True),
      ("on_non_main_thread", False),
  )
  def testSequentialExecution(self, on_main_thread: bool):
    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])
    # Make sure that this input array is ready for use by the colocated Python
    # function and does not disrupt elapsed time measurement.
    jax.block_until_ready(x)

    @colocated_python.colocated_python
    def sleep(x: jax.Array) -> jax.Array:
      time.sleep(5)
      return x

    # Specify out_specs_fn so that all executions are asynchronously dispatched.
    sleep = sleep.specialize(out_specs_fn=lambda x: x)

    def sleep_twice_and_wait(x: jax.Array) -> None:
      _ = sleep(x)
      jax.block_until_ready(sleep(x))

    start_time = time.time()

    # Two executions of `sleep` within `sleep_twice_and_wait` should run
    # sequentially.
    if on_main_thread:
      sleep_twice_and_wait(x)
    else:
      t = threading.Thread(target=sleep_twice_and_wait, args=(x,))
      t.start()
      t.join()

    elapsed_time = time.time() - start_time

    # If sequential execution did not happen, elapsed time typically will be
    # around 5 seconds.
    self.assertGreaterEqual(elapsed_time, 10)

  def testConcurrentExecution(self):
    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])
    # Make sure that this input array is ready for use by the colocated Python
    # function and does not disrupt elapsed time measurement.
    jax.block_until_ready(x)

    @colocated_python.colocated_python
    def sleep(x: jax.Array) -> jax.Array:
      time.sleep(5)
      return x

    # Specify out_specs_fn so that all executions are asynchronously dispatched.
    sleep = sleep.specialize(out_specs_fn=lambda x: x)

    def sleep_and_wait(x: jax.Array) -> None:
      jax.block_until_ready(sleep(x))

    start_time = time.time()

    # All three executions of `sleep_and_wait` should run concurrently.
    t1 = threading.Thread(target=sleep_and_wait, args=(x,))
    t2 = threading.Thread(target=sleep_and_wait, args=(x,))
    t1.start()
    t2.start()
    sleep_and_wait(x)
    t1.join()
    t2.join()

    elapsed_time = time.time() - start_time

    self.assertGreaterEqual(elapsed_time, 5)
    # If concurrent execution did not happen, elapsed time typically will be
    # around 15 seconds.
    self.assertLess(elapsed_time, 10)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
