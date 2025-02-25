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

import base64
import struct
import tempfile
import threading
import time
from typing import Sequence
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental import colocated_python
from jax.experimental.colocated_python import serialization
from jax.extend.ifrt_programs import ifrt_programs
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

try:
  import cloudpickle  # noqa
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("tests depend on cloudpickle library")


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


_count_colocated_python_specialization_cache_miss = jtu.count_events(
    "colocated_python_func._get_specialized_func"
)


class ColocatedPythonTest(jtu.JaxTestCase):

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
      self.assertEqual(count(), 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count(), 1)

  def testSimpleFunctionWithTree(self):
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
      self.assertEqual(count(), 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count(), 1)

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
      self.assertEqual(count(), 1)

      out = make_zero()
      out = jax.device_get(out)
      self.assertEqual(out, np.array(0))
      self.assertEqual(count(), 1)

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
      self.assertEqual(count(), 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count(), 1)

      # Different input tree structure and dtype/shape.
      x = [np.array(1), (np.array(2), {"v": np.array(3)})]
      x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count(), 2)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count(), 2)

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
      self.assertEqual(count(), 1)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, np.array(2))
      self.assertEqual(count(), 1)

      # Different input tree structure and dtype/shape.
      x = [np.array(1), (np.array(2), {"v": np.array(3)})]
      x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count(), 2)

      out = add_one(x)
      out = jax.device_get(out)
      self.assertEqual(out, [np.array(2), (np.array(3), {"v": np.array(4)})])
      self.assertEqual(count(), 2)

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

  def testInputsWithDifferentDeviceOrders(self):
    cpu_devices = _colocated_cpu_devices(jax.local_devices())[:2]
    if len(cpu_devices) < 2:
      self.skipTest("Not enough CPU devices")

    @colocated_python.colocated_python
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
      arrays = [
          x.addressable_shards[1].data + y.addressable_shards[0].data,
          x.addressable_shards[0].data + y.addressable_shards[1].data,
      ]
      return jax.make_array_from_single_device_arrays(
          y.shape, y.sharding, arrays
      )

    # The execution will use mixed device orders. We should specialize the
    # function with devices to avoid the argument-dependent device selection.
    add = add.specialize(devices=cpu_devices)

    mesh1 = jax.sharding.Mesh([cpu_devices[0], cpu_devices[1]], "x")
    sharding1 = jax.sharding.NamedSharding(
        mesh1, jax.sharding.PartitionSpec("x")
    )
    mesh2 = jax.sharding.Mesh([cpu_devices[1], cpu_devices[0]], "x")
    sharding2 = jax.sharding.NamedSharding(
        mesh2, jax.sharding.PartitionSpec("x")
    )

    x = np.array([0, 2])
    x = jax.device_put(x, sharding1)
    y = np.array([4, 8])
    y = jax.device_put(y, sharding2)

    out = add(x, y)

    self.assertEqual(out.sharding, sharding2)
    out_device_list = [shard.device for shard in out.addressable_shards]
    self.assertEqual(out_device_list, [cpu_devices[1], cpu_devices[0]])

    out = jax.device_get(out)
    np.testing.assert_equal(out, np.array([2 + 4, 0 + 8]))

  def testModuleVariableAccess(self):
    try:
      # The following pattern of storing and accessing non-serialized state in
      # the Python module is discouraged for storing user-defined state.
      # However, it should still work because many caching mechanisms rely on
      # this behavior.

      # Poison the test's own `colocated_python` module with a non-serializable
      # object (file) to detect any invalid attempt to serialize the module as
      # part of a colocated Python function.
      colocated_python._testing_non_serializable_object = (
          tempfile.TemporaryFile()
      )

      @colocated_python.colocated_python
      def set_global_state(x: jax.Array) -> jax.Array:
        colocated_python._testing_global_state = x
        return x + 1

      @colocated_python.colocated_python
      def get_global_state(x: jax.Array) -> jax.Array:
        del x
        return colocated_python._testing_global_state

      cpu_devices = _colocated_cpu_devices(jax.local_devices())
      x = np.array(1)
      x = jax.device_put(x, cpu_devices[0])
      y = np.array(2)
      y = jax.device_put(y, cpu_devices[0])

      jax.block_until_ready(set_global_state(x))
      out = jax.device_get(get_global_state(y))

      np.testing.assert_equal(out, np.array(1))
    finally:
      if "_testing_non_serializable_object" in colocated_python.__dict__:
        colocated_python._testing_non_serializable_object.close()
        del colocated_python._testing_non_serializable_object
      if "_testing_global_state" in colocated_python.__dict__:
        del colocated_python._testing_global_state

  def testStringProcessing(self):
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
      self.skipTest("StringDType requires NumPy 2.0.0 or later")
    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    if len(cpu_devices) < 2:
      self.skipTest(f"Need at least two CPU devices, got: {len(cpu_devices)}")

    @colocated_python.colocated_python
    def f(x):
      out_arrays = []
      upper_caser = np.vectorize(
          lambda x: x.upper(), otypes=[np.dtypes.StringDType()]
      )
      for shard in x.addressable_shards:
        np_array = jax.device_get(shard.data)
        out_np_array = upper_caser(np_array)
        out_arrays.append(jax.device_put(out_np_array, device=shard.device))
      return jax.make_array_from_single_device_arrays(
          sharding=x.sharding, shape=x.shape, arrays=out_arrays
      )

    # Make a string array.
    numpy_string_array = np.array(
        [["abcd", "efgh"], ["ijkl", "mnop"]], dtype=np.dtypes.StringDType()  # type: ignore
    )
    mesh = jax.sharding.Mesh(
        np.array(cpu_devices[:2]).reshape((2, 1)), ("x", "y")
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    x = jax.device_put(numpy_string_array, device=sharding)

    # Run the colocated Python function with the string array as input.
    out = f(x)
    out = jax.device_get(out)

    # Should have gotten the strings with all upper case letters.
    np.testing.assert_equal(
        out,
        np.array(
            [["ABCD", "EFGH"], ["IJKL", "MNOP"]], dtype=np.dtypes.StringDType()
        ),
    )

  def testBinaryDataProcessing(self):
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
      self.skipTest("StringDType requires NumPy 2.0.0 or later")
    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    if len(cpu_devices) < 1:
      self.skipTest("Need at least one CPU devices")

    @colocated_python.colocated_python
    def f(x):
      out_arrays = []
      for shard in x.addressable_shards:
        np_array = jax.device_get(shard.data)
        input_ints = struct.unpack(
            "<ii", base64.b64decode(np_array[0].encode("ascii"))
        )
        output_string = base64.b64encode(
            struct.pack("<ii", input_ints[0] + 1, input_ints[1] + 1)
        ).decode("ascii")
        out_np_array = np.array([output_string], dtype=np.dtypes.StringDType())
        out_arrays.append(jax.device_put(out_np_array, device=shard.device))

      out = jax.make_array_from_single_device_arrays(
          sharding=x.sharding, shape=x.shape, arrays=out_arrays
      )
      return out

    # Make the input array with the binary data that packs two integers as ascii
    # string.
    input_string = base64.b64encode(struct.pack("<ii", 1001, 1002)).decode(
        "ascii"
    )
    numpy_string_array = np.array([input_string], dtype=np.dtypes.StringDType())
    sharding = jax.sharding.SingleDeviceSharding(cpu_devices[0])
    x = jax.device_put(numpy_string_array, device=sharding)

    out = f(x)
    out = jax.device_get(out)

    # Should have gotten the binary data with the incremented integers as a
    # ascii string.
    out_ints = struct.unpack("<ii", base64.b64decode(out[0].encode("ascii")))
    self.assertEqual(out_ints[0], 1002)
    self.assertEqual(out_ints[1], 1003)

  def testDetectInvalidMeshDevice(self):
    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    if jax.local_devices()[0].id == cpu_devices[0].id:
      self.skipTest(
          "This test only works in a setup where accelerator and CPU devices"
          " use different device IDs."
      )

    # mesh contains non-CPU devices. To be used in colocated Python, it should
    # have contained CPU devices only.
    mesh = jax.sharding.Mesh(jax.local_devices(), "x")
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    @colocated_python.colocated_python
    def make_zero() -> jax.Array:
      return jax.make_array_from_callback((), sharding, lambda _: np.array(0))

    with self.assertRaisesRegex(ValueError, "Invalid device ID"):
      make_zero = make_zero.specialize(devices=cpu_devices)
      jax.block_until_ready(make_zero())

  def testObjectLifecycle(self):
    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    sharding = jax.sharding.SingleDeviceSharding(cpu_devices[0])

    @colocated_python.colocated_python_class
    class Object:

      def __init__(self) -> None:
        colocated_python._testing_initialized = True

      def __del__(self) -> None:
        colocated_python._testing_destroyed = True

      # TODO(hyeontaek): Support method calls with no arguments and remove
      # `x` parameter.
      def echo(self, x: jax.Array) -> jax.Array:
        return x

    @colocated_python.colocated_python
    def check_initialized() -> jax.Array:
      initialized = getattr(colocated_python, "_testing_initialized", False)
      return jax.device_put(np.array(initialized), sharding)

    @colocated_python.colocated_python
    def check_destroyed() -> jax.Array:
      destroyed = getattr(colocated_python, "_testing_destroyed", False)
      return jax.device_put(np.array(destroyed), sharding)

    @colocated_python.colocated_python
    def cleanup():
      if "_testing_initialized" in colocated_python.__dict__:
        del colocated_python._testing_initialized
      if "_testing_destroyed" in colocated_python.__dict__:
        del colocated_python._testing_destroyed

    check_initialized = check_initialized.specialize(devices=cpu_devices[:1])
    check_destroyed = check_destroyed.specialize(devices=cpu_devices[:1])
    cleanup = cleanup.specialize(devices=cpu_devices[:1])

    try:
      # Object initialization is deferred until the first method call.
      obj = Object()
      self.assertEqual(jax.device_get(check_initialized()), False)
      self.assertEqual(jax.device_get(check_destroyed()), False)

      # If the object is destroyed without any method calls, the object is
      # destroyed without initialization.
      del obj
      self.assertEqual(jax.device_get(check_initialized()), False)
      self.assertEqual(jax.device_get(check_destroyed()), False)
    finally:
      cleanup()

    try:
      # Object initialization is deferred until the first method call.
      obj = Object()
      self.assertEqual(jax.device_get(check_initialized()), False)
      self.assertEqual(jax.device_get(check_destroyed()), False)

      # The first method call on a process triggers object initialization there.
      x = np.array(1)
      x = jax.device_put(x, sharding)
      obj.echo(x)
      self.assertEqual(jax.device_get(check_initialized()), True)
      self.assertEqual(jax.device_get(check_destroyed()), False)

      del obj
      self.assertEqual(jax.device_get(check_initialized()), True)
      self.assertEqual(jax.device_get(check_destroyed()), True)
    finally:
      cleanup()

  def testStatefulObject(self):
    cpu_devices = _colocated_cpu_devices(jax.local_devices())

    @colocated_python.colocated_python_class
    class Value:

      def __init__(self, initial_value: np.ndarray) -> None:
        self.value = initial_value

      def add(self, x: jax.Array) -> jax.Array:
        self.value += np.asarray(x)
        return jax.device_put(self.value, x.sharding)

      # TODO(hyeontaek): Support method calls with no arguments and remove
      # `x` parameter.
      def fetch(self, x: jax.Array) -> jax.Array:
        return jax.device_put(self.value, x.sharding)

    value = Value(np.array(5))

    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])

    out = jax.device_get(value.add(x))
    self.assertEqual(out, np.array(6))

    out = jax.device_get(value.add(x))
    self.assertEqual(out, np.array(7))

    out = jax.device_get(value.fetch(x))
    self.assertEqual(out, np.array(7))

  def testObjectWithCapturedSharding(self):
    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    if len(cpu_devices) < 2:
      self.skipTest(f"Need at least two CPU devices, got: {len(cpu_devices)}")

    mesh = jax.sharding.Mesh(cpu_devices[0:2], "x")
    sharding1 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    sharding2 = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x")
    )

    @colocated_python.colocated_python_class
    class Value:

      def __init__(self, initial_value: np.ndarray) -> None:
        self.value = initial_value
        # Captured shardings in the closure.
        self.sharding1 = sharding1
        self.sharding2 = sharding2

      def add_sharding1(self, x: jax.Array) -> jax.Array:
        self.value += np.asarray(x)
        return jax.device_put(self.value, self.sharding1)

      def add_sharding2(self, x: jax.Array) -> jax.Array:
        self.value += np.asarray(x)
        return jax.device_put(self.value, self.sharding2)

    value = Value(np.array([5, 15]))

    x = np.array([1])
    x = jax.device_put(
        x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    )

    out = value.add_sharding1(x)
    self.assertEqual(out.sharding, sharding1)
    out = jax.device_get(out)
    self.assertArraysEqual(out, np.array([6, 16]))

    out = value.add_sharding2(x)
    self.assertEqual(out.sharding, sharding2)
    out = jax.device_get(out)
    self.assertArraysEqual(out, np.array([7, 17]))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
