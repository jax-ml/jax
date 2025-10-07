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
  HAS_CLOUDPICKLE = True
except (ModuleNotFoundError, ImportError):
  HAS_CLOUDPICKLE = False


_count_colocated_python_specialization_cache_miss = jtu.count_events(
    "colocated_python_func._get_specialized_func"
)


class ColocatedPythonTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not HAS_CLOUDPICKLE:
      self.skipTest(
        "ColocatedPythonTest depends on cloudpickle library"
      )
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
      self.skipTest(
        "Serialization in Colocated Python needs StringDType, and thus"
        " requires NumPy 2.0.0 or later"
      )

  def test_colocated_cpu_devices(self):
    mesh = jax.sharding.Mesh(
        np.array(jax.local_devices()[:1]).reshape((1, 1)), ("x", "y")
    )
    cpu_mesh1 = colocated_python.colocated_cpu_devices(mesh)

    cpu_devices = colocated_python.colocated_cpu_devices(
        jax.local_devices()[:1]
    )
    cpu_mesh2 = jax.sharding.Mesh(
        np.array(cpu_devices).reshape((1, 1)), ("x", "y")
    )
    self.assertEqual(cpu_mesh1, cpu_mesh2)

  def test_serialization_roundtrip(self):
    cpu_devices = colocated_python.colocated_cpu_devices(
        jax.local_devices()[:1])

    mesh = jax.sharding.Mesh(np.array(cpu_devices).reshape((1, 1)), ("x", "y"))
    self.assertEqual(
        serialization._deserialize(serialization._serialize(mesh)), mesh)

    sharding1 = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x"))
    self.assertEqual(
        serialization._deserialize(serialization._serialize([sharding1])),
        [sharding1])

    sharding2 = jax.sharding.SingleDeviceSharding(
        cpu_devices[0], memory_kind="pinned_host")
    self.assertEqual(
        serialization._deserialize(serialization._serialize((sharding2,))),
        (sharding2,))

    def func(x):
      return x + 1

    self.assertEqual(
        serialization._deserialize(serialization._serialize(func))(1), func(1))

  def test_make_colocated_python_program(self):
    def add_one(x):
      return x + 1

    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
    sharding = jax.sharding.SingleDeviceSharding(cpu_devices[0])
    sds = jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding)

    fun_and_specialization = (
        add_one,
        None,  # dummy in_specs_treedef
        None,  # dummy in_specs_leaves
        None,  # dummy out_specs_treedef
        None,  # dummy out_specs_leaves
        None,  # dummy devices
    )
    pickled_function = serialization._serialize(fun_and_specialization)
    program = ifrt_programs.make_colocated_python_program(
        "add_one", pickled_function, [cpu_devices[0]], [sds], [sds]
    )
    del program

  def test_serialize_with_shared_obj(self):
    cpu_devices = colocated_python.colocated_cpu_devices(
        jax.local_devices()[:1])
    mesh = jax.sharding.Mesh(
        np.array(cpu_devices).reshape((1, 1)),
        ("long_axis_name_1", "long_axis_name_2"))
    sharding1 = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("long_axis_name_1"))
    sharding2 = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("long_axis_name_2"))

    serialized1 = serialization._serialize([sharding1])
    serialized2 = serialization._serialize([sharding1, sharding2])
    serialized3 = serialization._serialize([sharding1, sharding1])

    # The total serialized size of two shardings of a shared mesh should be less
    # than twice the serialized size of a single sharding.
    self.assertLess(len(serialized2), len(serialized1) * 2)

    # The total serialized size of two identical shardings should be less than
    # that of two shardings that only share the mesh.
    self.assertLess(len(serialized3), len(serialized2))

    self.assertEqual(serialization._deserialize(serialized1), [sharding1])
    self.assertEqual(
        serialization._deserialize(serialized2), [sharding1, sharding2])
    self.assertEqual(
        serialization._deserialize(serialized3), [sharding1, sharding1])

  def test_simple_function(self):
    @colocated_python.colocated_python
    def add_one(x):
      return x + 1

    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  def test_simple_function_with_tree(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  def test_empty_input_fails_without_specialization(self):
    @colocated_python.colocated_python
    def make_zero():
      return jnp.array(0)

    with self.assertRaisesRegex(
        ValueError,
        "No devices found. colocated_python function without input arguments"
        " must be first specialized with devices."):
      _ = make_zero()

  def test_empty_input_with_devices_specialization(self):
    @colocated_python.colocated_python
    def make_zero():
      return jnp.array(0)

    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())

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

  def test_input_polymorphism_without_out_specs_fn(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  def test_input_polymorphism_allowed_with_out_specs_fn(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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
  # Cannot run concurrently with other tests using `colocated_python._testing_global_state`.
  @jtu.thread_unsafe_test()
  def test_sequential_execution(self, on_main_thread: bool):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])

    @colocated_python.colocated_python
    def func0(x: jax.Array) -> jax.Array:
      colocated_python._testing_global_state = 100
      return x

    @colocated_python.colocated_python
    def func1(x: jax.Array) -> jax.Array:
      assert "_testing_global_state" in colocated_python.__dict__
      assert colocated_python._testing_global_state == 100
      colocated_python._testing_global_state += 1
      return x

    @colocated_python.colocated_python
    def func2(x: jax.Array) -> jax.Array:
      assert "_testing_global_state" in colocated_python.__dict__
      assert colocated_python._testing_global_state == 101
      return x

    @colocated_python.colocated_python
    def cleanup(x: jax.Array) -> jax.Array:
      if "_testing_global_state" in colocated_python.__dict__:
        del colocated_python._testing_global_state
      return x

    # Specify out_specs_fn so that their executions are asynchronously
    # dispatched.
    func0 = func0.specialize(out_specs_fn=lambda x: x)
    func1 = func1.specialize(out_specs_fn=lambda x: x)
    func2 = func2.specialize(out_specs_fn=lambda x: x)

    def calls(x: jax.Array) -> None:
      # No explicit blocking before making the next call.
      func0(x)
      func1(x)
      jax.block_until_ready(func2(x))

    try:
      # Executions in `calls` should run sequentially.
      if on_main_thread:
        calls(x)
      else:
        t = threading.Thread(target=calls, args=(x,))
        t.start()
        t.join()
      # Executions should succeed without an error.
    finally:
      jax.block_until_ready(cleanup(x))

  # Cannot run concurrently with other tests using `colocated_python._testing_global_state`.
  @jtu.thread_unsafe_test()
  def test_concurrent_execution(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])

    @colocated_python.colocated_python
    def init(x: jax.Array) -> jax.Array:
      colocated_python._testing_global_state = threading.Barrier(3)
      return x

    @colocated_python.colocated_python
    def func(x: jax.Array) -> jax.Array:
      assert "_testing_global_state" in colocated_python.__dict__
      colocated_python._testing_global_state.wait(timeout=5)
      return x

    @colocated_python.colocated_python
    def cleanup(x: jax.Array) -> jax.Array:
      if "_testing_global_state" in colocated_python.__dict__:
        del colocated_python._testing_global_state
      return x

    # Specify out_specs_fn so that their executions are asynchronously
    # dispatched.
    func = func.specialize(out_specs_fn=lambda x: x)

    try:
      jax.block_until_ready(init(x))

      # All func calls should run concurrently and enter/exit the barrier.
      t1 = threading.Thread(target=func, args=(x,))
      t2 = threading.Thread(target=func, args=(x,))
      t3 = threading.Thread(target=func, args=(x,))
      t1.start()
      t2.start()
      t3.start()
      t1.join()
      t2.join()
      t3.join()
      # Executions should succeed without a deadlock.
    finally:
      jax.block_until_ready(cleanup(x))

  def test_inputs_with_different_device_orders(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())[:2]
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

  def test_module_variable_access(self):
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

      cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  def test_string_processing(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  def test_binary_data_processing(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  def test_detect_invalid_mesh_device(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  # Cannot run concurrently with other tests using `colocated_python._testing_global_state`.
  @jtu.thread_unsafe_test()
  def test_object_lifecycle(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
    sharding = jax.sharding.SingleDeviceSharding(cpu_devices[0])
    x = jax.device_put(np.array(0), sharding)

    @colocated_python.colocated_python_class
    class Object:

      def __init__(self) -> None:
        colocated_python._testing_initialized = True

      def __del__(self) -> None:
        colocated_python._testing_destroyed = True

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
    def cleanup(x: jax.Array) -> jax.Array:
      if "_testing_initialized" in colocated_python.__dict__:
        del colocated_python._testing_initialized
      if "_testing_destroyed" in colocated_python.__dict__:
        del colocated_python._testing_destroyed
      return x

    check_initialized = check_initialized.specialize(devices=cpu_devices[:1])
    check_destroyed = check_destroyed.specialize(devices=cpu_devices[:1])

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
      jax.block_until_ready(cleanup(x))

    try:
      # Object initialization is deferred until the first method call.
      obj = Object()
      self.assertEqual(jax.device_get(check_initialized()), False)
      self.assertEqual(jax.device_get(check_destroyed()), False)

      # The first method call on a process triggers object initialization there.
      x = np.array(1)
      x = jax.device_put(x, sharding)
      jax.block_until_ready(obj.echo(x))
      self.assertEqual(jax.device_get(check_initialized()), True)
      self.assertEqual(jax.device_get(check_destroyed()), False)

      del obj
      self.assertEqual(jax.device_get(check_initialized()), True)
      self.assertEqual(jax.device_get(check_destroyed()), True)
    finally:
      jax.block_until_ready(cleanup(x))

  def test_stateful_object(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())

    @colocated_python.colocated_python_class
    class Value:

      def __init__(self, initial_value: np.ndarray) -> None:
        self.value = initial_value

      def add(self, x: jax.Array) -> jax.Array:
        self.value += np.asarray(x)
        return jax.device_put(self.value, x.sharding)

      def fetch_like(self, x: jax.Array) -> jax.Array:
        return jax.device_put(self.value, x.sharding)

    value = Value(np.array(5))

    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])

    out = jax.device_get(value.add(x))
    self.assertEqual(out, np.array(6))

    out = jax.device_get(value.add(x))
    self.assertEqual(out, np.array(7))

    out = jax.device_get(value.fetch_like(x))
    self.assertEqual(out, np.array(7))

  def test_object_with_captured_sharding(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
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

  def test_object_method_specialization(self):
    cpu_devices = colocated_python.colocated_cpu_devices(jax.local_devices())
    cpu_devices = cpu_devices[:1]
    sharding = jax.sharding.SingleDeviceSharding(cpu_devices[0])

    @colocated_python.colocated_python_class
    class Object:

      def __init__(self, sharding: jax.sharding.Sharding) -> None:
        self.sharding = sharding

      def fetch_with_devices(self) -> jax.Array:
        return jax.device_put(np.array(1, dtype=np.int32), self.sharding)

      def fetch_with_output_spec(self) -> np.ndarray:
        return jax.device_put(np.array(1, dtype=np.int32), self.sharding)

    obj = Object(sharding)

    with self.assertRaisesRegex(
        ValueError,
        "No devices found. colocated_python function without input arguments"
        " must be first specialized with devices."):
      jax.block_until_ready(obj.fetch_with_devices())

    with self.assertRaisesRegex(
        ValueError,
        "No devices found. colocated_python function without input arguments"
        " must be first specialized with devices."):
      jax.block_until_ready(obj.fetch_with_output_spec())

    obj.fetch_with_devices = (
        obj.fetch_with_devices.specialize(devices=cpu_devices))
    out = obj.fetch_with_devices()
    self.assertArraysEqual(out, np.array(1, dtype=np.int32))

    # TODO(hyeontaek): Infer `devices` from the output spec computed using the
    # output spec function.
    obj.fetch_with_output_spec = obj.fetch_with_output_spec.specialize(
        devices=cpu_devices,
        out_specs_fn=lambda: jax.ShapeDtypeStruct(
            shape=(), dtype=np.int32, sharding=sharding))
    out = obj.fetch_with_output_spec()
    self.assertArraysEqual(out, np.array(1, dtype=np.int32))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
