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

"""Multihost tests for pjit."""

import collections
from concurrent import futures
import contextlib
import functools
import io
import math
import unittest

import jax
from jax import numpy as jnp
from jax._src import array
from jax._src import debugging
from jax._src import pjit
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec as P
import numpy as np

X_SIZE = 2
Y_SIZE = 2
CHIPS_SIZE = 2
ALL_AXES = ("x", "y", "chips")


def sorted_devices():
  devices = sorted(
      jax.devices(), key=lambda d: (d.host_id, getattr(d, "core_on_chip", 0))
  )
  if len(devices) != 8:
    raise unittest.SkipTest("Test assumes that it runs on a TPU donut")
  return devices


@contextlib.contextmanager
def use_default_mesh():
  devices = sorted_devices()
  mesh_devices = np.array(devices).reshape((X_SIZE, Y_SIZE, CHIPS_SIZE))
  with jax.sharding.Mesh(mesh_devices, ("x", "y", "chips")):
    yield


def create_2d_non_contiguous_mesh():
  devices = sorted_devices()
  device_mesh = np.array([
      [devices[0], devices[2]],
      [devices[3], devices[1]],
      [devices[4], devices[6]],
      [devices[7], devices[5]],
  ])
  # On TPUv3, the mesh looks like this (the integers are process index):
  #   0 1
  #   1 0
  #   2 3
  #   3 2
  return jax.sharding.Mesh(device_mesh, ("x", "y"))


def create_2d_non_contiguous_mesh2():
  devices = sorted_devices()
  device_mesh = np.array([
      [devices[0], devices[2]],
      [devices[1], devices[3]],
      [devices[4], devices[6]],
      [devices[5], devices[7]],
  ])
  # On TPUv3, the mesh looks like this (the integers are process index):
  #   0 1
  #   0 1
  #   2 3
  #   2 3
  return jax.sharding.Mesh(device_mesh, ("x", "y"))


# TODO(apaszke): Test with mesh that has host-tiled axes (especially nesting!)
class PJitTestMultiHost(jt_multiprocess.MultiProcessTest):

  @jtu.ignore_warning(category=DeprecationWarning)
  def testLocalInputsWithJaxArray(self):
    # Note that this is too small to shard over the global mesh, but fine for
    # the local mesh and so should be accepted.
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    elems_per_host = 4
    x = jnp.arange(elems_per_host) + jax.process_index() * elems_per_host
    iar = jax.sharding.PartitionSpec("x")
    oar = jax.sharding.PartitionSpec("x")
    with mesh:
      f = pjit.pjit(lambda x, y: (x, y), in_shardings=iar, out_shardings=oar)
      gx = jax.experimental.multihost_utils.host_local_array_to_global_array(
          (x, x), mesh, iar
      )
      global_out = f(*gx)
      out1, out2 = (
          jax.experimental.multihost_utils.global_array_to_host_local_array(
              global_out, mesh, oar
          )
      )
      np.testing.assert_array_equal(out1, x)
      np.testing.assert_array_equal(out2, x)


class ArrayPjitMultiHost(jt_multiprocess.MultiProcessTest):

  def test_pjit_array_single_output(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    global_input_shape = (8, 2)
    mesh_axes = jax.sharding.PartitionSpec("x", "y")
    global_input_data = np.arange(math.prod(global_input_shape)).reshape(
        global_input_shape
    )
    s = jax.sharding.NamedSharding(global_mesh, mesh_axes)

    arr = array.make_array_from_callback(
        global_input_shape, s, lambda idx: global_input_data[idx]
    )

    @functools.partial(pjit.pjit, out_shardings=s)
    def f(x):
      return x @ x.T

    expected_matrix_mul = global_input_data @ global_input_data.T

    out = f(arr)
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertEqual(out.shape, (8, 8))
    self.assertEqual(out.addressable_shards[0].data.shape, (2, 4))
    for s in out.addressable_shards:
      np.testing.assert_array_equal(
          np.asarray(s.data), expected_matrix_mul[s.index]
      )

  # Test does not work with non-contiguous device IDs.
  @jtu.skip_on_devices("cpu")
  def test_pjit_array_non_contiguous_mesh_2d(self):
    global_mesh = create_2d_non_contiguous_mesh()
    global_input_shape = (8, 2)
    pspec = jax.sharding.PartitionSpec("x", "y")
    input_data = np.arange(math.prod(global_input_shape)).reshape(
        global_input_shape
    )
    in_sharding = jax.sharding.NamedSharding(global_mesh, pspec)
    out_sharding = jax.sharding.NamedSharding(global_mesh, pspec)

    a1 = array.make_array_from_callback(
        global_input_shape, in_sharding, lambda idx: input_data[idx]
    )

    # device_id -> (index, replica_id)
    expected_idx_rid = {
        0: ((slice(0, 2), slice(0, 1)), 0),
        1: ((slice(2, 4), slice(1, 2)), 0),
        2: ((slice(0, 2), slice(1, 2)), 0),
        3: ((slice(2, 4), slice(0, 1)), 0),
        4: ((slice(4, 6), slice(0, 1)), 0),
        5: ((slice(6, 8), slice(1, 2)), 0),
        6: ((slice(4, 6), slice(1, 2)), 0),
        7: ((slice(6, 8), slice(0, 1)), 0),
    }

    with global_mesh:
      f = pjit.pjit(lambda x: x, out_shardings=out_sharding)
      out = f(a1)

      for s in out.addressable_shards:
        device_id = s.device.id
        expected_index = expected_idx_rid[device_id][0]
        expected_replica_id = expected_idx_rid[device_id][1]
        self.assertEqual(s.index, expected_index)
        self.assertEqual(s.replica_id, expected_replica_id)
        self.assertEqual(s.data.shape, (2, 1))
        np.testing.assert_array_equal(s.data._value, input_data[expected_index])

    with global_mesh:
      f = pjit.pjit(lambda x: x)
      out = f(a1)

      for s in out.addressable_shards:
        device_id = s.device.id
        expected_index = expected_idx_rid[device_id][0]
        expected_replica_id = expected_idx_rid[device_id][1]
        self.assertEqual(s.index, expected_index)
        self.assertEqual(s.replica_id, expected_replica_id)
        self.assertEqual(s.data.shape, (2, 1))
        np.testing.assert_array_equal(s.data._value, input_data[expected_index])

    none_sharding = jax.sharding.NamedSharding(
        global_mesh, jax.sharding.PartitionSpec(None)
    )

    with global_mesh:
      f = pjit.pjit(
          lambda x: x, in_shardings=none_sharding, out_shardings=out_sharding
      )
      # Fully replicated values allows a non-contiguous mesh.
      out = f(input_data)
      self.assertIsInstance(out, array.ArrayImpl)

    a2 = array.make_array_from_callback(
        global_input_shape, none_sharding, lambda idx: input_data[idx]
    )

    with global_mesh:
      f = pjit.pjit(
          lambda x, y: (x, y),
          in_shardings=(none_sharding, none_sharding),
          out_shardings=(out_sharding, out_sharding),
      )
      # Fully replicated values + Array allows a non-contiguous mesh.
      out1, out2 = f(input_data, a2)
      self.assertIsInstance(out1, array.ArrayImpl)
      self.assertIsInstance(out2, array.ArrayImpl)

  def test_sharded_add(self):
    global_mesh = create_2d_non_contiguous_mesh()
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    a_s = jax.sharding.NamedSharding(global_mesh, P("x", "y"))
    b_s = jax.sharding.NamedSharding(global_mesh, P("x"))

    a = array.make_array_from_callback(
        input_shape, a_s, lambda idx: input_data[idx]
    )
    b = array.make_array_from_callback(
        input_shape, b_s, lambda idx: input_data[idx]
    )

    out = a + b
    for s in out.addressable_shards:
      np.testing.assert_array_equal(
          s.data, (input_data + input_data)[s.index]
      )

  def test_sharded_jit_add(self):
    global_mesh = create_2d_non_contiguous_mesh()
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    a_s = jax.sharding.NamedSharding(global_mesh, P("x", "y"))
    b_s = jax.sharding.NamedSharding(global_mesh, P("x"))

    a = array.make_array_from_callback(
        input_shape, a_s, lambda idx: input_data[idx]
    )
    b = array.make_array_from_callback(
        input_shape, b_s, lambda idx: input_data[idx]
    )

    out = jax.jit(lambda x, y: x + y)(a, b)
    for s in out.addressable_shards:
      np.testing.assert_array_equal(s.data, (input_data + input_data)[s.index])

  def test_sharded_copy(self):
    global_mesh = create_2d_non_contiguous_mesh()
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    s = jax.sharding.NamedSharding(global_mesh, P("x", "y"))
    arr = array.make_array_from_callback(
        input_shape, s, lambda idx: input_data[idx]
    )
    # Copy the array sharded over multiple devices across multiple processes.
    copy_arr = jnp.copy(arr)

    for c, a in zip(copy_arr.addressable_shards, arr.addressable_shards):
      self.assertNotEqual(
          c.data.unsafe_buffer_pointer(), a.data.unsafe_buffer_pointer()
      )
      self.assertEqual(c.index, a.index)
      self.assertEqual(c.replica_id, a.replica_id)
      self.assertEqual(c.device, a.device)
      np.testing.assert_array_equal(c.data, a.data)

  def test_sharded_mul(self):
    global_mesh = create_2d_non_contiguous_mesh()
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    a_s = jax.sharding.NamedSharding(global_mesh, P("x", "y"))

    a = array.make_array_from_callback(
        input_shape, a_s, lambda idx: input_data[idx]
    )

    out = a @ a.T
    for s in out.addressable_shards:
      np.testing.assert_array_equal(
          s.data, (input_data @ input_data.T)[s.index]
      )

  def test_pjit_array_eval_shape(self):
    with jtu.create_mesh((8,), "x"):

      @functools.partial(
          pjit.pjit,
          in_shardings=jax.sharding.PartitionSpec(None),
          out_shardings=jax.sharding.PartitionSpec("x"),
      )
      def f():
        return jnp.zeros([32, 10])

      self.assertEqual(f().shape, (32, 10))
      self.assertEqual(jax.eval_shape(f).shape, (32, 10))

  def test_trace_with_global_avals(self):
    devices = sorted_devices()
    mesh_devices = np.array(devices[::2] + devices[1::2])
    # The device order in the below mesh is:
    #   (0, 2, 4, 6, 1, 3, 5, 7)
    # each having the following process index:
    #   (0, 1, 2, 3, 0, 1, 2, 3)
    # self.assertListEqual([d.process_index for d in mesh_devices],
    #                      [0, 1, 2, 3, 0, 1, 2, 3])
    global_mesh = jax.sharding.Mesh(mesh_devices, ("x",))
    x = jnp.arange(16)

    def check_shape(x):
      self.assertEqual(x.shape, (16,))
      return x

    with global_mesh:
      f = pjit.pjit(
          check_shape,
          in_shardings=jax.sharding.PartitionSpec("x"),
          out_shardings=None,
      )
      np.testing.assert_array_equal(f(x), jnp.arange(16))

  @use_default_mesh()
  def test_pjit_in_pjit(self):
    # The global size of x is 16. The shape should remain constant i.e. (16,)
    # within all `pjit`'s since with Array, pjit only accepts global shaped
    # inputs and doesn't lift the shape.
    x = jnp.arange(16)

    def pjit_all(f):
      return pjit.pjit(
          f,
          in_shardings=jax.sharding.PartitionSpec(ALL_AXES),
          out_shardings=jax.sharding.PartitionSpec(ALL_AXES),
      )

    def check_shape(x):
      assert x.shape == (16,)
      return x

    pjit_all(check_shape)(x)
    pjit_all(pjit_all(check_shape))(x)
    pjit_all(pjit_all(pjit_all(check_shape)))(x)

  def test_compile_parallel(self):
    x = jnp.arange(16)
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))

    def _lower_compile(inp):
      with global_mesh:
        f = pjit.pjit(
            lambda x: x.sum(),
            in_shardings=jax.sharding.PartitionSpec("x"),
            out_shardings=None,
        )
        exe = f.lower(inp).compile()
        return exe

    with futures.ThreadPoolExecutor(max_workers=5) as executor:
      result = executor.map(_lower_compile, [x] * 5)

    expected_out = np.arange(16).sum()

    for out in list(result):
      np.testing.assert_array_equal(out(x), expected_out)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message='`with mesh:` context manager')
  def test_fully_sharded_on_all_devices(self):
    if jax.local_device_count() > 1:
      self.skipTest("This test only works with 1 process per device.")
    num_devices = jax.device_count()
    x = jnp.arange(num_devices)
    global_mesh = jtu.create_mesh((num_devices,), "x")
    with global_mesh:
      f = pjit.pjit(
          lambda x: x,
          in_shardings=jax.sharding.PartitionSpec("x"),
          out_shardings=jax.sharding.PartitionSpec("x"),
      )
      out = f(x)
      expected_out = np.arange(num_devices)
      for s in out.addressable_shards:
        np.testing.assert_array_equal(s.data, expected_out[s.index])

  def test_on_device_size_in_bytes(self):
    global_mesh = create_2d_non_contiguous_mesh()
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    a_s = jax.sharding.NamedSharding(global_mesh, P("x", "y"))

    a = array.make_array_from_callback(
        input_shape, a_s, lambda idx: input_data[idx]
    )
    shard_size = a.addressable_shards[0].data.on_device_size_in_bytes()
    self.assertGreaterEqual(shard_size, 4 * 2)
    self.assertEqual(
        shard_size * len(a.global_shards), a.on_device_size_in_bytes()
    )

  def test_numpy_input_error_with_non_trivial_sharding(self):
    global_mesh = jtu.create_mesh((8,), "x")
    inp = np.arange(8)
    with global_mesh:
      f = pjit.pjit(
          lambda x: x,
          in_shardings=jax.sharding.PartitionSpec(None),
          out_shardings=jax.sharding.PartitionSpec(None),
      )
      out = f(inp)
      np.testing.assert_array_equal(out, inp)

      # If no in_axis_resources are specified, then pjit assumes that the
      # numpy input is fully replicated.
      f = pjit.pjit(lambda x: x, out_shardings=jax.sharding.PartitionSpec(None))
      out = f(inp)
      np.testing.assert_array_equal(out, inp)

      f = pjit.pjit(
          lambda x: x,
          in_shardings=jax.sharding.PartitionSpec("x"),
          out_shardings=jax.sharding.PartitionSpec("x"),
      )
      with self.assertRaisesRegex(
          ValueError,
          "Passing non-trivial shardings for numpy inputs is not allowed",
      ):
        f(inp)

  def test_non_contiguous_mesh_fetch_to_host(self):
    if jax.local_device_count() != 2:
      raise unittest.SkipTest("Test assumes 2 devices per process")
    global_mesh = create_2d_non_contiguous_mesh()
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    a_s = jax.sharding.NamedSharding(global_mesh, P(None, "y"))
    a = array.make_array_from_callback(
        input_shape, a_s, lambda idx: input_data[idx]
    )
    np.testing.assert_array_equal(a, input_data)

  def test_non_contiguous_mesh_fetch_to_host2(self):
    global_mesh = create_2d_non_contiguous_mesh2()
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    a_s = jax.sharding.NamedSharding(global_mesh, P(None, "y"))
    a = array.make_array_from_callback(
        input_shape, a_s, lambda idx: input_data[idx]
    )
    with self.assertRaisesRegex(
        RuntimeError,
        r"Fetching value for `jax.Array` that spans non-addressable \(non"
        r" process local\) devices is not possible",
    ):
      _ = a._value

  def test_no_python_shard_arg_fallback(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    input_shape = (8, 2)
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    a_s = jax.sharding.NamedSharding(global_mesh, P("x", "y"))
    arr = array.make_array_from_callback(
        input_shape, a_s, lambda idx: input_data[idx])

    @jax.jit
    def f(x):
      return x * 2

    with jtu.count_jax_array_shard_arg_calls() as count:
      f(arr)
      f(arr)
    self.assertEqual(count(), 1)


@contextlib.contextmanager
def capture_stdout():
  with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as fp:

    def _read() -> str:
      return fp.getvalue()

    yield _read


class MultiHostDebuggingTest(jt_multiprocess.MultiProcessTest):

  def _assert_lines_equal(self, text1, text2):
    def _count(lines):
      return collections.Counter(lines)

    self.assertDictEqual(_count(text1.split("\n")), _count(text2.split("\n")))

  @use_default_mesh()
  def test_print_in_multihost_pjit_array(self):
    x = jnp.arange(16, dtype=jnp.int32)

    def f(x):
      debugging.debug_print("{}", x, ordered=False)
      return x

    f = pjit.pjit(
        f,
        in_shardings=jax.sharding.PartitionSpec(ALL_AXES),
        out_shardings=jax.sharding.PartitionSpec(ALL_AXES),
    )
    with capture_stdout() as output:
      f(x)
      jax.effects_barrier()
    if jax.process_index() == 0:
      self.assertEqual(output(), f"{np.arange(16, dtype=np.int32)}\n")
    else:
      self.assertEqual(output(), "")

  def test_print_in_multihost_shard_map(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("i",))
    num_devices = jax.local_device_count()
    local_x = (
        jnp.arange(num_devices, dtype=jnp.int32)
        + jax.process_index() * num_devices
    )
    global_shape = (jax.device_count(),)
    sharding = jax.NamedSharding(mesh, jax.P("i"))
    global_x = jax.make_array_from_process_local_data(sharding, local_x, global_shape)

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=jax.P("i"), out_specs=jax.P("i"))
    def f(x):
      debugging.debug_print("{}", x[0], ordered=False)
      return x

    with capture_stdout() as output:
      out = f(global_x)
      out.block_until_ready()
      jax.effects_barrier()

    lines = [f"{i}" for i in local_x] + [""]
    self._assert_lines_equal(output(), "\n".join(lines))

if __name__ == "__main__":
  jt_multiprocess.main()
