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

"""Multihost tests for jax.Array."""

import math
import unittest

from absl.testing import parameterized
import jax
from jax._src import array
from jax._src import sharding_impls
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np


def create_array(shape, arr_sharding, global_data=None):
  if global_data is None:
    global_data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

  return array.make_array_from_callback(
      shape, arr_sharding, lambda idx: global_data[idx],
      dtype=global_data.dtype), global_data


def create_nonaddressable_array(shape, spec=None):
  """Creates an array that is non-addressable in half of the processes.

  Args:
    shape: Shape of the array.
    spec: Sharding spec of the array. If None, the array is sharded over all
      participating devices.

  Returns:
    A tuple of the created array and the global data.
  """
  n_dev = len(jax.devices()) // 2
  mesh = jax.make_mesh((n_dev,), ("x",), devices=jax.devices()[:n_dev],
                       axis_types=(jax.sharding.AxisType.Explicit,))
  if spec is None:
    spec = P("x")
  s = jax.sharding.NamedSharding(mesh, spec)
  return create_array(shape, s)


class ArrayTestMultiHost(jt_multiprocess.MultiProcessTest):

  @parameterized.named_parameters(
      (
          "mesh_x_y",
          P("x", "y"),
          # device_id -> (index, replica_id)
          {
              0: ((slice(0, 2), slice(0, 1)), 0),
              1: ((slice(0, 2), slice(1, 2)), 0),
              2: ((slice(2, 4), slice(0, 1)), 0),
              3: ((slice(2, 4), slice(1, 2)), 0),
              4: ((slice(4, 6), slice(0, 1)), 0),
              5: ((slice(4, 6), slice(1, 2)), 0),
              6: ((slice(6, 8), slice(0, 1)), 0),
              7: ((slice(6, 8), slice(1, 2)), 0),
          },
          (2, 1),
          False,
          False,
      ),
      (
          "mesh_x",
          P("x"),
          # device_id -> (index, replica_id)
          {
              0: ((slice(0, 2), slice(None)), 0),
              1: ((slice(0, 2), slice(None)), 1),
              2: ((slice(2, 4), slice(None)), 0),
              3: ((slice(2, 4), slice(None)), 1),
              4: ((slice(4, 6), slice(None)), 0),
              5: ((slice(4, 6), slice(None)), 1),
              6: ((slice(6, 8), slice(None)), 0),
              7: ((slice(6, 8), slice(None)), 1),
          },
          (2, 2),
          False,
          False,
      ),
      (
          "mesh_y",
          P("y"),
          # device_id -> (index, replica_id)
          {
              0: ((slice(0, 4), slice(None)), 0),
              1: ((slice(4, 8), slice(None)), 0),
              2: ((slice(0, 4), slice(None)), 1),
              3: ((slice(4, 8), slice(None)), 1),
              4: ((slice(0, 4), slice(None)), 2),
              5: ((slice(4, 8), slice(None)), 2),
              6: ((slice(0, 4), slice(None)), 3),
              7: ((slice(4, 8), slice(None)), 3),
          },
          (4, 2),
          False,
          True,
      ),
      (
          "mesh_xy",
          P(("x", "y")),
          # device_id -> (index, replica_id)
          {
              0: ((slice(0, 1), slice(None)), 0),
              1: ((slice(1, 2), slice(None)), 0),
              2: ((slice(2, 3), slice(None)), 0),
              3: ((slice(3, 4), slice(None)), 0),
              4: ((slice(4, 5), slice(None)), 0),
              5: ((slice(5, 6), slice(None)), 0),
              6: ((slice(6, 7), slice(None)), 0),
              7: ((slice(7, 8), slice(None)), 0),
          },
          (1, 2),
          False,
          False,
      ),
      (
          "mesh_fully_replicated",
          P(),
          # device_id -> (index, replica_id)
          {
              0: ((slice(None), slice(None)), 0),
              1: ((slice(None), slice(None)), 1),
              2: ((slice(None), slice(None)), 2),
              3: ((slice(None), slice(None)), 3),
              4: ((slice(None), slice(None)), 4),
              5: ((slice(None), slice(None)), 5),
              6: ((slice(None), slice(None)), 6),
              7: ((slice(None), slice(None)), 7),
          },
          (8, 2),
          True,
          True,
      ),
  )
  # Test does not work with non-contiguous device IDs.
  @jtu.skip_on_devices("cpu")
  def test_array_2d_shard(self, pspec, expected_idx_rid,
                          expected_shard_shape, expected_is_fully_replicated,
                          fetch_to_host):
    if jtu.is_device_tpu("5", "e"):
      raise unittest.SkipTest("Test fails on v5e")
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"), iota_order=True)
    global_input_shape = (8, 2)
    arr, global_input_data = create_array(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, pspec))

    self.assertEqual(arr.is_fully_replicated, expected_is_fully_replicated)

    for s in arr.addressable_shards:
      sd = s.device.id
      expected_index = expected_idx_rid[sd][0]
      expected_replica_id = expected_idx_rid[sd][1]
      self.assertEqual(s.index, expected_index)
      self.assertEqual(s.replica_id, expected_replica_id)
      self.assertEqual(s.data.shape, expected_shard_shape)
      np.testing.assert_array_equal(np.asarray(s.data),
                                    global_input_data[expected_index])

    for s in arr.global_shards:
      sd = s.device.id
      expected_index = expected_idx_rid[sd][0]
      expected_replica_id = expected_idx_rid[sd][1]
      self.assertEqual(s.index, expected_index)
      self.assertEqual(s.replica_id, expected_replica_id)
      if s.data is not None:
        self.assertEqual(s.data.shape, expected_shard_shape)
        np.testing.assert_array_equal(np.asarray(s.data),
                                      global_input_data[expected_index])

    if fetch_to_host:
      np.testing.assert_array_equal(arr._value, global_input_data)
    else:
      with self.assertRaisesRegex(
          RuntimeError,
          r"Fetching value for `jax.Array` that spans non-addressable \(non"
          r" process local\) devices is not possible",
      ):
        _ = arr._value

  @parameterized.named_parameters(
      ("mesh_x_y_z", P("x", "y", "z"), (4, 2, 1), False),
      ("mesh_xy_z", P(("x", "y"), "z"), (2, 2, 2), False),
      ("mesh_z", P("z"), (4, 4, 2), True),
      ("mesh_None_z", P(None, None, "z"), (8, 4, 1), True),
  )
  def test_array_3d_shard(self, pspec, expected_shard_shape, fetch_to_host):
    if jtu.is_device_tpu("5", "e"):
      raise unittest.SkipTest("Test fails on v5e")
    global_mesh = jtu.create_mesh((2, 2, 2), ("x", "y", "z"))
    global_input_shape = (8, 4, 2)
    arr, global_input_data = create_array(
        global_input_shape, jax.sharding.NamedSharding(global_mesh, pspec))

    self.assertEqual(arr.ndim, 3)
    self.assertEqual(arr.size, 64)
    self.assertEqual(arr.addressable_data(0).shape, expected_shard_shape)
    if fetch_to_host:
      np.testing.assert_array_equal(arr._value, global_input_data)
    else:
      with self.assertRaisesRegex(
          RuntimeError,
          r"Fetching value for `jax.Array` that spans non-addressable \(non"
          r" process local\) devices is not possible",
      ):
        _ = arr._value

  def test_sharded_zeros_like(self):
    if jtu.is_device_tpu("5", "e"):
      raise unittest.SkipTest("Test fails on v5e")
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    input_shape = (8, 2)
    a, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P("x", "y")))
    out = jnp.zeros_like(a)
    expected = np.zeros_like(input_data)
    self.assertLen(out.addressable_shards, 2)
    for i in out.addressable_shards:
      np.testing.assert_array_equal(i.data, expected[i.index])
      self.assertEqual(i.replica_id, 0)
      self.assertEqual(i.data.shape, (2, 1))

  @parameterized.product(
      spec=[
          (("a", "b", "c"),),
          (("a", "b"), "c"),
          (("a", "b"),),
          (("a",),),
          (("b",),),
          (("c",),),
          ((),),
      ],
      infer_shape=[True, False],
  )
  def test_make_array_from_process_data(self, spec, infer_shape):
    mesh = jtu.create_mesh((2, 2, 2), ("a", "b", "c"), iota_order=True)
    # Key: number of processes. Value: axes corresponding to hosts.
    host_axes_dict = {1: (), 2: ("a",), 4: ("a", "b"), 8: ("a", "b", "c")}

    host_axes = set(host_axes_dict[jax.process_count()])
    axis0_spec = (spec[0],) if isinstance(spec[0], str) else spec[0]
    expected_process_shards = 2 ** len(host_axes.intersection(axis0_spec))
    sharding = jax.sharding.NamedSharding(mesh, P(*spec))
    replicated = jax.sharding.NamedSharding(mesh, P(None))
    num_indices0 = sharding_impls.num_addressable_indices(sharding, 0, (8, 4))
    num_indices1 = sharding_impls.num_addressable_indices(sharding, 1, (8, 4))
    global_shape = None if infer_shape else (8, 4)
    if infer_shape and num_indices1 < 4:
      # If 2nd dimension is sharded across hosts (as it is on v5e 4x2)
      # it would affect the computed global_shape, for test's simplicity we
      # set explicit global_shape global shape to be 2.
      global_shape = (8, 4)

    process_index, num_shards = sharding_impls.get_process_index_and_count(
        sharding,
        0,
        ndims=2,
    )
    self.assertEqual(num_shards, expected_process_shards)
    process_data = np.arange(4)[None, :] + 4 * process_index
    b = np.broadcast_to(process_data, (num_indices0, 4))
    r = jax.make_array_from_process_local_data(sharding, b, global_shape)
    self.assertEqual(r.shape, (8, 4))
    self.assertEqual(r.sharding, sharding)
    r = np.array(jax.jit(lambda x: x, out_shardings=replicated)(r))
    global_target = [np.arange(4) + 4 * (i * num_shards // 8) for i in range(8)]
    np.testing.assert_array_equal(sorted(r, key=lambda x: x[0]), global_target)

  def test_make_array_from_process_data_shape_inference(self):
    mesh = jtu.create_mesh((2, 2, 2), ("a", "b", "c"), iota_order=True)
    sharding = jax.sharding.NamedSharding(mesh, P(("a", "b"), "c"))
    r = jax.make_array_from_process_local_data(sharding, np.ones([4, 4]))
    self.assertEqual(r.sharding, sharding)
    process_to_target_shape = {1: (4, 4), 2: (8, 4), 4: (16, 4), 8: (16, 8)}
    target_shape = process_to_target_shape[jax.process_count()]
    self.assertEqual(target_shape, r.shape)

    # Check if we can specify that local input actually contains full-span
    # across different axes.
    r2 = jax.make_array_from_process_local_data(
        sharding, np.ones([4, 4]), global_shape=(target_shape[0], 4)
    )
    self.assertEqual(r2.sharding, sharding)
    self.assertEqual((target_shape[0], 4), r2.shape)

    r2 = jax.make_array_from_process_local_data(
        sharding, np.ones([4, 4]), global_shape=(4, target_shape[1])
    )
    self.assertEqual(r2.sharding, sharding)
    self.assertEqual((4, target_shape[1]), r2.shape)

    r2 = jax.make_array_from_process_local_data(
        sharding, np.ones([4, 4]), global_shape=(4, 4)
    )
    self.assertEqual(r2.sharding, sharding)
    self.assertEqual((4, 4), r2.shape)
    # Verify that we get not-supported message rather than non-uniform
    with self.assertRaisesRegex(ValueError, ".*supported"):
      jax.make_array_from_process_local_data(
          sharding, np.ones([4, 4]), global_shape=(4, None)
      )

  @parameterized.named_parameters(
      ("shape_none", None),
      ("shape_tuple", (16, 4)),
      ("shape_pytree", {"a": (16, 4), "b": (16, 4)}),
  )
  @jtu.run_on_devices("cpu")
  def test_make_array_from_process_local_data_pytree(self, global_shape):
    mesh = jtu.create_mesh((2, 2, 2), ("a", "b", "c"), iota_order=True)
    with jax.set_mesh(mesh):
      r = jax.make_array_from_process_local_data(
          P(("a", "b"), "c"),
          {"a": np.ones([4, 4]), "b": np.ones([4, 4])},
          global_shape=global_shape,
      )
      self.assertTupleEqual(r["a"].shape, (16, 4))
      self.assertTupleEqual(r["b"].shape, (16, 4))

  def test_multi_process_to_py(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    input_shape = (8, 2)
    a, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P(None))
    )
    self.assertIsInstance(np.asarray(a), np.ndarray)
    np.testing.assert_array_equal(np.asarray(a), input_data)

    a, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P("x"))
    )
    with self.assertRaisesRegex(
        RuntimeError,
        r"Fetching value for `jax.Array` that spans non-addressable \(non"
        r" process local\) devices is not possible.",
    ):
      np.asarray(a)

  def test_multi_process_repr(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    input_shape = (8, 2)
    a, _ = create_array(input_shape,
                        jax.sharding.NamedSharding(global_mesh, P(None)))
    val = repr(a)
    self.assertIn("Array([[ 0.,  1.]", val)

    a, _ = create_array(input_shape,
                        jax.sharding.NamedSharding(global_mesh, P("x")))
    val = repr(a)
    self.assertEqual(val, "Array(shape=(8, 2), dtype=float32)")

  def test_getitem(self):
    if jtu.is_device_tpu("5", "e"):
      raise unittest.SkipTest("Test fails on v5e")
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    input_shape = (16, 8)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P("x", "y")))

    s = arr[2:4, 0:1]
    np.testing.assert_array_equal(s, input_data[2:4, 0:1])

    p = arr[:2]
    np.testing.assert_array_equal(p, input_data[:2])

  def test_array_fully_replicated_shard(self):
    global_mesh = jtu.create_mesh((4, 2), ("x", "y"))
    inp_shape = (8, 2)
    arr, inp_data = create_array(
        inp_shape, jax.sharding.NamedSharding(global_mesh, P()))
    fs = arr._fully_replicated_shard()
    self.assertEqual(fs.shape, inp_shape)
    self.assertLen(fs.sharding.device_set, 1)
    self.assertEqual(fs.devices(), {jax.local_devices()[0]})
    np.testing.assert_array_equal(fs, inp_data)
    np.testing.assert_array_equal(arr.addressable_data(0), inp_data)

  def test_device_put_uncommitted_array(self):
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P("x", "y"))
    inp = jnp.arange(16).reshape(8, 2)
    out = jax.device_put(inp, s)

    for shard in out.addressable_shards:
      np.testing.assert_array_equal(shard.data, inp[shard.index])
    self.assertEqual(out.sharding, s)

  def test_device_put_np_array(self):
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P("x", "y"))
    inp = np.arange(16).reshape(8, 2)
    out = jax.device_put(inp, s)

    for shard in out.addressable_shards:
      np.testing.assert_array_equal(shard.data, inp[shard.index])
    self.assertEqual(out.sharding, s)

  def test_device_put_python_scalar(self):
    mesh = jtu.create_mesh((2, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P())
    out = jax.device_put(1, s)

    for shard in out.addressable_shards:
      np.testing.assert_array_equal(shard.data, 1)
    self.assertEqual(out.sharding, s)

  def test_device_put_python_scalar_different_error(self):
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P())

    with self.assertRaisesRegex(
        AssertionError,
        ".*passed to device_put is not the same on each process.*"):
      if jax.process_index() == 0:
        jax.device_put(1., s)
      else:
        jax.device_put(2., s)

  def test_device_put_uncommitted_array_different_inputs_error(self):
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P("x", "y"))

    with self.assertRaisesRegex(
        AssertionError,
        ".*passed to device_put is not the same on each process.*"):
      if jax.process_index() == 0:
        jax.device_put(jnp.arange(16).reshape(8, 2), s)
      else:
        jax.device_put(jnp.arange(16, stop=32).reshape(8, 2), s)

  def test_device_put_committed_array_error(self):
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P("x", "y"))
    inp = jax.device_put(jnp.arange(16).reshape(8, 2), jax.local_devices()[0])

    with self.assertRaisesRegex(ValueError, "device_put's second argument.*"):
      jax.device_put(inp, s)

  def test_closed_over_global_array_error(self):
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P("x", "y"))
    arr, np_inp = create_array((8, 2), s)

    @jax.jit
    def f(x):
      return x + arr

    with self.assertRaisesRegex(
        RuntimeError,
        r"Closing over jax.Array that spans non-addressable \(non process"
        r" local\) devices is not allowed"):
      f(np_inp)

  def test_zeros_like_use_mesh(self):
    mesh = jtu.create_mesh((4, 2), ("x", "y"))
    s = jax.sharding.NamedSharding(mesh, P())
    np_inp = np.array(0, dtype=np.float32)
    arr = jax.device_put(np_inp, s)

    with jax.set_mesh(mesh):
      out = jnp.zeros_like(arr)
    np.testing.assert_array_equal(out, np_inp)

  def test_sharding_process_indices_all_devices(self):
    mesh = jax.make_mesh((jax.device_count(),), ("x",), devices=jax.devices(),
                         axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))

    expected_pids = {d.process_index for d in s.device_set}
    self.assertEqual(s._internal_device_list.process_indices, expected_pids)
    self.assertLen(s._internal_device_list.process_indices, jax.process_count())


class NonaddressableArrayTestMultiHost(jt_multiprocess.MultiProcessTest):

  def test_create_nonaddressable_array(self):
    y, x = create_nonaddressable_array((8, 8))
    # The array is non-addressable in at least one process.
    self.assertLess(len(y.sharding._internal_device_list.process_indices),
                    jax.process_count())
    for a in y.addressable_shards:
      np.testing.assert_array_equal(a.data, x[a.index])

    fr, x = create_nonaddressable_array((8, 8), spec=P())
    self.assertTrue(fr.sharding.is_fully_replicated)
    self.assertLess(len(fr.sharding._internal_device_list.process_indices),
                    jax.process_count())
    if fr.sharding.has_addressable_devices:
      np.testing.assert_array_equal(x, fr)

  def test_named_sharding_is_fully_addressable(self):
    pid = 0
    ds = jax.local_devices(process_index=pid)
    mesh = jtu.create_mesh((len(ds),), ("x",))
    s = jax.sharding.NamedSharding(mesh, P("x"))
    self.assertEqual(s.is_fully_addressable, jax.process_index() == pid)

  def test_single_device_sharding_is_fully_addressable(self):
    d = jax.devices()[0]
    s = jax.sharding.SingleDeviceSharding(d)
    self.assertEqual(s.is_fully_addressable,
                     jax.process_index() == d.process_index)

  def test_array_with_no_local_shards_has_valid_layout(self):
    d = jax.devices()[0]
    s = jax.sharding.SingleDeviceSharding(d)
    shape = (8, 8)
    np_inp = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    xs = []
    if jax.process_index() == d.process_index:
      x = jax.device_put(np_inp, s)
      xs.append(x)

    arr = jax.make_array_from_single_device_arrays(
        shape, s, xs, dtype=jnp.int32)
    self.assertIsNotNone(arr.format.layout)

  def test_device_put_uncommitted_array_namedsharding(self):
    n_local = len(jax.local_devices())
    pid = 0
    mesh = jax.make_mesh(
        (n_local,), ("x",), devices=jax.local_devices(process_index=pid),
        axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))
    inp = jnp.arange(16).reshape(8, 2)
    out = jax.device_put(inp, s)

    # device_put of an uncommitted array to a sharding that is addressable only
    # in process `pid` should return an array with addressable shards only in
    # process `pid`. In other processes, the returned array has no addressable
    # shards.
    expected_num_shards = n_local if jax.process_index() == pid else 0
    self.assertLen(out.addressable_shards, expected_num_shards)
    for shard in out.addressable_shards:
      np.testing.assert_array_equal(shard.data, inp[shard.index])
    self.assertEqual(out.sharding, s)

  def test_device_put_numpy_array_namedsharding(self):
    n_local = len(jax.local_devices())
    pid = 1
    mesh = jax.make_mesh(
        (n_local,), ("x",), devices=jax.local_devices(process_index=pid),
        axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))
    inp = np.arange(16).reshape(8, 2)
    out = jax.device_put(inp, s)

    # device_put of a numpy array to a sharding that is addressable only in
    # process `pid` should return an array with addressable shards only in
    # process `pid`. In other processes, the returned array has no addressable
    # shards.
    expected_num_shards = n_local if jax.process_index() == pid else 0
    self.assertLen(out.addressable_shards, expected_num_shards)
    for shard in out.addressable_shards:
      np.testing.assert_array_equal(shard.data, inp[shard.index])
    self.assertEqual(out.sharding, s)

  def test_device_put_numpy_array_singledevice(self):
    inp = np.arange(16).reshape(8, 2)
    d = jax.devices()[0]
    out = jax.device_put(inp, d)

    # device_put of a numpy array to a sharding that is addressable only in
    # process `pid` should return an array with addressable shards only in
    # process `pid`. In other processes, the returned array has no addressable
    # shards.
    expected_num_shards = 1 if jax.process_index() == d.process_index else 0
    self.assertLen(out.addressable_shards, expected_num_shards)
    for shard in out.addressable_shards:
      np.testing.assert_array_equal(shard.data, inp[shard.index])
    self.assertEqual(out.sharding, jax.sharding.SingleDeviceSharding(d))

  def test_device_put_to_device_error(self):
    mesh = jax.make_mesh((jax.device_count(),), ("x",), devices=jax.devices(),
                         axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))
    inp = jax.device_put(jnp.arange(16).reshape(8, 2), s)

    with self.assertRaisesRegex(ValueError,
                                "must be a fully addressable array or"):
      nonlocal_pid = (jax.process_index() + 1) % jax.process_count()
      jax.device_put(inp, jax.local_devices(process_index=nonlocal_pid)[0])

  def test_make_array_from_callback(self):
    n_local = jax.local_device_count()
    pid = 1

    mesh = jax.make_mesh(
        (n_local,), ("x",), devices=jax.local_devices(process_index=pid),
        axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))

    # Create an array that is non-addressable in processes besides `pid`.
    global_data = np.arange(16, dtype=np.int32).reshape(8, 2)
    arr = jax.make_array_from_callback(
        global_data.shape, s, lambda idx: global_data[idx],
        dtype=global_data.dtype)

    # The returned array should only contain addressable shards in process
    # `pid`.
    expected_num_shards = n_local if jax.process_index() == pid else 0
    self.assertLen(arr.addressable_shards, expected_num_shards)
    np.testing.assert_array_equal(arr.shape, global_data.shape)
    for shard in arr.addressable_shards:
      np.testing.assert_array_equal(shard.data, global_data[shard.index])

  def test_make_array_from_callback_prngkey(self):
    n_local = jax.local_device_count()
    pid = 1
    mesh = jax.make_mesh(
        (n_local,), ("x",), devices=jax.local_devices(process_index=pid),
        axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))

    # Create a PRNG key array that is non-addressable in processes besides
    # `pid`.
    seeds = jnp.arange(8)
    global_data = jax.vmap(lambda x: jax.random.key(seed=x))(seeds)
    k = jax.random.key(0)
    arr = jax.make_array_from_callback(
        global_data.shape, s, lambda idx: global_data[idx],
        dtype=k.dtype)

    # The returned array should only contain addressable shards in process
    # `pid`.
    expected_num_shards = n_local if jax.process_index() == pid else 0
    self.assertLen(arr.addressable_shards, expected_num_shards)
    np.testing.assert_array_equal(arr.shape, global_data.shape)
    for shard in arr.addressable_shards:
      np.testing.assert_array_equal(shard.data.shape, (8 // n_local,))

  def test_sharding_process_indices_device_subset(self):
    n_devices = jax.device_count()
    mesh = jax.make_mesh(
        (n_devices // 2,), ("x",), devices=jax.devices()[:n_devices // 2],
        axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))

    expected_pids = {d.process_index for d in s.device_set}
    self.assertEqual(s._internal_device_list.process_indices, expected_pids)
    self.assertLen(s._internal_device_list.process_indices,
                   jax.process_count() // 2)

  def test_jit_no_local_devices_named_sharding(self):
    x = np.arange(64).reshape(8, 8)
    n_local = jax.local_device_count()
    pid = 1

    # Create a sharding that is non-addressable in processes besides `pid`.
    mesh = jax.make_mesh(
        (n_local,), ("x",), devices=jax.local_devices(process_index=pid),
        axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))
    y = jax.device_put(x, s)
    expected_num_shards = n_local if jax.process_index() == pid else 0
    self.assertLen(y.addressable_shards, expected_num_shards)

    @jax.jit
    def f(x):
      return x + 1

    # The returned array should only contain addressable shards in process
    # `pid`. No work is done in other processes.
    z = f(y)
    z.block_until_ready()
    self.assertLen(z.addressable_shards, expected_num_shards)
    if jax.process_index() == pid:
      for shard in z.addressable_shards:
        np.testing.assert_array_equal(shard.data, x[shard.index] + 1)

  def test_jit_no_local_devices_named_sharding_collective(self):
    x = np.arange(64).reshape(8, 8)
    n_local = jax.local_device_count()
    pid = 1

    # Create a sharding that is non-addressable in processes besides `pid`.
    mesh = jax.make_mesh(
        (n_local,), ("x",), devices=jax.local_devices(process_index=pid),
        axis_types=(jax.sharding.AxisType.Explicit,))
    s = jax.sharding.NamedSharding(mesh, P("x",))
    y = jax.device_put(x, s)
    expected_num_shards = n_local if jax.process_index() == pid else 0
    self.assertLen(y.addressable_shards, expected_num_shards)

    @jax.jit
    def f(x):
      return jnp.sum(x)

    # The returned array should only contain addressable shards in process
    # `pid`. No work is done in other processes.
    z = f(y)
    z.block_until_ready()
    self.assertLen(z.addressable_shards, expected_num_shards)
    if jax.process_index() == pid:
      expected = x.sum()
      for shard in z.addressable_shards:
        np.testing.assert_array_equal(shard.data, expected)

  def test_jit_no_local_devices_single_device_sharding(self):
    x = np.arange(64).reshape(8, 8)
    pid = 1

    # Create a single device sharding for a device local to process `pid`.
    s = jax.sharding.SingleDeviceSharding(
        jax.local_devices(process_index=pid)[0])
    y = jax.device_put(x, s)
    expected_num_shards = 1 if jax.process_index() == pid else 0
    self.assertLen(y.addressable_shards, expected_num_shards)

    @jax.jit
    def f(x):
      return x + 1

    # The returned array should only contain an addressable shard in process
    # `pid`. No work is done in other processes.
    z = f(y)
    z.block_until_ready()
    self.assertLen(z.addressable_shards, expected_num_shards)
    if jax.process_index() == pid:
      np.testing.assert_array_equal(z.addressable_shards[0].data, x + 1)

  def test_jit_fastpath_matmul(self):
    mesh = jax.sharding.Mesh(
        jax.devices()[: len(jax.devices()) // 2], axis_names=("devices"))
    sharding = jax.sharding.NamedSharding(mesh, P())

    x = jax.device_put(
        jnp.arange(8 * 16, dtype=jnp.float32).reshape((8, 16)), sharding)
    w = jax.device_put(
        jnp.arange(16 * 4, dtype=jnp.float32).reshape((16, 4)), sharding)

    jax.experimental.multihost_utils.sync_global_devices("start")
    matmul = jax.jit(lambda x, w: x @ w, out_shardings=sharding)

    _ = matmul(x, w)
    y = matmul(x, w)  # doesn't crash on second call
    expected = x @ w
    for shard in y.addressable_shards:
      np.testing.assert_array_equal(shard.data, expected[shard.index])

  def test_numpy_asarray_no_local_devices(self):
    y, x = create_nonaddressable_array((8, 8), spec=P())

    # In processes with local shards, we can fetch the value of the array using
    # np.asarray, since the sharding is fully replicated. In processes with no
    # local shards, attempting to fetch the NumPy array is an error.
    if y.sharding.has_addressable_devices:
      np.testing.assert_array_equal(np.asarray(y), x)
    else:
      with self.assertRaisesRegex(
          RuntimeError,
          r"Fetching value for `jax.Array` that spans non-addressable \(non"
          r" process local\) devices is not possible."):
        np.asarray(y)

  def test_shard_map_no_local_devices(self):
    x, x_np = create_nonaddressable_array((8, 8))

    # shard_map works as expected when there are nonparticipating hosts.
    shard_map_f = jax.shard_map(
        lambda x: jax.lax.psum(x, "x"), mesh=x.sharding.mesh, in_specs=P("x"),
        out_specs=P())
    y = shard_map_f(x)
    expected_y = sum(np.split(x_np, len(x.sharding.device_set)))
    sharding_process_indices = x.sharding._internal_device_list.process_indices
    expected_num_shards = (jax.local_device_count()
                           if jax.process_index() in sharding_process_indices
                           else 0)
    self.assertLen(y.addressable_shards, expected_num_shards)
    for shard in y.addressable_shards:
      np.testing.assert_array_equal(shard.data, expected_y[shard.index])

  def test_array_delete(self):
    y, _ = create_nonaddressable_array((8, 8))
    y.delete()
    with self.assertRaisesRegex(RuntimeError, "Array has been deleted."):
      y._check_if_deleted()
    self.assertIsNone(y._npy_value)
    self.assertIsNone(y._arrays)

  def test_single_device_array_usage_after_delete(self):
    y, _ = create_nonaddressable_array((8, 8))
    y.delete()

    with self.assertRaisesRegex(RuntimeError, "Array has been deleted."):
      _ = y + 1

  def test_repr(self):
    y, _ = create_nonaddressable_array((8, 8))
    if y.is_fully_addressable:
      self.assertStartsWith(repr(y), "Array([[ 0.,  1.,  2.,  3.,")
    else:
      self.assertEqual(repr(y), "Array(shape=(8, 8), dtype=float32)")

  def test_str(self):
    y, _ = create_nonaddressable_array((8, 8))
    if y.is_fully_addressable:
      self.assertStartsWith(str(y), "[[ 0.  1.  2.  3.")
    else:
      self.assertEqual(str(y), "Array(shape=(8, 8), dtype=float32)")

  def test_format(self):
    y, _ = create_nonaddressable_array((8, 8))
    if y.is_fully_addressable:
      self.assertStartsWith(format(y), "[[ 0.  1.  2.  3.")
    else:
      self.assertEqual(format(y), "Array(shape=(8, 8), dtype=float32)")

  def test_array_astype(self):
    y, _ = create_nonaddressable_array((8, 8))
    y = y.astype(np.int32)
    self.assertEqual(y.dtype, np.int32)

  def test_sharded_add(self):
    y, y_np = create_nonaddressable_array((8, 8))
    z, z_np = create_nonaddressable_array((8, 8), spec=P())
    out = y + z
    expected = y_np + z_np
    self.assertLen(out.addressable_shards, len(y.sharding.addressable_devices))
    for shard in out.addressable_shards:
      np.testing.assert_array_equal(shard.data, expected[shard.index])

  def test_sharded_zeros_like(self):
    y, _ = create_nonaddressable_array((8, 8))
    out = jnp.zeros_like(y)
    expected = jnp.zeros(y.shape, dtype=y.dtype)
    self.assertLen(out.addressable_shards, len(y.sharding.addressable_devices))
    for i in out.addressable_shards:
      np.testing.assert_array_equal(i.data, expected[i.index])

  def test_array_not_hashable(self):
    y, _ = create_nonaddressable_array((8, 8))
    with self.assertRaisesRegex(TypeError, "unhashable type"):
      hash(y)

  def test_on_device_size_in_bytes(self):
    a, _ = create_nonaddressable_array((8, 8))
    if not a.sharding.has_addressable_devices:
      with self.assertRaisesRegex(
          RuntimeError,
          r"GetOnDeviceSizeInBytes\(\) is not yet supported for arrays with no "
          r"addressable devices"):
        a.on_device_size_in_bytes()
    else:
      shard_size = a.addressable_shards[0].data.on_device_size_in_bytes()
      self.assertEqual(shard_size * len(a.global_shards),
                       a.on_device_size_in_bytes())

  def test_array_is_ready(self):
    y, _ = create_nonaddressable_array((8, 8))
    y.is_ready()  # doesn't crash

  def test_array_copy_to_host_async(self):
    y, x = create_nonaddressable_array((8, 8))
    y.copy_to_host_async()  # doesn't crash
    for shard in y.addressable_shards:
      np.testing.assert_array_equal(shard.data, x[shard.index])

  def test_device_get_replicated(self):
    y, x = create_nonaddressable_array((8, 8), spec=P())
    if y.sharding.has_addressable_devices:
      np.testing.assert_array_equal(jax.device_get(y), x)
    else:
      with self.assertRaisesRegex(
          RuntimeError,
          r"Fetching value for `jax.Array` that spans non-addressable \(non"
          r" process local\) devices is not possible."):
        jax.device_get(y)

  # Skipped on GPU since there are two processes with one device each, so we
  # can't construct a sharding that is nonaddressable in one of the processes
  # and also not fully replicated (since the sharding must contain one device).
  @jtu.skip_on_devices("gpu")
  def test_device_get_sharded(self):
    y, _ = create_nonaddressable_array((8, 8))
    with self.assertRaisesRegex(
        RuntimeError,
        r"Fetching value for `jax.Array` that spans non-addressable \(non"
        r" process local\) devices is not possible."):
      jax.device_get(y)

  def test_array_fully_replicated_shard(self):
    y, x = create_nonaddressable_array((8, 8), spec=P())
    if y.sharding.has_addressable_devices:
      fs = y.addressable_data(0)
      self.assertEqual(fs.shape, x.shape)
      self.assertLen(fs.sharding.device_set, 1)
      self.assertEqual(fs.devices(), {jax.local_devices()[0]})
      np.testing.assert_array_equal(fs, x)
      np.testing.assert_array_equal(y.addressable_data(0), x)
    else:
      with self.assertRaisesRegex(
          RuntimeError, "FullyReplicatedShard: Array has no addressable shards."
      ):
        y.addressable_data(0)

  def test_array_iter_replicated(self):
    y, _ = create_nonaddressable_array((8, 8), spec=P())
    y_iter = iter(y)
    self.assertLen(list(y_iter), 8)

  # Skipped on GPU since the sharding contains one device and is therefore fully
  # replicated.
  @jtu.skip_on_devices("gpu")
  def test_array_iter_sharded(self):
    y, _ = create_nonaddressable_array((8, 8))
    with self.assertRaises(AssertionError):
      iter(y)


class CrossHostTransferTest(jt_multiprocess.MultiProcessTest):

  @jtu.run_on_devices("cpu")
  def test_cross_host_transfer_cpu_error(self):
    x = np.arange(64).reshape(8, 8)
    src_pid = 0
    dst_pid = 1
    src_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices(process_index=src_pid)[0])
    dst_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices(process_index=dst_pid)[0])
    y = jax.device_put(x, src_sharding)
    with self.assertRaisesRegex(
        ValueError, "does not support cross-host device transfers"):
      jax.device_put(y, dst_sharding)

  @parameterized.named_parameters(
      ("numpy", np.arange),
      ("uncommitted", jnp.arange),
  )
  @jtu.skip_on_devices("cpu")
  def test_cross_host_transfer_single_device_sharding(self, arange_fn):
    x = arange_fn(64).reshape(8, 8)
    src_pid = 0
    dst_pid = 1
    src_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices(process_index=src_pid)[0])
    dst_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices(process_index=dst_pid)[0])
    y = jax.device_put(x, src_sharding)
    z = jax.device_put(y, dst_sharding)
    if jax.process_index() == dst_pid:
      self.assertLen(z.addressable_shards, 1)
      np.testing.assert_array_equal(z.addressable_shards[0].data, x)
    else:
      self.assertEmpty(z.addressable_shards)

  @parameterized.named_parameters(
      ("numpy", np.arange),
      ("uncommitted", jnp.arange),
  )
  @jtu.skip_on_devices("cpu")
  def test_cross_host_transfer_named_sharding(self, arange_fn):
    x = arange_fn(64).reshape(8, 8)
    n_local = jax.local_device_count()
    src_pid = 0
    dst_pid = 1
    src_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_local,), ("x",),
                      devices=jax.local_devices(process_index=src_pid),
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    dst_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_local,), ("x",),
                      devices=jax.local_devices(process_index=dst_pid),
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    y = jax.device_put(x, src_sharding)
    z = jax.device_put(y, dst_sharding)
    if jax.process_index() == dst_pid:
      self.assertLen(z.addressable_shards, n_local)
      for shard in z.addressable_shards:
        np.testing.assert_array_equal(shard.data, x[shard.index])
    else:
      self.assertEmpty(z.addressable_shards)

  @jtu.skip_on_devices("cpu")
  def test_cross_host_transfer_named_sharding_replicated(self):
    x = np.arange(64).reshape(8, 8)
    n_dev = jax.device_count() // 2
    src_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_dev,), ("x",), devices=jax.devices()[:n_dev],
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P()
    )
    dst_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_dev,), ("x",), devices=jax.devices()[n_dev:],
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P()
    )
    y = jax.device_put(x, src_sharding)
    z = jax.device_put(y, dst_sharding)
    for shard in z.addressable_shards:
      np.testing.assert_array_equal(shard.data, x[shard.index])

  @parameterized.named_parameters(
      ("numpy", np.arange),
      ("uncommitted", jnp.arange),
  )
  @jtu.skip_on_devices("cpu")
  def test_cross_host_transfer_batched(self, arange_fn):
    if jaxlib_extension_version < 400 and arange_fn == np.arange:
      self.skipTest("This functionality is not yet supported in jaxlib.")
    num_arrays = 3
    xs = []
    for i in range(1, num_arrays + 1):
      xs.append(arange_fn(64 * i).reshape(8, 8 * i))
      # TODO(emilyaf): Smaller sizes fail on TPU because the dst buffer size
      # returned by TransferSizeUtil::ShapeSizeCompact is larger than the src
      # buffer size. Investigate this further.
      # xs.append(jnp.arange(16 * i).reshape(8, 2 * i))
    xs[0] = xs[0].astype(jnp.float32)

    n_local = jax.local_device_count()
    src_pid = 0
    dst_pid = 1
    src_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_local,), ("x",),
                      devices=jax.local_devices(process_index=src_pid),
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    dst_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((n_local,), ("x",),
                      devices=jax.local_devices(process_index=dst_pid),
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))

    ys = jax.device_put(xs, src_sharding)
    zs = jax.device_put(ys, dst_sharding)
    for (x, z) in zip(xs, zs):
      if jax.process_index() == dst_pid:
        self.assertLen(z.addressable_shards, n_local)
        for shard in z.addressable_shards:
          np.testing.assert_array_equal(shard.data, x[shard.index])
      else:
        self.assertEmpty(z.addressable_shards)

  @jtu.skip_on_devices("cpu")
  def test_device_to_cpu_transfer_jit(self):
    x = jnp.arange(64).reshape(8, 8)
    with self.assertWarnsRegex(
        DeprecationWarning,
        r"backend and device argument on jit is deprecated",
    ):
      cpu_transfer_f = jax.jit(lambda x: x + 1, backend="cpu")
    cpu_transfer_f(x)  # Should not raise a cross-host transfer error.

  @jtu.skip_on_devices("cpu")
  def test_device_put_to_cpu(self):
    x = jnp.arange(64).reshape(8, 8)
    devices = jax.devices()
    cpu_devices = jax.devices(backend="cpu")
    num_devices = min(len(devices), len(cpu_devices))

    # Create CPU and GPU/TPU shardings that are not fully addressable.
    cpu_sharding = jax.sharding.NamedSharding(
        jax.make_mesh(
            (num_devices,), ("x",), devices=cpu_devices[:num_devices],
            axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    sharding = jax.sharding.NamedSharding(
        jax.make_mesh(
            (num_devices,), ("x",), devices=devices[:num_devices],
            axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    y = jax.device_put(x, sharding)

    # device_put of a GPU/TPU array to the CPU sharding should raise a helpful
    # error.
    with self.assertRaisesRegex(
        ValueError, ("For a cross-host reshard in multi-controller JAX|"
                     "device_put's second argument must be a Device")):
      jax.device_put(y, cpu_sharding)

  @jtu.skip_on_devices("cpu")
  def test_device_put_with_mixed_local_and_remote_transfers(self):
    if jaxlib_extension_version < 398:
      self.skipTest("This functionality is not yet supported in jaxlib.")
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 local devices for this test.")

    x = jnp.arange(64).reshape(8, 8)
    src_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((jax.local_device_count(),), ("x",),
                      devices=jax.local_devices(process_index=0),
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))

    # Half of the shards require cross-host transfers and half require local
    # transfers.
    dst_devices = (
        jax.local_devices(process_index=1)[:jax.local_device_count() // 2]
        + jax.local_devices(process_index=0)[:jax.local_device_count() // 2])
    dst_sharding = jax.sharding.NamedSharding(
        jax.make_mesh((jax.local_device_count(),), ("x",), devices=dst_devices,
                      axis_types=(jax.sharding.AxisType.Explicit,)),
        P("x"))
    y = jax.device_put(x, src_sharding)
    z = jax.device_put(y, dst_sharding)
    if jax.process_index() in (0, 1):
      self.assertLen(z.addressable_shards, jax.local_device_count() // 2)
    for shard in z.addressable_shards:
      np.testing.assert_array_equal(shard.data, x[shard.index])

  @parameterized.named_parameters(
      ("numpy", np.arange),
      ("uncommitted", jnp.arange),
  )
  @jtu.skip_on_devices("cpu")
  def test_device_put_to_device(self, arange_fn):
    if jaxlib_extension_version < 400:
      self.skipTest("This functionality is not yet supported in jaxlib.")
    x = arange_fn(64).reshape(8, 8)
    src_pid = 0
    dst_pid = 1
    src_device = jax.local_devices(process_index=src_pid)[0]
    dst_device = jax.local_devices(process_index=dst_pid)[0]
    y = jax.device_put(x, src_device)
    z = jax.device_put(y, dst_device)
    if jax.process_index() == dst_pid:
      self.assertLen(z.addressable_shards, 1)
      np.testing.assert_array_equal(z.addressable_shards[0].data, x)
    else:
      self.assertEmpty(z.addressable_shards)


if __name__ == "__main__":
  jt_multiprocess.main()
