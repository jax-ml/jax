# Copyright 2026 The JAX Authors.
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

"""Tests for xla_debug_log."""

import functools

from absl.testing import absltest
import jax
from jax import sharding
from jax._src import test_util as jtu
from jax._src import xla_debug_log
import numpy as np

NamedSharding = sharding.NamedSharding
Mesh = sharding.Mesh
PartitionSpec = sharding.PartitionSpec
partial = functools.partial
xla_debug_log = xla_debug_log.log
jax.config.parse_flags_with_absl()


class XlaDebugLogTest(jtu.JaxTestCase):

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()

  def test_callback_invoked(self):
    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    @jax.jit
    def demo(x, y):
      sum_val = x + y
      xla_debug_log(log_mode="GUARANTEED")(my_callback, x)
      xla_debug_log(log_mode="GUARANTEED")(my_callback, y)
      xla_debug_log(log_mode="GUARANTEED")(my_callback, sum_val)
      return sum_val

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)

    demo(x, y).block_until_ready()

    if not recorded_literals:
      self.fail("Callback was not invoked.")

    self.assertEqual(len(recorded_literals), 3)

    found_x = any(np.allclose(lit, x) for _, _, lit in recorded_literals)
    found_y = any(np.allclose(lit, y) for _, _, lit in recorded_literals)
    found_sum = any(np.allclose(lit, x + y) for _, _, lit in recorded_literals)

    self.assertTrue(found_x)
    self.assertTrue(found_y)
    self.assertTrue(found_sum)

  def test_callback_with_sharding(self):
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("This test requires at least 2 devices for sharding.")

    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    mesh = Mesh(devices, axis_names=("data_axis",))
    sharding_spec = PartitionSpec("data_axis")
    data_sharding = NamedSharding(mesh, sharding_spec)

    @jax.jit
    def demo(x):
      xla_debug_log()(my_callback, x)
      return x * 2

    tensor_shape = (len(devices) * 4,)
    x_data = np.arange(np.prod(tensor_shape), dtype=np.int32).reshape(
        tensor_shape
    )
    x_sharded = jax.device_put(x_data, data_sharding)

    demo(x_sharded).block_until_ready()

    if not recorded_literals:
      self.fail("Callback was not invoked.")

    found = False
    for replica_id, partition_id, literal in recorded_literals:
      self.assertEqual(replica_id, 0)
      self.assertEqual(partition_id, 0)
      if literal.shape == x_data.shape and np.allclose(literal, x_data):
        found = True

    self.assertTrue(found, "Did not find expected literal in callbacks.")

  def test_callback_with_partitioned(self):
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("This test requires at least 2 devices for sharding.")

    recorded_70 = []
    recorded_71 = []

    mesh = Mesh(devices, axis_names=("data_axis",))
    sharding_spec = PartitionSpec("data_axis")
    data_sharding = NamedSharding(mesh, sharding_spec)

    @jax.jit
    def demo(x):
      xla_debug_log(
          partitioned=True,
          log_mode="GUARANTEED",
      )(lambda r, p, l: recorded_70.append((p, l)), x)
      xla_debug_log(
          partitioned=True,
      )(lambda r, p, l: recorded_71.append((p, l)), x)
      return x * 2

    tensor_shape = (len(devices) * 4,)
    x_data = np.arange(np.prod(tensor_shape), dtype=np.int32).reshape(
        tensor_shape
    )
    x_sharded = jax.device_put(x_data, data_sharding)

    demo(x_sharded).block_until_ready()

    def verify_literals(recorded):
      self.assertEqual(len(recorded), len(devices))
      for part_id, lit in recorded:
        self.assertEqual(lit.shape, (4,))
        start = part_id * 4
        end = start + 4
        np.testing.assert_allclose(lit, x_data[start:end])

    verify_literals(recorded_70)
    verify_literals(recorded_71)

  def test_callback_with_shard_map(self):
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("This test requires at least 2 devices for shard_map.")

    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    mesh = Mesh(np.array(devices[:2]), axis_names=("x",))

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=PartitionSpec("x", None),
        out_specs=PartitionSpec("x", None),
    )
    def demo(x):
      res = x @ x
      xla_debug_log(log_mode="GUARANTEED")(my_callback, res)
      return res

    x_data = np.arange(8, dtype=np.float32).reshape(4, 2)
    expected_shard0 = x_data[0:2, :]
    expected_shard1 = x_data[2:4, :]
    expected_res0 = expected_shard0 @ expected_shard0
    expected_res1 = expected_shard1 @ expected_shard1

    demo(x_data).block_until_ready()

    if not recorded_literals:
      self.fail("Callback was not invoked.")

    self.assertEqual(len(recorded_literals), 2)

    found_res0 = any(
        np.allclose(lit, expected_res0) for _, _, lit in recorded_literals
    )
    found_res1 = any(
        np.allclose(lit, expected_res1) for _, _, lit in recorded_literals
    )

    self.assertTrue(found_res0, "Did not find expected result for shard 0.")
    self.assertTrue(found_res1, "Did not find expected result for shard 1.")

  def test_callback_with_2d_shard_map(self):
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest("This test requires at least 4 devices for 2D shard_map.")

    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), axis_names=("x", "y"))

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=PartitionSpec("x", "y"),
        out_specs=PartitionSpec("x", "y"),
    )
    def demo(x):
      res = x * 2
      xla_debug_log(log_mode="GUARANTEED")(my_callback, res)
      return res

    x_data = np.arange(16, dtype=np.int32).reshape(4, 4)
    demo(x_data).block_until_ready()

    if not recorded_literals:
      self.fail("Callback not invoked.")

    self.assertEqual(len(recorded_literals), 4)
    for _, _, lit in recorded_literals:
      self.assertEqual(lit.shape, (2, 2))

  def test_callback_with_sharding_in_shard_map(self):
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(
          "This test requires at least 4 devices for sharding in shard_map."
      )

    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), axis_names=("x", "y"))

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=PartitionSpec("x"),
        out_specs=PartitionSpec("x"),
        axis_names={"x"},
        check_vma=False,
    )
    def demo(x):
      x = jax.lax.with_sharding_constraint(x, PartitionSpec(None, "y"))
      res = x * 2
      xla_debug_log(log_mode="GUARANTEED")(my_callback, res)
      return res

    x_data = np.arange(16, dtype=np.int32).reshape(4, 4).astype(np.float32)
    x_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("x", None))
    x_data = jax.device_put(x_data, x_sharding)
    demo(x_data).block_until_ready()

    if not recorded_literals:
      self.fail("Callback not invoked.")

    self.assertEqual(len(recorded_literals), 2)
    for _, _, lit in recorded_literals:
      self.assertEqual(lit.shape, (2, 4))

  def test_callback_with_replication(self):
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(
          "This test requires at least 4 devices for replication test."
      )

    recorded_x = []
    recorded_y = []

    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), axis_names=("x", "y"))

    @jax.jit
    def demo(x, y):
      xla_debug_log(
          log_mode="GUARANTEED",
      )(lambda r, p, l: recorded_x.append(l), x)
      xla_debug_log(
          log_mode="GUARANTEED",
      )(lambda r, p, l: recorded_y.append(l), y)
      xla_debug_log()(lambda r, p, l: recorded_x.append(l), x)
      xla_debug_log()(lambda r, p, l: recorded_y.append(l), y)
      return x, y

    sharding_x = NamedSharding(mesh, PartitionSpec(None, None))
    sharding_y = NamedSharding(mesh, PartitionSpec(None, "y"))

    x_data = np.arange(256, dtype=np.float32).reshape(16, 16)
    y_data = np.arange(256, dtype=np.float32).reshape(16, 16) + 10.0

    x_placed = jax.device_put(x_data, sharding_x)
    y_placed = jax.device_put(y_data, sharding_y)

    res = demo(x_placed, y_placed)
    res[0].block_until_ready()
    res[1].block_until_ready()

    self.assertTrue(recorded_x)
    self.assertTrue(recorded_y)

    for lit in recorded_x:
      self.assertEqual(lit.shape, (16, 16))
      np.testing.assert_allclose(lit, x_data)
    for lit in recorded_y:
      self.assertEqual(lit.shape, (16, 16))
      np.testing.assert_allclose(lit, y_data)

  def test_callback_with_f64(self):
    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    with jax.enable_x64():

      @jax.jit
      def demo(x, y):
        sum_val = x + y
        xla_debug_log(log_mode="GUARANTEED")(my_callback, sum_val)
        return sum_val

      x = np.array([1.1, 2.2, 3.3], dtype=np.float64)
      y = np.array([4.4, 5.5, 6.6], dtype=np.float64)

      demo(x, y).block_until_ready()

    if not recorded_literals:
      self.fail("Callback not invoked.")

    found = False
    for _, _, literal in recorded_literals:
      if literal.dtype == np.float64 and np.allclose(literal, x + y):
        found = True

    self.assertTrue(found)

  def test_callback_with_s64(self):
    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    with jax.enable_x64():

      @jax.jit
      def demo(x, y):
        sum_val = x + y
        xla_debug_log(log_mode="GUARANTEED")(my_callback, sum_val)
        return sum_val

      x = np.array([10, 20, 30], dtype=np.int64)
      y = np.array([40, 50, 60], dtype=np.int64)

      demo(x, y).block_until_ready()

    if not recorded_literals:
      self.fail("Callback not invoked.")

    found = False
    for _, _, literal in recorded_literals:
      if literal.dtype == np.int64 and np.allclose(literal, x + y):
        found = True

    self.assertTrue(found)

  def test_callback_with_u64(self):
    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    with jax.enable_x64():

      @jax.jit
      def demo(x, y):
        sum_val = x + y
        xla_debug_log(log_mode="GUARANTEED")(my_callback, sum_val)
        return sum_val

      # Use large values to test high bits.
      x = np.array([1 << 40, 2 << 40, 3 << 40], dtype=np.uint64)
      y = np.array([4 << 40, 5 << 40, 6 << 40], dtype=np.uint64)

      demo(x, y).block_until_ready()

    if not recorded_literals:
      self.fail("Callback not invoked.")

    found = False
    for _, _, literal in recorded_literals:
      if literal.dtype == np.uint64 and np.allclose(literal, x + y):
        found = True

    self.assertTrue(found)

  def test_callback_with_multiple_modes(self):
    recorded_literals = []

    def my_callback(replica_id, partition_id, literal):
      recorded_literals.append((replica_id, partition_id, literal))

    @jax.jit
    def demo(a, b):
      r = a + b
      xla_debug_log(log_mode="GUARANTEED")(my_callback, r)
      xla_debug_log()(my_callback, r)
      return r

    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32)

    demo(a, b).block_until_ready()

    if not recorded_literals:
      self.fail("Callback not invoked.")

    self.assertEqual(len(recorded_literals), 2)
    for _, _, literal in recorded_literals:
      np.testing.assert_allclose(literal, a + b)

  def test_multi_value_callback_with_missing_operand(self):
    recorded_results = []

    def my_callback(replica_id, partition_id, *args):
      recorded_results.append(args)

    @jax.jit
    def demo(x):
      sin_x = jax.lax.sin(x)

      def cond_fun(state):
        loop_iter, _ = state
        return loop_iter < 2

      def body_fun(state):
        loop_iter, val = state
        new_loop_iter = loop_iter + 1
        xla_debug_log()(my_callback, new_loop_iter, sin_x)
        return new_loop_iter, val + 1.0

      return jax.lax.while_loop(cond_fun, body_fun, (0, x)), sin_x

    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    res = demo(x)
    res[0][0].block_until_ready()

    if not recorded_results:
      self.fail("Callback was not invoked.")

    self.assertLen(recorded_results, 2)

    self.assertLen(recorded_results[0], 2)
    self.assertEqual(recorded_results[0][0], 1)
    self.assertIsNone(recorded_results[0][1])

    self.assertLen(recorded_results[1], 2)
    self.assertEqual(recorded_results[1][0], 2)
    self.assertIsNone(recorded_results[1][1])

  def test_multi_value_callback_guaranteed(self):
    recorded_results = []

    def my_callback(replica_id, partition_id, x, y, sum_val):
      recorded_results.append((x, y, sum_val))

    @jax.jit
    def demo(x, y):
      sum_val = x + y
      xla_debug_log(log_mode="GUARANTEED")(my_callback, x, y, sum_val)
      return sum_val

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)

    demo(x, y).block_until_ready()

    if not recorded_results:
      self.fail("Callback was not invoked.")

    self.assertEqual(len(recorded_results), 1)
    val_x, val_y, val_sum = recorded_results[0]
    np.testing.assert_allclose(val_x, x)
    np.testing.assert_allclose(val_y, y)
    np.testing.assert_allclose(val_sum, x + y)

  def test_multi_value_callback_non_guaranteed(self):
    recorded_results = []

    def my_callback(replica_id, partition_id, x, y, sum_val):
      recorded_results.append((x, y, sum_val))

    @jax.jit
    def demo(x, y):
      sum_val = x + y
      xla_debug_log()(my_callback, x, y, sum_val)
      return sum_val

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)

    demo(x, y).block_until_ready()

    if not recorded_results:
      self.fail("Callback was not invoked.")

    self.assertEqual(len(recorded_results), 1)
    val_x, val_y, val_sum = recorded_results[0]
    np.testing.assert_allclose(val_x, x)
    np.testing.assert_allclose(val_y, y)
    np.testing.assert_allclose(val_sum, x + y)

  def test_callback_with_free_form_arguments(self):
    recorded_args = []
    recorded_kwargs = []

    def my_callback(replica_id, partition_id, *args, **kwargs):
      recorded_args.append(args)
      recorded_kwargs.append(kwargs)

    @jax.jit
    def demo(a, b, c, d, e, f, g, h, i, j, k):
      xla_debug_log(log_mode="GUARANTEED")(
          my_callback,
          a,
          [b, c],
          (d, (e, f)),
          {"g": g},
          h=h,
          ijk={"i": i, "jk": [j, k]},
      )
      return a

    a = np.array([1], dtype=np.int32)
    b = np.array([2], dtype=np.int32)
    c = np.array([3], dtype=np.int32)
    d = np.array([4], dtype=np.int32)
    e = np.array([5], dtype=np.int32)
    f = np.array([6], dtype=np.int32)
    g = np.array([7], dtype=np.int32)
    h = np.array([8], dtype=np.int32)
    i = np.array([9], dtype=np.int32)
    j = np.array([10], dtype=np.int32)
    k = np.array([11], dtype=np.int32)

    demo(a, b, c, d, e, f, g, h, i, j, k).block_until_ready()

    if not recorded_args:
      self.fail("Callback was not invoked.")

    self.assertEqual(len(recorded_args), 1)
    self.assertEqual(len(recorded_kwargs), 1)

    args = recorded_args[0]
    kwargs = recorded_kwargs[0]

    self.assertEqual(len(args), 4)
    np.testing.assert_allclose(args[0], a)

    self.assertIsInstance(args[1], list)
    self.assertEqual(len(args[1]), 2)
    np.testing.assert_allclose(args[1][0], b)
    np.testing.assert_allclose(args[1][1], c)

    self.assertIsInstance(args[2], tuple)
    self.assertEqual(len(args[2]), 2)
    np.testing.assert_allclose(args[2][0], d)
    self.assertIsInstance(args[2][1], tuple)
    self.assertEqual(len(args[2][1]), 2)
    np.testing.assert_allclose(args[2][1][0], e)
    np.testing.assert_allclose(args[2][1][1], f)

    self.assertIsInstance(args[3], dict)
    self.assertIn("g", args[3])
    np.testing.assert_allclose(args[3]["g"], g)

    self.assertIn("h", kwargs)
    np.testing.assert_allclose(kwargs["h"], h)

    self.assertIn("ijk", kwargs)
    self.assertIsInstance(kwargs["ijk"], dict)
    self.assertIn("i", kwargs["ijk"])
    np.testing.assert_allclose(kwargs["ijk"]["i"], i)
    self.assertIn("jk", kwargs["ijk"])
    self.assertIsInstance(kwargs["ijk"]["jk"], list)
    self.assertEqual(len(kwargs["ijk"]["jk"]), 2)
    np.testing.assert_allclose(kwargs["ijk"]["jk"][0], j)
    np.testing.assert_allclose(kwargs["ijk"]["jk"][1], k)


if __name__ == "__main__":
  absltest.main()
