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

import unittest
import numpy as np
from functools import partial
from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax import P
from jax._src import test_util as jtu
from jax._src import config
from jax._src.named_sharding import NamedSharding
from jax.experimental.custom_partitioning import (
    custom_partitioning, SdyShardingRule, BATCHING)

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


@jtu.pytest_mark_if_available('multiaccelerator')
class CustomPartitionerTest(jtu.JaxTestCase):

  def skip_if_custom_partitioning_not_supported(self):
    if jtu.is_cloud_tpu():
      raise unittest.SkipTest("Custom partitioning is not supported on libtpu.")

  @jtu.skip_on_devices('cpu')  # Collectives don't seem to work on CPU.
  def test_custom_partitioner(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(precision, mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
      result_sharding = result_shape[0].sharding
      self.assertEqual(arg_shardings[0], result_sharding)
      self.assertEqual(P('x', None), result_sharding.spec)
      self.assertEqual(P('y', None), arg_shardings[1].spec)

      def lower_fn(x, y):
        axis_name = arg_shardings[1].spec[0][0]
        i = jax.lax.axis_index(axis_name)
        # Use offset i * 0 instead of 0 to ensure that the two offsets have the
        # same dtype regardless the value of config.enable_x64.
        z = jax.lax.psum(
            jax.lax.dynamic_slice(x, (i * 0, i * 8), (8, 8)) @ y, (axis_name)
        )
        return z, z * z

      return mesh, lower_fn, (result_sharding, result_sharding), arg_shardings

    def infer_sharding_from_operands(precision, mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
      x_shard, y_shard = arg_shardings
      x_shape, y_shape = arg_shapes
      x_names = tuple(x_shard.spec) + tuple(
          None for _ in range(len(x_shape.shape) - len(x_shard.spec)))
      y_names = tuple(y_shard.spec) + tuple(
          None for _ in range(len(y_shape.shape) - len(y_shard.spec)))
      z_shard = NamedSharding(y_shard.mesh, P(*(x_names[:-1] + y_names[1:])))
      return z_shard, z_shard

    @partial(custom_partitioning, static_argnums=(2,))
    def f(x, y, precision=None):
      z = jnp.matmul(x, y, precision=precision)
      return z, z * z

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule=SdyShardingRule(operand_mappings=(('i', 'j'), ('j', 'k')), result_mappings=(('i', 'k'), ('i', 'k'))))

    with jax.set_mesh(jtu.create_mesh((4, 2), ('x', 'y'))):
      jit_f = jax.jit(f, in_shardings=(P('x'), P('y')), out_shardings=P('x'))
      x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
      y = np.asarray(np.random.randint(0, 20, (16, 32)), dtype=np.float32)
      x_sharded = jax.device_put(x, P('x'))
      y_sharded = jax.device_put(y, P('y'))
      result1 = jax.jit(f)(x_sharded, y_sharded)
      result2 = f(x, y)
      result0 = jit_f(x_sharded, y_sharded)
      self.assertArraysEqual(result0, result1)
      self.assertArraysEqual(result1, result2)

  def test_custom_partitioner_propagate_user_sharding(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      return (
          mesh,
          lower_fn,
          arg_shapes[0].sharding,
          (arg_shapes[0].sharding,),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return arg_shapes[0].sharding

    def propagate_user_sharding(mesh, user_shape):
      return user_shape.sharding

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        propagate_user_sharding=propagate_user_sharding,
        sharding_rule='i j -> i j',
    )

    def f2(a):
      return a + f(a)

    with jax.set_mesh(jtu.create_mesh((4, 2), ('x', 'y'))):
      jit_f = jax.jit(f2, in_shardings=(P(None, 'x')), out_shardings=P('x'))
      x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
      self.assertArraysEqual(x + x, jit_f(jax.device_put(x, P(None, 'x'))))

  def test_custom_partitioner_sharding_override(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      y_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(y_shard.mesh, P(None)),
          (NamedSharding(y_shard.mesh, P(None)),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      y_shard = arg_shapes[0].sharding
      return NamedSharding(y_shard.mesh, P('x'))

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule=SdyShardingRule(operand_mappings=((BATCHING, 'i'),), result_mappings=((BATCHING, 'i'),)))

    with jax.set_mesh(jtu.create_mesh((4, 2), ('x', 'y'))):
      jit_f = jax.jit(f, in_shardings=(P(None, 'x')), out_shardings=P('x'))
      x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
      self.assertArraysEqual(x, jit_f(jax.device_put(x, P(None, 'x'))))

  def test_custom_partitioner_invalid_sharding(self):
    self.skip_if_custom_partitioning_not_supported()
    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      y_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(y_shard.mesh, P(None)),
          (NamedSharding(y_shard.mesh, P(None, 'x')),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      y_shard = arg_shapes[0].sharding
      return NamedSharding(y_shard.mesh, P('x'))

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='i j -> i j',
    )

    with jax.set_mesh(jtu.create_mesh((4, 2), ('x', 'y'))):
      jit_f = jax.jit(f, in_shardings=(P(None, 'x')), out_shardings=P('x'))
      x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)

      with self.assertRaisesRegex(Exception, 'Mismatch in result shapes.'):
        jit_f(jax.device_put(x, P(None, 'x'))).block_until_ready()

  def test_custom_partitioner_jit_annotated_function(self):
    """Test correct lowering of function with a @jax.jit annotated callee.

    Annotating a callee with @jax.jit results in a module with a HLO CallOp.
    This test is makes sure that the custom partitioner lowering supports
    CallOps.
    """

    self.skip_if_custom_partitioning_not_supported()

    @custom_partitioning
    def f(x):
      return x

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        @jax.jit
        def g(y):
          return y

        return g(x)

      x_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(x_shard.mesh, P('x')),
          (NamedSharding(x_shard.mesh, P('x')),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      x_shard = arg_shapes[0].sharding
      return NamedSharding(x_shard.mesh, P('x'))

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='i -> i',
    )

    with jax.set_mesh(jtu.create_mesh((4,), ('x',))):
      jit_f = jax.jit(f)
      x = np.asarray(np.random.randint(0, 20, (32,)), dtype=np.float32)
      jit_f = jax.jit(jit_f, in_shardings=(P('x')), out_shardings=P('x'))
      self.assertArraysEqual(x, jit_f(jax.device_put(x, P('x'))))

  def test_custom_partitioner_with_scan(self):
    self.skip_if_custom_partitioning_not_supported()

    # This is a reproducer from https://github.com/jax-ml/jax/issues/20864.

    @custom_partitioning
    def f(x):
      return jnp.sum(x)

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(xs):
        def f(carry, x):
          return carry + jax.lax.psum(jnp.sum(x), axis_name='x'), None

        carry, _ = jax.lax.scan(f, 0, xs)
        return carry

      result_shardings = jax.tree.map(lambda x: x.sharding, result_shape)
      arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
      return mesh, lower_fn, result_shardings, arg_shardings

    f.def_partition(
        partition,
        infer_sharding_from_operands=lambda mesh, *_: NamedSharding(mesh, P()),
        propagate_user_sharding=lambda _, user_shape: user_shape.sharding,
        sharding_rule='i j -> ')  # Result is a scalar.

    with jax.set_mesh(jtu.create_mesh((4,), ('x',))):
      jit_f = jax.jit(f, in_shardings=P(None, 'x'))
      xs = jax.device_put(jnp.ones([32, 16]), P(None, 'x'))
      self.assertEqual(jit_f(xs), xs.sum())

  def test_custom_partitioning_no_mesh_context(self):
    self.skip_if_custom_partitioning_not_supported()

    @custom_partitioning
    def f(x):
      return x

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        @jax.jit
        def g(y):
          return y

        return g(x)

      x_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(x_shard.mesh, P('x')),
          (NamedSharding(x_shard.mesh, P('x')),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      x_shard = arg_shapes[0].sharding
      return NamedSharding(x_shard.mesh, P('x'))

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='i -> i',
    )

    mesh = jtu.create_mesh((4,), ('x',))
    x = np.asarray(np.random.randint(0, 20, (32,)), dtype=np.float32)
    s = NamedSharding(mesh, P('x'))

    jit_f = jax.jit(f, in_shardings=s, out_shardings=s)
    self.assertArraysEqual(x, jit_f(x))

  def test_custom_partitioner_pytree_inputs(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(xs):
        x, y, z = xs
        return x + y + z

      return (
          mesh,
          lower_fn,
          arg_shapes[0][0].sharding,
          jax.tree.map(lambda x: x.sharding, arg_shapes),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return arg_shapes[0][0].sharding

    def propagate_user_sharding(mesh, user_shape):
      return user_shape.sharding

    @custom_partitioning
    def f(xs):
      x, y, z = xs
      return x + y + z

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        propagate_user_sharding=propagate_user_sharding,
        sharding_rule='i j, i j, i j -> i j',
    )

    def f2(a):
      return a + f((a, a, a))

    with jax.set_mesh(jtu.create_mesh((4, 2), ('x', 'y'))):
      jit_f = jax.jit(f2, in_shardings=(P(None, 'x')), out_shardings=P('x'))
      x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
      self.assertArraysEqual(x * 4, jit_f(jax.device_put(x, P(None, 'x'))))

  @jtu.skip_on_devices('cpu')
  def test_custom_partition_with_sharding_rule_callback(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(static_arg0, static_arg1, mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
      result_sharding = result_shape.sharding
      rank = len(arg_shapes[0].shape)

      self.assertEqual(static_arg0, 1)
      self.assertEqual(static_arg1, 2)
      def lower_fn(x, y):
        axis_name = arg_shardings[1].spec[rank-2][0]
        i = jax.lax.axis_index(axis_name)
        z = jax.lax.psum(jax.lax.dynamic_slice_in_dim(
            jax.lax.dynamic_slice_in_dim(x, i * 0, 8, axis=rank-2),
            i * 8, 8, axis=rank-1) @ y, (axis_name))
        return z

      return mesh, lower_fn, (result_sharding), arg_shardings

    def produce_sharding_rule(static_arg0, static_arg1, mesh, arg_shapes, result_shape):
      self.assertEqual(static_arg0, 1)
      self.assertEqual(static_arg1, 2)
      rank = len(arg_shapes[0].shape)
      leading_axes = ""
      for i in range(rank - 2):
        leading_axes += f" b{i}"
      return f"{leading_axes} i j, {leading_axes} j k -> {leading_axes} i k" , dict(reduction_factors=("j",))

    @partial(custom_partitioning, static_argnums=(2,3))
    def f(x, y, static_arg0=1, static_arg1=2):
      return jnp.matmul(x, y)

    f.def_partition(
        infer_sharding_from_operands=None,
        partition=partition,
        sharding_rule=produce_sharding_rule)

    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    x = jax.device_put(np.arange(2 * 3 * 32 * 16).reshape(2, 3, 32, 16),
                       NamedSharding(mesh, P(None, None, 'x')))
    y = jax.device_put(np.arange(2 * 3 * 16 * 32).reshape(2, 3, 16, 32),
                       NamedSharding(mesh, P(None, None,'y')))
    result = jax.jit(f)(x, y)
    expected_result = f(x, y)
    self.assertArraysEqual(result, expected_result)
    self.assertEqual(result.sharding, NamedSharding(mesh, P(None, None, 'x')))

  def test_custom_partition_shardy_migration(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      return (
          mesh,
          lower_fn,
          arg_shapes[0].sharding,
          (arg_shapes[0].sharding,),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return arg_shapes[0].sharding

    def propagate_user_sharding(mesh, user_shape):
      return user_shape.sharding

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        propagate_user_sharding=propagate_user_sharding,
    )

    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    x = jax.device_put(np.arange(32 * 16).reshape(32, 16),
                       NamedSharding(mesh, P(None, 'x')))
    with self.assertRaisesRegex(
        NotImplementedError, 'provide sharding_rule to migrate to Shardy'):
      jax.jit(f)(x)

  def test_custom_partitioner_reshape(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
      result_sharding = result_shape.sharding

      def lower_fn(x, y):
        return x.reshape((4,)) + y
      return mesh, lower_fn, (result_sharding), arg_shardings

    @partial(custom_partitioning)
    def f(x, y):
      x = x.reshape((8,))
      return x + y

    f.def_partition(
      infer_sharding_from_operands=None,
      propagate_user_sharding=None,
      partition=partition,
      sharding_rule='(i k) j, (i k j) -> (i k j)', i=2, k=2, need_replication_factors=('k',))

    mesh = jtu.create_mesh((2, 4), ('x', 'y'))
    x = jax.device_put(np.arange(8).reshape(4, 2),
                       NamedSharding(mesh, P('x', None)))
    y = jax.device_put(np.arange(8),
                       NamedSharding(mesh, P('x')))
    jitted_result = jax.jit(f)(x, y)
    unjitted_result = f(x, y)
    self.assertArraysEqual(jitted_result, unjitted_result)
    self.assertEqual(jitted_result.sharding, NamedSharding(mesh, P('x')))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
