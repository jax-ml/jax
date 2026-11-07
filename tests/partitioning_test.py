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

import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import parallel
from jax._src.layout import Layout
from jax._src.pjit import with_layout_constraint
import jax.numpy as jnp
from jax.sharding import AxisType
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()

jtu.request_cpu_devices(8)


@jtu.pytest_mark_if_available('multiaccelerator')
class ShardMapPartitioningTest(jtu.JaxTestCase):

  @parameterized.product(
      op=((jnp.sum, np.sum), (jnp.max, np.max), (jnp.min, np.min)),
      case=(
          (None, P('x', 'y'), P(), True),
          (0, P('x', 'y'), P('y'), True),
          (0, P(('x', 'y'), None), P(None), True),
          (1, P(('x', 'y'), None), P(('x', 'y')), False),
      ),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce(self, op, case, mesh):
    jnp_op, np_op = op
    axis, in_spec, out_spec, has_ar = case
    np_inp = np.arange(32, dtype=np.float32).reshape(4, 8)
    arr = jax.device_put(np_inp, in_spec)

    def f(x):
      y = jnp_op(x, axis=axis)
      self.assertEqual(jax.typeof(y).sharding.spec, out_spec)
      return y

    f_jit = jax.jit(f)
    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f_jit.lower(arr).as_text())
      expected = f_jit(arr)
    lowered_text = f_jit.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered_text)
    self.assertEqual('all_reduce' in lowered_text, has_ar)

    out = f_jit(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axis))
    self.assertArraysEqual(out, expected)

  @parameterized.product(
      op=(
          (lax.reduce_sum, np.sum),
          (lax.reduce_max, np.max),
          (lax.reduce_min, np.min),
      ),
      case=(
          ((4, 8), P('x', None), (0,), P('x'), True, False),
          ((2, 2, 8), P('x', 'y', None), (0, 1), P(('y', 'x')), True, False),
          ((2, 8), P('x', 'y'), (0,), P(('y', 'x')), True, False),
          ((2, 2, 8), P('x', 'y', None), (0, 1), P('x'), True, True),
          ((4, 8), P('x', 'y'), (0,), P(None), False, False),
          ((4, 8), P('x', 'y'), (0,), P(('x', 'y')), False, False),
          ((2, 2, 4, 4), P('x', 'y', None, None), (0, 1), P('x', 'y'),
           True, False),
          (
              (4, 8),
              P(None, 'y', reduced={'x'}),
              (0,),
              P('y', reduced={'x'}),
              False,
              False,
          ),
      ),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce_scatter(self, op, case, mesh):
    lax_op, np_op = op
    shape, in_spec, axes, out_spec, is_rs, has_ar = case
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    arr = jax.device_put(np_inp, in_spec)
    out_s = None if out_spec.reduced else out_spec

    @jax.jit
    def f(x):
      y = lax_op(x, axes=axes, out_sharding=out_s)
      self.assertEqual(jax.typeof(y).sharding.spec, out_spec)
      return y

    uses_shmap = is_rs or bool(out_spec.reduced)
    if not uses_shmap:
      with self.assertRaises(NotImplementedError):
        f.lower(arr)
      return

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(arr).as_text())
      expected = f(arr)
    lowered_text = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered_text)
    self.assertEqual('reduce_scatter' in lowered_text, is_rs)
    self.assertEqual('all_reduce' in lowered_text, has_ar)

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axes))
    self.assertArraysEqual(out, expected)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_missing_rule_raises_not_implemented(self, mesh):
    arr = jax.device_put(np.ones((4, 8), dtype=np.float32), P('x', 'y'))

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda x: jnp.expand_dims(x, 0)).lower(arr)

    with self.assertRaises(NotImplementedError):
      f_bad = jax.jit(
          lambda x: lax.reduce_sum(x, axes=(0,), out_sharding=P(None)))
      f_bad.lower(arr)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('m', 'e'))
  def test_manual_collectives_under_partial_shard_map(self, mesh):
    np_x = np.arange(16, dtype=np.float32).reshape(2, 8)
    arr = jax.device_put(np_x, P('m', 'e'))

    @jax.jit
    @jax.shard_map(out_specs=P(None), axis_names={'m'})
    def f(x):
      y = lax.psum(x, 'm')
      z = lax.pmax(y, 'm')
      return lax.reduce_sum(z, axes=(0,))

    with config.shard_map_partitioning(False):
      expected = f(arr)
    out = f(arr)
    self.assertAllClose(out, np.sum(np_x, axis=0))
    self.assertArraysEqual(out, expected)

    @jax.jit
    @jax.shard_map(out_specs=P(), axis_names={'m'})
    def f_idx_psum():
      return lax.psum(lax.axis_index('m'), 'm')

    with config.shard_map_partitioning(False):
      expected_idx = f_idx_psum()
    out_idx = f_idx_psum()
    self.assertArraysEqual(out_idx, np.int32(1))
    self.assertArraysEqual(out_idx, expected_idx)

    @jax.jit
    @jax.shard_map(out_specs=P(None, None), axis_names={'m'})
    def f_ag_unsharded_dim(x):
      return lax.all_gather(x, 'm', axis=0, tiled=True, to='invarying')

    with config.shard_map_partitioning(False):
      expected_ag = f_ag_unsharded_dim(arr)
    out_ag = f_ag_unsharded_dim(arr)
    self.assertAllClose(out_ag, np_x)
    self.assertArraysEqual(out_ag, expected_ag)

    @jax.jit
    @jax.shard_map(out_specs=P(None, None), axis_names={'m'})
    def f_ag_const():
      c = lax.pcast(np.ones((1, 4), dtype=np.float32), 'm', to='varying')
      return lax.all_gather(c, 'm', axis=1, tiled=True, to='invarying')

    with config.shard_map_partitioning(False):
      expected_ag_const = f_ag_const()
    out_ag_const = f_ag_const()
    self.assertAllClose(out_ag_const, np.ones((1, 8), dtype=np.float32))
    self.assertArraysEqual(out_ag_const, expected_ag_const)

    @jax.jit
    @jax.shard_map(out_specs=P(None, None), axis_names={'m'})
    def f_ag_sharded_dim(x):
      return lax.all_gather(x, 'm', axis=1, tiled=True, to='invarying')

    with self.assertRaises(NotImplementedError):
      f_ag_sharded_dim.lower(arr)

  @parameterized.product(
      jnp_op=(jnp.sin, jnp.exp, jnp.negative, jnp.round, lax.erf),
      in_spec=(P('x', 'y'), P(('x', 'y'), None), P(None, None)),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_unop(self, jnp_op, in_spec, mesh):
    np_inp = np.arange(1, 33, dtype=np.float32).reshape(4, 8) / 8.0
    arr = jax.device_put(np_inp, in_spec)

    def f(x):
      self.assertEqual(jax.typeof(x).sharding.spec, in_spec)
      y = jnp_op(x)
      self.assertEqual(jax.typeof(y).sharding.spec, in_spec)
      return y

    with config.shard_map_partitioning(False):
      expected_np = jnp_op(np_inp)
      expected_eager = f(arr)

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, in_spec))
    self.assertArraysEqual(out, expected_np)
    self.assertArraysEqual(out, expected_eager)

    f_jit = jax.jit(f)
    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f_jit.lower(arr).as_text())
      expected_jit = f_jit(arr)
    self.assertIn('sdy.manual_computation', f_jit.lower(arr).as_text())

    out = f_jit(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, in_spec))
    self.assertArraysEqual(out, expected_np)
    self.assertArraysEqual(out, expected_jit)

    if jnp_op is jnp.negative:
      @jax.jit
      def vjp_fn(x, ct):
        return jax.vjp(f, x)[1](ct)

      with config.shard_map_partitioning(False):
        expected_grad, = vjp_fn(arr, arr)
      grad, = vjp_fn(arr, arr)
      self.assertEqual(grad.sharding, arr.sharding)
      self.assertArraysEqual(grad, -np_inp)
      self.assertArraysEqual(grad, expected_grad)

  @parameterized.parameters(
      ((4, 8), P('x', 'y'), (4, 8), P('x', 'y'), P('x', 'y'), False),
      ((4, 8), P('x', None), (4, 8), P(None, 'y'), P('x', 'y'), True),
      ((16, 4), P(None, None), (16, 4), P(('y', 'x'), None),
       P(('y', 'x'), None), True),
      ((4, 8), P('x', 'y'), (1, 8), P(None, None), P('x', 'y'), True),
      ((), P(), (4, 8), P('x', 'y'), P('x', 'y'), False),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_binop(
      self, x_shape, x_spec, y_shape, y_spec, out_spec, has_slice, mesh):
    np_x = np.arange(math.prod(x_shape), dtype=np.float32).reshape(x_shape)
    np_y = np.arange(math.prod(y_shape), dtype=np.float32).reshape(y_shape)
    arr_x, arr_y = jax.device_put(np_x, x_spec), jax.device_put(np_y, y_spec)

    f = jax.jit(jnp.subtract)
    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(arr_x, arr_y).as_text())
      expected = f(arr_x, arr_y)
    lowered = f.lower(arr_x, arr_y).as_text()
    self.assertIn('sdy.manual_computation', lowered)
    self.assertEqual('stablehlo.dynamic_slice' in lowered, has_slice)

    out = f(arr_x, arr_y)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_x - np_y)
    self.assertArraysEqual(out, expected)

  @parameterized.product(
      op=(
          (parallel.pmax_scatter, np.max),
          (parallel.pmin_scatter, np.min),
      ),
      tiled=(False, True),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_pmin_pmax_scatter_primitives(self, op, tiled, mesh):
    scatter_op, np_op = op
    shape = (2, 4) if tiled else (2, 2)
    np_inp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    arr = jax.device_put(np_inp, P('x', None))

    @jax.shard_map(out_specs=P('x'))
    def f(x):
      y = jnp.squeeze(x, axis=0)
      out = scatter_op(y, 'x', scatter_dimension=0, tiled=tiled)
      return out if tiled else out[None]

    expected = np_op(np_inp, axis=0)
    f_jit = jax.jit(f)
    f_vmap = jax.vmap(f)
    batched = jax.device_put(
        np.stack([np_inp, np_inp + 10]), P(None, 'x', None))
    with config.shard_map_partitioning(False):
      expected_eager = f(arr)
      expected_jit = f_jit(arr)
      expected_vmap = f_vmap(batched)
    out_eager = f(arr)
    out_jit = f_jit(arr)
    out_vmap = f_vmap(batched)
    self.assertArraysEqual(out_eager, expected)
    self.assertArraysEqual(out_eager, expected_eager)
    self.assertArraysEqual(out_jit, expected)
    self.assertArraysEqual(out_jit, expected_jit)
    self.assertArraysEqual(out_vmap, np.stack([expected, expected + 10]))
    self.assertArraysEqual(out_vmap, expected_vmap)

  @parameterized.parameters(
      (lax.reduce_prod, np.prod, np.float32, P('x', None), (1,), P('x'), False),
      (lax.reduce_and, np.all, np.bool_, P('x', 'y'), (0,), P('y'), True),
      (lax.reduce_or, np.any, np.bool_, P('x', 'y'), (0,), P('y'), True),
      (lax.reduce_xor, np.bitwise_xor.reduce, np.int32, P('x', None), (1,),
       P('x'), False),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce_prod_and_bitwise(
      self, lax_op, np_op, dtype, in_spec, axes, out_spec, has_ar, mesh):
    if dtype == np.bool_:
      np_inp = np.array([[0, 1, 0, 1], [0, 1, 1, 0],
                         [0, 1, 0, 1], [0, 1, 0, 0]], dtype=np.bool_)
    elif dtype == np.int32:
      np_inp = np.array([[1, 2, 3, 2], [2, 1, 2, 3],
                         [3, 2, 1, 2], [2, 3, 2, 1]], dtype=np.int32)
    else:
      np_inp = ((np.arange(16, dtype=np.float32) % 3) + 1).reshape(4, 4)
    arr = jax.device_put(np_inp, in_spec)

    @jax.jit
    def f(x):
      y = lax_op(x, axes=axes)
      self.assertEqual(jax.typeof(y).sharding.spec, out_spec)
      return y

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(arr).as_text())
      expected = f(arr)
    lowered = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered)
    self.assertEqual('all_reduce' in lowered, has_ar)
    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axes).astype(dtype))
    self.assertArraysEqual(out, expected)

    int_arr = jax.device_put(np.ones((4, 4), dtype=np.int32), P('x', 'y'))
    for op in (lax.reduce_prod, lax.reduce_and, lax.reduce_or, lax.reduce_xor):
      with self.assertRaises(NotImplementedError):
        jax.jit(lambda x, op=op: op(x, axes=(0,))).lower(int_arr)
    c_arr = jax.device_put(np.ones((4,), dtype=np.complex64), P('x'))
    with self.assertRaises(NotImplementedError):
      jax.jit(lambda x: lax.reduce_prod(x, axes=(0,))).lower(c_arr)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_per_shard_params(self, mesh):
    x = jax.device_put(np.arange(8, dtype=np.int32).reshape(4, 2), P('x', None))
    y = jax.jit(lambda a: lax.convert_element_type_p.bind(
        a, new_dtype=np.dtype(np.float32), weak_type=False,
        sharding=NamedSharding(mesh, P('x', None))))(x)
    self.assertArraysEqual(y, x.astype(np.float32))
    s = jax.jit(lambda a: lax.reduce_sum(
        a, axes=(1,), out_sharding=NamedSharding(mesh, P('x'))))(y)
    self.assertArraysEqual(s, np.sum(x.astype(np.float32), axis=1))

  @parameterized.parameters(
      (P(None, None), P('x', 'y'), 'dynamic_slice'),
      (P('x', 'y'), P(None, None), 'all_gather'),
      (P('x', None), P(None, 'x'), 'all_to_all'),
      (P('x', 'y'), P('y', 'x'), 'collective_permute'),
      (P(('x', 'y'), None), P('x', None), 'all_gather'),
      (P(None, None, reduced={'x'}), P('x', None), 'dynamic_slice'),
      (P('x', None), P(None, None, reduced={'x'}), 'all_gather'),
      (P(('x', 'y'), None), P('x', 'y'), 'all_to_all'),
      (P('x', 'y'), P(None, ('y', 'x')), 'all_to_all'),
      (P('x', 'y', None), P(None, 'x', 'y'), 'all_to_all'),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshard(self, in_spec, out_spec, expected_op, mesh):
    shape = (4,) * len(in_spec)
    np_x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    x = jax.device_put(np_x, in_spec)

    @jax.jit
    def f(a):
      return jax.reshard(a, out_spec)

    hlo = f.lower(x).as_text()
    self.assertIn('sdy.manual_computation', hlo)
    self.assertIn(expected_op, hlo)
    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_x)

    with config.shard_map_partitioning(False):
      expected = f(x)
      self.assertArraysEqual(expected, out)

  @parameterized.parameters(
      # Both axes move together in one all_to_all.
      (P(('x', 'y'), None), P(None, ('x', 'y')), (1, 0, 0)),
      # Both axes are gathered together in one all_gather.
      (P(('x', 'y'), None), P(None, None), (0, 1, 0)),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshard_collective_counts(self, in_spec, out_spec, counts, mesh):
    np_x = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = jax.device_put(np_x, in_spec)

    @jax.jit
    def f(a):
      return jax.reshard(a, out_spec)

    hlo = f.lower(x).as_text()
    num_a2a, num_ag, num_slice = counts
    self.assertEqual(hlo.count('stablehlo.all_to_all'), num_a2a)
    self.assertEqual(hlo.count('stablehlo.all_gather'), num_ag)
    self.assertEqual(hlo.count('stablehlo.dynamic_slice'), num_slice)
    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_x)

  @parameterized.parameters(
      # Reorder axes within a dim.
      ((2, 4), ('x', 'y'), P(('x', 'y'), None), P(('y', 'x'), None)),
      # Swap axes between dims.
      ((2, 2), ('x', 'y'), P('x', 'y'), P('y', 'x')),
      # Swap two axes while keeping a third in place.
      ((2, 2, 2), ('x', 'y', 'z'), P('x', 'y', 'z'), P('x', 'z', 'y')),
      # Rotate three axes.
      ((2, 2, 2), ('x', 'y', 'z'), P('x', 'y', 'z'), P('z', 'x', 'y')),
  )
  @config.shard_map_partitioning(True)
  def test_reshard_permute(self, mesh_shape, axis_names, in_spec, out_spec):
    mesh = jtu.create_mesh(mesh_shape, axis_names,
                           axis_types=(AxisType.Explicit,) * len(axis_names))
    shape = (8,) * len(in_spec)
    np_x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    @jax.jit
    def f(a):
      return jax.reshard(a, out_spec)

    with jax.set_mesh(mesh):
      x = jax.device_put(np_x, in_spec)
      hlo = f.lower(x).as_text()
      out = f(x)
      with config.shard_map_partitioning(False):
        expected = f(x)

    self.assertEqual(hlo.count('stablehlo.collective_permute'), 1)
    for op in ('all_gather', 'all_to_all', 'dynamic_slice'):
      self.assertNotIn(op, hlo)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_x)
    self.assertArraysEqual(out, expected)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 4), ('x', 'y'))
  def test_reshard_no_permute_when_shard_count_changes(self, mesh):
    # Dim 0 goes from 2 to 4 shards, so no device holds a whole new block:
    # f32[8@x, 8@y] -> f32[8, 8@y]    all_gather 'x'
    #               -> f32[8@y, 8]    all_to_all 'y' to dim 0
    #               -> f32[8@y, 8@x]  slice 'x'
    np_x = np.arange(64, dtype=np.float32).reshape(8, 8)
    x = jax.device_put(np_x, P('x', 'y'))
    out_spec = P('y', 'x')

    @jax.jit
    def f(a):
      return jax.reshard(a, out_spec)

    hlo = f.lower(x).as_text()
    self.assertNotIn('collective_permute', hlo)
    for op in ('all_gather', 'all_to_all', 'dynamic_slice'):
      self.assertIn(op, hlo)
    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_x)

    with config.shard_map_partitioning(False):
      expected = f(x)
    self.assertArraysEqual(out, expected)

  @parameterized.parameters(
      ((8, 12), P(('x', 'y'), None), (0, 3), (8, 11), (1, 2)),
      ((4, 10, 4), P('x', None, 'y'), (0, 1, 0), (4, 9, 4), (1, 3, 1)),
      ((8, 4), P(None, None), (2, 1), (5, 3), None),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_slice(self, shape, in_spec, start, limit, strides, mesh):
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    arr = jax.device_put(np_inp, in_spec)
    f = jax.jit(lambda x: lax.slice(x, start, limit, strides))
    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(arr).as_text())
      expected = f(arr)
    txt = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', txt)
    self.assertNotIn('all_gather', txt)
    out = f(arr)
    slices = tuple(map(slice, start, limit, strides or (None,) * len(shape)))
    self.assertEqual(out.sharding, NamedSharding(mesh, in_spec))
    self.assertArraysEqual(out, np_inp[slices])
    self.assertArraysEqual(out, expected)

  @parameterized.parameters(
      (((8, 3), (8, 5)), P(('x', 'y'), None), 1),
      (((4, 3, 4), (4, 5, 4)), P('x', None, 'y'), 1),
      (((4, 0), (4, 5)), P('x', None), 1),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_concatenate(self, shapes, spec, dim, mesh):
    np_inps = [np.arange(math.prod(s), dtype=np.float32).reshape(s)
               for s in shapes]
    arrs = [jax.device_put(x, spec) for x in np_inps]
    f = jax.jit(lambda *xs: lax.concatenate(xs, dimension=dim))
    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(*arrs).as_text())
      expected = f(*arrs)
    txt = f.lower(*arrs).as_text()
    self.assertIn('sdy.manual_computation', txt)
    self.assertNotIn('all_gather', txt)
    out = f(*arrs)
    self.assertEqual(out.sharding, NamedSharding(mesh, spec))
    self.assertArraysEqual(out, np.concatenate(np_inps, axis=dim))
    self.assertArraysEqual(out, expected)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('m', 'e'))
  def test_slice_concatenate_mixed_and_unsupported(self, mesh):
    np_x = np.arange(128, dtype=np.float32).reshape(4, 8, 4)
    arr = jax.device_put(np_x, NamedSharding(mesh, P('m', 'e', None)))

    @jax.jit
    @jax.shard_map(out_specs=P('m'), axis_names={'m'})
    def f(x):
      y = lax.slice(x, (0, 0, 1), (2, 8, 3))
      return lax.concatenate([y, y], dimension=2)

    with config.shard_map_partitioning(False):
      expected = f(arr)
    self.assertNotIn('all_gather', f.lower(arr).as_text())
    out = f(arr)
    self.assertEqual(out.sharding.spec, P('m', 'e', None))
    np_expected = np.concatenate([np_x[:, :, 1:3], np_x[:, :, 1:3]], axis=2)
    self.assertArraysEqual(out, np_expected)
    self.assertArraysEqual(out, expected)

    rep = jax.device_put(np.ones((2, 2), np.float32), P(None, None))
    lit = np.zeros((2, 2), np.float32)
    lit_fn = jax.jit(lambda x: lax.concatenate([x, lit], 1))
    with config.shard_map_partitioning(False):
      expected_lit = lit_fn(rep)
    lit_out = lit_fn(rep)
    self.assertArraysEqual(lit_out, np.concatenate([rep, lit], 1))
    self.assertArraysEqual(lit_out, expected_lit)

    arr_e = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2),
                           P('e', None))
    ones = np.ones((8, 2), np.float32)
    with self.assertRaises(NotImplementedError):
      jax.jit(lambda x: lax.slice(x, (2, 0), (6, 2))).lower(arr_e)
    with self.assertRaises(NotImplementedError):
      jax.jit(lambda x: lax.slice(x, (0, 0), (8, 2), (2, 1))).lower(arr_e)
    with self.assertRaises(NotImplementedError):
      jax.jit(lambda x: lax.concatenate([x, ones], 1)).lower(arr_e)

  @parameterized.parameters((None,), ('x',))
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_auto_axes(self, axes, mesh):
    np_x = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = jax.device_put(np_x, P('x', 'y'))

    def g(b):
      return b * 2

    @jax.jit
    def f(a):
      return jax.sharding.auto_axes(g, axes=axes, out_sharding=P('y', 'x'))(a)

    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'x')))
    self.assertArraysEqual(out, np_x * 2)

  @parameterized.parameters(
      ((AxisType.Auto, AxisType.Auto), None),
      ((AxisType.Explicit, AxisType.Auto), 'y'),
  )
  @config.shard_map_partitioning(True)
  def test_explicit_axes_in_auto_mesh(self, axis_types, axes):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'), axis_types=axis_types)
    np_x = np.arange(16, dtype=np.float32).reshape(4, 4)

    def g(b):
      return jax.reshard(b * 2, P('y', 'x'))

    @jax.jit
    def f(a):
      return jax.sharding.explicit_axes(
          g, axes=axes, in_sharding=P('x', 'y'))(a)

    with jax.set_mesh(mesh):
      x = jax.device_put(np_x, NamedSharding(mesh, P('x', 'y')))
      # The reshard inside the explicit region still uses shard_map.
      self.assertIn('sdy.manual_computation', f.lower(x).as_text())
      out = f(x)
    self.assertArraysEqual(out, np_x * 2)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2, 2), ('x', 'y', 'z'))
  def test_partial_manual_auto_axes(self, mesh):
    np_x = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    x = jax.device_put(np_x, P('x', 'y', 'z'))

    def auto_fn(c):
      return c * 2

    def manual_fn(b):
      # 'x' is Manual here; drop into Auto on the remaining axes ('y', 'z').
      b = jax.reshard(b, P(None, 'z', 'y'))
      return jax.sharding.auto_axes(
          auto_fn, axes=('y', 'z'), out_sharding=P(None, 'y', 'z'))(b)

    @jax.jit
    def f(a):
      return jax.shard_map(
          manual_fn, mesh=mesh, in_specs=P('x', None, None),
          out_specs=P('x', None, None), axis_names={'x'})(a)

    lowered_text = f.lower(x).as_text()
    # One for the outer shard_map on 'x', one for the explicit reshard on
    # ('y', 'z'), and none from `auto_axes`.
    self.assertEqual(lowered_text.count('sdy.manual_computation'), 2)
    self.assertIn('sdy.sharding_constraint', lowered_text)

    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y', 'z')))
    self.assertArraysEqual(out, np_x * 2)

  @parameterized.parameters(
      # f32[4@x, 4] -> f32[4@x, 4@y] -> f32[4, 4@y]
      (P('x', None), P(None, 'y'), 'all_gather'),
      # f32[4@x, 4, 4] -> f32[4@x, 4, 4@y] -> f32[4, 4@x, 4@y]
      (P('x', None, None), P(None, 'x', 'y'), 'all_to_all'),
      # f32[4@x, 4@y, 4] -> f32[4@x, 4@y, 4@z] -> f32[4@y, 4@x, 4@z]
      (P('x', 'y', None), P('y', 'x', 'z'), 'collective_permute'),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2, 2), ('x', 'y', 'z'))
  def test_reshard_slices_before_collective(self, in_spec, out_spec,
                                            collective, mesh):
    shape = (4,) * len(in_spec)
    np_x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    x = jax.device_put(np_x, in_spec)

    @jax.jit
    def f(a):
      return jax.reshard(a, out_spec)

    hlo = f.lower(x).as_text()
    self.assertEqual(hlo.count('stablehlo.dynamic_slice'), 1)
    self.assertLess(hlo.index('stablehlo.dynamic_slice'),
                    hlo.index(f'stablehlo.{collective}'))
    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_x)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2, 2), ('x', 'y', 'z'))
  def test_reshard_slice_collective_slice(self, mesh):
    # f32[4@(x,y), 4, 4] -> f32[4@(x,y), 4, 4@z]  slice 'z'
    #                    -> f32[4@x, 4, 4@z]      all_gather 'y'
    #                    -> f32[4, 4@x, 4@z]      all_to_all 'x' to dim 1
    #                    -> f32[4@y, 4@x, 4@z]    slice 'y'
    np_x = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    x = jax.device_put(np_x, P(('x', 'y'), None, None))
    out_spec = P('y', 'x', 'z')

    @jax.jit
    def f(a):
      return jax.reshard(a, out_spec)

    hlo = f.lower(x).as_text()
    self.assertEqual(hlo.count('stablehlo.dynamic_slice'), 2)
    first_slice = hlo.index('stablehlo.dynamic_slice')
    ag = hlo.index('stablehlo.all_gather')
    a2a = hlo.index('stablehlo.all_to_all')
    second_slice = hlo.rindex('stablehlo.dynamic_slice')
    self.assertTrue(first_slice < ag < a2a < second_slice)

    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_x)

  @parameterized.parameters(
      ((), P(), (1, 1), None),
      ((4, 16), P('x', 'y'), (4, 2, 8), None),
      ((2, 2, 4), P('x', 'y', None), (16,), None),
      ((8, 4), P(('x', 'y'), None), (2, 4, 4), P('x', 'y', None)),
      ((4, 16), P('x', 'y'), (4, 2, 8), P('x', 'y', None)),
      ((4, 16), P('x', None, reduced={'y'}), (4, 2, 8),
       P('x', None, None, reduced={'y'})),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshape_no_collectives(
      self, in_shape, in_spec, out_shape, out_sharding, mesh):
    np_inp = np.arange(math.prod(in_shape), dtype=np.float32).reshape(in_shape)
    arr = jax.device_put(np_inp, in_spec)

    @jax.jit
    def f(x):
      return lax.reshape(x, out_shape, out_sharding=out_sharding)

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(arr).as_text())
      expected = f(arr)
    txt = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', txt)
    self.assertNotIn('all_to_all', txt)
    self.assertNotIn('all_gather', txt)
    out = f(arr)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshape_collectives_needed_raises(self, mesh):
    for shape, in_spec, dims, out_shape, out_spec in (
        ((4, 4), P('x', 'y'), None, (16,), None),
        ((16,), P(('x', 'y')), None, (4, 4), P('x', 'y')),
        ((4, 8), P('x', None), None, (2, 2, 8), P(None, 'x', None)),
        ((4, 4), P('x', 'y'), None, (16,), P('x')),
        ((4, 6), P('x', 'y'), (1, 0), (6, 2, 2), P('y', 'x', None)),
    ):
      arr = jax.device_put(np.zeros(shape, dtype=np.float32), in_spec)

      @jax.jit
      def f(x):
        return lax.reshape(x, out_shape, dimensions=dims, out_sharding=out_spec)

      with self.assertRaises(NotImplementedError):
        f.lower(arr)

  @parameterized.parameters(
      (
          (8, 4),
          P('x', None),
          (4, 8),
          P(None, 'y'),
          (((1,), (0,)), ((), ())),
          None,
      ),
      (
          (8, 4),
          P(('x', 'y'), None),
          (4, 8),
          P(None, None),
          (((1,), (0,)), ((), ())),
          P(('x', 'y'), None),
      ),
      (
          (4, 8, 4),
          P('x', None, None),
          (4, 4, 8),
          P('x', None, None),
          (((2,), (1,)), ((0,), (0,))),
          None,
      ),
      (
          (4, 2, 4, 4),
          P('x', 'y', 'z', None),
          (2, 4, 4, 4),
          P('y', 'x', None, None),
          (((3,), (3,)), ((0, 1), (1, 0))),
          P('x', 'y', 'z', None),
      ),
      (
          (0, 4),
          P('x', None),
          (4, 8),
          P(None, 'y'),
          (((1,), (0,)), ((), ())),
          None,
      ),
      (
          (4, 4),
          P('x', None, reduced={'z'}),
          (4, 8),
          P(None, 'y'),
          (((1,), (0,)), ((), ())),
          None,
      ),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2, 2), ('x', 'y', 'z'))
  def test_dot_general_no_collectives(
      self, lhs_shape, lhs_spec, rhs_shape, rhs_spec, dnums, out_spec, mesh
  ):
    np_lhs = np.arange(math.prod(lhs_shape), dtype=np.float32).reshape(
        lhs_shape
    )
    np_rhs = np.arange(math.prod(rhs_shape), 0, -1, dtype=np.float32).reshape(
        rhs_shape
    )
    lhs = jax.device_put(np_lhs, lhs_spec)
    rhs = jax.device_put(np_rhs, rhs_spec)

    @jax.jit
    def f(a, b):
      return lax.dot_general(
          a, b, dnums, precision=lax.Precision.HIGHEST, out_sharding=out_spec
      )

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(lhs, rhs).as_text())
      expected = f(lhs, rhs)
    self.assertIn('sdy.manual_computation', f.lower(lhs, rhs).as_text())

    out = f(lhs, rhs)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_dot_general_collectives_needed_raises(self, mesh):
    d2 = (((1,), (0,)), ((), ()))
    d3 = (((2,), (1,)), ((0,), (0,)))
    for shape, dnums, lhs_spec, rhs_spec, out_spec in (
        ((8, 8), d2, P(None, 'x'), P(None, None), None),
        ((8, 8), d2, P(None, None), P('x', None), None),
        ((8, 8), d2, P('x', None), P('x', None), None),
        (
            (4, 4, 4),
            d3,
            P('x', None, None),
            P(None, None, None),
            P('x', None, None),
        ),
        ((8, 8), d2, P('x', None), P(None, 'x'), P('x', None)),
        ((8, 8), d2, P('x', None), P(None, None), P(('x', 'y'), None)),
    ):
      z = np.zeros(shape, dtype=np.float32)
      lhs = jax.device_put(z, lhs_spec)
      rhs = jax.device_put(z, rhs_spec)

      @jax.jit
      def f(a, b):
        return lax.dot_general(a, b, dnums, out_sharding=out_spec)

      with self.assertRaises(NotImplementedError):
        f.lower(lhs, rhs)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_layout_constraint(self, mesh):
    np_x = np.arange(64, dtype=np.float32).reshape(8, 8)
    x = jax.device_put(np_x, P('x', 'y'))
    layout = Layout(major_to_minor=(1, 0))

    @jax.jit
    def f(a):
      return with_layout_constraint(a * 2, layout)

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(x).as_text())
      expected = f(x)
    hlo = f.lower(x).as_text()
    self.assertIn('sdy.manual_computation', hlo)
    self.assertIn('LayoutConstraint', hlo)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertEqual(out.format.layout.major_to_minor, (1, 0))
    self.assertArraysEqual(out, expected)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
