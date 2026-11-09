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
from jax._src import ad_util
from jax._src import config
from jax._src import core
from jax._src import dtypes
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

  def setUp(self):
    super().setUp()
    self.enterContext(config.shard_map_partitioning(True))

  @parameterized.product(
      op=((jnp.sum, np.sum), (jnp.max, np.max), (jnp.min, np.min)),
      case=(
          (None, P('x', 'y'), P(), True),
          (0, P('x', 'y'), P('y'), True),
          (0, P(('x', 'y'), None), P(None), True),
          (1, P(('x', 'y'), None), P(('x', 'y')), False),
      ),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce(self, op, case, mesh):
    jnp_op, np_op = op
    axis, in_spec, out_spec, has_ar = case
    np_inp = np.arange(32, dtype=np.float32).reshape(4, 8)
    arr = jax.device_put(np_inp, in_spec)

    @jax.jit
    def f(x):
      y = jnp_op(x, axis=axis)
      self.assertEqual(jax.typeof(y).sharding.spec, out_spec)
      return y

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(arr).as_text())
      expected = f(arr)

    lowered_text = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered_text)
    self.assertEqual('all_reduce' in lowered_text, has_ar)

    out = f(arr)
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

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_missing_rule_raises_not_implemented(self, mesh):
    arr = jax.device_put(np.ones((4, 8), dtype=np.float32), P('x', 'y'))

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda x: lax.rev(x, (0,))).lower(arr)

    with self.assertRaises(NotImplementedError):
      f_bad = jax.jit(
          lambda x: lax.reduce_sum(x, axes=(0,), out_sharding=P(None)))
      f_bad.lower(arr)

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

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_per_shard_params(self, mesh):
    x = jax.device_put(np.arange(8, dtype=np.int32).reshape(4, 2), P('x', None))

    @jax.jit
    def f(a):
      return lax.convert_element_type_p.bind(
          a, new_dtype=np.dtype(np.float32), weak_type=False,
          sharding=NamedSharding(mesh, P('x', None)))

    y = f(x)
    self.assertArraysEqual(y, x.astype(np.float32))

    @jax.jit
    def g(a):
      return lax.reduce_sum(
          a, axes=(1,), out_sharding=NamedSharding(mesh, P('x')))

    s = g(y)
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
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_slice(self, shape, in_spec, start, limit, strides, mesh):
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    arr = jax.device_put(np_inp, in_spec)

    @jax.jit
    def f(x):
      return lax.slice(x, start, limit, strides)

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
      (((4, 2), (4, 3)), P('x', None, reduced={'y'}), 1),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_concatenate(self, shapes, spec, dim, mesh):
    np_inps = [np.arange(math.prod(s), dtype=np.float32).reshape(s)
               for s in shapes]
    arrs = [jax.device_put(x, spec) for x in np_inps]

    @jax.jit
    def f(*xs):
      return lax.concatenate(xs, dimension=dim)

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

    @jax.jit
    def lit_fn(x):
      return lax.concatenate([x, lit], 1)

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

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                          axis_types=(AxisType.Explicit, AxisType.Auto))
  def test_mixed_explicit_auto_mesh(self, mesh):
    x = jax.device_put(np.arange(16).reshape(4, 4), P('x', 'y'))

    @jax.jit
    def f(a):
      return jnp.sum(a * 2)

    self.assertNotIn('sdy.manual_computation', f.lower(x).as_text())
    out = f(x)

    with config.shard_map_partitioning(False):
      expected = f(x)

    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @parameterized.parameters(
      # f32[4@x, 4] -> f32[4@x, 4@y] -> f32[4, 4@y]
      (P('x', None), P(None, 'y'), 'all_gather'),
      # f32[4@x, 4, 4] -> f32[4@x, 4, 4@y] -> f32[4, 4@x, 4@y]
      (P('x', None, None), P(None, 'x', 'y'), 'all_to_all'),
      # f32[4@x, 4@y, 4] -> f32[4@x, 4@y, 4@z] -> f32[4@y, 4@x, 4@z]
      (P('x', 'y', None), P('y', 'x', 'z'), 'collective_permute'),
  )
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
      ((4, 4), P('x', 'y'), (2, 2, 2, 2), P('x', None, 'y', None)),
      ((2, 2, 2, 2), P('x', None, 'y', None), (4, 4), P('x', 'y')),
      ((4, 16), P('x', None, reduced={'y'}), (4, 2, 8),
       P('x', None, None, reduced={'y'})),
  )
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

  @parameterized.parameters(
      # Merge all_to_all (propagated and explicit out_sharding)
      ((4, 4), P('x', 'y'), (16,), None, P(('x', 'y')), {'all_to_all': 1}),
      ((4, 4), P('x', 'y'), (16,), P(('x', 'y')), P(('x', 'y')),
       {'all_to_all': 1}),
      ((8, 4), P('x', 'y'), (32,), None, P(('x', 'y')), {'all_to_all': 1}),
      ((2, 1, 4, 4, 3), P('z', None, 'x', 'y', None), (2, 16, 3), None,
       P('z', ('x', 'y'), None), {'all_to_all': 1}),
      ((8, 4), P('x', ('y', 'z')), (32,), None, P(('x', 'y', 'z')),
       {'all_to_all': 1}),
      ((2, 4, 4), P('x', 'y', 'z'), (32,), None, P(('x', 'y', 'z')),
       {'all_to_all': 1}),
      ((4, 2, 4), P('x', 'y', 'z'), (32,), None, P(('x', 'y', 'z')),
       {'all_to_all': 2}),
      ((8, 4, 4), P('x', 'y', 'z'), (128,), None, P(('x', 'y', 'z')),
       {'all_to_all': 2}),
      ((4, 4), P('x', 'y', reduced={'z'}), (16,), None,
       P(('x', 'y'), reduced={'z'}), {'all_to_all': 1}),
      # Merge with unsharded gap
      ((4, 2, 2), P('x', None, 'y'), (16,), P(('x', 'y')), P(('x', 'y')),
       {'all_to_all': 1}),
      # Split all_to_all and split + move
      ((16,), P(('x', 'y')), (4, 4), P('x', 'y'), P('x', 'y'),
       {'all_to_all': 1}),
      ((4, 4), P('x', 'y'), (4, 2, 2), P('x', None, 'y'), P('x', None, 'y'),
       {'all_to_all': 1}),
      ((4, 8), P('x', 'y'), (4, 2, 4), P('x', None, 'y'), P('x', None, 'y'),
       {'all_to_all': 1}),
      ((4, 8), P('x', None), (2, 2, 8), P(None, 'x', None), P(None, 'x', None),
       {'all_to_all': 1}),
      # Multi-axis split and merge with collectives
      ((4, 4), P('x', 'y'), (2, 2, 2, 2), P(None, 'x', None, 'y'),
       P(None, 'x', None, 'y'), {'all_to_all': 2}),
      ((2, 2, 2, 2), P(None, 'x', None, 'y'), (4, 4), P('x', 'y'),
       P('x', 'y'), {'all_to_all': 2}),
      ((4, 4, 1, 2, 2), P('x', 'y', None, None, 'z'), (16, 1, 4),
       P(('x', 'y'), None, 'z'), P(('x', 'y'), None, 'z'), {'all_to_all': 2}),
      ((4, 4, 2, 2), P('x', 'y', 'z', None), (16, 4), P('x', 'z'),
       P('x', 'z'), {'all_gather': 1}),
      ((16, 4), P('x', 'z'), (4, 4, 2, 2), P('x', 'y', None, 'z'),
       P('x', 'y', None, 'z'), {'all_to_all': 1, 'dynamic_slice': 1}),
      # Axis drop (all_gather) and axis add (dynamic_slice)
      ((4, 4), P('x', 'y'), (16,), P('x'), P('x'), {'all_gather': 1}),
      ((16,), P('x'), (4, 4), P('x', 'y'), P('x', 'y'), {'dynamic_slice': 1}),
      # Prefer all_to_all over all_gather when step counts tie
      ((2, 2, 4), P('x', None, None), (4, 4), P(('y', 'x'), None),
       P(('y', 'x'), None), {'all_to_all': 1, 'dynamic_slice': 1}),
      ((2, 2, 4), P('x', 'y', None), (4, 4), P('y', None), P('y', None),
       {'all_gather': 1, 'all_to_all': 1}),
      # Axis swap (collective_permute)
      ((2, 2, 4), P('x', None, 'y'), (4, 4), P('y', 'x'), P('y', 'x'),
       {'collective_permute': 1}),
      ((2, 2, 4), P('x', 'y', None), (4, 4), P(('y', 'x'), None),
       P(('y', 'x'), None), {'collective_permute': 1}),
      ((4, 4), P('x', 'y'), (16,), P(('y', 'x')), P(('y', 'x')),
       {'all_to_all': 1, 'collective_permute': 1}),
  )
  @jtu.with_explicit_mesh((2, 2, 2), ('x', 'y', 'z'))
  def test_reshape_collectives(
      self, in_shape, in_spec, out_shape, out_spec, expected_spec,
      expected_collectives, mesh):
    np_inp = np.arange(math.prod(in_shape), dtype=np.float32).reshape(in_shape)
    arr = jax.device_put(np_inp, in_spec)

    @jax.jit
    def f(x):
      y = lax.reshape(x, out_shape, out_sharding=out_spec)
      self.assertEqual(jax.typeof(y).sharding.spec, expected_spec)
      return y

    lowered_txt = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered_txt)

    for op in ('all_to_all', 'all_gather', 'dynamic_slice',
               'collective_permute'):
      self.assertEqual(
          lowered_txt.count(f'stablehlo.{op}'), expected_collectives.get(op, 0),
          msg=op)

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(arr).as_text())
      expected = f(arr)

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, expected_spec))
    self.assertArraysEqual(out, np_inp.reshape(out_shape))
    self.assertArraysEqual(out, expected)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshape_collectives_vjp(self, mesh):
    np_inp = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr = jax.device_put(np_inp, P('x', 'y'))

    def reshape_fn(x):
      return lax.reshape(x, (16,))

    @jax.jit
    def f(x):
      y, vjp_fn = jax.vjp(reshape_fn, x)
      dx, = vjp_fn(y * 2)
      return y, dx

    with config.shard_map_partitioning(False):
      expected_y, expected_dx = f(arr)

    txt = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', txt)
    self.assertEqual(txt.count('stablehlo.all_to_all'), 2)

    out_y, out_dx = f(arr)
    self.assertEqual(out_y.sharding, NamedSharding(mesh, P(('x', 'y'))))
    self.assertEqual(out_dx.sharding, NamedSharding(mesh, P('x', 'y')))
    self.assertArraysEqual(out_y, expected_y)
    self.assertArraysEqual(out_dx, expected_dx)

  @parameterized.parameters(
      ((4, 6), P('x', 'y'), (1, 0), (6, 2, 2), P('y', 'x', None)),
      ((4, 6), P('x', 'y'), None, (3, 8), P(None, 'x')),
      ((2, 2, 8), P('x', None, 'y'), None, (4, 2, 4), P('x', None, 'y')),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshape_unsupported_raises(
      self, shape, in_spec, dims, out_shape, out_spec, mesh):
    arr = jax.device_put(np.zeros(shape, dtype=np.float32), in_spec)

    @jax.jit
    def f(x):
      return lax.reshape(x, out_shape, dimensions=dims, out_sharding=out_spec)

    with self.assertRaises(NotImplementedError):
      f(arr)

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

  @parameterized.parameters(
      ((8, 8), (((1,), (0,)), ((), ())), P(None, 'x'), P(None, None), None),
      ((8, 8), (((1,), (0,)), ((), ())), P(None, None), P('x', None), None),
      ((8, 8), (((1,), (0,)), ((), ())), P('x', None), P('x', None), None),
      ((4, 4, 4), (((2,), (1,)), ((0,), (0,))), P('x', None, None),
       P(None, None, None), P('x', None, None)),
      ((8, 8), (((1,), (0,)), ((), ())), P('x', None), P(None, 'x'),
       P('x', None)),
      ((8, 8), (((1,), (0,)), ((), ())), P('x', None), P(None, None),
       P(('x', 'y'), None)),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_dot_general_collectives_needed_raises(
      self, shape, dnums, lhs_spec, rhs_spec, out_spec, mesh):
    z = np.zeros(shape, dtype=np.float32)
    lhs = jax.device_put(z, lhs_spec)
    rhs = jax.device_put(z, rhs_spec)

    @jax.jit
    def f(a, b):
      return lax.dot_general(a, b, dnums, out_sharding=out_spec)

    with self.assertRaises(NotImplementedError):
      f.lower(lhs, rhs)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_layout_constraint(self, mesh):
    np_x = np.arange(64, dtype=np.float32).reshape(8, 8)
    x = jax.device_put(np_x, P('x', 'y'))

    @jax.jit
    def f(a):
      return with_layout_constraint(a * 2, Layout(major_to_minor=(1, 0)))

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

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_broadcast_in_dim(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))
    row = jax.device_put(
        np.arange(4, dtype=np.float32).reshape(1, 4), P(None, 'y'))

    @jax.jit
    def f(a, r):
      b0 = lax.broadcast_in_dim(
          r, (4, 3, 4), (0, 2), out_sharding=P('x', None, 'y'))
      b1 = lax.broadcast_in_dim(
          a, (4, 3, 4), (0, 2), out_sharding=P('x', None, 'y'))
      return b0 + b1

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(x, row).as_text())
      expected = f(x, row)

    self.assertIn('sdy.manual_computation', f.lower(x, row).as_text())

    out = f(x, row)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    for bad_spec in (P(None, None, 'y'), P(None, 'y', 'x')):
      with self.assertRaises(NotImplementedError):
        jax.jit(lambda a: lax.broadcast_in_dim(
            a, (2, 4, 4), (1, 2), out_sharding=bad_spec)).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_squeeze(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 1, 4), P('x', None, 'y'))

    @jax.jit
    def f(a):
      return lax.squeeze(a, dimensions=(1,))

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    with self.assertRaises(ValueError):
      jax.jit(lambda a: lax.squeeze(a, dimensions=(0,))).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_transpose(self, mesh):
    x = jax.device_put(
        np.arange(48, dtype=np.float32).reshape(4, 2, 6), P('x', None, 'y'))
    y = jax.device_put(
        np.arange(24, dtype=np.float32).reshape(8, 3), P(('x', 'y'), None))

    @jax.jit
    def f(a, b):
      return lax.transpose(a, (2, 0, 1)), lax.transpose(b, (1, 0))

    with config.shard_map_partitioning(False):
      exp0, exp1 = f(x, y)

    out0, out1 = f(x, y)
    self.assertEqual(out0.sharding, exp0.sharding)
    self.assertEqual(out1.sharding, exp1.sharding)
    self.assertArraysEqual(out0, exp0)
    self.assertArraysEqual(out1, exp1)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_split(self, mesh):
    x = jax.device_put(
        np.arange(24, dtype=np.float32).reshape(4, 6), P('x', None))

    @jax.jit
    def f(a):
      return lax.split(a, (2, 4), axis=1)

    with config.shard_map_partitioning(False):
      exp0, exp1 = f(x)

    out0, out1 = f(x)
    self.assertEqual(out0.sharding, exp0.sharding)
    self.assertEqual(out1.sharding, exp1.sharding)
    self.assertArraysEqual(out0, exp0)
    self.assertArraysEqual(out1, exp1)

    xy = jax.device_put(
        np.arange(24, dtype=np.float32).reshape(4, 6), P('x', 'y'))

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: lax.split(a, (2, 2), axis=0)).lower(xy)

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda a: lax.split(a, (1, 3), axis=0)).lower(xy)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_stack(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))
    y = jax.device_put(
        np.arange(16, 32, dtype=np.float32).reshape(4, 4), P('x', 'y'))

    @jax.jit
    def f(a, b):
      return jnp.stack([a, b], axis=1)

    with config.shard_map_partitioning(False):
      expected = f(x, y)

    out = f(x, y)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    bad_y = jax.device_put(np.zeros((4, 4), np.float32), P('y', 'x'))

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda a, b: jnp.stack([a, b], axis=0)).lower(x, bad_y)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: jnp.stack(
          [a, np.zeros(a.shape, np.float32)], axis=0)).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_unstack(self, mesh):
    x = jax.device_put(
        np.arange(48, dtype=np.float32).reshape(4, 3, 4), P('x', None, 'y'))

    @jax.jit
    def f(a):
      return jnp.unstack(a, axis=1)

    with config.shard_map_partitioning(False):
      expected = f(x)

    for out, exp in zip(f(x), expected):
      self.assertEqual(out.sharding, exp.sharding)
      self.assertArraysEqual(out, exp)

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda a: jnp.unstack(a, axis=0)).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_pad(self, mesh):
    x = jax.device_put(
        np.arange(32, dtype=np.float32).reshape(4, 2, 4), P('x', None, 'y'))

    @jax.jit
    def f(a):
      return lax.pad(a, jnp.float32(0.0), ((0, 0, 0), (1, 2, 1), (0, 0, 0)))

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(x).as_text())
      expected = f(x)

    self.assertIn('sdy.manual_computation', f.lower(x).as_text())

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: lax.pad(
          a, jnp.float32(0.0), ((2, 0, 0), (0, 0, 0), (0, 0, 0)))).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_tile(self, mesh):
    x = jax.device_put(
        np.arange(32, dtype=np.float32).reshape(4, 2, 4), P('x', None, 'y'))

    @jax.jit
    def f(a):
      return lax.tile(a, (1, 2, 1))

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda a: lax.tile(a, (2, 1, 1))).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_clamp(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))
    lo = jax.device_put(
        np.full((4, 4), 3.0, dtype=np.float32), P(None, 'y'))

    @jax.jit
    def f(a, l):
      return lax.clamp(l, a, jnp.float32(12.0))

    with config.shard_map_partitioning(False):
      expected = f(x, lo)

    out = f(x, lo)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: lax.clamp(a, np.ones(a.shape, np.float32), a)).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_select_n(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))
    c = np.full((4, 4), -1.0, dtype=np.float32)

    @jax.jit
    def f(a):
      return lax.select_n(a > 7.0, a * 2, c)

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: lax.select_n(
          a > 0, np.zeros(a.shape, np.float32),
          np.ones(a.shape, np.float32))).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_add_any(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))
    y = jax.device_put(
        np.ones((4, 4), dtype=np.float32), P('x', 'y'))

    @jax.jit
    def f(a, b):
      _, vjp_fn = jax.vjp(lambda u: (u * 2.0, u * 3.0), a)
      (g,) = vjp_fn((b, b))
      return ad_util.add_jaxvals_p.bind(g, b)

    with config.shard_map_partitioning(False):
      expected = f(x, y)

    out = f(x, y)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_iota(self, mesh):
    @jax.jit
    def f():
      return lax.broadcasted_iota(
          np.int32, (4, 3, 4), 1, out_sharding=P('x', None, 'y'))

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower().as_text())
      expected = f()

    self.assertIn('sdy.manual_computation', f.lower().as_text())

    out = f()
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda: lax.broadcasted_iota(
          np.int32, (4, 4), 0, out_sharding=P('x', 'y'))).lower()

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_top_k(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', None))

    @jax.jit
    def f(a):
      return lax.top_k(a, k=3, axis=1)

    with config.shard_map_partitioning(False):
      exp_v, exp_i = f(x)

    out_v, out_i = f(x)
    self.assertEqual(out_v.sharding, exp_v.sharding)
    self.assertEqual(out_i.sharding, exp_i.sharding)
    self.assertArraysEqual(out_v, exp_v)
    self.assertArraysEqual(out_i, exp_i)

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda a: lax.top_k(a, k=2, axis=0)).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce_precision(self, mesh):
    x = jax.device_put(
        (np.arange(16, dtype=np.float32).reshape(4, 4) + 0.12345),
        P('x', 'y'))

    @jax.jit
    def f(a):
      return lax.reduce_precision(a, exponent_bits=8, mantissa_bits=7)

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_threefry_and_iota_2x32(self, mesh):
    from jax._src.random import prng, threefry2x32

    k = jax.device_put(
        np.arange(4, dtype=np.uint32).reshape(4, 1), P('x', None))
    c_sharded = jax.device_put(
        np.arange(16, dtype=np.uint32).reshape(4, 4), P('x', None))

    @jax.jit
    def f(k, cs):
      c0, c1 = prng.iota_2x32_shape((4, 4))
      r0, r1 = threefry2x32.threefry2x32_p.bind(
          k, k + np.uint32(1), c0 + cs, c1)
      return r0, r1

    with config.shard_map_partitioning(False):
      exp0, exp1 = f(k, c_sharded)

    out0, out1 = f(k, c_sharded)
    self.assertEqual(out0.sharding, exp0.sharding)
    self.assertEqual(out1.sharding, exp1.sharding)
    self.assertArraysEqual(out0, exp0)
    self.assertArraysEqual(out1, exp1)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_edtype_casts(self, mesh):
    dt = dtypes.primal_tangent_dtype(jnp.int8, jnp.float32)
    x = jax.device_put(
        np.arange(16, dtype=np.int8).reshape(4, 4), P('x', 'y'))

    @jax.jit
    def f(a):
      return a.astype(dt).astype(jnp.int8)

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @parameterized.parameters(
      ((4, 8), P('x', 'y')),
      ((8, 3), P(('x', 'y'), None)),
      ((2, 2), P(None, None)),
      ((), None),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_empty(self, shape, spec, mesh):
    @jax.jit
    def f():
      return lax.empty(shape, jnp.float32, out_sharding=spec)

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower().as_text())
      expected = f()

    self.assertIn('sdy.manual_computation', f.lower().as_text())

    # lax.empty returns uninitialized memory, so compare aval and sharding.
    out = f()
    self.assertEqual(out.aval, expected.aval)
    self.assertEqual(out.sharding, expected.sharding)
    expected_spec = P() if spec is None else spec
    self.assertEqual(out.sharding, NamedSharding(mesh, expected_spec))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_passthrough_and_composite_primitives(self, mesh):
    from jax.ad_checkpoint import checkpoint_name
    from jax.experimental import xla_metadata

    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y')
    )
    y = jax.device_put(np.arange(4, dtype=np.int32), P('x'))

    def f(x, y):
      x = xla_metadata.set_xla_metadata(x + x, test_attr='keep_me')
      x = checkpoint_name(lax.stop_gradient(x), 'act')
      x, y = lax.optimization_barrier((x, y))
      k1, k2 = jax.random.split(jax.random.key(0))
      k1 = jax.random.fold_in(k1, 1)
      bits = jax.random.bits(k2, (4,))
      return x, y, jax.random.key_data(k1), bits

    f_jit = jax.jit(f)

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f_jit.lower(x, y).as_text())
      expected = f_jit(x, y)

    lowered = f_jit.lower(x, y).as_text()
    self.assertIn('sdy.manual_computation', lowered)
    self.assertIn('keep_me', lowered)

    for out, exp in zip(f_jit(x, y), expected):
      self.assertEqual(out.sharding, exp.sharding)
      self.assertArraysEqual(out, exp)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_remat_opt(self, mesh):
    x = jax.device_put(
        np.arange(1, 17, dtype=np.float32).reshape(4, 4), P('x', 'y'))

    g = jax.custom_vjp(jnp.sin)
    g.defvjp(lambda u: (jnp.sin(u), jnp.cos(u)), lambda c, ct: (c * ct,),
             optimize_remat=True)

    @jax.jit
    def f(a):
      _, vjp_fn = jax.vjp(jax.checkpoint(g), a)
      (grad_a,) = vjp_fn(a)
      return grad_a

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_linear_call(self, mesh):
    from jax._src import custom_derivatives

    x = jax.device_put(
        np.arange(1, 17, dtype=np.float32).reshape(4, 4), P('x', 'y'))

    @jax.jit
    def f(a):
      def h(u):
        return custom_derivatives.linear_call(
            lambda r, v: r * v, lambda r, ct: r * ct, jnp.float32(3.0), u)

      _, vjp_fn = jax.vjp(h, a)
      (grad_a,) = vjp_fn(a)
      return grad_a

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_integer_pow(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))

    @jax.jit
    def f(a):
      return a ** 3

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_copy(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))

    @jax.jit
    def f(a):
      return jnp.copy(a)

    with config.shard_map_partitioning(False):
      expected = f(x)

    out = f(x)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_bitcast_convert_type(self, mesh):
    x32 = jax.device_put(
        np.arange(16, dtype=np.int32).reshape(4, 4), P('x', 'y'))
    x16 = jax.device_put(
        np.arange(8, dtype=np.int16).reshape(4, 2), P('x', None))

    @jax.jit
    def f(a32, a16):
      same = lax.bitcast_convert_type(a32, np.float32)
      narrow = lax.bitcast_convert_type(a32, np.int16)
      wide = lax.bitcast_convert_type(a16, np.int32)
      return same, narrow, wide

    with config.shard_map_partitioning(False):
      expected = f(x32, x16)

    for out, exp in zip(f(x32, x16), expected):
      self.assertEqual(out.sharding, exp.sharding)
      self.assertArraysEqual(out, exp)

    bad_x16 = jax.device_put(
        np.arange(8, dtype=np.int16).reshape(4, 2), P('x', 'y'))

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: lax.bitcast_convert_type(a, np.int32)).lower(bad_x16)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_argmin_and_argmax(self, mesh):
    x = jax.device_put(
        np.array([[3, 1, 4, 0], [2, 9, 5, 1], [8, 0, 2, 7], [6, 4, 3, 5]],
                 dtype=np.float32),
        P('x', None))

    @jax.jit
    def f(a):
      return jnp.argmin(a, axis=1), jnp.argmax(a, axis=1)

    with config.shard_map_partitioning(False):
      exp_min, exp_max = f(x)

    out_min, out_max = f(x)
    self.assertEqual(out_min.sharding, exp_min.sharding)
    self.assertEqual(out_max.sharding, exp_max.sharding)
    self.assertArraysEqual(out_min, exp_min)
    self.assertArraysEqual(out_max, exp_max)

    for op in (jnp.argmin, jnp.argmax):
      with self.assertRaises(NotImplementedError):
        jax.jit(lambda a: op(a, axis=0)).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_sort(self, mesh):
    keys = jax.device_put(
        np.array([[3, 1, 4, 0], [2, 9, 5, 1], [8, 0, 2, 7], [6, 4, 3, 5]],
                 dtype=np.float32),
        P('x', None))
    vals = jax.device_put(
        np.arange(16, dtype=np.int32).reshape(4, 4), P('x', None))

    @jax.jit
    def f(k, v):
      return lax.sort(k, dimension=1), lax.sort((k, v), dimension=1, num_keys=1)

    with config.shard_map_partitioning(False):
      exp_s, (exp_k, exp_v) = f(keys, vals)

    out_s, (out_k, out_v) = f(keys, vals)
    for out, exp in ((out_s, exp_s), (out_k, exp_k), (out_v, exp_v)):
      self.assertEqual(out.sharding, exp.sharding)
      self.assertArraysEqual(out, exp)

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda k: lax.sort(k, dimension=0)).lower(keys)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda k: lax.sort(
          (k, np.zeros((4, 4), np.int32)), dimension=1, num_keys=1)).lower(keys)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_cumred(self, mesh):
    x = jax.device_put(
        np.arange(1, 17, dtype=np.float32).reshape(4, 4), P('x', None))

    @jax.jit
    def f(a):
      return jnp.cumsum(a, axis=1), jnp.cumprod(a, axis=1)

    with config.shard_map_partitioning(False):
      exp_sum, exp_prod = f(x)

    out_sum, out_prod = f(x)
    self.assertEqual(out_sum.sharding, exp_sum.sharding)
    self.assertEqual(out_prod.sharding, exp_prod.sharding)
    self.assertArraysEqual(out_sum, exp_sum)
    self.assertArraysEqual(out_prod, exp_prod)

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda a: jnp.cumsum(a, axis=0)).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_dynamic_slice(self, mesh):
    x = jax.device_put(
        np.arange(48, dtype=np.float32).reshape(4, 3, 4), P('x', None, 'y'))
    off = jax.device_put(np.int32(1), P())

    @jax.jit
    def f(a, i):
      return lax.dynamic_slice(a, (i, i, i), (4, 1, 4))

    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f.lower(x, off).as_text())
      expected = f(x, off)

    self.assertIn('sdy.manual_computation', f.lower(x, off).as_text())

    out = f(x, off)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: lax.dynamic_slice(a, (1, 0, 0), (2, 2, 4))).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_dynamic_update_slice(self, mesh):
    x = jax.device_put(
        np.arange(48, dtype=np.float32).reshape(4, 3, 4), P('x', None, 'y'))
    upd = jax.device_put(
        np.ones((4, 1, 4), dtype=np.float32), P('x', None, 'y'))
    off = jax.device_put(np.int32(1), P())

    @jax.jit
    def f(a, u, i):
      return lax.dynamic_update_slice(a, u, (i, i, i))

    with config.shard_map_partitioning(False):
      expected = f(x, upd, off)

    out = f(x, upd, off)
    self.assertEqual(out.sharding, expected.sharding)
    self.assertArraysEqual(out, expected)

    bad_upd = jax.device_put(
        np.zeros((2, 3, 4), dtype=np.float32), P('x', None, 'y'))

    with self.assertRaises(NotImplementedError):
      jax.jit(
          lambda a, u: lax.dynamic_update_slice(a, u, (0, 0, 0))
      ).lower(x, bad_upd)

    rep_upd = jax.device_put(
        np.zeros((4, 1, 4), dtype=np.float32), P(None, None, None))

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(
          lambda a, u: lax.dynamic_update_slice(a, u, (0, 0, 0))
      ).lower(x, rep_upd)

    # lax.scan allocates its output buffer via lax.empty2 and writes via
    # dynamic_update_slice.
    c0 = jax.device_put(
        np.arange(8, dtype=np.float32).reshape(2, 4), P('x', 'y'))

    @jax.jit
    def run_scan(c):
      def body(carry, _):
        return carry + 1, carry * 2

      return lax.scan(body, c, None, length=3)

    with config.shard_map_partitioning(False):
      expected_carry, expected_ys = run_scan(c0)

    carry, ys = run_scan(c0)
    self.assertEqual(carry.sharding, expected_carry.sharding)
    self.assertEqual(ys.sharding, expected_ys.sharding)
    self.assertArraysEqual(carry, expected_carry)
    self.assertArraysEqual(ys, expected_ys)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_gather(self, mesh):
    x = jax.device_put(
        np.arange(48, dtype=np.float32).reshape(4, 3, 4), P('x', None, 'y'))
    table = jax.device_put(
        np.arange(32, dtype=np.float32).reshape(8, 4), P(None, 'y'))
    sharded_idx = jax.device_put(
        np.array([2, 0, 5, 1], dtype=np.int32), P('x'))
    idx = jax.device_put(np.array([2, 0], dtype=np.int32), P(None))

    @jax.jit
    def f(a, tbl, s_idx, i):
      vds = jax.vmap(
          lambda r, j: lax.dynamic_slice(r, (j, np.int32(0)), (1, 4))
      )(a, s_idx)
      emb = tbl.at[s_idx].get(out_sharding=P('x', 'y'))
      return vds, emb, tbl[i]

    with config.shard_map_partitioning(False):
      expected = f(x, table, sharded_idx, idx)

    for out, exp in zip(f(x, table, sharded_idx, idx), expected):
      self.assertEqual(out.sharding, exp.sharding)
      self.assertArraysEqual(out, exp)

    with self.assertRaises(NotImplementedError):
      jax.jit(
          lambda t: t.at[idx].get(out_sharding=P(None, None))
      ).lower(table)

    with self.assertRaises(NotImplementedError):
      jax.jit(
          lambda a: a.at[idx].get(out_sharding=P(None, None, 'y'))
      ).lower(x)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_scatter(self, mesh):
    x = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P(None, 'y'))
    upd = jax.device_put(
        np.arange(8, dtype=np.float32).reshape(2, 4), P(None, 'y'))
    idx = jax.device_put(np.array([3, 1], dtype=np.int32), P(None))
    bx = jax.device_put(
        np.arange(48, dtype=np.float32).reshape(4, 3, 4), P('x', None, 'y'))
    bupd = jax.device_put(np.ones((4, 4), dtype=np.float32), P('x', 'y'))
    bidx = jax.device_put(np.array([2, 0, 1, 2], dtype=np.int32), P('x'))

    full_upd = jax.device_put(np.ones((4, 4), dtype=np.float32), P(None, 'y'))
    full_dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(0, 1),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0, 1),
    )

    @jax.jit
    def f(a, u, i, ba, bu, bi, fu):
      s_set = a.at[i].set(u)
      s_add = a.at[i].add(u, out_sharding=P(None, 'y'))
      s_max = a.at[i].max(u)
      s_vmap = jax.vmap(lambda r, j, v: r.at[j].add(v))(ba, bi, bu)
      s_clip = lax.scatter(
          a, jnp.array([0, 1], dtype=np.int32), fu, full_dnums, mode='clip')
      return s_set, s_add, s_max, s_vmap, s_clip

    args = (x, upd, idx, bx, bupd, bidx, full_upd)

    with config.shard_map_partitioning(False):
      expected = f(*args)

    for out, exp in zip(f(*args), expected):
      self.assertEqual(out.sharding, exp.sharding)
      self.assertArraysEqual(out, exp)

    xy = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4), P('x', 'y'))

    with self.assertRaises(core.ShardingTypeError):
      jax.jit(lambda a: a.at[idx].add(upd)).lower(xy)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: a.at[idx].add(upd, out_sharding=P('x', 'y'))).lower(xy)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: a.at[idx].add(upd, out_sharding=P(None, None))).lower(x)

    op_nx = jax.device_put(np.zeros((2, 2), dtype=np.float32), P(None, 'x'))
    idx_xn = jax.device_put(np.zeros((2, 1), dtype=np.int32), P('x', None))
    batch_dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a, i: lax.scatter_add(
          a, i, a, batch_dnums, out_sharding=P(None, 'x')
      )).lower(op_nx, idx_xn)

    rep1d = jax.device_put(np.ones((4,), dtype=np.float32), P(None))

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda r: r.at[0].apply(lambda v: v + 1.0)).lower(rep1d)

    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: lax.scatter(
          a, jnp.array([0, 1], dtype=np.int32), full_upd, full_dnums,
          mode='fill')).lower(x)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
