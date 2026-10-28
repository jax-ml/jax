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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
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
    lowered_text = f_jit.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered_text)
    self.assertEqual('all_reduce' in lowered_text, has_ar)

    out = f_jit(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axis))

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
    np_inp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    arr = jax.device_put(np_inp, in_spec)
    out_s = None if out_spec.reduced else out_spec

    @jax.jit
    def f(x):
      y = lax_op(x, axes=axes, out_sharding=out_s)
      self.assertEqual(jax.typeof(y).sharding.spec, out_spec)
      return y

    uses_shmap = (is_rs and (lax_op is lax.reduce_sum)) or bool(out_spec.reduced)
    if not uses_shmap:
      with self.assertRaises(NotImplementedError):
        f.lower(arr)
      return

    lowered_text = f.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered_text)
    self.assertEqual('reduce_scatter' in lowered_text, is_rs)
    self.assertEqual('all_reduce' in lowered_text, has_ar)

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axes))

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh(
      (2, 2, 2), ('m', 'e', 'a'),
      axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Auto))
  def test_mixed_manual_explicit_auto(self, mesh):
    np_x = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    arr = jax.device_put(np_x, NamedSharding(mesh, P('m', 'e', 'a')))

    @jax.jit
    @jax.shard_map(out_specs=P('m', None), axis_names={'m'})
    def f(x):
      return lax.reduce_max(x, axes=(1,))

    hlo_txt = f.lower(arr).compile().as_text()
    self.assertIn("replica_groups=mesh['m'=2,'e'=2,'a'=2] {'e'}", hlo_txt)
    out = f(arr)
    self.assertEqual(out.sharding.spec, P('m', 'a'))
    self.assertArraysEqual(out, np.max(np_x, axis=1))

    g = jax.jit(jax.sharding.auto_axes(
        lambda y: jnp.sum(y, axis=0), axes='e', out_sharding=P(None, None)))
    with self.assertRaises(NotImplementedError):
      g.lower(arr)

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

    self.assertAllClose(f(arr), np.sum(np_x, axis=0))

    @jax.jit
    @jax.shard_map(out_specs=P(), axis_names={'m'})
    def f_idx_psum():
      return lax.psum(lax.axis_index('m'), 'm')

    self.assertArraysEqual(f_idx_psum(), np.int32(1))

    @jax.jit
    @jax.shard_map(out_specs=P(None, None), axis_names={'m'})
    def f_ag_unsharded_dim(x):
      return lax.all_gather(x, 'm', axis=0, tiled=True, to='invarying')

    self.assertAllClose(f_ag_unsharded_dim(arr), np_x)

    @jax.jit
    @jax.shard_map(out_specs=P(None, None), axis_names={'m'})
    def f_ag_const():
      c = lax.pcast(np.ones((1, 4), dtype=np.float32), 'm', to='varying')
      return lax.all_gather(c, 'm', axis=1, tiled=True, to='invarying')

    self.assertAllClose(f_ag_const(), np.ones((1, 8), dtype=np.float32))

    @jax.jit
    @jax.shard_map(out_specs=P(None, None), axis_names={'m'})
    def f_ag_sharded_dim(x):
      return lax.all_gather(x, 'm', axis=1, tiled=True, to='invarying')

    with self.assertRaises(NotImplementedError):
      f_ag_sharded_dim.lower(arr)

  @parameterized.product(
      op=(
          (jnp.sin, np.sin),
          (jnp.exp, np.exp),
          (jnp.negative, np.negative),
          (jnp.round, np.round),
          (lax.lgamma, None),
      ),
      in_spec=(P('x', 'y'), P(('x', 'y'), None), P(None, None)),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_unop(self, op, in_spec, mesh):
    jnp_op, np_op = op
    np_inp = np.arange(1, 33, dtype=np.float32).reshape(4, 8) / 8.0
    arr = jax.device_put(np_inp, in_spec)

    def f(x):
      self.assertEqual(jax.typeof(x).sharding.spec, in_spec)
      y = jnp_op(x)
      self.assertEqual(jax.typeof(y).sharding.spec, in_spec)
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, in_spec))
    expected = jnp_op(np_inp) if np_op is None else np_op(np_inp)
    self.assertAllClose(out, expected)

    f_jit = jax.jit(f)
    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f_jit.lower(arr).as_text())
    self.assertIn('sdy.manual_computation', f_jit.lower(arr).as_text())

    out = f_jit(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, in_spec))
    self.assertAllClose(out, expected)

    if jnp_op is jnp.negative:
      grad, = jax.jit(lambda x, ct: jax.vjp(f, x)[1](ct))(arr, arr)
      self.assertEqual(grad.sharding, arr.sharding)
      self.assertAllClose(grad, -np_inp)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh(
      (2, 2, 2), ('m', 'e', 'a'),
      axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Auto))
  def test_unop_mixed_axes(self, mesh):
    np_inp = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4) / 16.0
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('m', 'e', 'a')))

    @jax.jit
    @jax.shard_map(out_specs=P('m'), axis_names={'m'})
    def f(a):
      self.assertEqual(jax.typeof(a).sharding.spec, P(None, 'e', None))
      self.assertEqual(a.aval.mat.varying, {'m'})
      y = jnp.tanh(jnp.sin(a))
      self.assertEqual(jax.typeof(y).sharding.spec, P(None, 'e', None))
      self.assertEqual(y.aval.mat.varying, {'m'})
      return jnp.sum(y, axis=1)

    out = f(arr)
    self.assertEqual(out.sharding.spec, P('m', 'a'))
    self.assertAllClose(out, np.sum(np.tanh(np.sin(np_inp)), axis=1))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
