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

    grad = jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(arr)
    self.assertEqual(grad.sharding, arr.sharding)
    self.assertAllClose(
        grad, jax.grad(lambda x: jnp.sum(jnp_op(x, axis=axis)))(np_inp))

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

    uses_shmap = is_rs and (lax_op is lax.reduce_sum)
    lowered_text = f.lower(arr).as_text()
    self.assertEqual('sdy.manual_computation' in lowered_text, uses_shmap)
    if uses_shmap:
      self.assertIn('reduce_scatter', lowered_text)
      self.assertEqual('all_reduce' in lowered_text, has_ar)

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axes))

    if lax_op is lax.reduce_sum and not out_spec.reduced:
      grad = jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(arr)
      self.assertEqual(grad.sharding, arr.sharding)
      self.assertArraysEqual(grad, np.ones_like(np_inp))

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
    self.assertArraysEqual(g(arr), np.sum(np_x, axis=0))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
