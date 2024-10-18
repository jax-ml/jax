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

from typing import Any, Sequence
import unittest
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from jax import lax
from jax._src import pjit
from jax._src import core
import jax._src.interpreters.jaxpr_passes as jaxpr_passes
import jax._src.test_util as jtu
from jax._src.lax import lax as lax_internal
from jax._src.sharding_impls import NamedSharding
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import shard_map


jax.config.parse_flags_with_absl()


Shape = Sequence[int]


def find_primitive(jaxpr: core.Jaxpr,
                  primitive: core.Primitive):
  for eqn in jaxpr.eqns:
    if eqn.primitive == primitive:
      return eqn
    if 'jaxpr' in eqn.params:
      try:
        return find_primitive(eqn.params['jaxpr'], primitive)
      except ValueError:
        continue
  raise ValueError(f"Primitive not found: {primitive.name}\n\tin JAXPR: {jaxpr}")


class PhysicalizeTest(unittest.TestCase):

  def assert_jaxpr(self,
                jaxpr: core.ClosedJaxpr,
                primitive: core.Primitive,
                expected_in_shapes: Sequence[Shape],
                expected_in_dtypes: Sequence[Any],
                expected_out_shapes: Sequence[Shape],
                expected_out_dtypes: Sequence[Any],
                ):
    broadcast_op = find_primitive(jaxpr, primitive)
    for invar, expected_shape, expected_dt in zip(
      broadcast_op.invars, expected_in_shapes, expected_in_dtypes):
      in_aval = invar.aval
      self.assertEqual(in_aval.shape, expected_shape)
      self.assertEqual(in_aval.dtype, expected_dt)
    for outvar, expected_shape, expected_dt in zip(
      broadcast_op.outvars, expected_out_shapes, expected_out_dtypes):
      out_aval = outvar.aval
      self.assertEqual(out_aval.shape, expected_shape)
      self.assertEqual(out_aval.dtype, expected_dt)

  def test_broadcast_in_dim(self):
    def fun(k):
      return lax.broadcast_in_dim(k, (2, 2), broadcast_dimensions=(0,))
    k = random.split(random.key(0), 2)
    result = jax.jit(fun)(k)
    self.assertEqual(result.shape, (2, 2))

    traced = jax.jit(fun).trace(k)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    phys_aval = phys_jaxpr.jaxpr.invars[0].aval
    self.assertEqual(phys_aval.shape, (2, 2))
    self.assertEqual(phys_aval.dtype, jnp.uint32)
    self.assert_jaxpr(phys_jaxpr,
                    lax.broadcast_in_dim_p,
                    expected_in_shapes=[(2, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(2, 2, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_slice(self):
    def fun(k):
      return lax.slice(k, (0, 0), (4, 1))
    k = random.split(random.key(0), (4, 4))
    result = jax.jit(fun)(k)
    self.assertEqual(result.shape, (4, 1))
    traced = jax.jit(fun).trace(k)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.slice_p,
                    expected_in_shapes=[(4, 4, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(4, 1, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_dynamic_slice(self):
    def fun(k, starts):
      return lax.dynamic_slice(k, starts, (2, 3))
    k = random.split(random.key(0), (4, 4))
    result = jax.jit(fun)(k, (0, 1))
    self.assertEqual(result.shape, (2, 3))
    traced = jax.jit(fun).trace(k, (0, 1))
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                      lax.dynamic_slice_p,
                      expected_in_shapes=[(4, 4, 2)],
                      expected_in_dtypes=[jnp.uint32],
                      expected_out_shapes=[(2, 3, 2)],
                      expected_out_dtypes=[jnp.uint32])


  def test_dynamic_slice_scalar(self):
    f = lambda k: lax.dynamic_slice(k, [], [])
    k = random.key(72)
    result = f(k)
    self.assertEqual(result, k)
    result = jax.jit(f)(k)
    self.assertEqual(result, k)

  def test_dynamic_update_slice(self):
    def fun(k, updates, starts):
      return lax.dynamic_update_slice(k, updates, starts)
    k = random.split(random.key(0), (4, 4))
    updates = random.split(random.key(1), (2, 3))
    result = jax.jit(fun)(k, updates, (0, 1))
    self.assertEqual(result.shape, (4, 4))
    traced = jax.jit(fun).trace(k, updates, (0, 1))
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                      lax.dynamic_update_slice_p,
                      expected_in_shapes=[(4, 4, 2), (2, 3, 2)],
                      expected_in_dtypes=[jnp.uint32, jnp.uint32],
                      expected_out_shapes=[(4, 4, 2)],
                      expected_out_dtypes=[jnp.uint32])

  def test_dynamic_update_slice_scalar(self):
    f = lambda k1, k2: lax.dynamic_update_slice(k1, k2, [])
    k1, k2 = random.key(72), jax.random.key(15)
    result = f(k1, k2)
    self.assertEqual(result, k2)
    result = jax.jit(f)(k1, k2)
    self.assertEqual(result, k2)

  def test_convert_element_type(self):
    def fun(x, y):
      # These converts are no-ops.
      x = lax.convert_element_type(x, core.bint(5))
      x = lax.convert_element_type(x, jnp.int32)
      # Only this convert should exist in the jaxpr.
      x = lax.convert_element_type(x, jnp.float32)
      # These converts are no-ops.
      y = lax.convert_element_type(y, core.bint(8))
      y = lax.convert_element_type(y, jnp.int32)
      return x + y
    result = jax.jit(fun)(2, 3)
    self.assertEqual(result, 5)
    traced = jax.jit(fun).trace(2, 3)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.convert_element_type_p,
                    expected_in_shapes=[()],
                    expected_in_dtypes=[jnp.int32],
                    expected_out_shapes=[()],
                    expected_out_dtypes=[jnp.float32],
                    )

  def test_empty(self):
    def fun():
      return lax_internal.empty(random.key(0).dtype)
    result = jax.jit(fun)()
    self.assertEqual(result.shape, ())
    np.testing.assert_array_equal(jax.random.key_data(result),
                                  jnp.zeros((2,)))

  def test_gather(self):
    def fun(k, starts):
      return lax.gather(k, starts,
                        lax.GatherDimensionNumbers((0, 1), (), (0, 1)),
                        (2, 3))
    k = random.split(random.key(0), (4, 4))
    starts = jnp.array([0, 1])
    result = jax.jit(fun)(k, starts)
    np.testing.assert_array_equal(jax.random.key_data(result),
                                  jax.random.key_data(k[0:2, 1:4]))
    traced = jax.jit(fun).trace(k, starts)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.gather_p,
                    expected_in_shapes=[(4, 4, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(2, 3, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_scatter(self):
    def fun(k, indices, updates):
      return lax.scatter(k,
                         indices,
                         updates,
                        lax.ScatterDimensionNumbers((0, 1), (), (0, 1)))
    k = random.split(random.key(0), (4, 4))
    updates = random.split(random.key(1), (2, 3))
    indices = jnp.array([0, 1])
    result = jax.jit(fun)(k, indices, updates)
    np.testing.assert_array_equal(jax.random.key_data(result),
                                  jax.random.key_data(k).at[0:2, 1:4].set(
                                    jax.random.key_data(updates)))
    traced = jax.jit(fun).trace(k, indices, updates)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.scatter_p,
                    expected_in_shapes=[(4, 4, 2), (2,), (2, 3, 2)],
                    expected_in_dtypes=[jnp.uint32, jnp.int32, jnp.uint32],
                    expected_out_shapes=[(4, 4, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_compare(self):
    def fun(x, y):
      return x == y
    x = random.split(random.key(0), (2, 3))
    y = random.split(random.key(1), (2, 3))
    result = jax.jit(fun)(x, y)
    np.testing.assert_array_equal(result,
                                  jnp.zeros((2, 3)))
    traced = jax.jit(fun).trace(x, y)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.eq_p,
                    expected_in_shapes=[(2, 3, 2), (2, 3, 2)],
                    expected_in_dtypes=[jnp.uint32, jnp.uint32],
                    expected_out_shapes=[(2, 3, 2)],
                    expected_out_dtypes=[jnp.bool_],
                    )

  def test_select(self):
    def fun(which, x, y):
      return lax.select(which, x, y)

    which = jnp.array([True, False, True])
    x = random.split(random.key(0), (3,))
    y = random.split(random.key(1), (3,))
    result = jax.jit(fun)(which, x, y)
    np.testing.assert_array_equal(result,
                                  jnp.stack([x[0], y[1], x[2]]))
    traced = jax.jit(fun).trace(which, x, y)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.select_n_p,
                    expected_in_shapes=[(3, 2), (3, 2), (3, 2)],
                    expected_in_dtypes=[jnp.bool_, jnp.uint32, jnp.uint32],
                    expected_out_shapes=[(3, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_transpose(self):
    def fun(x):
      return jnp.transpose(x, (1, 2, 0))
    x = random.split(random.key(0), (2, 3, 7))
    result = jax.jit(fun)(x)
    self.assertEqual(result.shape, (3, 7, 2))
    traced = jax.jit(fun).trace(x)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.transpose_p,
                    expected_in_shapes=[(2, 3, 7, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(3, 7, 2, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_reshape(self):
    def fun(x):
      return jnp.reshape(x, (6, 7))
    x = random.split(random.key(0), (2, 3, 7))
    result = jax.jit(fun)(x)
    self.assertEqual(result.shape, (6, 7))
    traced = jax.jit(fun).trace(x)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    self.assert_jaxpr(phys_jaxpr,
                    lax.reshape_p,
                    expected_in_shapes=[(2, 3, 7, 2)],
                    expected_in_dtypes=[jnp.uint32],
                    expected_out_shapes=[(6, 7, 2)],
                    expected_out_dtypes=[jnp.uint32],
                    )

  def test_random(self):
    def fun(x):
      k = random.key(x)
      k1, k2 = random.split(k)
      k1 = random.fold_in(k1, 0)
      x1 = random.uniform(k1, shape=(4, 8))
      k2 = random.key_data(k2)
      k2 = random.wrap_key_data(k2, impl=k.dtype._impl)
      x2 = random.uniform(k2, shape=(4, 8))
      return x1 + x2
    result = jax.jit(fun)(0)
    self.assertEqual(result.shape, (4, 8))

    vmapped_fun = jax.vmap(fun)
    seeds = jnp.arange(10)
    with self.subTest('vmapped'):
      result = jax.jit(vmapped_fun)(seeds)
      self.assertEqual(result.shape, (10, 4, 8))


class PhysicalizeTransformationsTest(parameterized.TestCase):
  def test_shard_map_eager(self):
    if jax.device_count() < 2:
      self.skipTest('sharding test requires at least 2 devices.')
    def fun(k, y):
      out = jax.random.uniform(k[0], shape=(5, 4)) + y[0, 0]
      return out
    n_devices = jax.device_count()
    x_dim = n_devices // 2
    mesh = Mesh(devices=mesh_utils.create_device_mesh((x_dim, 2)), axis_names=('x', 'y'))
    keys = random.split(random.key(0), (x_dim,))
    keys = jax.device_put(keys, NamedSharding(mesh, P('x',)))
    y = jnp.ones((x_dim, 2))
    y = jax.device_put(y, NamedSharding(mesh, P('x', 'y')))
    mapped_fun = shard_map.shard_map(jax.jit(fun),
                        mesh=mesh,
                        in_specs=(P('x',), P('x', 'y')),
                        out_specs=P('x', 'y')
                        )
    result = mapped_fun(keys, y)
    self.assertEqual(result.shape, (x_dim * 5, 8))

  def test_shard_map_inside_jit(self):
    if jax.device_count() < 2:
      self.skipTest('sharding test requires at least 2 devices.')
    def fun(k, y):
      out = jax.random.uniform(k[0], shape=(5, 4)) + y[0, 0]
      return out
    n_devices = jax.device_count()
    x_dim = n_devices // 2
    mesh = Mesh(devices=mesh_utils.create_device_mesh((x_dim, 2)), axis_names=('x', 'y'))
    keys = random.split(random.key(0), (x_dim,))
    keys = jax.device_put(keys, NamedSharding(mesh, P('x',)))
    y = jnp.ones((x_dim, 2))
    y = jax.device_put(y, NamedSharding(mesh, P('x', 'y')))
    mapped_fun = shard_map.shard_map(fun,
                        mesh=mesh,
                        in_specs=(P('x',), P('x', 'y')),
                        out_specs=P('x', 'y')
                        )
    mapped_fun = jax.jit(mapped_fun)
    result = mapped_fun(keys, y)
    self.assertEqual(result.shape, (x_dim * 5, 8))

  def test_pjit_sharding(self):
    if jax.device_count() < 2:
      self.skipTest('sharding test requires at least 2 devices.')
    def fun(k):
      return jax.random.key_data(k)
    n_devices = jax.device_count()
    mesh = Mesh(devices=mesh_utils.create_device_mesh((n_devices, )), axis_names=('x',))
    keys = random.split(random.key(0), (n_devices,))
    keys = jax.device_put(keys, NamedSharding(mesh, P('x',)))
    jitted_fun = pjit.pjit(fun,
                        in_shardings=NamedSharding(mesh, P('x')),
                        out_shardings=NamedSharding(mesh, P('x')),
                        )
    result = jax.jit(jitted_fun)(keys)
    self.assertEqual(result.shape, (n_devices, 2))
    traced = jax.jit(jitted_fun).trace(keys)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    pjit_eqn = find_primitive(phys_jaxpr, pjit.pjit_p)
    in_shardings = pjit_eqn.params['in_shardings']
    self.assertEqual(len(in_shardings), 1)
    # pjit should insert an extra None to account for the physical dimension.
    self.assertEqual(in_shardings[0].spec, P('x', None))
    out_shardings = pjit_eqn.params['out_shardings']
    self.assertEqual(len(out_shardings), 1)
    self.assertEqual(out_shardings[0].spec, P('x',))

  def test_sharding_constraint(self):
    if jax.device_count() < 2:
      self.skipTest('sharding test requires at least 2 devices.')
    def fun(k):
      x = lax.with_sharding_constraint(k, NamedSharding(mesh, P('x')))
      x = jax.random.key_data(k)
      return x
    n_devices = jax.device_count()
    mesh = Mesh(devices=mesh_utils.create_device_mesh((n_devices, )), axis_names=('x',))
    keys = random.split(random.key(0), (n_devices, 4))
    keys = jax.device_put(keys, NamedSharding(mesh, P('x')))
    jitted_fun = pjit.pjit(fun,
                        in_shardings=NamedSharding(mesh, P('x')),
                        out_shardings=NamedSharding(mesh, P('x')),
                        )
    result = jax.jit(jitted_fun)(keys)
    self.assertEqual(result.shape, (n_devices, 4, 2))
    traced = jax.jit(jitted_fun).trace(keys)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    sharding_constraint_eqn = find_primitive(phys_jaxpr, pjit.sharding_constraint_p)
    sharding = sharding_constraint_eqn.params['sharding']
    # sharding_constraint should insert an extra None to account for the physical dimension.
    self.assertEqual(sharding.spec, P('x', None))

  def test_all_gather(self):
    def fun(k):
      k = lax.all_gather(k, 'x')  # Shape: [n_devices, 1]
      return jax.random.key_data(k) # Shape: [n_devices, 1, 2]
    n_devices = jax.device_count()
    mesh = Mesh(devices=mesh_utils.create_device_mesh((n_devices, )), axis_names=('x',))
    keys = random.split(random.key(0), (n_devices,))
    keys = jax.device_put(keys, NamedSharding(mesh, P('x',)))
    mapped_fun = shard_map.shard_map(fun,
                        mesh=mesh,
                        in_specs=(P('x'),),
                        out_specs=P('x')
                        )
    result = jax.jit(mapped_fun)(keys)
    self.assertEqual(result.shape, (n_devices * n_devices, 1, 2))

  def test_batched_while_loop(self):
    def _cond_fun(val):
      return val[0] < 10

    def _body_fun(val):
      return (val[0] + 1, val[1])

    def fun(keys):
      def _generate_loop(k):
        return lax.while_loop(_cond_fun,
                              _body_fun,
                              (0, k))[1]
      return jax.vmap(_generate_loop)(keys)

    keys = random.split(random.key(0), (10,))
    result = jax.jit(fun)(keys)
    np.testing.assert_array_equal(result, keys)
    eager_result = jax.jit(fun)(keys)
    np.testing.assert_array_equal(eager_result, keys)
    traced = jax.jit(fun).trace(keys)
    phys_jaxpr = jaxpr_passes.resolve_edtypes_jaxpr(traced.jaxpr)
    while_eqn = find_primitive(phys_jaxpr, lax.while_p)
    body_invars = while_eqn.params['body_jaxpr'].jaxpr.invars
    self.assertEqual(body_invars[1].aval.shape, (10, 2))
    self.assertEqual(body_invars[1].aval.dtype, jnp.uint32)
    cond_invars = while_eqn.params['cond_jaxpr'].jaxpr.invars
    self.assertEqual(cond_invars[1].aval.shape, (10, 2))
    self.assertEqual(cond_invars[1].aval.dtype, jnp.uint32)

  def test_scan(self):
    def _body_fun(carry, _):
      return (carry, carry)

    def fun(keys):
      final_carry, _ = lax.scan(_body_fun, keys, jnp.arange(5))
      return final_carry

    keys = random.split(random.key(0), (10,))
    result = jax.jit(fun)(keys)
    np.testing.assert_array_equal(result, keys)

  @parameterized.parameters((True,), (False,))
  def test_cond(self, pred):
    def true_fn(key):
      return random.split(key)[0]

    def false_fn(key):
      return random.split(key)[1]

    def fun_with_cond(key, pred):
      return lax.cond(pred, true_fn, false_fn, key)

    key = random.key(0)
    true_key, false_key = random.split(key)
    result = jax.jit(fun_with_cond)(key, pred)
    if pred:
      np.testing.assert_array_equal(result, true_key)
    else:
      np.testing.assert_array_equal(result, false_key)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
