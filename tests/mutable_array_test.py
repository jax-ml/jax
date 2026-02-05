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

from __future__ import annotations

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from functools import partial
import itertools as it
import numpy as np
import jax
from jax._src import core
from jax._src import config
from jax._src import test_util as jtu
from jax._src.util import safe_map, safe_zip
from jax._src.interpreters import mlir
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType
import jax.numpy as jnp

from jax._src.state.types import (RefEffect)

config.parse_flags_with_absl()

jtu.request_cpu_devices(8)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


class MutableArrayTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_basic(self, jit):
    def f(x_mut):
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    if jit:
      f = jax.jit(f)

    x_mut = core.new_ref(jnp.zeros(3))
    f(x_mut)

    self.assertAllClose(x_mut[...], jnp.array([2., 6., 1.]),
                        check_dtypes=False)

    jaxpr = jax.make_jaxpr(f)(x_mut)
    self.assertTrue(any(isinstance(e, RefEffect) for e in jaxpr.effects))

  def test_basic_aot(self):
    @jax.jit
    def f(x_mut):
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    x_mut = core.new_ref(jnp.zeros(3))
    f.lower(x_mut).compile()(x_mut)
    self.assertAllClose(x_mut[...], jnp.array([2., 6., 1.]),
                        check_dtypes=False)

  def test_basic_aot_closure(self):
    x_mut = core.new_ref(jnp.zeros(3))

    @jax.jit
    def f():
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    c = f.lower().compile()
    c()
    c()
    self.assertAllClose(x_mut[...], jnp.array([4., 12., 2.]),
                        check_dtypes=False)

  def test_basic_sharded_aot(self):
    mesh = jtu.create_mesh((2,), ('x',))
    arr = jax.device_put(np.arange(8.), NamedSharding(mesh, P('x')))

    @jax.jit
    def f(x_mut):
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    x_mut = core.new_ref(arr)
    f.lower(x_mut).compile()(x_mut)
    expected = np.arange(8.) + 1
    expected[0] += 1
    expected[1] += 5
    self.assertAllClose(x_mut[...], expected)

  def test_sharded_aot_mutable_sds(self):
    mesh = jtu.create_mesh((2,), ('x',))
    arr = jax.device_put(np.arange(8.), NamedSharding(mesh, P('x')))

    @jax.jit
    def f(x_mut):
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    sds_mut = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=arr.sharding,
                                   is_ref=True)
    compiled = f.lower(sds_mut).compile()

    x_mut = core.new_ref(arr)
    compiled(x_mut)

    expected = np.arange(8.) + 1
    expected[0] += 1
    expected[1] += 5
    self.assertAllClose(x_mut[...], expected)

  @parameterized.parameters([True, False])
  def test_multiple_inputs_and_outputs(self, jit):
    def f(x_mut, y, z_mut, w):
      x_mut[...] += 1
      z_mut[...] += 1
      return x_mut[...] + y + z_mut[...] + w, y + w

    if jit:
      f = jax.jit(f)

    x_mut = core.new_ref(jnp.zeros((1, 3)))
    y = jnp.ones((2, 3))
    z_mut = core.new_ref(jnp.zeros((2, 3)))
    w = jnp.ones((2, 1))

    out1, out2 = f(x_mut, y, z_mut, w)

    self.assertAllClose(x_mut[...], jnp.ones((1, 3)), check_dtypes=False)
    self.assertAllClose(z_mut[...], jnp.ones((2, 3)), check_dtypes=False)
    self.assertAllClose(out1, 4 * jnp.ones((2, 3)), check_dtypes=False)
    self.assertAllClose(out2, y + w, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_closed_over_basic(self, jit):
    x_mut = core.new_ref(jnp.zeros(3))
    def f():
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

    if jit:
      f = jax.jit(f)

    f()

    self.assertAllClose(x_mut[...], jnp.array([2., 6., 1.]),
                        check_dtypes=False)

    jaxpr = jax.make_jaxpr(f)()
    self.assertTrue(any(isinstance(e, RefEffect) for e in jaxpr.effects))

  @parameterized.parameters([True, False])
  def test_closed_over_nested(self, jit):
    x_mut = core.new_ref(jnp.zeros(3))

    @jax.jit
    def f(y_mut, z):
      x_mut[...] += 1.
      x_mut[0] += 1
      x_mut[1] += 5

      y_mut[2] += 7
      return z + 9

    if jit:
      f = jax.jit(f)

    y_mut = core.new_ref(np.zeros(3))

    w = f(y_mut, 1)

    self.assertAllClose(x_mut[...], jnp.array([2., 6., 1.]),
                        check_dtypes=False)
    self.assertAllClose(y_mut[...], jnp.array([0., 0., 7.]),
                        check_dtypes=False)
    self.assertAllClose(w, 10, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_len_mutable_array(self, jit):
    x_mut = core.new_ref(jnp.zeros(3))

    def f():
      return jnp.int32(len(x_mut))

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(), 3)

  @parameterized.parameters([True, False])
  def test_internal_mutarray_basic(self, jit):
    def f():
      x_mut = core.new_ref(jnp.zeros(3))
      x_mut[0] += 1
      x_mut[0] += 1
      x_mut[2] += 1
      return x_mut[...]

    if jit:
      f = jax.jit(f)

    out = f()
    self.assertAllClose(out, jnp.array([2., 0., 1.]), check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_internal_mut_array(self, jit):
    def body_fun(_, x):
      x_mut = core.new_ref(x)
      x_mut[...] += 2
      return ((), x_mut[...])
    doit = lambda: jax.lax.scan(body_fun, (), np.arange(5))
    if jit:
      doit = jax.jit(doit)
    _, xs = doit()
    self.assertAllClose(xs, (np.arange(5) + 2), check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_closed_over_mut_array(self, jit):
    x_mut = core.new_ref(0)
    def body_fun(_, x):
      x_mut[...] += 2
      return ((), x_mut[...])

    doit = lambda: jax.lax.scan(body_fun, (), np.arange(5))
    if jit:
      doit = jax.jit(doit)
    _, xs = doit()
    self.assertAllClose(x_mut[...], 10)
    self.assertAllClose(xs, np.arange(5) * 2 + 2, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_scanned_mut_array(self, jit):
    def body_fun(_, index_x):
      (index, x) = index_x
      x[...] += index
      return ((), x[...])

    x_mut = core.new_ref(np.arange(5))
    doit = lambda: jax.lax.scan(body_fun, (), (np.arange(5), x_mut))
    if jit:
      doit = jax.jit(doit)
    _, xs = doit()
    self.assertAllClose(xs, (np.arange(5) * 2), check_dtypes=False)

  def test_double_jit_mutable_array(self):
    @jax.jit
    @jax.jit
    def f():
      x_ref = core.new_ref(jnp.zeros(8))
      return x_ref[...]
    x = f()
    self.assertArraysEqual(x, jnp.zeros(8))

  @parameterized.parameters([False, True])
  def test_grad_mutable_array(self, jit):

    def f(x):
      x_ = core.new_ref(x)
      x_[()] = x_[()] + x_[()]
      y = core.freeze(x_)
      return y

    if jit:
      f = jax.jit(f)

    ans = jax.grad(f)(1.)
    expected = 2.0
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_defensive_copy(self):
    x = jnp.arange(3.)
    _ = jax.jit(lambda x_ref: x_ref[...])(core.new_ref(x))
    x + 1  # don't crash

  def test_sharding_persists(self):
    mesh = jtu.create_mesh((1,), ('i',))
    x = jax.device_put(jnp.arange(2), NamedSharding(mesh, P('i')))
    s = x.sharding
    a = core.new_ref(x)
    self.assertEqual(s, a.sharding)
    self.assertEqual(s, a[...].sharding)
    f = jax.jit(lambda: a[...])
    y = f()
    self.assertEqual(s, a.sharding)
    self.assertEqual(s, y.sharding)

  def test_explicit_sharding_after_indexing(self):
    # https://github.com/jax-ml/jax/issues/26936
    mesh = jtu.create_mesh((1, 1), ('x', 'y'),
                           axis_types=(AxisType.Explicit,) * 2)
    sharding = NamedSharding(mesh, P('x', 'y'))

    @jax.jit
    def f(x_ref):
      self.assertEqual(core.typeof(x_ref).sharding.spec,
                       core.typeof(x_ref[...]).sharding.spec)
      y = x_ref[...] + 1
      return y

    with jax.set_mesh(mesh):
      x = jnp.zeros((4, 4), jnp.int32, device=sharding)
      x_ref = core.new_ref(x)
      y = f(x_ref)

  def test_vmap_basic(self):
    @jax.vmap
    def f(x):
      x_ref = core.new_ref(x)
      x_ref[...] =  x_ref[...] * x_ref[...]
      return x_ref[...]
    xs = jnp.arange(4.)
    ys = f(xs)
    self.assertAllClose(ys, xs ** 2, check_dtypes=False)

  def test_vmap_extensive_inputs(self):
    def f(x_ref, val):
      x_ref[...] += val
      x_ref[...] += val

    xs_ref = core.new_ref(jnp.array([0, 0, 0]))
    vals = jnp.arange(3)
    jax.vmap(f)(xs_ref, vals)
    self.assertAllClose(xs_ref[...], 2 * vals, check_dtypes=False)

  def test_vmap_closed_over_read_only(self):
    y_ref = core.new_ref(1)

    def f(x_ref):
      x_ref[...] += y_ref[...]
      x_ref[...] += y_ref[...]

    xs_ref = core.new_ref(jnp.array([0, 0, 0]))
    jax.vmap(f)(xs_ref)
    self.assertAllClose(xs_ref[...], jnp.array([2, 2, 2]), check_dtypes=False)

  def test_implicit_bitcast_regression(self):
    # https://github.com/jax-ml/jax/issues/27683
    v = core.new_ref(jnp.array([0, 0, 0]))
    with self.assertRaises(ValueError):
      v[...] += 1.0

  def test_implicit_cast_in_swap(self):
    v = core.new_ref(jnp.array(0, dtype='bfloat16'))
    v[...] += 1.0  # don't crash

  def test_rng_key(self):
    key = core.new_ref(jax.random.key(0))
    # test read/write
    key[...] = jax.random.fold_in(key[...], 1) # don't crash

  def test_scan_grad_doesnt_hoist_mutable_stuff(self):
    x_ref = core.new_ref(0)

    def f(x):
      def body(c, _):
        x_ref[...] += 1
        return c, ()
      x, () = jax.lax.scan(body, x, (), length=3)
      return x

    jax.grad(f)(1.0)
    self.assertAllClose(x_ref[...], 3, check_dtypes=False)

  def test_scan_grad_doesnt_hoist_mutable_stuff2(self):
    x_ref = core.new_ref(0)
    const = jnp.arange(3)
    const2 = jnp.zeros(())

    def f(x):
      def body(c, _):
        x_ref[...] += const.sum()
        return c + const2, ()
      x, () = jax.lax.scan(body, x, (), length=4)
      return x

    jax.grad(f)(1.0)
    self.assertAllClose(x_ref[...], 12, check_dtypes=False)

  @parameterized.parameters([False, True])
  def test_custom_vjp_grad_stats_plumbing(self, jit):

   @jax.custom_vjp
   def gradient_history_calculator(x, ref):
     del ref
     return x

   def gradient_history_calculator_fwd(x, ref):
     return x, ref

   def gradient_history_calculator_bwd(amax_history, grad_output):
     amax_update = jnp.max(jnp.abs(grad_output))
     shifted = jnp.roll(amax_history[:], 1)
     shifted = shifted.at[0].set(amax_update)
     amax_history[:] = shifted
     amax_from_history = jnp.max(amax_history[:])
     grad_output = grad_output / amax_from_history
     return grad_output, None

   gradient_history_calculator.defvjp(
     gradient_history_calculator_fwd,
     gradient_history_calculator_bwd)

   class DotOp:
     def __init__(self):
       self.amax_history = core.new_ref(jnp.zeros(5,))

     def forward(self, x, y):
       out = jnp.dot(x, y)
       out = gradient_history_calculator(out, self.amax_history)
       return out

   dot_op = DotOp()
   x_top = jnp.ones((5,))
   y_top = jnp.ones((5,))

   def loss(x, y):
     return dot_op.forward(x, y).sum()

   if jit:
     loss = jax.jit(loss)

   for i in range(3):
     jax.grad(loss, (0,1))(x_top, y_top)
     self.assertAllClose(dot_op.amax_history[:], jnp.zeros((5,)).at[:i+1].set(1.0), check_dtypes=False)

  @parameterized.parameters([False, True])
  def test_custom_vjp_grad_stats_plumbing_basic(self, jit):
    def primal(grads_ref, x):  # note: jit-abstracted!
      x = jnp.sin(x)
      x = stash_grads(grads_ref, x)
      x = jnp.sin(x)
      x = stash_grads(grads_ref, x)  # ignored, order-preserved
      return x

    if jit:
      primal = jax.jit(primal)

    @jax.custom_vjp
    def stash_grads(grads_ref, x):
      return x
    def stash_grads_fwd(grads_ref, x):
      return x, grads_ref
    def stash_grads_bwd(grads_ref, g):
      grads_ref[...] = g
      return None, g
    stash_grads.defvjp(stash_grads_fwd, stash_grads_bwd)

    grads_ref = core.new_ref(jnp.float32(0.))
    jax.grad(primal, 1)(grads_ref, jnp.float32(1.0))
    self.assertAllClose(grads_ref[...], jnp.cos(jnp.sin(1.)), check_dtypes=False)

  @parameterized.parameters(it.product([False, True], repeat=2))
  def test_custom_vjp_grad_stats_plumbing_scan(self, jit, remat):
    def primal(grads_ref, x):  # note: jit-abstracted!
      def body(x, _):
        x = jnp.sin(x)
        x = stash_grads(grads_ref, x)
        x = jnp.sin(x)
        return x, ()
      if remat:
        body = jax.remat(body)
      x, () = jax.lax.scan(body, x, None, length=1)
      return x

    if jit:
      primal = jax.jit(primal)

    @jax.custom_vjp
    def stash_grads(grads_ref, x):
      return x
    def stash_grads_fwd(grads_ref, x):
      return x, grads_ref
    def stash_grads_bwd(grads_ref, g):
      grads_ref[...] = g
      return None, g
    stash_grads.defvjp(stash_grads_fwd, stash_grads_bwd)

    grads_ref = core.new_ref(jnp.float32(0.))
    jax.grad(primal, argnums=1)(grads_ref, jnp.float32(1.0))
    self.assertAllClose(grads_ref[...], jnp.cos(jnp.sin(1.)), check_dtypes=False)

  @parameterized.product(jit=[False, True], has_aux=[False, True])
  def test_custom_vjp_grad_stats_plumbing_basic_vjp3(self, jit, has_aux):
    def primal(grads_ref, x):  # note: abstracts over jit and has_aux!
      x0 = x
      x = jnp.sin(x)
      x = stash_grads(grads_ref, x)
      x = jnp.sin(x)
      x = stash_grads(grads_ref, x)  # ignored, order-preserved
      return (x, x0) if has_aux else x

    if jit:
      primal = jax.jit(primal)

    @jax.custom_vjp
    def stash_grads(grads_ref, x):
      return x
    def stash_grads_fwd(grads_ref, x):
      return x, grads_ref
    def stash_grads_bwd(grads_ref, g):
      grads_ref[...] = g
      return None, g
    stash_grads.defvjp(stash_grads_fwd, stash_grads_bwd)

    grads_ref = core.new_ref(jnp.float32(0.))
    x = jnp.float32(1.)
    _, f_vjp, *maybe_aux = jax.vjp(
        lambda x: primal(grads_ref, x), x, has_aux=has_aux)
    _ = f_vjp(jnp.float32(1.))
    self.assertAllClose(grads_ref[...], jnp.cos(jnp.sin(1.)), check_dtypes=False)
    if has_aux:
      aux, = maybe_aux
      self.assertAllClose(aux, x)

  def test_custom_vjp_grad_stats_plumbing_scan_vjp3(self):
    def primal(stash_ref, x):  # note: jit-abstracted!
      def body(x, _):
        x = jnp.sin(x)
        x = stash_grads(stash_ref, x)
        x = jnp.sin(x)
        return x, ()
      x, () = jax.lax.scan(body, x, None, length=1)
      return x

    @jax.custom_vjp
    def stash_grads(stash_ref, x):
      return x
    def stash_grads_fwd(stash_ref, x):
      return x, stash_ref
    def stash_grads_bwd(stash_ref, g):
      stash_ref[...] = g
      return None, g
    stash_grads.defvjp(stash_grads_fwd, stash_grads_bwd)

    stash_ref = core.new_ref(jnp.float32(0.))
    _, f_vjp = jax.vjp(lambda x: primal(stash_ref, x), jnp.float32(1.))
    grads_val, = f_vjp(jnp.float32(1.))
    self.assertAllClose(stash_ref[...], jnp.cos(jnp.sin(1.)), check_dtypes=False)
    self.assertAllClose(grads_val, jnp.cos(jnp.sin(1.)) * jnp.cos(1.),
                        check_dtypes=False)

    stash_ref = core.new_ref(jnp.float32(0.))
    grads_ref = core.new_ref(jnp.float32(0.))
    _, f_vjp = jax.vjp(lambda x: primal(stash_ref, x), jnp.float32(1.))
    _ = f_vjp.with_refs(grads_ref)(jnp.float32(1.))
    self.assertAllClose(stash_ref[...], jnp.cos(jnp.sin(1.)), check_dtypes=False)
    self.assertAllClose(grads_ref[...], jnp.cos(jnp.sin(1.)) * jnp.cos(1.),
                        check_dtypes=False)

  @parameterized.parameters([False, True], [False, True])
  def test_freeze_insertion(self, inner_jit, outer_jit):
    def f(x):
      x_ref = core.new_ref(x)

      def g():
        x_ref[...] = x_ref[...] + x_ref[...]
      if inner_jit:
        g = jax.jit(g)
      g()

      return x_ref[...]

    if outer_jit:
      f = jax.jit(f)

    self.assertAllClose(jax.grad(f)(3.), 2., check_dtypes=False)

  @parameterized.parameters([False, True])
  def test_grad_jit(self, jit):
    def f(x):
      x_ref = core.new_ref(x)

      @jax.jit
      def g():
        x_ref[...] = jnp.sin(x_ref[...]) + jnp.sin(x_ref[...])
      g()

      return core.freeze(x_ref)

    if jit:
      f = jax.jit(f)

    ans = jax.grad(f)(2.)
    expected = 2. * jnp.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.parameters([False, True])
  def test_grad_scan(self, jit):
    def f(x):
      x_ref = core.new_ref(x)

      def g(_, __):
        x_ref[...] = jnp.sin(x_ref[...]) + jnp.sin(x_ref[...])
        return None, None
      jax.lax.scan(g, None, None, length=1)

      return core.freeze(x_ref)

    if jit:
      f = jax.jit(f)

    ans = jax.grad(f)(2.)
    expected = 2. * jnp.cos(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_grad_scan_extensive(self):
    def f(xs):
      xs_ref = core.new_ref(xs)

      def g(c, x_ref):
        return c + x_ref[...], None
      out, _ = jax.lax.scan(g, 0., xs_ref)

      return out

    ans = jax.grad(f)(jnp.arange(3.))
    expected = jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.parameters([False, True])
  def test_grad_jit_readonly(self, jit):
    def f(x):
      x_ref = core.new_ref(jnp.zeros_like(x))
      x_ref[...] = x
      return x_ref[...]

    if jit:
      f = jax.jit(f)

    jtu.check_grads(f, (1.5,), 2, ['fwd', 'rev'])

  def test_grad_jit_readonly_1(self):
    @jax.jit
    def f(x):
      x_ref = core.new_ref(x)

      def inner():
        return jnp.sin(x_ref[...])

      return inner()

    jtu.check_grads(f, (1.5,), 2, ['fwd', 'rev'])

  def test_grad_jit_readonly_2(self):
    def f(x):
      x_ref = core.new_ref(x)

      @jax.jit
      def inner():
        return jnp.sin(x_ref[...])

      return inner()

    jtu.check_grads(f, (1.5,), 2, ['fwd', 'rev'])

  @jtu.sample_product(
      seed=range(6),
      num_consts=range(2, 6),
      num_args=[0, 3],
  )
  @jtu.run_on_devices("cpu")
  def test_jit_vjp_systematic_readonly(self, seed, num_consts, num_args):
    num_mut_consts = num_consts // 2
    num_pure_consts = num_consts - num_mut_consts

    rng = np.random.RandomState(seed)
    pure_consts = [rng.normal() for _ in range(num_pure_consts)]
    mut_const_vals = [rng.normal() for _ in range(num_mut_consts)]

    args = [rng.normal() for _ in range(num_args)]

    mutable_bools = rng.permutation([True] * num_mut_consts +
                                    [False] * num_pure_consts)

    def f(mut_const_vals, pure_consts, args):
      consts = pure_consts[:], map(core.new_ref, mut_const_vals)

      @jax.jit
      def inner(args):
        tot = 0.
        for is_mut in mutable_bools:
          const = consts[int(is_mut)].pop()
          if is_mut: const = const[...]
          tot += jnp.sin(const)
        for x in args:
          tot += jnp.sin(x)
        return tot

      return inner(args)

    jtu.check_grads(f, (mut_const_vals, pure_consts, args), 2, ['rev'])

  @jtu.sample_product(
      seed=range(6),
      num_consts=range(2, 6),
      num_carry=[0, 3],
      num_ext_in=[0, 3],
      num_iters=[1, 3],
  )
  @jtu.run_on_devices("cpu")
  def test_scan_vjp_systematic_readonly(
      self, seed, num_consts, num_carry, num_ext_in, num_iters):
    num_mut_consts = num_consts // 2
    num_pure_consts = num_consts - num_mut_consts

    rng = np.random.RandomState(seed)
    pure_consts = [rng.normal() for _ in range(num_pure_consts)]
    mut_const_vals = [rng.normal() for _ in range(num_mut_consts)]

    init_carry = [rng.normal() for _ in range(num_carry)]
    xs = [rng.normal(size=num_iters) for _ in range(num_ext_in)]

    mutable_bools = rng.permutation([True] * num_mut_consts +
                                    [False] * num_pure_consts)

    def f(mut_const_vals, pure_consts, c, xs):
      consts = pure_consts[:], map(core.new_ref, mut_const_vals)

      def body(c, x):
        tot = 0.
        for is_mut in mutable_bools:
          const = consts[int(is_mut)].pop()
          if is_mut: const = const[...]
          tot += jnp.sin(const)
        new_c = [jnp.sin(carry) + tot for carry in c]
        y = sum(map(jnp.sin, x)) * 1.0
        return new_c, y

      return jax.lax.scan(body, init_carry, xs, length=num_iters)

    jtu.check_grads(f, (mut_const_vals, pure_consts, init_carry, xs),
                    2, ['fwd', 'rev'], rtol=1.5e-2)

  @parameterized.parameters([False, True])
  def test_remat_basic_internal(self, jit):
    @jax.remat
    def f(y, x):
      x_ref = jax.new_ref(x)
      out = y * x_ref[...]
      x_ref[...] += 1
      return out

    if jit:
      f = jax.jit(f)

    g = jax.grad(f)(2., 1.)
    self.assertAllClose(g, 1.)

  @parameterized.parameters([False, True])
  def test_remat_basic_arg(self, jit):
    @jax.remat
    def f(y, x_ref):
      out = y * y
      x_ref[...] += out
      return out

    if jit:
      f = jax.jit(f)

    x_ref = core.new_ref(1., kind='anselm_ref')
    g = jax.grad(f)(2., x_ref)
    self.assertAllClose(x_ref[...], 5.)
    self.assertAllClose(g, 4.)

  @parameterized.parameters([False, True])
  def test_remat_basic_closed_over(self, jit):
    @jax.remat
    def f(y):
      out = y * x_ref[...]
      x_ref[...] += 1
      return out

    if jit:
      f = jax.jit(f)

    x_ref = core.new_ref(1., kind='anselm_ref')
    g = jax.grad(f)(2.)
    self.assertAllClose(x_ref[...], 2.)
    self.assertAllClose(g, 1.)

  def test_remat_basic_closed_over_nested(self):
    @jax.remat
    @partial(jax.remat, policy=lambda *_, **__: False)
    @jax.remat
    def f(y):
      jax.debug.callback(lambda _: lst.append('hi'), y)
      out = y * x_ref[...]
      x_ref[...] += 1
      return jnp.sin(out)

    lst = []
    x_ref = core.new_ref(1., kind='anselm_ref')
    g = jax.grad(f)(2.)
    self.assertAllClose(x_ref[...], 2.)
    self.assertAllClose(g, jnp.cos(2.))
    self.assertLen(lst, 4)

  def test_remat_grad_stats_plumbing_basic(self):
    @jax.remat
    def f(x_ref, y):
      stash_grads(x_ref, y)
      return y

    @jax.custom_vjp
    def stash_grads(grads_ref, x):
      return x
    def stash_grads_fwd(grads_ref, x):
      return x, grads_ref
    def stash_grads_bwd(grads_ref, g):
      grads_ref[...] = g
      return None, g
    stash_grads.defvjp(stash_grads_fwd, stash_grads_bwd)

    x_ref = core.new_ref(0)
    jax.grad(f, 1)(x_ref, 3.14)

  @jtu.run_on_devices("cpu")  # tolerances, lol
  def test_vjp3_ref_grads_for_val_primals(self):
    NUM_LAYERS = 3
    NUM_MUBATCHES = 5
    MUBATCH_SIZE = 7

    def mubatch_loss(Ws, xs):
      # Inner loop: scan over layers
      act, _ = jax.lax.scan(lambda xs, W: (jnp.dot(xs, W), None), xs, Ws)
      return jnp.mean(act)

    def process_batch(Ws, xs_batch):
      grad_acc = jax.new_ref(jnp.zeros_like(Ws))               # CHANGED

      def process_mubatch(_, xs):
        loss, f_vjp = jax.vjp(lambda Ws: mubatch_loss(Ws, xs), Ws)  # CHANGED
        f_vjp.with_refs(grad_acc)(jnp.ones_like(loss))           # CHANGED
        return (), loss

      assert xs_batch.shape[0] == NUM_MUBATCHES * MUBATCH_SIZE
      xs_mubatches = xs_batch.reshape(NUM_MUBATCHES, MUBATCH_SIZE, *xs_batch.shape[1:])

      # Outer loop: scan over microbatches
      (), _losses = jax.lax.scan(process_mubatch, (), xs_mubatches)
      return jax.ref.freeze(grad_acc)

    Ws = jnp.ones((NUM_LAYERS, 4, 4))
    xs_batch = jnp.ones((NUM_MUBATCHES * MUBATCH_SIZE, 4))
    g = process_batch(Ws, xs_batch)
    self.assertAllClose(g, 20. * jnp.ones_like(Ws), atol=1e-3, rtol=1e-3)

  @parameterized.parameters([False, True])
  def test_custom_vjp_internal_ref(self, jit):
    @jax.custom_vjp
    def f(x):
      x_ref = jax.new_ref(jnp.zeros_like(x))
      x_ref[...] = x
      return x_ref[...]
    def f_fwd(x):
      return x, None
    def f_bwd(_, g):
      return g,
    f.defvjp(f_fwd, f_bwd)

    if jit:
      f = jax.jit(f)

    x = jax.jit(f)(3.)  # no ad, doesn't crash
    self.assertAllClose(x, 3., check_dtypes=False)

    g = jax.grad(f)(3.)
    self.assertAllClose(g, 1., check_dtypes=False)

  def test_custom_vjp_ad_after_discharge_error(self):
    @jax.custom_vjp
    def f(x):
      x_ref = jax.new_ref(jnp.zeros_like(x))
      x_ref[...] = x
      return x_ref[...]
    def f_fwd(x):
      return x, None
    def f_bwd(_, g):
      return g,

    f.defvjp(f_fwd, f_bwd)
    from jax._src import core
    from jax._src.state.discharge import discharge_state
    jaxpr = jax.make_jaxpr(f)(3.)
    jaxpr_, consts_ = discharge_state(jaxpr.jaxpr, jaxpr.consts)
    with self.assertRaises(Exception):
      jax.grad(lambda x: core.eval_jaxpr(jaxpr_, consts_, x)[0])(3.)

  @parameterized.parameters([False, True])
  def test_custom_vjp_differentiated_ref(self, jit):
    @jax.custom_vjp
    def f(x_ref):
      return x_ref[...]
    def f_fwd(x_ref):
      return f(x_ref), None
    def f_bwd(_, g):
      return g,
    f.defvjp(f_fwd, f_bwd)

    if jit:
      f = jax.jit(f)

    y = f(jax.new_ref(3.14))
    self.assertAllClose(y, 3.14, check_dtypes=False)

    # this exercises the fallback path, not a fancy transpose
    _, f_vjp = jax.vjp(lambda x: f(jax.new_ref(x)), 3.14)
    g, = f_vjp(1.)
    self.assertAllClose(g, 1., check_dtypes=False)

  def test_get_transpose_uninstantiated_grad_ref(self):
    # from https://github.com/jax-ml/jax/pull/31412#discussion_r2308151559
    f = lambda x: jax.new_ref(x)[0]
    jax.grad(f)(jnp.array([3.]))  # don't crash

  def test_vmap_create_ref_from_unbatched_value(self):
    @jax.jit
    def internally_pure(x):
      ref = jax.new_ref(1.)
      ref[...] += x
      return ref[...]

    ans = jax.vmap(internally_pure)(jnp.arange(4.))
    self.assertAllClose(ans, jnp.array([1., 2., 3., 4.]))

  def test_isinstance(self):
    ref = jax.new_ref(1.)
    self.assertIsInstance(ref, jax.Ref)

    @jax.jit
    def f(x_ref):
      self.assertIsInstance(x_ref, jax.Ref)
    f(ref)

    self.assertNotIsInstance(ref, jax.Array)

    arr = jnp.ones(3)
    self.assertNotIsInstance(arr, jax.Ref)

  def test_scan_vjp3_reverse(self):
    # https://github.com/jax-ml/jax/issues/32411
    def f(xs):
      ys = jnp.arange(5.)

      def body(_, xy):
        x, y = xy
        return (), x * y
      (), z = jax.lax.scan(body, (), (xs, ys))
      return z.sum()

    grad_accum = jax.new_ref(jnp.zeros(5))
    _, f_vjp = jax.vjp(f, jnp.ones(5))
    _, = f_vjp.with_refs(grad_accum)(1.)
    self.assertAllClose(grad_accum[...], jnp.arange(5.))

  def test_vmap_with_vjp3(self):
    # https://github.com/jax-ml/jax/issues/32479
    def grad_via_ref(f):
      def wrapper(*args):
        grad_accum = jax.tree.map(lambda x: jax.new_ref(jnp.zeros_like(x)), args)
        out, f_vjp = jax.vjp(f, *args)
        f_vjp.with_refs(*grad_accum)(jnp.ones_like(out))
        return jax.tree.map(lambda x: jax.freeze(x), grad_accum)
      return wrapper

    def issue_vmap1():
      def f(x):
        return x + 1
      x = jnp.ones((4,))
      # g = grad_via_ref(jax.vmap(f))  # good
      g = jax.vmap(grad_via_ref(f))    # bad
      g(x)  # crash

    def issue_vmap1_minimized():
      def f(x):
        x.addupdate(1.0)  # bad (assumes non-empty list of indexers)
      jax.vmap(f)(jax.new_ref(jnp.zeros((4,))))  # crash

    def issue_vmap2():
      def f(x):
        x[...] = 1.0  # bad (mismatched shapes)
      jax.vmap(f)(jax.new_ref(jnp.zeros((4,))))  # crash

    # don't crash
    issue_vmap1()
    issue_vmap1_minimized()
    issue_vmap2()

  def test_slicing_with_vjp3(self):
    @jax.jit
    def f(x, i):
      return x[i] ** 2

    x = jnp.arange(10.)

    grad_accum = jax.new_ref(jnp.zeros(10))
    not_needed = object()

    @jax.make_jaxpr
    def run():
      _, f_vjp = jax.vjp(f, x, 5)
      f_vjp = f_vjp.with_refs(grad_accum, not_needed)
      f_vjp(1.)

    jaxpr = run()
    self.assertIn('+=', str(jaxpr))
    self.assertNotIn('0.0', str(jaxpr))

  @absltest.skip("Not yet implemented")
  def test_none_index(self):
    ref = jax.new_ref(jnp.array([1, 2, 3]))
    y = ref[None]
    self.assertEqual(y.shape, (1, 3))

  def test_what_if_you_lower_fun_something_with_internal_effects(self):
    bjp_p = core.Primitive('bjp')

    @bjp_p.def_abstract_eval
    def _(aval):
      return aval

    def lowering(x):
      x_ref = jax.new_ref(x)
      x_ref[...] += 1
      x_ref[...] += -1
      return jax.freeze(x_ref)

    mlir.register_lowering(bjp_p, mlir.lower_fun(lowering, multiple_results=False))

    @jax.jit
    def f(x):
      return bjp_p.bind(x)

    f(3.)  # don't crash

  def test_remat_while_loop_residuals(self):
    @jax.custom_vjp
    def ra2a(x):
      return jax.freeze(jax.new_ref(x))

    def ra2a_fwd(x):
      o = ra2a(x)
      return o, ()

    def ra2a_bwd(res, g):
      return (ra2a(g),)

    ra2a.defvjp(ra2a_fwd, ra2a_bwd)

    @jax.jit
    @jax.remat
    def f(x):

      def g(x):
        def body(carry):
          i, x = carry
          x = ra2a(x)
          return i + 1, x
        return jax.lax.while_loop(lambda x: x[0] < 5, body, (0, x))[1]
      return g(x)

    jax.linearize(f, 5.)  # don't crash

  def test_empty_ref_basic(self):
    @jax.jit
    def f():
      ref = jax.empty_ref(jax.ShapeDtypeStruct((), 'float32'))
      ref[...] = 1.
      return ref[...]

    y = f()
    self.assertAllClose(y, jnp.ones((), 'float32'))

  def test_empty_ref_jvp(self):
    @jax.jit
    def f(x):
      ref = jax.empty_ref(jax.ShapeDtypeStruct((), 'float32'))
      ref[...] = 2. * x
      return ref[...]

    y, y_dot = jax.jvp(f, (3.,), (1.,))
    self.assertAllClose(y, 6., check_dtypes=False)
    self.assertAllClose(y_dot, 2., check_dtypes=False)

  def test_empty_ref_grad(self):
    @jax.jit
    def f(x):
      ref = jax.empty_ref(jax.ShapeDtypeStruct((), 'float32'))
      ref[...] = 2. * x
      return ref[...]

    y_bar = jax.grad(f)(3.)
    self.assertAllClose(y_bar, 2., check_dtypes=False)

  def test_free_ref_basic(self):
    @jax.jit
    def f(x):
      ref = jax.empty_ref(jax.ShapeDtypeStruct((), 'float32'))
      ref[...] = 2. * x
      result = ref[...]
      jax.free_ref(ref)
      return result
    self.assertArraysEqual(f(3.), 6., check_dtypes=False)

  def test_double_free_ref_raises(self):
    @jax.jit
    def f():
      ref = jax.empty_ref(jax.ShapeDtypeStruct((), 'float32'))
      jax.free_ref(ref)
      jax.free_ref(ref)  # double free, should raise

    with self.assertRaises(Exception):
      f()

  def test_deref_after_free_ref_raises(self):
    @jax.jit
    def f(x):
      ref = jax.new_ref(x)
      jax.free_ref(ref)
      return ref[...]  # deref after free, should raise
    with self.assertRaises(Exception):
      f(1.)

  def test_free_ref_jvp(self):
    @jax.jit
    def f(x):
      ref = jax.empty_ref(jax.ShapeDtypeStruct((), 'float32'))
      ref[...] = 2. * x
      result = ref[...]
      jax.free_ref(ref)
      return result

    y, y_dot = jax.jvp(f, (3.,), (1.,))
    self.assertArraysEqual(y, 6., check_dtypes=False)
    self.assertArraysEqual(y_dot, 2., check_dtypes=False)

  def test_free_ref_grad(self):
    @jax.jit
    def f(x):
      ref = jax.empty_ref(jax.ShapeDtypeStruct((), 'float32'))
      ref[...] = 2. * x
      result = ref[...]
      jax.free_ref(ref)
      return result

    y_bar = jax.grad(f)(3.)
    self.assertArraysEqual(y_bar, 2., check_dtypes=False)

  @jtu.sample_product([
    dict(shape=(3, 4), indexer=np.index_exp[1]),
    dict(shape=(3, 4), indexer=np.index_exp[1:4]),
    dict(shape=(2, 3, 4), indexer=np.index_exp[..., 0]),
    dict(shape=(2, 3, 4), indexer=np.index_exp[:, 1]),
    dict(shape=(3, 4), indexer=np.index_exp[np.arange(2), np.arange(2)]),
    dict(shape=(3, 4), indexer=np.index_exp[np.arange(2), ..., np.arange(2)]),
  ])
  def test_indexing_patterns(self, shape, indexer):
    x = self.rng().uniform(size=shape)
    x_ref = jax.new_ref(x)
    self.assertArraysAllClose(x[indexer], x_ref[indexer])


@jtu.with_config(jax_mutable_array_checks=True)
class MutableArrayErrorsTest(jtu.JaxTestCase):
  def test_return_from_jit(self):
    with self.assertRaisesRegex(
        ValueError,
        r"traced for jit returned a mutable array reference.*\n\n"
        r".*was created on line"):
      jax.jit(core.new_ref)(jnp.arange(3))

  def test_return_from_jit_arg(self):
    with self.assertRaisesRegex(
        ValueError,
        r"traced for jit returned a mutable array reference.*\n\n"
        r".*was passed in as the argument x_ref"):
      jax.jit(lambda x_ref: x_ref)(core.new_ref(jnp.arange(3)))

  @unittest.skip("regressed")  # TODO(mattjj): fix
  def test_return_from_jit_pytree(self):
    with self.assertRaisesRegex(
        ValueError,
        r"tree path result\['hi'\]"):
      jax.jit(lambda x_ref: {'hi': x_ref})(core.new_ref(jnp.arange(3)))

  @unittest.skip("regressed")  # TODO(mattjj): fix
  def test_return_from_jit_closure(self):
    with self.assertRaisesRegex(
        ValueError,
        r"tree path result\['hi'\]"):
      x_ref = core.new_ref(jnp.arange(3))
      jax.jit(lambda: {'hi': x_ref})()

  def test_argument_aliases_jit(self):
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(
        ValueError, "appeared at both x_ref and y_ref"):
      jax.jit(lambda x_ref, y_ref: x_ref[...] + y_ref[...])(x_ref, x_ref)

  def test_closure_and_argument_aliases_jit(self):
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(
        ValueError, "closed over and passed as the argument y_ref"):
      jax.jit(lambda y_ref: x_ref[...] + y_ref[...])(x_ref)

  def test_return_from_scan(self):
    with self.assertRaisesRegex(
        ValueError, "traced for scan returned a mutable array reference of type"):
      jax.lax.scan(lambda c, x: (core.new_ref(c), x), 0, jnp.arange(3))

  def test_argument_aliases_scan(self):
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(
        ValueError, r"appeared at both c\[0\] and c\[1\]"):
      jax.lax.scan(lambda c, _: (None, None), (x_ref, x_ref), None, length=1)

  def test_closure_and_argument_aliases_scan(self):
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(
        ValueError, r"closed over and passed as the argument y_ref"):
      jax.lax.scan(lambda y_ref, _: (x_ref[...] + y_ref[...], None), x_ref,
                   None, length=1)

  def test_return_from_cond(self):
    with self.assertRaisesRegex(
        ValueError, "traced for cond returned a mutable array reference of type"):
      jax.lax.cond(True, lambda: core.new_ref(1.0), lambda: core.new_ref(2.0))

  def test_argument_aliases_cond(self):
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex( ValueError, r"for cond.*at both x1 and x2"):
      jax.lax.cond(True, lambda x1, x2: ..., lambda x1, x2: ..., x_ref, x_ref)

  def test_closure_and_argument_aliases_cond(self):
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(
        ValueError, r"closed over and passed as the argument y_ref"):
      jax.lax.cond(True,
                   lambda y_ref: x_ref[...] + y_ref[...],
                   lambda y_ref: x_ref[...] + y_ref[...],
                   x_ref)

  @parameterized.parameters([False, True])
  def test_return_from_custom_vjp_primal(self, jit):
    @jax.custom_vjp
    def f(ref):
      return ref
    f.defvjp(lambda ref: ..., lambda *_: ...)
    if jit:
      f = jax.jit(f)
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(
        ValueError, "custom_vjp primal function"):
      f(x_ref)

  @parameterized.parameters([False, True])
  def test_return_from_custom_vjp_primal_nondiff_argnum(self, jit):
    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def f(_, ref):
      return ref
    f.defvjp(lambda _, ref: ..., lambda *_: ...)
    if jit:
      f = jax.jit(f, static_argnums=0)
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(
        ValueError, "custom_vjp primal function"):
      f('hi', x_ref)

  @parameterized.parameters([False, True])
  def test_return_from_custom_vjp_fwd(self, jit):
    @jax.custom_vjp
    def f(x, ref):
      return x
    f.defvjp(lambda x, ref: (x, ref), lambda ref, g: g)
    if jit:
      f = jax.jit(f)
    x_ref = core.new_ref(0.)

    jax.vjp(f, 3., x_ref)  # returning input ref, okay

    @jax.custom_vjp
    def g(x, ref):
      return x
    def g_fwd(x, _):
      y_ref = core.new_ref(0)
      return x, y_ref
    g.defvjp(g_fwd, lambda ref, g: g)
    if jit:
      g = jax.jit(g)
    x_ref = core.new_ref(0.)

    with self.assertRaisesRegex(
        ValueError, "custom_vjp fwd function"):
      jax.vjp(g, 3., x_ref)

  @parameterized.parameters([False, True])
  def test_argument_aliases_custom_vjp_primal(self, jit):
    @jax.custom_vjp
    def f(x_ref, y_ref):
      ...
    f.defvjp(lambda x_ref, y_ref: (None, None), lambda _, g: (None, None))
    if jit:
      f = jax.jit(f)
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(ValueError, "x_ref and y_ref"):
      f(x_ref, x_ref)

  @parameterized.parameters([False, True])
  def test_argument_aliases_custom_vjp_fwd(self, jit):
    @jax.custom_vjp
    def f(x_ref, y_ref):
      ...
    f.defvjp(lambda x_ref, y_ref: (None, None), lambda _, g: (None, None))
    if jit:
      f = jax.jit(f)
    x_ref = core.new_ref(0.)
    with self.assertRaisesRegex(ValueError, "x_ref and y_ref"):
      jax.vjp(f, x_ref, x_ref)

  # TODO(mattjj): add test test_closure_and_argument_aliases_custom_vjp

  @parameterized.parameters([False, True])
  def test_cond_both_branches_close_over_same_mutable_array(self, jit):
    # see also test_cond_with_ref_reuse in state_test.py
    x_ref = core.new_ref(0.)
    def f(pred):
      def true_fun():
        x_ref[()] = 1.
      def false_fun():
        x_ref[()] = 2.
      jax.lax.cond(pred, true_fun, false_fun)
    if jit:
      f = jax.jit(f)
    out_true = f(True)
    self.assertAllClose(x_ref[...], 1.)
    out_false = f(False)
    self.assertAllClose(x_ref[...], 2.)

  def test_vmap_closed_over_ref_write(self):
    x_ref = core.new_ref(jnp.zeros((), 'int32'))

    def f(val):
      x_ref[...] += val

    vals = jnp.arange(3, dtype='int32')
    with self.assertRaisesRegex(Exception, "unbatched array reference"):
      jax.vmap(f)(vals)

  def test_vmap_aliased_arguments(self):
    def f(ref_1, ref_2):
      pass

    x_ref = core.new_ref(jnp.zeros((3, 3)))
    with self.assertRaisesRegex(
        ValueError,
        "only one reference to a mutable array may be passed as an argument"):
      jax.vmap(f)(x_ref, x_ref)

  def test_jvp_closed_over_ref_error(self):
    ref = core.new_ref(0.)
    def f(x):
      ref[...] = x
      return x
    with self.assertRaisesRegex(
        Exception, "Move the array reference"):
      jax.jvp(f, [1.], [1.])


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
