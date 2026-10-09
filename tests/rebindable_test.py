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

from functools import partial

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental.rebindable import (
    extract_rebindables, rebind, rebindable)

config.parse_flags_with_absl()
jtu.request_cpu_devices(2)


class RebindableTest(jtu.JaxTestCase):

  def test_rebindable_extract_and_lazy_rebind(self):
    traces = []

    @partial(rebindable, hyperparams=('bm', 'bn'))
    def tiled_dot(x, y, *, bm, bn):
      traces.append((bm, bn))
      return jnp.dot(x, y)

    x = jnp.ones((32, 64), jnp.float32)
    y = jnp.ones((64, 32), jnp.float32)
    model = lambda a, b: tiled_dot(
        tiled_dot(a, b, bm=16, bn=32), a, bm=16, bn=32)

    traced = jax.jit(model).trace(x, y)
    self.assertEqual(traces, [(16, 32)] * 2)  # two distinct input shapes
    traced.lower()
    self.assertLen(traces, 2)  # lowering reuses the bind-time trace
    sites = extract_rebindables(traced)
    self.assertEqual([t.name for t in sites], ['tiled_dot'] * 2)
    self.assertEqual(sites[0].rebindable.hyperparams, {'bm': 16, 'bn': 32})

    rebound = rebind(traced, lambda t: dict(bm=32, bn=64))
    self.assertLen(traces, 2)  # rebinding does not trace anything
    self.assertEqual(
        [s.rebindable.hyperparams for s in extract_rebindables(rebound)],
        [{'bm': 32, 'bn': 64}] * 2)
    out_orig = traced.lower().compile()(x, y)
    self.assertLen(traces, 2)  # lowering the original again hits the cache
    out_rebound = rebound.lower().compile()(x, y)
    self.assertEqual(traces[2:], [(32, 64)] * 2)  # one trace per rebound site
    self.assertAllClose(out_orig, out_rebound)
    with self.assertRaisesRegex(ValueError, "no hyperparameters"):
      sites[0].rebindable.rebind(bk=8)
    with self.assertRaisesRegex(TypeError, "missing hyperparameters"):
      tiled_dot(x, y, bm=16)
    self.assertIs(rebind(traced, lambda s: None), traced)

  def test_rebindable_declared_hyperparams_and_static_kwargs(self):
    traces = []

    @partial(rebindable, hyperparams=('bm',))
    def kernel(x, *, bm=8, scale, table):
      traces.append(bm)
      return x * scale + table['bias']

    x = jnp.ones(4, jnp.float32)
    # `scale` and the unhashable `table` dict are static config, not hp.
    f = lambda a: kernel(a, scale=3.0, table={'bias': 1.0})
    traced = jax.jit(f).trace(x)
    [site] = extract_rebindables(traced)
    self.assertEqual(site.rebindable.hyperparams, {'bm': 8})  # default recorded
    self.assertEqual(site.name, 'kernel')
    rebound = rebind(traced, lambda s: dict(bm=16))
    self.assertAllClose(rebound.lower().compile()(x), x * 3.0 + 1.0)
    self.assertEqual(traces, [8, 16])
    with self.assertRaisesRegex(ValueError, "no hyperparameters"):
      site.rebindable.rebind(scale=4.0)  # static kwargs are not rebindable
    with self.assertRaisesRegex(TypeError, "not hashable"):
      jax.jit(lambda a: kernel(a, bm=[8], scale=3., table={})).trace(x)
    with self.assertRaisesRegex(jax.errors.UnexpectedTracerError,
                                "explicit argument"):
      jax.jit(lambda a: kernel(a, scale=a, table={'bias': 1.0})).trace(x)

  def test_rebind_sites_by_key(self):
    @partial(rebindable, hyperparams='k')
    def scale(x, *, k):
      return x * k
    head = rebindable(lambda x, *, k: x * k, hyperparams='k', name='scale',
                   key='head')

    x = jnp.ones(4)
    traced = jax.jit(
        lambda x: (scale(x, k=2.), head(x, k=2.), head(3. * x, k=2.))
    ).trace(x)
    self.assertEqual([s.key for s in extract_rebindables(traced)],
                     [None, 'head', 'head'])
    rebound = rebind(traced, {'head': dict(k=5.)})
    self.assertEqual([s.rebindable.hyperparams['k']
                      for s in extract_rebindables(rebound)], [2., 5., 5.])
    self.assertAllClose(rebound.lower().compile()(x), (2 * x, 5 * x, 15 * x))
    self.assertIs(rebind(traced, {}), traced)
    self.assertIs(rebind(traced, {'tail': dict(k=5.)}), traced)

  def test_site_call_on_one_chip_and_on_a_replica_group(self):
    if jax.device_count() < 2:
      self.skipTest('requires 2 devices')
    @partial(rebindable, hyperparams='bm')
    def local(x, *, bm):  # does not communicate
      del bm
      return jnp.sin(x)

    @partial(rebindable, hyperparams='bm', key='summed')
    def summed(x, *, bm):  # communicates over 'data'
      del bm
      return x + jax.lax.psum(x, 'data')

    @partial(rebindable, hyperparams='bm')
    def accumulate(x, acc, *, bm):  # in-place into a per-shard Ref
      del bm
      acc[...] += x

    def body(x):
      acc = jax.new_ref(jnp.zeros_like(x))
      accumulate(x, acc, bm=8)
      return local(x, bm=8), summed(x, bm=8), acc[...]

    mesh = jax.make_mesh((2,), ('data',))
    model = jax.jit(jax.shard_map(
        body, mesh=mesh, in_specs=jax.P('data'), out_specs=jax.P('data')))
    h = jnp.arange(8.)
    x = jax.device_put(h, jax.NamedSharding(mesh, jax.P('data')))
    s_acc, s_local, s_summed = extract_rebindables(model.trace(x))
    self.assertEqual(s_local.abstract_mesh.manual_axes, ('data',))

    # Tuning harnesses run `site.call` on a submesh of their choice, checking
    # that each shard sees exactly the site's per-shard operand types.
    def on_submesh(site, n, spec=jax.P('data'), **hp):
      submesh = jax.make_mesh((n,), ('data',), devices=jax.devices()[:n],
                              axis_types=(jax.sharding.AxisType.Auto,))
      per_shard = lambda a: (a.shape, a.dtype, a.manual_axis_type.varying)

      def kernel(x, *rest):
        self.assertEqual([per_shard(jax.typeof(v)) for v in (x, *rest)],
                         [per_shard(a) for a in site.rebindable.in_avals_flat])
        return site.call(x, *rest, **hp)
      return jax.jit(jax.shard_map(kernel, mesh=submesh, in_specs=spec,
                                   out_specs=jax.P('data')))

    one_chip = h[:4]
    self.assertAllClose(on_submesh(s_local, 1)(one_chip), jnp.sin(one_chip))
    # It does not name the mesh axes, so it also runs under plain jit.
    self.assertAllClose(jax.jit(s_local.call)(one_chip), jnp.sin(one_chip))
    with self.assertRaises(AssertionError):  # wrong spec: shards see (8,)
      on_submesh(s_local, 1, spec=jax.P())(h)

    def acc_on_ones(x):
      acc = jax.new_ref(jnp.ones_like(x))
      s_acc.call(x, acc)
      return acc[...]
    chip = jax.make_mesh((1,), ('data',), devices=jax.devices()[:1],
                         axis_types=(jax.sharding.AxisType.Auto,))
    ones = jax.jit(jax.shard_map(acc_on_ones, mesh=chip, in_specs=jax.P('data'),
                                 out_specs=jax.P('data')))
    self.assertAllClose(ones(one_chip), one_chip + 1.)

    # summed communicates over 'data': tune it on a replica group.
    self.assertAllClose(on_submesh(s_summed, 2, bm=16)(h), model(x)[1])

    # Rebinding a shard_map site retraces it at lowering, in its context.
    traced = model.trace(x)
    rebound = rebind(traced, {'summed': dict(bm=16)})
    self.assertEqual([s.rebindable.hyperparams['bm']
                      for s in extract_rebindables(rebound)], [8, 8, 16])
    self.assertAllClose(rebound.lower().compile()(x)[1], model(x)[1])

  def test_site_call_with_ref_operand(self):
    @partial(rebindable, hyperparams='bm')
    def accumulate(x, acc, *, bm):
      del bm
      acc[...] += x
      return x * 2.

    def f(x, a):
      acc = jax.new_ref(a)
      return accumulate(x, acc, bm=8), acc[...]

    x, a = jnp.ones(4), jnp.arange(4.)
    [site] = extract_rebindables(jax.jit(f).trace(x, a))

    def run(x, a):
      acc = jax.new_ref(a)
      return site.call(x, acc, bm=16), acc[...]

    y, acc = jax.jit(run)(x, a)
    self.assertAllClose(y, 2 * x)
    self.assertAllClose(acc, a + x)
    with self.assertRaisesRegex(ValueError, "has no hyperparameters"):
      site.call(x, jax.new_ref(a), bk=4)

  def test_lowering_reuses_the_body_traced_at_bind_time(self):
    traces = []

    @partial(rebindable, hyperparams='k')
    def scale(x, *, k):
      traces.append(k)
      return x * k

    with jax.numpy_rank_promotion('warn'):  # part of the trace context
      traced = jax.jit(lambda x: scale(x, k=2.)).trace(jnp.ones(3))
    traced.lower()  # under a different trace context
    self.assertEqual(traces, [2.])

  def test_rebindable_rebind_cannot_change_types(self):
    @partial(rebindable, hyperparams='n')
    def f(x, *, n):
      return x[:n]
    traced = jax.jit(lambda x: f(x, n=2)).trace(jnp.ones(4))
    rebound = rebind(traced, lambda t: dict(n=3))
    with self.assertRaisesRegex(TypeError, "must not depend on"):
      rebound.lower()

  def test_rebindable_rejects_direct_ad(self):
    op = rebindable(lambda x, *, bm: x * 2.0, hyperparams='bm')
    x = jnp.ones((16,), jnp.float32)
    with self.assertRaises(NotImplementedError):
      jax.grad(lambda a: jnp.sum(op(a, bm=16)))(x)
    with self.assertRaises(NotImplementedError):
      jax.jvp(partial(op, bm=16), (x,), (x,))

  def test_rebindable_raises_under_grad_even_off_the_diff_path(self):
    scale = rebindable(lambda c, *, k: c * k, hyperparams='k')
    ones = jnp.ones(3)
    cases = {
        'non-differentiated arg': lambda a, c: jnp.sum(a * scale(c, k=2.0)),
        'constant input': lambda a, c: jnp.sum(a * scale(ones, k=2.0)),
        'output unused': lambda a, c: (scale(a, k=2.0), jnp.sum(a))[1],
    }
    for name, f in cases.items():
      for wrap in (lambda f: f, jax.jit):
        with self.subTest(name), self.assertRaisesRegex(
            NotImplementedError, "is not differentiable"):
          jax.grad(wrap(f))(ones, ones)

  def test_rebindable_vmap_stays_rebindable(self):
    @partial(rebindable, hyperparams='k', name='scale')
    def scale(x, *, k):
      return x * k
    xs = jnp.arange(6.).reshape(3, 2)
    traced = jax.jit(jax.vmap(lambda x: scale(x, k=2.))).trace(xs)
    rebound = rebind(traced, lambda s: dict(k=5.))
    self.assertAllClose(rebound.lower().compile()(xs), 5. * xs)

  def test_nested_rebindables_raise(self):
    @partial(rebindable, hyperparams='k', name='inner')
    def inner(x, *, k):
      return x * k

    @partial(rebindable, hyperparams='j', name='outer')
    def outer(x, *, j):
      return inner(x, k=j + 1.)

    with self.assertRaisesRegex(ValueError,
                                "inside the body of rebindable outer"):
      jax.jit(lambda x: outer(x, j=1.)).trace(jnp.ones(3))
    # The body stack is reset after the error.
    self.assertAllClose(jax.jit(lambda x: inner(x, k=2.))(jnp.ones(3)),
                        2. * jnp.ones(3))

  def test_rebindable_vmap_batches_only_dependent_outputs(self):
    @partial(rebindable, hyperparams='k')
    def f(x, c, *, k):
      return x * k, c + k

    xs, c = jnp.arange(6.).reshape(3, 2), jnp.ones(2)
    g = jax.vmap(lambda x: f(x, c, k=2.), out_axes=(0, None))
    traced = jax.jit(g).trace(xs)
    rebound = rebind(traced, lambda s: dict(k=5.))
    y, d = rebound.lower().compile()(xs)
    self.assertAllClose(y, 5. * xs)
    self.assertAllClose(d, c + 5.)

  def test_rebindable_vmap_with_refs_axis_names_and_in_axes(self):
    @partial(rebindable, hyperparams='k')
    def center(x, acc, *, k):  # collective over the vmapped axis + a Ref
      acc[...] += x
      return k * (x - jax.lax.pmean(x, 'i'))

    def f(x):
      acc = jax.new_ref(jnp.zeros_like(x))
      return center(x, acc, k=2.), acc[...]

    xs = jnp.arange(6.).reshape(2, 3)
    g = jax.jit(jax.vmap(f, in_axes=1, out_axes=1, axis_name='i'))
    traced = g.trace(xs)
    rebound = rebind(traced, lambda s: dict(k=3.))
    y, acc = rebound.lower().compile()(xs)
    self.assertAllClose(y, 3. * (xs - xs.mean(1, keepdims=True)))
    self.assertAllClose(acc, xs)

  def test_rebindable_nested_vmap_stages_the_fully_batched_call(self):
    traces = []

    @partial(rebindable, hyperparams='k')
    def f(x, *, k):
      traces.append(jax.typeof(x).shape)
      return x * k

    xs = jnp.arange(24.).reshape(2, 3, 4)
    traced = jax.jit(jax.vmap(jax.vmap(lambda x: f(x, k=2.)))).trace(xs)
    [site] = extract_rebindables(traced)  # intermediate calls not staged
    self.assertEqual([a.shape for a in site.rebindable.in_avals_flat],
                     [(2, 3, 4)])
    self.assertEqual([a.shape for a in site.rebindable.out_avals_flat],
                     [(2, 3, 4)])
    traces.clear()
    rebound = rebind(traced, lambda s: dict(k=5.))
    self.assertAllClose(rebound.lower().compile()(xs), 5. * xs)
    self.assertEqual(traces, [(4,)])  # fn itself sees one example

  def test_rebindable_vmap_over_mesh_axes(self):
    if jax.device_count() < 2:
      self.skipTest('requires 2 devices')

    @partial(rebindable, hyperparams='k')
    def f(x, *, k):
      return x * k

    x = jnp.arange(8.).reshape(2, 4)
    auto, explicit = jax.sharding.AxisType.Auto, jax.sharding.AxisType.Explicit
    for axis_type, kw in [(auto, {'spmd_axis_name': 'data'}), (explicit, {})]:
      mesh = jax.make_mesh((2,), ('data',), axis_types=(axis_type,))
      xs = jax.device_put(x, jax.NamedSharding(mesh, jax.P('data')))
      with self.subTest(str(axis_type)), jax.set_mesh(mesh):
        traced = jax.jit(jax.vmap(lambda x: f(x, k=2.), **kw)).trace(xs)
        [site] = extract_rebindables(traced)
        [aval] = site.rebindable.in_avals_flat
        self.assertEqual(aval.shape, (2, 4))
        if axis_type == explicit:
          self.assertEqual(aval.sharding.spec, jax.P('data', None))
        rebound = rebind(traced, lambda s: dict(k=5.))
        out = rebound.lower().compile()(xs)
        self.assertAllClose(out, 5. * x)
        self.assertEqual(out.sharding.spec[0], 'data')

  def test_rebindable_inside_custom_vjp_rules(self):
    @partial(rebindable, hyperparams='bm')
    def mm(a, b, *, bm):
      del bm
      return jnp.dot(a, b)

    @jax.custom_vjp
    def linear(x, w):
      return mm(x, w, bm=16)
    linear.defvjp(lambda x, w: (mm(x, w, bm=16), (x, w)),
                  lambda res, g: (mm(g, res[1].T, bm=16),
                                  mm(res[0].T, g, bm=16)))

    x = jnp.ones((8, 16), jnp.float32)
    w = jnp.ones((16, 4), jnp.float32)
    traced = jax.jit(jax.value_and_grad(
        lambda a, b: jnp.sum(linear(a, b)), argnums=(0, 1))).trace(x, w)
    self.assertLen(extract_rebindables(traced), 3)
    bms = iter([32, 64, 128])
    rebound = rebind(traced, lambda t: dict(bm=next(bms)))
    self.assertEqual([s.rebindable.hyperparams['bm']
                      for s in extract_rebindables(rebound)], [32, 64, 128])
    self.assertAllClose(traced.lower().compile()(x, w),
                        rebound.lower().compile()(x, w))

  @parameterized.parameters([False, True])
  def test_rebindable_under_remat(self, use_remat3):
    @partial(rebindable, hyperparams='bm')
    def mm(a, b, *, bm):
      del bm
      return jnp.dot(a, b)

    @jax.custom_vjp
    def block(x, w):
      return jnp.sin(mm(x, w, bm=16))
    def block_fwd(x, w):
      d = mm(x, w, bm=16)
      return jnp.sin(d), (x, w, d)
    def block_bwd(res, g):
      x, w, d = res
      dd = g * jnp.cos(d)
      return mm(dd, w.T, bm=16), mm(x.T, dd, bm=16)
    block.defvjp(block_fwd, block_bwd)

    x = jnp.ones((8, 16), jnp.float32)
    w = jnp.ones((16, 16), jnp.float32)
    with config.remat3(use_remat3):
      for policy in [None, jax.checkpoint_policies.everything_saveable]:
        f = jax.remat(block, policy=policy)
        traced = jax.jit(jax.grad(lambda a, b: jnp.sum(f(a, b)))).trace(x, w)
        # The forward kernel is recomputed unless the policy saves its output;
        # remat2 and remat3 agree here (the rebindables are in custom_vjp
        # rules).
        saved = policy is jax.checkpoint_policies.everything_saveable
        self.assertLen(extract_rebindables(traced), 3 if saved else 4)
        rebound = rebind(traced, lambda t: dict(bm=32))
        self.assertAllClose(traced.lower().compile()(x, w),
                            rebound.lower().compile()(x, w))

  @parameterized.parameters([False, True])
  def test_rebindable_under_scan_cond_remat_and_vmap(self, use_remat3):
    @partial(rebindable, hyperparams='k', name='scale')
    def scale(x, *, k):
      return x * k

    def body(c, _):
      c = jax.lax.cond(c.sum() > 0, lambda c: scale(c, k=2.), lambda c: c, c)
      return jax.checkpoint(lambda c: jnp.sin(scale(c, k=2.)))(c), None

    f = lambda x: jax.lax.scan(body, x, None, length=3)[0]

    def expected(x, k):
      for _ in range(3):
        x = k * x if x.sum() > 0 else x
        x = jnp.sin(k * x)
      return x

    x = jnp.ones(4)
    with config.remat3(use_remat3):
      traced = jax.jit(f).trace(x)
      sites = extract_rebindables(traced)
      self.assertEqual([s.name for s in sites], ['scale'] * 2)
      rebound = rebind(traced, lambda s: dict(k=3.))
      self.assertAllClose(rebound.lower().compile()(x), expected(x, 3.))

      xs = jnp.stack([x, -x])
      traced_v = jax.jit(jax.vmap(f)).trace(xs)
      sites_v = extract_rebindables(traced_v)
      self.assertGreaterEqual(len(sites_v), 2)
      self.assertTrue(all(s.name == 'scale' for s in sites_v))
      rebound_v = rebind(traced_v, lambda s: dict(k=3.))
      self.assertAllClose(rebound_v.lower().compile()(xs),
                          jnp.stack([expected(x, 3.), expected(-x, 3.)]))

  @jtu.with_explicit_mesh((2,), ('data',))
  def test_rebindable_rebind_under_shard_map(self, mesh):
    @partial(rebindable, hyperparams='bm')
    def shift(x, *, bm):
      del bm
      i = jax.lax.axis_index('data') + jax.lax.psum(1, 'data')
      return x + i.astype(x.dtype)

    f = jax.shard_map(lambda x: shift(x, bm=8), in_specs=jax.P('data'),
                      out_specs=jax.P('data'))
    x = jax.device_put(jnp.zeros(8), jax.P('data'))
    traced = jax.jit(f).trace(x)
    rebound = rebind(traced, lambda t: dict(bm=16))
    self.assertEqual(extract_rebindables(rebound)[0].rebindable.hyperparams,
                     {'bm': 16})
    self.assertAllClose(rebound.lower().compile()(x),
                        jnp.array([2.] * 4 + [3.] * 4))


@jtu.with_config(jax_custom_vjp3=True)
class RebindableCustomVJP3Test(RebindableTest):
  pass


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
