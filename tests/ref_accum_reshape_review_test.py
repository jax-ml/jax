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

"""Repros from a review of https://github.com/jax-ml/jax/pull/41876.

Each test pushes a gradient Ref through a trivial reshape (lax.squeeze,
x[None], x[0]) that the PR now aliases as a RefAccum view, and then into a
consumer that handled the old ValAccum but doesn't handle the view. See
PR_41876_REVIEW.md at the repo root for the write-up.

Note that jnp.squeeze is an inlined jit, so under an eager jax.vjp it doesn't
alias, and these tests use lax.squeeze instead.
"""

from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import ad
from jax._src.state.types import AbstractRef
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


def grad_into_ref(f, x):
  ref = jax.new_ref(jnp.zeros_like(x))
  jax.vjp(f, x)[1].with_refs(ref)(1.)
  return jax.freeze(ref)


class RefAccumReshapeRegressionTest(jtu.JaxTestCase):

  # Regressions: each of these passes on main and fails at the PR head.

  @parameterized.parameters(['slice', 'gather', 'dynamic_slice'])
  def test_vmap_batched_grad_ref_through_squeeze(self, op):
    # Under vmap the gradient Ref is a batched tracer, and the view's
    # TransformedRef(ReshapeTransform) can't compute its type:
    #   TypeError: Cannot reshape VmapTracer(aval=Ref{float32[1,6]}, ...) to (6,)
    index = {'slice': lambda y: y[1:3].sum(),
             'gather': lambda y: y[jnp.array([1, 3])].sum(),
             'dynamic_slice': lambda y: lax.dynamic_slice_in_dim(y, 1, 2).sum(),
             }[op]
    f = lambda x: index(lax.squeeze(x, (0,)))
    xs = jnp.ones((3, 1, 6))
    self.assertAllClose(jax.vmap(partial(grad_into_ref, f))(xs),
                        jax.vmap(jax.grad(f))(xs))

  def test_vmapped_custom_vjp_with_accums_through_squeeze(self):
    # Batched custom_vjp rules with accumulators reject RefAccums. On main the
    # squeeze output was a ValAccum, so this worked; now it raises
    # batching.ref_accum_error ("under vmap, the custom_vjp function total has
    # a bwd rule that takes gradient accumulators, ...").
    @jax.custom_vjp
    def total(y):
      return y.sum()
    def total_bwd(_, g, y_acc):
      y_acc.accum(jnp.broadcast_to(g, y_acc.aval.shape))
    total.defvjp_with_accums(lambda y: (total(y), None), total_bwd)

    f = lambda x: jax.vmap(total)(lax.squeeze(x, (0,))).sum()
    x = jnp.ones((1, 3, 4))
    self.assertAllClose(grad_into_ref(f, x), jax.grad(f)(x))

  @jtu.with_explicit_mesh((2,), 'x')
  def test_shard_map_check_vma_false_reshaped_view(self, mesh):
    # _check_grad_refs gives the view the spec P(), which mentions no manual
    # axes, so with check_vma=False it always raises "Differentiating a
    # shard_map with check_vma=False into a gradient Ref ...".
    g = jax.shard_map(lambda y: y.sum(keepdims=True), in_specs=P(),
                      out_specs=P(), check_vma=False)
    f = lambda w: g(lax.squeeze(w, (0,))).sum()
    w = jnp.arange(4.).reshape(1, 4)
    self.assertAllClose(jax.jit(partial(grad_into_ref, f))(w), jax.grad(f)(w))

  @parameterized.named_parameters(('sharded', P('x')), ('replicated', P(None)))
  def test_shard_map_root_and_reshaped_view(self, root_spec):
    # The root records root_spec and the view records P(). PartitionSpecs
    # compare by identity, and P(None) is not P(), so this raises
    # "Differentiating a shard_map into one gradient Ref through arguments with
    # different in_specs" even when both are replicated.
    mesh = jtu.create_mesh((2,), 'x')
    g = jax.shard_map(lambda a, b: a * b.sum(), mesh=mesh,
                      in_specs=(root_spec, P()), out_specs=root_spec)
    f = lambda w: g(w, w[None]).sum()
    w = jnp.arange(4.) + 1.
    self.assertAllClose(jax.jit(partial(grad_into_ref, f))(w), jax.grad(f)(w))

  def test_scan_over_x_closing_over_reshaped_x(self):
    # The const x[None] is now a view sharing x's root Ref, so the scan
    # transpose's const/xs overlap check raises "Differentiating a scan with
    # respect to an array that is both a constant and a scanned-over input".
    def f(x):
      x_row = x[None]
      c, _ = lax.scan(lambda c, xi: (c + xi * x_row[0, 0], None), 0., x)
      return c
    x = jnp.arange(4.) + 1.
    self.assertAllClose(grad_into_ref(f, x), jax.grad(f)(x))

  # Not regressions, but worth a look.

  def test_custom_gradient_ref_accum_through_squeeze(self):
    # The argument is now a RefAccum view, so _apply_accums_closure_residuals
    # passes acc.inst().ref, a TransformedRef, to eval_jaxpr, which fails with
    #   TransformedRefAvalError: TransformedRefs cannot be abstractified.
    # On main this raised the documented TypeError ("declared a RefAccum ...
    # but its backward pass got a ValAccum"). This test expects the PR's
    # intended behavior, where the rule gets the view and the gradient lands
    # in the caller's Ref.
    @partial(jax.custom_gradient, with_accums=ad.RefAccum)
    def second(y):
      def rule(g, y_acc):
        y_acc.ref[1] += g
      return y[1], rule

    f = lambda x: second(lax.squeeze(x, (0,)))
    x = jnp.arange(4.).reshape(1, 4)
    self.assertAllClose(grad_into_ref(f, x), jnp.zeros_like(x).at[0, 1].set(1.))

  @parameterized.named_parameters(
      ('unspecified', None), ('P()', P()), ('P(None)', P(None)))
  def test_jit_in_shardings_reshaped_view_is_aliased(self, spec):
    # _can_keep_reshaped_ref_accum compares a NamedSharding on a concrete Mesh
    # with the root aval's sharding, which is on an AbstractMesh, so they never
    # match. With any explicit in_shardings the view falls back to a ValAccum,
    # and the transposed jit doesn't take the Ref. The unspecified case passes
    # and is here for contrast.
    mesh = jtu.create_mesh((1,), 'a')
    kw = {} if spec is None else dict(in_shardings=NamedSharding(mesh, spec))
    g = jax.jit(lambda y: y[1] * 3., **kw)
    f = lambda x: g(lax.squeeze(x, (0,)))
    x = jnp.ones((1, 4))

    def run(ref):
      jax.vjp(f, x)[1].with_refs(ref)(1.)

    jaxpr = jax.make_jaxpr(run)(jax.new_ref(jnp.zeros_like(x))).jaxpr
    bwd_jit = [e for e in jaxpr.eqns if e.primitive.name == 'jit'][-1]
    self.assertTrue(any(isinstance(v.aval, AbstractRef)
                        for v in bwd_jit.invars))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
