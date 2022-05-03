# Copyright 2022 Google LLC
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
import jax.numpy as jnp
from jax import ad_checkpoint
from jax import core
from jax import lax
from jax import linear_util as lu
from jax.experimental import maps
from jax.experimental import pjit
from jax.config import config
from jax.interpreters import mlir
from jax._src import test_util as jtu
from jax._src import util
import numpy as np

config.parse_flags_with_absl()

effect_p = core.Primitive('effect')
effect_p.multiple_results = True

@effect_p.def_effectful_abstract_eval
def _(*, effect):
  return [], {effect}

mlir.lowerable_effects.add('foo')
mlir.lowerable_effects.add('foo2')
mlir.lowerable_effects.add('bar')
core.ordered_effects.add('foo')
core.ordered_effects.add('foo2')


def trivial_effect_lowering(ctx, *, effect):
  ctx.set_tokens_out(ctx.tokens_in)
  return []

def function_effect_lowering(ctx, *, effect):
  def _f(ctx):
    ctx.set_tokens_out(ctx.tokens_in)
    return []
  func = mlir._emit_lowering_rule_as_fun(_f, ctx)

  output_types = map(mlir.aval_to_ir_types, ctx.avals_out)
  token_types = [mlir.token_type() for _ in ctx.tokens_in.items()]
  output_types = [*token_types, *output_types]
  flat_output_types = util.flatten(output_types)
  call = mlir.func_dialect.CallOp(flat_output_types,
                                  mlir.ir.FlatSymbolRefAttr.get(func.name.value),
                                  mlir.flatten_lowering_ir_args(ctx.tokens_in.tokens()))
  tokens, out = util.split_list(call.results, [len(ctx.tokens_in)])
  ctx.set_tokens_out(mlir.TokenSet(zip(ctx.tokens_in.effects(), tokens)))
  return out


class JaxprEffectsTest(jtu.JaxTestCase):

  def test_trivial_jaxpr_has_no_effects(self):
    def f(x):
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual(core.no_effects, jaxpr.effects)

  def test_effectful_primitive_in_jaxpr_creates_effects(self):
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual({'foo'}, jaxpr.jaxpr.eqns[0].effects)
    self.assertEqual({'foo'}, jaxpr.effects)

  def test_different_effects_in_jaxpr(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual({'foo'}, jaxpr.jaxpr.eqns[0].effects)
    self.assertEqual({'bar'}, jaxpr.jaxpr.eqns[1].effects)
    self.assertEqual({'foo', 'bar'}, jaxpr.effects)

  def test_jaxpr_typecheck_should_verify_eqn_effects_are_subset(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.).jaxpr

    # Edit jaxpr to make its type wrong
    jaxpr = jaxpr.replace(effects={'foo'})

    with self.assertRaisesRegex(core.JaxprTypeError,
        'Equation effects are not subset of Jaxpr effects.'):
      core.check_jaxpr(jaxpr)

class HigherOrderPrimitiveTest(jtu.JaxTestCase):

  def test_core_call_primitive_inherits_effects(self):

    def f(x):
      @lu.wrap_init
      def f_(x):
        effect_p.bind(effect='foo')
        effect_p.bind(effect='bar')
        return [x]
      return core.call(f_, x)[0]
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

  def test_xla_call_primitive_inherits_effects(self):

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{flavor}", flavor=flavor)
    for flavor in ["old", "new"]))
  def test_remat_call_primitive_inherits_effects(self, flavor):
    remat = jax.remat if flavor == "old" else ad_checkpoint.checkpoint

    @remat
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

  def test_custom_jvp_primitive_inherits_effects(self):

    @jax.custom_jvp
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    f.defjvp(lambda x, t: (x, t))
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

  def test_custom_vjp_primitive_inherits_effects(self):

    @jax.custom_vjp
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    f.defvjp(
        fwd=lambda x: (x, ()),
        bwd=lambda _, g: g)
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

  def test_pmap_inherits_effects(self):

    @jax.pmap
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(jnp.arange(jax.local_device_count()))

  def test_xmap_inherits_effects(self):

    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    f = maps.xmap(f, in_axes=['a'], out_axes=['a'])
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(jnp.arange(jax.local_device_count()))

  def test_pjit_inherits_effects(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    f = pjit.pjit(f, in_axis_resources=pjit.PartitionSpec('x'),
        out_axis_resources=pjit.PartitionSpec('x'))
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      with maps.Mesh(np.array(jax.devices()), ['x']):
        jax.make_jaxpr(f)(jnp.arange(jax.local_device_count()))


class EffectfulJaxprLoweringTest(jtu.JaxTestCase):

  def test_should_pass_tokens_into_ordered_effect(self):

    def _effect_lowering(ctx, *, effect):
      self.assertListEqual(list(ctx.tokens_in.effects()), ['foo'])
      ctx.set_tokens_out(ctx.tokens_in)
      return []
    mlir.register_lowering(effect_p, _effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    f.lower(2.)

  def test_should_not_pass_tokens_into_unordered_effect(self):

    def effect_lowering(ctx, *, effect):
      self.assertEmpty(ctx.tokens_in)
      return []
    mlir.register_lowering(effect_p, effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='bar')
      return x + 1.
    f.lower(2.)

  def test_lowering_that_doesnt_set_tokens_should_cause_error(self):

    def bad_effect_lowering(ctx, *, effect):
      # Doesn't call `ctx.set_tokens_out`!
      return []
    mlir.register_lowering(effect_p, bad_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    with self.assertRaisesRegex(ValueError, 'Lowering rule for `effect` needs to '
        'set `tokens_out`'):
      f.lower(2.)

  def test_lowering_that_sets_wrong_tokens_should_cause_error(self):

    def bad_effect_lowering(ctx, *, effect):
      ctx.set_tokens_out(mlir.TokenSet(bar=ctx.tokens_in.get('foo')))
      return []
    mlir.register_lowering(effect_p, bad_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    with self.assertRaisesRegex(ValueError, 'Lowering rule for `effect` returns '
        'incorrect set of output token.'):
      f.lower(2.)

  def test_lowering_ordered_effect_should_create_tokens(self):

    def effect_lowering(ctx, *, effect):
      ctx.set_tokens_out(ctx.tokens_in)
      return []
    mlir.register_lowering(effect_p, effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    mhlo = f.lower(2.).compiler_ir()
    main = mhlo.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertEqual(first_op.operation.name, "mhlo.create_token")

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='foo2')
      return x + 1.
    mhlo = f.lower(2.).compiler_ir()
    main = mhlo.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertEqual(first_op.operation.name, "mhlo.create_token")
    second_op = main.body.blocks[0].operations[1]
    self.assertEqual(second_op.operation.name, "mhlo.create_token")

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    mhlo = f.lower(2.).compiler_ir()
    main = mhlo.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertEqual(first_op.operation.name, "mhlo.create_token")

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='foo2')
      return x + 1.
    mhlo = f.lower(2.).compiler_ir()
    main = mhlo.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertEqual(first_op.operation.name, "mhlo.create_token")
    second_op = main.body.blocks[0].operations[1]
    self.assertEqual(second_op.operation.name, "mhlo.create_token")

  def test_nontrivial_lowering_with_ordered_effect_should_consume_token(self):

    mlir.register_lowering(effect_p, function_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.

    mhlo = f.lower(2.).compiler_ir()
    main = mhlo.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertEqual(first_op.operation.name, "mhlo.create_token")
    second_op = main.body.blocks[0].operations[1]
    self.assertEqual(second_op.operation.name, "func.call")
    self.assertEqual(str(second_op.attributes["callee"]), "@effect")
    self.assertEqual(second_op.operands[0].owner, first_op)
    func = mhlo.body.operations[1]
    self.assertEqual(func.name.value, "effect")
    self.assertEqual(str(func.type.inputs[0]), "!mhlo.token")
    self.assertEqual(str(func.type.results[0]), "!mhlo.token")

  def test_nontrivial_lowering_with_unordered_effect_should_consume_token(self):

    mlir.register_lowering(effect_p, function_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='bar')
      return x + 1.

    mhlo = f.lower(2.).compiler_ir()
    main = mhlo.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertEqual(first_op.operation.name, "func.call")
    self.assertEqual(str(first_op.attributes["callee"]), "@effect")
    self.assertLen(list(first_op.operands), 0)
    func = mhlo.body.operations[1]
    self.assertEqual(func.name.value, "effect")
    self.assertLen(list(func.type.inputs), 0)
    self.assertLen(list(func.type.results), 0)


class ControlFlowEffectsTest(jtu.JaxTestCase):

  def test_effects_disallowed_in_cond(self):
    def f1(x):
      def true_fun(x):
        effect_p.bind(effect='foo')
        return x
      def false_fun(x):
        return x
      return lax.cond(True, true_fun, false_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f1)(2.)

    def f2(x):
      def true_fun(x):
        return x
      def false_fun(x):
        effect_p.bind(effect='foo')
        return x
      return lax.cond(True, true_fun, false_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f2)(2.)

  def test_effects_disallowed_in_while(self):
    def f1(x):
      def cond_fun(x):
        effect_p.bind(effect='foo')
        return False
      def body_fun(x):
        return x
      return lax.while_loop(cond_fun, body_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f1)(2.)

    def f2(x):
      def cond_fun(x):
        return False
      def body_fun(x):
        effect_p.bind(effect='foo')
        return x
      return lax.while_loop(cond_fun, body_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f2)(2.)

  def test_effects_disallowed_in_scan(self):

    def f(x):
      def body(carry, x):
        effect_p.bind(effect='foo')
        return carry, x
      return lax.scan(body, x, jnp.arange(4))

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
