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

"""Tests for `lax.fusion` values, jaxpr checks, transforms, and CPU lowering."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow.pipeline import PipelineParams
from jax._src.lax.fusion import fusion_p
from jax._src.lax.fusion import FusionParams
from jax._src.lib.mlir import ir
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class _NoAttributes(FusionParams):
  """Test `FusionParams` without a registered platform handler."""

  tag: str = ""


@dataclasses.dataclass(frozen=True)
class _PipelineParams(PipelineParams):
  """Test params implementing `PipelineParams` only."""


@dataclasses.dataclass
class _MutableParams(FusionParams):
  pass


def _lower(f, *args, platform):
  platforms = (platform,) if isinstance(platform, str) else platform
  return jax.jit(f).trace(*args).lower(lowering_platforms=platforms)


def _frontend_attributes(op: ir.Operation) -> dict[str, str]:
  raw = op.attributes.get("mhlo.frontend_attributes")
  if raw is None:
    return {}
  return {a.name: ir.StringAttr(a.attr).value for a in ir.DictAttr(raw)}


class _Op(NamedTuple):
  func: str  # Enclosing function name.
  name: str
  callee: str | None  # Callee symbol for calls.
  attrs: dict[str, str]


class _Module:
  """Ops of a lowered module, with queries by enclosing function."""

  def __init__(self, lowered):
    self.ops: list[_Op] = []
    module = lowered.compiler_ir("stablehlo")
    for func in module.body.operations:
      name = ir.StringAttr(func.operation.attributes["sym_name"]).value
      self._walk(func.operation, name)

  def _walk(self, op, func_name):
    for region in op.regions:
      for block in region.blocks:
        for inner in block.operations:
          inner = inner.operation
          callee = None
          if inner.name == "func.call":
            callee = ir.FlatSymbolRefAttr(inner.attributes["callee"]).value
          self.ops.append(
              _Op(func_name, inner.name, callee, _frontend_attributes(inner))
          )
          self._walk(inner, func_name)

  def __iter__(self):
    return iter(self.ops)

  def fusion_calls(self, func=None) -> list[_Op]:
    """Returns fusion `func.call` ops, restricted to `func` when given."""
    return [
        o
        for o in self.ops
        if o.name == "func.call"
        and o.callee.startswith("fusion")
        and func in (None, o.func)
    ]

  def helper_calls(self, func) -> list[_Op]:
    """Returns `func.call` ops in `func` targeting helper callees."""
    return [
        o
        for o in self.ops
        if o.func == func
        and o.name == "func.call"
        and not o.callee.startswith("fusion")
    ]


class FusionValueTest(jtu.JaxTestCase):

  def test_matches_body_on_nested_pytrees(self):

    def body(tree, scale, *, offset):
      a, (b, c) = tree["a"], tree["bc"]
      return {"sum": a + b * scale, "prod": (a * c, c - offset)}

    tree = {
        "a": jnp.arange(6.0).reshape(2, 3),
        "bc": (jnp.ones((2, 3)), jnp.full((2, 3), 2.0)),
    }
    expected = body(tree, 3.0, offset=1.5)
    got = lax.fusion(body)(tree, 3.0, offset=1.5)
    self.assertAllClose(got, expected)
    self.assertAllClose(
        jax.jit(lax.fusion(body))(tree, 3.0, offset=1.5), expected
    )

  def test_body_without_inputs_or_outputs(self):
    self.assertAllClose(lax.fusion(lambda: jnp.arange(3))(), jnp.arange(3))
    self.assertEqual(lax.fusion(lambda x: ())(1.0), ())

  def test_decorator_forms(self):
    params = _NoAttributes(tag="7")

    @lax.fusion
    def bare(x):
      return x + 1.0

    @lax.fusion(compiler_params=params)
    def with_params(x):
      return x * 2.0

    self.assertAllClose(jax.jit(bare)(3.0), 4.0)
    if jtu.test_device_matches(["cpu"]):
      self.assertAllClose(with_params(3.0), 6.0)
    self.assertEqual(with_params.__name__, "with_params")
    jaxpr = jax.make_jaxpr(with_params)(3.0)
    (eqn,) = jaxpr.eqns
    self.assertEqual(eqn.primitive, fusion_p)
    self.assertIs(eqn.params["compiler_params"], params)
    self.assertLen(
        _Module(_lower(with_params, 3.0, platform="cpu")).fusion_calls(), 1
    )
    with self.assertRaisesRegex(TypeError, "lax.fusion.*compiler_params"):
      lax.fusion(compiler_params={"vmem": 1})

  def test_hoisted_constant_on_each_platform(self):
    # Exceeds jax_embedded_constants_max_bytes, so lowering hoists it as an arg.
    table = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    g = lax.fusion(lambda x: x * table)
    f = lambda x: g(x) + g(x + 1.0)
    x = jnp.ones((64, 64), jnp.float32)
    self.assertAllClose(jax.jit(f)(x), x * table + (x + 1.0) * table)
    calls = _Module(_lower(f, x, platform="cpu")).fusion_calls()
    self.assertLen(calls, 2)
    # The default lowering shares one callee between equal bodies.
    self.assertEqual(calls[0].callee, calls[1].callee)
    other_calls = _Module(_lower(f, x, platform="tpu")).fusion_calls()
    self.assertLen(other_calls, 2)
    self.assertNotEqual(other_calls[0].callee, other_calls[1].callee)

  @parameterized.parameters("cpu", "tpu")
  def test_constants_stay_in_the_callee_without_hoisting(self, platform):
    # Export embeds constants inside the callee.
    table = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    f = jax.jit(lax.fusion(lambda x: x * table))
    exp = jax.export.export(f, platforms=(platform,))(
        jax.ShapeDtypeStruct((64, 64), jnp.float32)
    )
    module = ir.Module.parse(exp.mlir_module(), context=mlir.make_ir_context())
    (callee,) = [
        op.operation
        for op in module.body.operations
        if ir.StringAttr(op.operation.attributes["sym_name"]).value.startswith(
            "fusion"
        )
    ]
    self.assertLen(callee.regions[0].blocks[0].arguments, 1)

  def test_unused_fusion_is_removed_unless_effectful(self):
    log = []

    def f(x):
      lax.fusion(jnp.sin)(x)
      return x + 1.0

    def g(x):
      lax.fusion(lambda x: io_callback(log.append, None, x, ordered=True))(x)
      return x + 1.0

    jaxpr = jax.make_jaxpr(f)(1.0).jaxpr
    self.assertEqual([e.primitive for e in jaxpr.eqns].count(fusion_p), 1)
    dced, _ = pe.dce_jaxpr(jaxpr, [True])
    self.assertNotIn(fusion_p, [e.primitive for e in dced.eqns])
    jaxpr = jax.make_jaxpr(g)(1.0).jaxpr
    dced, _ = pe.dce_jaxpr(jaxpr, [True])
    self.assertIn(fusion_p, [e.primitive for e in dced.eqns])
    self.assertAllClose(jax.jit(g)(2.0), 3.0)
    self.assertEqual(log, [2.0])

  def test_ordered_effect_in_body(self):
    # Threads the ordered effect token through the private fusion callee.
    log = []

    def body(x):
      io_callback(log.append, None, x, ordered=True)
      return x * 2.0

    f = jax.jit(lambda x: lax.fusion(body)(x) + 1.0)
    self.assertAllClose(f(3.0), 7.0)
    self.assertAllClose(f(4.0), 9.0)
    self.assertEqual(log, [3.0, 4.0])

  @parameterized.named_parameters(
      (
          "dict",
          lambda: lax.fusion(jnp.sin, compiler_params={"vmem": 1}),
          "lax.fusion.*compiler_params",
      ),
      (
          "mutable",
          lambda: lax.fusion(jnp.sin, compiler_params=_MutableParams()),
          "frozen dataclass",
      ),
      ("not_callable", lambda: lax.fusion(3), "callable"),
      ("none_f", lambda: lax.fusion(None), "callable"),
  )
  def test_rejects_bad_arguments(self, make, msg):
    with self.assertRaisesRegex(TypeError, msg):
      make()

  def test_rejects_other_primitive_params(self):
    pipe_params = _PipelineParams()
    self.assertIsInstance(pipe_params, PipelineParams)
    self.assertNotIsInstance(pipe_params, FusionParams)
    with self.assertRaisesRegex(
        TypeError, "lax.fusion.*FusionParams.*_PipelineParams"
    ):
      lax.fusion(jnp.sin, compiler_params=pipe_params)

    @dataclasses.dataclass(frozen=True)
    class _BothParams(_NoAttributes, _PipelineParams):
      pass

    both = _BothParams(tag="7")
    self.assertIsInstance(both, FusionParams)
    self.assertIsInstance(both, PipelineParams)
    fused = lax.fusion(jnp.sin, compiler_params=both)
    ops = _Module(
        _lower(
            lambda init: lax.pipeline(
                lambda i, c: fused(c),
                grid=(2,),
                init=init,
                compiler_params=both,
            ),
            jnp.ones(4),
            platform="cpu",
        )
    )
    (call,) = ops.fusion_calls()
    self.assertEqual(call.attrs, {})

  def test_block_ops_inside_fusion_in_pipeline_body_raise(self):
    slice_step = lambda i, carry, src: carry + lax.fusion(
        lambda s: lax.block_slice(s, (4, 8), (i * 4, 0))
    )(src)
    update_step = lambda i, carry: lax.fusion(
        lambda c: lax.block_update(c, jnp.ones((4, 8), jnp.float32), (i * 4, 0))
    )(carry)
    src = jnp.zeros((16, 8), jnp.float32)
    init = jnp.zeros((4, 8), jnp.float32)
    for checks in (True, False):
      with config.enable_checks(checks):
        with self.assertRaisesRegex(
            TypeError, "lax.block_slice may only appear directly in the body"
        ):
          jax.make_jaxpr(
              lambda s: lax.pipeline(
                  lambda i, c: slice_step(i, c, s), grid=(4,), init=init
              )
          )(src)
        with self.assertRaisesRegex(
            TypeError, "lax.block_update may only appear directly in the body"
        ):
          jax.make_jaxpr(
              lambda x: lax.pipeline(update_step, grid=(4,), init=x)
          )(src)


class FusionJaxprTest(jtu.JaxTestCase):

  def test_make_jaxpr_prints_params_and_body(self):
    params = _NoAttributes(tag="7")
    jaxpr = jax.make_jaxpr(
        lax.fusion(lambda x: jnp.sin(x) + 1.0, compiler_params=params)
    )(1.0)
    text = str(jaxpr)
    self.assertIn("fusion[", text)
    self.assertIn(f"compiler_params={params!r}", text)
    self.assertIn("sin", text)
    (eqn,) = jaxpr.eqns
    self.assertEqual(eqn.primitive, fusion_p)
    self.assertIs(eqn.params["compiler_params"], params)

  def test_public_surface(self):
    self.assertEqual(lax.fusion.__module__, "jax.lax")
    self.assertFalse(hasattr(lax, "fusion_p"))

  def test_closed_over_tracers_and_constants(self):
    (eqn,) = jax.make_jaxpr(lambda x, y: lax.fusion(lambda x: x * y)(x))(
        1.0, 2.0
    ).eqns
    self.assertLen(eqn.invars, 2)
    self.assertEmpty(eqn.params["jaxpr"].consts)
    self.assertEqual(eqn.params["num_consts"], 1)

    table = np.arange(4.0, dtype=np.float32)
    f = lambda x, y: lax.fusion(lambda x: x * y + table)(x)
    (eqn,) = jax.make_jaxpr(f)(jnp.ones(4, jnp.float32), 2.0).eqns
    self.assertEqual([v.aval.shape for v in eqn.invars], [(), (4,)])
    self.assertEqual(eqn.params["num_consts"], 1)
    body = eqn.params["jaxpr"]
    self.assertLen(body.consts, 1)
    self.assertAllClose(body.consts[0], table)
    core.check_jaxpr(body)
    self.assertAllClose(jax.jit(f)(jnp.ones(4, jnp.float32), 2.0), 2.0 + table)

  def test_typecheck_accepts_valid_jaxpr(self):
    jaxpr = jax.make_jaxpr(lax.fusion(lambda x, y: (x * y, x + y)))(1.0, 2.0)
    core.check_jaxpr(jaxpr.jaxpr)

  @parameterized.named_parameters(
      (
          "operand_types",
          lambda e: e.replace(
              params=dict(
                  e.params, jaxpr=jax.make_jaxpr(lambda x: x + 1)(np.int32(1))
              )
          ),
          "fusion",
      ),
      (
          "mutable_params",
          lambda e: e.replace(
              params=dict(e.params, compiler_params=_MutableParams())
          ),
          "frozen dataclass",
      ),
      (
          "pipeline_params",
          lambda e: e.replace(
              params=dict(e.params, compiler_params=_PipelineParams())
          ),
          "fusion.*FusionParams.*_PipelineParams",
      ),
      (
          "operand_count",
          lambda e: e.replace(invars=[*e.invars, *e.invars]),
          "fusion passes operands",
      ),
      (
          "num_consts",
          lambda e: e.replace(params=dict(e.params, num_consts=2)),
          "num_consts=2",
      ),
      (
          "undefined_body_output",
          lambda e: e.replace(
              params=dict(
                  e.params,
                  jaxpr=e.params["jaxpr"].replace(
                      outvars=[core.Var(e.params["jaxpr"].out_avals[0])]
                  ),
              )
          ),
          "not defined",
      ),
  )
  def test_typecheck_rejects(self, change, msg):
    jaxpr = jax.make_jaxpr(lax.fusion(lambda x: x + 1.0))(1.0).jaxpr
    (eqn,) = jaxpr.eqns
    with self.assertRaisesRegex(core.JaxprTypeError, msg):
      core.check_jaxpr(jaxpr.replace(eqns=[change(eqn)]))

  def test_typecheck_rejects_duplicate_refs(self):

    def f(x):
      r1, r2 = jax.new_ref(x), jax.new_ref(x)
      return lax.fusion(lambda a, b: a[...] + b[...])(r1, r2)

    jaxpr = jax.make_jaxpr(f)(1.0).jaxpr
    (i,) = [i for i, e in enumerate(jaxpr.eqns) if e.primitive is fusion_p]
    eqn = jaxpr.eqns[i]
    eqns = list(jaxpr.eqns)
    eqns[i] = eqn.replace(invars=[eqn.invars[0], eqn.invars[0]])
    with self.assertRaisesRegex(core.JaxprTypeError, "same Ref"):
      core.check_jaxpr(jaxpr.replace(eqns=eqns))


class FusionTransformTest(jtu.JaxTestCase):

  def test_vmap_matches_batched_reference(self):

    def body(p, x):
      return {"y": p["w"] @ x + p["b"], "n": jnp.sum(x * x)}

    p = {"w": jnp.arange(12.0).reshape(3, 4), "b": jnp.arange(3.0)}
    xs = jnp.arange(20.0).reshape(5, 4)
    in_axes = ({"w": None, "b": None}, 0)
    expected = jax.vmap(body, in_axes)(p, xs)
    got = jax.vmap(lax.fusion(body), in_axes)(p, xs)
    self.assertAllClose(got, expected)
    got = jax.jit(jax.vmap(lax.fusion(body), in_axes))(p, xs)
    self.assertAllClose(got, expected)

  def test_vmap_keeps_params_and_primitive(self):
    params = _NoAttributes(tag="group")
    jaxpr = jax.make_jaxpr(
        jax.vmap(
            lax.fusion(lambda x, y: x * y, compiler_params=params),
            in_axes=(0, None),
        )
    )(jnp.arange(3.0), 2.0)
    core.check_jaxpr(jaxpr.jaxpr)
    (eqn,) = jaxpr.eqns
    self.assertEqual(eqn.primitive, fusion_p)
    self.assertIs(eqn.params["compiler_params"], params)
    self.assertEqual(eqn.outvars[0].aval.shape, (3,))

  def test_vmap_of_unbatched_output(self):
    xs = jnp.arange(4.0)
    got = jax.vmap(
        lax.fusion(lambda x, c: (x + 1.0, c * 2.0)),
        in_axes=(0, None),
        out_axes=(0, None),
    )(xs, 3.0)
    self.assertAllClose(got, (xs + 1.0, 6.0))

  def test_vmap_with_no_batched_operand(self):
    f = jax.vmap(
        lax.fusion(lambda x, y: x * y + 1.0), in_axes=(None, None), axis_size=3
    )
    self.assertAllClose(f(2.0, 3.0), jnp.full((3,), 7.0))

  def test_vmap_closing_over_batched_tracer_and_constant(self):
    table = np.arange(4.0, dtype=np.float32)

    def f(x, y):
      return lax.fusion(lambda x: x * y + table)(x)

    x = jnp.ones(4, jnp.float32)
    ys = jnp.arange(3.0, dtype=jnp.float32)
    expected = ys[:, None] * x[None, :] + table[None, :]
    self.assertAllClose(jax.vmap(f, in_axes=(None, 0))(x, ys), expected)
    self.assertAllClose(
        jax.jit(jax.vmap(f, in_axes=(None, 0)))(x, ys), expected
    )

  def test_grad_without_tangent_into_fusion(self):

    def f(w, x):
      features = lax.fusion(lambda x: jnp.tanh(x) * 2.0)(x)
      return jnp.sum(w * features)

    w, x = jnp.arange(3.0), jnp.ones(3)
    self.assertAllClose(jax.grad(f)(w, x), jnp.tanh(x) * 2.0)
    self.assertAllClose(jax.jit(jax.grad(f))(w, x), jnp.tanh(x) * 2.0)

  def test_grad_with_tangent_into_fusion_raises(self):
    f = lambda x: jnp.sum(lax.fusion(jnp.sin)(x))
    with self.assertRaisesRegex(
        NotImplementedError, "Differentiation rule for 'fusion' not implemented"
    ):
      jax.grad(f)(jnp.ones(3))
    with self.assertRaisesRegex(
        NotImplementedError, "Differentiation rule for 'fusion' not implemented"
    ):
      jax.jvp(lax.fusion(jnp.sin), (1.0,), (1.0,))

  def test_custom_vjp_around_fusion(self):
    @jax.custom_vjp
    def f(x):
      return lax.fusion(jnp.sin)(x)

    f.defvjp(
        lambda x: (f(x), x),
        lambda x, g: (lax.fusion(lambda x, g: jnp.cos(x) * g)(x, g),),
    )
    x = jnp.arange(3.0)
    self.assertAllClose(jax.grad(lambda x: f(x).sum())(x), jnp.cos(x))
    self.assertAllClose(jax.vmap(jax.grad(f))(x), jnp.cos(x))


class FusionLoweringTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ("default_none", None),
      ("unregistered_params", _NoAttributes(tag="ignored")),
  )
  def test_cpu_lowering_is_a_plain_call(self, params):
    f = lax.fusion(lambda x: jnp.sin(x) * 2.0, compiler_params=params)
    lowered = _lower(f, jnp.ones(4), platform="cpu")
    ops = _Module(lowered)
    (call,) = ops.fusion_calls()
    self.assertEqual(call.func, "main")
    self.assertEqual(call.attrs, {})
    self.assertTrue(all(not attrs for *_, attrs in ops), ops)
    text = lowered.as_text()
    self.assertNotIn("ignored", text)
    self.assertNotIn("MAXIMAL_FUSE", text)

  def test_multiplatform_default_rule_does_not_share_callees_across_branches(
      self,
  ):
    inner = lax.fusion(lambda x: x + 1.0)
    helper = jax.jit(lambda x: inner(x) * 2.0)
    outer = lax.fusion(lambda x: helper(x) - 3.0)
    for f in (lambda x: outer(x) + helper(x), lambda x: helper(x) + outer(x)):
      ops = _Module(_lower(f, jnp.ones(8), platform=("cpu", "tpu")))
      # The top level helper is lowered for both platforms and dispatches on
      # the platform; branch callees are lowered for one platform only.
      top_helpers = {c.callee for c in ops.helper_calls("main")}
      outer_calls = ops.fusion_calls("main")
      self.assertLen(outer_calls, 2)
      branch_helpers = set()
      for c in outer_calls:
        branch_helpers |= {h.callee for h in ops.helper_calls(c.callee)}
      self.assertLen(branch_helpers, 2)
      self.assertEmpty(top_helpers & branch_helpers)
      for name in branch_helpers:
        self.assertNotIn(
            "stablehlo.case", [o.name for o in ops if o.func == name]
        )
      (top,) = top_helpers
      self.assertIn("stablehlo.case", [o.name for o in ops if o.func == top])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
