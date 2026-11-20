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

"""Tests for TPU lowering and numerics of lax.fusion."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lax.control_flow.pipeline import PipelineParams
from jax._src.lax.fusion import fusion_p
from jax._src.lax.fusion import FusionParams
from jax._src.lib.mlir import ir
from jax._src.tpu import frontend_attrs as tpu_attrs
from jax._src.tpu import fusion as tpu_fusion
from jax._src.tpu import pipeline as tpu_pipeline
from jax._src.tpu.fusion import _combine
from jax._src.tpu.fusion import _Directive
from jax.experimental import layout
from jax.experimental.scheduling_groups import scheduling_group
from jax.experimental.xla_metadata import set_xla_metadata
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()

# Representative TPU fusion frontend attribute keys.
_NORI_CALL_KEYS = (
    ("FUSE_LIMIT", "5"),
    ("MUST_FUSE", "88"),
    ("_xla_tpu_native_bf16", "true"),
    ("buffering_level", "3"),
    ("input_window_bounds", "2,2"),
    ("integer", "64"),
    ("kernel_window_bounds", "4,8"),
    ("output_window_bounds", "4,1"),
    ("scoped_vmem_limit_kib", "7168"),
)


class _NoAttributes(FusionParams):
  """Default `FusionParams` base returning empty attribute dicts."""

  def fusion_call_attributes(
      self, ctx: tpu_fusion.FusionContext
  ) -> Mapping[str, str]:
    del ctx
    return {}

  def fusion_body_attributes(
      self, ctx: tpu_fusion.FusionContext
  ) -> Mapping[str, str]:
    del ctx
    return {}


class _MethodFusionHandler:
  """TPU fusion handler forwarding to methods on `_NoAttributes` subclasses."""

  def fusion_call_attributes(
      self, params: Any, ctx: tpu_fusion.FusionContext
  ) -> Mapping[str, str]:
    return params.fusion_call_attributes(ctx)

  def fusion_body_attributes(
      self, params: Any, ctx: tpu_fusion.FusionContext
  ) -> Mapping[str, str]:
    return params.fusion_body_attributes(ctx)

  def verify_fusion_body(
      self, params: Any, ctx: tpu_fusion.FusionContext, jaxpr: core.Jaxpr
  ) -> None:
    if (verify := getattr(params, "verify_fusion_body", None)) is not None:
      verify(ctx, jaxpr)

  def verify_fusion_operands(
      self, params: Any, ctx: tpu_fusion.FusionContext, operands
  ) -> None:
    if (verify := getattr(params, "verify_fusion_operands", None)) is not None:
      verify(ctx, operands)


tpu_fusion.register_fusion_handler(_NoAttributes, _MethodFusionHandler())


@dataclasses.dataclass(frozen=True)
class _TpuParams(_NoAttributes):
  """Test `FusionParams` with fixed call attributes and optional body group."""

  call: tuple[tuple[str, str], ...] = ()
  body: tuple[tuple[str, str], ...] = ()
  group: bool = False
  must: bool = False

  def fusion_call_attributes(
      self, ctx: tpu_fusion.FusionContext
  ) -> Mapping[str, str]:
    del ctx
    return dict(self.call)

  def fusion_body_attributes(
      self, ctx: tpu_fusion.FusionContext
  ) -> Mapping[str, str]:
    attrs = dict(self.body)
    if self.must:
      attrs["MUST_FUSE"] = str(ctx.fusion_id)
    elif self.group:
      attrs["MAXIMAL_FUSE"] = str(ctx.fusion_id)
    return attrs


_GROUP = _TpuParams(group=True)
_MUST = _TpuParams(must=True)


def _call_params(**call: str) -> _TpuParams:
  """Returns `_TpuParams` with `call` as its call attributes."""
  return _TpuParams(call=tuple(call.items()))


def _group_params(spec: str | None) -> _TpuParams:
  """Returns `_TpuParams` with `spec` (`KEY:id`) on its body ops, or none."""
  if spec is None:
    return _TpuParams()
  key, group = spec.split(":")
  return _TpuParams(body=((key, group),))


@dataclasses.dataclass(frozen=True)
class _PipelineParams(PipelineParams):
  """Test params implementing `PipelineParams` only."""

  loop: tuple[tuple[str, str], ...] = ()
  body: tuple[tuple[str, str], ...] = ()
  read: tuple[tuple[str, str], ...] = ()
  write: tuple[tuple[str, str], ...] = ()


class _PipelineParamsHandler:
  """TPU pipeline handler for `_PipelineParams`."""

  def pipeline_loop_attributes(self, params, facts) -> Mapping[str, str]:
    del facts
    return dict(params.loop)

  def pipeline_body_attributes(self, params, facts) -> Mapping[str, str]:
    del facts
    return dict(params.body)

  def block_read_attributes(self, params, facts) -> Mapping[str, str]:
    del facts
    return dict(params.read)

  def block_write_attributes(self, params, facts) -> Mapping[str, str]:
    del facts
    return dict(params.write)


tpu_pipeline.register_pipeline_handler(
    _PipelineParams, _PipelineParamsHandler()
)


@dataclasses.dataclass(frozen=True)
class _CheckedParams(_NoAttributes):
  seen: list[Any] = dataclasses.field(
      default_factory=list, hash=False, compare=False
  )

  def verify_fusion_body(
      self, ctx: tpu_fusion.FusionContext, jaxpr: core.Jaxpr
  ) -> None:
    self.seen.append((ctx, jaxpr))
    if any(e.primitive.name == "cos" for e in jaxpr.eqns):
      raise ValueError("cos is not allowed in this fusion")


@dataclasses.dataclass(frozen=True)
class _OperandParams(_NoAttributes):
  seen: list[Any] = dataclasses.field(
      default_factory=list, hash=False, compare=False
  )

  def verify_fusion_operands(
      self, ctx: tpu_fusion.FusionContext, operands
  ) -> None:
    self.seen.append((ctx, [str(o.type) for o in operands]))
    if any("i32" in str(o.type) for o in operands):
      raise ValueError("int32 operands are not allowed in this fusion")


# Sentinels for expected group ids in nested fusion test tables.
_OUTER = object()
_OWN = object()


def _lower(f, *args, platform):
  platforms = (platform,) if isinstance(platform, str) else platform
  return jax.jit(f).trace(*args).lower(lowering_platforms=platforms)


class _Op(NamedTuple):
  func: str  # Enclosing function name.
  name: str
  callee: str | None  # Callee symbol for calls.
  attrs: dict[str, str]


_SKIPPED = ("func.return", "stablehlo.return", "stablehlo.constant")
_GROUP_KEYS = ("MUST_FUSE", "MAXIMAL_FUSE")


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
              _Op(
                  func_name,
                  inner.name,
                  callee,
                  tpu_attrs.frontend_attributes(inner),
              )
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

  def body(self, func) -> list[_Op]:
    """Returns the body ops of `func`, skipping returns and constants."""
    return [o for o in self.ops if o.func == func and o.name not in _SKIPPED]

  def only(self, func, name) -> _Op:
    """Returns the single op named `name` in `func`."""
    (op,) = [o for o in self.body(func) if o.name == name]
    return op

  def group(self, func, key) -> str:
    """Returns the shared value of `key` across fusible ops in `func`."""
    (group,) = {o.attrs[key] for o in self.body(func) if o.name != "func.call"}
    return group

  def helper_calls(self, func) -> list[_Op]:
    """Returns `func.call` ops in `func` targeting helper callees."""
    return [
        o
        for o in self.body(func)
        if o.name == "func.call" and not o.callee.startswith("fusion")
    ]

  def helper_ops(self, func) -> list[_Op]:
    """Returns body ops of helper callees transitively called from `func`."""
    out, seen, todo = [], set(), [c.callee for c in self.helper_calls(func)]
    while todo:
      name = todo.pop()
      if name not in seen:
        seen.add(name)
        out += self.body(name)
        todo += [c.callee for c in self.helper_calls(name)]
    return out


def _tpu_module(test, f, *args) -> _Module:
  """Lowers `f` for TPU and checks call and single group key invariants."""
  # Outlined `lax.fusion` calls keep group keys off `func.call`, `MUST_FUSE`
  # never stays on any `func.call`, and no op carries both group keys.
  m = _Module(_lower(f, *args, platform="tpu"))
  for o in m:
    test.assertFalse(_GROUP_KEYS[0] in o.attrs and _GROUP_KEYS[1] in o.attrs, o)
    if o.name == "func.call":
      test.assertNotIn("MUST_FUSE", o.attrs, o)
      if o.callee.startswith("fusion"):
        test.assertNotIn("MAXIMAL_FUSE", o.attrs, o)
  return m


def _uses_helpers(x):
  # `jnp.where` and `jax.nn.silu` lower to calls of shared jitted callees.
  return jnp.where(x > 0, x, 0.0) * jax.nn.silu(x)


class FusionLoweringTest(jtu.JaxTestCase):

  def test_unregistered_fusion_params_succeed_on_cpu_and_fail_on_tpu(self):
    # Registered TPU params are also ignored on CPU.
    tpu_f = lax.fusion(
        lambda x: jnp.sin(x) * 2.0,
        compiler_params=_TpuParams(call=_NORI_CALL_KEYS, group=True),
    )
    cpu_lowered = _lower(tpu_f, jnp.ones(4), platform="cpu")
    cpu_ops = _Module(cpu_lowered)
    self.assertTrue(all(not attrs for *_, attrs in cpu_ops), cpu_ops)
    cpu_text = cpu_lowered.as_text()
    for key, _ in _NORI_CALL_KEYS:
      self.assertNotIn(key, cpu_text)
    self.assertNotIn("MAXIMAL_FUSE", cpu_text)

    @dataclasses.dataclass(frozen=True)
    class UnregisteredParams(FusionParams):
      tag: str = "x"

    class Handler:

      def fusion_call_attributes(self, params, ctx) -> Mapping[str, str]:
        del ctx
        return {"tag": params.tag}

      def fusion_body_attributes(self, params, ctx) -> Mapping[str, str]:
        del params, ctx
        return {}

    f = lax.fusion(lambda x: x + 1.0, compiler_params=UnregisteredParams())
    (cpu_call,) = _Module(_lower(f, jnp.ones(4), platform="cpu")).fusion_calls()
    self.assertEqual(cpu_call.attrs, {})
    with self.assertRaisesRegex(NotImplementedError, "UnregisteredParams"):
      _lower(f, jnp.ones(4), platform="tpu")
    tpu_fusion.register_fusion_handler(UnregisteredParams, Handler())
    self.addCleanup(tpu_fusion.unregister_fusion_handler, UnregisteredParams)
    (tpu_call,) = _tpu_module(self, f, jnp.ones(4)).fusion_calls()
    self.assertEqual(tpu_call.attrs, {"tag": "x"})
    tpu_fusion.unregister_fusion_handler(UnregisteredParams)
    with self.assertRaisesRegex(NotImplementedError, "UnregisteredParams"):
      _lower(f, jnp.ones(4), platform="tpu")

  def test_tpu_call_attributes(self):
    params = _TpuParams(call=_NORI_CALL_KEYS)
    f = lax.fusion(lambda x: jnp.sin(x) * 2.0, compiler_params=params)
    ops = _tpu_module(self, f, jnp.ones(4))
    (call,) = ops.fusion_calls()
    # `MUST_FUSE` moves to body ops; remaining keys stay on `func.call`.
    self.assertEqual(
        call.attrs, {k: v for k, v in _NORI_CALL_KEYS if k != "MUST_FUSE"}
    )
    body = ops.body(call.callee)
    self.assertNotEmpty(body)
    for op in body:
      self.assertEqual(op.attrs, {"MUST_FUSE": "88"}, op)

  @parameterized.named_parameters(
      ("body_and_helpers", "body_and_helpers"),
      ("distinct_ids", "distinct_ids"),
      ("nested_with_must_fuse", "nested_with_must_fuse"),
  )
  def test_tpu_default_maximal_fuse(self, case):
    x = jnp.ones((8, 128))
    if case == "body_and_helpers":
      ops = _tpu_module(self, lax.fusion(lambda a: _uses_helpers(a) + 1.0), x)
      (call,) = ops.fusion_calls("main")
      self.assertEqual(call.attrs, {})
      tag = ops.group(call.callee, "MAXIMAL_FUSE")
      for o in ops.body(call.callee):
        self.assertEqual(o.attrs, {"MAXIMAL_FUSE": tag}, o)
      self.assertNotEmpty(ops.helper_calls(call.callee))
      for o in ops.helper_ops(call.callee):
        self.assertEqual(o.attrs, {"MAXIMAL_FUSE": tag}, o)
    elif case == "distinct_ids":
      g = lax.fusion(lambda a: jnp.sin(a) * 2.0 + 1.0)
      h = lax.fusion(lambda a: jnp.cos(a) - 1.0, compiler_params=None)
      ops = _tpu_module(self, lambda a: h(g(g(a))), x)
      calls = ops.fusion_calls("main")
      self.assertLen(calls, 3)
      self.assertLen({c.callee for c in calls}, 3)
      for c in calls:
        self.assertEqual(c.attrs, {})
        for o in ops.body(c.callee):
          self.assertEqual(list(o.attrs), ["MAXIMAL_FUSE"], o)
      ids = [ops.group(c.callee, "MAXIMAL_FUSE") for c in calls]
      self.assertLen(set(ids), 3)
    else:
      default_f = lax.fusion(lambda a: a + 1.0)
      must_outer = lax.fusion(
          lambda a: default_f(a) * 2.0, compiler_params=_MUST
      )
      must_inner = lax.fusion(lambda a: a - 1.0, compiler_params=_MUST)
      default_outer = lax.fusion(lambda a: must_inner(a) * 3.0)
      ops = _tpu_module(
          self, lambda a: (default_f(a), must_outer(a), default_outer(a)), x
      )
      plain_call, must_call, def_outer_call = ops.fusion_calls("main")
      plain_tag = ops.group(plain_call.callee, "MAXIMAL_FUSE")
      must_tag = ops.group(must_call.callee, "MUST_FUSE")
      (nested_def,) = ops.fusion_calls(must_call.callee)
      self.assertEqual(nested_def.attrs, {})
      self.assertEqual(
          ops.only(nested_def.callee, "stablehlo.add").attrs,
          {"MUST_FUSE": must_tag},
      )
      def_outer_tag = ops.group(def_outer_call.callee, "MAXIMAL_FUSE")
      (nested_must,) = ops.fusion_calls(def_outer_call.callee)
      inner_must_tag = ops.group(nested_must.callee, "MUST_FUSE")
      self.assertLen({plain_tag, must_tag, def_outer_tag, inner_must_tag}, 4)

  def test_tpu_group_ids_are_deterministic(self):
    params = _GROUP
    g = lax.fusion(lambda x: x * 3.0, compiler_params=params)
    f = lambda x: g(x) + g(x + 1.0)
    first = _tpu_module(self, f, jnp.ones(4))
    second = _tpu_module(self, f, jnp.ones(4))
    self.assertEqual(first.ops, second.ops)

  @parameterized.named_parameters(
      ("maximal_fuse_while", False),
      ("must_fuse_while_and_cond", True),
  )
  def test_tpu_body_tags_reach_control_flow_regions(self, with_cond_and_must):
    if not with_cond_and_must:
      loop_body = lambda x: lax.fori_loop(0, 3, lambda i, c: c * 2.0 + 1.0, x)
      ops = _tpu_module(
          self, lax.fusion(loop_body, compiler_params=_GROUP), jnp.ones(4)
      )
      (call,) = ops.fusion_calls()
      body_ops = ops.body(call.callee)
      tag = {
          "MAXIMAL_FUSE": (
              ops.only(call.callee, "stablehlo.while").attrs["MAXIMAL_FUSE"]
          )
      }
      for name in (
          "stablehlo.while",
          "stablehlo.compare",
          "stablehlo.multiply",
          "stablehlo.add",
      ):
        found = [o for o in body_ops if o.name == name]
        self.assertNotEmpty(found, name)
        for o in found:
          self.assertEqual(o.attrs, tag, o)
      return

    inner = lax.fusion(lambda x: x + 1.0, compiler_params=_MUST)

    def body(x):
      x = lax.fori_loop(0, 3, lambda i, c: inner(c) * 2.0, x)
      return lax.cond(x[0, 0] > 0, lambda z: z - 1.0, lambda z: z + 1.0, x)

    outer = lax.fusion(body, compiler_params=_call_params(MUST_FUSE="5"))
    ops = _tpu_module(self, outer, jnp.ones((8, 128)))
    (outer_call,) = ops.fusion_calls("main")
    self.assertEqual(outer_call.attrs, {})
    body_ops = ops.body(outer_call.callee)
    names = {o.name for o in body_ops}
    self.assertIn("stablehlo.while", names)
    self.assertTrue({"stablehlo.case", "stablehlo.if"} & names, names)
    for o in body_ops:
      if o.name != "func.call":
        self.assertEqual(o.attrs.get("MUST_FUSE"), "5", o)
    (inner_call,) = ops.fusion_calls(outer_call.callee)
    for o in ops.body(inner_call.callee):
      self.assertEqual(o.attrs.get("MUST_FUSE"), "5", o)

  def test_tpu_body_tags_skip_directives_and_reduction_computations(self):
    params = _GROUP

    def body(x):
      y = jnp.sum(x * x, axis=0)
      return layout.with_layout_constraint(y, layout.Layout((0,)))

    ops = _tpu_module(
        self, lax.fusion(body, compiler_params=params), jnp.ones((4, 8))
    )
    custom_calls = [o for o in ops if o.name == "stablehlo.custom_call"]
    self.assertNotEmpty(custom_calls)
    (reduce_op,) = [o for o in ops if o.name == "stablehlo.reduce"]
    self.assertIn("MAXIMAL_FUSE", reduce_op.attrs)
    # `stablehlo.add` lives inside the reducer region, which stays untagged.
    adds = [o for o in ops if o.name == "stablehlo.add"]
    self.assertTrue(adds)
    for o in adds:
      self.assertNotIn("MAXIMAL_FUSE", o.attrs)
    for o in custom_calls:
      self.assertNotIn("MAXIMAL_FUSE", o.attrs, o)

  @parameterized.named_parameters(
      # (name, inner params, outer params, inner call attrs, inner body attrs,
      #  outer body group key, outer call attrs)
      (
          "both_maximal_keep_own_call_keys",
          _TpuParams(call=(("integer", "9"),), group=True),
          _TpuParams(call=(("integer", "1"),), group=True),
          {"integer": "9"},
          {"MAXIMAL_FUSE": _OUTER},
          "MAXIMAL_FUSE",
          {"integer": "1"},
      ),
      (
          "outer_must_overrides_inner_call_must",
          _call_params(MUST_FUSE="88"),
          _MUST,
          {},
          {"MUST_FUSE": _OUTER},
          "MUST_FUSE",
          {},
      ),
      (
          "inner_call_must_stays_inside_outer_maximal",
          _call_params(MUST_FUSE="88"),
          _GROUP,
          {},
          {"MUST_FUSE": "88"},
          "MAXIMAL_FUSE",
          {},
      ),
      (
          "inner_body_must_stays_inside_outer_maximal",
          _MUST,
          _GROUP,
          {},
          {"MUST_FUSE": _OWN},
          "MAXIMAL_FUSE",
          {},
      ),
      (
          "non_group_body_attrs_keep_inner_call_group",
          _call_params(MUST_FUSE="88"),
          _TpuParams(body=(("k", "1"),)),
          {"k": "1"},
          {"MUST_FUSE": "88"},
          None,
          {},
      ),
  )
  def test_tpu_two_level_groups(
      self,
      inner_params,
      outer_params,
      inner_call_attrs,
      inner_body_attrs,
      outer_key,
      outer_call_attrs,
  ):
    inner = lax.fusion(lambda x: x + 1.0, compiler_params=inner_params)
    outer = lax.fusion(lambda x: inner(x) * 2.0, compiler_params=outer_params)
    ops = _tpu_module(self, outer, jnp.ones((8, 128)))
    (outer_call,) = ops.fusion_calls("main")
    (inner_call,) = ops.fusion_calls(outer_call.callee)
    self.assertEqual(outer_call.attrs, outer_call_attrs)
    self.assertEqual(inner_call.attrs, inner_call_attrs)
    outer_group = None
    if outer_key is not None:
      outer_group = ops.only(outer_call.callee, "stablehlo.multiply").attrs[
          outer_key
      ]
    inner_add = ops.only(inner_call.callee, "stablehlo.add").attrs
    self.assertEqual(inner_add.keys(), inner_body_attrs.keys())
    for key, want in inner_body_attrs.items():
      if want is _OUTER:
        self.assertEqual(inner_add[key], outer_group)
      elif want is _OWN:
        self.assertNotEqual(inner_add[key], outer_group)
      else:
        self.assertEqual(inner_add[key], want)

  @parameterized.named_parameters(
      ("must_over_maximal_over_maximal", _GROUP, _GROUP, _MUST, _OUTER),
      (
          "maximal_over_call_must_over_maximal",
          _GROUP,
          _call_params(MUST_FUSE="88"),
          _GROUP,
          "88",
      ),
      (
          "call_must_over_call_must_over_maximal",
          _GROUP,
          _call_params(MUST_FUSE="7"),
          _call_params(MUST_FUSE="5"),
          "5",
      ),
  )
  def test_tpu_three_level_groups(
      self, leaf_params, mid_params, top_params, want
  ):
    leaf = lax.fusion(lambda x: x + 1.0, compiler_params=leaf_params)
    mid = lax.fusion(lambda x: leaf(x) * 2.0, compiler_params=mid_params)
    top = lax.fusion(lambda x: mid(x) - 3.0, compiler_params=top_params)
    ops = _tpu_module(self, top, jnp.ones((8, 128)))
    (top_call,) = ops.fusion_calls("main")
    self.assertEqual(top_call.attrs, {})
    (mid_call,) = ops.fusion_calls(top_call.callee)
    (leaf_call,) = ops.fusion_calls(mid_call.callee)
    if want is _OUTER:
      want = ops.group(top_call.callee, "MUST_FUSE")
    for call in (mid_call, leaf_call):
      self.assertEqual(ops.group(call.callee, "MUST_FUSE"), want)

  @parameterized.named_parameters(
      ("none_none", None, None, None),
      ("none_maximal", None, "MAXIMAL_FUSE:j", "MAXIMAL_FUSE:j"),
      ("none_must", None, "MUST_FUSE:j", "MUST_FUSE:j"),
      ("maximal_none", "MAXIMAL_FUSE:i", None, "MAXIMAL_FUSE:i"),
      (
          "maximal_maximal",
          "MAXIMAL_FUSE:i",
          "MAXIMAL_FUSE:j",
          "MAXIMAL_FUSE:i",
      ),
      ("maximal_must", "MAXIMAL_FUSE:i", "MUST_FUSE:j", "MUST_FUSE:j"),
      ("must_none", "MUST_FUSE:i", None, "MUST_FUSE:i"),
      ("must_maximal", "MUST_FUSE:i", "MAXIMAL_FUSE:j", "MUST_FUSE:i"),
      ("must_must", "MUST_FUSE:i", "MUST_FUSE:j", "MUST_FUSE:i"),
  )
  def test_combine_group_directives(self, outer, inner, want):
    parse = lambda s: None if s is None else _Directive(*s.split(":"))
    self.assertEqual(_combine(parse(outer), parse(inner)), parse(want))
    if outer is None and inner is None:
      ds = lambda g: (
          None,
          _Directive("MUST_FUSE", g),
          _Directive("MAXIMAL_FUSE", g),
      )
      for d1 in ds("a"):
        for d2 in ds("b"):
          for d3 in ds("c"):
            self.assertEqual(
                _combine(_combine(d1, d2), d3), _combine(d1, _combine(d2, d3))
            )

  @parameterized.named_parameters(
      ("none_must", None, "MUST_FUSE:b"),
      ("must_none", "MUST_FUSE:a", None),
      ("must_maximal", "MUST_FUSE:a", "MAXIMAL_FUSE:b"),
      ("maximal_must", "MAXIMAL_FUSE:a", "MUST_FUSE:b"),
      ("maximal_maximal", "MAXIMAL_FUSE:a", "MAXIMAL_FUSE:b"),
  )
  def test_tpu_call_and_body_group_on_same_fusion(self, call_spec, body_spec):
    parse = lambda s: None if s is None else _Directive(*s.split(":"))
    want = _combine(parse(call_spec), parse(body_spec))
    want_attrs = {} if want is None else {want.key: want.group}
    params = _TpuParams(
        call=() if call_spec is None else (tuple(call_spec.split(":")),),
        body=() if body_spec is None else (tuple(body_spec.split(":")),),
    )
    ops = _tpu_module(
        self, lax.fusion(lambda x: x + 1.0, compiler_params=params), jnp.ones(8)
    )
    (call,) = ops.fusion_calls("main")
    self.assertEqual(call.attrs, {})
    for o in ops.body(call.callee):
      self.assertEqual(o.attrs, want_attrs, o)

  @parameterized.named_parameters(
      ("none_must_none", None, "MUST_FUSE:b", None),
      ("maximal_none_maximal", "MAXIMAL_FUSE:a", None, "MAXIMAL_FUSE:c"),
      (
          "maximal_must_maximal",
          "MAXIMAL_FUSE:a",
          "MUST_FUSE:b",
          "MAXIMAL_FUSE:c",
      ),
      ("must_none_maximal", "MUST_FUSE:a", None, "MAXIMAL_FUSE:c"),
  )
  def test_tpu_group_of_three_nested_fusions(self, g1, g2, g3):
    parse = lambda s: None if s is None else _Directive(*s.split(":"))
    as_attrs = lambda d: {} if d is None else {d.key: d.group}
    d1, d2, d3 = map(parse, (g1, g2, g3))
    d12 = _combine(d1, d2)
    want = _combine(d12, d3)
    self.assertEqual(want, _combine(d1, _combine(d2, d3)))
    leaf = lax.fusion(lambda x: x + 1.0, compiler_params=_group_params(g3))
    mid = lax.fusion(lambda x: leaf(x) * 2.0, compiler_params=_group_params(g2))
    top = lax.fusion(lambda x: mid(x) - 3.0, compiler_params=_group_params(g1))
    ops = _tpu_module(self, top, jnp.ones((8, 128)))
    for o in ops:
      self.assertFalse(
          _GROUP_KEYS[0] in o.attrs and _GROUP_KEYS[1] in o.attrs, o
      )
    (top_call,) = ops.fusion_calls("main")
    (mid_call,) = ops.fusion_calls(top_call.callee)
    (leaf_call,) = ops.fusion_calls(mid_call.callee)
    for o in ops.body(top_call.callee):
      if o.name != "func.call":
        self.assertEqual(o.attrs, as_attrs(d1), o)
    for o in ops.body(mid_call.callee):
      if o.name != "func.call":
        self.assertEqual(o.attrs, as_attrs(d12), o)
    for o in ops.body(leaf_call.callee):
      self.assertEqual(o.attrs, as_attrs(want), o)

  @parameterized.named_parameters(
      ("none_must", None, {"MUST_FUSE": "b"}, {"MUST_FUSE": "b"}),
      (
          "maximal_must",
          "MAXIMAL_FUSE:a",
          {"MUST_FUSE": "b"},
          {"MUST_FUSE": "b"},
      ),
      (
          "maximal_maximal",
          "MAXIMAL_FUSE:a",
          {"MAXIMAL_FUSE": "b"},
          {"MAXIMAL_FUSE": "a"},
      ),
      (
          "must_maximal",
          "MUST_FUSE:a",
          {"MAXIMAL_FUSE": "b"},
          {"MUST_FUSE": "a"},
      ),
  )
  def test_tpu_xla_metadata_and_scheduling_group_in_fusion_body(
      self, outer, inner_meta, want_add
  ):
    @scheduling_group("grp_sub")
    def sub(z):
      return jnp.sin(z)

    def body(x):
      with set_xla_metadata(**inner_meta):
        y = x + 1.0
      with set_xla_metadata(_scheduling_group_id="7:2"):
        z = y * 2.0
      return sub(z)

    fused = lax.fusion(body, compiler_params=_group_params(outer))
    ops = _tpu_module(self, fused, jnp.ones((8, 128)))
    for o in ops:
      self.assertFalse(
          _GROUP_KEYS[0] in o.attrs and _GROUP_KEYS[1] in o.attrs, o
      )
    (call,) = ops.fusion_calls("main")
    self.assertEqual(ops.only(call.callee, "stablehlo.add").attrs, want_add)
    outer_attrs = {} if outer is None else dict([outer.split(":")])
    self.assertEqual(
        ops.only(call.callee, "stablehlo.multiply").attrs,
        {"_scheduling_group_id": "7:2", **outer_attrs},
    )
    (sched_call,) = ops.helper_calls(call.callee)
    self.assertEqual(sched_call.attrs, {"scheduling_group": "grp_sub"})
    self.assertEqual(
        ops.only(sched_call.callee, "stablehlo.sine").attrs, outer_attrs
    )

  def test_tpu_body_tags_do_not_overwrite(self):
    inner = lax.fusion(
        lambda x: x + 1.0,
        compiler_params=_TpuParams(call=(("k", "inner"),)),
    )
    outer = lax.fusion(
        lambda x: inner(x) * 2.0,
        compiler_params=_TpuParams(body=(("k", "outer"), ("only_outer", "1"))),
    )
    ops = _tpu_module(self, outer, 1.0)
    (outer_call,) = ops.fusion_calls("main")
    (inner_call,) = ops.fusion_calls(outer_call.callee)
    self.assertEqual(inner_call.attrs, {"k": "inner", "only_outer": "1"})

  def test_tpu_call_in_pipeline_body_orders_attributes(self):
    pipe = _PipelineParams(
        body=(("scoped_vmem_limit_kib", "512"), ("from_pipeline", "1"))
    )
    own = _call_params(scoped_vmem_limit_kib="64")
    inner = lax.fusion(
        lambda x: x - 1.0, compiler_params=_call_params(integer="3")
    )
    fused = lax.fusion(lambda x: inner(x) * 2.0, compiler_params=own)

    def step(i, carry):
      tile = lax.block_slice(carry, (4,), (i * 4,))
      return lax.block_update(carry, fused(tile), (i * 4,))

    f = lambda init: lax.pipeline(
        step, grid=(2,), init=init, compiler_params=pipe
    )
    ops = _tpu_module(self, f, jnp.ones(8))
    (call,) = ops.fusion_calls("main")
    self.assertEqual(
        call.attrs, {"scoped_vmem_limit_kib": "64", "from_pipeline": "1"}
    )
    for op in ops.body(call.callee):
      self.assertNotIn("scoped_vmem_limit_kib", op.attrs, op)
      self.assertNotIn("from_pipeline", op.attrs, op)
    # Only top level fusions in the pipeline body inherit `pipe` attributes.
    (nested,) = ops.fusion_calls(call.callee)
    self.assertEqual(nested.attrs, {"integer": "3"})

    @dataclasses.dataclass(frozen=True)
    class _BothParams(_TpuParams, _PipelineParams):
      pass

    both = _BothParams(
        call=(("integer", "7"),), body=(("scoped_vmem_limit_kib", "512"),)
    )
    both_fused = lax.fusion(jnp.sin, compiler_params=both)
    both_ops = _tpu_module(
        self,
        lambda init: lax.pipeline(
            lambda i, c: both_fused(c),
            grid=(2,),
            init=init,
            compiler_params=both,
        ),
        jnp.ones(4),
    )
    (both_call,) = both_ops.fusion_calls()
    self.assertEqual(
        both_call.attrs, {"integer": "7", "scoped_vmem_limit_kib": "512"}
    )

  def test_tpu_fusion_without_params_in_pipeline_body(self):
    pipe = _PipelineParams(body=(("scoped_vmem_limit_kib", "512"),))
    fused = lax.fusion(jnp.sin)

    def step(i, carry):
      tile = lax.block_slice(carry, (4,), (i * 4,))
      return lax.block_update(carry, fused(tile), (i * 4,))

    f = lambda init: lax.pipeline(
        step, grid=(2,), init=init, compiler_params=pipe
    )
    ops = _tpu_module(self, f, jnp.ones(8))
    (call,) = ops.fusion_calls()
    self.assertEqual(call.attrs, {"scoped_vmem_limit_kib": "512"})
    tag = ops.group(call.callee, "MAXIMAL_FUSE")
    self.assertEqual(
        [o.attrs for o in ops.body(call.callee)], [{"MAXIMAL_FUSE": tag}]
    )

  def test_tpu_fusion_inside_jit_in_pipeline_body(self):
    pipe = _PipelineParams(
        body=(("scoped_vmem_limit_kib", "512"), ("from_pipeline", "1"))
    )
    inner = lax.fusion(
        lambda x: x - 1.0, compiler_params=_call_params(integer="3")
    )
    inherited = lax.fusion(
        lambda x: inner(x) + 1.0,
        compiler_params=_TpuParams(
            call=(("output_window_bounds", "1,1"),), group=True
        ),
    )
    overridden = lax.fusion(
        lambda x: x * 2.0,
        compiler_params=_TpuParams(
            call=(
                ("scoped_vmem_limit_kib", "64"),
                ("output_window_bounds", "2,1"),
            ),
            group=True,
        ),
    )
    helper = jax.jit(lambda t: overridden(inherited(t)))

    def step(i, carry):
      tile = lax.block_slice(carry, (4,), (i * 4,))
      return lax.block_update(carry, helper(tile), (i * 4,))

    f = lambda init: lax.pipeline(
        step, grid=(2,), init=init, compiler_params=pipe
    )
    ops = _tpu_module(self, f, jnp.ones(8))
    (jit_call,) = [o for o in ops if o.func == "main" and o.name == "func.call"]
    first_call, second_call = [
        o for o in ops.fusion_calls() if o.func == jit_call.callee
    ]
    self.assertEqual(
        first_call.attrs,
        {
            "scoped_vmem_limit_kib": "512",
            "from_pipeline": "1",
            "output_window_bounds": "1,1",
        },
    )
    self.assertEqual(
        second_call.attrs,
        {
            "scoped_vmem_limit_kib": "64",
            "from_pipeline": "1",
            "output_window_bounds": "2,1",
        },
    )
    (nested,) = ops.fusion_calls(first_call.callee)
    self.assertEqual(nested.attrs, {"integer": "3"})
    for c in (first_call, second_call):
      for op in ops.body(c.callee):
        self.assertNotIn("scoped_vmem_limit_kib", op.attrs, op)
        self.assertNotIn("from_pipeline", op.attrs, op)

  def test_lowering_is_not_cached(self):
    # Lowering allocates per call ids and reads nesting state.
    self.assertIn(fusion_p, mlir._uncacheable_primitives)

  def test_tpu_helper_callees_are_not_shared_with_outside_callers(self):
    params = _GROUP

    def f(x):
      inside = lax.fusion(_uses_helpers, compiler_params=params)(x)
      return _uses_helpers(x) + inside

    ops = _tpu_module(self, f, jnp.ones(8))
    (call,) = ops.fusion_calls()
    main_helpers = {c.callee for c in ops.helper_calls("main")}
    body_helpers = {c.callee for c in ops.helper_calls(call.callee)}
    self.assertLen(main_helpers, 2)
    self.assertLen(body_helpers, 2)
    self.assertEmpty(main_helpers & body_helpers)
    # Top level helper ops stay untagged.
    self.assertTrue(all(not o.attrs for o in ops.helper_ops("main")))
    # Body helper calls and ops carry `tag`.
    (tag,) = {
        o.attrs["MAXIMAL_FUSE"]
        for o in ops.body(call.callee)
        if o.name == "stablehlo.multiply"
    }
    for c in ops.helper_calls(call.callee):
      self.assertEqual(c.attrs, {"MAXIMAL_FUSE": tag}, str(c))
    helper_ops = ops.helper_ops(call.callee)
    self.assertTrue(any(o.name == "stablehlo.select" for o in helper_ops))
    for o in helper_ops:
      self.assertEqual(o.attrs, {"MAXIMAL_FUSE": tag}, str(o))

  def test_tpu_maximal_fuse_tags_inlined_helper_calls(self):
    @jax.jit
    def scale_add(u, v):
      return u * v + u

    def body(x, y):
      a = jnp.where(x > y, x, y)
      b = jnp.where(x < y, x, y)
      return scale_add(a, b)

    ops = _tpu_module(
        self,
        lax.fusion(body, compiler_params=_GROUP),
        jnp.ones((8, 128)),
        jnp.zeros((8, 128)),
    )
    (call,) = ops.fusion_calls("main")
    tag = ops.group(call.callee, "MAXIMAL_FUSE")
    helper_calls = ops.helper_calls(call.callee)
    self.assertLen(helper_calls, 3)
    for c in helper_calls:
      self.assertEqual(c.attrs, {"MAXIMAL_FUSE": tag}, str(c))

    # A jitted helper wrapping a `MUST_FUSE` fusion stays untagged on its call
    # op so XLA inlining preserves the inner `MUST_FUSE` group.
    inner_must = lax.fusion(lambda z: z + 1.0, compiler_params=_MUST)
    wrap = jax.jit(lambda z: inner_must(z) * 2.0)
    wrap_ops = _tpu_module(
        self, lax.fusion(wrap, compiler_params=_GROUP), jnp.ones((8, 128))
    )
    (outer_call,) = wrap_ops.fusion_calls("main")
    (wrap_call,) = wrap_ops.helper_calls(outer_call.callee)
    self.assertEqual(wrap_call.attrs, {})
    (inner_call,) = wrap_ops.fusion_calls(wrap_call.callee)
    self.assertIn(
        "MUST_FUSE", wrap_ops.only(inner_call.callee, "stablehlo.add").attrs
    )

  def test_tpu_two_fusions_get_their_own_helper_callees(self):
    params = _GROUP
    g = lax.fusion(_uses_helpers, compiler_params=params)
    ops = _tpu_module(self, lambda x: g(g(x)), jnp.ones(8))
    first, second = ops.fusion_calls()
    self.assertNotEqual(first.callee, second.callee)
    self.assertEmpty(
        {c.callee for c in ops.helper_calls(first.callee)}
        & {c.callee for c in ops.helper_calls(second.callee)}
    )
    body_tags = [
        {o.attrs["MAXIMAL_FUSE"] for o in ops.body(c.callee)}
        for c in (first, second)
    ]
    tags = [
        {
            o.attrs["MAXIMAL_FUSE"]
            for o in ops.helper_ops(c.callee)
            if o.name != "func.call"
        }
        for c in (first, second)
    ]
    self.assertLen(body_tags[0], 1)
    self.assertLen(tags[0], 1)
    self.assertLen(tags[1], 1)
    self.assertEqual(body_tags, tags)
    self.assertNotEqual(tags[0], tags[1])

  def test_tpu_late_inlined_helper_is_not_part_of_the_body(self):
    params = _GROUP
    late = jax.jit(lambda x: x * 3.0, inline=jax.Inline.XLA_LATE)
    ops = _tpu_module(
        self,
        lax.fusion(lambda x: late(x) + 1.0, compiler_params=params),
        jnp.ones(8),
    )
    (call,) = ops.fusion_calls()
    (helper,) = ops.helper_calls(call.callee)
    self.assertEqual(helper.attrs, {"inlineable": "xla_late"})
    self.assertTrue(
        all("MAXIMAL_FUSE" not in o.attrs for o in ops.body(helper.callee))
    )

  def test_tpu_must_fuse_body_with_nested_fusion_and_helpers(self):
    inner = lax.fusion(lambda x: x * 2.0 + 1.0, compiler_params=_GROUP)
    first = lax.fusion(lambda x: x - 1.0, compiler_params=_MUST)
    second = lax.fusion(
        lambda x: inner(_uses_helpers(x)) + 3.0,
        compiler_params=_MUST,
    )
    ops = _tpu_module(self, lambda x: second(first(x)), jnp.ones(8))
    first_call, second_call = ops.fusion_calls("main")
    (first_tag,) = {o.attrs["MUST_FUSE"] for o in ops.body(first_call.callee)}
    outer_add = ops.only(second_call.callee, "stablehlo.add")
    second_tag = outer_add.attrs["MUST_FUSE"]
    self.assertNotEqual(first_tag, second_tag)
    # Inner call ops have empty group keys so XLA inlines them cleanly.
    for o in ops:
      if o.name == "func.call" and o.func != "main":
        self.assertNotIn("MUST_FUSE", o.attrs, o)
    for o in ops.helper_ops(second_call.callee):
      if o.name != "func.call":
        self.assertEqual(o.attrs["MUST_FUSE"], second_tag, o)
    # Inner fusion ops take the outer `MUST_FUSE` group.
    (inner_call,) = ops.fusion_calls(second_call.callee)
    inner_ops = ops.body(inner_call.callee)
    self.assertNotEmpty(inner_ops)
    for o in inner_ops:
      self.assertEqual(o.attrs, {"MUST_FUSE": second_tag}, o)

  def test_multiplatform_lowering_narrows_the_body_platform(self):
    inner = lax.fusion(lambda x: jnp.sin(x) + 1.0, compiler_params=_GROUP)
    params = _TpuParams(call=(("integer", "3"),), group=True)
    f = lax.fusion(lambda x: inner(x) * 2.0, compiler_params=params)
    ops = _Module(_lower(f, jnp.ones(8), platform=("cpu", "tpu")))
    outer_calls = ops.fusion_calls("main")
    self.assertLen(outer_calls, 2)
    by_attrs = {bool(c.attrs): c for c in outer_calls}
    self.assertEqual(by_attrs[True].attrs, {"integer": "3"})
    self.assertEqual(by_attrs[False].attrs, {})
    # Each branch lowers its callee for a single platform.
    for o in ops:
      if o.func != "main":
        self.assertNotEqual(o.name, "stablehlo.case", o)
    (cpu_inner,) = ops.fusion_calls(by_attrs[False].callee)
    cpu_ops = ops.body(by_attrs[False].callee) + ops.body(cpu_inner.callee)
    self.assertTrue(all(not o.attrs for o in cpu_ops))
    (tpu_inner,) = ops.fusion_calls(by_attrs[True].callee)
    self.assertTrue(
        all("MAXIMAL_FUSE" in o.attrs for o in ops.body(tpu_inner.callee))
    )

  def test_tpu_jitted_helper_with_fusion_inside_and_outside_a_fusion(self):
    inner = lax.fusion(
        lambda x: x + 1.0,
        compiler_params=_TpuParams(call=(("integer", "9"),), group=True),
    )
    helper = jax.jit(lambda x: inner(x) * 2.0)
    outer = lax.fusion(lambda x: helper(x) - 3.0, compiler_params=_GROUP)
    ops = _tpu_module(self, lambda x: helper(x) + outer(x), jnp.ones(8))
    (top_helper,) = ops.helper_calls("main")
    (outer_call,) = ops.fusion_calls("main")
    (body_helper,) = ops.helper_calls(outer_call.callee)
    # The helper inside the fusion body has a callee of its own.
    self.assertNotEqual(top_helper.callee, body_helper.callee)
    self.assertEqual(top_helper.attrs, {})
    self.assertEqual(body_helper.attrs, {})
    outer_sub = ops.only(outer_call.callee, "stablehlo.subtract")
    outer_tag = outer_sub.attrs["MAXIMAL_FUSE"]
    # Top level copy: only the nested fusion's own tags.
    top_mul = ops.only(top_helper.callee, "stablehlo.multiply")
    self.assertEqual(top_mul.attrs, {})
    (top_inner,) = ops.fusion_calls(top_helper.callee)
    self.assertEqual(top_inner.attrs, {"integer": "9"})
    # Body copy: helper and inner ops join `outer_tag`; call keeps `integer`.
    body_mul = ops.only(body_helper.callee, "stablehlo.multiply")
    self.assertEqual(body_mul.attrs, {"MAXIMAL_FUSE": outer_tag})
    (body_inner,) = ops.fusion_calls(body_helper.callee)
    self.assertEqual(body_inner.attrs, {"integer": "9"})
    self.assertNotEqual(top_inner.callee, body_inner.callee)
    inner_tags = [
        {o.attrs["MAXIMAL_FUSE"] for o in ops.body(c.callee)}
        for c in (top_inner, body_inner)
    ]
    self.assertLen(inner_tags[0], 1)
    self.assertEqual(inner_tags[1], {outer_tag})
    self.assertNotIn(outer_tag, inner_tags[0])

  def test_tpu_ungrouped_outer_with_call_tagged_must_fuse_fusion(self):
    inner = lax.fusion(
        lambda x: x + 1.0, compiler_params=_call_params(MUST_FUSE="88")
    )
    helper = jax.jit(lambda x: inner(x) * 3.0)
    outer = lax.fusion(
        lambda x: helper(inner(x)) * 2.0,
        compiler_params=_call_params(scoped_vmem_limit_kib="512"),
    )
    ops = _tpu_module(self, outer, jnp.ones((8, 128)))
    (outer_call,) = ops.fusion_calls("main")
    self.assertNotIn("MUST_FUSE", outer_call.attrs)
    inner_calls = [o for o in ops.fusion_calls() if o.func != "main"]
    self.assertLen(inner_calls, 2)
    for c in inner_calls:
      self.assertEqual(c.attrs, {})
      # The inner fusion keeps its own group on its body ops.
      self.assertEqual(
          ops.only(c.callee, "stablehlo.add").attrs, {"MUST_FUSE": "88"}
      )
    # Outer body ops stay untagged.
    self.assertEqual(
        ops.only(outer_call.callee, "stablehlo.multiply").attrs, {}
    )

  def test_tpu_two_calls_sharing_a_call_level_group(self):
    first = lax.fusion(
        lambda x: x + 1.0,
        compiler_params=_call_params(MUST_FUSE="88", FUSE_LIMIT="1"),
    )
    second = lax.fusion(
        lambda x: x * 2.0,
        compiler_params=_call_params(MUST_FUSE="88"),
    )
    ops = _tpu_module(self, lambda x: (first(x), second(x)), jnp.ones((8, 128)))
    first_call, second_call = ops.fusion_calls()
    self.assertEqual(first_call.attrs, {"FUSE_LIMIT": "1"})
    self.assertEqual(second_call.attrs, {})
    callee_ops = ops.body(first_call.callee) + ops.body(second_call.callee)
    groups = {o.attrs.get("MUST_FUSE") for o in callee_ops}
    self.assertEqual(groups, {"88"})

  def test_tpu_fusion_facts(self):

    @dataclasses.dataclass(frozen=True)
    class Recording(FusionParams):
      seen: list[Any] = dataclasses.field(
          default_factory=list, hash=False, compare=False
      )

    class RecordingHandler:

      def fusion_call_attributes(
          self, params: Recording, ctx: tpu_fusion.FusionContext
      ) -> Mapping[str, str]:
        params.seen.append(("call", ctx, tpu_fusion.in_fusion_body()))
        return {}

      def fusion_body_attributes(
          self, params: Recording, ctx: tpu_fusion.FusionContext
      ) -> Mapping[str, str]:
        params.seen.append(("body", ctx, tpu_fusion.in_fusion_body()))
        return {}

      def verify_fusion_body(
          self, params: Recording, ctx: tpu_fusion.FusionContext, jaxpr
      ):
        del jaxpr
        params.seen.append(("verify_body", ctx, tpu_fusion.in_fusion_body()))

      def verify_fusion_operands(
          self, params: Recording, ctx: tpu_fusion.FusionContext, operands
      ):
        del operands
        params.seen.append(
            ("verify_operands", ctx, tpu_fusion.in_fusion_body())
        )

    tpu_fusion.register_fusion_handler(Recording, RecordingHandler())
    self.addCleanup(tpu_fusion.unregister_fusion_handler, Recording)
    params = Recording()
    inner = lax.fusion(lambda u: u + 1.0, compiler_params=params)

    def f(x, y, z):
      return lax.fusion(
          lambda a, d: inner(a) * y + d["b"], compiler_params=params
      )(x, {"b": z})

    self.assertFalse(tpu_fusion.in_fusion_body())
    _lower(f, jnp.ones(8), 2.0, jnp.ones(8), platform="tpu")
    self.assertFalse(tpu_fusion.in_fusion_body())
    outer_ctx = tpu_fusion.FusionContext(fusion_id=1, num_consts=1, num_args=2)
    inner_ctx = tpu_fusion.FusionContext(fusion_id=2, num_consts=0, num_args=1)
    expected_hooks = ["verify_body", "call", "body", "verify_operands"]
    self.assertEqual(
        params.seen,
        [(h, outer_ctx, False) for h in expected_hooks]
        + [(h, inner_ctx, True) for h in expected_hooks],
    )

    bad_inner = lax.fusion(jnp.cos, compiler_params=_CheckedParams())
    bad_outer = lax.fusion(lambda x: bad_inner(x) + 1.0)
    with self.assertRaisesRegex(ValueError, "cos is not allowed"):
      _lower(bad_outer, 1.0, platform="tpu")
    self.assertFalse(tpu_fusion.in_fusion_body())

  @jtu.with_explicit_mesh((1,), ("x",))
  def test_explicit_mesh_lowering(self, mesh):
    del mesh
    params = _TpuParams(call=(("integer", "2"),), group=True)
    f = jax.jit(lax.fusion(lambda x: jnp.sin(x) * 2.0, compiler_params=params))
    x = jnp.ones(8)
    with config.jax_partitioning(True):
      for platform in ("cpu", "tpu"):
        calls = _Module(_lower(f, x, platform=platform)).fusion_calls()
        self.assertLen(calls, 1)
      self.assertAllClose(f(x), jnp.sin(x) * 2.0)

  def test_tpu_verify_operands(self):
    params = _OperandParams()
    table = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)

    def f(x, y):
      # `y` is a closed over operand; `table` is a hoisted constant.
      return lax.fusion(lambda x: x * table + y, compiler_params=params)(x)

    x = jnp.ones((64, 64), jnp.float32)
    _lower(f, x, jnp.float32(2.0), platform="tpu")
    ((ctx, types),) = params.seen
    self.assertEqual(
        ctx, tpu_fusion.FusionContext(fusion_id=1, num_consts=1, num_args=1)
    )
    self.assertEqual(types, ["tensor<f32>", "tensor<64x64xf32>"])
    bad = lax.fusion(lambda x: x + 1, compiler_params=params)
    with self.assertRaisesRegex(ValueError, "int32 operands"):
      _lower(bad, np.int32(1), platform="tpu")
    _lower(bad, np.int32(1), platform="cpu")  # Only the TPU lowering checks.

  def test_tpu_verify_body(self):
    params = _CheckedParams()
    ok = lax.fusion(jnp.sin, compiler_params=params)
    _lower(ok, 1.0, platform="tpu")
    ((ctx, jaxpr),) = params.seen
    self.assertEqual(
        ctx, tpu_fusion.FusionContext(fusion_id=1, num_consts=0, num_args=1)
    )
    self.assertEqual([e.primitive.name for e in jaxpr.eqns], ["sin"])
    bad = lax.fusion(jnp.cos, compiler_params=params)
    with self.assertRaisesRegex(ValueError, "cos is not allowed"):
      _lower(bad, 1.0, platform="tpu")
    _lower(bad, 1.0, platform="cpu")  # Only the TPU lowering verifies.

  def test_tpu_lowering_inside_vmap_and_scan(self):
    params = _GROUP
    g = lax.fusion(lambda x: x * 2.0, compiler_params=params)

    def f(xs):
      return lax.scan(lambda c, x: (c + g(x), None), 0.0, xs)[0]

    ops = _tpu_module(self, jax.vmap(f), jnp.ones((2, 3)))
    (call,) = ops.fusion_calls()
    self.assertTrue(
        all("MAXIMAL_FUSE" in o.attrs for o in ops.body(call.callee))
    )


def _numerics_program(kind):
  """Builds a program for `test_numerics_next_to_an_unrelated_must_fuse`."""
  if kind == "nested_maximal":
    inner = lax.fusion(lambda x: x * 2.0 + 1.0, compiler_params=_GROUP)
    return lax.fusion(lambda x: inner(x) - 1.0, compiler_params=_GROUP)
  if kind == "three_level_must":
    leaf = lax.fusion(lambda x: x + 1.0, compiler_params=_GROUP)
    mid = lax.fusion(lambda x: leaf(x) * 2.0, compiler_params=_GROUP)
    return lax.fusion(lambda x: mid(x) - 3.0, compiler_params=_MUST)
  if kind == "call_must_jit":
    inner = lax.fusion(lambda x: x * 3.0, compiler_params=_GROUP)
    helper = jax.jit(lambda x: inner(x) - 1.0)
    return lax.fusion(
        lambda x: helper(x) + 1.0, compiler_params=_call_params(MUST_FUSE="88")
    )
  if kind == "two_calls_shared":
    p = _call_params(MUST_FUSE="88")
    a = lax.fusion(lambda x: x + 1.0, compiler_params=p)
    b = lax.fusion(lambda x: x * 2.0, compiler_params=p)
    return lambda x: a(x) + b(x)
  if kind == "call_group_loop":
    inner = lax.fusion(lambda x: x + 1.0, compiler_params=_MUST)
    return lax.fusion(
        lambda x: lax.fori_loop(0, 3, lambda i, c: inner(c) * 2.0, x),
        compiler_params=_call_params(MUST_FUSE="5"),
    )
  inner = lax.fusion(
      lambda x: x + 1.0, compiler_params=_call_params(MUST_FUSE="88")
  )
  if kind == "ungrouped_over_call_must":
    return lax.fusion(
        inner, compiler_params=_call_params(scoped_vmem_limit_kib="512")
    )
  return lax.fusion(lambda x: inner(x) * 2.0, compiler_params=_MUST)


class FusionNumericsTest(jtu.JaxTestCase):
  """Runs on TPU where fusion attributes reach XLA."""

  def test_numerics_with_tpu_attributes(self):
    params = _TpuParams(call=(("scoped_vmem_limit_kib", "512"),), group=True)

    def body(x, y):
      return jnp.tanh(x) * y + 1.0

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128) / 1000.0
    y = jnp.full((8, 128), 0.5, jnp.float32)
    f = jax.jit(
        lambda x, y: lax.fusion(body, compiler_params=params)(x, y) * 2.0
    )
    self.assertAllClose(f(x, y), body(x, y) * 2.0, rtol=1e-6, atol=1e-6)

  def test_numerics_with_must_fuse_bodies_helpers_and_nested_fusion(self):
    inner = lax.fusion(lambda x: x * 2.0 + 1.0, compiler_params=_GROUP)
    first = lax.fusion(lambda x: x - 1.0, compiler_params=_MUST)
    second = lax.fusion(
        lambda x: inner(_uses_helpers(x)) + 3.0,
        compiler_params=_MUST,
    )

    def f(x):
      return second(first(x)) + _uses_helpers(x)

    def reference(x):
      y = _uses_helpers(x - 1.0) * 2.0 + 1.0 + 3.0
      return y + _uses_helpers(x)

    x = jnp.linspace(-3.0, 3.0, 8 * 128, dtype=jnp.float32).reshape(8, 128)
    self.assertAllClose(jax.jit(f)(x), reference(x), rtol=1e-5, atol=1e-5)

  def test_numerics_nested_maximal_with_unfusable_inner_body(self):
    inner = lax.fusion(
        lambda x, y: (jnp.exp(x) + 1.0, jnp.sin(y) + 2.0),
        compiler_params=_GROUP,
    )
    outer = jax.jit(lax.fusion(inner, compiler_params=_GROUP))
    x = jnp.linspace(-1.0, 1.0, 8 * 128, dtype=jnp.float32).reshape(8, 128)
    y = jnp.linspace(-2.0, 2.0, 16 * 256, dtype=jnp.float32).reshape(16, 256)
    a, b = outer(x, y)
    self.assertAllClose(a, jnp.exp(x) + 1.0, rtol=1e-5, atol=1e-5)
    self.assertAllClose(b, jnp.sin(y) + 2.0, rtol=1e-5, atol=1e-5)

  # Lowering `must` first claims XLA:TPU group 1 on an incompatible shape;
  # XLA:TPU also assigns group 1 to any grouped nested call op, which would
  # collide with `must` at compile time.
  @parameterized.named_parameters(
      ("nested_maximal_next_to_must_fuse", "nested_maximal", lambda x: x * 2.0),
      (
          "three_level_must_fuse_over_maximal",
          "three_level_must",
          lambda x: (x + 1.0) * 2.0 - 3.0,
      ),
      ("call_must_fuse_over_jit_helper", "call_must_jit", lambda x: x * 3.0),
      (
          "two_calls_sharing_a_call_level_group",
          "two_calls_shared",
          lambda x: (x + 1.0) + x * 2.0,
      ),
      (
          "call_level_group_with_loop_in_body",
          "call_group_loop",
          lambda x: ((((x + 1.0) * 2.0) + 1.0) * 2.0 + 1.0) * 2.0,
      ),
      (
          "ungrouped_outer_over_call_must_fuse",
          "ungrouped_over_call_must",
          lambda x: x + 1.0,
      ),
      (
          "must_fuse_body_over_call_must_fuse",
          "must_over_call_must",
          lambda x: (x + 1.0) * 2.0,
      ),
  )
  def test_numerics_next_to_an_unrelated_must_fuse(self, kind, reference):
    must = lax.fusion(lambda y: jnp.cos(y) + 1.0, compiler_params=_MUST)
    fused = _numerics_program(kind)
    x = jnp.linspace(-1.0, 1.0, 8 * 128, dtype=jnp.float32).reshape(8, 128)
    y = jnp.linspace(-2.0, 2.0, 16 * 256, dtype=jnp.float32).reshape(16, 256)
    a, b = jax.jit(lambda x, y: (must(y), fused(x)))(x, y)
    self.assertAllClose(a, jnp.cos(y) + 1.0, rtol=1e-5, atol=1e-5)
    self.assertAllClose(b, reference(x), rtol=1e-5, atol=1e-5)

  def test_compiled_maximal_fuse_assigns_must_fuse_to_helper_calls(self):
    @jax.jit
    def scale_add(u, v):
      return u * v + u

    fused = lax.fusion(
        lambda x, y: scale_add(jnp.where(x > y, x, y), y),
        compiler_params=_GROUP,
    )
    x = jnp.linspace(-1.0, 1.0, 8 * 128, dtype=jnp.float32).reshape(8, 128)
    y = jnp.zeros((8, 128), dtype=jnp.float32)
    compiled = jax.jit(fused).lower(x, y).compile()
    self.assertIn("MUST_FUSE=", compiled.as_text() or "")
    self.assertAllClose(
        compiled(x, y),
        scale_add(jnp.where(x > y, x, y), y),
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
