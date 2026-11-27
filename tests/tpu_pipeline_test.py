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

"""Tests for TPU lowering of lax.pipeline, block_slice, and block_update."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import threading
from typing import Any, ClassVar, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import core
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lax.control_flow import pipeline as pipeline_lib
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.tpu import frontend_attrs as tpu_attrs
from jax._src.tpu import pipeline as tpu_pipeline
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class _Log(threading.local):
  """Per thread query log; the test runner may run tests in threads."""

  def __init__(self):
    super().__init__()
    self.entries: list[tuple[str, Any]] = []

  def append(self, entry):
    self.entries.append(entry)

  def clear(self):
    self.entries.clear()

  def __iter__(self):
    return iter(list(self.entries))


@dataclasses.dataclass(frozen=True)
class _Params(pipeline_lib.PipelineParams):
  """Test only TPU params with a simple pipelining decision."""

  # Each query appends (method, details) here; tests clear it in setUp.
  log: ClassVar[_Log] = _Log()

  levels: int = 2
  vmem_kib: int | None = None
  read_tag: str | None = None
  write_tag: str | None = None
  # When set, the loop attributes returned whatever the loop facts say.
  loop_attrs: tuple[tuple[str, str], ...] | None = None


class _ParamsHandler:
  """TPU pipeline handler for `_Params`."""

  def pipeline_loop_attributes(
      self, params: Any, facts: tpu_pipeline.LoopFacts
  ) -> Mapping[str, str]:
    params.log.append(("loop", facts))
    if params.loop_attrs is not None:
      return dict(params.loop_attrs)
    # Pipeline a loop with a static trip count of at least 4 and no nested
    # loop.
    if facts.trip_count < 4 or facts.has_nested_loop:
      return {}
    return {
        "xla_disable_while_loop_copies": "true",
        "xla_while_loop_async_pipelining_levels": str(params.levels),
    }

  def pipeline_body_attributes(
      self, params: Any, facts: tpu_pipeline.LoopFacts
  ) -> Mapping[str, str]:
    del facts
    if params.vmem_kib is None:
      return {}
    return {"scoped_vmem_limit_kib": str(params.vmem_kib)}

  def block_read_attributes(
      self, params: Any, facts: tpu_pipeline.LoopFacts
  ) -> Mapping[str, str]:
    params.log.append(("read", (tpu_pipeline.enclosing_pipeline(), facts)))
    return {} if params.read_tag is None else {"test_read": params.read_tag}

  def block_write_attributes(
      self, params: Any, facts: tpu_pipeline.LoopFacts
  ) -> Mapping[str, str]:
    params.log.append(("write", (tpu_pipeline.enclosing_pipeline(), facts)))
    return {} if params.write_tag is None else {"test_write": params.write_tag}

  def verify_pipeline_body(
      self,
      params: Any,
      facts: tpu_pipeline.LoopFacts,
      jaxpr: core.Jaxpr,
  ) -> None:
    params.log.append(("verify", (jaxpr, facts)))


tpu_pipeline.register_pipeline_handler(_Params, _ParamsHandler())


def _logged_facts() -> list[tpu_pipeline.LoopFacts]:
  return _logged("loop")


def _lower(f, *args, platform="tpu"):
  return jax.jit(f).trace(*args).lower(lowering_platforms=(platform,))


class _Op(NamedTuple):
  """An op of a lowered module.

  `path` holds one (op name, region index) entry per enclosing op; a while's
  condition is region 0 and its body region 1.
  """

  name: str
  attrs: dict[str, str]
  path: tuple[tuple[str, int], ...]
  op: ir.Operation


def _ops(lowered) -> list[_Op]:
  """Every op of `lowered`, outermost first."""
  module = lowered.compiler_ir("stablehlo")
  out = []

  def visit(op, path):
    for r, region in enumerate(op.regions):
      inner = path + ((op.name, r),)
      for block in region.blocks:
        for child in block.operations:
          c = child.operation
          out.append(_Op(c.name, tpu_attrs.frontend_attributes(c), inner, c))
          visit(c, inner)

  with module.context:
    visit(module.operation, ())
  return out


def _attrs(ops, name, where=lambda path: True) -> list[dict[str, str]]:
  """The attributes of the ops named `name` whose path satisfies `where`."""
  return [o.attrs for o in ops if o.name == name and where(o.path)]


def _only(ops, name) -> dict[str, str]:
  """The attributes of the one op named `name`."""
  (attrs,) = _attrs(ops, name)
  return attrs


_BODY_SKIP = ("stablehlo.return", "stablehlo.constant")


def _in_while_body(path, depth=1):
  whiles = [r for name, r in path if name == "stablehlo.while"]
  return len(whiles) >= depth and whiles[depth - 1] == 1


def _in_while_cond(path):
  return any(name == "stablehlo.while" and r == 0 for name, r in path)


def _pipeline_eqn(jaxpr: core.Jaxpr) -> core.JaxprEqn:
  (eqn,) = (e for e in jaxpr.eqns if e.primitive is pipeline_lib.pipeline_p)
  return eqn


_DISJOINT = "xla_disjoint_read_write_regions"
_VMEM = "scoped_vmem_limit_kib"
_REDUCE = "stablehlo.reduce"


def _ones(shape=(64, 128)):
  return jnp.ones(shape, jnp.float32)


def _tile():
  return jnp.zeros((16, 128), jnp.float32)


def _logged(kind: str) -> list[Any]:
  return [d for s, d in _Params.log if s == kind]


def _tiled_copy(x, tile=16, **kw):
  def body(i, acc):
    t = lax.block_slice(x, (tile, x.shape[1]), (i * tile, 0))
    return lax.block_update(acc, t * 2.0, (i * tile, 0))

  return lax.pipeline(
      body, grid=(x.shape[0] // tile,), init=jnp.zeros_like(x), **kw
  )


def _copy_with(params):
  return lambda x: _tiled_copy(x, compiler_params=params)


def _bind_slice(x, *index, block_shape=(16, 128), revisits_tile=True):
  # Binding the primitive directly bypasses the trace time placement check.
  return pipeline_lib.block_slice_p.bind(
      x,
      *index,
      block_shape=block_shape,
      revisits_tile=revisits_tile,
  )


def _bind_block_slice(x, i):
  return _bind_slice(x, i, jnp.int32(0))


def _top_level_block_slice(x):
  return _bind_block_slice(x, jnp.int32(0))


def _top_level_block_update(x):
  zero = jnp.int32(0)
  return pipeline_lib.block_update_p.bind(x, x[:16], zero, zero, disjoint=True)


def _block_slice_in_cond(x):
  def body(i, acc):
    return lax.cond(i > 0, lambda: acc + _bind_block_slice(x, i), lambda: acc)

  return lax.pipeline(body, grid=(4,), init=_tile())


def _block_slice_in_jit(x):
  def body(i, acc):
    return jax.jit(lambda: acc + _bind_block_slice(x, i))()

  return lax.pipeline(body, grid=(4,), init=_tile())


# Programs that put a block op outside the direct body of a pipeline. They
# reach lowering only from jaxprs that skipped the typecheck.
_MISPLACED_BLOCK_OPS = (
    ("top_level_block_slice", _top_level_block_slice),
    ("top_level_block_update", _top_level_block_update),
    ("block_slice_in_cond", _block_slice_in_cond),
    ("block_slice_in_jit", _block_slice_in_jit),
)


class TpuPipelineTest(jtu.JaxTestCase):

  def test_values_with_tpu_params(self):
    # On TPU the loop lowers with the pipelining keys and ColorToVMEM tags;
    # elsewhere the lowering ignores the params. The value is the same.
    x = jnp.arange(64 * 256, dtype=jnp.float32).reshape(64, 256)
    params = _Params(levels=2, vmem_kib=1024)

    def f(x):
      def body(i, j, carry):
        out, rowmax = carry
        t = lax.block_slice(x, (8, 128), (i * 8, j * 128))
        out = lax.block_update(out, t * 2.0 + 1.0, (i * 8, j * 128))
        return out, jnp.maximum(rowmax, jnp.max(t, axis=1, keepdims=True))

      init = (jnp.zeros_like(x), jnp.full((8, 1), -jnp.inf, jnp.float32))
      return lax.pipeline(body, grid=(8, 2), init=init, compiler_params=params)

    out, rowmax = jax.jit(f)(x)
    self.assertAllClose(out, x * 2.0 + 1.0)
    self.assertAllClose(rowmax, x.reshape(8, 8, 256).max(axis=(0, 2))[:, None])


class PipelineLoweringTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    _Params.log.clear()

  @parameterized.named_parameters(
      *[(f"tpu_{name}", program) for name, program in _MISPLACED_BLOCK_OPS]
  )
  def test_misplaced_block_op_fails_to_lower(self, program):
    with (
        jax.enable_checks(False),
        self.assertRaisesRegex(ValueError, "directly in the body"),
    ):
      _lower(program, _ones(), platform="tpu")

  def test_loop_keys_on_while(self):
    ops = _ops(_lower(_copy_with(_Params(levels=3)), _ones()))
    self.assertEqual(
        _attrs(ops, "stablehlo.while"),
        [{
            "xla_disable_while_loop_copies": "true",
            "xla_while_loop_async_pipelining_levels": "3",
        }],
    )

  def test_one_while_with_divmod_coordinates_for_nd_grid(self):
    x = jnp.ones((32, 48), jnp.float32)

    def f(x):
      def body(i, j, acc):
        t = lax.block_slice(x, (8, 16), (i * 8, j * 16))
        return lax.block_update(acc, t, (i * 8, j * 16))

      return lax.pipeline(
          body, grid=(4, 3), init=jnp.zeros_like(x), compiler_params=_Params()
      )

    ops = _ops(_lower(f, x))
    self.assertLen(_attrs(ops, "stablehlo.while"), 1)
    body_names = {o.name for o in ops if _in_while_body(o.path)}
    self.assertContainsSubset(
        {"stablehlo.remainder", "stablehlo.divide"}, body_names
    )
    (facts,) = _logged_facts()
    self.assertEqual(facts.trip_count, 12)

  @parameterized.named_parameters(
      ("disjoint_true", True, ["stablehlo.dynamic_update_slice"], "true"),
      ("disjoint_false", False, [], None),
  )
  def test_disjoint_key_on_dynamic_update_slice(
      self, disjoint, want_tagged, want_val
  ):
    def f(x):
      def body(i, acc):
        t = lax.block_slice(x, (16, 128), (i * 16, 0))
        return lax.block_update(acc, t * 2.0, (i * 16, 0), disjoint=disjoint)

      return lax.pipeline(
          body, grid=(4,), init=jnp.zeros_like(x), compiler_params=_Params()
      )

    ops = _ops(_lower(f, _ones()))
    self.assertEqual([o.name for o in ops if _DISJOINT in o.attrs], want_tagged)
    self.assertEqual(
        _only(ops, "stablehlo.dynamic_update_slice").get(_DISJOINT), want_val
    )
    # The index arithmetic of the block ops carries no block attribute.
    for name in ("stablehlo.compare", "stablehlo.select", "stablehlo.multiply"):
      for attrs in _attrs(ops, name, _in_while_body):
        self.assertNotIn(_DISJOINT, attrs)

  def test_revisits_tile_only_on_requested_dynamic_slice(self):
    def f(x):
      def body(i, acc):
        a = lax.block_slice(x, (16, 128), (i * 16, 0), revisits_tile=False)
        b = lax.block_slice(x, (16, 128), (0, 0))
        return acc + a + b

      return lax.pipeline(
          body, grid=(4,), init=_tile(), compiler_params=_Params()
      )

    slices = _attrs(_ops(_lower(f, _ones())), "stablehlo.dynamic_slice")
    self.assertEqual(
        sorted(slices, key=len),
        [{}, {"xla_while_loop_async_immediate_revisiting": "false"}],
    )

  def test_scoped_vmem_on_body_ops_not_on_condition(self):
    ops = _ops(_lower(_copy_with(_Params(vmem_kib=512)), _ones()))
    body = [
        o for o in ops if _in_while_body(o.path) and o.name not in _BODY_SKIP
    ]
    self.assertNotEmpty(body)
    for o in body:
      self.assertEqual(o.attrs.get(_VMEM), "512", o.name)
    for o in ops:
      if _in_while_cond(o.path) or o.name == "stablehlo.while":
        self.assertNotIn(_VMEM, o.attrs)

  def test_no_params_means_no_loop_keys_and_color_input_to_hbm(self):
    ops = _ops(_lower(_tiled_copy, _ones()))
    self.assertEqual(_attrs(ops, "stablehlo.while"), [{}])
    self.assertEqual(
        _only(ops, "stablehlo.dynamic_update_slice"),
        {_DISJOINT: "true", "ColorInputToHBM": "false"},
    )
    self.assertEqual(_only(ops, "stablehlo.dynamic_slice"), {})

  def test_block_sites_see_enclosing_params_and_facts(self):
    params = _Params(read_tag="r", write_tag="w")
    ops = _ops(_lower(_copy_with(params), _ones()))
    self.assertEqual(_only(ops, "stablehlo.dynamic_slice"), {"test_read": "r"})
    self.assertEqual(
        _only(ops, "stablehlo.dynamic_update_slice"),
        {"test_write": "w", _DISJOINT: "true"},
    )
    (loop_facts,) = _logged_facts()
    frame = tpu_pipeline.EnclosingPipeline(params, loop_facts)
    self.assertEqual(_logged("read"), [(frame, loop_facts)])
    self.assertEqual(_logged("write"), [(frame, loop_facts)])
    self.assertIsNone(tpu_pipeline.enclosing_pipeline())
    self.assertEqual(tpu_pipeline.enclosing_body_attributes(frame), {})

  def test_unregistered_pipeline_params_succeed_on_cpu_and_fail_on_tpu(self):
    @dataclasses.dataclass(frozen=True)
    class _Unregistered(pipeline_lib.PipelineParams):
      levels: int = 2

    program = _copy_with(_Unregistered())
    ops = _ops(_lower(program, _ones(), platform="cpu"))
    self.assertEqual(_attrs(ops, "stablehlo.while"), [{}])
    with self.assertRaisesRegex(NotImplementedError, "_Unregistered"):
      _lower(program, _ones(), platform="tpu")
    tpu_pipeline.register_pipeline_handler(_Unregistered, _ParamsHandler())
    tpu_pipeline.unregister_pipeline_handler(_Unregistered)
    with self.assertRaisesRegex(NotImplementedError, "_Unregistered"):
      _lower(program, _ones(), platform="tpu")

  def test_loop_facts(self):
    def f(x):
      def outer(i, acc):
        def inner(j, a):
          return a + jnp.sum(lax.block_slice(x, (16, 128), (j * 16, 0)))

        s = lax.pipeline(
            inner, grid=(4,), init=jnp.float32(0), compiler_params=_Params()
        )
        return lax.block_update(acc, acc[:16] + s, (i * 16, 0))

      return lax.pipeline(outer, grid=(4,), init=x, compiler_params=_Params())

    _lower(f, _ones())
    outer = tpu_pipeline.LoopFacts(
        trip_count=4,
        has_nested_loop=True,
        has_block_write=True,
        pipeline_depth=0,
        num_consts=1,
        num_coords=1,
        num_carries=1,
    )
    inner = dataclasses.replace(
        outer, has_nested_loop=False, has_block_write=False, pipeline_depth=1
    )
    self.assertEqual(_logged_facts(), [outer, inner])

    _Params.log.clear()
    _lower(
        lambda x, y: lax.pipeline(
            lambda i, j, c: (
                c[0] + lax.block_slice(x, (8, 16), (i * 8, j * 16)) * y,
                c[1] + 1,
            ),
            grid=(4, 3),
            init=(jnp.zeros((8, 16), jnp.float32), jnp.int32(0)),
            compiler_params=_Params(),
        ),
        jnp.ones((32, 48), jnp.float32),
        jnp.ones((8, 16), jnp.float32),
    )
    (facts,) = _logged_facts()
    self.assertEqual(
        (facts.num_consts, facts.num_coords, facts.num_carries), (2, 2, 2)
    )

  @parameterized.named_parameters(
      ("while_and_dus", "while_dus", None, True, True),
      ("scan", "scan", None, True, False),
      ("pallas_call", "opaque", "pallas_call", False, False),
      ("mpmd_map", "opaque", "mpmd_map", False, False),
  )
  def test_loop_facts_body_inspection(
      self, kind, kernel_name, want_nested, want_write
  ):
    x = _ones()
    if kind == "while_dus":
      f = lambda y: lax.pipeline(
          lambda i, acc: lax.dynamic_update_slice(
              lax.fori_loop(0, i, lambda k, a: a + 1.0, acc),
              y[:1],
              (i, jnp.int32(0)),
          ),
          grid=(4,),
          init=y,
          compiler_params=_Params(),
      )
      _lower(f, x)
      (facts,) = _logged_facts()
    elif kind == "scan":
      f = lambda y: lax.pipeline(
          lambda i, acc: lax.fori_loop(0, 4, lambda k, a: a * 2.0, acc)
          + lax.block_slice(y, (16, 128), (i * 16, 0)),
          grid=(4,),
          init=_tile(),
          compiler_params=_Params(),
      )
      eqn = _pipeline_eqn(jax.make_jaxpr(f)(x).jaxpr)
      self.assertIn(
          "scan", [e.primitive.name for e in eqn.params["jaxpr"].eqns]
      )
      _lower(f, x)
      (facts,) = _logged_facts()
    else:
      kernel_p = core.Primitive(kernel_name)
      kernel_p.def_abstract_eval(lambda y, *, jaxpr: y)
      inner_jaxpr = jax.make_jaxpr(
          lambda t: lax.dynamic_update_slice(
              lax.fori_loop(0, 4, lambda k, a: a + jnp.sum(t), t),
              t[:1],
              (jnp.int32(0), jnp.int32(0)),
          )
      )(_tile()).jaxpr
      outer_jaxpr = jax.make_jaxpr(
          lambda: lax.pipeline(
              lambda i, acc: acc
              + kernel_p.bind(
                  lax.block_slice(x, (16, 128), (i * 16, 0)), jaxpr=inner_jaxpr
              ),
              grid=(4,),
              init=_tile(),
          )
      )().jaxpr
      facts = tpu_pipeline.loop_facts(
          _pipeline_eqn(outer_jaxpr).params["jaxpr"], (4,), 1
      )
      with pipeline_lib.checking_pipeline_bodies():
        bad_inner = jax.make_jaxpr(
            lambda t: _bind_block_slice(t, jnp.int32(0))
        )(x)
        bad_body = jax.make_jaxpr(
            lambda t: kernel_p.bind(t, jaxpr=bad_inner.jaxpr)
        )(x).jaxpr
      with self.assertRaisesRegex(ValueError, "directly in the body"):
        pipeline_lib.check_body_only_placement(bad_body)

    self.assertEqual(facts.has_nested_loop, want_nested)
    self.assertEqual(facts.has_block_write, want_write)

  def test_jitted_helper_lowered_in_and_out_of_body(self):
    # The same jitted helper lowers once at top level and once in a body.
    # Only the body copy carries the body keys.
    x = jnp.ones((16, 128), jnp.float32)
    helper = jax.jit(lambda t: jnp.sin(t) * 3.0)

    def f(x):
      top = helper(x)

      def body(i, acc):
        del i
        return acc + helper(lax.block_slice(x, (16, 128), (0, 0)))

      looped = lax.pipeline(
          body,
          grid=(4,),
          init=jnp.zeros_like(x),
          compiler_params=_Params(vmem_kib=512),
      )
      return top, looped

    ops = _ops(_lower(f, x))
    sines = _attrs(ops, "stablehlo.sine")
    self.assertLen(sines, 2)
    for attrs in sines:
      self.assertNotIn(_VMEM, attrs)
    in_body = _attrs(ops, "func.call", _in_while_body)
    at_top = _attrs(ops, "func.call", lambda p: not _in_while_body(p))
    self.assertEqual([a.get(_VMEM) for a in in_body], ["512"])
    self.assertNotIn("512", [a.get(_VMEM) for a in at_top])
    callees = {
        ir.FlatSymbolRefAttr(o.op.attributes["callee"]).value
        for o in ops
        if o.name == "func.call"
    }
    self.assertLen(callees, 2)

  def test_verify_body_gets_the_body_jaxpr(self):
    _lower(_copy_with(_Params()), _ones())
    ((jaxpr, facts),) = _logged("verify")
    # The body is verified with the facts its loop attributes get.
    self.assertEqual(facts, _logged_facts()[0])
    self.assertIn(
        pipeline_lib.block_update_p, [e.primitive for e in jaxpr.eqns]
    )

  def test_color_to_vmem_false_on_reductions_in_pipelined_body(self):
    reduce_tags = lambda grid, params: [
        t
        for n, t, _ in self._reduction_tags(grid=grid, params=params)
        if n == _REDUCE
    ]
    self.assertEqual(reduce_tags(4, _Params()), ["false"])
    self.assertEqual(reduce_tags(2, _Params()), [None])
    self.assertEqual(reduce_tags(4, None), [None])

  def _reduction_tags(self, loop_attrs=None, *, grid=4, params=None):
    """Returns (op name, ColorToVMEM, path) of each reduction related op."""
    if loop_attrs is not None:
      params = _Params(loop_attrs=tuple(loop_attrs.items()))
    f = lambda x: lax.pipeline(
        lambda i, acc: acc
        + jnp.max(
            lax.block_slice(x, (16, 128), (i * 16, 0)), axis=1, keepdims=True
        ),
        grid=(grid,),
        init=jnp.zeros((16, 1), jnp.float32),
        compiler_params=params,
    )
    return [
        (o.name, o.attrs.get("ColorToVMEM"), o.path)
        for o in _ops(_lower(f, _ones()))
        if o.name in (
            _REDUCE,
            "stablehlo.broadcast_in_dim",
            "stablehlo.reshape",
        )
    ]

  # ColorToMemPass pins the body of a while that has any of these keys,
  # whatever their values and the pipelining levels.
  @parameterized.named_parameters(
      (
          "copies_true_levels_0",
          {
              "xla_disable_while_loop_copies": "true",
              "xla_while_loop_async_pipelining_levels": "0",
          },
      ),
      ("copies_false", {"xla_disable_while_loop_copies": "false"}),
      ("disjoint", {_DISJOINT: "true"}),
      ("dce", {"xla_disable_while_loop_dce": "true"}),
  )
  def test_color_to_vmem_on_color_to_mem_pass_trigger(self, loop_attrs):
    tags = self._reduction_tags(loop_attrs)
    self.assertEqual([t for n, t, _ in tags if n == _REDUCE], ["false"])

  @parameterized.named_parameters(
      ("no_keys", {}),
      ("levels_only", {"xla_while_loop_async_pipelining_levels": "2"}),
  )
  def test_no_color_to_vmem_without_trigger(self, loop_attrs):
    tags = self._reduction_tags(loop_attrs)
    self.assertEqual([t for _, t, _ in tags], [None] * len(tags))

  def test_nested_pipeline_inherits_outer_body_keys(self):
    inner = lambda x, acc: lax.pipeline(
        lambda j, a: a + lax.block_slice(x, (16, 128), (j * 16, 0)),
        grid=(4,),
        init=acc,
        compiler_params=_Params(vmem_kib=64),
    )
    f = lambda x: lax.pipeline(
        lambda i, acc: inner(x, acc),
        grid=(2,),
        init=_tile(),
        compiler_params=_Params(vmem_kib=512),
    )
    ops = _ops(_lower(f, _ones()))
    outer_w, inner_w = [o for o in ops if o.name == "stablehlo.while"]
    self.assertEqual(outer_w.attrs, {})
    self.assertEqual(
        inner_w.attrs,
        {
            _VMEM: "512",
            "xla_disable_while_loop_copies": "true",
            "xla_while_loop_async_pipelining_levels": "2",
        },
    )
    self.assertTrue(_in_while_body(inner_w.path))
    while_regions = lambda o: [r for n, r in o.path if n == "stablehlo.while"]
    cond_vmem = {
        o.attrs.get(_VMEM)
        for o in ops
        if while_regions(o) == [1, 0] and o.name not in _BODY_SKIP
    }
    self.assertEqual(cond_vmem, {"512"})
    self.assertEqual(
        _attrs(ops, "stablehlo.dynamic_slice", lambda p: _in_while_body(p, 2)),
        [{_VMEM: "64"}],
    )

  def test_lowerings_are_not_cached_across_pipelines(self):
    def f(x):
      a = _tiled_copy(x, compiler_params=_Params(write_tag="a"))
      b = _tiled_copy(x, compiler_params=_Params(write_tag="b"))
      return a, b

    tags = [
        a.get("test_write")
        for a in _attrs(
            _ops(_lower(f, _ones())), "stablehlo.dynamic_update_slice"
        )
    ]
    self.assertEqual(tags, ["a", "b"])


class TpuFrontendAttrsTest(jtu.JaxTestCase):

  def test_attach_and_read_frontend_attributes(self):
    with mlir.make_ir_context(), ir.Location.unknown():
      module = ir.Module.create()
      with ir.InsertionPoint(module.body):
        x = mlir.ir_constant(np.float32(1))
        op = hlo.AddOp(x, x).operation
        self.assertEqual(tpu_attrs.frontend_attributes(op), {})
        tpu_attrs.attach_frontend_attributes(op, {"a": "1"})
        tpu_attrs.attach_frontend_attributes(op, {"b": "2", "a": "3"})
        tpu_attrs.attach_frontend_attributes(op, {})
        self.assertEqual(
            tpu_attrs.frontend_attributes(op), {"a": "3", "b": "2"}
        )
        tpu_attrs.attach_missing_frontend_attributes(op, {"a": "4", "c": "5"})
        self.assertEqual(
            tpu_attrs.frontend_attributes(op), {"a": "3", "b": "2", "c": "5"}
        )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
