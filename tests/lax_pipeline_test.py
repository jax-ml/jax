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

"""Tests for lax.pipeline, lax.block_slice and lax.block_update."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import core
from jax._src import test_util as jtu
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow import pipeline as pipeline_lib
from jax._src.lib.mlir import ir
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class _Params(pipeline_lib.PipelineParams):
  """Test only pipeline params without a registered platform handler."""

  levels: int = 2


@dataclasses.dataclass(frozen=True)
class _FusionOnlyParams:
  """Params for another primitive, not subclassing `PipelineParams`."""

  def fusion_call_attributes(self, ctx: Any) -> Mapping[str, str]:
    del ctx
    return {}


def _cpu_lower(f, *args):
  return jax.jit(f).trace(*args).lower(lowering_platforms=("cpu",))


def _stablehlo_ops(lowered) -> list[tuple[str, dict[str, str]]]:
  """Every op name and `mhlo.frontend_attributes` dict of `lowered`."""
  module = lowered.compiler_ir("stablehlo")
  out = []

  def visit(op):
    for region in op.regions:
      for block in region.blocks:
        for child in block.operations:
          c = child.operation
          raw = c.attributes.get("mhlo.frontend_attributes")
          attrs = {}
          if raw is not None:
            attrs = {
                a.name: ir.StringAttr(a.attr).value for a in ir.DictAttr(raw)
            }
          out.append((c.name, attrs))
          visit(c)

  with module.context:
    visit(module.operation)
  return out


def _pipeline_eqn(jaxpr: core.Jaxpr) -> core.JaxprEqn:
  (eqn,) = (e for e in jaxpr.eqns if e.primitive is pipeline_lib.pipeline_p)
  return eqn


def _ones(shape=(64, 128)):
  return jnp.ones(shape, jnp.float32)


def _tile():
  return jnp.zeros((16, 128), jnp.float32)


def _tiled_copy(x, tile=16, **kw):
  def body(i, acc):
    t = lax.block_slice(x, (tile, x.shape[1]), (i * tile, 0))
    return lax.block_update(acc, t * 2.0, (i * tile, 0))

  return lax.pipeline(
      body, grid=(x.shape[0] // tile,), init=jnp.zeros_like(x), **kw
  )


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


class PipelineValueTest(jtu.JaxTestCase):

  def test_tiled_copy(self):
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    self.assertAllClose(jax.jit(_tiled_copy)(x), x * 2.0)
    self.assertAllClose(_tiled_copy(x), x * 2.0)

  def test_tiled_accumulate_matches_fori_loop(self):
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)

    def f(x):
      def body(i, acc):
        return acc + lax.block_slice(x, (16, 128), (i * 16, 0))

      return lax.pipeline(body, grid=(4,), init=_tile())

    def ref(x):
      def body(i, acc):
        return acc + lax.dynamic_slice(x, (i * 16, 0), (16, 128))

      return lax.fori_loop(
          jnp.int32(0), jnp.int32(4), body, jnp.zeros((16, 128), jnp.float32)
      )

    self.assertAllClose(jax.jit(f)(x), ref(x))

  def test_nd_grid_coordinates_are_row_major(self):
    x = jnp.arange(4 * 6 * 8, dtype=jnp.float32).reshape(4 * 2, 6 * 4)

    def f(x):
      def body(i, j, carry):
        out, order, step = carry
        t = lax.block_slice(x, (2, 4), (i * 2, j * 4))
        out = lax.block_update(out, t + 1.0, (i * 2, j * 4))
        order = order.at[step].set(i * 10 + j)
        return out, order, step + 1

      init = (jnp.zeros_like(x), jnp.zeros((24,), jnp.int32), jnp.int32(0))
      return lax.pipeline(body, grid=(4, 6), init=init)

    out, order, steps = jax.jit(f)(x)
    self.assertAllClose(out, x + 1.0)
    expected_order = (
        np.arange(4, dtype=np.int32)[:, None] * 10
        + np.arange(6, dtype=np.int32)
    ).ravel()
    self.assertArraysEqual(order, expected_order)
    self.assertEqual(int(steps), 24)

  def test_multiple_carries_as_pytree(self):
    x = jnp.arange(32 * 128, dtype=jnp.float32).reshape(32, 128)

    def f(x):
      def body(i, acc):
        t = lax.block_slice(x, (8, 128), (i * 8, 0))
        return {
            "sum": acc["sum"] + t,
            "max": jnp.maximum(acc["max"], t),
            "n": acc["n"] + 1,
        }

      init = {
          "sum": jnp.zeros((8, 128), jnp.float32),
          "max": jnp.full((8, 128), -jnp.inf, jnp.float32),
          "n": jnp.int32(0),
      }
      return lax.pipeline(body, grid=4, init=init)

    out = jax.jit(f)(x)
    tiles = x.reshape(4, 8, 128)
    self.assertAllClose(out["sum"], tiles.sum(0))
    self.assertAllClose(out["max"], tiles.max(0))
    self.assertEqual(int(out["n"]), 4)

  def test_zero_trip_grid_returns_init(self):
    x = jnp.ones((8, 128), jnp.float32)
    out = jax.jit(
        lambda x: lax.pipeline(
            lambda i, acc: acc + lax.block_slice(x, (8, 128), (i, 0)),
            grid=(0,),
            init=x * 5.0,
        )
    )(x)
    self.assertAllClose(out, x * 5.0)

  def test_negative_index_follows_dynamic_slice(self):
    x = jnp.arange(16, dtype=jnp.float32).reshape(16, 1)

    def f(x):
      def body(i, acc):
        del i
        return lax.block_update(
            acc, lax.block_slice(x, (4, 1), (-4, 0)), (-8, 0)
        )

      return lax.pipeline(body, grid=(1,), init=jnp.zeros_like(x))

    ref = lax.dynamic_update_slice(
        jnp.zeros_like(x), lax.dynamic_slice(x, (-4, 0), (4, 1)), (-8, 0)
    )
    self.assertAllClose(jax.jit(f)(x), ref)

  def test_nested_pipeline(self):
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

    def f(x):
      def outer(i, acc):
        def inner(j, acc2):
          t = lax.block_slice(x, (1, 128), (i * 4 + j, 0))
          return lax.block_update(acc2, t, (i * 4 + j, 0))

        return lax.pipeline(inner, grid=(4,), init=acc)

      return lax.pipeline(outer, grid=(2,), init=jnp.zeros_like(x))

    self.assertAllClose(jax.jit(f)(x), x)

  def test_jaxpr_layout_is_consts_coords_carries(self):
    x = jnp.ones((32, 128), jnp.float32)
    y = jnp.ones((8, 128), jnp.float32)

    def f(x, y):
      def body(i, j, acc):
        return acc + lax.block_slice(x, (8, 128), (i * 8 + j, 0)) * y

      return lax.pipeline(
          body, grid=(2, 2), init=jnp.zeros((8, 128), jnp.float32)
      )

    closed = jax.make_jaxpr(f)(x, y)
    eqn = _pipeline_eqn(closed.jaxpr)
    self.assertEqual(eqn.params["grid"], (2, 2))
    self.assertEqual(eqn.params["num_consts"], 2)
    self.assertIsNone(eqn.params["compiler_params"])
    body = eqn.params["jaxpr"]
    self.assertEmpty(body.consts)
    avals = [v.aval for v in body.invars]
    self.assertEqual({a.shape for a in avals[:2]}, {(32, 128), (8, 128)})
    self.assertEqual([a.shape for a in avals[2:]], [(), (), (8, 128)])
    self.assertEqual(avals[2].dtype, np.int32)
    self.assertEqual(avals[3].dtype, np.int32)
    prims = [e.primitive for e in body.eqns]
    self.assertIn(pipeline_lib.block_slice_p, prims)
    self.assertEqual(set(eqn.invars[:2]), set(closed.jaxpr.invars))

  def test_zero_dimensional_grid_runs_once(self):
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

    def f(x):
      def body(acc):
        return lax.block_update(
            acc, lax.block_slice(x, (8, 128), (0, 0)) + 1.0, (0, 0)
        )

      return lax.pipeline(body, grid=(), init=jnp.zeros_like(x))

    self.assertAllClose(jax.jit(f)(x), x + 1.0)

  def test_public_surface(self):
    for name in ("pipeline", "block_slice", "block_update"):
      self.assertTrue(callable(getattr(lax, name)), name)
    for name in ("pipeline_p", "block_slice_p", "block_update_p"):
      self.assertFalse(hasattr(lax, name), name)
    from jax._src.lax import control_flow  # pylint: disable=g-import-not-at-top

    self.assertIs(control_flow.pipeline, pipeline_lib)
    self.assertIsInstance(pipeline_lib, type(jax))

  def test_effectful_pipeline_is_kept(self):
    def body(i, acc):
      jax.debug.print("step {}", i)
      return acc + 1.0

    jaxpr = jax.make_jaxpr(lambda x: lax.pipeline(body, grid=(2,), init=x))(
        jnp.float32(0)
    ).jaxpr
    dced, _ = pe.dce_jaxpr(jaxpr, [False])
    self.assertIn(pipeline_lib.pipeline_p, [e.primitive for e in dced.eqns])


class PipelineAutodiffTest(jtu.JaxTestCase):
  """grad and jvp work while no tangent reaches the pipeline."""

  def test_grad_and_jvp_without_tangents_into_pipeline(self):
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    y = jnp.full((3,), 2.0, jnp.float32)

    def f(y):
      return jnp.sum(_tiled_copy(x)) + jnp.sum(y**2)

    self.assertAllClose(jax.grad(f)(y), 2.0 * y)
    _, tangent = jax.jvp(f, (y,), (jnp.ones_like(y),))
    self.assertAllClose(tangent, jnp.sum(2.0 * y))

  @parameterized.product(through=("init", "const"), transform=("grad", "jvp"))
  def test_tangent_into_pipeline_raises(self, through, transform):
    def f(x):
      if through == "init":
        out = lax.pipeline(lambda i, acc: acc * 2.0, grid=(4,), init=x)
      else:
        out = _tiled_copy(x)
      return jnp.sum(out)

    x = _ones()
    with self.assertRaisesRegex(
        NotImplementedError,
        "Differentiation rule for 'pipeline' not implemented",
    ):
      _ = jax.grad(f)(x) if transform == "grad" else jax.jvp(f, (x,), (x,))


class PipelineTransformationTest(jtu.JaxTestCase):

  def test_vmap_is_not_implemented(self):
    xs = jnp.ones((3, 64, 128), jnp.float32)
    with self.assertRaisesRegex(NotImplementedError, "pipeline"):
      jax.vmap(_tiled_copy)(xs)


class PipelineLoweringTest(jtu.JaxTestCase):

  def test_cpu_lowering_is_a_plain_loop(self):
    params = _Params(levels=3)
    ops = _stablehlo_ops(
        _cpu_lower(lambda x: _tiled_copy(x, compiler_params=params), _ones())
    )
    names = {name for name, _ in ops}
    self.assertContainsSubset(
        {
            "stablehlo.while",
            "stablehlo.dynamic_slice",
            "stablehlo.dynamic_update_slice",
        },
        names,
    )
    self.assertEqual([attrs for _, attrs in ops if attrs], [])

  @parameterized.named_parameters(
      *[(f"cpu_{name}", program) for name, program in _MISPLACED_BLOCK_OPS]
  )
  def test_misplaced_block_op_fails_to_lower(self, program):
    with (
        jax.enable_checks(False),
        self.assertRaisesRegex(ValueError, "directly in the body"),
    ):
      _cpu_lower(program, _ones())


def _read(k):
  return lax.block_slice(_ones(), (16, 128), (k, 0))


def _read_in_cond(i, acc):
  return lax.cond(i > 0, lambda: acc + _read(0), lambda: acc)


def _read_in_fori_loop(i, acc):
  del i
  return lax.fori_loop(0, 2, lambda k, a: a + _read(k), acc)


def _read_in_jit(i, acc):
  return jax.jit(lambda: acc + _read(i))()


# A wrong type on purpose.
_NOT_BOOL: Any = None


def _scalar_pipeline(body, grid=(2,), **kw):
  return lax.pipeline(body, grid=grid, init=jnp.zeros((), jnp.float32), **kw)


class PipelineTypecheckTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      (
          "block_slice",
          lambda: jax.jit(lambda x: lax.block_slice(x, (8, 128), (0, 0)))(
              _ones((16, 128))
          ),
      ),
      (
          "block_update",
          lambda: lax.block_update(_ones((16, 128)), _ones((8, 128)), (0, 0)),
      ),
  )
  def test_block_op_outside_pipeline(self, f):
    with self.assertRaisesRegex(TypeError, "directly in the body"):
      f()

  @parameterized.named_parameters(
      ("cond", _read_in_cond),
      ("fori_loop", _read_in_fori_loop),
      ("jit", _read_in_jit),
  )
  def test_block_op_nested_within_body(self, body):
    with self.assertRaisesRegex(TypeError, "directly in the body"):
      lax.pipeline(body, grid=(4,), init=_tile())

  def test_jaxpr_typecheck_rejects_nested_block_op(self):
    def body(i, acc):
      return lax.cond(
          i > 0, lambda: acc + _bind_block_slice(_ones(), i), lambda: acc
      )

    with jax.enable_checks(False):
      jaxpr = jax.make_jaxpr(
          lambda: lax.pipeline(body, grid=(4,), init=_tile())
      )()
    with self.assertRaisesRegex(core.JaxprTypeError, "directly in the body"):
      core.check_jaxpr(jaxpr.jaxpr)

  def test_jaxpr_typecheck_accepts_valid_pipeline(self):
    core.check_jaxpr(jax.make_jaxpr(_tiled_copy)(_ones()).jaxpr)

  @parameterized.named_parameters(
      ("slice_index", _bind_slice, "index"),
      (
          "slice_rank",
          lambda x, z: _bind_slice(x, z, z, block_shape=(16, 128, 1)),
          "block_shape",
      ),
      (
          "slice_list",
          lambda x, z: _bind_slice(x, z, z, block_shape=[16, 128]),
          "block_shape",
      ),
      (
          "slice_revisiting",
          lambda x, z: _bind_slice(x, z, z, revisits_tile="no"),
          "revisits_tile",
      ),
      # Not 1: bind caches abstract evaluation by params, and 1 == True.
      (
          "update_disjoint",
          lambda x, z: pipeline_lib.block_update_p.bind(
              x, x[:16], z, z, disjoint="yes"
          ),
          "disjoint",
      ),
  )
  def test_block_primitive_abstract_eval_checks(self, bind, msg):
    with self.assertRaisesRegex(TypeError, msg), jax.enable_checks(False):
      jax.make_jaxpr(lambda: bind(_ones(), jnp.int32(0)))()

  @parameterized.named_parameters(
      ("too_large", (2**16, 2**16), ValueError, "int32"),
      ("negative", (-1,), ValueError, "grid"),
      ("float", 4.0, TypeError, "grid"),
      ("float_entry", (2, 2.5), TypeError, "grid"),
      ("string", "4", TypeError, "grid"),
      ("bool", (True,), TypeError, "grid"),
      ("none", None, TypeError, "grid"),
      ("array_entry", (np.float32(2),), TypeError, "grid"),
      ("rank_mismatch", (2, 2), TypeError, ""),
  )
  def test_bad_grid(self, grid, error, msg):
    with self.assertRaisesRegex(error, msg):
      _scalar_pipeline(lambda i, acc: acc, grid=grid)

  def test_traced_grid(self):
    with self.assertRaisesRegex(TypeError, "grid"):
      jax.jit(lambda n: _scalar_pipeline(lambda i, acc: acc, grid=(n,)))(4)

  @parameterized.named_parameters(
      ("smaller", lambda i, acc: acc[:8]),
      ("dtype", lambda i, acc: acc.astype(jnp.int32)),
      ("structure", lambda i, acc: (acc, acc)),
  )
  def test_carry_type_mismatch(self, body):
    with self.assertRaisesRegex(TypeError, "carry"):
      lax.pipeline(body, grid=(4,), init=_ones())

  def test_bad_compiler_params_at_bind(self):
    with self.assertRaisesRegex(TypeError, "PipelineParams"):
      _scalar_pipeline(lambda i, acc: acc, compiler_params=_FusionOnlyParams())

  @parameterized.named_parameters(
      (
          "slice_too_large",
          lambda acc: acc + lax.block_slice(_ones(), (128, 128), (0, 0)),
          TypeError,
      ),
      (
          "slice_short_index",
          lambda acc: acc + lax.block_slice(_ones(), (16, 128), (0,)),
          ValueError,
      ),
      (
          "slice_float_index",
          lambda acc: acc + lax.block_slice(_ones(), (16, 128), (0.5, 0)),
          TypeError,
      ),
      (
          "slice_revisiting",
          lambda acc: acc
          + lax.block_slice(
              _ones(), (16, 128), (0, 0), revisits_tile=_NOT_BOOL
          ),
          TypeError,
      ),
      (
          "update_disjoint",
          lambda acc: lax.block_update(
              acc, acc[:, :8], (0, 0), disjoint=_NOT_BOOL
          ),
          TypeError,
      ),
  )
  def test_bad_block_arguments(self, body, error):
    with self.assertRaises(error):
      lax.pipeline(lambda i, acc: body(acc), grid=(4,), init=_tile())

  def test_body_effects_are_rejected(self):
    x = jnp.zeros((), jnp.float32)

    def ordered_body(i, acc):
      jax.debug.print("{}", i, ordered=True)
      return acc

    with self.assertRaisesRegex(NotImplementedError, "pipeline"):
      jax.jit(lambda: _scalar_pipeline(ordered_body))()

    body = jax.make_jaxpr(ordered_body)(jnp.int32(0), x).jaxpr
    with self.assertRaisesRegex(NotImplementedError, "Ordered effects"):
      pipeline_lib.pipeline_p.bind(
          x, grid=(2,), num_consts=0, jaxpr=body, compiler_params=None
      )
    jaxpr = jax.make_jaxpr(
        lambda y: lax.pipeline(lambda i, a: a, grid=(2,), init=y)
    )(x).jaxpr
    (eqn,) = jaxpr.eqns
    bad = jaxpr.replace(
        eqns=[
            eqn.replace(
                params={**eqn.params, "jaxpr": body}, effects=body.effects
            )
        ],
        effects=body.effects,
    )
    with self.assertRaisesRegex(core.JaxprTypeError, "Ordered effects"):
      core.check_jaxpr(bad)

    ref = jax.new_ref(x)

    def ref_body(i, acc):
      del i
      ref[...] = acc
      return acc

    with self.assertRaisesRegex(NotImplementedError, "Ref effects"):
      _scalar_pipeline(ref_body)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
