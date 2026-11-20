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

from __future__ import annotations

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import reduction_order as ro
from jax._src.interpreters import partial_eval as pe
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


def _eqns_by_name(closed_jaxpr, name: str):
  jaxpr = closed_jaxpr.jaxpr if hasattr(closed_jaxpr, "jaxpr") else closed_jaxpr
  return [e for e in jtu.iter_eqns(jaxpr) if e.primitive.name == name]

def _lo_eqns(closed_jaxpr, name: str):
  return _eqns_by_name(pe.lower_jaxpr2(closed_jaxpr.jaxpr), name)


_MAT_DNUMS = (((1,), (0,)), ((), ()))
_REASSOC_TOL = {
    np.float16: 1e-2,
    jnp.bfloat16: 2e-2,
    np.float32: 1e-4,
    np.float64: 1e-12,
}

_REDUCTION_OP_CASES = [
    ("jnp_sum", "reduce_sum", lambda a, b, **kw: jnp.sum(a, axis=1, **kw)),
    ("jnp_max", "reduce_max", lambda a, b, **kw: jnp.max(a, axis=1, **kw)),
    ("jnp_dot", "dot_general", lambda a, b, **kw: jnp.dot(a, b, **kw)),
    ("jnp_matmul", "dot_general", lambda a, b, **kw: jnp.matmul(a, b, **kw)),
    ("jnp_einsum_sum", "reduce_sum", lambda a, b, **kw: jnp.einsum("ij->i", a, **kw)),
    ("jnp_einsum_dot", "dot_general", lambda a, b, **kw: jnp.einsum("ik,kj->ij", a, b, **kw)),
    ("lax_reduce_sum", "reduce_sum", lambda a, b, **kw: lax.reduce_sum(a, (1,), **kw)),
    ("lax_reduce_max", "reduce_max", lambda a, b, **kw: lax.reduce_max(a, (1,), **kw)),
    ("lax_dot", "dot_general", lambda a, b, **kw: lax.dot(a, b, **kw)),
    ("lax_dot_general", "dot_general", lambda a, b, **kw: lax.dot_general(a, b, _MAT_DNUMS, **kw)),
]


class ReductionOrderTest(jtu.JaxTestCase):

  @parameterized.parameters(
      (None, None),
      ("unordered", ro.ReductionOrder.unordered),
      ("linear", ro.ReductionOrder.linear),
      ("unreduced", ro.ReductionOrder.unreduced),
      (ro.Intrinsic("tpu_xlane_128"), ro.Intrinsic("tpu_xlane_128")),
      (ro.Unordered(), ro.Unordered(None)),
      (ro.Unordered("xla"), ro.Unordered("xla")),
      (
          ro.tiled(4, "linear", "unordered", split="lanes", pad="absent"),
          ro.Tile(
              4, ro.ReductionOrder.linear, ro.ReductionOrder.unordered,
              ro.Split.strided, ro.PadVal.absent,
          ),
      ),
      (
          ro.padded(16, "linear"),
          ro.Pad(16, ro.ReductionOrder.linear, ro.PadVal.zero),
      ),
      (
          ro.cast("bfloat16", "linear"),
          ro.Cast(np.dtype(jnp.bfloat16), ro.ReductionOrder.linear),
      ),
      (ro.ring(), ro.Ring(ro.RingDir.bidir, split_axis=None)),
      (ro.ring("bidi", split_axis=0), ro.Ring(ro.RingDir.bidir, split_axis=0)),
  )
  def test_normalize_order_valid(self, order, expected):
    self.assertEqual(ro.normalize_order(order), expected)
    self.assertEqual(hash(ro.normalize_order(order)), hash(expected))

  def test_enums_and_leaf_classification(self):
    self.assertEqual(repr(ro.ReductionOrder.linear), "linear")
    self.assertEqual(repr(ro.RingDir.bidir), "bidir")
    self.assertNotEqual(ro.Unordered("xla"), ro.Unordered("mosaic"))
    self.assertTrue(ro.is_leaf_order(ro.ReductionOrder.linear))
    self.assertFalse(ro.is_leaf_order(ro.ReductionOrder.unreduced))
    self.assertFalse(ro.is_leaf_order(ro.tiled(4)))
    with self.assertRaises(TypeError):
      getattr(ro, "ring")("asc", 0, "linear")

  @parameterized.parameters(
      ("",), ("tree",), (123,), (["linear"],),
      (lambda: ro.Intrinsic(""),), (lambda: ro.Intrinsic(42),),
      (lambda: ro.Unordered(""),), (lambda: ro.Unordered(123),),
      (lambda: ro.tiled(4, split="diagonal"),),
      (lambda: ro.tiled(4, pad="bogus"),),
      (lambda: ro.padded(4, pad=None),),
      (lambda: ro.ring("clockwise"),),
  )
  def test_normalize_order_invalid(self, bad_order):
    with self.assertRaises(ValueError):
      ro.normalize_order(bad_order() if callable(bad_order) else bad_order)

  @parameterized.named_parameters(*_REDUCTION_OP_CASES)
  def test_jaxpr_default_omits_reduction_order(self, prim, op_fn):
    rng = jtu.rand_default(self.rng())
    x = rng((4, 8), np.float32)
    w = rng((8, 3), np.float32)
    eqns = _eqns_by_name(jax.make_jaxpr(op_fn)(x, w), prim)
    self.assertLen(eqns, 1)
    self.assertNotIn("reduction_order", eqns[0].params)

  @parameterized.product(
      op_case=_REDUCTION_OP_CASES,
      reduction_order=[
          "unordered",
          "linear",
          ro.ReductionOrder.unordered,
          ro.ReductionOrder.linear,
          ro.Intrinsic("test_hw"),
          ro.Unordered(),
          ro.Unordered("xla"),
          ro.Unordered("mosaic"),
      ],
  )
  def test_jaxpr_explicit_reduction_order_param(self, op_case, reduction_order):
    _, prim, op_fn = op_case
    rng = jtu.rand_default(self.rng())
    x = rng((4, 8), np.float32)
    w = rng((8, 3), np.float32)
    eqn, = _eqns_by_name(
        jax.make_jaxpr(lambda a, b: op_fn(a, b, reduction_order=reduction_order))(x, w),
        prim,
    )
    self.assertEqual(
        eqn.params["reduction_order"], ro.normalize_order(reduction_order))
    self.assertIsInstance(
        eqn.params["reduction_order"],
        (ro.ReductionOrder, ro.Intrinsic, ro.Unordered),
    )

  @jtu.sample_product(
      op_case=_REDUCTION_OP_CASES,
      reduction_order=[
          "unordered",
          ro.Unordered(),
          ro.Unordered("xla"),
          ro.Unordered("mosaic"),
      ],
      dtype=jtu.dtypes.floating,
  )
  def test_unordered_lowering_matches_default(self, op_case, reduction_order, dtype):
    _, _, op_fn = op_case
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((4, 8), dtype), rng((8, 3), dtype)]
    fn = lambda a, b: op_fn(a, b, reduction_order=reduction_order)
    ref_fn = lambda a, b: op_fn(a, b)
    a, b = args_maker()
    self.assertAllClose(fn(a, b), ref_fn(a, b), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(fn, args_maker)

  @parameterized.product(
      op_fn=[
          lambda a: jnp.sum(a, reduction_order="linear"),
          lambda a: jnp.dot(a, jnp.ones(4, dtype=jnp.float32), reduction_order="linear"),
          lambda a: jnp.matmul(a, jnp.ones(4, dtype=jnp.float32), reduction_order="linear"),
          lambda a: jnp.einsum("i->", a, reduction_order="linear"),
          lambda a: lax.reduce_sum(a, (0,), reduction_order="linear"),
          lambda a: lax.dot(a, jnp.ones(4, dtype=jnp.float32), reduction_order="linear"),
      ],
      jit=[False, True],
  )
  def test_linear_sequential_ieee754_cancellation(self, op_fn, jit):
    # ((1 + 1e8) + 1) - 1e8 == 0.0 in f32 left-to-right, whereas reassociated is 2.0.
    v = jnp.array([1.0, 1e8, 1.0, -1e8], dtype=jnp.float32)
    expected = np.float32(0.0)
    fn = jax.jit(op_fn) if jit else op_fn
    out = np.asarray(fn(v))
    self.assertEqual(out.view(np.uint32), expected.view(np.uint32))

  @parameterized.parameters(False, True)
  def test_linear_all_neg_zero_preserves_signbit(self, jit):
    # Starting from x[0] preserves -0.0 on all -0.0 inputs.
    neg_zeros = jnp.array([-0.0, -0.0, -0.0], dtype=jnp.float32)
    fn = lambda a: jnp.sum(a, reduction_order="linear")
    out = np.asarray((jax.jit(fn) if jit else fn)(neg_zeros))
    self.assertTrue(np.signbit(out))

  @jtu.sample_product(
      [
          dict(
              op_fn=lambda e, z: jnp.sum(e, axis=1, reduction_order="linear"),
              expected_shape=(2,),
          ),
          dict(
              op_fn=lambda e, z: jnp.dot(e, z, reduction_order="linear"),
              expected_shape=(2, 3),
          ),
          dict(
              op_fn=lambda e, z: jnp.matmul(e, z, reduction_order="linear"),
              expected_shape=(2, 3),
          ),
          dict(
              op_fn=lambda e, z: jnp.einsum("ik,kj->ij", e, z, reduction_order="linear"),
              expected_shape=(2, 3),
          ),
          dict(
              op_fn=lambda e, z: lax.reduce_sum(e, (1,), reduction_order="linear"),
              expected_shape=(2,),
          ),
          dict(
              op_fn=lambda e, z: lax.dot(e, z, reduction_order="linear"),
              expected_shape=(2, 3),
          ),
      ],
      dtype=jtu.dtypes.floating,
  )
  def test_linear_empty_reduction_returns_pos_zero(self, op_fn, expected_shape, dtype):
    args_maker = lambda: [jnp.zeros((2, 0), dtype=dtype), jnp.zeros((0, 3), dtype=dtype)]
    e, z = args_maker()
    expected = jnp.zeros(expected_shape, dtype=dtype)
    out = op_fn(e, z)
    self.assertArraysEqual(out, expected)
    self.assertFalse(bool(jnp.any(jnp.signbit(out))))
    self._CompileAndCheck(op_fn, args_maker)

  @parameterized.named_parameters(
      ("small_unrolled_dot", (2,), (2,), jnp.dot, ()),
      ("large_loop_matmul", (64, 64, 2), (2, 2), jnp.matmul, (64, 64, 2)),
  )
  def test_linear_dot_no_fma_contraction(self, a_shape, b_shape, op_fn, out_shape):
    # 0-ULP separate-rounding IEEE-754 fl(acc + fl(a * b)) without FMA contraction (<=8KiB & >8KiB).
    u = np.float32(1.0 + 2.0**-23)
    base_a = jnp.array([u, -u], dtype=jnp.float32)
    base_b = jnp.array([u, u], dtype=jnp.float32)
    a = jnp.broadcast_to(base_a, a_shape)
    b = jnp.broadcast_to(base_b[:, None] if len(b_shape) == 2 else base_b, b_shape)
    out = jax.jit(lambda x, y: op_fn(x, y, reduction_order="linear"))(a, b)
    self.assertArraysEqual(out, jnp.zeros(out_shape, dtype=jnp.float32))

  @jtu.sample_product(
      [
          dict(shape=shape, axis=axis)
          for shape in [(5,), (3, 7), (2, 4, 5)]
          for axis in range(-len(shape), len(shape))
      ],
      op=["sum", "max"],
      dtype=jtu.dtypes.floating + jtu.dtypes.integer,
  )
  def test_linear_matches_sequential_numpy_reference(self, shape, axis, op, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    ax = axis % len(shape)
    if op == "sum":
      jnp_fn = lambda x: lax.reduce_sum(x, (ax,), reduction_order="linear")
      np_fn = lambda x: functools.reduce(np.add, np.moveaxis(np.asarray(x), ax, 0))
    else:
      jnp_fn = lambda x: lax.reduce_max(x, (ax,), reduction_order="linear")
      np_fn = lambda x: functools.reduce(np.maximum, np.moveaxis(np.asarray(x), ax, 0))
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, atol=0, rtol=0)
    self._CompileAndCheck(jnp_fn, args_maker, atol=0, rtol=0)

  @parameterized.named_parameters(
      ("sum_axis", "ij->i", "reduce_sum", False, lambda a, b: jnp.sum(a, axis=1)),
      ("dot_standard", "ik,kj->ij", "dot_general", True, lambda a, b: a @ b),
      ("dot_transposed_out", "ik,kj->ji", "dot_general", True, lambda a, b: (a @ b).T),
  )
  @jax.default_matmul_precision("float32")
  def test_einsum_single_reduction(self, subscripts, prim, binary, ref_fn):
    rng = jtu.rand_default(self.rng())
    a = rng((2, 3), np.float32)
    b = rng((3, 4), np.float32)
    args = (a, b) if binary else (a,)
    fn = lambda *xs: jnp.einsum(subscripts, *xs, reduction_order="linear")
    jaxpr = jax.make_jaxpr(fn)(*args)
    eqn, = _eqns_by_name(jaxpr, prim)
    self.assertIs(eqn.params["reduction_order"], ro.ReductionOrder.linear)
    if binary:
      # Preserves user (arg0, arg1) operand order on dot_general even when output is transposed.
      self.assertEqual(eqn.invars, list(jaxpr.jaxpr.invars))
    self.assertAllClose(fn(*args), ref_fn(a, b), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(fn, lambda: list(args))

  @parameterized.named_parameters(
      ("transpose_no_red", "ij->ji", [(2, 3)], "linear"),
      ("diag_no_red", "ii->i", [(3, 3)], "linear"),
      ("trace_repeated_idx", "ii->", [(3, 3)], "linear"),
      ("outer_product_no_red", "i,j->ij", [(3,), (4,)], "linear"),
      ("hadamard_no_red", "ij,ij->ij", [(2, 3), (2, 3)], "unordered"),
      ("three_operand_multi_red", "ab,bc,cd->ad", [(2, 3), (3, 4), (4, 2)], "linear"),
      ("contract_plus_marginal_sum", "ij,jk->i", [(2, 3), (3, 4)], "linear"),
      ("trace_plus_outer", "ii,jk->jk", [(3, 3), (2, 3)], "linear"),
  )
  def test_einsum_invalid_reduction_count_raises(self, subscripts, shapes, reduction_order):
    rng = jtu.rand_default(self.rng())
    ops = [rng(s, np.float32) for s in shapes]
    with self.assertRaisesRegex(ValueError, "reduction_order="):
      jnp.einsum(subscripts, *ops, reduction_order=reduction_order)

  def test_jvp_preserves_reduction_order(self):
    rng = jtu.rand_default(self.rng())
    x = rng((4, 8), np.float32)
    dx = rng((4, 8), np.float32)
    w = rng((8, 4), np.float32)
    jvp_jaxpr = jax.make_jaxpr(
        lambda a, da: jax.jvp(lambda z: jnp.dot(z, w, reduction_order="linear"), (a,), (da,))
    )(x, dx)
    dot_eqns = _eqns_by_name(jvp_jaxpr, "dot_general")
    self.assertLen(dot_eqns, 2)
    self.assertTrue(
        all(e.params["reduction_order"] is ro.ReductionOrder.linear
            for e in dot_eqns))

  @jax.default_matmul_precision("float32")
  def test_grad_leaves_transpose_reductions_unordered(self):
    rng = jtu.rand_default(self.rng())

    def predict(params, inputs):
      for weight, bias in params:
        outputs = jnp.dot(inputs, weight, reduction_order="linear") + bias
        inputs = jnp.maximum(outputs, 0)
      return outputs

    def loss(params, batch):
      inputs, targets = batch
      preds = predict(params, inputs)
      return jnp.sum((preds - targets) ** 2, reduction_order="linear")

    params = [(rng((8, 4), np.float32), rng((4,), np.float32)),
              (rng((4, 4), np.float32), rng((4,), np.float32))]
    batch = (rng((4, 8), np.float32), rng((4, 4), np.float32))

    with jax.numpy_rank_promotion("allow"):
      grad_jaxpr = jax.make_jaxpr(jax.grad(loss))(params, batch)
      grad_dots = _eqns_by_name(grad_jaxpr, "dot_general")
      self.assertEqual(
          [e.params["reduction_order"] for e in grad_dots[:2]],
          [ro.ReductionOrder.linear, ro.ReductionOrder.linear],
      )
      self.assertTrue(all(e.params.get("reduction_order") is None for e in grad_dots[2:]))
      grad_sums = _eqns_by_name(grad_jaxpr, "reduce_sum")
      self.assertIn(None, [e.params.get("reduction_order") for e in grad_sums])

      g = jax.jit(jax.grad(loss))(params, batch)
      pe_g = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0)))(params, batch)
      self.assertAllClose(g[0][0], jnp.sum(pe_g[0][0], axis=0), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)

  @parameterized.parameters("linear", ro.Unordered("xla"))
  def test_grad_through_max_with_reduction_order(self, reduction_order):
    # Use distinct elements so max gradient is unambiguous.
    a = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    gm = jax.grad(lambda x: jnp.sum(jnp.max(x, axis=1, reduction_order=reduction_order)))(a)
    ref_gm = jax.grad(lambda x: jnp.sum(jnp.max(x, axis=1)))(a)
    self.assertArraysEqual(gm, ref_gm)

  @parameterized.named_parameters(("sum", jnp.sum), ("max", jnp.max))
  def test_named_axis_reduction_order_not_implemented(self, op_fn):
    with self.assertRaises(NotImplementedError):
      jax.vmap(lambda z: op_fn(z, axis="x", reduction_order="linear"), axis_name="x")(jnp.ones(4))

  def test_custom_vjp_pins_backward_reduction_order(self):
    rng = jtu.rand_default(self.rng())
    x = rng((4, 8), np.float32)
    w = rng((8, 4), np.float32)

    @jax.custom_vjp
    def pinned_linear(inp, weight):
      return jnp.dot(inp, weight, reduction_order="linear")

    pinned_linear.defvjp(
        lambda inp, weight: (pinned_linear(inp, weight), (inp, weight)),
        lambda res, ct: (jnp.dot(ct, res[1].T, reduction_order="linear"),
                         jnp.dot(res[0].T, ct, reduction_order="linear")),
    )
    cvjp_jaxpr = jax.make_jaxpr(
        jax.grad(
            lambda inp, weight: jnp.sum(pinned_linear(inp, weight), reduction_order="linear"),
            argnums=(0, 1),
        )
    )(x, w)
    self.assertTrue(all(
        e.params["reduction_order"] is ro.ReductionOrder.linear
        for e in _eqns_by_name(cvjp_jaxpr, "dot_general")
    ))

  @jtu.sample_product(
      [
          dict(shape=shape, axis=axis)
          for shape in [(2, 5), (3, 8), (2, 7, 4)]
          for axis in range(-len(shape), len(shape))
      ],
      tile=[2, 4],
      split=["block", "lanes", "strided"],
      keepdims=[False, True],
      op=["sum", "max"],
      dtype=jtu.dtypes.floating,
  )
  def test_tiled_sum_and_max(self, shape, axis, tile, split, keepdims, op, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    order = ro.tiled(tile, split=split)
    jnp_op = jnp.sum if op == "sum" else jnp.max
    fn = lambda x: jnp_op(
        x, axis=axis, keepdims=keepdims, reduction_order=order)
    ref_fn = lambda x: jnp_op(x, axis=axis, keepdims=keepdims)
    x, = args_maker()
    jaxpr = jax.make_jaxpr(fn)(x)
    prim = "reduce_sum" if op == "sum" else "reduce_max"
    self.assertEqual(
        [e.params["reduction_order"] for e in _eqns_by_name(jaxpr, prim)],
        [order],
    )
    self.assertTrue(all(
        ro.is_leaf_order(e.params.get("reduction_order"))
        for e in _lo_eqns(jaxpr, prim)
    ))
    self.assertAllClose(
        fn(x), ref_fn(x), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(fn, args_maker)

  @parameterized.parameters(
      lambda a, _: jnp.sum(a, axis=1, reduction_order=ro.tiled(4, pad=None)),
      lambda a, _: jnp.max(a, axis=1, reduction_order=ro.tiled(4, pad=None)),
      lambda a, b: jnp.dot(a, b, reduction_order=ro.tiled(4, pad=None)),
      lambda a, _: jnp.max(a, axis=0, reduction_order=ro.ring("asc")),
      lambda a, _: lax.reduce_sum(
          a, (1,), reduction_order=ro.tiled(4, pad=None)),
      lambda a, _: lax.reduce_sum(a, (-1,), reduction_order=ro.tiled(4)),
      lambda a, _: lax.reduce_max(a, (0, 1), reduction_order=ro.padded(8)),
      lambda a, b: lax.dot(a, b, reduction_order=ro.tiled(4, pad=None)),
      lambda a, _: jax.make_jaxpr(
          lambda z: lax.reduce_sum(
              z, (1,), reduction_order=ro.tiled(4, pad=None)))(a),
      lambda a, _: jnp.sum(a, axis=1, reduction_order=ro.padded(-4)),
  )
  def test_invalid_composite_orders_raise(self, fn):
    a, b = jnp.ones((4, 5), jnp.float32), jnp.ones((5, 3), jnp.float32)
    with self.assertRaises(ValueError):
      fn(a, b)

  def test_vmap_unmapped_composite_order(self):
    f = lambda _: jnp.sum(jnp.ones(8), reduction_order=ro.tiled(4))
    self.assertAllClose(jax.vmap(f)(jnp.arange(3)), jnp.full(3, 8.0))

  @parameterized.parameters(64, 128, 300)
  def test_xlane_jaxpr_structure(self, n):
    v = jnp.ones((2, n), dtype=jnp.float32)
    xl = ro.xlane("linear", ro.Intrinsic("tpu_xlane_128"))
    j_xs = jax.make_jaxpr(lambda z: jnp.sum(z, axis=1, reduction_order=xl))(v)
    self.assertEqual(
        [e.params["reduction_order"]
         for e in _eqns_by_name(j_xs, "reduce_sum")],
        [xl],
    )
    s_eqns = _lo_eqns(j_xs, "reduce_sum")
    self.assertLen(s_eqns, 2)
    self.assertEqual(
        s_eqns[-1].params["reduction_order"], ro.Intrinsic("tpu_xlane_128"))
    j_xm = jax.make_jaxpr(lambda z: jnp.max(z, axis=1, reduction_order=xl))(v)
    self.assertEqual(
        [e.params["reduction_order"]
         for e in _eqns_by_name(j_xm, "reduce_max")],
        [xl],
    )
    m_eqns = _lo_eqns(j_xm, "reduce_max")
    self.assertLen(m_eqns, 2)
    self.assertEqual(
        m_eqns[-1].params["reduction_order"], ro.Intrinsic("tpu_xlane_128"))
    self.assertAllClose(
        jnp.sum(v, axis=1, reduction_order=ro.xlane()),
        jnp.sum(v, axis=1),
        atol=_REASSOC_TOL,
        rtol=_REASSOC_TOL,
    )

  @jtu.sample_product(
      [
          dict(lhs_shape=(4, 5), rhs_shape=(5, 3), dnums=None),
          dict(lhs_shape=(5,), rhs_shape=(5, 3), dnums="matmul"),
          dict(lhs_shape=(4, 5), rhs_shape=(5,), dnums="matmul"),
          dict(lhs_shape=(2, 3, 5), rhs_shape=(5, 2, 4), dnums="bik,kbn->bin"),
          dict(lhs_shape=(2, 8, 3), rhs_shape=(2, 8, 4), dnums="bki,bkn->bin"),
      ],
      bk=[2, 4],
      split=["block", "strided"],
      dtype=jtu.dtypes.floating,
  )
  @jax.default_matmul_precision("float32")
  def test_tiled_dot_multi_block(
      self, lhs_shape, rhs_shape, dnums, bk, split, dtype
  ):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    order = ro.tiled(bk, split=split)
    if dnums is None:
      fn = lambda l, r: jnp.dot(
          l, r, preferred_element_type=dtype, reduction_order=order)
      ref_fn = lambda l, r: jnp.dot(l, r, preferred_element_type=dtype)
    elif dnums == "matmul":
      fn = lambda l, r: jnp.matmul(
          l, r, preferred_element_type=dtype, reduction_order=order)
      ref_fn = lambda l, r: jnp.matmul(l, r, preferred_element_type=dtype)
    else:
      fn = lambda l, r: jnp.einsum(
          dnums, l, r, preferred_element_type=dtype, reduction_order=order
      )
      ref_fn = lambda l, r: jnp.einsum(
          dnums, l, r, preferred_element_type=dtype)
    lhs, rhs = args_maker()
    bd_jaxpr = jax.make_jaxpr(fn)(lhs, rhs)
    self.assertEqual(
        [e.params["reduction_order"]
         for e in _eqns_by_name(bd_jaxpr, "dot_general")],
        [order],
    )
    dot_eqns = _lo_eqns(bd_jaxpr, "dot_general")
    sum_eqns = _lo_eqns(bd_jaxpr, "reduce_sum")
    self.assertLen(dot_eqns, 1)
    self.assertLen(sum_eqns, 1)
    self.assertTrue(all(
        ro.is_leaf_order(e.params.get("reduction_order"))
        for e in dot_eqns + sum_eqns
    ))
    self.assertAllClose(
        fn(lhs, rhs), ref_fn(lhs, rhs), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(fn, args_maker)

  @jtu.sample_product(
      op=["sum", "max", "dot"],
      order=[
          ro.tiled(2, "linear", ro.tiled(2, "linear", "linear")),
          ro.padded(6, ro.tiled(3, "linear", "linear")),
          ro.tiled(4, ro.cast(jnp.bfloat16, "linear"), "linear"),
      ],
  )
  @jax.default_matmul_precision("float32")
  def test_nested_combinators_leaf_only_jaxpr(self, op, order):
    rng = jtu.rand_default(self.rng())
    a = rng((2, 15), np.float32)
    b = rng((15, 4), np.float32)
    if op == "sum":
      fn = lambda x, _: jnp.sum(x, axis=1, reduction_order=order)
      ref = jnp.sum(a, axis=1)
    elif op == "max":
      fn = lambda x, _: jnp.max(x, axis=1, reduction_order=order)
      ref = jnp.max(a, axis=1)
    else:
      fn = lambda x, y: jnp.dot(x, y, reduction_order=order)
      ref = jnp.dot(a, b)
    jaxpr = jax.make_jaxpr(fn)(a, b)
    prim = {"sum": "reduce_sum", "max": "reduce_max", "dot": "dot_general"}[op]
    self.assertEqual(
        [e.params["reduction_order"] for e in _eqns_by_name(jaxpr, prim)],
        [order],
    )
    prim_eqns = (
        _lo_eqns(jaxpr, "reduce_sum")
        + _lo_eqns(jaxpr, "reduce_max")
        + _lo_eqns(jaxpr, "dot_general")
    )
    self.assertNotEmpty(prim_eqns)
    self.assertTrue(all(
        ro.is_leaf_order(e.params.get("reduction_order")) for e in prim_eqns
    ))
    has_cast = isinstance(order, ro.Tile) and isinstance(order.inner, ro.Cast)
    tol = 2e-1 if has_cast else 1e-4
    self.assertAllClose(fn(a, b), ref, atol=tol, rtol=tol)
    with self.assertRaises(ValueError):
      jnp.sum(a, axis=(0, 1), reduction_order=order)

  @jax.default_matmul_precision("float32")
  def test_unordered_scoped_combinators(self):
    rng = jtu.rand_default(self.rng())
    a = rng((4, 8), np.float32)
    b = rng((8, 3), np.float32)
    inner_u, outer_u = ro.Unordered("mosaic"), ro.Unordered("xla")
    order = ro.tiled(4, inner=inner_u, outer=outer_u)

    ts_fn = lambda x: jnp.sum(x, axis=1, reduction_order=order)
    ts_eqns = _lo_eqns(jax.make_jaxpr(ts_fn)(a), "reduce_sum")
    self.assertEqual(
        [e.params["reduction_order"] for e in ts_eqns], [inner_u, outer_u])
    self.assertAllClose(
        ts_fn(a), jnp.sum(a, axis=1), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(ts_fn, lambda: [a])

    bd_fn = lambda x, y: jnp.dot(x, y, reduction_order=order)
    bd_jaxpr = jax.make_jaxpr(bd_fn)(a, b)
    self.assertEqual(
        _lo_eqns(bd_jaxpr, "dot_general")[0].params["reduction_order"], inner_u)
    self.assertEqual(
        _lo_eqns(bd_jaxpr, "reduce_sum")[0].params["reduction_order"], outer_u)
    self.assertAllClose(
        bd_fn(a, b), a @ b, atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(bd_fn, lambda: [a, b])

  @jax.default_matmul_precision("float32")
  def test_composite_ad_remat_and_vmap(self):
    rng = jtu.rand_default(self.rng())
    a = rng((3, 4, 8), np.float32)
    w = rng((8, 5), np.float32)
    order = ro.tiled(4, ro.cast(jnp.bfloat16, "linear"), "linear")
    loss = lambda x, y: jnp.sum(
        jax.checkpoint(lambda u, v: jnp.dot(u, v, reduction_order=order))(x, y))
    jvp_jaxpr = jax.make_jaxpr(
        lambda x, dx: jax.jvp(
            lambda z: jnp.dot(z, w, reduction_order=order), (x,), (dx,))
    )(a[0], a[0])
    self.assertEqual(
        [e.params.get("reduction_order")
         for e in _eqns_by_name(jvp_jaxpr, "dot_general")],
        [order, None],
    )
    grad_jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(a[0], w)
    self.assertEqual(
        [e.params.get("reduction_order")
         for e in _eqns_by_name(grad_jaxpr, "dot_general")],
        [order, None, None],
    )
    got_g = jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1)), in_axes=(0, None)))
    ref_g = jax.vmap(
        jax.grad(lambda x, y: jnp.sum(jnp.dot(x, y)), argnums=(0, 1)),
        in_axes=(0, None),
    )
    self.assertAllClose(got_g(a, w), ref_g(a, w), atol=1e-4, rtol=1e-4)

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      mode=["sum_block", "sum_lanes", "dot_lhs", "dot_rhs"],
      pad=["absent", "zero"],
      jit=[False, True],
  )
  def test_tiled_zero_vs_absent_signbit(self, dtype, mode, pad, jit):
    nz = jnp.full((2, 5), -0.0, dtype=dtype)
    ones = jnp.ones((5, 3), dtype=dtype)
    if mode.startswith("sum"):
      split = "block" if mode == "sum_block" else "lanes"
      fn = lambda: jnp.sum(
          nz, axis=1, dtype=dtype,
          reduction_order=ro.tiled(4, split=split, pad=pad))
    elif mode == "dot_lhs":
      fn = lambda: jnp.dot(
          nz, ones, preferred_element_type=dtype,
          reduction_order=ro.tiled(4, pad=pad))
    else:
      fn = lambda: jnp.dot(
          jnp.ones((2, 5), dtype), jnp.full((5, 3), -0.0, dtype),
          preferred_element_type=dtype, reduction_order=ro.tiled(8, pad=pad))
    out = (jax.jit(fn) if jit else fn)()
    self.assertEqual(out.dtype, dtype)
    self.assertTrue(bool(jnp.all(out == 0)))
    self.assertEqual(bool(jnp.all(jnp.signbit(out))), pad == "absent")

  @parameterized.named_parameters(
      ("large_tiled_dot", lambda: jax.jit(
          lambda x, y: jnp.dot(
              x, y, reduction_order=ro.tiled(4, pad="absent")))(
          jnp.full((64, 5), -0.0, jnp.float32),
          jnp.ones((5, 32), jnp.float32))),
      ("long_linear_dot", lambda: jax.jit(
          lambda x, y: jnp.dot(x, y, reduction_order="linear"))(
          jnp.full((129, 17), -0.0, jnp.float32),
          jnp.ones((17, 1), jnp.float32))),
      ("mixed_int_float_dot", lambda: jax.jit(
          lambda x, y: jnp.dot(
              x, y, preferred_element_type=jnp.float32,
              reduction_order=ro.tiled(4, pad="absent")))(
          jnp.ones((2, 5), jnp.int32), jnp.full((5, 3), -0.0, jnp.float32))),
      ("padded_sum", lambda: jax.jit(
          lambda x: jnp.sum(
              x, axis=1, reduction_order=ro.padded(4, pad="absent")))(
          jnp.full((2, 5), -0.0, jnp.float32))),
  )
  def test_absent_padding_loop_and_mixed_dtype_signbit(self, compute_fn):
    out = compute_fn()
    self.assertTrue(bool(jnp.all(out == 0) & jnp.all(jnp.signbit(out))))

  def test_absent_padding_int_and_complex(self):
    int_lhs = jtu.rand_default(self.rng())((2, 5), np.int32)
    self.assertArraysEqual(
        jnp.sum(
            int_lhs, axis=1, dtype=jnp.int32,
            reduction_order=ro.tiled(4, pad="absent")),
        jnp.sum(int_lhs, axis=1, dtype=jnp.int32),
    )
    c_nz = jnp.full((2, 5), complex(-0.0, -0.0), dtype=jnp.complex64)
    c_out = jax.jit(
        lambda a: jnp.sum(a, axis=1, reduction_order=ro.tiled(4, pad="absent"))
    )(c_nz)
    re_neg, im_neg = jnp.signbit(jnp.real(c_out)), jnp.signbit(jnp.imag(c_out))
    self.assertTrue(bool(jnp.all(re_neg & im_neg)))

  @parameterized.product(
      use_general=[False, True],
      reduction_order=[
          None,
          "unordered",
          "linear",
          ro.Intrinsic("test_hw"),
          ro.Unordered(),
          ro.Unordered("xla"),
      ],
  )
  def test_ragged_dot_jaxpr_and_lowering(self, use_general, reduction_order):
    rng = jtu.rand_default(self.rng())
    lhs, rhs = rng((4, 8), np.float32), rng((2, 8, 3), np.float32)
    gs = jnp.array([1, 3], dtype=jnp.int32)
    dnums = lax.RaggedDotDimensionNumbers((([1], [1]), ([], [])), [0], [0])
    call = (functools.partial(lax.ragged_dot_general,
                              ragged_dot_dimension_numbers=dnums)
            if use_general else lax.ragged_dot)
    fn = lambda l, r, s: call(l, r, s, reduction_order=reduction_order)
    eqn, = _eqns_by_name(jax.make_jaxpr(fn)(lhs, rhs, gs), "ragged_dot_general")
    self.assertEqual(eqn.params.get("reduction_order"),
                     ro.normalize_order(reduction_order))
    if reduction_order == "linear" or isinstance(reduction_order, ro.Intrinsic):
      with self.assertRaisesRegex(NotImplementedError, "no XLA lowering"):
        jax.jit(fn)(lhs, rhs, gs)
    else:
      self.assertAllClose(fn(lhs, rhs, gs), call(lhs, rhs, gs))
      self._CompileAndCheck(fn, lambda: [lhs, rhs, gs])

  @parameterized.parameters(
      ro.tiled(4, "linear", "linear"),
      ro.tiled(4, "linear", "linear", split="strided"),
      ro.padded(6, "linear"),
      ro.tiled(4, ro.cast(jnp.bfloat16, "linear"), "linear"),
      ro.xlane("linear", "linear", num_lanes=4, pad="absent"),
  )
  def test_ragged_dot_composite_orders(self, order):
    rng = jtu.rand_default(self.rng())
    args = rng((5, 7), np.float32), rng((2, 7, 3), np.float32)
    gs = jnp.array([2, 3], dtype=jnp.int32)
    fn = lambda l, r, s: lax.ragged_dot(
        l, r, s, preferred_element_type=jnp.float32, reduction_order=order)
    jaxpr = jax.make_jaxpr(fn)(*args, gs)
    self.assertEqual(
        [e.params["reduction_order"]
         for e in _eqns_by_name(jaxpr, "ragged_dot_general")],
        [order],
    )
    lo_eqns = (
        _lo_eqns(jaxpr, "ragged_dot_general") + _lo_eqns(jaxpr, "reduce_sum")
    )
    self.assertNotEmpty(lo_eqns)
    self.assertTrue(all(
        ro.is_leaf_order(e.params.get("reduction_order")) for e in lo_eqns
    ))
    with self.assertRaisesRegex(NotImplementedError, "no XLA lowering"):
      jax.jit(fn)(*args, gs)

  def test_ragged_dot_invalid_composite_orders(self):
    gs = jnp.array([1, 1], dtype=jnp.int32)
    with self.assertRaises(ValueError):
      lax.ragged_dot(
          jnp.ones((2, 5)), jnp.ones((2, 5, 2)), gs,
          reduction_order=ro.tiled(4, pad=None))
    rc_dnums = lax.RaggedDotDimensionNumbers((([1], [0]), ([], [])), [1], [])
    with self.assertRaises(ValueError):
      lax.ragged_dot_general(
          jnp.ones((2, 4)), jnp.ones((4, 2)), jnp.array([2, 2], jnp.int32),
          rc_dnums, reduction_order=ro.tiled(2))
    mc_dnums = lax.RaggedDotDimensionNumbers(
        (([1, 2], [1, 2]), ([], [])), [0], [0])
    with self.assertRaises(ValueError):
      lax.ragged_dot_general(
          jnp.ones((2, 2, 2)), jnp.ones((2, 2, 2, 3)), gs,
          mc_dnums, reduction_order=ro.tiled(2))

  @parameterized.product(
      use_general=[False, True],
      order=["unreduced", ro.tiled(4, "unreduced", "linear"),
             ro.tiled(4, "linear", "unreduced")],
  )
  def test_ragged_dot_unreduced_not_implemented(self, use_general, order):
    lhs, rhs = jnp.ones((4, 8)), jnp.ones((2, 8, 3))
    gs = jnp.array([1, 3], dtype=jnp.int32)
    dnums = lax.RaggedDotDimensionNumbers((([1], [1]), ([], [])), [0], [0])
    call = (functools.partial(lax.ragged_dot_general,
                              ragged_dot_dimension_numbers=dnums)
            if use_general else lax.ragged_dot)
    with self.assertRaisesRegex(NotImplementedError, "unreduced"):
      call(lhs, rhs, gs, reduction_order=order)

  def test_ragged_dot_ad_remat_and_vmap_jaxprs(self):
    rng = jtu.rand_default(self.rng())
    a = rng((3, 4, 8), np.float32)
    w = rng((3, 2, 8, 5), np.float32)
    gs = jnp.array([1, 3], dtype=jnp.int32)
    orders = lambda jaxpr: [e.params.get("reduction_order")
                            for e in _eqns_by_name(jaxpr, "ragged_dot_general")]
    jvp_orders = lambda o: orders(jax.make_jaxpr(lambda x, dx: jax.jvp(
        lambda z: lax.ragged_dot(z, w[0], gs, reduction_order=o), (x,), (dx,))
    )(a[0], a[0]))
    lin = ro.ReductionOrder.linear
    self.assertEqual(jvp_orders("linear"), [lin, lin])

    order = ro.tiled(4, ro.cast(jnp.bfloat16, "linear"), "linear")
    self.assertEqual(jvp_orders(order), [order, None])
    loss = lambda x, y: jnp.sum(jax.checkpoint(
        lambda u, v: lax.ragged_dot(u, v, gs, reduction_order=order))(x, y))
    grad_jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(a[0], w[0])
    self.assertEqual(orders(grad_jaxpr), [order, None, None])
    vmap_jaxpr = jax.make_jaxpr(jax.vmap(
        lambda x, y, s: lax.ragged_dot(x, y, s, reduction_order=order)))(
            a, w, jnp.broadcast_to(gs, (3, 2)))
    self.assertEqual(orders(vmap_jaxpr), [order])


@jtu.thread_unsafe_test_class()  # Mutates module-level ro._INTRINSIC_IMPLS registry.
class ReductionOrderIntrinsicRegistrationTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ("sum", "reduce_sum", lambda a, w, o: jnp.sum(a, reduction_order=o)),
      ("max", "reduce_max", lambda a, w, o: jnp.max(a, reduction_order=o)),
      ("dot", "dot_general", lambda a, w, o: jnp.dot(a, w, reduction_order=o)),
  )
  def test_unregistered_intrinsic_raises(self, prim, op_fn):
    rng = jtu.rand_default(self.rng())
    x = rng((4, 8), np.float32)
    w = rng((8, 3), np.float32)
    unreg = ro.Intrinsic("unregistered_test_leaf")
    jaxpr = jax.make_jaxpr(lambda a, b: op_fn(a, b, unreg))(x, w)
    self.assertEqual(_eqns_by_name(jaxpr, prim)[0].params["reduction_order"], unreg)
    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a, b: op_fn(a, b, unreg))(x, w)

  @parameterized.named_parameters(("sum", jnp.sum), ("max", jnp.max))
  def test_registered_intrinsic_dispatch(self, op_fn):
    rng = jtu.rand_default(self.rng())
    x = rng((8,), np.float32)
    name = "test_registered_leaf"
    ro.register_intrinsic(
        name,
        lambda a, *, axes, op: (
            lax.reduce_sum(a, axes, reduction_order="linear")
            if op == "sum"
            else lax.reduce_max(a, axes, reduction_order="linear")
        ),
    )
    try:
      leaf = ro.Intrinsic(name)
      self.assertArraysEqual(
          jax.jit(lambda a: op_fn(a, reduction_order=leaf))(x),
          op_fn(x, reduction_order="linear"),
      )
    finally:
      ro._INTRINSIC_IMPLS.pop(name, None)

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      k=[5, 130],
      pad=["absent", "zero"],
      jit=[False, True],
  )
  def test_xlane_sum_zero_vs_absent_signbit(self, dtype, k, pad, jit):
    ro.register_intrinsic(
        "tpu_xlane_128",
        lambda a, *, axes, op: lax.reduce_sum(a, axes, reduction_order="linear"),
    )
    try:
      x_nz = jnp.full((2, k), -0.0, dtype=dtype)
      xl = ro.xlane("linear", ro.Intrinsic("tpu_xlane_128"), pad=pad)
      fn = lambda a: jnp.sum(a, axis=1, dtype=dtype, reduction_order=xl)
      out = (jax.jit(fn) if jit else fn)(x_nz)
      self.assertTrue(bool(jnp.all(out == 0)))
      if pad == "absent":
        self.assertTrue(bool(jnp.all(jnp.signbit(out))))
      else:
        self.assertFalse(bool(jnp.any(jnp.signbit(out))))
    finally:
      ro._INTRINSIC_IMPLS.pop("tpu_xlane_128", None)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
