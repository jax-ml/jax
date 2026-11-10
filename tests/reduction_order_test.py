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
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


def _eqns_by_name(closed_jaxpr, name: str):
  return [e for e in jtu.iter_eqns(closed_jaxpr.jaxpr) if e.primitive.name == name]


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
      None,
      "unordered",
      "linear",
      ro.Intrinsic("tpu_xlane_128"),
      ro.Unordered(),
      ro.Unordered(None),
      ro.Unordered("xla"),
      ro.Unordered("mosaic"),
  )
  def test_normalize_order_valid(self, order):
    self.assertEqual(ro.normalize_order(order), order)

  def test_unordered_equality(self):
    self.assertEqual(ro.Unordered(), ro.Unordered(None))
    self.assertNotEqual(ro.Unordered("xla"), ro.Unordered("mosaic"))

  @parameterized.parameters(("",), ("tree",), (123,), (["linear"],))
  def test_normalize_order_invalid(self, bad_order):
    with self.assertRaises(ValueError):
      ro.normalize_order(bad_order)

  @parameterized.parameters(("",), (None,), (42,))
  def test_intrinsic_invalid_name(self, bad_name):
    with self.assertRaises(ValueError):
      ro.Intrinsic(bad_name)

  @parameterized.parameters(("",), (123,), (["xla"],))
  def test_unordered_invalid_scope(self, bad_scope):
    with self.assertRaises(ValueError):
      ro.Unordered(bad_scope)

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
    self.assertEqual(eqn.params["reduction_order"], reduction_order)

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
    self.assertEqual(eqn.params["reduction_order"], "linear")
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
    self.assertTrue(all(e.params["reduction_order"] == "linear" for e in dot_eqns))

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
      self.assertEqual([e.params["reduction_order"] for e in grad_dots[:2]], ["linear", "linear"])
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
    self.assertTrue(
        all(e.params["reduction_order"] == "linear" for e in _eqns_by_name(cvjp_jaxpr, "dot_general"))
    )

  @jtu.sample_product(
      [
          dict(shape=shape, axis=axis)
          for shape in [(2, 5), (3, 8), (2, 7, 4)]
          for axis in range(-len(shape), len(shape))
      ],
      tile=[2, 4],
      split=["block", "lanes"],
      keepdims=[False, True],
      op=["sum", "max"],
      dtype=jtu.dtypes.floating,
  )
  def test_tiled_sum_and_max(self, shape, axis, tile, split, keepdims, op, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    if op == "sum":
      fn = lambda x: ro.tiled_sum(x, tile, axis=axis, split=split, keepdims=keepdims)
      ref_fn = lambda x: jnp.sum(x, axis=axis, keepdims=keepdims)
    else:
      fn = lambda x: ro.tiled_max(x, tile, axis=axis, split=split, keepdims=keepdims)
      ref_fn = lambda x: jnp.max(x, axis=axis, keepdims=keepdims)
    x, = args_maker()
    self.assertAllClose(fn(x), ref_fn(x), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(fn, args_maker)

  def test_tiled_max_unaligned_pad_false_raises(self):
    rng = jtu.rand_default(self.rng())
    x = rng((2, 5), np.float32)
    with self.assertRaises(ValueError):
      ro.tiled_max(x, 4, axis=1, pad=False)

  @parameterized.named_parameters(
      ("lt_128", 64, 1),
      ("eq_128", 128, 1),
      ("gt_128", 300, 2),
  )
  def test_xlane_jaxpr_structure(self, n, expected_eqns):
    v = jnp.ones((2, n), dtype=jnp.float32)
    j_xs = jax.make_jaxpr(lambda z: ro.xlane_sum(z, axis=1))(v)
    s_eqns = _eqns_by_name(j_xs, "reduce_sum")
    self.assertLen(s_eqns, expected_eqns)
    self.assertEqual(s_eqns[-1].params["reduction_order"], ro.Intrinsic("tpu_xlane_128"))
    j_xm = jax.make_jaxpr(lambda z: ro.xlane_max(z, axis=1))(v)
    m_eqns = _eqns_by_name(j_xm, "reduce_max")
    self.assertLen(m_eqns, expected_eqns)
    self.assertEqual(m_eqns[-1].params["reduction_order"], ro.Intrinsic("tpu_xlane_128"))

  @jtu.sample_product(
      [
          dict(lhs_shape=(4, 5), rhs_shape=(5, 3), dnums=None),
          dict(lhs_shape=(2, 3, 5), rhs_shape=(5, 2, 4), dnums=(((2,), (0,)), ((0,), (1,)))),
          dict(lhs_shape=(2, 8, 3), rhs_shape=(2, 8, 4), dnums=(((1,), (1,)), ((0,), (0,)))),
      ],
      bk=[2, 4],
      dtype=jtu.dtypes.floating,
  )
  @jax.default_matmul_precision("float32")
  def test_blocked_dot_multi_block(self, lhs_shape, rhs_shape, dnums, bk, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    fn = lambda l, r: ro.blocked_dot(
        l, r, bk, dimension_numbers=dnums, preferred_element_type=dtype
    )
    ref_fn = (
        (lambda l, r: jnp.dot(l, r, preferred_element_type=dtype))
        if dnums is None
        else (lambda l, r: lax.dot_general(l, r, dnums, preferred_element_type=dtype))
    )
    lhs, rhs = args_maker()
    bd_jaxpr = jax.make_jaxpr(fn)(lhs, rhs)
    self.assertLen(_eqns_by_name(bd_jaxpr, "dot_general"), 1)
    self.assertLen(_eqns_by_name(bd_jaxpr, "reduce_sum"), 1)
    self.assertAllClose(fn(lhs, rhs), ref_fn(lhs, rhs), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(fn, args_maker)

  @parameterized.parameters(5, 8)
  @jax.default_matmul_precision("float32")
  def test_blocked_dot_single_block_skips_outer_reduce_sum(self, bk_single):
    # 1-block blocked_dot (K <= bk) skips redundant 1-block reshape + outer reduce_sum.
    rng = jtu.rand_default(self.rng())
    lhs = rng((2, 3, 5), np.float32)
    rhs = rng((5, 2, 4), np.float32)
    dnums = (((2,), (0,)), ((0,), (1,)))
    bd1_jaxpr = jax.make_jaxpr(
        lambda l, r: ro.blocked_dot(
            l, r, bk_single, dimension_numbers=dnums,
            inner_order=ro.Intrinsic("tpu_matmul_bf16"),
            outer_order=ro.Intrinsic("tpu_xlane_128"),
        )
    )(lhs, rhs)
    self.assertLen(_eqns_by_name(bd1_jaxpr, "dot_general"), 1)
    self.assertEmpty(_eqns_by_name(bd1_jaxpr, "reduce_sum"))
    self.assertAllClose(
        ro.blocked_dot(lhs, rhs, bk_single, dimension_numbers=dnums),
        lax.dot_general(lhs, rhs, dnums),
        atol=_REASSOC_TOL,
        rtol=_REASSOC_TOL,
    )

  @jax.default_matmul_precision("float32")
  def test_unordered_scoped_helpers(self):
    rng = jtu.rand_default(self.rng())
    a = rng((4, 8), np.float32)
    b = rng((8, 3), np.float32)
    inner_u, outer_u = ro.Unordered("mosaic"), ro.Unordered("xla")

    ts_fn = lambda x: ro.tiled_sum(x, 4, axis=1, inner_order=inner_u, outer_order=outer_u)
    ts_eqns = _eqns_by_name(jax.make_jaxpr(ts_fn)(a), "reduce_sum")
    self.assertEqual([e.params["reduction_order"] for e in ts_eqns], [inner_u, outer_u])
    self.assertAllClose(ts_fn(a), jnp.sum(a, axis=1), atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(ts_fn, lambda: [a])

    bd_fn = lambda x, y: ro.blocked_dot(x, y, 4, inner_order=inner_u, outer_order=outer_u)
    bd_jaxpr = jax.make_jaxpr(bd_fn)(a, b)
    self.assertEqual(_eqns_by_name(bd_jaxpr, "dot_general")[0].params["reduction_order"], inner_u)
    self.assertEqual(_eqns_by_name(bd_jaxpr, "reduce_sum")[0].params["reduction_order"], outer_u)
    self.assertAllClose(bd_fn(a, b), a @ b, atol=_REASSOC_TOL, rtol=_REASSOC_TOL)
    self._CompileAndCheck(bd_fn, lambda: [a, b])

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      split=["block", "lanes"],
      pad=["absent", "zero"],
      jit=[False, True],
  )
  def test_tiled_sum_zero_vs_absent_signbit(self, dtype, split, pad, jit):
    nz = jnp.full((2, 5), -0.0, dtype=dtype)
    fn = lambda a: ro.tiled_sum(a, 4, axis=1, split=split, pad=pad)
    out = (jax.jit(fn) if jit else fn)(nz)
    self.assertEqual(out.dtype, dtype)
    self.assertTrue(bool(jnp.all(out == 0)))
    if pad == "absent":
      self.assertTrue(bool(jnp.all(jnp.signbit(out))))
    else:
      self.assertFalse(bool(jnp.any(jnp.signbit(out))))

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
      bk=[4, 8],
      nz_side=["lhs", "rhs"],
      pad=["absent", "zero"],
      jit=[False, True],
  )
  def test_blocked_dot_zero_vs_absent_signbit(self, dtype, bk, nz_side, pad, jit):
    lhs = jnp.full((2, 5), -0.0, dtype=dtype) if nz_side == "lhs" else jnp.ones((2, 5), dtype=dtype)
    rhs = jnp.ones((5, 3), dtype=dtype) if nz_side == "lhs" else jnp.full((5, 3), -0.0, dtype=dtype)
    fn = lambda x, y: ro.blocked_dot(x, y, bk, preferred_element_type=dtype, pad=pad)
    out = (jax.jit(fn) if jit else fn)(lhs, rhs)
    self.assertEqual(out.dtype, dtype)
    self.assertTrue(bool(jnp.all(out == 0)))
    if pad == "absent":
      self.assertTrue(bool(jnp.all(jnp.signbit(out))))
    else:
      self.assertFalse(bool(jnp.any(jnp.signbit(out))))

  @parameterized.named_parameters(
      (
          "large_blocked_dot_loop",
          lambda: jax.jit(lambda x, y: ro.blocked_dot(x, y, 4, pad="absent"))(
              jnp.full((64, 5), -0.0, dtype=jnp.float32),
              jnp.ones((5, 32), dtype=jnp.float32),
          ),
      ),
      (
          "long_linear_dot_loop",
          lambda: jax.jit(lambda x, y: jnp.dot(x, y, reduction_order="linear"))(
              jnp.full((129, 17), -0.0, dtype=jnp.float32),
              jnp.ones((17, 1), dtype=jnp.float32),
          ),
      ),
      (
          "mixed_int32_float32_blocked_dot",
          lambda: jax.jit(lambda x, y: ro.blocked_dot(x, y, 4, pad="absent"))(
              jnp.ones((2, 5), dtype=jnp.int32),
              jnp.full((5, 3), -0.0, dtype=jnp.float32),
          ),
      ),
  )
  def test_absent_padding_loop_and_mixed_dtype_signbit(self, compute_fn):
    out = compute_fn()
    self.assertTrue(bool(jnp.all(out == 0)))
    self.assertTrue(bool(jnp.all(jnp.signbit(out))))

  def test_absent_padding_int_and_complex(self):
    rng = jtu.rand_default(self.rng())
    int_lhs = rng((2, 5), np.int32)
    self.assertArraysEqual(
        ro.tiled_sum(int_lhs, 4, axis=1, pad="absent"),
        jnp.sum(int_lhs, axis=1, dtype=jnp.int32),
    )
    c_nz = jnp.full((2, 5), complex(-0.0, -0.0), dtype=jnp.complex64)
    c_out = jax.jit(lambda a: ro.tiled_sum(a, 4, axis=1, pad="absent"))(c_nz)
    self.assertTrue(bool(jnp.all(jnp.signbit(jnp.real(c_out)) & jnp.signbit(jnp.imag(c_out)))))

  @parameterized.product(
      bad_pad=["bogus", "", False, True, None, 0],
      call_fn=[
          lambda p: ro.tiled_sum(jnp.ones((2, 8), jnp.float32), 4, axis=1, pad=p),
          lambda p: ro.tiled_sum(jnp.ones((2, 5), jnp.float32), 4, axis=1, pad=p),
          lambda p: ro.xlane_sum(jnp.ones((2, 128), jnp.float32), axis=1, pad=p),
          lambda p: ro.xlane_sum(jnp.ones((2, 5), jnp.float32), axis=1, pad=p),
          lambda p: ro.blocked_dot(jnp.ones((2, 8), jnp.float32), jnp.ones((8, 3), jnp.float32), 4, pad=p),
          lambda p: ro.blocked_dot(jnp.ones((2, 5), jnp.float32), jnp.ones((5, 3), jnp.float32), 4, pad=p),
      ],
  )
  def test_invalid_pad_raises(self, bad_pad, call_fn):
    with self.assertRaises(ValueError):
      call_fn(bad_pad)


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
      fn = lambda a: ro.xlane_sum(a, axis=1, pad=pad)
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
