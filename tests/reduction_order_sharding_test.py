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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import core
from jax._src import mesh as mesh_lib
from jax._src import reduction_order as ro
from jax._src import test_util as jtu
from jax._src.lax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


def _ring_permute_np(
    x: np.ndarray, split_axis: int, direction: str
) -> np.ndarray:
  x_oax = (split_axis % (x.ndim - 1)) + 1
  s, chunk = x.shape[0], x.shape[x_oax] // x.shape[0]
  parts = []
  for c in range(s):
    xc = np.take(x, range(c * chunk, (c + 1) * chunk), axis=x_oax)
    asc = np.roll(xc, -(c + 1), axis=0)
    desc = np.flip(np.roll(xc, -c, axis=0), axis=0)
    if direction in ("asc", "desc"):
      parts.append(asc if direction == "asc" else desc)
    else:
      h = chunk // 2
      top = np.take(asc, range(h), axis=x_oax)
      bot = np.take(desc, range(h, chunk), axis=x_oax)
      parts.append(np.concatenate([top, bot], axis=x_oax))
  return np.concatenate(parts, axis=x_oax)


def _single_device_eval(fn, *np_args):
  with mesh_lib.use_abstract_mesh(mesh_lib.empty_abstract_mesh):
    return fn(*(jnp.asarray(a) for a in np_args))


def _ring_ref(partials_np: np.ndarray, direction: str, split_axis: int = 0):
  arr = _ring_permute_np(partials_np, split_axis, direction)
  return _single_device_eval(
      lambda z: jnp.sum(z, axis=0, dtype=z.dtype, reduction_order="linear"), arr
  )


def _run_op(op: str, x, y, *, out_sharding=None, reduction_order=None):
  kw = dict(out_sharding=out_sharding, reduction_order=reduction_order)
  if op == "sum":
    return jnp.sum(x, axis=1, **kw)
  if op == "max":
    return jnp.max(x, axis=1, **kw)
  if op == "dot":
    return jnp.dot(x, y, **kw)
  return jnp.einsum("mk,kn->mn", x, y, **kw)


class ReductionOrderShardingTest(jtu.JaxTestCase):

  @parameterized.parameters(False, True)
  @jtu.with_explicit_mesh((2,), ("X",))
  def test_canonical_sum_combinator_table(self, use_jit, mesh):
    wrap = jax.jit if use_jit else (lambda f: f)
    x1_np = np.array(
        [1e8, 1.0, 2.0, -1e8, -1e8, 3.0, 4.0, 1e8], dtype=np.float32)
    x2_np = np.stack([x1_np + np.float32(i) for i in range(16)], axis=1)
    x1 = jax.device_put(x1_np, NamedSharding(mesh, P("X")))
    x2 = jax.device_put(x2_np, NamedSharding(mesh, P("X", None)))

    out_uo = wrap(lambda z: jnp.sum(z, reduction_order="unordered"))(x1)
    self.assertEqual(out_uo.sharding, NamedSharding(mesh, P()))
    with self.assertRaises(core.ShardingTypeError):
      wrap(lambda z: jnp.sum(z, reduction_order="linear"))(x1)

    r_unred = ro.tiled(4, "linear", "unreduced")
    u1 = wrap(lambda z: jnp.sum(
        z, out_sharding=P(unreduced={"X"}), reduction_order=r_unred))(x1)
    self.assertEqual(u1.sharding, NamedSharding(mesh, P(unreduced={"X"})))
    self.assertAllClose(
        jax.sharding.reshard(u1, P()), jnp.sum(jnp.asarray(x1_np)), atol=2.0)

    x_1ps = jax.device_put(
        np.array([3.0, 5.0], dtype=np.float32), NamedSharding(mesh, P("X")))
    u_leaf_fn = lambda z: lax.reduce_sum(
        z, (0,), out_sharding=P(unreduced={"X"}), reduction_order="unreduced")
    u_leaf_jaxpr = jax.make_jaxpr(u_leaf_fn)(x_1ps)
    self.assertEqual(
        u_leaf_jaxpr.eqns[0].params["reduction_order"],
        ro.ReductionOrder.unreduced,
    )
    self.assertAllClose(
        jax.sharding.reshard(wrap(u_leaf_fn)(x_1ps), P()), np.float32(8.0))
    u_dot = wrap(lambda z: lax.dot(
        z, z, out_sharding=P(unreduced={"X"}),
        reduction_order="unreduced"))(x_1ps)
    self.assertAllClose(jax.sharding.reshard(u_dot, P()), np.float32(34.0))
    with self.assertRaises(core.ShardingTypeError):
      wrap(lambda z: lax.reduce_sum(z, (0,), reduction_order="unreduced"))(
          x_1ps)
    with self.assertRaises(core.ShardingTypeError):
      _single_device_eval(
          lambda z: lax.reduce_sum(z, (0,), reduction_order="unreduced"),
          np.ones(2, dtype=np.float32),
      )

    r_tu = ro.tiled(4, "linear", "unordered")
    out_tu = wrap(lambda z: jnp.sum(z, reduction_order=r_tu))(x1)
    ref_tu = _single_device_eval(
        lambda z: jnp.sum(z, reduction_order=r_tu), x1_np)
    self.assertEqual(out_tu.sharding, NamedSharding(mesh, P()))
    self.assertAllClose(out_tu, ref_tu, atol=1e-5)

    with self.assertRaises(ValueError):
      wrap(lambda z: jnp.sum(
          z, reduction_order=ro.tiled(4, "linear", ro.ring("asc"))))(x1)

    p2 = _single_device_eval(
        lambda z: jnp.sum(z, axis=1, reduction_order="linear"),
        x2_np.reshape(2, 4, 16),
    )
    ref_rb = _ring_ref(np.asarray(p2), "bidir", split_axis=0)

    r_ar = ro.tiled(4, "linear", ro.ring("bidir", split_axis=0))
    out_ar = wrap(lambda z: jnp.sum(z, axis=0, reduction_order=r_ar))(x2)
    self.assertEqual(out_ar.sharding, NamedSharding(mesh, P(None)))
    self.assertArraysEqual(out_ar, ref_rb, check_dtypes=True)

    r_bi = ro.tiled(4, "linear", ro.ring("bidir"))
    out_rb = wrap(lambda z: jnp.sum(
        z, axis=0, out_sharding=P("X"), reduction_order=r_bi))(x2)
    self.assertEqual(out_rb.sharding, NamedSharding(mesh, P("X")))
    self.assertArraysEqual(out_rb, ref_rb, check_dtypes=True)
    out_kd = wrap(lambda z: jnp.sum(
        z, axis=0, keepdims=True, out_sharding=P(None, "X"),
        reduction_order=r_bi))(x2)
    self.assertEqual(out_kd.sharding, NamedSharding(mesh, P(None, "X")))
    self.assertArraysEqual(out_kd, ref_rb[None, :], check_dtypes=True)
    with self.assertRaises(core.ShardingTypeError):
      wrap(lambda z: jnp.sum(z, axis=0, keepdims=True, out_sharding=P("X")))(x2)
    g_rb = wrap(jax.grad(lambda z: jnp.sum(
        jnp.sum(z, axis=0, out_sharding=P("X"), reduction_order=r_bi) ** 2
    )))(x2)
    self.assertArraysEqual(
        g_rb, np.broadcast_to(2.0 * ref_rb, x2.shape), check_dtypes=True)
    x2_b = jax.device_put(
        np.stack([x2_np] * 3), NamedSharding(mesh, P(None, "X", None)))
    got_vmap1 = wrap(
        jax.vmap(lambda z: jnp.sum(z, axis=0, reduction_order=r_ar)))(x2_b)
    self.assertArraysEqual(got_vmap1, np.stack([ref_rb] * 3))
    r_str = ro.tiled(
        4, ro.ring("bidir", split_axis=0), "linear", split="strided")
    got_vmap2 = wrap(
        jax.vmap(lambda z: jnp.sum(z, axis=0, reduction_order=r_str)))(x2_b)
    self.assertArraysEqual(
        got_vmap2, jnp.stack([jnp.sum(x2, axis=0, reduction_order=r_str)] * 3))
    r_nest = ro.tiled(
        4,
        ro.tiled(2, ro.ring("bidir", split_axis=1), "linear", split="strided"),
        "linear",
        split="strided",
    )
    xn_np = np.arange(64, dtype=np.float32).reshape(16, 4)
    xn = jax.device_put(xn_np, NamedSharding(mesh, P("X", None)))
    xn_b = jax.device_put(
        np.stack([xn_np] * 3, axis=1), NamedSharding(mesh, P("X", None, None)))
    got_nest = wrap(jax.vmap(
        lambda z: jnp.sum(z, axis=0, reduction_order=r_nest), in_axes=1))(xn_b)
    self.assertArraysEqual(
        got_nest, jnp.stack([jnp.sum(xn, axis=0, reduction_order=r_nest)] * 3))

    u2 = wrap(lambda z: jnp.sum(
        z, axis=0, out_sharding=P(None, unreduced={"X"}),
        reduction_order=r_unred))(x2)
    with self.assertRaises(core.ShardingTypeError):
      jnp.dot(u2, jnp.ones_like(x2_np[0]))
    out_rs = wrap(lambda u: jax.sharding.reshard(
        u, P("X"), reduction_order=ro.ring("bidir")))(u2)
    self.assertEqual(out_rs.sharding, NamedSharding(mesh, P("X")))
    self.assertArraysEqual(out_rs, ref_rb, check_dtypes=True)
    rs_jaxpr = jax.make_jaxpr(lambda u: jax.sharding.reshard(
        u, P("X"), reduction_order=ro.ring("bidir")))(u2)
    self.assertLen(rs_jaxpr.eqns, 1)
    self.assertEqual(rs_jaxpr.eqns[0].primitive.name, "reshard")
    self.assertEqual(
        rs_jaxpr.eqns[0].params["reduction_order"], ro.ring("bidir"))
    self.assertIn("reduction_order=Ring", str(rs_jaxpr))
    g_rs = wrap(jax.grad(lambda z: jnp.sum(jax.sharding.reshard(
        jnp.sum(
            z, axis=0, out_sharding=P(None, unreduced={"X"}),
            reduction_order=r_unred),
        P("X"), reduction_order=ro.ring("bidir")) ** 2)))(x2)
    self.assertArraysEqual(
        g_rs, np.broadcast_to(2.0 * ref_rb, x2.shape), check_dtypes=True)
    to_u = lambda z: jnp.sum(
        z, axis=0, out_sharding=P(None, unreduced={"X"}),
        reduction_order=r_unred)
    got_vmap_ar = wrap(jax.vmap(lambda z: jax.sharding.reshard(
        to_u(z), P(), reduction_order=ro.ring("bidir", split_axis=0))))(x2_b)
    self.assertArraysEqual(got_vmap_ar, np.stack([ref_rb] * 3))
    got_vmap_rs = wrap(jax.vmap(lambda z: jax.sharding.reshard(
        to_u(z), P("X"), reduction_order=ro.ring("bidir"))))(x2_b)
    self.assertArraysEqual(got_vmap_rs, np.stack([ref_rb] * 3))
    with self.assertRaises(core.ShardingTypeError):
      wrap(lambda u: jax.sharding.reshard(
          u, P(None, unreduced={"X"}), reduction_order="linear"))(u2)
    with self.assertRaises(core.ShardingTypeError):
      wrap(lambda u: jax.sharding.reshard(
          u, P(), reduction_order="linear"))(u2)
    with self.assertRaises(ValueError):
      wrap(lambda u: jax.sharding.reshard(
          u, P(), reduction_order=ro.ring("asc")))(u1)

  @jtu.sample_product(
      op=["sum", "max", "dot", "einsum"],
      split=["block", "strided"],
      use_jit=[False, True],
  )
  @jtu.with_explicit_mesh((2, 2), ("x", "y"))
  def test_sharded_ops_linear_tiled_and_errors(self, op, split, use_jit, mesh):
    rng = jtu.rand_default(self.rng())
    wrap = jax.jit if use_jit else (lambda f: f)
    a_np, b_np = rng((4, 16), np.float32), rng((16, 4), np.float32)

    a_free = jax.device_put(a_np, NamedSharding(mesh, P("x", None)))
    b_free = jax.device_put(b_np, NamedSharding(mesh, P(None, "y")))
    ref = _single_device_eval(
        lambda x, y: _run_op(op, x, y, reduction_order="linear"), a_np, b_np)
    got_free = wrap(
        lambda x, y: _run_op(op, x, y, reduction_order="linear")
    )(a_free, b_free)
    self.assertArraysEqual(got_free, ref, check_dtypes=True)

    a_red = jax.device_put(a_np, NamedSharding(mesh, P("y", "x")))
    b_red = jax.device_put(b_np, NamedSharding(mesh, P("x", None)))
    u_out_s = P("y", None) if op in ("dot", "einsum") else None
    _ = wrap(lambda x, y: _run_op(
        op, x, y, out_sharding=u_out_s,
        reduction_order=ro.Unordered("xla")))(a_red, b_red)
    with self.assertRaises(core.ShardingTypeError):
      wrap(lambda x, y: _run_op(op, x, y, reduction_order="linear"))(
          a_red, b_red)
    with self.assertRaises(core.ShardingTypeError):
      wrap(lambda x, y: _run_op(
          op, x, y, reduction_order=ro.Intrinsic("tpu_xlane_128")))(
          a_red, b_red)

    t_ord = ro.tiled(4, "linear", "unordered", split=split)
    ta, tb = (a_red, b_red) if split == "block" else (a_free, b_free)
    t_out_s = u_out_s if split == "block" else None
    got_tiled = wrap(lambda x, y: _run_op(
        op, x, y, out_sharding=t_out_s, reduction_order=t_ord))(ta, tb)
    self.assertAllClose(got_tiled, ref, rtol=1e-4, atol=1e-4)

    if op in ("dot", "einsum") and split == "block":
      with self.assertRaises(core.ShardingTypeError):
        wrap(lambda x, y: _run_op(op, x, y, reduction_order=t_ord))(
            a_red, b_red)
      b_dup = jax.device_put(b_np, NamedSharding(mesh, P(None, "x")))
      out_dup = wrap(lambda x, y: _run_op(
          op, x, y, out_sharding=P("x", None),
          reduction_order="linear"))(a_free, b_dup)
      self.assertEqual(out_dup.sharding, NamedSharding(mesh, P("x", None)))
      self.assertArraysEqual(out_dup, ref, check_dtypes=True)
      a_half = jax.device_put(a_np, NamedSharding(mesh, P(None, "x")))
      b_unsh = jax.device_put(b_np, NamedSharding(mesh, P(None, None)))
      out_half = wrap(
          lambda x, y: _run_op(op, x, y, reduction_order=t_ord))(a_half, b_unsh)
      self.assertEqual(out_half.sharding, NamedSharding(mesh, P(None, None)))
      self.assertAllClose(out_half, ref, rtol=1e-4, atol=1e-4)

  @parameterized.parameters(
      lambda a, _, r, m: jnp.sum(
          a, axis=1, reduction_order=ro.tiled(16, "linear", "unordered")),
      lambda a, _, r, m: jnp.sum(
          jax.device_put(
              r((4, 10), np.float32), NamedSharding(m, P(None, "x"))),
          axis=1, reduction_order=ro.tiled(4, "linear", "unordered")),
      lambda a, _, r, m: jnp.sum(
          a, axis=1,
          reduction_order=ro.tiled(4, "linear", "unordered", split="strided")),
      lambda a, b, r, m: jnp.dot(
          a, jax.device_put(b, NamedSharding(m, P("y", None))),
          out_sharding=P(),
          reduction_order=ro.tiled(4, "linear", "unordered")),
      lambda a, b, r, m: jnp.dot(
          a, jax.device_put(b, NamedSharding(m, P("x", None))),
          reduction_order=ro.tiled(4, "linear", "unordered")),
  )
  @jtu.with_explicit_mesh((2, 2), ("x", "y"))
  def test_tiled_sharding_type_errors(self, bad_fn, mesh):
    rng = jtu.rand_default(self.rng())
    a = jax.device_put(
        rng((4, 16), np.float32), NamedSharding(mesh, P(None, "x")))
    b = rng((16, 4), np.float32)
    with self.assertRaises(core.ShardingTypeError):
      bad_fn(a, b, rng, mesh)

  @jtu.sample_product(direction=["asc", "desc", "bidir"], use_jit=[False, True])
  @jtu.with_explicit_mesh((4,), ("x",))
  def test_sharded_ring_0ulp_and_errors(self, direction, use_jit, mesh):
    wrap = jax.jit if use_jit else (lambda f: f)
    base = np.array([1e8, 1.0, -1e8, 3.0], dtype=np.float32)
    x_flat = np.array(
        [[np.roll(base, r + c)[i] / 4.0 for c in range(16)]
         for r in range(4) for i in range(4)],
        dtype=np.float32,
    )
    with self.assertRaises(core.ShardingTypeError):
      _single_device_eval(
          lambda z: jnp.sum(
              z, axis=0, reduction_order=ro.ring(direction, split_axis=0)),
          x_flat,
      )

    x_sh = jax.device_put(x_flat, NamedSharding(mesh, P("x", None)))
    r_scatter = ro.tiled(4, "linear", ro.ring(direction))
    out_sc = wrap(lambda z: jnp.sum(
        z, axis=0, out_sharding=P("x"), reduction_order=r_scatter))(x_sh)
    partials = np.asarray(_single_device_eval(
        lambda z: jnp.sum(z, axis=1, reduction_order="linear"),
        x_flat.reshape(4, 4, 16),
    ))
    ref_sc = _ring_ref(partials, direction, split_axis=0)
    self.assertEqual(out_sc.sharding, NamedSharding(mesh, P("x")))
    self.assertArraysEqual(out_sc, ref_sc, check_dtypes=True)

    r_allreduce = ro.tiled(4, "linear", ro.ring(direction, split_axis=0))
    out_ar = wrap(
        lambda z: jnp.sum(z, axis=0, reduction_order=r_allreduce))(x_sh)
    self.assertEqual(out_ar.sharding, NamedSharding(mesh, P(None)))
    self.assertArraysEqual(out_ar, ref_sc, check_dtypes=True)

    with self.assertRaises(ValueError):
      jnp.sum(x_sh, axis=0, reduction_order=r_scatter)
    x_odd = jax.device_put(
        np.ones((4, 4), np.float32), NamedSharding(mesh, P("x", None)))
    with self.assertRaises(ValueError):
      jnp.sum(x_odd, axis=0, out_sharding=P("x"),
              reduction_order=ro.ring("bidir"))
    x3_sh = jax.device_put(
        x_flat.reshape(16, 4, 4), NamedSharding(mesh, P("x", None, None)))
    with self.assertRaises(core.ShardingTypeError):
      jnp.sum(
          x3_sh, axis=0, out_sharding=P("x", None),
          reduction_order=ro.tiled(
              4, "linear", ro.ring(direction, split_axis=1)),
      )
    with self.assertRaises(core.ShardingTypeError):
      jnp.sum(x_sh, axis=0, out_sharding=P("x"), reduction_order=ro.ring("asc"))
    with self.assertRaises(core.ShardingTypeError):
      x_col = jax.device_put(x_flat, NamedSharding(mesh, P(None, "x")))
      jnp.sum(x_col, axis=0, reduction_order=ro.ring("asc", split_axis=0))

  @jtu.thread_unsafe_test()
  @parameterized.parameters("asc", "desc", "bidir")
  @jtu.with_explicit_mesh((2, 4), ("dp", "tp"))
  def test_2d_mesh_tp_dot_bf16_cast_ring_and_unreduced_reshard_0ulp(
      self, direction, mesh
  ):
    ro.register_intrinsic(
        "tpu_v7_mxu_256",
        lambda l, r, *, dimension_numbers, preferred_element_type=None, **_: (
            lax._linear_dot(
                preferred_element_type or jnp.result_type(l, r),
                l, r, dimension_numbers=dimension_numbers,
            )
        ),
    )
    try:
      rng = jtu.rand_default(self.rng())
      a_np, b_np = rng((4, 32), np.float32), rng((32, 8), np.float32)
      a = jax.device_put(a_np, NamedSharding(mesh, P("dp", "tp")))
      b = jax.device_put(b_np, NamedSharding(mesh, P("tp", None)))
      mxu = ro.Intrinsic("tpu_v7_mxu_256")
      inner = ro.cast(jnp.bfloat16, ro.tiled(4, mxu, "linear"))
      r_tp = ro.tiled(8, inner=inner, outer=ro.ring(direction))

      out_tp = jax.jit(lambda x, y: jnp.dot(
          x, y, preferred_element_type=jnp.float32,
          out_sharding=P("dp", "tp"), reduction_order=r_tp))(a, b)
      tp_partials = _single_device_eval(
          lambda x, y: jnp.einsum(
              "mtk,tkn->tmn", x.reshape(4, 4, 8), y.reshape(4, 8, 8),
              preferred_element_type=jnp.float32, reduction_order=inner,
          ).astype(jnp.bfloat16),
          a_np, b_np,
      )
      ref_tp = _ring_ref(
          np.asarray(tp_partials), direction, split_axis=1
      ).astype(jnp.float32)
      self.assertEqual(out_tp.sharding, NamedSharding(mesh, P("dp", "tp")))
      self.assertArraysEqual(out_tp, ref_tp, check_dtypes=True)

      r_tp_ar = ro.tiled(8, inner=inner, outer=ro.ring(direction, split_axis=1))
      out_tp_ar = jax.jit(lambda x, y: jnp.dot(
          x, y, preferred_element_type=jnp.float32,
          out_sharding=P("dp", None), reduction_order=r_tp_ar))(a, b)
      self.assertEqual(out_tp_ar.sharding, NamedSharding(mesh, P("dp", None)))
      self.assertArraysEqual(out_tp_ar, ref_tp, check_dtypes=True)

      u = jax.jit(lambda x, y: jnp.dot(
          x, y, preferred_element_type=jnp.float32,
          out_sharding=P("dp", None, unreduced={"tp"}),
          reduction_order=ro.tiled(8, "linear", "unreduced")))(a, b)
      self.assertEqual(
          u.sharding, NamedSharding(mesh, P("dp", None, unreduced={"tp"})))
      out_2stage = jax.jit(lambda z: jax.sharding.reshard(
          z, P("dp", "tp"), reduction_order=ro.ring(direction)))(u)
      u_partials = _single_device_eval(
          lambda x, y: jnp.einsum(
              "mtk,tkn->tmn", x.reshape(4, 4, 8), y.reshape(4, 8, 8),
              preferred_element_type=jnp.float32, reduction_order="linear",
          ),
          a_np, b_np,
      )
      ref_2stage = _ring_ref(np.asarray(u_partials), direction, split_axis=1)
      self.assertArraysEqual(out_2stage, ref_2stage, check_dtypes=True)

      with self.assertRaises(core.ShardingTypeError):
        jnp.sum(
            a, axis=1, out_sharding=P("dp", unreduced={"tp"}),
            reduction_order="unreduced")
      with self.assertRaises(core.ShardingTypeError):
        jnp.max(
            a, axis=1, out_sharding=P("dp", unreduced={"tp"}),
            reduction_order=ro.tiled(8, "linear", "unreduced"))
      with self.assertRaises(core.ShardingTypeError):
        jax.sharding.reshard(a, P("dp", None), reduction_order="unordered")
    finally:
      ro._INTRINSIC_IMPLS.pop("tpu_v7_mxu_256", None)

  @jtu.with_explicit_mesh((2, 4), ("dp", "tp"))
  def test_sharded_ragged_dot_orders(self, mesh):
    rng = jtu.rand_default(self.rng())
    a = jax.device_put(rng((4, 32), np.float32),
                       NamedSharding(mesh, P("dp", "tp")))
    b = jax.device_put(rng((2, 32, 8), np.float32),
                       NamedSharding(mesh, P(None, "tp", None)))
    gs = jax.device_put(np.array([1, 3], np.int32), NamedSharding(mesh, P()))
    rd = lambda x, y, s, out_s=P("dp", None), **kw: lax.ragged_dot(
        x, y, s, out_sharding=out_s, **kw)
    with self.assertRaises(core.ShardingTypeError):
      rd(a, b, gs, reduction_order="linear")
    with self.assertRaises(core.ShardingTypeError):
      rd(a, b, gs, reduction_order=ro.Intrinsic("test_hw"))

    orders = lambda f: [e.params.get("reduction_order")
                        for e in jax.make_jaxpr(f)(a, b, gs).eqns
                        if e.primitive.name == "ragged_dot_general"]
    xla = ro.Unordered("xla")
    self.assertEqual(orders(lambda x, y, s: rd(x, y, s, reduction_order=xla)),
                     [xla])
    u_s = P("dp", None, unreduced={"tp"})
    with self.assertRaises(NotImplementedError):
      jax.eval_shape(lambda x, y, s: rd(x, y, s, out_s=u_s), a, b, gs)
    with self.assertRaises(NotImplementedError):
      jax.eval_shape(lambda x, y, s: rd(x, y, s, out_s=u_s,
                                        reduction_order="unreduced"), a, b, gs)
    with self.assertRaisesRegex(NotImplementedError, "unreduced"):
      jax.eval_shape(lambda x, y, s: rd(x, y, s, reduction_order="unreduced"),
                     a, b, gs)
    tiled_u = ro.tiled(8, "linear", "unreduced")
    with self.assertRaisesRegex(NotImplementedError, "unreduced"):
      jax.eval_shape(lambda x, y, s: rd(x, y, s, reduction_order=tiled_u),
                     a, b, gs)

    order = ro.tiled(8, "linear", ro.ring("asc"))
    ring = lambda x, y, s: rd(x, y, s, out_s=P("dp", "tp"),
                              reduction_order=order)
    self.assertEqual(orders(ring), [order])
    with self.assertRaisesRegex(NotImplementedError, "no XLA lowering"):
      jax.jit(ring)(a, b, gs)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
