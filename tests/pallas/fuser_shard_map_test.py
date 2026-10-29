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
import jax
from jax import lax
from jax._src import core as jax_core
from jax._src import test_util as jtu
from jax._src.pallas.fuser import fusible_dtype
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P
import numpy as np

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(4)


@fuser.fusible
def _id_fusible(x_fn, out_fn):
  x = x_fn()
  return x if out_fn is None else out_fn(x)


class ShardMapFusionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 devices")
    self.mesh = jtu.create_mesh(
        (2, 2), ("x", "y"), axis_types=(AxisType.Explicit,) * 2
    )

  def test_shard_map_all_four_regions(self):
    @fuser.fusible
    def add_fusible(x_fn, y_fn, out_fn):
      res = x_fn() + y_fn()
      return res if out_fn is None else out_fn(res)

    def fn(x, y, z, bias):
      iota = lax.broadcasted_iota(jnp.int32, x.shape, 0).astype(jnp.float32)
      x_pre = jnp.sin(x) * 2.0 + iota
      y_pre = (
          lax.squeeze(lax.slice(y, (1, 0, 0), (2, 8, 8)), (0,)) + bias[None, :]
      )

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(P("x", "y"), P("x", "y"), P("x", "y")),
          out_specs=P("x", "y"),
      )
      def shmapped(x_s, y_s, z_s):
        return (add_fusible(x_s + 1.0, jnp.cos(y_s)) - z_s) * 0.5

      return jnp.tanh(shmapped(x_pre, y_pre, z)) + x_pre * 0.25 + bias[None, :]

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 64.0, P("x", "y")
      )
      y = jax.device_put(
          jnp.arange(192, dtype=jnp.float32).reshape(3, 8, 8) / 192.0,
          P(None, "x", "y"),
      )
      z = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32) * 0.1, P("x", "y"))
      bias = jax.device_put(jnp.arange(8, dtype=jnp.float32), P("y"))
      np.testing.assert_allclose(
          jax.jit(fuser.fuse(fn))(x, y, z, bias),
          jax.jit(fn)(x, y, z, bias),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_shard_map_with_tiled_pull_and_push_block_spec(self):
    @fuser.fusible
    def tiled_matmul(x_fn, y_fn, out_fn):
      has_out_fn = out_fn is not None
      if out_fn is None:
        out_fn = lambda x: (x,)
      m, k = x_fn.out_type.shape
      _, n = y_fn.out_type.shape
      bm, bk, bn = m // 2, k, n // 2
      grid = (2, 2)
      x_bs = pl.BlockSpec((bm, bk), lambda i, j: (i, 0))
      y_bs = pl.BlockSpec((bk, bn), lambda i, j: (0, j))
      z_bs = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
      mm_out_type = jax.ShapeDtypeStruct((m, n), x_fn.out_type.dtype)
      x_fn2, x_vals, x_sp = fuser.get_fusion_values(x_fn)
      y_fn2, y_vals, y_sp = fuser.get_fusion_values(y_fn)
      out_fn2, out_vals, out_sp = fuser.get_fusion_values(out_fn, mm_out_type)
      sp_handler = fuser.make_scalar_prefetch_handler(0)
      x_kfn, (x_val_bs,), _ = fuser.pull_block_spec(
          x_fn2, x_bs, scalar_prefetch_handler=sp_handler, grid_len=len(grid)
      )(x_vals)
      y_kfn, (y_val_bs,), _ = fuser.pull_block_spec(
          y_fn2, y_bs, scalar_prefetch_handler=sp_handler, grid_len=len(grid)
      )(y_vals)
      out_kfn, out_val_bs, _, out_type, out_bs = fuser.push_pull_block_spec(
          out_fn2,
          z_bs,
          scalar_prefetch_handler=sp_handler,
          grid_len=len(grid),
      )(out_vals, mm_out_type)

      def kernel(x_val_refs, y_val_refs, out_val_refs, o_ref):
        pids = (pl.program_id(0), pl.program_id(1))
        x_t = x_kfn(pids, x_sp, jax.tree.map(lambda r: r[...], x_val_refs))
        y_t = y_kfn(pids, y_sp, jax.tree.map(lambda r: r[...], y_val_refs))
        (o_ref[...],) = out_kfn(
            pids,
            out_sp,
            jax.tree.map(lambda r: r[...], out_val_refs),
            x_t @ y_t,
        )

      res = pl.pallas_call(
          kernel,
          out_shape=out_type,
          grid=grid,
          in_specs=(x_val_bs, y_val_bs, out_val_bs),
          out_specs=out_bs,
          interpret=True,
      )(x_vals, y_vals, out_vals)
      return res if has_out_fn else res[0]

    def fn(x, y, z):
      iota = lax.broadcasted_iota(jnp.int32, x.shape, 0).astype(jnp.float32)
      x_pre = x * 2.0 + iota
      y_pre = lax.squeeze(lax.slice(y, (1, 0, 0), (2, 8, 16)), (0,))

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(P("x", None), P(None, "y"), P("x", "y")),
          out_specs=P("x", "y"),
          check_vma=False,
      )
      def shmapped(x_s, y_s, z_s):
        return tiled_matmul(x_s + 1.0, y_s) + z_s

      return shmapped(x_pre, y_pre, z) * 0.5 - z

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(128, dtype=jnp.float32).reshape(16, 8) / 128.0,
          P("x", None),
      )
      y = jax.device_put(
          jnp.arange(384, dtype=jnp.float32).reshape(3, 8, 16) / 384.0,
          P(None, None, "y"),
      )
      z = jax.device_put(
          jnp.ones((16, 16), dtype=jnp.float32) * 0.25, P("x", "y")
      )
      fused_fn = fuser.fuse(fn)
      fused_jaxpr = jax.make_jaxpr(fused_fn)(x, y, z).jaxpr
      self.assertLen(fused_jaxpr.eqns, 1)
      self.assertEqual(fused_jaxpr.eqns[0].primitive.name, "shard_map")
      inner_prims = [
          e.primitive.name for e in fused_jaxpr.eqns[0].params["jaxpr"].eqns
      ]
      self.assertEqual(inner_prims, ["pallas_call"])
      np.testing.assert_allclose(
          jax.jit(fused_fn)(x, y, z),
          jax.jit(fn)(x, y, z),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_shard_map_replicated_outer_input_in_sharded_op(self):
    def fn(x, rep_scale, rep_shift):
      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return _id_fusible(x_s + 1.0)

      return shmapped(x * rep_scale) + rep_shift, rep_scale

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", "y")
      )
      rep_scale = jax.device_put(
          jnp.full((8, 8), 2.0, dtype=jnp.float32), P(None, None)
      )
      rep_shift = jax.device_put(
          jnp.full((8, 8), 3.0, dtype=jnp.float32), P(None, None)
      )
      exp_out, exp_bypass = jax.jit(fn)(x, rep_scale, rep_shift)
      act_out, act_bypass = jax.jit(fuser.fuse(fn))(x, rep_scale, rep_shift)
      np.testing.assert_allclose(act_out, exp_out)
      np.testing.assert_allclose(act_bypass, exp_bypass)

  def test_shard_map_separate_output_fusions_and_permutation(self):
    @fuser.fusible(output_fusion_prefix=(True, True))
    def multi_fusible(x_fn, y_fn, out_fns):
      fn1, fn2 = out_fns
      return (x_fn() + 1.0 if fn1 is None else fn1(x_fn() + 1.0)), (
          y_fn() + 2.0 if fn2 is None else fn2(y_fn() + 2.0)
      )

    def fn(x, y):
      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(P("x", None), P(None, "y")),
          out_specs=(P("x", None), P(None, "y")),
      )
      def shmapped(x_s, y_s):
        a, b = multi_fusible(x_s, y_s)
        return a * 2.0, b * 5.0

      a_out, b_out = shmapped(x * 3.0, y * 4.0)
      return b_out - 1.0, (a_out + 10.0, a_out * 0.5)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", None)
      )
      y = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P(None, "y")
      )
      b_exp, (a1_exp, a2_exp) = jax.jit(fn)(x, y)
      b_act, (a1_act, a2_act) = jax.jit(fuser.fuse(fn))(x, y)
      np.testing.assert_allclose(b_act, b_exp)
      np.testing.assert_allclose(a1_act, a1_exp)
      np.testing.assert_allclose(a2_act, a2_exp)

  def test_shard_map_reshape_and_transpose_outer_ops(self):
    def fn(x):
      x_r = jnp.reshape(
          jnp.swapaxes(x, 1, 2), (32, 8), out_sharding=P("x", "y")
      )

      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return _id_fusible(x_s + 1.0) * 2.0

      return jnp.swapaxes(shmapped(x_r), 0, 1)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(256, dtype=jnp.float32).reshape(4, 8, 8), P("x", "y", None)
      )
      np.testing.assert_allclose(jax.jit(fuser.fuse(fn))(x), jax.jit(fn)(x))

  def test_shard_map_compound_axes_and_independent_outer_output(self):
    def fn(x):
      x_pre = x * 2.0 + 1.0

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=P(("x", "y"), None),
          out_specs=P(("x", "y"), None),
      )
      def shmapped(x_s):
        return _id_fusible(x_s + 3.0)

      return shmapped(x_pre) * 0.5, x_pre + 5.0

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P(("x", "y"), None)
      )
      exp_shmap, exp_indep = jax.jit(fn)(x)
      act_shmap, act_indep = jax.jit(fuser.fuse(fn))(x)
      np.testing.assert_allclose(act_shmap, exp_shmap)
      np.testing.assert_allclose(act_indep, exp_indep)

  def test_shard_map_multiple_fusibles_raises(self):
    @jax.jit
    @fuser.fuse
    def fn_two_in_shmap(x):
      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return _id_fusible(_id_fusible(x_s))

      return shmapped(x)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32), P("x", "y"))
      with self.assertRaisesRegex(
          ValueError,
          "Only a single fusible is allowed in a fused function, but found 2",
      ):
        fn_two_in_shmap(x)

  def test_shard_map_auto_axes_raises(self):
    mesh = jtu.create_mesh((2, 2), ("x", "y"), axis_types=(AxisType.Auto,) * 2)

    @jax.jit
    @fuser.fuse
    def fn(x):
      return jax.shard_map(
          _id_fusible,
          mesh=mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
      )(x)

    with self.assertRaisesRegex(NotImplementedError, "Explicit mesh axes"):
      fn(jnp.ones((8, 8), dtype=jnp.float32))

  def test_shard_map_dynamic_slice_scalar_prefetch_and_literals(self):
    def fn(x, idx):
      zero = jnp.int32(0)
      x_squeezed = lax.squeeze(
          lax.dynamic_slice(x, (idx, zero, zero), (1, 8, 8)), (0,)
      )

      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return _id_fusible(x_s * 2.0) + 1.0

      return shmapped(x_squeezed), jnp.int32(42)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(192, dtype=jnp.float32).reshape(3, 8, 8), P(None, "x", "y")
      )
      idx = jax.device_put(jnp.int32(2), P())
      exp_out, exp_lit = jax.jit(fn)(x, idx)
      act_out, act_lit = jax.jit(fuser.fuse(fn))(x, idx)
      np.testing.assert_allclose(act_out, exp_out)
      np.testing.assert_array_equal(act_lit, exp_lit)

  def test_shard_map_literal_operand(self):
    def fn(x):
      @jax.shard_map(
          mesh=self.mesh, in_specs=(P("x", "y"), P()), out_specs=P("x", "y")
      )
      def shmapped(x_s, c):
        return _id_fusible(x_s * c)

      return shmapped(x, 2.0)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", "y")
      )
      shmap_eqn = jax.make_jaxpr(fn)(x).eqns[-1]
      self.assertIsInstance(shmap_eqn.invars[1], jax_core.Literal)
      np.testing.assert_allclose(jax.jit(fuser.fuse(fn))(x), jax.jit(fn)(x))

  def test_shard_map_with_fusion_dtype_physicalization(self):
    class PairFusionDType(fusible_dtype.FusionDType):

      def __str__(self):
        return "pair_fusion_dtype"

      def abstract_unpack(self, x):
        return (x.update(dtype=jnp.float32), x.update(dtype=jnp.float32))

      def abstract_pack(self, x, y):
        del y
        return x.update(dtype=self)

      def pull_block_spec_one_step(self, aval_out, block_spec):
        del aval_out
        return block_spec, block_spec

      def unpack_push_block_spec(self, aval_in, block_spec):
        del aval_in
        return block_spec, block_spec

      def unpack_pull_block_spec(self, aval_in, block_spec1, block_spec2):
        del aval_in, block_spec2
        return (block_spec1,)

      def pack_eval_rule(self, ctx, x, y):
        del ctx
        return fusible_dtype.pack(x, y, dtype=self)

    @fuser.fusible
    def inner_fusible(x_fn, out_fn):
      has_out_fn = out_fn is not None
      if out_fn is None:
        out_fn = lambda x: (x,)
      bs = pl.BlockSpec(x_fn.out_type.shape, lambda i: (i, 0))
      out_type = jax.ShapeDtypeStruct(x_fn.out_type.shape, jnp.float32)
      x_fn2, x_vals, x_sp = fuser.get_fusion_values(x_fn)
      out_fn2, out_vals, out_sp = fuser.get_fusion_values(out_fn, out_type)
      sp_handler = fuser.make_scalar_prefetch_handler(0)
      x_kfn, (x_val_bs,), _ = fuser.pull_block_spec(
          x_fn2, bs, scalar_prefetch_handler=sp_handler, grid_len=1
      )(x_vals)
      out_kfn, out_val_bs, _, z_out_type, out_bs = fuser.push_pull_block_spec(
          out_fn2, bs, scalar_prefetch_handler=sp_handler, grid_len=1
      )(out_vals, out_type)

      def kernel(x_val_refs, out_val_refs, o_ref):
        pids = (pl.program_id(0),)
        u, v = fusible_dtype.unpack(
            x_kfn(pids, x_sp, jax.tree.map(lambda r: r[...], x_val_refs))
        )
        (o_ref[...],) = out_kfn(
            pids, out_sp, jax.tree.map(lambda r: r[...], out_val_refs), u + v
        )

      res = pl.pallas_call(
          kernel,
          out_shape=z_out_type,
          grid=(1,),
          in_specs=(x_val_bs, out_val_bs),
          out_specs=out_bs,
          interpret=True,
      )(x_vals, out_vals)
      return res if has_out_fn else res[0]

    @jax.jit
    @fuser.fuse(resolve_fusion_dtypes=True)
    def fn(x, y):
      packed = fusible_dtype.pack(x * 2.0, y + 1.0, dtype=PairFusionDType())

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
          check_vma=False,
      )
      def shmapped(p_s):
        return inner_fusible(p_s)

      return shmapped(packed) * 3.0

    @jax.jit
    @fusible_dtype.physicalize
    def direct_shmap(x, y):
      p = fusible_dtype.pack(x, y, dtype=PairFusionDType())
      p_out = jax.shard_map(
          lambda p_s: p_s,
          mesh=self.mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
      )(p)
      u, v = fusible_dtype.unpack(p_out)
      return u - v

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", "y")
      )
      y = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32), P("x", "y"))
      fused_jaxpr = jax.make_jaxpr(fn)(x, y).jaxpr.eqns[0].params["jaxpr"]
      self.assertLen(fused_jaxpr.eqns, 1)
      self.assertEqual(fused_jaxpr.eqns[0].primitive.name, "shard_map")
      inner_prims = [
          e.primitive.name for e in fused_jaxpr.eqns[0].params["jaxpr"].eqns
      ]
      self.assertEqual(inner_prims, ["pallas_call"])
      np.testing.assert_allclose(fn(x, y), (x * 2.0 + (y + 1.0)) * 3.0)
      np.testing.assert_allclose(direct_shmap(x, y), x - y)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
