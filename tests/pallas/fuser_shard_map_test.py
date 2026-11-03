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
import functools
from absl.testing import absltest
from absl.testing import parameterized
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


def _run_pallas_fusible(
    in_fns_and_specs, out_fn, core_out_type, z_bs, grid, body_fn
):
  has_out_fn = out_fn is not None
  if out_fn is None:
    out_fn = lambda x: (x,)
  g_len = 0 if grid is None else len(grid)
  in_kfns, in_vals, in_sps, in_bss = [], [], [], []
  for fn, bs in in_fns_and_specs:
    fn2, vals, sp = fuser.get_fusion_values(fn)
    kfn, (val_bs,), _ = fuser.pull_block_spec(fn2, bs, grid_len=g_len)(vals)
    in_kfns.append(kfn)
    in_vals.append(vals)
    in_sps.append(sp)
    in_bss.append(val_bs)
  out_fn2, out_vals, out_sp = fuser.get_fusion_values(out_fn, core_out_type)
  out_kfn, out_val_bs, _, out_type, out_bs = fuser.push_pull_block_spec(
      out_fn2, z_bs, grid_len=g_len
  )(out_vals, core_out_type)
  if grid is None:
    ins = [k((), s, v) for k, s, v in zip(in_kfns, in_sps, in_vals)]
    res = out_kfn((), out_sp, out_vals, body_fn(*ins))
    return res if has_out_fn else res[0]
  load = lambda refs: jax.tree.map(lambda r: r[...], refs)

  def kernel(*refs):
    *in_val_refs, out_val_refs, o_ref = refs
    pids = tuple(pl.program_id(i) for i in range(g_len))
    ins = [k(pids, s, load(r)) for k, s, r in zip(in_kfns, in_sps, in_val_refs)]
    (o_ref[...],) = out_kfn(pids, out_sp, load(out_val_refs), body_fn(*ins))

  res = pl.pallas_call(
      kernel,
      out_shape=out_type,
      grid=grid,
      in_specs=(*in_bss, out_val_bs),
      out_specs=out_bs,
      interpret=not jtu.test_device_matches(["tpu"]),
  )(*in_vals, out_vals)
  return res if has_out_fn else res[0]


class ShardMapFusionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 devices")
    self.mesh = jtu.create_mesh(
        (2, 2), ("x", "y"), axis_types=(AxisType.Explicit,) * 2
    )

  def test_shard_map_with_tiled_pull_and_push_block_spec(self):
    @fuser.fusible
    def tiled_matmul(x_fn, y_fn, out_fn):
      m, k = x_fn.out_type.shape
      bm, bn = m // 2, y_fn.out_type.shape[1] // 2
      return _run_pallas_fusible(
          [
              (x_fn, pl.BlockSpec((bm, k), lambda i, j: (i, 0))),
              (y_fn, pl.BlockSpec((k, bn), lambda i, j: (0, j))),
          ],
          out_fn,
          jax.eval_shape(jnp.matmul, x_fn.out_type, y_fn.out_type),
          pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
          (2, 2),
          lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32),
      )

    def make_fn(check_vma=True):
      def fn(x, y, z, bias):
        iota = lax.broadcasted_iota(jnp.int32, x.shape, 0).astype(jnp.float32)
        x_pre = jnp.sin(x) * 2.0 + iota
        y_pre = lax.squeeze(lax.slice(y, (1, 0, 0), (2, 128, 512)), (0,)) + bias
        z_pre = z * 0.5
        shmapped = jax.shard_map(
            lambda a, b, c: tiled_matmul(a + 1.0, jnp.cos(b)) + c,
            mesh=self.mesh,
            in_specs=(P("x", None), P(None, "y"), P("x", "y")),
            out_specs=P("x", "y"),
            check_vma=check_vma,
        )
        return jnp.tanh(shmapped(x_pre, y_pre, z_pre)) * 0.5 - z_pre + bias

      return fn

    with jax.set_mesh(self.mesh):
      lin = lambda n: jnp.linspace(-1.0, 1.0, n, dtype=jnp.float32)
      x = jax.device_put(lin(32 * 128).reshape(32, 128), P("x", None))
      y = jax.device_put(lin(3 * 128 * 512).reshape(3, 128, 512), P(None, None, "y"))
      z = jax.device_put(jnp.full((32, 512), 0.25, jnp.float32), P("x", "y"))
      bias = jax.device_put(lin(512)[None, :], P(None, "y"))
      fused_jaxpr = jax.make_jaxpr(fuser.fuse(make_fn(True)))(x, y, z, bias).jaxpr
      self.assertEqual([e.primitive.name for e in fused_jaxpr.eqns], ["shard_map"])
      inner_eqns = fused_jaxpr.eqns[0].params["jaxpr"].eqns
      self.assertEqual([e.primitive.name for e in inner_eqns], ["pallas_call"])
      # CPU `pallas_call(interpret=True)` lowering does not propagate VMA.
      fn = make_fn(check_vma=jtu.test_device_matches(["tpu"]))
      np.testing.assert_allclose(
          jax.jit(fuser.fuse(fn))(x, y, z, bias),
          jax.jit(fn)(x, y, z, bias),
          rtol=1e-4,
          atol=1e-4,
      )

  @parameterized.named_parameters(
      ("replicated_outer_input", "replicated"),
      ("compound_axes_and_independent_output", "compound"),
      ("literal_operand", "literal"),
  )
  def test_shard_map_sharding_and_bypass_variants(self, mode):
    arr = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    shmap_xy = functools.partial(
        jax.shard_map, mesh=self.mesh, out_specs=P("x", "y")
    )
    with jax.set_mesh(self.mesh):
      if mode == "replicated":
        args = (
            jax.device_put(arr, P("x", "y")),
            jax.device_put(jnp.full((8, 8), 2.0, jnp.float32), P()),
            jax.device_put(jnp.full((8, 8), 3.0, jnp.float32), P()),
        )
        shmapped = shmap_xy(lambda x_s: _id_fusible(x_s + 1.0), in_specs=P("x", "y"))
        fn = lambda x, s, b: (shmapped(x * s) + b, s, jnp.int32(42))
      elif mode == "compound":
        spec = P(("x", "y"), None)
        args = (jax.device_put(arr, spec),)
        shmapped = jax.shard_map(
            lambda a: _id_fusible(a + 3.0),
            mesh=self.mesh,
            in_specs=spec,
            out_specs=spec,
        )
        fn = lambda x: (shmapped(x * 2.0 + 1.0) * 0.5, (x * 2.0 + 1.0) + 5.0)
      else:
        args = (jax.device_put(arr, P("x", "y")),)
        shmapped = shmap_xy(
            lambda x_s, c: _id_fusible(x_s * c), in_specs=(P("x", "y"), P())
        )
        fn = lambda x: shmapped(x, 2.0)
        self.assertIsInstance(
            jax.make_jaxpr(fn)(*args).eqns[-1].invars[1], jax_core.Literal
        )
      jax.tree.map(
          np.testing.assert_allclose,
          jax.jit(fuser.fuse(fn))(*args),
          jax.jit(fn)(*args),
      )

  def test_shard_map_separate_output_fusions_and_permutation(self):
    @fuser.fusible(output_fusion_prefix=(True, True))
    def multi_fusible(x_fn, y_fn, out_fns):
      fn1, fn2 = out_fns
      return (x_fn() + 1.0 if fn1 is None else fn1(x_fn() + 1.0)), (
          y_fn() + 2.0 if fn2 is None else fn2(y_fn() + 2.0)
      )

    def fn(x, y):
      specs = (P("x", None), P(None, "y"))
      shmapped = jax.shard_map(
          lambda a, b: tuple(
              v * s for v, s in zip(multi_fusible(a, b), (2.0, 5.0))
          ),
          mesh=self.mesh,
          in_specs=specs,
          out_specs=specs,
      )
      a_out, b_out = shmapped(x * 3.0, y * 4.0)
      return b_out - 1.0, (a_out + 10.0, a_out * 0.5)

    arr = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
    with jax.set_mesh(self.mesh):
      x, y = jax.device_put(arr, P("x", None)), jax.device_put(arr, P(None, "y"))
      jax.tree.map(
          np.testing.assert_allclose,
          jax.jit(fuser.fuse(fn))(x, y),
          jax.jit(fn)(x, y),
      )

  def test_shard_map_reshape_and_transpose_outer_ops(self):
    def fn(x):
      x_r = jnp.reshape(
          jnp.swapaxes(x, 1, 2), (32, 8), out_sharding=P("x", "y")
      )
      shmapped = jax.shard_map(
          lambda a: _id_fusible(a + 1.0) * 2.0,
          mesh=self.mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
      )
      return jnp.swapaxes(shmapped(x_r), 0, 1)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(256, dtype=jnp.float32).reshape(4, 8, 8), P("x", "y", None)
      )
      np.testing.assert_allclose(jax.jit(fuser.fuse(fn))(x), jax.jit(fn)(x))

  def test_shard_map_unsupported_cases_raise(self):
    shmap = functools.partial(
        jax.shard_map, mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
    )
    with jax.set_mesh(self.mesh):
      x = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32), P("x", "y"))
      with self.assertRaisesRegex(
          ValueError, "Only a single fusible is allowed"
      ):
        jax.jit(fuser.fuse(shmap(lambda x_s: _id_fusible(_id_fusible(x_s)))))(x)
      ref = jax.new_ref(jax.device_put(jnp.zeros((), jnp.float32), P()))

      @fuser.fuse
      def fn_ref_void(x):
        shmap(_id_fusible)(x)
        ref[...] = jnp.float32(1.0)

      with self.assertRaisesRegex(NotImplementedError, "Refs or tokens"):
        fn_ref_void(x)

      def effectful_body(x_s):
        jax.debug.callback(lambda _: None, x_s)
        return _id_fusible(x_s)

      with self.assertRaisesRegex(NotImplementedError, "side effects"):
        jax.jit(fuser.fuse(shmap(effectful_body)))(x)

    auto_mesh = jtu.create_mesh(
        (2, 2), ("x", "y"), axis_types=(AxisType.Auto,) * 2
    )
    shmap_auto = jax.shard_map(
        _id_fusible, mesh=auto_mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
    )
    with self.assertRaisesRegex(NotImplementedError, "Explicit mesh axes"):
      jax.jit(fuser.fuse(shmap_auto))(jnp.ones((8, 8), dtype=jnp.float32))

  def test_shard_map_with_fusion_dtype_physicalization(self):
    class PairFusionDType(fusible_dtype.FusionDType):

      def __init__(self, second_shape=None):
        self.second_shape = second_shape

      def __str__(self):
        return "pair_fusion_dtype"

      def abstract_unpack(self, x):
        s2 = x.shape if self.second_shape is None else self.second_shape
        return x.update(dtype=jnp.float32), x.update(shape=s2, dtype=jnp.float32)

      def abstract_pack(self, x, y):
        return x.update(dtype=self)

      pull_block_spec_one_step = lambda self, _, bs: (bs, bs)
      unpack_push_block_spec = lambda self, _, bs: (bs, bs)
      unpack_pull_block_spec = lambda self, _, bs1, bs2: (bs1,)
      pack_eval_rule = lambda self, _, x, y: fusible_dtype.pack(x, y, dtype=self)

    @fuser.fusible
    def inner_fusible(x_fn, out_fn):
      bs = pl.BlockSpec(x_fn.out_type.shape, lambda i: (i, 0))
      return _run_pallas_fusible(
          [(x_fn, bs)],
          out_fn,
          x_fn.out_type.update(dtype=jnp.float32),
          bs,
          (1,),
          lambda p: sum(fusible_dtype.unpack(p)),
      )

    def make_fn(check_vma=True):
      @jax.jit
      @fuser.fuse(resolve_fusion_dtypes=True)
      def fn(x, y):
        packed = fusible_dtype.pack(x * 2.0, y + 1.0, dtype=PairFusionDType())
        return (
            jax.shard_map(
                inner_fusible,
                mesh=self.mesh,
                in_specs=P("x", "y"),
                out_specs=P("x", "y"),
                check_vma=check_vma,
            )(packed)
            * 3.0
        )

      return fn

    def run_direct(dtype, spec, a_in, b_in):
      @jax.jit
      @fusible_dtype.physicalize
      def direct(a, b):
        p = jax.shard_map(
            lambda p_s: p_s, mesh=self.mesh, in_specs=spec, out_specs=spec
        )(fusible_dtype.pack(a, b, dtype=dtype))
        return sum(v.sum() for v in fusible_dtype.unpack(p))

      return direct(a_in, b_in)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(4096, dtype=jnp.float32).reshape(16, 256) / 256.0,
          P("x", "y"),
      )
      y = jax.device_put(jnp.ones((16, 256), dtype=jnp.float32), P("x", "y"))
      fused = jax.make_jaxpr(make_fn(True))(x, y).jaxpr.eqns[0].params["jaxpr"]
      self.assertEqual([e.primitive.name for e in fused.eqns], ["shard_map"])
      inner_eqns = fused.eqns[0].params["jaxpr"].eqns
      self.assertEqual([e.primitive.name for e in inner_eqns], ["pallas_call"])
      fn = make_fn(check_vma=jtu.test_device_matches(["tpu"]))
      np.testing.assert_allclose(fn(x, y), (x * 2.0 + (y + 1.0)) * 3.0)
      np.testing.assert_allclose(
          run_direct(PairFusionDType(), P("x", "y"), x, y), x.sum() + y.sum()
      )
      x_rep = jax.device_put(x, P())
      s_rep = jax.device_put(jnp.ones((16, 2), jnp.float32), P())
      mismatched_dtype = PairFusionDType(second_shape=(16, 2))
      np.testing.assert_allclose(
          run_direct(mismatched_dtype, P(), x_rep, s_rep),
          x_rep.sum() + s_rep.sum(),
      )
      with self.assertRaisesRegex(
          NotImplementedError, "requires unpacked leaves to match logical shape"
      ):
        run_direct(
            mismatched_dtype, P("x", "y"), x, jax.device_put(s_rep, P("x", "y"))
        )

  def test_shard_map_custom_jvp(self):
    @fuser.fusible
    def block_fusible(x_fn, out_fn):
      bs = pl.BlockSpec(x_fn.shape, lambda: (0,) * len(x_fn.shape))
      return _run_pallas_fusible(
          [(x_fn, bs)], out_fn, x_fn.out_type, bs, None, lambda v: v
      )

    @jax.custom_jvp
    def ste_clip_with_sign(x):
      return jnp.clip(x, -2.0, 2.0), (x > 0).astype(jnp.int32)

    @functools.partial(ste_clip_with_sign.defjvp, symbolic_zeros=True)
    def _(primals, tangents):
      p_out = ste_clip_with_sign(primals[0])
      z = jax.custom_derivatives.SymbolicZero(
          jax.typeof(p_out[1]).to_tangent_aval()
      )
      return p_out, (tangents[0], z)

    def fn(x, rep_bias):
      y, sign_pre = ste_clip_with_sign(x)
      shmapped = jax.shard_map(
          lambda a, b: (
              (r := ste_clip_with_sign(block_fusible(a * 1.5 + b)))[0] + b,
              r[1],
          ),
          mesh=self.mesh,
          in_specs=(P("x", "y"), P()),
          out_specs=(P("x", "y"), P("x", "y")),
      )
      z, sign_post = shmapped(y, rep_bias)
      return jax.nn.relu(z) + sign_pre.astype(jnp.float32), sign_post

    fused_fn, ref_fn = jax.jit(fuser.fuse(fn)), jax.jit(fn)
    with jax.set_mesh(self.mesh):
      lin = jnp.linspace(-4.0, 4.0, 64, dtype=jnp.float32).reshape(8, 8)
      x = jax.device_put(lin, P("x", "y"))
      t = jax.device_put(jnp.ones_like(x) * 0.5, P("x", "y"))
      b, tb = jax.device_put(0.25, P()), jax.device_put(0.1, P())
      (p_f, (t_f, z_f)) = jax.jvp(fused_fn, (x, b), (t, tb))
      (p_r, (t_r, z_r)) = jax.jvp(ref_fn, (x, b), (t, tb))
      jax.tree.map(np.testing.assert_allclose, (p_f, t_f), (p_r, t_r))
      self.assertEqual((z_f.shape, z_f.dtype), (z_r.shape, z_r.dtype))
      out_f, vjp_f = jax.vjp(lambda a, c: fused_fn(a, c)[0], x, b)
      out_r, vjp_r = jax.vjp(lambda a, c: ref_fn(a, c)[0], x, b)
      jax.tree.map(
          np.testing.assert_allclose,
          (out_f, vjp_f(jnp.ones_like(out_f))),
          (out_r, vjp_r(jnp.ones_like(out_r))),
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
