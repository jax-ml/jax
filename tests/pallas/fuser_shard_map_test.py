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
from jax._src import tpu_info
from jax._src.pallas.fuser import fusible_dtype
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P
import numpy as np

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(4)
tpu_info.registry.setdefault(
    "cpu",
    lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V6E, 1),
)


def _run_kernel_fusible(
    in_fns_and_specs, out_fn, core_out_type, z_bs, grid, body_fn
):
  has_out_fn = out_fn is not None
  if out_fn is None:
    out_fn = lambda x: (x,)
  g_len = 0 if grid is None else len(grid)
  in_fusions = [fuser.get_fusion_values(fn) for fn, _ in in_fns_and_specs]
  out_fn2, out_vals, out_sp = fuser.get_fusion_values(out_fn, core_out_type)
  all_sp = (*(sp for _, _, sp in in_fusions), out_sp)
  has_sp = bool(jax.tree.leaves(all_sp))
  sp_h = fuser.make_scalar_prefetch_handler if has_sp else lambda _: None
  in_kfns, in_vals, in_sps, in_bss = [], [], [], []
  for idx, ((_, bs), (fn2, vals, sp)) in enumerate(
      zip(in_fns_and_specs, in_fusions)
  ):
    kfn, (val_bs,), _ = fuser.pull_block_spec(
        fn2, bs, scalar_prefetch_handler=sp_h(idx), grid_len=g_len
    )(vals)
    in_kfns.append(kfn)
    in_vals.append(vals)
    in_sps.append(sp)
    in_bss.append(val_bs)
  out_kfn, out_val_bs, _, out_type, out_bs = fuser.push_pull_block_spec(
      out_fn2,
      z_bs,
      scalar_prefetch_handler=sp_h(len(in_fns_and_specs)),
      grid_len=g_len,
  )(out_vals, core_out_type)
  if grid is None:
    ins = [
        k((), all_sp if has_sp else s, v)
        for k, s, v in zip(in_kfns, in_sps, in_vals)
    ]
    res = out_kfn(
        (), all_sp if has_sp else out_sp, out_vals, body_fn(*ins)
    )
    return res if has_out_fn else res[0]
  load = lambda refs: jax.tree.map(lambda r: r[...], refs)
  sp_1d = jax.tree.map(lambda x: x[None], all_sp) if has_sp else ()

  def kernel(*refs):
    if has_sp:
      *in_val_refs, out_val_refs, sp_refs, o_ref, sp_smem = refs
      pltpu.sync_copy(sp_refs, sp_smem)
      in_sp_args = [sp_smem] * len(in_kfns)
      out_sp_arg = sp_smem
    else:
      *in_val_refs, out_val_refs, o_ref = refs
      in_sp_args = in_sps
      out_sp_arg = out_sp

    def step_body(step, *step_refs):
      *step_in_refs, step_out_val_refs, step_o_ref = step_refs
      ins = [
          k(step.index, s, load(r))
          for k, s, r in zip(in_kfns, in_sp_args, step_in_refs)
      ]
      out = out_kfn(
          step.index, out_sp_arg, load(step_out_val_refs), body_fn(*ins)
      )
      jax.tree.map(lambda ref, x: ref.set(x), step_o_ref, out)

    in_specs = (*in_bss, out_val_bs)
    out_specs = (out_bs,)
    if has_sp:
      in_specs = fuser.block_spec_with_prefetch(in_specs, sp_smem)
      out_specs = fuser.block_spec_with_prefetch(out_specs, sp_smem)
    pltpu.emit_pipeline(
        step_body,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        core_axis_name="core",
        _explicit_indices=True,
    )(*in_val_refs, out_val_refs, o_ref)

  args = (*in_vals, out_vals, *((sp_1d,) if has_sp else ()))
  scratch = [jax.tree.map(pltpu.SMEM.like, sp_1d)] if has_sp else []
  (res,) = pl.kernel(
      kernel,
      out_type=(out_type,),
      mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
      scratch_types=scratch,
      interpret=(
          pltpu.InterpretParams()
          if not jtu.test_device_matches(["tpu"])
          else False
      ),
  )(*args)
  return res if has_out_fn else res[0]


@fuser.fusible
def _id_fusible(x_fn, out_fn):
  shape = x_fn.out_type.shape
  if not shape:
    x = x_fn()
    return x if out_fn is None else out_fn(x)
  bs = pl.BlockSpec(shape, lambda i, *_: (0,) * len(shape))
  return _run_kernel_fusible(
      [(x_fn, bs)], out_fn, x_fn.out_type, bs, (1,), lambda x: x
  )


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
      return _run_kernel_fusible(
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
      self.assertEqual([e.primitive.name for e in inner_eqns], ["mpmd_map"])
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
    arr = jnp.arange(16 * 256, dtype=jnp.float32).reshape(16, 256)
    cv = jtu.test_device_matches(["tpu"])
    shmap_xy = functools.partial(
        jax.shard_map, mesh=self.mesh, out_specs=P("x", "y"), check_vma=cv
    )
    with jax.set_mesh(self.mesh):
      if mode == "replicated":
        args = (
            jax.device_put(arr, P("x", "y")),
            jax.device_put(jnp.full((16, 256), 2.0, jnp.float32), P()),
            jax.device_put(jnp.full((16, 256), 3.0, jnp.float32), P()),
        )
        shmapped = shmap_xy(lambda x_s: _id_fusible(x_s + 1.0), in_specs=P("x", "y"))
        fn = lambda x, s, b: (shmapped(x * s) + b, s, jnp.int32(42))
      elif mode == "compound":
        spec = P(("x", "y"), None)
        args = (jax.device_put(arr, spec),)
        shmapped = jax.shard_map(
            lambda a: _id_fusible(a + 3.0), mesh=self.mesh, in_specs=spec, out_specs=spec
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
      bs_x = pl.BlockSpec(x_fn.out_type.shape, lambda i: (0, 0))
      bs_y = pl.BlockSpec(y_fn.out_type.shape, lambda i: (0, 0))
      out1 = _run_kernel_fusible(
          [(x_fn, bs_x)], fn1, x_fn.out_type, bs_x, (1,), lambda x: x + 1.0
      )
      out2 = _run_kernel_fusible(
          [(y_fn, bs_y)], fn2, y_fn.out_type, bs_y, (1,), lambda y: y + 2.0
      )
      return out1, out2

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
      return _run_kernel_fusible(
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
      self.assertEqual([e.primitive.name for e in inner_eqns], ["mpmd_map"])
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
      return _run_kernel_fusible(
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


@fuser.fusible
def _kernel_scale_fusible(x_fn, out_fn):
  bs = pl.BlockSpec(x_fn.out_type.shape, lambda i: (0, 0))
  return _run_kernel_fusible(
      [(x_fn, bs)],
      out_fn,
      jax.ShapeDtypeStruct(x_fn.out_type.shape, x_fn.out_type.dtype),
      bs,
      (1,),
      lambda x: x * 2.0,
  )


class ShardMapFusionBlockContainmentTest(jtu.JaxTestCase):
  """Tests for shard containment and multi-fusible validation in shard_map fusion."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 devices")
    self.mesh_1d = jtu.create_mesh(
        (2,), ("x",), axis_types=(AxisType.Explicit,)
    )
    self.mesh = jtu.create_mesh(
        (2, 2), ("x", "y"), axis_types=(AxisType.Explicit,) * 2
    )

  @parameterized.named_parameters(
      ("1d", (512,), P("x"), 128, 384, _id_fusible),
      ("2d_scale", (64, 128), P("x", None), 16, 48, _kernel_scale_fusible),
  )
  def test_shard_map_centered_slice_prologue_within_shards(
      self, shape, spec, start, stop, kernel_fn
  ):
    # Static centered slice on P("x") over 2 devices: each device's slice lies
    # within its local shard, so local shard extraction succeeds.
    make_fn = lambda cv: (
        lambda x: jax.shard_map(
            kernel_fn, mesh=self.mesh_1d, in_specs=spec, out_specs=spec, check_vma=cv
        )(x[start:stop])
    )
    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(
          jnp.arange(int(np.prod(shape)), dtype=jnp.float32).reshape(shape), spec
      )
      jax.make_jaxpr(fuser.fuse(make_fn(True)))(x)
      fn = make_fn(jtu.test_device_matches(["tpu"]))
      self.assertAllClose(jax.jit(fuser.fuse(fn))(x), jax.jit(fn)(x))

      if shape == (512,):
        # Inner scalar prefetch passed explicitly into `fuse` (not closed over).
        start_idx = jnp.int32(128)
        make_sp_fn = lambda cv: (
            lambda y, s: jax.shard_map(
                lambda y_s, s_s: _id_fusible(
                    lax.dynamic_slice_in_dim(
                        y_s, s_s - lax.axis_index("x") * 128, 128
                    )
                ),
                mesh=self.mesh_1d,
                in_specs=(P("x"), P()),
                out_specs=P("x"),
                check_vma=cv,
            )(y * 2.0, s)
        )
        jax.make_jaxpr(fuser.fuse(make_sp_fn(True)))(x, start_idx)
        sp_fn = make_sp_fn(jtu.test_device_matches(["tpu"]))
        self.assertAllClose(
            jax.jit(fuser.fuse(sp_fn))(x, start_idx),
            jax.jit(sp_fn)(x, start_idx),
        )

  @parameterized.named_parameters(
      ("prologue", lambda shmap, x: shmap(x.sum(axis=1)), False),
      ("epilogue", lambda shmap, x: shmap(x).sum(axis=1), False),
      (
          "epilogue_keepdims",
          lambda shmap, x: shmap(x).sum(axis=1, keepdims=True),
          True,
      ),
  )
  def test_shard_map_reduction_over_unsharded_axis(self, apply_fn, keepdims):
    if (
        not keepdims
        and jtu.test_device_matches(["tpu"])
        and not jtu.is_device_tpu_at_least(5)
    ):
      self.skipTest(
          "Requires TPUv5+ for minor-axis reduction with keepdims=False"
      )

    def fn(x):
      return apply_fn(
          lambda y: jax.shard_map(
              _id_fusible,
              mesh=self.mesh_1d,
              in_specs=P("x") if y.ndim == 1 else P("x", None),
              out_specs=P("x") if y.ndim == 1 else P("x", None),
          )(y),
          x,
      )

    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(
          jnp.arange(32, dtype=jnp.float32).reshape(8, 4), P("x", None)
      )
      self.assertAllClose(jax.jit(fuser.fuse(fn))(x), jax.jit(fn)(x))

  @parameterized.named_parameters(
      ("fusible_check_vma_true", _id_fusible, True),
      ("kernel", _kernel_scale_fusible, False),
  )
  def test_shard_map_reshape_out_sharding_cross_axis_swap(
      self, kernel_fn, check_vma
  ):
    def fn(x):
      x_resharded = jnp.reshape(x, (16, 16), out_sharding=P("y", "x"))
      return jax.shard_map(
          kernel_fn,
          mesh=self.mesh,
          in_specs=P("y", "x"),
          out_specs=P("y", "x"),
          check_vma=check_vma,
      )(x_resharded)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(256, dtype=jnp.float32).reshape(16, 16), P("x", "y")
      )
      with self.assertRaisesRegex(
          NotImplementedError, "lies outside local shard"
      ):
        jax.jit(fuser.fuse(fn))(x)

  @parameterized.named_parameters(
      ("prologue_positive_offset_clamp", lambda shmap, x: shmap(x[4:8])),
      ("epilogue_positive_offset_clamp", lambda shmap, x: shmap(x)[4:8]),
      (
          "prologue_dynamic_slice_positive_offset_clamp",
          lambda shmap, x: shmap(lax.dynamic_slice_in_dim(x, 4, 4)),
      ),
      ("prologue_negative_local_offset_wrap", lambda shmap, x: shmap(x[0:4])),
  )
  def test_shard_map_cross_shard_slice_1d(self, apply_fn):
    def fn(x):
      shmap = jax.shard_map(
          _id_fusible, mesh=self.mesh_1d, in_specs=P("x"), out_specs=P("x")
      )
      return apply_fn(shmap, x)

    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(jnp.arange(8, dtype=jnp.float32), P("x"))
      with self.assertRaisesRegex(
          NotImplementedError, "lies outside local shard"
      ):
        jax.jit(fuser.fuse(fn))(x)

  @parameterized.named_parameters(
      ("compound_axes", 16, P(("x", "y")), set()),
      ("partial_manual_axes", 8, P("x"), {"x"}),
  )
  def test_shard_map_cross_shard_slice_2d_mesh_variants(
      self, size, spec, axis_names
  ):
    fn = lambda x: jax.shard_map(
        _id_fusible, mesh=self.mesh, in_specs=spec, out_specs=spec, axis_names=axis_names
    )(x[4:8])
    with jax.set_mesh(self.mesh):
      x = jax.device_put(jnp.arange(size, dtype=jnp.float32), spec)
      with self.assertRaisesRegex(
          NotImplementedError, "lies outside local shard"
      ):
        jax.jit(fuser.fuse(fn))(x)

  def test_shard_map_kernel_cross_shard_slice_prologue(self):
    def fn(x):
      return jax.shard_map(
          _kernel_scale_fusible,
          mesh=self.mesh_1d,
          in_specs=P("x", None),
          out_specs=P("x", None),
          check_vma=False,
      )(x[16:32])

    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(
          jnp.arange(32 * 128, dtype=jnp.float32).reshape(32, 128),
          P("x", None),
      )
      with self.assertRaisesRegex(
          NotImplementedError, "lies outside local shard"
      ):
        jax.jit(fuser.fuse(fn))(x)

  @parameterized.named_parameters(
      ("check_vma_false", False, P("x")),
      ("check_vma_true", True, P()),
  )
  def test_shard_map_size1_slice_static_shortcut(self, check_vma, out_specs):
    def fn(x):
      return jax.shard_map(
          _id_fusible,
          mesh=self.mesh_1d,
          in_specs=P(),
          out_specs=out_specs,
          check_vma=check_vma,
      )(x[1:2])

    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(jnp.arange(2, dtype=jnp.float32), P("x"))
      with self.assertRaisesRegex(
          NotImplementedError, "lies outside local shard"
      ):
        jax.jit(fuser.fuse(fn))(x)

  @parameterized.named_parameters(
      ("sharded_dim", 0),
      ("unsharded_dim", 1),
  )
  def test_shard_map_outer_scalar_prefetch_raises(self, axis):
    fn = lambda x, start: jax.shard_map(
        _id_fusible, mesh=self.mesh_1d, in_specs=P("x", None), out_specs=P("x", None)
    )(lax.dynamic_slice_in_dim(x, start, 4, axis=axis))
    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", None)
      )
      start = jnp.int32(2)
      for call in (
          lambda: jax.jit(fuser.fuse(fn))(x, start),
          lambda: jax.jit(fuser.fuse(lambda y: fn(y, start)))(x),
      ):
        with self.assertRaisesRegex(
            NotImplementedError, "scalar prefetch in outer operations"
        ):
          call()

  def test_shard_map_prologue_reduction_over_sharded_axis(self):
    def fn(x):
      return jax.shard_map(
          _id_fusible, mesh=self.mesh_1d, in_specs=P(), out_specs=P()
      )(x.sum(axis=0))

    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(
          jnp.arange(32, dtype=jnp.float32).reshape(8, 4), P("x", None)
      )
      with self.assertRaisesRegex(
          NotImplementedError, "exceeds local shard size"
      ):
        jax.jit(fuser.fuse(fn))(x)

  @parameterized.named_parameters(
      (
          "1d_keepdims",
          (8,),
          P("x"),
          lambda y: y.sum(keepdims=True),
          (
              "Scalar prefetch values in a fusion cannot depend on the fusion"
              " inputs"
          ),
      ),
      (
          "2d_sharded_axis",
          (8, 4),
          P("x", None),
          lambda y: y.sum(axis=0),
          "exceeds local shard size",
      ),
  )
  def test_shard_map_epilogue_reduction_over_sharded_axis(
      self, shape, spec, reduce_fn, err_regex
  ):
    def fn(x):
      return reduce_fn(
          jax.shard_map(
              _id_fusible, mesh=self.mesh_1d, in_specs=spec, out_specs=spec
          )(x)
      )

    with jax.set_mesh(self.mesh_1d):
      x = jax.device_put(
          jnp.arange(int(np.prod(shape)), dtype=jnp.float32).reshape(shape),
          spec,
      )
      with self.assertRaisesRegex(NotImplementedError, err_regex):
        jax.jit(fuser.fuse(fn))(x)

  def test_shard_map_top_level_and_inner_fusible_raises(self):
    @fuser.fusible
    def top_fusible(x_fn, out_fn):
      return x_fn() if out_fn is None else out_fn(x_fn())

    @jax.jit
    @fuser.fuse
    def fn(x):
      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return _id_fusible(x_s)

      return top_fusible(x) + shmapped(x)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32), P("x", "y"))
      with self.assertRaisesRegex(
          ValueError,
          "Only a single fusible is allowed in a fused function, but found 2",
      ):
        fn(x)

  def test_shard_map_4way_reshape_and_symbolic_large_abstract_mesh(self):
    cv = jtu.test_device_matches(["tpu"])
    mesh_4 = jtu.create_mesh((4,), ("x",), axis_types=(AxisType.Explicit,))
    fn_replicated_reshape = lambda x: jax.shard_map(
        _id_fusible, mesh=mesh_4, in_specs=P("x"), out_specs=P("x"), check_vma=cv
    )(jnp.reshape(x, (512,), out_sharding=P("x")))
    with jax.set_mesh(mesh_4):
      x_rep = jax.device_put(
          jnp.arange(512, dtype=jnp.float32).reshape(2, 256), P(None, None)
      )
      self.assertAllClose(
          jax.jit(fuser.fuse(fn_replicated_reshape))(x_rep),
          jax.jit(fn_replicated_reshape)(x_rep),
      )

    fn_compound_reshape = lambda x: jax.shard_map(
        _id_fusible,
        mesh=self.mesh,
        in_specs=P(("x", "y")),
        out_specs=P(("x", "y")),
        check_vma=cv,
    )(jnp.reshape(x, (16,), out_sharding=P(("x", "y"))))
    with jax.set_mesh(self.mesh):
      x_3d = jax.device_put(
          jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4), P("x", "y", None)
      )
      with self.assertRaisesRegex(
          NotImplementedError, "lies outside local shard"
      ):
        jax.jit(fuser.fuse(fn_compound_reshape))(x_3d)
      x_2d = jax.device_put(
          jnp.arange(16, dtype=jnp.float32).reshape(4, 4), P("x", "y")
      )
      with self.assertRaisesRegex(
          NotImplementedError, "exceeds local shard size"
      ):
        jax.jit(fuser.fuse(fn_compound_reshape))(x_2d)

      # Concatenate two partially sharded (4, 4) @ P("x", None) arrays along
      # axis 1 into (4, 8) @ P("x", "y"): each device needs its local (2, 4)
      # shard of either operand, exercising core.max_dim/min_dim bounds.
      a = jax.device_put(
          jnp.arange(16, dtype=jnp.float32).reshape(4, 4), P("x", None)
      )
      b = jax.device_put(
          jnp.arange(16, 32, dtype=jnp.float32).reshape(4, 4), P("x", None)
      )
      fn_concat = lambda a, b: jax.shard_map(
          _id_fusible, mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y"), check_vma=cv
      )(
          jnp.reshape(
              jnp.concatenate([a, b], axis=1), (4, 8), out_sharding=P("x", "y")
          )
      )
      self.assertAllClose(
          jax.jit(fuser.fuse(fn_concat))(a, b), jax.jit(fn_concat)(a, b)
      )

    # Verify symbolic bounds analysis runs in O(eqns) on a 1M-device abstract
    # mesh without enumerating mesh coordinates.
    large_mesh = jax.sharding.AbstractMesh(
        (1024, 1024), ("x", "y"), axis_types=(AxisType.Explicit,) * 2
    )
    with jax.sharding.use_abstract_mesh(large_mesh):
      x_sds = jax.ShapeDtypeStruct(
          (2048, 2048),
          jnp.float32,
          sharding=jax.sharding.NamedSharding(large_mesh, P("x", "y")),
      )
      valid_fn = lambda x: jax.shard_map(
          _id_fusible,
          mesh=large_mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
      )(x * 2.0)
      jax.make_jaxpr(fuser.fuse(valid_fn))(x_sds)

      invalid_fn = lambda x: jax.shard_map(
          _id_fusible,
          mesh=large_mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
      )(x[512:1536, :])
      with self.assertRaisesRegex(
          NotImplementedError, "lies outside local shard"
      ):
        jax.make_jaxpr(fuser.fuse(invalid_fn))(x_sds)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
