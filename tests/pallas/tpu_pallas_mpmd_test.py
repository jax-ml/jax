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
"""Tests for Pallas MPMD kernels."""
import dataclasses
import functools
import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import core as jax_core
from jax._src import hijax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

TC = pltpu.CoreType.TC
SCV = pltpu.CoreType.SC_VECTOR_SUBCORE
SCS = pltpu.CoreType.SC_SCALAR_SUBCORE


def from_core_type(core_type):
  match core_type:
    case pltpu.CoreType.TC:
      return pltpu.create_tensorcore_mesh(axis_name="tc_core", num_cores=1)
    case pltpu.CoreType.SC_VECTOR_SUBCORE:
      return plsc.VectorSubcoreMesh(
          core_axis_name="s_core",
          subcore_axis_name="subcore",
          num_cores=1,
          num_subcores=1,
      )
    case pltpu.CoreType.SC_SCALAR_SUBCORE:
      return plsc.ScalarSubcoreMesh(
          axis_name="s_core",
          num_cores=1,
      )
    case _:
      raise ValueError(f"Unsupported core type: {core_type}")


class PallasSCTest(jtu.JaxTestCase):
  USE_TC_TILING = False

  def setUp(self):
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")
    super().setUp()

  @property
  def num_lanes(self) -> int:
    return plsc.get_sparse_core_info().num_lanes

  @property
  def sc_info(self):
    return plsc.get_sparse_core_info()


class MpmdAsyncTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")
    if jtu.is_cloud_tpu():
      self.skipTest("Not yet supported on Cloud TPU.")
    super().setUp()

  @parameterized.parameters([SCS, SCV])
  def test_async_sc_tc_prefetch_vmem(self, sc_core_type):
    mesh = from_core_type(sc_core_type)
    tc_mesh = pltpu.create_tensorcore_mesh(axis_name="tc", num_cores=1)

    def scalar_subcore_fn(x_ref, out_tc_vmem_ref, tc_sem, sem):
      pltpu.async_remote_copy(
          x_ref, out_tc_vmem_ref, sem, tc_sem, device_id={"tc": 0}
      ).wait_send()

    def tc_fn(x_ref, out_tc_vmem_ref, tc_sem):
      pltpu.make_async_copy(x_ref, out_tc_vmem_ref, tc_sem).wait()
      out_tc_vmem_ref[...] += 1

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)
      out, sem = pl.kernel(
          mesh=mesh,
          out_type=(
              pltpu.VMEM(x.shape, x.dtype) @ tc_mesh,
              pltpu.SemaphoreType.DMA(()) @ tc_mesh,
          ),
          scratch_types=[pltpu.SemaphoreType.DMA(())],
          compiler_params=pltpu.CompilerParams(
              use_tc_tiling_on_sc=True,
          ),
          name=f"sc_copy_start_{x.shape[0]}",
      )(scalar_subcore_fn)(x_ref)
      out_ref = jax.new_ref(out)
      sem_ref = jax.new_ref(sem)
      pl.kernel(
          mesh=tc_mesh,
          name=f"tc_copy_end_{x.shape[0]}",
      )(tc_fn)(x_ref, out_ref, sem_ref)
      return jax.freeze(out_ref)

    x = jnp.arange(8 * 128).reshape(8, 128)
    out = f(x)
    np.testing.assert_array_equal(out, x + 1)


class MpmdTest(PallasSCTest):

  @staticmethod
  def from_core_type(core_type):
    match core_type:
      case pltpu.CoreType.TC:
        return pltpu.create_tensorcore_mesh(axis_name="tc_core", num_cores=1)
      case pltpu.CoreType.SC_VECTOR_SUBCORE:
        return plsc.VectorSubcoreMesh(
            core_axis_name="s_core",
            subcore_axis_name="subcore",
            num_cores=1,
            num_subcores=1,
        )
      case pltpu.CoreType.SC_SCALAR_SUBCORE:
        return plsc.ScalarSubcoreMesh(
            axis_name="s_core",
            num_cores=1,
        )
      case _:
        raise ValueError(f"Unsupported core type: {core_type}")

  def test_mismatched_core_axis_name(self):
    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="wrong_core", subcore_axis_name="subcore", num_cores=1
    )
    s_mesh = plsc.ScalarSubcoreMesh(axis_name="s_core", num_cores=1)

    with self.assertRaisesRegex(
        ValueError,
        r".*(Vector|Scalar)SubcoreMesh.*should have the same core axis name .*"
    ):
      pl.kernel(
          body=[lambda *_: None, lambda *_: None],
          mesh=[v_mesh, s_mesh],
          out_type=jax.ShapeDtypeStruct([], jnp.int32),
      )()

  @parameterized.parameters([TC, SCS, SCV])
  def test_mpmd_capture_scalar(self, core_type):
    mesh = from_core_type(core_type)
    axis_name = list(mesh.shape.keys())[0]
    def f(x, i):
      def body(x_ref, out_ref):
        idx = jax.lax.axis_index(axis_name)
        pltpu.sync_copy(x_ref.at[i], out_ref.at[idx])

      return pl.kernel(
          body=body,
          mesh=mesh,
          out_type=jax.ShapeDtypeStruct((1, *x.shape[1:]), jnp.int32),
      )(x)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    for i in range(x.shape[0]):
      out = jax.jit(f)(x, i)
      np.testing.assert_array_equal(out[0], x[i])

  def test_mpmd_capture_multiple_scalars(self):
    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)

    def f(x, i, j):
      def body(x_ref, out_ref):
        idx = jax.lax.axis_index("x")
        pltpu.sync_copy(x_ref.at[i + j], out_ref.at[idx])

      return pl.kernel(
          body=body,
          mesh=mesh,
          out_type=jax.ShapeDtypeStruct((1, *x.shape[1:]), jnp.int32),
      )(x)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    out = jax.jit(f)(x, 1, 2)
    np.testing.assert_array_equal(out[0], x[3])

  def test_mpmd_capture_scalar_indexing(self):
    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)
    def f(x, i):
      def body(x_ref, out_ref):
        idx = jax.lax.axis_index("x")
        pltpu.sync_copy(x_ref.at[i], out_ref.at[idx])

      return pl.kernel(
          body=body,
          mesh=mesh,
          out_type=jax.ShapeDtypeStruct((1, *x.shape[1:]), jnp.int32),
      )(x)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    out = f(x, 1)
    np.testing.assert_array_equal(out[0], x[1])

  def test_mpmd_capture_scalar_and_ref(self):
    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)
    @jax.jit
    def f(x, i):
      y = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)
      y_ref = jax.new_ref(y, memory_space=pl.ANY)
      def body(x_ref, out_ref):
        idx = jax.lax.axis_index("x")
        pltpu.sync_copy(x_ref.at[i], out_ref.at[idx])
        vmem_buf = jax.empty_ref(
            jax.typeof(y_ref).inner_aval, memory_space=pltpu.MemorySpace.VMEM
        )
        pltpu.sync_copy(y_ref, vmem_buf)

      return pl.kernel(
          body=body,
          mesh=mesh,
          out_type=jax.ShapeDtypeStruct((1, *x.shape[1:]), jnp.int32),
      )(x)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    out = f(x, 1)
    np.testing.assert_array_equal(out[0], x[1])

  @parameterized.product(use_tc_tiling=[False, True],
                         scratch_structure=[tuple, dict])
  def test_parallel_subkernels(self, use_tc_tiling, scratch_structure):
    if not jtu.is_cloud_tpu_at_least(2026, 3, 28):
      self.skipTest("Needs a newer libtpu")

    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="s_core",
        subcore_axis_name="subcore",
        num_cores=self.sc_info.num_cores,
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="s_core", num_cores=self.sc_info.num_cores
    )

    x = jnp.arange(128 if use_tc_tiling else self.num_lanes, dtype=jnp.int32)

    def vector_subcore_fn(x_hbm_ref, out_hbm_ref,
                          scratch_vmem_shd_ref, nested_in_dict):
      pltpu.sync_copy(x_hbm_ref, scratch_vmem_shd_ref)
      pltpu.sync_copy(x_hbm_ref, nested_in_dict["vmshd"])
      scratch_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.VMEM)
      pltpu.sync_copy(scratch_vmem_shd_ref, scratch_ref)

      @pl.loop(0, x.size, step=self.num_lanes)
      def _(i):
        s = pl.ds(i, self.num_lanes)
        scratch_ref[s] += 2 * scratch_ref[s]

      pltpu.sync_copy(scratch_ref, out_hbm_ref.at[:x.size])

    def scalar_subcore_fn(x_hbm_ref, out_hbm_ref,
                          scratch_vmem_shd_ref, nested_in_dict):
      del scratch_vmem_shd_ref, nested_in_dict
      scratch_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.SMEM)
      pltpu.sync_copy(x_hbm_ref, scratch_ref)

      @pl.loop(0, x.size)
      def _(i):
        scratch_ref[i] += 3 * scratch_ref[i]

      pltpu.sync_copy(scratch_ref, out_hbm_ref.at[x.size:])

    if scratch_structure is dict:
      scratch_shapes = dict(
          scratch_vmem_shd_ref=pltpu.VMEM_SHARED(x.shape, x.dtype),
          nested_in_dict=dict(vmshd=pltpu.VMEM_SHARED(x.shape, x.dtype)))
    else:
      scratch_shapes = (pltpu.VMEM_SHARED(x.shape, x.dtype),
                        dict(vmshd=pltpu.VMEM_SHARED(x.shape, x.dtype)))
    out = pl.kernel(
        body=[vector_subcore_fn, scalar_subcore_fn],
        mesh=[v_mesh, s_mesh],
        out_type=jax.ShapeDtypeStruct([x.size * 2], x.dtype),
        scratch_types=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=use_tc_tiling,
        ),
    )(x)
    np.testing.assert_array_equal(out[:x.size], x + 2 * x)
    np.testing.assert_array_equal(out[x.size:], x + 3 * x)

  @parameterized.product(use_tc_tiling=[False, True])
  def test_parallel_subkernels_with_kernel(self, use_tc_tiling):
    if not jtu.is_cloud_tpu_at_least(2026, 3, 28):
      self.skipTest("Needs a newer libtpu")

    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="s_core",
        subcore_axis_name="subcore",
        num_cores=self.sc_info.num_cores,
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="s_core", num_cores=self.sc_info.num_cores
    )

    x = jnp.arange(128 if use_tc_tiling else self.num_lanes, dtype=jnp.int32)

    def vector_subcore_fn(x_hbm_ref, out_hbm_ref, scratch_vmem_shd_ref):
      pltpu.sync_copy(x_hbm_ref, scratch_vmem_shd_ref)
      scratch_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.VMEM)
      pltpu.sync_copy(scratch_vmem_shd_ref, scratch_ref)

      @pl.loop(0, x.size, step=self.num_lanes)
      def _(i):
        s = pl.ds(i, self.num_lanes)
        scratch_ref[s] += 2 * scratch_ref[s]

      pltpu.sync_copy(scratch_ref, out_hbm_ref.at[:x.size])

    def scalar_subcore_fn(x_hbm_ref, out_hbm_ref, scratch_vmem_shd_ref):
      del scratch_vmem_shd_ref
      scratch_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.SMEM)
      pltpu.sync_copy(x_hbm_ref, scratch_ref)

      @pl.loop(0, x.size)
      def _(i):
        scratch_ref[i] += 3 * scratch_ref[i]

      pltpu.sync_copy(scratch_ref, out_hbm_ref.at[x.size:])

    scratch_shapes = (pltpu.VMEM_SHARED(x.shape, x.dtype),)

    out = pl.kernel(
        body=[vector_subcore_fn, scalar_subcore_fn],
        mesh=[v_mesh, s_mesh],
        out_type=jax.ShapeDtypeStruct([x.size * 2], x.dtype),
        scratch_types=scratch_shapes,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=use_tc_tiling,
        ),
    )(x)
    np.testing.assert_array_equal(out[:x.size], x + 2 * x)
    np.testing.assert_array_equal(out[x.size:], x + 3 * x)

  @parameterized.parameters([TC, SCS, SCV])
  def test_passing_in_refs(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)
      o_ref = jax.empty_ref(jax.typeof(x))
      pl.kernel(body=pltpu.sync_copy, mesh=mesh)(x_ref, o_ref)
      return jax.freeze(o_ref)

    np.testing.assert_array_equal(x, f(x))

  @parameterized.parameters([TC, SCS])
  def test_passing_in_multiple_refs(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)
    y = jnp.zeros_like(x)

    @jax.jit
    def f(x, y):
      x_ref = jax.new_ref(x)
      y_ref = jax.new_ref(y)

      def fn(x_ref, y_ref, scratch_x, scratch_y):
        pltpu.sync_copy(x_ref, scratch_x)
        pltpu.sync_copy(y_ref, scratch_y)
        pltpu.sync_copy(scratch_x, y_ref)
        pltpu.sync_copy(scratch_y, x_ref)

      pl.kernel(
          body=fn,
          mesh=mesh,
          scratch_types=(
              pltpu.SMEM(x.shape, x.dtype),
              pltpu.SMEM(y.shape, y.dtype),
          ),
      )(x_ref, y_ref)
      return jax.freeze(x_ref), jax.freeze(y_ref)

    out_x, out_y = f(x, y)
    np.testing.assert_array_equal(out_x, y)
    np.testing.assert_array_equal(out_y, x)

  @parameterized.parameters([TC, SCV])
  def test_mixed_outputs_and_refs(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)
    num_lanes = pltpu.get_tpu_info().sparse_core.num_lanes

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)

      def fn(x_ref, out_ref, scratch_ref):
        pltpu.sync_copy(x_ref, out_ref)
        pltpu.sync_copy(x_ref, scratch_ref)
        scratch_ref[0, :num_lanes] += jnp.ones(num_lanes, dtype=jnp.int32)
        pltpu.sync_copy(scratch_ref, x_ref)

      out = pl.kernel(
          body=fn,
          mesh=mesh,
          out_type=jax.typeof(x),
          scratch_types=(pltpu.VMEM(x.shape, x.dtype),),
      )(x_ref)
      return out, jax.freeze(x_ref)

    out, mutated_x = f(x)
    np.testing.assert_array_equal(out, x)
    np.testing.assert_array_equal(mutated_x, x.at[0, :num_lanes].add(1))

  @parameterized.parameters([TC, SCS, SCV])
  def test_passing_in_refs_read_only(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)

      def fn(_):
        pass

      pl.kernel(body=fn, mesh=mesh)(x_ref)
      return jax.freeze(x_ref)

    np.testing.assert_array_equal(x, f(x))

  @parameterized.parameters([TC, SCS])
  def test_passing_in_refs_with_scratch(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)
      o_ref = jax.empty_ref(jax.typeof(x))

      def fn(x_hbm_ref, out_hbm_ref, scratch_ref):
        pltpu.sync_copy(x_hbm_ref, scratch_ref)
        pltpu.sync_copy(scratch_ref, out_hbm_ref)

      pl.kernel(
          body=fn,
          mesh=mesh,
          scratch_types=(pltpu.SMEM(x.shape, x.dtype),),
      )(x_ref, o_ref)
      return jax.freeze(o_ref)

    np.testing.assert_array_equal(x, f(x))

  @parameterized.parameters([TC, SCS, SCV])
  def test_passing_in_duplicate_refs_errors(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)
      pl.kernel(body=lambda *_: None, mesh=mesh)(x_ref, x_ref)

    with self.assertRaisesRegex(
        NotImplementedError,
        "Cannot pass the same ref into a mpmd map multiple times",
    ):
      f(x)

  @parameterized.parameters([TC, SCS, SCV])
  def test_closed_over_refs(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)
      o_ref = jax.empty_ref(jax.typeof(x))

      @pl.kernel(mesh=mesh)
      def fn():
        pltpu.sync_copy(x_ref, o_ref)
      fn()

      return jax.freeze(o_ref)

    np.testing.assert_array_equal(x, f(x))

  @parameterized.parameters([TC, SCS, SCV])
  def test_closed_over_refs_with_scratch(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)

    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)
      o_ref = jax.empty_ref(jax.typeof(x))

      mem_type = pltpu.SMEM if core_type == SCS else pltpu.VMEM
      @pl.kernel(
          mesh=mesh,
          scratch_types=(mem_type(x.shape, x.dtype),),
      )
      def fn(scratch_ref):
        pltpu.sync_copy(x_ref, scratch_ref)
        pltpu.sync_copy(scratch_ref, o_ref)
      fn()

      return jax.freeze(o_ref)

    np.testing.assert_array_equal(x, f(x))

  @parameterized.product(
      use_tc_tiling=(False, True), full_core_spec=(True, False),
      signalling_direction=("scs_to_tec", "tec_to_scs", "both"),
      subcores=(2, 16))
  def test_parallel_subkernels_semaphores(
      self, use_tc_tiling, full_core_spec, signalling_direction, subcores
  ):
    if not jtu.is_cloud_tpu_at_least(2026, 5, 15):
      self.skipTest("Needs a newer libtpu")

    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="s_core",
        subcore_axis_name="subcore",
        num_cores=self.sc_info.num_cores,
        num_subcores=min(self.sc_info.num_subcores, subcores),
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="s_core", num_cores=self.sc_info.num_cores
    )

    x = jnp.arange(128 if use_tc_tiling else self.num_lanes, dtype=jnp.int32)

    def vector_subcore_fn(_, tec_sem, scs_sem):
      device_id = ({"s_core": jax.lax.axis_index("s_core")} if full_core_spec
                   else None)
      if signalling_direction in ("tec_to_scs", "both"):
        pl.semaphore_signal(scs_sem, 1, device_id=device_id)
      if signalling_direction in ("scs_to_tec", "both"):
        pl.semaphore_wait(tec_sem, 1)

    def scalar_subcore_fn(_, tec_sem, scs_sem):
      if signalling_direction in ("scs_to_tec", "both"):
        for i in range(jax.lax.axis_size("subcore")):
          device_id = {"subcore": i}
          if full_core_spec:
            device_id |= {"s_core": jax.lax.axis_index("s_core")}
          pl.semaphore_signal(tec_sem, device_id=device_id)
      if signalling_direction in ("tec_to_scs", "both"):
        pl.semaphore_wait(scs_sem, jax.lax.axis_size("subcore"))

    device_mesh = jax.make_mesh((jax.device_count(),), axis_names=("x",))

    @functools.partial(jax.shard_map, out_specs=None, check_vma=False)
    def test_mpmd_map():
      pl.kernel(
          body=[vector_subcore_fn, scalar_subcore_fn],
          mesh=[v_mesh, s_mesh],
          out_type=jax.ShapeDtypeStruct([x.size * 2], x.dtype),
          compiler_params=pltpu.CompilerParams(
              use_tc_tiling_on_sc=use_tc_tiling,
          ),
          scratch_types=[
              # SCS -> TEC
              pltpu.SemaphoreType.REGULAR(()) @ v_mesh,
              # TEC -> SCS
              pltpu.SemaphoreType.REGULAR(()) @ s_mesh,
          ],
      )()

    with jax.sharding.set_mesh(device_mesh):
      test_mpmd_map()

  def test_parallel_subkernels_semaphores_missing_subcore_axis(self):
    if not jtu.is_cloud_tpu_at_least(2026, 3, 1):
      self.skipTest("Need a newer libtpu")

    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="s_core",
        subcore_axis_name="subcore",
        num_cores=self.sc_info.num_cores,
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="s_core", num_cores=self.sc_info.num_cores
    )

    def vector_subcore_fn(_, tec_sem):
      pl.semaphore_wait(tec_sem, 1)

    def scalar_subcore_fn(_, tec_sem):
      pl.semaphore_signal(
          tec_sem, device_id={"s_core": jax.lax.axis_index("s_core")})

    device_mesh = jax.make_mesh((jax.device_count(),), axis_names=("x",))

    @functools.partial(jax.shard_map, out_specs=None, check_vma=False)
    def test_mpmd_map():
      pl.kernel(
          body=[vector_subcore_fn, scalar_subcore_fn],
          mesh=[v_mesh, s_mesh],
          out_type=jax.ShapeDtypeStruct([8], jnp.int32),
          scratch_types=[pltpu.SemaphoreType.REGULAR(()) @ v_mesh],
      )()
    with self.assertRaisesRegex(
        ValueError,
        re.compile(
            r"When addressing SC_VECTOR_SUBCORE from SC_SCALAR_SUBCORE and"
            r" specifying .* the following axes are missing from the mesh:"
            r" \{'subcore'\}",
            re.IGNORECASE,
        ),
    ):
      with jax.sharding.set_mesh(device_mesh):
        test_mpmd_map()


@dataclasses.dataclass(frozen=True)
class WeirdTuple:
  x0: jax.Array
  x1: jax.Array


@dataclasses.dataclass(frozen=True)
class WeirdTupleTy(hijax.HiType):
  x0_aval: jax_core.ShapedArray
  x1_aval: jax_core.ShapedArray

  @property
  def shape(self) -> tuple[int, ...]:
    return self.x0_aval.shape

  @property
  def dtype(self) -> jnp.dtype:
    return self.x0_aval.dtype

  def lo_ty(self) -> list[jax_core.ShapedArray]:
    return [self.x0_aval, self.x1_aval]

  def lower_val(self, hi_val: WeirdTuple) -> list[jax.Array]:
    assert isinstance(hi_val, WeirdTuple), f"Expected WeirdTuple, got {type(hi_val)}"
    return [hi_val.x0, hi_val.x1]

  def raise_val(self, x0, x1) -> WeirdTuple:
    return WeirdTuple(x0, x1)

  def get_ref_aval(self):
    from jax._src import state

    return state.AbstractRef(self, memory_space=self.memory_space)

  def dma_start(
      self,
      src_ref,
      dst_ref,
      src_sem,
      dst_sem,
      device_id,
      device_id_type,
      priority,
      add,
  ) -> None:
    assert device_id is None
    assert src_sem is None
    src_x0_ref = src_ref._refs.x0
    src_x1_ref = src_ref._refs.x1
    dst_x0_ref = dst_ref._refs.x0
    dst_x1_ref = dst_ref._refs.x1

    desc_x0 = pltpu.make_async_copy(src_x0_ref, dst_x0_ref, dst_sem)
    desc_x0.start(priority=priority, add=add)

    desc_x1 = pltpu.make_async_copy(src_x1_ref, dst_x1_ref, dst_sem)
    desc_x1.start(priority=priority, add=add)

  def dma_wait(
      self, src_ref, dst_ref, src_sem, dst_sem, device_id, device_id_type
  ):
    assert device_id is None
    assert src_sem is None

    src_x0_ref = src_ref._refs.x0
    src_x1_ref = src_ref._refs.x1
    dst_x0_ref = dst_ref._refs.x0
    dst_x1_ref = dst_ref._refs.x1

    desc_x0 = pltpu.make_async_copy(src_x0_ref, dst_x0_ref, dst_sem)
    desc_x0.wait()

    desc_x1 = pltpu.make_async_copy(src_x1_ref, dst_x1_ref, dst_sem)
    desc_x1.wait()

hijax.register_hitype(
    WeirdTuple, lambda t: WeirdTupleTy(jax.typeof(t.x0), jax.typeof(t.x1))
)

unpack_p = hijax.HiPrimitive("unpack")
unpack = unpack_p.bind
unpack_p.multiple_results = True
unpack_p.is_high = lambda *_: True
unpack_p.def_abstract_eval(lambda x: [x.x0_aval, x.x1_aval])
unpack_p.to_lojax = lambda x: [x.x0, x.x1]

pack_p = hijax.HiPrimitive("pack")
pack = pack_p.bind
pack_p.is_high = lambda *_: True
pack_p.def_abstract_eval(lambda x0, x1: WeirdTupleTy(x0, x1))
pack_p.to_lojax = lambda x0, x1: WeirdTuple(x0, x1)


class MpmdHijaxTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu():
      self.skipTest("Only works on TPU.")
    super().setUp()

  def test_pass_weird_tuple_into_mpmd_map(self):
    xt = WeirdTuple(
        x0=jnp.ones((8, 8), dtype=jnp.int32),
        x1=jnp.zeros((8,), dtype=jnp.int32),
    )

    def kernel(xt_ref, ot_ref, xt_vmem_ref, ot_vmem_ref):
      pltpu.sync_copy(xt_ref, xt_vmem_ref)
      ot_vmem_ref[...] = xt_vmem_ref[...]
      pltpu.sync_copy(ot_vmem_ref, ot_ref)

    mesh = pltpu.create_tensorcore_mesh("tc_core", num_cores=1)

    ot = pl.kernel(
        body=kernel,
        mesh=mesh,
        out_type=jax.typeof(xt),
        scratch_types=(
            pltpu.VMEM.like(WeirdTupleTy(jax.typeof(xt.x0), jax.typeof(xt.x1))),
            pltpu.VMEM.like(WeirdTupleTy(jax.typeof(xt.x0), jax.typeof(xt.x1))),
        ),
    )(xt)

    self.assertArraysEqual(ot.x0, xt.x0)
    self.assertArraysEqual(ot.x1, xt.x1)

  def test_mpmd_map_hijax_input_output_aliasing(self):
    xt = WeirdTuple(
        x0=jnp.ones((8, 8), dtype=jnp.int32),
        x1=jnp.zeros((8,), dtype=jnp.int32),
    )
    mesh = pltpu.create_tensorcore_mesh("tc_core", num_cores=1)

    def kernel(xt_ref_inner, scratch_vmem_ref):
      pltpu.sync_copy(xt_ref_inner, scratch_vmem_ref)
      x0, x1 = unpack(scratch_vmem_ref[...])
      scratch_vmem_ref[...] = pack(x0 + 1, x1)
      pltpu.sync_copy(scratch_vmem_ref, xt_ref_inner)

    @jax.jit
    def f(xt):
      xt_ref = jax.new_ref(xt)
      pl.kernel(
          body=kernel,
          mesh=mesh,
          scratch_types=(pltpu.VMEM.like(jax.typeof(xt)),),
      )(xt_ref)
      return jax.freeze(xt_ref)

    x1 = f(xt)
    self.assertArraysEqual(x1.x0, xt.x0 + 1)
    self.assertArraysEqual(x1.x1, xt.x1)

  def test_parallel_subkernels_hijax(self):
    if not jtu.is_cloud_tpu_at_least(2026, 3, 28):
      self.skipTest("Needs a newer libtpu")
    xt = WeirdTuple(
        x0=jnp.ones((8, 128), dtype=jnp.int32),
        x1=jnp.zeros((8,), dtype=jnp.int32),
    )
    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="s_core",
        subcore_axis_name="subcore",
        num_cores=1,
        num_subcores=1,
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="s_core",
        num_cores=1,
    )

    def scalar_subcore_fn(x_ref, os_ref, ov_ref):
      del ov_ref
      pltpu.sync_copy(x_ref, os_ref)

    def vector_subcore_fn(x_ref, os_ref, ov_ref):
      del os_ref
      pltpu.sync_copy(x_ref, ov_ref)

    ot_s, ot_v = pl.kernel(
        body=[vector_subcore_fn, scalar_subcore_fn],
        mesh=[v_mesh, s_mesh],
        out_type=[jax.typeof(xt), jax.typeof(xt)],
    )(xt)
    self.assertArraysEqual(ot_s.x0, xt.x0)
    self.assertArraysEqual(ot_s.x1, xt.x1)
    self.assertArraysEqual(ot_v.x0, xt.x0)
    self.assertArraysEqual(ot_v.x1, xt.x1)

  def test_closed_over_hijax_refs(self):
    xt = WeirdTuple(
        x0=jnp.ones((8, 8), dtype=jnp.int32),
        x1=jnp.zeros((8,), dtype=jnp.int32),
    )
    mesh = pltpu.create_tensorcore_mesh("tc_core", num_cores=1)

    @jax.jit
    def f(xt_in):
      xt_ref = jax.new_ref(xt_in)
      ot_ref = jax.empty_ref(jax.typeof(xt_in))

      def kernel(xt_vmem_ref, ot_vmem_ref):
        pltpu.sync_copy(xt_ref, xt_vmem_ref)
        ot_vmem_ref[...] = xt_vmem_ref[...]
        pltpu.sync_copy(ot_vmem_ref, ot_ref)

      pl.kernel(
          body=kernel,
          mesh=mesh,
          scratch_types=(
              pltpu.VMEM.like(
                  WeirdTupleTy(jax.typeof(xt.x0), jax.typeof(xt.x1))
              ),
              pltpu.VMEM.like(
                  WeirdTupleTy(jax.typeof(xt.x0), jax.typeof(xt.x1))
              ),
          ),
      )()
      return jax.freeze(ot_ref)

    ot = f(xt)
    self.assertArraysEqual(ot.x0, xt.x0)
    self.assertArraysEqual(ot.x1, xt.x1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
