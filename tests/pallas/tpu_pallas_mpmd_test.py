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
from jax._src.pallas.fuser import fusible_dtype
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()

TC = pltpu.CoreType.TC
SCV = pltpu.CoreType.SC_VECTOR_SUBCORE
SCS = pltpu.CoreType.SC_SCALAR_SUBCORE


def from_core_type(core_type):
  match core_type:
    case pltpu.CoreType.TC:
      return pltpu.TensorCoreMesh(axis_name="tc_core", num_cores=1)
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


# TODO(rdyro): A temporary workaround to avoid flakiness.
@jtu.thread_unsafe_test_class()
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


# TODO(rdyro): A temporary workaround to avoid flakiness.
@jtu.thread_unsafe_test_class()
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
    tc_mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)

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


# TODO(rdyro): A temporary workaround to avoid flakiness.
@jtu.thread_unsafe_test_class()
class MpmdTest(PallasSCTest):

  @staticmethod
  def from_core_type(core_type):
    match core_type:
      case pltpu.CoreType.TC:
        return pltpu.TensorCoreMesh(axis_name="tc_core", num_cores=1)
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

  def test_mpmd_with_name_arg(self):
    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x, memory_space=pltpu.HBM)
      out_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.HBM)
      def f_scs(scratch):
        del scratch
        pltpu.sync_copy(x_ref, out_ref)
      def f_scv(scratch):
        scratch[pl.ds(pltpu.get_tpu_info().sparse_core.num_lanes)] = jnp.zeros(
            pltpu.get_tpu_info().sparse_core.num_lanes, dtype=jnp.int32
        )

      scv_mesh = from_core_type(SCV)
      pl.kernel(
          body=(f_scs, f_scv),
          mesh=(from_core_type(SCS), scv_mesh),
          scratch_types=[pltpu.VMEM([128], jnp.int32) @ scv_mesh],
          name="test_mpmd_with_name_arg",
      )()
      return jax.freeze(out_ref)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    np.testing.assert_array_equal(f(x), x)

  def test_mpmd_with_dup_fn_names(self):
    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x, memory_space=pltpu.HBM)
      out_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.HBM)
      def f_scs(scratch):
        del scratch
        pltpu.sync_copy(x_ref, out_ref)
      def f_scv(scratch):
        scratch[pl.ds(pltpu.get_tpu_info().sparse_core.num_lanes)] = jnp.zeros(
            pltpu.get_tpu_info().sparse_core.num_lanes, dtype=jnp.int32
        )
      def wrapper_fn(fn, scratch):
        return fn(scratch)

      scv_mesh = from_core_type(SCV)
      pl.kernel(
          body=(functools.partial(wrapper_fn, f_scs),
                functools.partial(wrapper_fn, f_scv)),
          mesh=(from_core_type(SCS), scv_mesh),
          scratch_types=[pltpu.VMEM([128], jnp.int32) @ scv_mesh],
      )()
      return jax.freeze(out_ref)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    np.testing.assert_array_equal(f(x), x)

  def test_mpmd_capture_multiple_scalars(self):
    mesh = pltpu.TensorCoreMesh(axis_name="x", num_cores=1)

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
    mesh = pltpu.TensorCoreMesh(axis_name="x", num_cores=1)
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
    mesh = pltpu.TensorCoreMesh(axis_name="x", num_cores=1)
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

  @parameterized.parameters([TC, SCS, SCV])
  def test_all_semaphores_support(self, core_type):
    mesh = from_core_type(core_type)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)

    @jax.jit
    def f(x):
      @pl.kernel(
          mesh=mesh,
          compiler_params=pltpu.CompilerParams(has_side_effects=True),
          out_type=[
              pltpu.HBM((8, 128), jnp.int32),
              pltpu.SemaphoreType.DMA(()) @ mesh,
              pltpu.SemaphoreType.REGULAR(()) @ mesh,
          ],
      )
      def body(x_ref, out_ref, dma_sem, regular_sem):
        pltpu.async_copy(x_ref, out_ref, dma_sem).wait()
        pl.semaphore_signal(regular_sem, 1)
        pl.semaphore_wait(regular_sem, 1)

      out, _, _ = body(x)
      return out

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

  def test_vmap(self):

    @pl.kernel(mesh=from_core_type(SCV))
    def kernel(x_hbm_ref):
      pltpu.touch(x_hbm_ref)

    x = jnp.arange(self.sc_info.num_lanes, dtype=jnp.int32)
    _ = jax.vmap(kernel)(x[jnp.newaxis])

  def test_vmap_captured_scalar_error(self):
    def run_kernel(zero):

      @pl.kernel(mesh=from_core_type(SCV))
      def kernel(x_hbm_ref):
        x_hbm_ref[...] = jnp.broadcast_to(zero, x_hbm_ref.shape)

      x = jnp.arange(self.sc_info.num_lanes, dtype=jnp.int32)
      kernel(x)

    with self.assertRaisesRegex(
        ValueError,
        "Closed-over scalar constants cannot be batched"
    ):
      jax.vmap(run_kernel)(jnp.int32(0)[jnp.newaxis])

  def test_vmap_with_refs(self):
    def run_kernel(x):
      x_hbm_ref = jax.new_ref(x)

      @pl.kernel(
          mesh=from_core_type(SCV),
          scratch_types=(pltpu.VMEM.like(x),),
      )
      def kernel(scratch_ref):
        pltpu.sync_copy(x_hbm_ref, scratch_ref)
        scratch_ref[...] += 1
        pltpu.sync_copy(scratch_ref, x_hbm_ref)

      _ = kernel()
      return jax.freeze(x_hbm_ref)

    x = jnp.arange(self.sc_info.num_lanes, dtype=jnp.int32)
    out = jax.vmap(run_kernel)(x[jnp.newaxis])
    np.testing.assert_array_equal(out, x[jnp.newaxis] + 1)

  def test_remat_with_checkpoint(self):
    mesh = pltpu.TensorCoreMesh(axis_name="tc", num_cores=1)

    kernel_impl = pl.kernel(
        pltpu.sync_copy,
        mesh=mesh,
        out_type=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )

    # NOTE: ``mpmd_map`` does not yet have a general purpose jvp rule, so we
    # need to use ``custom_vjp`` to define one.
    @jax.custom_vjp
    def kernel(x):
      return kernel_impl(x)

    kernel.defvjp(lambda x: (kernel_impl(x), ()), lambda res, g: (g,))

    def f(x):
      def block(x):
        y = jax.ad_checkpoint.checkpoint_name(x * 2.0, name="y")
        return kernel(y)
      policy = jax.checkpoint_policies.save_only_these_names("y")
      return jax.remat(block, policy=policy)(x).sum()

    x = jnp.ones((8, 128), dtype=jnp.float32)
    np.testing.assert_array_equal(jax.grad(f)(x), jnp.full_like(x, 2.0))

  @parameterized.product(
      use_tc_tiling=(False, True), full_core_spec=(True, False),
      signalling_direction=("scs_to_tec", "tec_to_scs", "both"),
      subcores=(2, 16))
  def test_parallel_subkernels_semaphores(
      self, use_tc_tiling, full_core_spec, signalling_direction, subcores
  ):
    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="s_core",
        subcore_axis_name="subcore",
        num_cores=self.sc_info.num_cores,
        num_subcores=min(self.sc_info.num_subcores, subcores),
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="s_core", num_cores=self.sc_info.num_cores
    )

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

    def test_mpmd_map():
      jax.jit(
          pl.kernel(
              body=[vector_subcore_fn, scalar_subcore_fn],
              mesh=[v_mesh, s_mesh],
              out_type=jax.ShapeDtypeStruct([256], jnp.int32),
              compiler_params=pltpu.CompilerParams(
                  use_tc_tiling_on_sc=use_tc_tiling,
              ),
              scratch_types=[
                  # SCS -> TEC
                  pltpu.SemaphoreType.REGULAR(()) @ v_mesh,
                  # TEC -> SCS
                  pltpu.SemaphoreType.REGULAR(()) @ s_mesh,
              ],
          )
      )()

    jax.block_until_ready(test_mpmd_map())

  def test_copy_with_cross_core_signaling(self):
    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore",
        num_cores=self.sc_info.num_cores,
        num_subcores=self.sc_info.num_subcores,
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="core", num_cores=self.sc_info.num_cores
    )
    num_cores = v_mesh.num_cores
    num_subcores = v_mesh.num_subcores
    x1 = jnp.arange(num_cores * 8 * 128, dtype=jnp.int32).reshape(
        num_cores, 8, 128
    )
    x2 = jnp.arange(
        num_cores * num_subcores * 8 * 128, dtype=jnp.int32
    ).reshape(num_cores, num_subcores, 8, 128)

    def _barrier(my_sem, scs_sem, tec_sem):
      num_cores = jax.lax.axis_size("core")
      num_subcores = jax.lax.axis_size("subcore")
      for i in range(num_cores):
        pl.semaphore_signal(scs_sem, device_id={"core": i})
        for j in range(num_subcores):
          pl.semaphore_signal(tec_sem, device_id={"core": i, "subcore": j})
      pl.semaphore_wait(my_sem, num_cores + num_cores * num_subcores)

    def vector_subcore_fn(x1_ref, x2_ref, o1_ref, o2_ref, tec_sem, scs_sem):
      del x2_ref, o2_ref
      num_cores = jax.lax.axis_size("core")
      _barrier(tec_sem, scs_sem, tec_sem)

      # Wait for scalar cores to tell us that we can start.
      pl.semaphore_wait(tec_sem, num_cores)

      # Copy from x1 to o1.
      i, j = jax.lax.axis_index("core"), jax.lax.axis_index("subcore")
      pltpu.sync_copy(x1_ref.at[i, j], o1_ref.at[i, j])

      # Tell scalar cores they can start
      for i in range(jax.lax.axis_size("core")):
        pl.semaphore_signal(scs_sem, device_id={"core": i})

      # Wait for scalar cores to tell us they are done.
      pl.semaphore_wait(tec_sem, num_cores)

    def scalar_subcore_fn(x1_ref, x2_ref, o1_ref, o2_ref, tec_sem, scs_sem):
      del x1_ref, o1_ref
      num_cores = jax.lax.axis_size("core")
      num_subcores = jax.lax.axis_size("subcore")

      _barrier(scs_sem, scs_sem, tec_sem)

      # Tell vector subcores to start.
      for i in range(num_cores):
        for j in range(num_subcores):
          pl.semaphore_signal(tec_sem, device_id={"core": i, "subcore": j})

      # Wait for vector subcores to tell us they are done.
      pl.semaphore_wait(scs_sem, num_cores * num_subcores)

      # Copy from x2 to o2.
      i = jax.lax.axis_index("core")
      pltpu.sync_copy(x2_ref.at[i], o2_ref.at[i])

      # Tell vector cores we are done.
      for i in range(num_cores):
        for j in range(num_subcores):
          pl.semaphore_signal(tec_sem, device_id={"core": i, "subcore": j})

    @jax.jit
    def f(x1, x2):
      return pl.kernel(
          body=[vector_subcore_fn, scalar_subcore_fn],
          mesh=[v_mesh, s_mesh],
          out_type=(jax.typeof(x1), jax.typeof(x2)),
          scratch_types=[
              # SCS -> TEC
              pltpu.SemaphoreType.REGULAR(()) @ v_mesh,
              # TEC -> SCS
              pltpu.SemaphoreType.REGULAR(()) @ s_mesh,
          ],
      )(x1, x2)

    o1, o2 = f(x1, x2)
    np.testing.assert_array_equal(o1, x1)
    np.testing.assert_array_equal(o2, x2)

  def test_parallel_subkernels_semaphores_missing_subcore_axis(self):
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

  def test_mpmd_map_semaphore_mesh_enforcement(self):
    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="s_core", subcore_axis_name="subcore", num_cores=1,
        num_subcores=1)
    s_mesh = plsc.ScalarSubcoreMesh(axis_name="s_core", num_cores=1)

    dummy_fn = lambda *_: None

    # Case 1: MPMD with 1 mesh and semaphore WITHOUT mesh should PASS
    pl.kernel(
        body=[dummy_fn],
        mesh=[v_mesh],
        out_type=jax.ShapeDtypeStruct([self.num_lanes], jnp.int32),
        scratch_types=[pltpu.SemaphoreType.REGULAR(())],
    )()

    # Case 2: MPMD with 2 meshes and semaphore WITHOUT mesh should FAIL
    with self.assertRaisesRegex(
        NotImplementedError,
        r"MPMD map with more than one mesh requires scratch_type to have",
    ):
      pl.kernel(
          body=[dummy_fn, dummy_fn],
          mesh=[v_mesh, s_mesh],
          out_type=jax.ShapeDtypeStruct([self.num_lanes], jnp.int32),
          scratch_types=[pltpu.SemaphoreType.REGULAR(())],
      )()

  @parameterized.parameters([False, True])
  def test_smem_vmem_shared_signaling(self, reverse):
    if not jtu.is_libtpu_at_least("0.0.43"):
      self.skipTest("Requires libtpu 0.0.43 or newer")
    num_cores = self.sc_info.num_cores
    num_subcores = self.sc_info.num_subcores

    v_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=num_cores,
        num_subcores=num_subcores,
    )
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="core", num_cores=num_cores
    )

    slice_size = 128
    x = jnp.arange(
        num_cores * num_subcores * slice_size, dtype=jnp.int32
    ).reshape(num_cores, num_subcores, slice_size)

    def vector_subcore_fn(x_ref, out_ref, smem_ref, vmem_shared_ref, scs_sem, tec_sem):
      del smem_ref
      core_id = jax.lax.axis_index("core")
      subcore_id = jax.lax.axis_index("subcore")

      # TEC waits on regular semaphore tec_sem
      if not reverse:
        pl.semaphore_wait(tec_sem, 1)

      if reverse:
        pltpu.sync_copy(
            x_ref.at[core_id, subcore_id], vmem_shared_ref.at[subcore_id]
        )
        pl.semaphore_signal(scs_sem, device_id={"core": core_id})
        del out_ref, tec_sem
      else:
        del x_ref, scs_sem
        pltpu.sync_copy(
            vmem_shared_ref.at[subcore_id], out_ref.at[core_id, subcore_id]
        )

    def scalar_subcore_fn(x_ref, out_ref, smem_ref, vmem_shared_ref, scs_sem, tec_sem):
      core_id = jax.lax.axis_index("core")

      if reverse:
        del x_ref, tec_sem
        pl.semaphore_wait(scs_sem, num_subcores)
        pltpu.sync_copy(vmem_shared_ref, smem_ref)
        pltpu.sync_copy(smem_ref, out_ref.at[core_id])
      else:
        del out_ref, scs_sem
        pltpu.sync_copy(x_ref.at[core_id], smem_ref)
        pltpu.sync_copy(smem_ref, vmem_shared_ref)

        # Signal TECs
        for j in range(num_subcores):
          device_id = {"core": core_id, "subcore": j}
          pl.semaphore_signal(tec_sem, device_id=device_id)

    scratch_types = [
        pltpu.SMEM((num_subcores, slice_size), jnp.int32) @ s_mesh,
        pltpu.VMEM_SHARED((num_subcores, slice_size), jnp.int32),
        pltpu.SemaphoreType.REGULAR(()) @ s_mesh,
        pltpu.SemaphoreType.REGULAR(()) @ v_mesh,
    ]

    out = pl.kernel(
        body=[vector_subcore_fn, scalar_subcore_fn],
        mesh=[v_mesh, s_mesh],
        out_type=jax.ShapeDtypeStruct(
            (num_cores, num_subcores, slice_size), jnp.int32
        ),
        scratch_types=scratch_types,
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True),
    )(x)
    expected = x
    np.testing.assert_array_equal(out, expected)


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


# TODO(rdyro): A temporary workaround to avoid flakiness.
@jtu.thread_unsafe_test_class()
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

    mesh = pltpu.TensorCoreMesh(axis_name="tc_core", num_cores=1)

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
    mesh = pltpu.TensorCoreMesh(axis_name="tc_core", num_cores=1)

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
    if pltpu.get_tpu_info().sparse_core is None:
      self.skipTest("Test needs a TPU with a sparse core")
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
    mesh = pltpu.TensorCoreMesh(axis_name="tc_core", num_cores=1)

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


# TODO(rdyro): A temporary workaround to avoid flakiness.
@jtu.thread_unsafe_test_class()
class MpmdPhysicalizeTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu():
      self.skipTest("Only works on TPU.")
    super().setUp()

  def test_mpmd_map_physicalize(self):

    @dataclasses.dataclass(frozen=True)
    class SimpleFusionDType(fusible_dtype.FusionDType):

      def __str__(self):
        return "simple_fusion_dtype"

      def abstract_unpack(self, x):
        if isinstance(x, jax_core.ShapedArray):
          return (x.update(dtype=jnp.float32), x.update(dtype=jnp.float32))
        raise NotImplementedError(type(x))

      def abstract_pack(self, x, y):
        if isinstance(x, jax_core.ShapedArray):
          return x.update(dtype=self)
        raise NotImplementedError(type(x))

      def pull_block_spec_one_step(self, aval_out, block_spec):
        return block_spec, block_spec

      def unpack_push_block_spec(self, aval_in, block_spec):
        return block_spec, block_spec

      def unpack_pull_block_spec(self, aval_in, block_spec1, block_spec2):
        return (block_spec1,)

      def pack_eval_rule(self, eval_ctx, x, y):
        return fusible_dtype.pack_dtype_p.bind(x, y, dtype=self)

      def unpack_eval_rule(self, eval_ctx, x):
        return fusible_dtype.unpack(x)

    mesh = pltpu.TensorCoreMesh(axis_name="tc_core", num_cores=1)

    def subkernel(x_ref, y_ref, out_ref, x_vmem, y_vmem, out_vmem):
      pltpu.sync_copy(x_ref, x_vmem)
      pltpu.sync_copy(y_ref, y_vmem)
      packed = fusible_dtype.pack(
          x_vmem[...], y_vmem[...], dtype=SimpleFusionDType()
      )
      x_val, y_val = fusible_dtype.unpack(packed)
      out_vmem[...] = x_val + y_val
      pltpu.sync_copy(out_vmem, out_ref)

    @fuser.fusible
    def mpmd_f(x_fn, y_fn, z_fn):
      x = x_fn()
      y = y_fn()
      out = pl.kernel(
          body=subkernel,
          mesh=mesh,
          out_type=jax.ShapeDtypeStruct.like(x),
          scratch_types=(
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.VMEM(x.shape, x.dtype),
          ),
      )(x, y)
      if z_fn is None:
        z_fn = lambda x: x
      return z_fn(out)

    x = jnp.ones((8, 8), dtype=jnp.float32)
    y = jnp.ones((8, 8), dtype=jnp.float32)

    physicalized_f = fusible_dtype.physicalize(mpmd_f)
    res = physicalized_f(x, y)
    np.testing.assert_allclose(res, x + y)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
