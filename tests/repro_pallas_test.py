# Copyright 2025 The JAX Authors.
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

import contextlib
import dataclasses
from functools import partial
from typing import Any, Callable

from absl.testing import absltest
import numpy as np

import jax
try:
  import jax.experimental.pallas.ops.tpu.matmul
except ImportError:
  pass
from jax import export
from jax import lax
from jax import numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu_sc as plsc
try:
  import jax.experimental.pallas.ops.tpu.matmul
except ImportError:
  pass
from jax.sharding import PartitionSpec as P

from jax._src import config
from jax._src import dtypes
from jax._src.pallas.mosaic import tpu_info
from jax._src.pallas import mpmd
from jax._src.repro import tracker
from jax._src.repro import emitter
from jax._src.repro import repro_test_util as rtu

from jax._src import test_util as jtu
from jax._src import tree_util

try:
  import tensorflow as tf  # type: ignore
except ImportError:
  tf = None


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

intx = dtypes.default_int_dtype()
floatx = dtypes.default_float_dtype()


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Value:
  a: float
  b: float

@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CallableValue:
  a: float
  b: float

  def __call__(self):
    return self.a + self.b


flatten_custom_pytree = rtu.flatten_custom_pytree


@contextlib.contextmanager
def missing_lax_Precision_emitter():
  tracker.lazy_init()
  # Pretend we don't have a lax.Precision emitter
  old_lax_Precision_emitter = emitter._operand_emitter_by_type[lax.Precision]
  try:
    del emitter._operand_emitter_by_type[lax.Precision]
    yield
  finally:
    emitter._operand_emitter_by_type[lax.Precision] = old_lax_Precision_emitter


@jtu.with_config(jax_traceback_filtering="off",
                 jax_enable_checks=True)
class ReproTest(rtu.ReproTestBase):

  def run_or_export(
      self,
      fun: Callable[..., Any],
      *,
      platform: str,
      interpret: bool = False,
      min_tpu_version: int | None = None,
      dummy_return: Any = 0.0,
  ) -> Callable[..., Any]:
    """Function transformation that wraps `fun` into a callable `f(*args, **kwargs)`.

    When `f` is called, it determines whether `fun` can be run natively:
    - True if `interpret` is True.
    - True if `jtu.device_under_test() == platform` (and, if `platform == "tpu"` and
      `min_tpu_version` is specified, `jtu.is_device_tpu_at_least(min_tpu_version)`).
    - False otherwise.

    If native execution is supported, it calls `fun(*args, **kwargs)`.
    Otherwise, it exports `fun` for `platform` (mapping "gpu" to "cuda" for export)
    and returns `dummy_return` (evaluating `dummy_return(*args, **kwargs)` if callable).
    """
    def run_or_export_wrapper(*args, **kwargs):
      can_run_natively = False
      if interpret:
        can_run_natively = True
      elif jtu.device_under_test() == platform:
        if platform == "tpu" and min_tpu_version is not None:
          can_run_natively = jtu.is_device_tpu_at_least(min_tpu_version)
        else:
          can_run_natively = True

      if can_run_natively:
        return fun(*args, **kwargs)
      else:
        export_platform = "cuda" if platform == "gpu" else platform
        _ = export.export(fun, platforms=(export_platform,))(*args, **kwargs)
        if callable(dummy_return):
          return dummy_return(*args, **kwargs)
        return dummy_return
    return run_or_export_wrapper

  def test_pallas_call_0(self):
    m, n = 32, 4
    out_shape = jax.ShapeDtypeStruct((4, n), floatx)
    @jax.jit
    @partial(
        pl.pallas_call,
        interpret=True,
        out_shape=out_shape,
    )
    def slice_kernel(x_ref, y_ref):
      y_ref[:4, :4] = x_ref[:4, :4]
    x = np.arange(m * n, dtype=floatx).reshape((m, n))
    y = slice_kernel(x)
    self.collect_and_check(slice_kernel, y)

  def test_pallas_call_1(self):
    @partial(jax.jit, static_argnames=["bm", "bn", "bk",
                                       "interpret", "debug"])
    def matmul_block_spec(x, y, *, bm, bn, bk, interpret, debug=False):
      m, n, k = x.shape[0], y.shape[1], x.shape[1]

      @partial(
          pl.pallas_call,
          out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
          interpret=interpret,
          debug=debug,
          in_specs=[
              pl.BlockSpec((bm, x.shape[1]), lambda i, _: (i, 0),
                           memory_space=pltpu.MemorySpace.SMEM),
              pl.BlockSpec((y.shape[0], bn), lambda _, j: (0, j)),
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
          grid=(pl.cdiv(m, bm), pl.cdiv(n, bn)),
      )
      def matmul_kernel(x_ref, y_ref, o_ref):
        acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)

        def body(i, acc):
          x_block = x_ref[:, pl.ds(i * bk, bk)]
          y_block = y_ref[pl.ds(i * bk, bk), :]
          return acc + pl.dot(x_block, y_block)

        acc = lax.fori_loop(0, k // bk, body, acc).astype(o_ref.dtype)
        o_ref[:, :] = acc

      return matmul_kernel(x, y)

    x = y = np.ones((16, 16), dtype=np.float32)

    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(matmul_block_spec,
                             x, y, bm=2, bn=4, bk=8, interpret=True)

  def test_pallas_blockspec_outside_collect(self):
    bs = pl.BlockSpec((8, 8), lambda i, j: (i, j))

    @jax.jit
    @partial(
        pl.pallas_call,
        interpret=True,
        out_shape=jax.ShapeDtypeStruct((16, 16), floatx),
        in_specs=[bs],
        out_specs=pl.BlockSpec((8, 8), lambda i, j: (i, j)),
        grid=(2, 2),
    )
    def copy_kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...]

    x = np.arange(16 * 16, dtype=floatx).reshape((16, 16))
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(copy_kernel, x)

  def test_pallas_blockspec_calls_blockspec(self):
    def f(x):
      bs = pl.BlockSpec((8, 8), lambda i, j: (i, j))
      bs1 = pl.BlockSpec((8, 8), lambda i, j: bs.index_map(j, i))
      @jax.jit
      @partial(
          pl.pallas_call,
          interpret=True,
          out_shape=jax.ShapeDtypeStruct((16, 16), floatx),
          in_specs=[bs1],
          out_specs=pl.BlockSpec((8, 8), lambda i, j: (i, j)),
          grid=(2, 2),
      )
      def copy_kernel(x_ref, y_ref):
        y_ref[...] = x_ref[...]
      return copy_kernel(x)

    x = np.arange(16 * 16, dtype=floatx).reshape((16, 16))
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  def test_pallas_blockspec_replace(self):

    def top(x):
      # BlockSpec.replace() triggers __post_init__ which re-wraps the index_map
      # This happens sometimes, and we must handle it.
      bs = pl.BlockSpec((8, 8), lambda i, j: (i, j))
      bs2 = bs.replace(memory_space=pl.MemorySpace.DEFAULT)

      @jax.jit
      @partial(
          pl.pallas_call,
          interpret=True,
          out_shape=jax.ShapeDtypeStruct((16, 16), floatx),
          in_specs=[bs2],
          out_specs=pl.BlockSpec((8, 8), lambda i, j: (i, j)),
          grid=(2, 2),
      )
      def copy_kernel(x_ref, y_ref):
        y_ref[...] = x_ref[...]

      return copy_kernel(x)

    x = np.arange(16 * 16, dtype=floatx).reshape((16, 16))
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(top, x)

  def test_pallas_blockspec_reuse(self):

    def top(x, y):
      # Reuse the same BlockSpec object
      bs = pl.BlockSpec((8, 8), lambda i, j: (i, j))
      @jax.jit
      @partial(
          pl.pallas_call,
          interpret=True,
          out_shape=jax.ShapeDtypeStruct((16, 16), floatx),
          in_specs=[bs, bs],
          out_specs=bs,
          grid=(2, 2),
      )
      def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

      return add_kernel(x, y)

    x = np.arange(16 * 16, dtype=floatx).reshape((16, 16))
    y = np.ones((16, 16), dtype=floatx)
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(top, x, y)

  def test_pallas_ref_transforms(self):

    def kernel(x_ref, y_ref):
      ref = (
          x_ref.at[:16, :256]  # i32(16, 256)
          .bitcast(jnp.int16)  # i16(32, 256)
          .reshape((2, 16, 256))  # i16(2, 16, 256)
          .bitcast(jnp.float16)  # bf16(2, 16, 256)
          .at[1:, :, :]  # bf16(1, 16, 256)
          .reshape((16, 256))  # bf16(16, 256)
          .at[:, :128]  # bf16(16, 128)
          .bitcast(jnp.int32)  # i32(8, 128)
      )
      y_ref[...] = ref[...]

    x = jnp.arange(32 * 256, dtype=jnp.int32).reshape((32, 256))
    def myf():
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
          interpret=True)(x)

    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(myf)

  def test_export_shape_poly_error(self):
    @jax.jit
    def f(x):  # x: f32[w, h]
      def copy_one(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      return pl.pallas_call(
          copy_one,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

    def g():
      # TODO: we must wrap the generation of the symbolic variables in the
      # same top-level API calls as their use!
      w, h = export.symbolic_shape("w, h")
      export.export(f, platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

    self.collect_and_check(
      g,
      expect_exception=(
          ValueError,
          "shape polymorphism for Pallas does not support dynamically-shaped blocks"))

  def test_pallas_cost_analysis(self):
    def kernel(x, y):
      y[:] = x[:]
    x = jnp.arange(1024., dtype=jnp.float32).reshape(8, 128)
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        cost_estimate=pl.CostEstimate(
            flops=1234, transcendentals=21, bytes_accessed=12345
        ),
        interpret=True,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  def test_pallas_estimate_cost(self):
    @jax.jit
    def f(x):
      cost = pl.estimate_cost(lambda a: a + 1, x)
      return x + cost.flops

    x = jnp.ones((4, 4), dtype=jnp.int32)
    self.collect_and_check(f, x)

  def test_pallas_call_scalar_prefetch(self):
    def body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...]

    s = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    x = jnp.arange(2 * 8 * 8 * 4, dtype=jnp.int32).reshape((2, 8 * 8, 4))

    def _x_transform(i, s_ref):
      s = s_ref[i]
      return (s, 0)

    s = jnp.tile(s[None], [2, 1])

    @jax.jit
    @jax.vmap
    def kernel(s, x):
      return pl.pallas_call(
          body,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[
                  pl.BlockSpec((x.shape[0] // 8, x.shape[1]), _x_transform),
              ],
              out_specs=pl.BlockSpec(
                  (x.shape[0] // 8, x.shape[1]), lambda i, _: (i, 0)
              ),
              grid=8,
          ),
          compiler_params=pltpu.CompilerParams(
              allow_input_fusion=[False, True]
          ),
          interpret=True,
      )(s, x)

    with tracker.flags_override(fake_array_threshold=s.size + x.size + 1):
      self.collect_and_check(kernel, s, x)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  def test_pallas_core_map(self, *, interpret: bool):
    if jtu.device_under_test() not in ["cpu", "tpu"]:
      self.skipTest("Test runs only on TPU or CPU")

    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)

    @jax.jit
    def g(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        @pl.core_map(mesh, interpret=interpret)
        def _():
          def body(x_vmem, y_vmem):
            pltpu.sync_copy(x_ref, x_vmem)
            y_vmem[...] = x_vmem[...] + 1
            pltpu.sync_copy(y_vmem, y_ref)
          pl.run_scoped(body,
                        pltpu.VMEM(x_ref.shape, x_ref.dtype),
                        pltpu.VMEM(y_ref.shape, y_ref.dtype))
      _, y = pl.run_state(inner)((x, y))
      return y

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="tpu",
        interpret=interpret,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  def test_pallas_run_scoped(self, *, interpret: bool):
    if jtu.device_under_test() not in ["cpu", "tpu"]:
      self.skipTest("Test runs only on TPU or CPU")
    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)

    @jax.jit
    def g(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        @pl.core_map(mesh, interpret=interpret)
        def _():
          def scoped_body(scratch_ref):
            pltpu.sync_copy(x_ref, scratch_ref)
            scratch_ref[...] = scratch_ref[...] + 1
            pltpu.sync_copy(scratch_ref, y_ref)
          pl.run_scoped(scoped_body,
                        pltpu.VMEM((8, 128), jnp.int32))
      _, y = pl.run_state(inner)((x, y))
      return y

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="tpu",
        interpret=interpret,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  def test_pallas_run_scoped_interpret(self):
    self.skipTest("TODO: interpret mode does not support repros")
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("Pallas TPU interpret mode test runs only on CPU")

    mesh = pltpu.create_tensorcore_mesh("x", num_cores=2)

    @jax.jit
    def g(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        @pl.core_map(mesh, interpret=pltpu.InterpretParams(allow_hbm_allocation_in_run_scoped=True))
        def _():
          def body(sem):
            pl.semaphore_signal(sem, 1)
            pl.semaphore_wait(sem, 1)
            def copy(x_hbm_ref):
              pltpu.sync_copy(x_ref, x_hbm_ref)
              pltpu.sync_copy(x_hbm_ref, y_ref)
            pl.run_scoped(copy, pltpu.HBM(x_ref.shape, x_ref.dtype))
          pl.run_scoped(body, pltpu.SemaphoreType.REGULAR)
      _, y = pl.run_state(inner)((x, y))
      return y

    x = jnp.zeros((16, 128), dtype=jnp.int32)
    f = self.run_or_export(
        g,
        platform="tpu",
        interpret=True,
        dummy_return=0,
    )
    self.collect_and_check(f, x)

  @jtu.thread_unsafe_test()
  @rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V6E)
  def test_pallas_mpmd_map(self):
    if jtu.device_under_test() not in ["tpu", "cpu"]:
      self.skipTest("Test runs only on CPU and TPU")

    from jax._src.pallas import mpmd

    v_mesh = plsc.VectorSubcoreMesh(core_axis_name="x", subcore_axis_name="y")
    s_mesh = plsc.ScalarSubcoreMesh(axis_name="x")

    def vector_subcore_fn(x_ref, o_ref):
      pass

    def scalar_subcore_fn(x_ref, o_ref):
      pass

    def kernel(x):
      return mpmd.mpmd_map(
          [(v_mesh, vector_subcore_fn), (s_mesh, scalar_subcore_fn)],
          out_types=jax.ShapeDtypeStruct.like(x),
      )(x)

    @jax.jit
    def g(x):
      return kernel(x)

    x = jnp.zeros((8, 128), dtype=jnp.float32)
    f = self.run_or_export(
        g,
        platform="tpu",
        min_tpu_version=6,
        dummy_return=0.,
    )
    _ = f(x)
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  @jtu.thread_unsafe_test()
  def test_kernel_multiple_bodies(self):
    self.enter_context(rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V6E))
    from jax._src.pallas.mosaic.tpu_info import get_tpu_info

    @jax.jit
    def g():
      sc_info = get_tpu_info().sparse_core
      v_mesh = plsc.VectorSubcoreMesh(
          core_axis_name="s_core",
          subcore_axis_name="subcore",
          num_cores=sc_info.num_cores,
      )
      s_mesh = plsc.ScalarSubcoreMesh(
          axis_name="s_core", num_cores=sc_info.num_cores
      )

      def vector_subcore_fn(_, tec_sem):
        pl.semaphore_wait(tec_sem, 1)

      def scalar_subcore_fn(_, tec_sem):
        pl.semaphore_signal(
            tec_sem, device_id={"s_core": jax.lax.axis_index("s_core")})

      @partial(jax.shard_map, out_specs=None, check_vma=False)
      def mpmd_map_fun():
        pl.kernel(
            body=[vector_subcore_fn, scalar_subcore_fn],
            mesh=[v_mesh, s_mesh],
            out_type=jax.ShapeDtypeStruct([8], jnp.int32),
            scratch_types=[pltpu.SemaphoreType.REGULAR(()) @ v_mesh],
        )()
      mpmd_map_fun()

    f = self.run_or_export(
        g,
        platform="tpu",
        min_tpu_version=6,
        dummy_return=0.,
    )
    device_mesh = jax.make_mesh((jax.device_count(),), axis_names=("x",))
    self.enter_context(jax.sharding.set_mesh(device_mesh))
    self.collect_and_check(f)

  @jtu.thread_unsafe_test()
  def test_mpmd_map(self):
    self.enter_context(rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V6E))
    from jax._src.pallas.mosaic.tpu_info import get_tpu_info

    @jax.jit
    def g():
      sc_info = get_tpu_info().sparse_core
      v_mesh = plsc.VectorSubcoreMesh(
          core_axis_name="s_core",
          subcore_axis_name="subcore",
          num_cores=sc_info.num_cores,
      )
      s_mesh = plsc.ScalarSubcoreMesh(
          axis_name="s_core", num_cores=sc_info.num_cores
      )

      def vector_subcore_fn(_, tec_sem):
        pl.semaphore_wait(tec_sem, 1)

      def scalar_subcore_fn(_, tec_sem):
        pl.semaphore_signal(
            tec_sem, device_id={"s_core": jax.lax.axis_index("s_core")})

      return mpmd.mpmd_map(
          [(v_mesh, vector_subcore_fn), (s_mesh, scalar_subcore_fn)],
          out_types=jax.ShapeDtypeStruct([8], jnp.int32),
          scratch_types=[pltpu.SemaphoreType.REGULAR(()) @ v_mesh],
      )()

    f = self.run_or_export(
        g,
        platform="tpu",
        min_tpu_version=6,
        dummy_return=0.,
    )
    self.collect_and_check(f)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  @jtu.thread_unsafe_test()
  @rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V6E)
  def test_pallas_parallel_loop(self, *, interpret: bool):
    if jtu.device_under_test() not in ["tpu", "cpu"]:
      self.skipTest("Test runs only on CPU and TPU")
    if interpret:
      self.skipTest("parallel_loop does not support interpret mode")

    mesh = plsc.VectorSubcoreMesh(core_axis_name="x", subcore_axis_name="y")

    def body(x_ref, o_ref):
      @plsc.parallel_loop(0, x_ref.shape[0], 1)
      def _(i):
        pltpu.sync_copy(x_ref, o_ref)

    @jax.jit
    def g(x):
      compiler_params = pltpu.CompilerParams()
      return pl.kernel(body, out_type=x, mesh=mesh,
                        compiler_params=compiler_params)(x)

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="tpu",
        min_tpu_version=6,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  @jtu.thread_unsafe_test()
  @rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V6E)
  def test_pallas_sync_copy(self):
    if jtu.device_under_test() not in ["tpu", "cpu"]:
      self.skipTest("Test runs only on CPU and TPU")

    mesh = plsc.VectorSubcoreMesh(core_axis_name="x", subcore_axis_name="y")

    def body(x_ref, o_ref):
      @plsc.parallel_loop(0, x_ref.shape[0], 1)
      def _(i):
        def scoped_body(sem):
          pltpu.sync_copy(x_ref, o_ref)
          dma1 = pltpu.make_async_copy(src_ref=x_ref, dst_ref=o_ref, sem=sem)
          dma1.start()
          dma1.wait()

        pl.run_scoped(scoped_body, pltpu.SemaphoreType.DMA(()))

    @jax.jit
    def g(x):
      compiler_params = pltpu.CompilerParams()
      return pl.kernel(body, out_type=x, mesh=mesh,
                        compiler_params=compiler_params)(x)

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="tpu",
        min_tpu_version=6,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  @jtu.thread_unsafe_test()
  @rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V6E)
  def test_pallas_sync_copy_with_indices(self):
    if jtu.device_under_test() == "cpu" and config.enable_x64.value:
      # TODO
      self.skipTest("Fails on CPU with x64")

    if jtu.device_under_test() not in ["tpu", "cpu"]:
      self.skipTest("Test runs only on CPU and TPU")

    mesh = plsc.VectorSubcoreMesh(core_axis_name="x", subcore_axis_name="y")
    sc_info = plsc.get_sparse_core_info()
    nl = sc_info.num_lanes

    def body(x_hbm_ref, indices_hbm_ref, o_hbm_ref):
      def scoped_body(o_vmem_ref, indices_vmem_ref):
        pltpu.sync_copy(indices_hbm_ref, indices_vmem_ref)
        o_vmem_ref[...] = jax.lax.broadcast(-100_000, o_vmem_ref.shape)
        pltpu.sync_copy(
            x_hbm_ref.at[plsc.Indices(indices_vmem_ref, ignored_value=4)],
            o_vmem_ref,
        )
        pltpu.sync_copy(o_vmem_ref, o_hbm_ref)
      pl.run_scoped(
          scoped_body,
          pltpu.VMEM(x_hbm_ref.shape, x_hbm_ref.dtype),
          pltpu.VMEM(indices_hbm_ref.shape, indices_hbm_ref.dtype),
      )

    @jax.jit
    def g(x, indices):
      compiler_params = pltpu.CompilerParams()
      return pl.kernel(body, out_type=x, mesh=mesh,
                       compiler_params=compiler_params)(x, indices)

    x = jnp.arange(nl, dtype=jnp.int32)
    indices = jax.random.permutation(jax.random.key(42), x).astype(jnp.int32)
    f = self.run_or_export(
        g,
        platform="tpu",
        min_tpu_version=6,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + indices.size + 1):
      self.collect_and_check(f, x, indices)

  @jtu.with_explicit_mesh((2,), ("x",))
  @jtu.thread_unsafe_test()
  @rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V6E)
  def test_pallas_async_remote_copy(self, mesh):
    if jtu.device_under_test() not in ["tpu", "cpu"]:
      self.skipTest("Test runs only on CPU and TPU")

    def body(x_ref, idx_ref, o_ref):
      def scoped_body(sem1, sem2, idx_vmem):
        dma0 = pltpu.make_async_copy(idx_ref, idx_vmem, sem1)
        dma0.start()
        dma0.wait()

        axis_index = idx_vmem.get()[0]
        target_idx = (axis_index + 1) % 2

        dma = pltpu.make_async_remote_copy(x_ref, o_ref, sem1, sem2,
                                            device_id={"x": target_idx})
        dma.start()
        dma.wait_send()
        dma.wait_recv()

      pl.run_scoped(scoped_body,
                    pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
                    pltpu.VMEM((1,), jnp.int32))

      @jax.jit
      def g(x, idx):
        compiler_params = pltpu.CompilerParams()
        pallas_mesh = pltpu.create_tensorcore_mesh("x", num_cores=2)
        return pl.kernel(body, out_type=x, mesh=pallas_mesh,
                         compiler_params=compiler_params)(x, idx)

      @jax.jit
      def f(x, indices):
        from jax._src.pjit import reshard
        s_x = jax.sharding.NamedSharding(mesh, P("x", None))
        x = reshard(x, s_x)
        s_idx = jax.sharding.NamedSharding(mesh, P("x"))
        indices = reshard(indices, s_idx)
        return jax.shard_map(g, mesh=mesh, in_specs=(P("x", None), P("x")),
                             out_specs=P("x", None), check_vma=False)(x, indices)

      x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
      indices = jnp.arange(2, dtype=jnp.int32)
      top = self.run_or_export(
          f,
          platform="tpu",
          min_tpu_version=6,
          dummy_return=0.,
      )
      with tracker.flags_override(fake_array_threshold=x.size + 1):
        self.collect_and_check(top, x, indices)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  @jtu.thread_unsafe_test()
  @rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V5E)
  def test_pallas_kernel_basic(self, *, interpret: bool):
    if jtu.device_under_test() not in ["tpu", "cpu"]:
      self.skipTest("Test runs only on CPU and TPU")
    if config.enable_x64.value and not interpret:
      # TODO
      self.skipTest("x64 not supported in non-interpret mode: internal failure")

    if interpret:
      self.skipTest("pl.kernel interpret=True")

    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)

    def body(x_hbm_ref, o_hbm_ref):
      pltpu.emit_pipeline(body_vmem,
                          in_specs=pl.BlockSpec((8, 128), lambda i: i),
                          out_specs=pl.BlockSpec((8, 128), lambda i: i),
                          grid=(1,),
      )(x_hbm_ref, o_hbm_ref)

    def body_vmem(x_vmem_ref, o_vmem_ref):
      o_vmem_ref[...] = x_vmem_ref[...] + 1

    @jax.jit
    def g(x):
      compiler_params = pltpu.CompilerParams()
      return pl.kernel(body, out_type=x, mesh=mesh, interpret=interpret,
                       compiler_params=compiler_params)(x)

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="tpu",
        interpret=interpret,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  def test_pallas_gpu_kernel(self, *, interpret: bool):
    import jax.experimental.pallas.mosaic_gpu as plgpu

    if jtu.device_under_test() not in ["gpu", "cpu"]:
      self.skipTest("Test runs only on CPU and GPU")
    if jtu.device_under_test() == "gpu" and interpret:
      # TODO: error in interpret mode, wrong results
      self.skipTest("TODO: GPU + non-interpret mode is unsupported")

    def body(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1

    @jax.jit
    def g(x):
      return plgpu.kernel(
          body,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(1,),
          grid_names=("sm",),
          compiler_params=plgpu.CompilerParams(approx_math=True),
          interpret=interpret,
      )(x)

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="gpu",
        interpret=interpret,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  def test_pallas_gpu_emitters(self):
    from jax._src import core as jax_core
    from jax._src.pallas.mosaic_gpu import core as plgpu_core
    from jax._src.repro import emitter

    # All 7 GPU-specific values under test:
    transforms = (
        plgpu_core.TilingTransform((8, 8)),
        plgpu_core.SwizzleTransform(32),
        plgpu_core.UntilingTransform((8, 8)),
        plgpu_core.UnswizzleRef(32),
    )
    compiler_params = plgpu_core.CompilerParams(approx_math=True)
    gpu_memory_ref = plgpu_core.GPUMemoryRef(
        jax_core.ShapedArray((64, 64), np.float32),
        plgpu_core.MemorySpace.SMEM,
        transforms=transforms[:2]
    )
    wgmma_accumulator_ref = plgpu_core.WGMMAAccumulatorRef((64, 64), np.float32)

    tracker.lazy_init()
    global_ctx = emitter.EmitGlobalContext()
    func_ctx = emitter.EmitFunctionDefContext("test", global_ctx, None)

    self.assertEqual(func_ctx.traverse_value(transforms[0]), "plgpu_core.TilingTransform((8, 8))")
    self.assertEqual(func_ctx.traverse_value(transforms[1]), "plgpu_core.SwizzleTransform(32)")
    self.assertEqual(func_ctx.traverse_value(transforms[2]), "plgpu_core.UntilingTransform((8, 8))")
    self.assertEqual(func_ctx.traverse_value(transforms[3]), "plgpu_core.UnswizzleRef(32)")

    func_ctx.traverse_value(compiler_params)
    func_ctx.traverse_value(gpu_memory_ref)
    func_ctx.traverse_value(wgmma_accumulator_ref)

    # Check constructor string representation in local lines
    text = "\n".join(func_ctx.emitted_function.lines)
    self.assertIn("plgpu_core.CompilerParams(", text)
    self.assertIn("plgpu_core.GPUMemoryRef(", text)
    self.assertIn("plgpu_core.WGMMAAccumulatorRef(", text)

  def test_pallas_gpu_transforms_and_ref(self):
    if jtu.device_under_test() not in ["gpu", "cpu"]:
      self.skipTest("Test runs only on GPU or CPU")
    from jax.experimental.pallas import mosaic_gpu as plgpu  # type: ignore

    shape1 = (6 * 64, 8)
    shape2 = (2, 3, 64, 8)

    transforms = (plgpu.TilingTransform((8, 8)), plgpu.SwizzleTransform(32))

    def body(x_ref, out_ref, scratch_ref):
      x = plgpu.load(x_ref, (), layout=plgpu.Layout.WGMMA, optimized=False)
      scratch_ref[...] = x.reshape(shape2)
      out_ref[...] = scratch_ref[...]

    @jax.jit
    def g(x):
      return plgpu.kernel(
          body,
          out_shape=jax.ShapeDtypeStruct(shape2, jnp.float32),
          scratch_shapes=[plgpu.SMEM(shape2, jnp.float32, transforms=transforms)],
          compiler_params=plgpu.CompilerParams(approx_math=True),
          interpret=False,
      )(x)

    x = jnp.arange(6 * 64 * 8, dtype=jnp.float32).reshape(shape1)
    f = self.run_or_export(
        g,
        platform="gpu",
        dummy_return=lambda x: jnp.zeros(shape2, dtype=jnp.float32),
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  def test_pallas_gpu_wgmma(self, *, interpret: bool):
    if jtu.device_under_test() not in ["gpu", "cpu"]:
      self.skipTest("Test runs only on CPU and GPU")
    if interpret:
      self.skipTest("wgmma does not support interpret mode")

    import jax.experimental.pallas.mosaic_gpu as plgpu

    transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))

    def kernel(x_gmem, y_gmem, o_gmem):
      def acc_scope(acc_ref):
        def pipeline_body(step, x_smem, y_smem):
          plgpu.wgmma(acc_ref, x_smem, y_smem)
          plgpu.wgmma_wait(0)

        plgpu.emit_pipeline(
            pipeline_body,
            grid=(1,),
            in_specs=[
                plgpu.BlockSpec((64, 64), lambda i: (0, 0), transforms=transforms),
                plgpu.BlockSpec((64, 64), lambda i: (0, 0), transforms=transforms),
            ],
        )(x_gmem, y_gmem)
        return acc_ref[...]

      acc = pl.run_scoped(acc_scope, plgpu.ACC((64, 64)))

      def store_scope(o_smem):
        o_smem[...] = acc.astype(o_smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_gmem)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

      pl.run_scoped(
          store_scope,
          o_smem=plgpu.SMEM((64, 64), dtype=o_gmem.dtype, transforms=transforms)
      )

    @jax.jit
    def g(x, y):
      return plgpu.kernel(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(1,),
          grid_names=("sm",),
          compiler_params=plgpu.CompilerParams(
              lowering_semantics=plgpu.LoweringSemantics.Warpgroup
          ),
          interpret=interpret,
      )(x, y)

    x = jnp.ones((64, 64), dtype=jnp.float16)
    y = jnp.ones((64, 64), dtype=jnp.float16)
    f = self.run_or_export(
        g,
        platform="gpu",
        dummy_return=lambda x, y: jnp.zeros_like(x),
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x, y)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  def test_pallas_gpu_emit_pipeline_warp_specialized(self, *, interpret: bool):
    if jtu.device_under_test() not in ["gpu", "cpu"]:
      self.skipTest("Test runs only on CPU and GPU")
    if interpret:
      self.skipTest("emit_pipeline_warp_specialized does not support interpret mode")

    import jax.experimental.pallas.mosaic_gpu as plgpu

    num_steps = 4

    def kernel_body(_, x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    def kernel(x_gmem, o_gmem):
      in_specs = [
          plgpu.BlockSpec((64, 64), lambda i: (0, i))
      ]
      out_specs = [plgpu.BlockSpec((64, 64), lambda i: (0, i))]
      for _ in range(3):
        plgpu.emit_pipeline_warp_specialized(
            kernel_body,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(num_steps,),
            max_concurrent_steps=2,
            num_compute_wgs=1,
            memory_registers=40,
            wg_axis="wg",
        )(x_gmem, o_gmem)

    @jax.jit
    def g(x):
      return plgpu.kernel(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          num_threads=2,
          thread_name="wg",
          interpret=interpret,
      )(x)

    x = jnp.arange(64 * num_steps * 64)
    x = x.reshape(-1, num_steps * 64).astype(jnp.float32)
    f = self.run_or_export(
        g,
        platform="gpu",
        interpret=interpret,
        dummy_return=lambda x: jnp.zeros_like(x),
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  def test_pallas_semaphore_signal(self):

    def kernel(o_ref, sem):
      pl.semaphore_signal(sem, 1)
      pl.semaphore_wait(sem)

    @jax.jit
    def g():
      if jtu.device_under_test() == "gpu":
        import jax.experimental.pallas.mosaic_gpu as plgpu
        sem_type = plgpu.SemaphoreType.REGULAR
        return plgpu.kernel(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
            scratch_shapes=[sem_type],
        )()
      else:
        sem_type = pltpu.SemaphoreType.REGULAR
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
            scratch_shapes=[sem_type],
            interpret=True,
        )()

    f = self.run_or_export(
        g,
        platform="tpu" if jtu.device_under_test() == "tpu" else "gpu",
        dummy_return=0,
    )
    self.collect_and_check(f, skip_repro_eval=True)

  def test_pallas_gpu_barrier(self):
    if jtu.device_under_test() not in ["gpu", "cpu"]:
      self.skipTest("Test runs only on CPU, GPU")

    # This test reproduces the ReproError: Undefined g_... = PyTreeDef... error
    # when plgpu.barrier_arrive, plgpu.barrier_test, or plgpu.barrier_wait is used within
    # a Pallas Mosaic GPU kernel.
    # Because these functions are missing an api_boundary annotation, the underlying
    # primitive binds (and their transforms_treedef PyTreeDef kwarg) get recorded
    # as user statements in the kernel body function, causing an undefined PyTreeDef error
    # during repro source generation when error_mode="raise".

    import jax.experimental.pallas.mosaic_gpu as plgpu

    def kernel(o_ref, barrier):
      plgpu.barrier_arrive(barrier)
      _ = plgpu.barrier_test(barrier)
      plgpu.barrier_wait(barrier)

    @jax.jit
    def g():
      sem_type = plgpu.Barrier()
      return plgpu.kernel(
          kernel,
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          scratch_shapes=[sem_type],
          interpret=True,
      )()

    f = self.run_or_export(
        g,
        platform="gpu",
        dummy_return=0,
    )
    self.collect_and_check(
        f,
        expect_exception=(NotImplementedError, "primitive: barrier_arrive|Unimplemented primitive in Pallas Mosaic GPU lowering"),
        skip_repro_eval=True,
    )

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  def test_pallas_gpu_wgmma_transpose(self, *, interpret: bool):
    if jtu.device_under_test() not in ["gpu", "cpu"]:
      self.skipTest("Test runs only on CPU and GPU")
    if interpret:
      self.skipTest("wgmma does not support interpret mode")

    import jax.experimental.pallas.mosaic_gpu as plgpu

    transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))

    def kernel(x_gmem, y_gmem, o_gmem):
      def acc_scope(acc_ref):
        def pipeline_body(step, x_smem, y_smem):
          plgpu.wgmma(acc_ref, plgpu.transpose_ref(x_smem, (1, 0)), y_smem)
          plgpu.wgmma_wait(0)

        plgpu.emit_pipeline(
            pipeline_body,
            grid=(1,),
            in_specs=[
                plgpu.BlockSpec((64, 64), lambda i: (0, 0), transforms=transforms),
                plgpu.BlockSpec((64, 64), lambda i: (0, 0), transforms=transforms),
            ],
        )(x_gmem, y_gmem)
        return acc_ref[...]

      acc = pl.run_scoped(acc_scope, plgpu.ACC((64, 64)))

      def store_scope(o_smem):
        o_smem[...] = acc.astype(o_smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_gmem)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

      pl.run_scoped(
          store_scope,
          o_smem=plgpu.SMEM((64, 64), dtype=o_gmem.dtype, transforms=transforms)
      )

    @jax.jit
    def g(x, y):
      return plgpu.kernel(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(1,),
          grid_names=("sm",),
          compiler_params=plgpu.CompilerParams(
              lowering_semantics=plgpu.LoweringSemantics.Warpgroup
          ),
          interpret=interpret,
      )(x, y)

    x = jnp.ones((64, 64), dtype=jnp.float16)
    y = jnp.ones((64, 64), dtype=jnp.float16)
    f = self.run_or_export(
        g,
        platform="gpu",
        dummy_return=lambda x, y: jnp.zeros_like(x),
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x, y)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  @jtu.thread_unsafe_test()
  @rtu.mock_tpu_context(tpu_info.ChipVersion.TPU_V5E)
  def test_pallas_emit_pipeline_with_allocations(self, *, interpret: bool):
    if jtu.device_under_test() != "cpu" and not jtu.is_device_tpu_at_least(5):
      self.skipTest("Test runs only on CPU and TPU v5+")
    if jtu.device_under_test() == "cpu" and interpret:
      self.skipTest("interpret mode returns different values!!!")
    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)

    def kernel(x_ref, o_ref):
      def pipeline_body(x_inner, o_inner):
        o_inner[...] = x_inner[...] + 1
      in_specs = [pl.BlockSpec((4, 128), lambda i: (i, 0))]
      out_specs = pl.BlockSpec((4, 128), lambda i: (i, 0))
      pipeline, make_allocations = pltpu.emit_pipeline_with_allocations(
          pipeline_body,
          grid=(2,),
          in_specs=in_specs,
          out_specs=out_specs,
      )
      allocations = make_allocations(x_ref, o_ref)
      def with_allocations(allocations):
        pipeline(x_ref, o_ref, allocations=allocations)
      pl.run_scoped(with_allocations, allocations)

    @jax.jit
    def g(x):
      return pl.kernel(
          kernel,
          out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
          mesh=mesh,
          interpret=interpret,
      )(x)

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="tpu",
        interpret=interpret,
        min_tpu_version=5,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  @jtu.parameterized_filterable(
    kwargs=[dict(interpret=interpret)
                 for interpret in [False, True]
    ])
  def test_pallas_emit_pipeline_gpu(self, *, interpret: bool):
    if jtu.device_under_test() not in ["gpu", "cpu"]:
      self.skipTest("Test runs only on CPU and GPU")
    import jax.experimental.pallas.mosaic_gpu as plgpu

    if interpret:
      self.skipTest("TODO: maybe interpret mode does not work? no state "
                    "discharge rule for copy_gmem_to_smem")

    def kernel(x_gmem, o_gmem):
      def pipeline_body(step, x_smem, o_smem):
        o_smem[...] = x_smem[...] + 1
      plgpu.emit_pipeline(
          pipeline_body,
          grid=(2,),
          in_specs=[plgpu.BlockSpec((4, 128), lambda i: (i, 0))],
          out_specs=[plgpu.BlockSpec((4, 128), lambda i: (i, 0))],
      )(x_gmem, o_gmem)

    @jax.jit
    def g(x):
      return plgpu.kernel(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(1,),
          grid_names=("sm",),
          interpret=interpret,
      )(x)

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    f = self.run_or_export(
        g,
        platform="gpu",
        interpret=interpret,
        dummy_return=0.,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  def test_pallas_call_fusions_trivial(self):
    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      # Check the properties of a Fusion object wrapper
      self.assertEqual(x_fn.dtype, np.float32)
      self.assertEqual(x_fn.shape, (128, 128))
      self.assertEqual(x_fn.type.shape, (128, 128))
      self.assertEqual(x_fn.out_type.shape, (128, 128))
      self.assertEqual(x_fn.in_dtype, ((), {}))
      self.assertEqual(x_fn.in_shape, ((), {}))
      self.assertEqual(x_fn.in_type, ((), {}))

      x = x_fn()
      y = y_fn()
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x, y = f(x, y)
      return x, y * 2

    x = np.ones((128, 128), dtype=np.float32)
    y = np.ones((1, 128), dtype=np.float32)
    self.collect_and_check(g, x, y)

  def test_fusion_with_jit_example(self):
    @fuser.fusible
    def fusible_sin(x_fn, y_fn):
      if y_fn is None:
        y_fn = lambda x: x
      @jax.jit
      def _impl():
        x = x_fn()
        out = jnp.sin(x)
        return y_fn(out)
      return _impl()

    @fuser.fuse
    def f(x):
      x = 2 * x
      out = fusible_sin(x)
      return out + 1.0

    x = np.ones( (128,), dtype=jnp.float32)
    self.collect_and_check(f, x)

  def test_pallas_call_fusible_matmul(self):
    # Copied from tpu_fusible_matmult_test.py
    def matmul_kernel(
        x_scalar_prefetch,
        y_scalar_prefetch,
        z_scalar_prefetch,
        x_value_refs,
        y_value_refs,
        z_value_refs,
        o_ref,
        acc_ref,
        *,
        x_fn: Any,
        y_fn: Any,
        z_fn: Any,
        out_dtype: jnp.dtype,
    ):
      @pl.when(pl.program_id(2) == 0)
      def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

      pids = pl.program_id(0), pl.program_id(1), pl.program_id(2)
      scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

      x_values = jax.tree.map(lambda ref: ref.get(), x_value_refs)
      x = x_fn(pids, scalar_prefetch, x_values)
      y_values = jax.tree.map(lambda ref: ref.get(), y_value_refs)
      y = y_fn(pids, scalar_prefetch, y_values)
      acc_ref[...] += jnp.dot(x, y, preferred_element_type=jnp.float32)

      @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
      def _():
        acc = acc_ref[...].astype(out_dtype)
        z_values = jax.tree.map(lambda ref: ref.get(), z_value_refs)
        out = z_fn(pids, scalar_prefetch, z_values, acc)
        jax.tree.map(lambda ref, x: ref.set(x), o_ref, out)

    def _fusible_matmul(
        x: fuser.Fusion[[], jax.Array],  # pytype: disable=invalid-annotation
        y: fuser.Fusion[[], jax.Array],  # pytype: disable=invalid-annotation
        z: fuser.Fusion[[jax.Array], jax.Array] | None,  # pytype: disable=invalid-annotation
        *,
        bm: int,
        bk: int,
        bn: int,
        interpret: bool,
        debug: bool,
    ) -> jax.Array:
      m, k = x.shape
      k_, n = y.shape
      out_dtype = jnp.float32
      z_type = jax.ShapeDtypeStruct((m, n), dtype=out_dtype)
      if not z:
        z = lambda x: x
      if k != k_:
        raise ValueError(f'X and Y shapes must be compatible. Got {k} != {k_}')

      assert m % bm == 0
      assert k % bk == 0
      assert n % bn == 0
      grid = (m // bm, n // bn, k // bk)

      def x_index_map(i, j, k, *_):
        del j
        return i, k

      x_block_spec = pl.BlockSpec(block_shape=(bm, bk), index_map=x_index_map)

      def y_index_map(i, j, k, *_):
        del i
        return k, j

      y_block_spec = pl.BlockSpec(block_shape=(bk, bn), index_map=y_index_map)

      def z_index_map(i, j, k, *_):
        del k
        return i, j

      z_block_spec = pl.BlockSpec(block_shape=(bm, bn), index_map=z_index_map)
      dimension_semantics = (pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY)

      z_out_type = jax.eval_shape(z, z_type)

      x_fn, x_values, x_scalar_prefetch = fuser.get_fusion_values(x)
      y_fn, y_values, y_scalar_prefetch = fuser.get_fusion_values(y)
      z_fn, z_values, z_scalar_prefetch = fuser.get_fusion_values(z, z_type)

      scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

      x_fn, (x_value_block_specs,), _ = fuser.pull_block_spec(
          x_fn,
          x_block_spec,
          scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(0),
          grid_len=len(grid),
      )(x_values)

      y_fn, (y_value_block_specs,), _ = fuser.pull_block_spec(
          y_fn,
          y_block_spec,
          scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(1),
          grid_len=len(grid),
      )(y_values)

      z_out_block_spec = fuser.push_block_spec(z, z_block_spec)(z_type)
      z_fn, (z_value_block_specs, _), _ = fuser.pull_block_spec(
          z_fn,
          z_out_block_spec,
          scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(2),
          grid_len=len(grid),
      )(z_values, z_type)

      scalar_prefetch = jax.tree.map(lambda x: x[None], scalar_prefetch)

      return pl.pallas_call(
          partial(
              matmul_kernel,
              x_fn=x_fn,
              y_fn=y_fn,
              z_fn=z_fn,
              out_dtype=out_dtype,
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=len(scalar_prefetch),
              grid=grid,
              scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
              in_specs=[
                  x_value_block_specs,
                  y_value_block_specs,
                  z_value_block_specs,
              ],
              out_specs=[z_out_block_spec],
          ),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=dimension_semantics,
          ),
          out_shape=[z_out_type],
          interpret=interpret,
          debug=debug,
      )(
          *scalar_prefetch,
          x_values,
          y_values,
          z_values,
      )[0]

    def fusible_matmul(
        x: jax.Array,
        y: jax.Array,
        *,
        bm: int = 128,
        bk: int = 128,
        bn: int = 128,
        debug: bool = False,
        interpret: bool = True,  # TODO: False on a TPU backend
    ) -> jax.Array:
      return fuser.fusible(
          partial(
              _fusible_matmul,
              bm=bm,
              bk=bk,
              bn=bn,
              interpret=interpret,
              debug=debug,
          )
      )(x, y)

    dtype = np.float32
    k0, k1 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_relu(x, y):
      z = fusible_matmul(x, y)
      v = jnp.maximum(z, 0.0)
      return v

    def mm_ref(x, y):
      return jnp.dot(x, y, preferred_element_type=jnp.float32)

    @partial(jax.jit, compiler_options={"xla_allow_excess_precision": False})
    def matmul_relu_ref(x, y):
      return jax.nn.relu(mm_ref(x, y))

    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(matmul_relu, x, y)

  def test_pallas_call_custom_fusion_0(self):
    @fuser.custom_fusion
    def c(x, y):
      return x + y

    c.def_pull_block_spec(lambda bss: (bss[0], bss[0]))
    c.def_push_block_spec(lambda bss: (bss[0],))
    c.def_eval_rule(lambda _, x, y: (c(x, y),))

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
      return z_fn1(x), z_fn2(y)

    def g(x, y, z):
      x, y = f(x, c(y, z))
      return c(x, z), y * 2

    x = jax.random.normal(jax.random.key(0), (4, 4), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 4), dtype=jnp.float32)
    z = jax.random.normal(jax.random.key(2), (1, 4), dtype=jnp.float32)
    g(x, y, x)
    self.collect_and_check(jax.jit(fuser.fuse(g)), x, y, z)

  def test_pallas_call_custom_fusion_with_kwargs(self):
    @fuser.custom_fusion
    def c(x, y):  # y is passed as kwarg
      return x + y

    c.def_pull_block_spec(lambda bss: (bss[0], bss[0]))
    c.def_push_block_spec(lambda bss: (bss[0],))
    c.def_eval_rule(lambda _, x, y: (c(x, y=y),))

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
      return z_fn1(x), z_fn2(y)

    def g(x, y1, y2):
      x, y3 = f(x, c(y1, y=y2))
      return c(x, y=y2), y3 * 2

    x = jax.random.normal(jax.random.key(0), (4, 4), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 4), dtype=jnp.float32)
    z = jax.random.normal(jax.random.key(2), (1, 4), dtype=jnp.float32)
    g(x, y, z)
    self.collect_and_check(jax.jit(fuser.fuse(g)), x, y, z)

  def test_fuser_pull_block_spec(self):
    def fn(x):
      return jnp.exp(x)

    def f(x):
      return fn(x)

    def top():
      in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)
      f2, new_values, scalar_prefetch_values = \
        fuser.get_fusion_values(f, in_type)

      block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
      kernel_fn, (value_block_specs, in_block_spec), _ = (
          fuser.pull_block_spec(
              f2,
              block_spec,
              grid_len=3,
              scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(0),
          )(new_values, in_type)
      )

      x = jnp.ones((128, 128), dtype=np.float32)
      return kernel_fn((0, 0, 0), scalar_prefetch_values, (), x)

    with tracker.flags_override(fake_array_threshold=128 * 128 + 1):
      self.collect_and_check(top)

  def test_fuser_pull_block_spec_out_block_specs_is_kwarg(self):
    def fn(x):
      return jnp.exp(x)

    def f(x):
      return fn(x)

    def top():
      in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)
      f2, new_values, scalar_prefetch_values = \
        fuser.get_fusion_values(f, in_type)

      block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
      kernel_fn, (value_block_specs, in_block_spec), _ = (
          fuser.pull_block_spec(
              f2,
              out_block_specs=block_spec,
              grid_len=3,
              scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(0),
          )(new_values, in_type)
      )

      x = jnp.ones((128, 128), dtype=np.float32)
      return kernel_fn((0, 0, 0), scalar_prefetch_values, (), x)

    with tracker.flags_override(fake_array_threshold=128 * 128 + 1):
      self.collect_and_check(top)

  def test_fuser_push_block_spec(self):
    def fn(x):
      return jnp.exp(x)

    def f(x):
      return fn(x)

    def top():
      in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)

      block_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
      out_block_spec = fuser.push_block_spec(f, block_spec)(in_type)
      return out_block_spec.block_shape

    with tracker.flags_override(fake_array_threshold=128 * 128 + 1):
      self.collect_and_check(top)

  def test_fuser_push_block_spec_pytrees_kwargs(self):
    def fn(x, y, *, z):
      return (jnp.exp(x) + y), (z + y)

    def f(x_and_y, *, z):
      x, y = x_and_y
      return fn(x, y, z=z)

    def top():
      in_type = (
          jax.ShapeDtypeStruct((512, 512), jnp.float32),
          jax.ShapeDtypeStruct((512, 512), jnp.float32),
      )
      z_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)

      block_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
      (out_block_spec1, out_block_spec2) = fuser.push_block_spec(
          f,
          (block_spec, pl.no_block_spec),
          z=block_spec,
      )(in_type, z=z_type)
      return out_block_spec1.block_shape, out_block_spec2.block_shape

    with tracker.flags_override(fake_array_threshold=128 * 128 + 1):
      self.collect_and_check(top)

  def test_fuser_evaluate(self):
    def f(x):
      return jnp.sin(x) + 1.0

    x = np.ones((128,), dtype=jnp.float32)
    self.collect_and_check(jax.jit(fuser.evaluate(f)), x)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
