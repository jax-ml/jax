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
"""Tests for Pallas on SparseCore."""

import collections
import functools
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic import sc_core
from jax._src.state import discharge as state_discharge
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np


jtu.setup_hypothesis()
jax.config.parse_flags_with_absl()


class PallasSCTest(jtu.JaxTestCase):
  USE_TC_TILING = False

  def setUp(self):
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")

    if self.USE_TC_TILING and jtu.is_cloud_tpu():
      # TODO(apaszke,slebedev): Fix those.
      self.skipTest("Many tests are failing on Cloud TPUs")

    super().setUp()

  @property
  def sc_info(self):
    return plsc.get_sparse_core_info()

  def vector_subcore_kernel(self, **kwargs):
    assert "compiler_params" not in kwargs
    return functools.partial(
        pl.pallas_call,
        **kwargs,
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
            use_tc_tiling_on_sc=self.USE_TC_TILING,
        ),
    )

  def kernel(self, **kwargs):
    assert "compiler_params" not in kwargs
    return functools.partial(
        pl.kernel,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=self.USE_TC_TILING
        ),
        **kwargs,
    )

  def skip_if_tc_tiling(self, reason: str = ""):
    if self.USE_TC_TILING:
      self.skipTest(f"TC tiling is not supported. {reason}")


class DebugPrintTest(PallasSCTest):

  def setUp(self):
    if jtu.is_cloud_tpu():
      # TODO(slebedev): Investigate this and remove the skip.
      self.skipTest("Fails on Cloud TPUs")

    super().setUp()

  @parameterized.product(dtype=[jnp.int32, jnp.float32])
  def test_vector_subcore(self, dtype):
    x = jnp.arange(16, dtype=dtype)
    debug_int = 1234552
    debug_float = 12344.625

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_hbm_ref, _):
      pl.debug_print("Memref", x_hbm_ref)
      x = x_hbm_ref[:8] + 100
      pl.debug_print("Vector value", x)
      masks = x < 103
      pl.debug_print("Masks", masks)
      pl.debug_print("Single int", debug_int)
      pl.debug_print("Single float", debug_float)
      pl.debug_print("No values")

    compiled_kernel = jax.jit(
        kernel,
        compiler_options={
            "xla_tpu_enable_sc_log_recorder": "true",
            "xla_tpu_enable_tile_log_recorder": "true",
        },
    )
    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(compiled_kernel(x))
    self.assertIn("Memref", get_output())
    self.assertIn("0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15", get_output())
    self.assertIn("Vector value", get_output())
    self.assertIn("100, 101, 102, 103, 104, 105, 106, 107", get_output())
    self.assertIn("Masks", get_output())
    self.assertIn("1, 1, 1, 0, 0, 0, 0, 0", get_output())
    self.assertIn("Single int, data: s32[1]", get_output())
    self.assertIn(str(debug_int), get_output())
    self.assertIn("Single float, data: f32[1]", get_output())
    self.assertIn(str(debug_float), get_output())
    self.assertIn("No values", get_output())

  def test_scalar_subcore(self):
    int32s = jnp.arange(512, dtype=jnp.int32).reshape(64, 8)
    int16s = jnp.arange(512, dtype=jnp.int16).reshape(32, 16)
    int8s = jnp.arange(512, dtype=jnp.int8).reshape(16, 32)
    debug_int = 1234552
    debug_float = 12344.625

    @self.kernel(
        out_shape=int32s,
        mesh=plsc.ScalarSubcoreMesh(
            axis_name="core", num_cores=self.sc_info.num_cores
        ),
    )
    def kernel(int32s_hbm_ref, int16s_hbm_ref, int8s_hbm_ref, o_hbm_ref):
      @functools.partial(
          pl.run_scoped,
          tmp_ref=pltpu.VMEM_SHARED(int32s.shape, int32s.dtype),
          sem=pltpu.SemaphoreType.DMA,
      )
      def _(tmp_ref, sem):
        @pl.when(lax.axis_index("core") == 0)
        def _():
          pltpu.async_copy(int32s_hbm_ref, tmp_ref, sem).wait()
          pltpu.async_copy(tmp_ref, o_hbm_ref, sem).wait()
          pl.debug_print("s32 array", tmp_ref)
          pl.debug_print("s16 array", int16s_hbm_ref)
          pl.debug_print("s8 array", int8s_hbm_ref)
          pl.debug_print("Single int", debug_int)
          pl.debug_print("Single float", debug_float)
          pl.debug_print("No values")

    compiled_kernel = jax.jit(
        kernel, compiler_options={"xla_tpu_enable_sc_log_recorder": "true"}
    )
    with jtu.capture_stderr() as get_output:
      jax.block_until_ready(compiled_kernel(int32s, int16s, int8s))
    print(get_output())
    self.assertIn("s32 array, data: s32", get_output())
    self.assertIn("{ 8, 9, 10, 11, 12, 13, 14, 15 }", get_output())
    self.assertIn("s16 array, data: s16", get_output())
    self.assertIn(
        "{ 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 }",
        get_output(),
    )
    self.assertIn("s8 array, data: s8", get_output())
    self.assertIn(
        "{ 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47"
        ", 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 }",
        get_output(),
    )
    self.assertIn("Single int", get_output())
    self.assertIn(str(debug_int), get_output())
    self.assertIn("Single float", get_output())
    self.assertIn(str(debug_float), get_output())
    self.assertIn("No values", get_output())


class VectorSubcoreTest(PallasSCTest):

  # Used for testing masked loads and stores below
  MASK_FNS = [lambda x: x < 4, lambda x: x >= 4, lambda x: x % 2 == 0]

  @parameterized.product(
      dtype=[jnp.int32, jnp.float32], op=[jnp.add, jnp.subtract]
  )
  def test_add_sub_one(self, dtype, op):
    x = jnp.arange(8, dtype=dtype)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = op(x, 1)

    np.testing.assert_array_equal(kernel(x), op(x, 1))

  def test_add_one_block_specs(self):
    x = jnp.arange(32, dtype=jnp.int32)

    @self.vector_subcore_kernel(
        out_shape=x,
        grid=(4,),
        out_specs=pl.BlockSpec([8], lambda i: i),
        in_specs=[pl.BlockSpec([8], lambda i: i)],
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1

    np.testing.assert_array_equal(kernel(x), x + 1)

  @parameterized.named_parameters(*(
      dict(
          testcase_name=(
              f"_{'x'.join(map(str, shape))}x{dtype.name}_{minor_scale}"
          ),
          dtype=dtype,
          out_shape=shape,
          minor_scale=minor_scale,
      )
      for dtype, shapes in sc_core.SUPPORTED_VECTOR_SHAPES.items()
      for shape in shapes
      if math.prod(shape) * dtype.itemsize == 32
      for minor_scale in [1, 2, 4]
  ))
  def test_slicing(self, dtype, out_shape, minor_scale):
    self.skip_if_tc_tiling()

    if dtype == jnp.float16 and jtu.is_device_tpu(6, "e"):
      # TODO(b/433704850): Remove this once the bug is fixed.
      self.skipTest("Crashes")

    crashing = {
        "int16": [(2, 8)],
        "uint16": [(2, 8)],
        "float16": [(2, 8)],
        "bfloat16": [(2, 8)],
        "int8": [(4, 8)],
        "uint8": [(4, 8)],
    }
    if out_shape in crashing.get(dtype.name, []):
      self.skipTest("Crashes")

    out_minor = out_shape[-1]
    in_minor = out_minor * minor_scale
    in_shape = (*out_shape[:-1], in_minor)
    indices = [
        slice(i * out_minor, (i + 1) * out_minor) for i in range(minor_scale)
    ]

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=out_shape, dtype=dtype),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = sum(x_ref[..., idx] for idx in indices)

    x = jnp.arange(math.prod(in_shape), dtype=dtype).reshape(in_shape)
    np.testing.assert_array_equal(
        kernel(x), sum(x[..., idx] for idx in indices)
    )

  @parameterized.product(major_dim=[2, 3, 4])
  def test_get_index(self, major_dim):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = lax.fori_loop(
          1, major_dim, lambda i, acc: acc + x_ref[i], x_ref[0]
      )

    x = jnp.arange(8 * major_dim).reshape(major_dim, 8)
    np.testing.assert_array_equal(kernel(x), x.sum(axis=0))

  def test_get_multi_index(self):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.zeros_like(o_ref)
      for i, j in itertools.product(*map(range, x_ref.shape[:-1])):
        o_ref[...] += x_ref.at[i][j]

    x = jnp.arange(3 * 4 * 8).reshape(3, 4, 8)
    np.testing.assert_array_equal(kernel(x), x.sum(axis=(0, 1)))

  @jtu.thread_unsafe_test(condition=not jtu.hypothesis_is_thread_safe())
  @hp.given(hps.data())
  def test_block_spec_untiled_slicing(self, data):
    if not self.USE_TC_TILING:
      self.skipTest(
          "Test uncovers a bug: @reproduce_failure('6.80.0', b'AAEBAQAAAAA=')"
      )
    else:
      self.skipTest(
          "Test uncovers a bug: @reproduce_failure('6.80.0', b'AAEAAQAAAAA=')"
      )
    slice_shape = data.draw(
        hps.lists(
            hps.integers(1, 3), min_size=(1 + self.USE_TC_TILING), max_size=4
        )
    )
    if self.USE_TC_TILING:
      slice_shape[-2] *= 8
      slice_shape[-1] *= 128
    else:
      slice_shape[-1] *= 8
    max_elems = 12000 if jtu.is_device_tpu(6, "e") else 25000
    hp.assume(math.prod(slice_shape) <= max_elems)  # Avoid OOMs.
    rank = len(slice_shape)
    offsets = data.draw(
        hps.lists(hps.integers(0, 4), min_size=rank, max_size=rank)
    )
    full_shape = tuple(s * (o + 2) for s, o in zip(slice_shape, offsets))

    def nd_loop(bounds, body, *, _idxs = ()):
      if not bounds:
        body(*_idxs)
        return
      bound, *other_bounds = bounds
      def _loop_body(i, _):
        nd_loop(other_bounds, body, _idxs=(*_idxs, i))
      jax.lax.fori_loop(0, bound, _loop_body, None)

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=slice_shape, dtype=jnp.int32),
        in_specs=[pl.BlockSpec(slice_shape, lambda: offsets)],
    )
    def kernel(x_ref, o_ref):
      slice_vec_shape = (*slice_shape[:-1], slice_shape[-1] // 8)
      def copy(*idxs):
        idxs = (*idxs[:-1], pl.ds(idxs[-1] * 8, 8))
        o_ref[idxs] = x_ref[idxs]
      nd_loop(slice_vec_shape, copy)

    x = jnp.arange(math.prod(full_shape)).reshape(full_shape)
    np_slc = tuple(slice(o * s, (o + 1) * s) for o, s in zip(offsets, slice_shape))
    np.testing.assert_array_equal(kernel(x), x[np_slc])

  @parameterized.product(major_dim=[2, 3, 4])
  def test_swap_index(self, major_dim):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(major_dim, 8), dtype=jnp.int32),
    )
    def kernel(x_ref, o_ref):
      @pl.loop(0, major_dim)
      def _(i):
        o_ref[major_dim - 1 - i] = x_ref[i]

    x = jnp.arange(major_dim * 8).reshape(major_dim, 8)
    np.testing.assert_array_equal(kernel(x), x[::-1])

  @parameterized.product(shape=[(8,), (16,), (8, 8), (16, 8), (8, 16, 8)])
  def test_scatter_major(self, shape):
    self.skip_if_tc_tiling()
    x = jnp.arange(math.prod(shape)).reshape(shape)
    major_dim, *_ = shape
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(major_dim))

    @self.vector_subcore_kernel(
        out_shape=x, out_specs=pl.BlockSpec(memory_space=pltpu.HBM)
    )
    def kernel(x_ref, indices_ref, o_hbm_ref):
      @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.DMA)
      def _(sem):
        pltpu.async_copy(x_ref, o_hbm_ref.at[indices_ref], sem).wait()

    np.testing.assert_array_equal(
        kernel(x, indices), jnp.empty_like(x).at[indices].set(x)
    )

  def test_scatter_1d_array(self):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(
        out_shape=x, out_specs=pl.BlockSpec(memory_space=pltpu.HBM)
    )
    def kernel(x_ref, indices_ref, o_hbm_ref):
      pltpu.sync_copy(x_ref, o_hbm_ref.at[indices_ref[...]])

    np.testing.assert_array_equal(
        kernel(x, indices), jnp.empty_like(x).at[indices].set(x)
    )

  def test_scatter_1d_array_from_transformed_src(self):
    self.skip_if_tc_tiling()
    x = jnp.arange(16).reshape(2, -1)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(
        out_shape=x[0], out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )
    def kernel(x_ref, indices_ref, o_hbm_ref):
      pltpu.sync_copy(x_ref.at[0], o_hbm_ref.at[indices_ref[...]])

    np.testing.assert_array_equal(
        kernel(x, indices), jnp.empty_like(x[0]).at[indices].set(x[0])
    )

  @parameterized.product(kind=["ref", "array"])
  def test_gather_1d(self, kind):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), x)

    @self.vector_subcore_kernel(
        out_shape=x,
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      indices = indices_ref if kind == "ref" else indices_ref[...]
      pltpu.sync_copy(x_hbm_ref.at[indices], o_ref)

    np.testing.assert_array_equal(kernel(x, indices), x[indices])

  @parameterized.product(kind=["ref", "array"])
  def test_gather_1d_to_transformed_dst(self, kind):
    self.skip_if_tc_tiling()
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), x)

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(2, 8,), dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      indices = indices_ref if kind == "ref" else indices_ref[...]
      pltpu.sync_copy(x_hbm_ref.at[indices], o_ref.at[0])

    np.testing.assert_array_equal(kernel(x, indices)[0], x[indices])

  def test_large_gather_1d(self):
    x = jnp.arange(1024)
    indices = jax.random.permutation(jax.random.key(42), x)

    @self.vector_subcore_kernel(
        out_shape=x,
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[indices_ref], o_ref)

    np.testing.assert_array_equal(kernel(x, indices), x[indices])

  def test_gather_1d_with_indexing(self):
    self.skip_if_tc_tiling("Small 1d gather does not work on TC tiling.")
    x = jnp.arange(4 * 4 * 8).reshape(4, 4, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[1, 2].at[indices_ref], o_ref)

    np.testing.assert_array_equal(kernel(x, indices), x[1, 2, indices])

  def test_gather_2d_with_indexing(self):
    x = jnp.arange(4 * 16 * 128).reshape(4, 16, 128)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8, 128,), dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[1, pl.ds(8, 8), :].at[indices_ref], o_ref)

    np.testing.assert_array_equal(kernel(x, indices), x[1, 8:][indices])

  def test_gather_1d_with_indexed_ref(self):
    x = jnp.arange(16)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(16))

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[indices_ref.at[:indices.size // 2]], o_ref)

    np.testing.assert_array_equal(
        kernel(x, indices), x[indices[:indices.size // 2]]
    )

  def test_gather_1d_with_dynamically_sized_ref(self):
    self.skip_if_tc_tiling()
    x = jnp.arange(16)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(16))

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32),
        grid=(1,),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pid = pl.program_id(0)  # Always zero.
      num_indices = pid + indices_ref.size // 2
      pltpu.sync_copy(
          x_hbm_ref.at[indices_ref.at[pl.ds(0, num_indices)]],
          o_ref.at[pl.ds(0, num_indices)],
      )

    np.testing.assert_array_equal(
        kernel(x, indices), x[indices[: indices.size // 2]]
    )

  def test_gather_1d_with_dynamically_sized_2d_ref(self):
    self.skip_if_tc_tiling()

    x = jnp.arange(16)
    indices = jax.random.permutation(
        jax.random.key(42), jnp.arange(2 * 16).reshape(2, -1), axis=1
    )

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(
            shape=(indices.size // 4,), dtype=jnp.int32
        ),
        grid=(1,),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pid = pl.program_id(0)  # Always zero.
      num_indices = pid + indices_ref.size // 4
      pltpu.sync_copy(
          x_hbm_ref.at[indices_ref.at[pid, pl.ds(0, num_indices)]],
          o_ref.at[pl.ds(0, num_indices)],
      )

    np.testing.assert_array_equal(
        kernel(x, indices), x[indices[0, : indices.size // 4]]
    )

  def test_invalid_gather_1d_with_extra_transforms(self):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), x)

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[indices_ref].reshape(o_ref.size), o_ref)

    with self.assertRaisesRegex(
        NotImplementedError, "cannot have any transforms following the indexer"
    ):
      kernel(x, indices)

  def test_invalid_gather_1d_with_indexed_destination(self):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), x)

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[indices_ref], o_ref.at[indices_ref])

    with self.assertRaisesRegex(ValueError, "source ref can be indexed"):
      kernel(x, indices)

  def test_invalid_gather_1d_memory_space(self):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), x)

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[indices_ref], o_ref)

    with self.assertRaisesRegex(
        NotImplementedError, "from HBM to HBM is not supported"
    ):
      kernel(x, indices)

  def test_invalid_gather_1d_offsets_memory_space(self):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), x)

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[indices_ref], o_ref)

    with self.assertRaisesRegex(
        NotImplementedError, "must be in VMEM, got HBM"
    ):
      kernel(x, indices)

  def test_implicit_gather_1d(self):
    self.skip_if_tc_tiling()
    num_steps = 4
    x = jnp.arange(num_steps * 8).reshape(num_steps, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(num_steps))

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(num_steps, 8), dtype=jnp.int32),
        grid=(num_steps,),
        in_specs=(
            plsc.BlockSpec((1, 8), indexed_by=1, indexed_dim=0),
            pl.BlockSpec((1,), lambda i: i),
        ),
        out_specs=pl.BlockSpec((1, 8), lambda i: (0, i)),
    )
    def kernel(x_ref, indices_ref, o_ref):
      del indices_ref  # Unused.
      o_ref[...] = x_ref[...]

    np.testing.assert_array_equal(
        kernel(x, indices), jnp.take(x, indices, axis=0)
    )

  def test_load_gather_1d(self):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      o_ref[...] = plsc.load_gather(x_ref, [indices_ref[...]])

    np.testing.assert_array_equal(kernel(x, indices), x[indices])

  def test_load_gather_2d(self):
    x = jnp.arange(8 * 8).reshape(8, -1)
    indices0 = indices1 = jax.random.permutation(
        jax.random.key(42), jnp.arange(8)
    )

    @self.vector_subcore_kernel(out_shape=jax.ShapeDtypeStruct((8,), x.dtype))
    def kernel(x_ref, indices0_ref, indices1_ref, o_ref):
      o_ref[...] = plsc.load_gather(
          x_ref, [indices0_ref[...], indices1_ref[...]]
      )

    np.testing.assert_array_equal(
        kernel(x, indices0, indices1), x[indices0, indices1]
    )

  def test_load_gather_with_indexing(self):
    self.skip_if_tc_tiling()
    num_steps = 4
    x = jnp.arange(num_steps * 8).reshape(num_steps, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      indices = indices_ref[...]
      for i in range(num_steps):
        o_ref[i] = plsc.load_gather(x_ref.at[i], [indices])

    out = kernel(x, indices)
    for i in range(num_steps):
      np.testing.assert_array_equal(out[i], x[i][indices])

  @parameterized.parameters(*MASK_FNS)
  def test_load_gather_masked(self, mask_fn):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      o_ref[...] = plsc.load_gather(
          x_ref, [indices_ref[...]], mask=mask_fn(x_ref[...])
      )

    mask = mask_fn(x)
    np.testing.assert_array_equal(kernel(x, indices)[mask], x[indices][mask])

  def test_store_scatter(self):
    self.skip_if_tc_tiling()
    num_steps = 4
    x = jnp.arange(num_steps * 8).reshape(num_steps, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      indices = indices_ref[...]
      o_ref[...] = jnp.zeros_like(o_ref)
      for i in range(num_steps):
        plsc.store_scatter(o_ref.at[i], [indices], x_ref[i])

    out = kernel(x, indices)
    for i in range(num_steps):
      np.testing.assert_array_equal(
          out[i], jnp.zeros_like(x[i]).at[indices].set(x[i])
      )

  @parameterized.parameters(*MASK_FNS)
  def test_store_scatter_masked(self, mask_fn):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = jnp.zeros_like(o_ref)
      plsc.store_scatter(o_ref, [indices_ref[...]], x, mask=mask_fn(x))

    mask = mask_fn(x)
    np.testing.assert_array_equal(
        kernel(x, indices),
        jnp.zeros_like(x).at[indices[mask]].set(x[mask]),
    )

  def test_store_scatter_2d(self):

    num_steps = 4
    x = jnp.arange(num_steps * 8).reshape(num_steps, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      indices = indices_ref[...]
      o_ref[...] = jnp.zeros_like(o_ref)
      for i in range(num_steps):
        plsc.store_scatter(
            o_ref, [jnp.full(indices.shape, i), indices], x_ref[i])

    out = kernel(x, indices)
    for i in range(num_steps):
      np.testing.assert_array_equal(
          out[i], jnp.zeros_like(x[i]).at[indices].set(x[i])
      )

  @parameterized.parameters(*MASK_FNS)
  def test_addupdate_scatter(self, mask_fn):
    self.skip_if_tc_tiling()
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = jnp.ones_like(o_ref)
      plsc.addupdate_scatter(o_ref, [indices_ref[...]], x, mask=mask_fn(x))

    mask = mask_fn(x)
    np.testing.assert_array_equal(
        kernel(x, indices),
        jnp.ones_like(x).at[indices[mask]].add(x[mask]),
    )

  @parameterized.parameters(*MASK_FNS)
  def test_load_expanded(self, mask_fn):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plsc.load_expanded(x_ref.at[...], mask=mask_fn(x_ref[...]))

    x = jnp.arange(8)
    mask = mask_fn(x)
    expected = jnp.zeros_like(x).at[mask].set(x[: mask.sum()])
    np.testing.assert_array_equal(kernel(x)[mask], expected[mask])

  @parameterized.parameters(*MASK_FNS)
  def test_store_compressed(self, mask_fn):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      x = x_ref[...]
      plsc.store_compressed(o_ref.at[...], x, mask=mask_fn(x))

    x = jnp.arange(8)
    mask = mask_fn(x)
    np.testing.assert_array_equal(kernel(x)[: mask.sum()], x[mask])

  def test_addupdate(self):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(o_ref):
      o_ref[...] = jnp.zeros_like(o_ref)
      for i in range(8):
        plsc.addupdate(o_ref.at[...], lax.broadcast(i, o_ref.shape))

    np.testing.assert_array_equal(kernel(), jnp.full(8, jnp.arange(8).sum()))

  @parameterized.parameters(*MASK_FNS)
  def test_addupdate_compressed(self, mask_fn):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.zeros_like(o_ref)
      for i in range(8):
        plsc.addupdate_compressed(
            o_ref.at[...],
            lax.broadcast(i, o_ref.shape),
            mask=mask_fn(x_ref[...]),
        )

    x = jnp.arange(8)
    mask = mask_fn(x)
    np.testing.assert_array_equal(
        kernel(x)[: mask.sum()], jnp.full(mask.sum(), jnp.arange(8).sum())
    )

  @parameterized.product(
      dtype=[jnp.int32], new_dtype=[jnp.int8, jnp.int16, jnp.float32]
  )
  def test_bitcast(self, dtype, new_dtype):
    self.skip_if_tc_tiling()
    new_shape = (
        8 * jnp.dtype(dtype).itemsize // jnp.dtype(new_dtype).itemsize,
    )

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=new_shape, dtype=new_dtype)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plsc.bitcast(x_ref[...], o_ref.dtype)

    x = jnp.arange(8, dtype=dtype)
    np.testing.assert_array_equal(kernel(x), x.view(new_dtype))

  def test_bitcast_invalid(self):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=[1], dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plsc.bitcast(x_ref[...], o_ref.dtype)

    x = jnp.arange(2, dtype=jnp.int8)
    with self.assertRaisesRegex(ValueError, "is not divisible"):
      kernel(x)

  def test_lax_bitcast(self):
    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct((8,), jnp.uint32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...].view(o_ref.dtype)

    x = jnp.arange(8, dtype=jnp.float32)
    np.testing.assert_array_equal(kernel(x), x.view(np.uint32))

  def test_ref_bitcast(self):
    # TODO: b/443906446 - Remove the skip once we can lower such bitcasts.
    self.skipTest("Ref bitcast is not supported yet")

    @self.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct((8,), jnp.uint32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref.bitcast(o_ref.dtype)[...]

    x = jnp.arange(8, dtype=jnp.float32)
    np.testing.assert_array_equal(kernel(x), x.view(np.uint32))

  @parameterized.product(
      pack_format=[*plsc.PackFormat],
      dtype=[jnp.float32, jnp.int32],
  )
  def test_pack_unpack(self, pack_format, dtype):
    shape = (8,)

    @self.vector_subcore_kernel(
        out_shape=(jax.ShapeDtypeStruct((8,), dtype),) * 2
    )
    def kernel(a_ref, b_ref, oa_ref, ob_ref):
      ab = plsc.pack(a_ref[...], b_ref[...], format=pack_format)
      oa_ref[...], ob_ref[...] = plsc.unpack(ab, format=pack_format)

    a = jnp.arange(math.prod(shape), dtype=dtype).reshape(shape)
    b = -a
    out_a, out_b = kernel(a, b)
    np.testing.assert_array_equal(out_a, a)
    np.testing.assert_array_equal(out_b, b)

  @parameterized.parameters(jnp.int32, jnp.float32)
  def test_scan_count(self, dtype):
    shape = [8]

    @self.vector_subcore_kernel(
        out_shape=(
            jax.ShapeDtypeStruct(shape, jnp.int32),
            jax.ShapeDtypeStruct(shape, jnp.int32),
        ),
    )
    def kernel(x_ref, counts_ref, mask_ref):
      counts_ref[...], mask = plsc.scan_count(x_ref[...])
      mask_ref[...] = mask.astype(jnp.int32)

    key = jax.random.key(42)
    x = jax.random.randint(key, shape, 0, 10, dtype=jnp.int32).astype(dtype)
    counts, mask = kernel(x)
    expected_counts = []
    expected_mask = []
    c = collections.Counter()
    for item in x:
      item = int(item)
      c[item] += 1
      expected_counts.append(c[item])
    for item in x:
      item = int(item)
      c[item] -= 1
      expected_mask.append(c[item] == 0)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(mask, expected_mask)

  def test_population_count(self):
    key = jax.random.key(42)
    x = jax.random.randint(key, [8], 0, 100)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      mask = x_ref[...] < 50
      # TODO: b/434208146 - Test with reduce!=1 when we support v6e packed masks
      o_ref[...] = plsc.all_reduce_population_count(mask)

    np.testing.assert_array_equal(
        kernel(x), np.broadcast_to(np.count_nonzero(x < 50), x.shape)
    )

  def test_iota(self):
    key = jax.random.key(42)
    x = jax.random.randint(key, [8], 0, 100)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.arange(8) + x_ref[...]

    np.testing.assert_array_equal(
        kernel(x), x + np.arange(8)
    )

  def test_write_to_transformed_ref(self):
    x = jnp.arange(16)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      plsc.store_compressed(
          o_ref.at[pl.ds(5, 8)], x_ref[pl.ds(2, 8)], mask=jnp.ones(8, jnp.bool),
      )
    np.testing.assert_array_equal(kernel(x)[5:13], x[2:10])

  def test_load_transformed_ref(self):
    x = jnp.arange(16)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      o_ref[pl.ds(5, 8)] = plsc.load_expanded(
          x_ref.at[pl.ds(2, 8)], mask=jnp.arange(8) % 2 == 0)
    np.testing.assert_array_equal(kernel(x)[5:13:2], x[2:6])

  def test_scalar_load_store(self):

    @self.vector_subcore_kernel(
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8,), jnp.int32),
        scratch_shapes=(pltpu.VMEM((1,), jnp.int32),),
    )
    def kernel(x_ref, o_ref, tmp_ref):
      pltpu.sync_copy(x_ref, tmp_ref)
      o_ref[...] = lax.broadcast(tmp_ref[0], o_ref.shape)

    np.testing.assert_array_equal(
        kernel(jnp.ones((1,), jnp.int32)), jnp.ones((8,), jnp.int32)
    )

  def test_scalar_load_hbm(self):

    @self.vector_subcore_kernel(
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8,), jnp.int32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = lax.broadcast(x_ref[0], o_ref.shape)

    with self.assertRaisesRegex(
        NotImplementedError, "Get does not support loading from HBM"
    ):
      _ = kernel(jnp.ones((1,), jnp.int32))

  @parameterized.named_parameters(
      ("mixed", [0, 0, 1, 0, 1, 0, 0, 0], 2),
      ("all_zero", [0, 0, 0, 0, 0, 0, 0, 0], 8),
      ("all_one", [1, 1, 1, 1, 1, 1, 1, 1], 0))
  def test_ffs(self, data, expected):
    x = jnp.array(data)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      mask = x_ref[...] == 1
      # TODO: b/434208146 - Test with reduce!=1 when we support v6e packed masks
      o_ref[...] = plsc.all_reduce_ffs(mask)

    np.testing.assert_array_equal(kernel(x), np.broadcast_to(expected, x.shape))

  def test_run_scoped(self):
    x = jnp.arange(8)

    @self.vector_subcore_kernel(
        out_shape=x, out_specs=pl.BlockSpec(memory_space=pltpu.HBM)
    )
    def kernel(x_ref, o_hbm_ref):
      pltpu.sync_copy(x_ref, o_hbm_ref)

    np.testing.assert_array_equal(kernel(x), x)

  def test_run_scoped_with_tiling(self):
    x = jnp.arange(2 * 8).reshape(-1, 8)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      def scoped_kernel(scratch_ref):
        scratch_ref[...] = x_ref[...]
        o_ref[...] = scratch_ref[...]

      pl.run_scoped(
          scoped_kernel,
          plsc.MemoryRef(
              x.shape, x_ref.dtype, memory_space=pltpu.VMEM, tiling=[(1, 8)]
          ),
      )

    # Just make sure it compiles. The unrolling logic in the SC compiler
    # does not yet handle tiled layouts properly, so the result is wrong.
    _ = kernel(x)

  @parameterized.product(sizes=[[1, 1], [2, 2], [1, 1, 1, 1]])
  def test_split_concatenate(self, sizes):

    shape = (sum(sizes), 8)
    x = jnp.arange(math.prod(shape)).reshape(-1, 8)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      chunks = lax.split(x_ref[...], sizes, 0)
      o_ref[...] = lax.concatenate(chunks, 0)

    np.testing.assert_array_equal(kernel(x), x)

  def test_scratch(self):
    x = jnp.arange(8)

    @self.vector_subcore_kernel(
        out_shape=x,
        scratch_shapes=(pltpu.VMEM([8], jnp.float32),),
    )
    def kernel(x_ref, o_ref, scratch_ref):
      scratch_ref[...] = x_ref[...].astype(jnp.float32)
      o_ref[...] = scratch_ref[...].astype(x.dtype)

    np.testing.assert_array_equal(kernel(x), x)

  def test_implicit_padding_unsupported(self):
    x = jnp.arange(8, dtype=jnp.int32).reshape((8, 1))

    @self.vector_subcore_kernel(out_shape=x, in_specs=(pl.BlockSpec((8, 1)),))
    def kernel(*args):
      del args  # Unused.

    with self.assertRaisesRegex(ValueError, "must be a multiple of 8"):
      kernel(x)

  def test_subcore_parallel(self):
    self.skip_if_tc_tiling()
    num_subcores = 16

    @self.kernel(
        out_shape=jax.ShapeDtypeStruct(
            shape=(num_subcores, 8), dtype=jnp.int32
        ),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=1
        ),
    )
    def kernel(x_ref, o_ref):
      # This is a smoke test, since it does not in fact check that the kernel
      # is executed in parallel over the subcores.
      subcore_id = lax.axis_index("subcore")
      pltpu.sync_copy(x_ref.at[subcore_id], o_ref.at[subcore_id])

    x = jnp.arange(num_subcores * 8, dtype=jnp.int32).reshape(-1, 8)
    np.testing.assert_array_equal(kernel(x), x)

  def test_smem_vmem_store_literals(self):
    self.skip_if_tc_tiling()
    num_subcores = 16

    @self.kernel(
        out_shape=jax.ShapeDtypeStruct(
            shape=(num_subcores, 8), dtype=jnp.float32
        ),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=1
        ),
        scratch_shapes=(pltpu.SMEM([1], jnp.float32),
                        pltpu.VMEM([8], jnp.float32)),
    )
    def kernel(x_ref, o_ref, scratch_scalar_ref, scratch_vec_ref):
      subcore_id = lax.axis_index("subcore")
      scratch_scalar_ref[0] = 7.
      pltpu.sync_copy(x_ref.at[subcore_id], scratch_vec_ref)
      scratch_vec_ref[...] = jnp.where(
          subcore_id < 3, scratch_scalar_ref[0], scratch_vec_ref[...])
      pltpu.sync_copy(scratch_vec_ref, o_ref.at[subcore_id])

    x = jnp.arange(num_subcores * 8, dtype=jnp.float32).reshape(-1, 8)
    expected = jnp.where(jnp.arange(num_subcores)[:, jnp.newaxis] < 3,
                         jnp.full((num_subcores, 8), 7.),
                         x)
    np.testing.assert_array_equal(kernel(x), expected)

  @parameterized.named_parameters(
      ("barrier", lambda _: plsc.subcore_barrier()),
      ("debug_print", lambda vec: pl.debug_print('test', vec)),
  )
  def test_effect_discharge(self, effectful_op):
    x = jnp.arange(self.sc_info.num_lanes)
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=1
    )
    def stateful(refs):
      def body(x_ref, o_ref):
        def with_scratch(scratch_ref):
          pltpu.sync_copy(x_ref, scratch_ref)
          scratch_ref[...] = scratch_ref[...] + 1
          effectful_op(scratch_ref[...])
          pltpu.sync_copy(scratch_ref, o_ref)
        pl.run_scoped(with_scratch, pltpu.VMEM(x.shape, x.dtype))
      pl.core_map(mesh)(lambda: body(*refs))

    _, out = jax.jit(state_discharge.run_state(stateful))(
        (x, jnp.empty_like(x)))
    np.testing.assert_array_equal(out, x + 1)

  def test_parallel_loop_effects(self):
    chunk_size = 8

    @self.kernel(
        out_shape=(),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=1
        ),
        scratch_shapes=(pltpu.VMEM((chunk_size,), jnp.uint32),) * 3,
    )
    def _kernel(a_ref, b_ref, c_ref):
      @pl.loop(0, 4)
      def outer(i):
        const = jnp.array(0, jnp.uint32)

        @plsc.parallel_loop(0, chunk_size)
        def body(_):
          x = a_ref[...] >> i.astype(jnp.uint32)
          plsc.store_compressed(c_ref.at[...], b_ref[...], mask=x > const)

    _kernel()

  def test_reshape(self):
    shape = (8,)
    dtype = jnp.int32

    @self.vector_subcore_kernel(out_shape=jax.ShapeDtypeStruct(shape, dtype))
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...].reshape(2, 4).reshape(8)

    x = jnp.arange(math.prod(shape), dtype=dtype).reshape(shape)
    np.testing.assert_array_equal(kernel(x), x)

  @parameterized.product(dtype=[jnp.int32, jnp.float32])
  def test_cumsum(self, dtype):
    x = jnp.arange(self.sc_info.num_lanes, dtype=dtype)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.cumsum(x_ref[...])

    np.testing.assert_array_equal(kernel(x), np.cumsum(x))

  @parameterized.product(dtype=[jnp.int32, jnp.float32], op=[jnp.sum, jnp.max])
  def test_reductions(self, dtype, op):
    x = jnp.arange(self.sc_info.num_lanes, dtype=dtype)
    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.full(o_ref.shape, op(x_ref[...]))
    np.testing.assert_array_equal(kernel(x)[0], op(x))

  @parameterized.product(dtype=[jnp.int32, jnp.float32])
  def test_cumsum_2d_not_supported(self, dtype):
    x = jnp.arange(self.sc_info.num_lanes, dtype=dtype)

    with self.assertRaisesRegex(NotImplementedError, r"must be rank 1"):
      @self.vector_subcore_kernel(out_shape=x)
      def kernel(x_ref, o_ref):
        o_ref[...] = jnp.cumsum(x_ref[...].reshape(4, 2), axis=0).reshape(-1)

      kernel(x)

  @parameterized.product(dtype=[jnp.int32, jnp.float32])
  def test_masked_cumsum(self, dtype):
    x = jnp.arange(self.sc_info.num_lanes, dtype=dtype)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      o_ref[...] = plsc.cumsum(x_ref[...], mask=(x_ref[...] % 2) == 1)

    np.testing.assert_array_equal(kernel(x), np.cumsum(x * (x % 2)))

  @parameterized.product(dtype=[jnp.int32, jnp.float32])
  def test_masked_cummax(self, dtype):
    x = np.arange(self.sc_info.num_lanes, dtype=dtype)
    np.random.shuffle(x)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      o_ref[...] = plsc.cummax(x_ref[...], mask=(x_ref[...] % 2) == 1)

    row = np.arange(self.sc_info.num_lanes)[:, np.newaxis]
    col = np.arange(self.sc_info.num_lanes)[np.newaxis, :]
    mask = x % 2
    expected = (x * mask * (col <= row)).max(axis=1)
    has_valid_value_so_far = np.cumsum(mask) > 0
    expected = np.where(has_valid_value_so_far, expected, x)
    np.testing.assert_array_equal(kernel(x), expected)

  def test_parallel_loop_with_carry(self):
    chunk_size = self.sc_info.num_lanes
    nchunks = 4
    per_step_increment = 10
    sentinel_multiplier = 1000
    x = jnp.arange(16 * chunk_size * nchunks, dtype=np.int32)

    @self.vector_subcore_kernel(
        out_shape=x,
        grid=(16,),
        in_specs=[pl.BlockSpec([chunk_size * nchunks], lambda i: (i,))],
        out_specs=pl.BlockSpec([chunk_size * nchunks], lambda i: (i,)),
    )
    def kernel(x_ref, o_ref):
      @pl.when(pl.program_id(0) < 16)
      def _():
        init = (jnp.zeros([], x_ref.dtype),  # scalar
                jnp.zeros([chunk_size], x_ref.dtype),  # vector
               )
        def for_each_chunk(i, carry):
          incr, running_sum = carry
          incr += per_step_increment
          o_ref[pl.ds(i, chunk_size)] = x_ref[pl.ds(i, chunk_size)] + incr
          return incr, running_sum + x_ref[pl.ds(i, chunk_size)]
        result = plsc.parallel_loop(0, x_ref.shape[0], chunk_size, carry=init)(
            for_each_chunk)
        o_ref[pl.ds(0, chunk_size)] = jnp.where(
            jnp.arange(chunk_size) == 0,
            result[0] * sentinel_multiplier,
            result[1])

    output = kernel(x)
    expected = np.array(x).reshape(16, nchunks, chunk_size)
    # Check that the increment was properly applied.
    expected += 10 * np.arange(1, 5)[:, None]
    # Check the final carry values:
    # - Scalar in 0th position.
    expected[:, 0, 0] = sentinel_multiplier * per_step_increment * nchunks
    # - Vector in 1:chunk_size-1 positions.
    expected[:, 0, 1:] = x.reshape(16, nchunks, chunk_size).sum(1)[:, 1:]
    np.testing.assert_array_equal(output, expected.reshape(-1))

  @parameterized.parameters(
      (lambda x_ref: x_ref, r"may not be.*Ref\{"),
      (lambda x_ref: x_ref.at[pl.ds(0, 8)], r"TransformedRef.*not a valid"),
  )
  def test_parallel_loop_disallows_ref_carries(self, carry_fn, expected_regex):
    x = jnp.arange(64, dtype=jnp.int32)

    with self.assertRaisesRegex(TypeError, expected_regex):
      @self.vector_subcore_kernel(out_shape=x)
      def kernel(x_ref, o_ref):
        @plsc.parallel_loop(0, 1, carry=carry_fn(x_ref))
        def _(i, carry):
          del i  # Unused.
          x_ref[...] = o_ref[...]
          return carry

      kernel(x)

  def test_parallel_loop_wrong_carry_return(self):
    x = jnp.arange(64, dtype=jnp.int32)

    with self.assertRaisesRegex(ValueError, "should have same structure"):
      @self.vector_subcore_kernel(out_shape=x)
      def kernel(x_ref, o_ref):
        init = dict(x=jnp.zeros([]), y=jnp.ones([8]))
        @plsc.parallel_loop(0, 1, carry=init)
        def _(i, carry):
          del i  # Unused.
          x_ref[...] = o_ref[...]
          return carry["x"]

      kernel(x)

  def test_squeezed_blockspec_error_message(self):
    shape = (16, 8, 32)
    spec_shape = (pl.squeezed, 8, 32)
    x = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(*shape)

    @self.vector_subcore_kernel(
        out_shape=x,
        grid=16,
        in_specs=[pl.BlockSpec(spec_shape, lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec(spec_shape, lambda i: (i, 0, 0)),
    )
    def kernel(x_ref, o_ref):
      del x_ref, o_ref  # Unused.

    with self.assertRaisesRegex(
        NotImplementedError, r"Unsupported block dimension type.*Squeezed"):
      kernel(x)

  def test_multiple_of(self):
    x = jnp.arange(16)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      @pl.loop(0, 16, step=8)
      def _(i):
        i = pl.multiple_of(i, 8)
        o_ref[pl.ds(i, 8)] = x_ref[pl.ds(i, 8)] + 1

    np.testing.assert_array_equal(kernel(x), x + 1)

  def test_barrier_via_mesh(self):
    self.skip_if_tc_tiling()
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=1
    )
    vec_dim = self.sc_info.num_lanes
    @self.kernel(
        out_shape=jax.ShapeDtypeStruct(
            shape=(mesh.num_subcores, vec_dim), dtype=jnp.uint32
        ),
        mesh=mesh,
        scratch_shapes=[pltpu.VMEM((mesh.num_subcores, vec_dim), jnp.uint32)],
    )
    def kernel(o_ref, vmem_ref):
      subcore_id = lax.axis_index("subcore")
      @pl.loop(0, 2 * subcore_id + 1)
      def _(i):
        vmem_ref[subcore_id] = jnp.full(vec_dim, i, dtype=jnp.uint32)
        pltpu.sync_copy(vmem_ref.at[subcore_id], o_ref.at[subcore_id])
      plsc.subcore_barrier()
      pltpu.sync_copy(o_ref.at[(subcore_id + 1) % mesh.num_subcores],
                      vmem_ref.at[subcore_id])
      pltpu.sync_copy(vmem_ref.at[subcore_id], o_ref.at[subcore_id])
    expected = 2 * jnp.roll(jnp.arange(mesh.num_subcores), -1)
    expected = jnp.broadcast_to(expected[:, None], (mesh.num_subcores, vec_dim))
    np.testing.assert_array_equal(kernel(), expected)

  def test_barrier_via_pallas_call(self):
    self.skip_if_tc_tiling()

    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=1
    )
    vec_dim = self.sc_info.num_lanes
    @functools.partial(
        pl.pallas_call,
        grid=16,
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
            dimension_semantics=["subcore_parallel"],
            use_tc_tiling_on_sc=self.USE_TC_TILING,
        ),
        out_shape=jax.ShapeDtypeStruct(
            shape=(mesh.num_subcores, vec_dim), dtype=jnp.uint32
        ),
        out_specs=pl.BlockSpec((1, vec_dim), lambda i: (i, 0)),
        scratch_shapes=(
            pltpu.VMEM_SHARED((mesh.num_subcores, vec_dim), jnp.uint32),
            pltpu.VMEM((vec_dim,), jnp.uint32),
        ),
    )
    def kernel(o_ref, shared_ref, vmem_ref):
      subcore_id = pl.program_id(0)
      @pl.loop(0, 10 * subcore_id + 1)
      def _(i):
        vmem_ref[:] = jnp.full(vec_dim, i, dtype=jnp.uint32)
        pltpu.sync_copy(vmem_ref, shared_ref.at[subcore_id])
      plsc.subcore_barrier()
      pltpu.sync_copy(shared_ref.at[(subcore_id + 1) % mesh.num_subcores],
                      o_ref.at[0])
    expected = 10 * jnp.roll(jnp.arange(mesh.num_subcores), -1)
    expected = jnp.broadcast_to(expected[:, None], (mesh.num_subcores, vec_dim))
    np.testing.assert_array_equal(kernel(), expected)

  @parameterized.parameters(jnp.int32, jnp.float32)
  def test_gather_add(self, dtype):
    """Gather from HBM at indices added to contiguous VMEM."""
    self.skip_if_tc_tiling()
    shape = (16, 64, 32)
    x = jnp.arange(np.prod(shape), dtype=dtype).reshape(*shape)

    @self.kernel(
        out_shape=x[:, :8],
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=1
        ),
        scratch_shapes=[
            pltpu.VMEM([8], jnp.int32),
            pltpu.VMEM([8, 32], dtype),
            pltpu.SemaphoreType.DMA,
        ],
    )
    def kernel(x_ref, indices_ref, o_ref, indices_vmem, scratch_ref, sem):
      subcore_id = lax.axis_index("subcore")
      pltpu.sync_copy(indices_ref, indices_vmem)
      # Initialize scratch space.
      pltpu.sync_copy(x_ref.at[subcore_id, pl.ds(0, 8)], scratch_ref)
      # Gather-add selected indices to scratch.
      pltpu.async_copy(
          # TODO: Can't mix array and ref indexers .at[subcore_id, indices_vmem]
          x_ref.at[subcore_id].at[indices_vmem],
          scratch_ref,
          sem,
          add=True,
      ).wait()
      pltpu.sync_copy(scratch_ref, o_ref.at[subcore_id])

    indices = jnp.arange(8) * 8
    np.testing.assert_array_equal(
        kernel(x, indices), x[:, :8] + x[:, indices])

  @parameterized.parameters(jnp.int32, jnp.float32)
  def test_scatter_add(self, dtype):
    """Scatter from contiguous VMEM added to VMEM_SHARED at indices."""
    self.skip_if_tc_tiling()
    shape = (16, 32)
    x = jnp.arange(np.prod(shape), dtype=dtype).reshape(*shape)

    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=1
    )
    @functools.partial(
        pl.pallas_call,
        grid=mesh.num_subcores,
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
            dimension_semantics=["subcore_parallel"],
            use_tc_tiling_on_sc=self.USE_TC_TILING,
        ),
        out_shape=jax.ShapeDtypeStruct(shape[1:], dtype),
        out_specs=pl.BlockSpec(
            shape[1:], lambda i: (0,), memory_space=pltpu.HBM
        ),
        in_specs=[
            pl.BlockSpec(shape, lambda *_: (0, 0), memory_space=pltpu.HBM),
            pl.BlockSpec(shape[1:], lambda _: (0,)),
        ],
        scratch_shapes=[
            pltpu.VMEM_SHARED(shape[1:], dtype),
            pltpu.VMEM(shape[1:], dtype),
            pltpu.SemaphoreType.DMA,
        ],
    )
    def kernel(x_ref, indices_ref, o_ref,
               shared_scratch_ref, scratch_ref, sem):
      subcore_id = pl.program_id(0)
      pltpu.sync_copy(x_ref.at[subcore_id], scratch_ref)
      # Subcore 0 to init shared scratch.
      @pl.when(subcore_id == 0)
      def _():
        pltpu.sync_copy(scratch_ref, shared_scratch_ref)
      plsc.subcore_barrier()
      # All cores to add their slice to shared scratch.
      pltpu.async_copy(
          scratch_ref,
          shared_scratch_ref.at[indices_ref],
          sem,
          add=True,
      ).wait()
      plsc.subcore_barrier()
      # Subcore 0 to copy shared scratch to output.
      @pl.when(subcore_id == 0)
      def _():
        pltpu.sync_copy(shared_scratch_ref, scratch_ref)
        pltpu.sync_copy(scratch_ref, o_ref)

    indices = 31 - jnp.arange(32)
    np.testing.assert_array_equal(kernel(x, indices), x[0] + x.sum(0)[::-1])

  def test_shared_scratch(self):
    self.skip_if_tc_tiling()
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=1
    )
    shape = (mesh.num_subcores, 8, 8)
    x = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(*shape)

    @self.kernel(out_shape=x, mesh=mesh)
    def kernel(x_ref, o_ref):
      subcore_id = lax.axis_index("subcore")
      shared_scratch_ref = pl.get_global(
          pltpu.VMEM_SHARED(shape[1:], jnp.int32))
      @pl.when(subcore_id == 0)
      def _():
        shared_scratch_ref2 = pl.get_global(pltpu.VMEM_SHARED(shape, jnp.int32))
        pltpu.sync_copy(
            x_ref.at[subcore_id], shared_scratch_ref2.at[subcore_id])
        pltpu.sync_copy(x_ref.at[subcore_id], shared_scratch_ref)
        pltpu.sync_copy(shared_scratch_ref, o_ref.at[subcore_id])

    np.testing.assert_array_equal(kernel(x)[0], x[0])

  def test_copy_in_shard_map(self):
    self.skip_if_tc_tiling()
    num_devices = len(jax.devices())
    mesh = jtu.create_mesh((num_devices,), ("x",))

    rng = np.random.default_rng(0)
    x = rng.integers(512, size=(num_devices * 1024, 16), dtype=np.int32)

    # The test ensures that JAX-level memory space for ``x`` is not propagated
    # into Pallas, since Pallas cannot use it.
    x = jax.device_put(x, jax.sharding.NamedSharding(mesh, jax.P("x", None)))
    self.assertEqual(jax.typeof(x).memory_space, jax.memory.Space.Device)

    @functools.partial(
        jax.shard_map,
        in_specs=(jax.P("x", None),),
        out_specs=jax.P("x", None),
        mesh=mesh,
        check_vma=True,
    )
    def f(x):
      @self.kernel(
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype, vma={"x"}),
          mesh=plsc.VectorSubcoreMesh(
              core_axis_name="core", subcore_axis_name="subcore", num_cores=1
          ),
          scratch_shapes=(pltpu.VMEM(x.shape, x.dtype),),
      )
      def kernel(in_ref, o_ref, scratch_ref):
        pltpu.sync_copy(in_ref, scratch_ref)
        pltpu.sync_copy(scratch_ref, o_ref)

      return kernel(x)

    np.testing.assert_array_equal(f(x), x)

  @parameterized.named_parameters(
      ("exp", jnp.exp), ("neg", lambda x: -x), ("abs", jnp.abs)
  )
  def test_unary_ops(self, op):
    x = jnp.arange(8, dtype=jnp.float32)

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      o_ref[...] = op(x_ref[...])

    np.testing.assert_array_equal(kernel(x), op(x))

  @parameterized.product(dtype=[np.int32, np.float32])
  def test_vector_gather(self, dtype):
    vec_dim = self.sc_info.num_lanes
    x = np.arange(vec_dim, dtype=dtype)
    indices = np.random.randint(0, vec_dim, size=vec_dim, dtype=np.int32)
    indices[[0, -2]] = 2  # Verify non-unique works.
    indices[1] = -2  # Verify negative indices work.

    @self.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, out_ref):
      out_ref[...] = x_ref[...][indices_ref[...]]

    np.testing.assert_array_equal(kernel(x, indices), x[indices])

  @parameterized.product(
      keys_dtype=[np.int32, np.float32],
      values_dtype=[np.int32, np.float32],
      use_mask=[False, True],
      descending=[False, True],
  )
  def test_sort_key_val(self, keys_dtype, values_dtype, use_mask, descending):
    vec_dim = self.sc_info.num_lanes
    keys = np.arange(vec_dim, dtype=keys_dtype)
    np.random.shuffle(keys)
    keys[3] = keys[1]  # Verify sort stability.
    values = np.arange(vec_dim, dtype=values_dtype)
    np.random.shuffle(values)
    mask = np.random.choice([True, False], size=vec_dim) if use_mask else None
    maybe_mask_arg = (mask.astype(jnp.int32),) if use_mask else ()

    @self.vector_subcore_kernel(out_shape=(keys, values, *maybe_mask_arg))
    def kernel(*args):
      if use_mask:
        mask_ref, *args, o_mask_ref = args
        mask = mask_ref[...].astype(jnp.bool)
      else:
        mask, o_mask_ref = None, None
      keys_ref, values_ref, o_keys_ref, o_vals_ref = args
      o_keys_ref[...], o_vals_ref[...], *maybe_out_mask = plsc.sort_key_val(
          keys_ref[...], values_ref[...], mask=mask, descending=descending)
      if use_mask:
        [out_mask] = maybe_out_mask
        o_mask_ref[...] = out_mask.astype(jnp.int32)

    out_keys, out_values, *maybe_out_mask = kernel(
        *maybe_mask_arg, keys, values)

    keys_arg = keys
    if descending:
      keys_arg = -keys_arg
    if use_mask:
      keys_arg = jnp.where(mask, keys_arg, 100)
    _, gt_keys = jax.lax.sort_key_val(keys_arg, keys)
    _, gt_values = jax.lax.sort_key_val(keys_arg, values)
    if use_mask:
      [out_mask] = maybe_out_mask
      gt_out_mask = jnp.arange(vec_dim) < mask.sum()
      np.testing.assert_array_equal(out_mask, gt_out_mask.astype(jnp.int32))
    np.testing.assert_array_equal(out_keys, gt_keys)
    np.testing.assert_array_equal(out_values, gt_values)

  @parameterized.product(dtype=[np.int32, np.float32])
  def test_rev_and_sort_desc(self, dtype):
    vec_dim = self.sc_info.num_lanes
    keys = np.arange(vec_dim, dtype=dtype)
    np.random.shuffle(keys)

    @self.vector_subcore_kernel(out_shape=(keys, keys))
    def kernel(x_ref, o1_ref, o2_ref):
      o1_ref[...] = jnp.sort(x_ref[...], descending=True)
      o2_ref[...] = jnp.flip(x_ref[...], axis=-1)

    sorted_desc, reversed_keys = kernel(keys)  # pylint: disable=unpacking-non-sequence
    np.testing.assert_array_equal(
        sorted_desc, jnp.arange(vec_dim, dtype=dtype)[::-1])
    np.testing.assert_array_equal(reversed_keys, keys[::-1])

  @parameterized.product(
      keys_dtype=[np.int32, np.float32],
      values_dtypes=[(), (np.int32,), (np.float32, np.int32)],
  )
  def test_sort(self, keys_dtype, values_dtypes):
    vec_dim = self.sc_info.num_lanes
    keys = np.arange(vec_dim, dtype=keys_dtype)
    np.random.shuffle(keys)
    values = [np.arange(vec_dim, dtype=dtype) for dtype in values_dtypes]
    _ = [np.random.shuffle(v) for v in values]

    @self.vector_subcore_kernel(out_shape=(keys, *values))
    def kernel(*args):
      keys_ref, *values_refs = args[: len(args) // 2]
      keys_out, *all_values_out = jax.lax.sort(
          (keys_ref[...], *(ref[...] for ref in values_refs))
      )
      keys_out_ref, *values_out_refs = args[len(args) // 2 :]
      keys_out_ref[...] = keys_out
      for values_out_ref, values_out in zip(
          values_out_refs, all_values_out, strict=True
      ):
        values_out_ref[...] = values_out

    perm = np.argsort(keys)
    keys_result, *values_results = kernel(keys, *values)
    np.testing.assert_array_equal(keys_result, keys[perm])
    for values_result, values_in in zip(values_results, values, strict=True):
      np.testing.assert_array_equal(values_result, values_in[perm])


class VectorSubcoreTestWithTCTiling(VectorSubcoreTest):
  USE_TC_TILING = True


class ScalarSubcoreTest(PallasSCTest):

  def test_copy(self):
    x = jnp.arange(16)

    @self.kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(
            axis_name="core", num_cores=self.sc_info.num_cores
        ),
    )
    def kernel(x_ref, o_ref):
      lax.cond(
          lax.axis_index("core") == lax.axis_size("core") - 1,
          lambda: pltpu.sync_copy(x_ref, o_ref),
          lambda: None,
      )

    np.testing.assert_array_equal(kernel(x), x)

  def test_sliced_copy(self):
    self.skip_if_tc_tiling()
    x = jnp.arange(self.sc_info.num_cores * 8).reshape(
        self.sc_info.num_cores, -1
    )

    @self.kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(
            axis_name="core", num_cores=self.sc_info.num_cores
        ),
    )
    def kernel(x_ref, o_ref):
      @functools.partial(pl.run_scoped, sems=pltpu.SemaphoreType.DMA(4))
      def _(sems):
        core_id = lax.axis_index("core")
        pltpu.async_copy(
            x_ref.at[core_id], o_ref.at[core_id], sems.at[core_id]
        ).wait()

    np.testing.assert_array_equal(kernel(x), x)

  def test_scalar_load_store(self):
    x = jnp.arange(8)

    @self.kernel(
        out_shape=x, mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)
    )
    def kernel(x_ref, o_ref):
      @functools.partial(
          pl.run_scoped,
          tmp_ref=pltpu.SMEM(x.shape, x.dtype),
          sem=pltpu.SemaphoreType.DMA,
      )
      def _(tmp_ref, sem):
        pltpu.async_copy(x_ref, tmp_ref, sem).wait()

        @pl.loop(1, *x.shape)
        def _(i):
          tmp_ref[i] += tmp_ref[i - 1]

        pltpu.async_copy(tmp_ref, o_ref, sem).wait()

    np.testing.assert_array_equal(kernel(x), jnp.cumsum(x))

  @parameterized.product(
      first_parallel=[False, True], second_parallel=[False, True]
  )
  def test_parallel_loop(self, first_parallel, second_parallel):
    self.skip_if_tc_tiling()
    x = jnp.arange(8*8).reshape(8, 8)

    loop = lambda start, end, parallel, **kwargs: (
        plsc.parallel_loop(start, end, **kwargs)
        if parallel
        else pl.loop(start, end, **kwargs)
    )

    @self.kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
        scratch_shapes=(
            pltpu.SMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ),
    )
    def kernel(x_ref, o_ref, tmp_ref, sem):
      pltpu.async_copy(x_ref, tmp_ref, sem).wait()

      @loop(0, tmp_ref.shape[0], first_parallel)
      def _(i):
        @loop(0, tmp_ref.shape[1], second_parallel, unroll=tmp_ref.shape[1])
        def _(j):
          tmp_ref[i, j] += 1

      pltpu.async_copy(tmp_ref, o_ref, sem).wait()

    np.testing.assert_array_equal(kernel(x), x + 1)

  def test_parallel_loop_with_carry(self):
    self.skip_if_tc_tiling()
    x = jnp.arange(8*8).reshape(8, 8)

    @self.kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
        scratch_shapes=(
            pltpu.SMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ),
    )
    def kernel(x_ref, o_ref, tmp_ref, sem):
      pltpu.async_copy(x_ref, tmp_ref, sem).wait()

      @plsc.parallel_loop(0, tmp_ref.shape[0], carry=jnp.zeros([], x.dtype))
      def _(i, carry):
        carry += 1
        @plsc.parallel_loop(0, tmp_ref.shape[1], unroll=2)
        def _(j):
          tmp_ref[i, j] += carry
        return carry

      pltpu.async_copy(tmp_ref, o_ref, sem).wait()

    np.testing.assert_array_equal(kernel(x), x + jnp.arange(1, 9)[:, None])


class ScalarSubcoreTestWithTCTiling(ScalarSubcoreTest):
  USE_TC_TILING = True


class PipelineTest(PallasSCTest):

  def test_basic(self):
    self.skip_if_tc_tiling()
    num_steps = 16
    x = jnp.arange(num_steps * 8).reshape(-1, 8)

    @self.vector_subcore_kernel(
        out_shape=x,
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )
    def kernel(x_hbm_ref, o_hbm_ref):
      @functools.partial(
          pltpu.emit_pipeline,
          grid=(num_steps // 2,),
          in_specs=pl.BlockSpec((2, 8), lambda i: (i, 0)),
          out_specs=pl.BlockSpec((2, 8), lambda i: (i, 0)),
      )
      def pipeline(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1

      pipeline(x_hbm_ref, o_hbm_ref)

    np.testing.assert_array_equal(kernel(x), x + 1)

  def test_explicit_sc_tiling_1d(self):
    self.skip_if_tc_tiling("The test uses SC tiling.")

    num_steps = 4
    x = jnp.arange(num_steps * 8)

    @self.vector_subcore_kernel(
        out_shape=x,
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )
    def kernel(x_hbm_ref, o_hbm_ref):
      spec = plsc.BlockSpec((8,), lambda i: (i,))

      @functools.partial(
          pltpu.emit_pipeline,
          grid=(num_steps,),
          in_specs=spec,
          out_specs=spec,
          tiling=pltpu.Tiling.SPARSE_CORE,
      )
      def pipeline(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1

      pipeline(x_hbm_ref, o_hbm_ref)

    np.testing.assert_array_equal(kernel(x), x + 1)

  def test_explicit_sc_tiling_2d(self):
    self.skip_if_tc_tiling("The test uses SC tiling.")

    num_steps = 16
    x = jnp.arange(num_steps * 8 * 128).reshape(-1, 8, 128)

    @self.vector_subcore_kernel(
        out_shape=x,
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )
    def kernel(x_hbm_ref, o_hbm_ref):
      spec = plsc.BlockSpec((pl.Squeezed(), 8, 128), lambda i: (i, 0, 0))

      @functools.partial(
          pltpu.emit_pipeline,
          grid=(num_steps,),
          in_specs=[spec],
          out_specs=[spec],
          tiling=pltpu.Tiling.SPARSE_CORE,
      )
      def pipeline(x_ref, o_ref):
        @pl.loop(0, 8)
        def _(i):
          @pl.loop(0, 128, step=8)
          def _(j):
            o_ref[i, pl.ds(j, 8)] = x_ref[i, pl.ds(j, 8)] + 1

      pipeline(x_hbm_ref, o_hbm_ref)

    np.testing.assert_array_equal(kernel(x), x + 1)


class PipelineTestWithTCTiling(PipelineTest):
  USE_TC_TILING = True


class PallasSparsecoreAsyncTest(PallasSCTest):

  def setUp(self):
    super().setUp()

  @parameterized.product(
      shape=[
          (8, 128),
          (8, 256),
          (8, 512),
          (8, 1024),
          (16, 128),
          (16, 256),
          (16, 512),
          (16, 1024),
          # TODO(sharadmv): These shapes fail right now.
          # (64, 8),
      ],
      dtype=[jnp.int32, jnp.float32, jnp.bfloat16],
  )
  def test_basic_async_kernel(self, shape, dtype):
    x = jnp.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)

    @jax.jit
    def foo(x):
      sc_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)

      sem = pl.pallas_call(
          lambda _: None,
          out_shape=pltpu.SemaphoreType.DMA(()),
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=["core_parallel"],
              kernel_type=pltpu.KernelType.SC_SCALAR_SUBCORE,
              use_tc_tiling_on_sc=self.USE_TC_TILING,
          ),
      )()

      sem_ref = jax.new_ref(sem, memory_space=pltpu.SEMAPHORE)
      y_ref = pl.empty_ref_like(pltpu.HBM(x.shape, x.dtype))
      x_ref = jax.new_ref(x)

      run_kernel = pl.core_map(mesh=sc_mesh)

      @run_kernel
      def _():
        pltpu.make_async_copy(x_ref, y_ref, sem_ref).start()

      @run_kernel
      def _():
        pltpu.make_async_copy(x_ref, y_ref, sem_ref).wait()

      return y_ref[...]

    o = jax.block_until_ready(foo(x))
    np.testing.assert_array_equal(o, x)


class PallasSparsecoreAsyncTestWithTCTiling(PallasSparsecoreAsyncTest):
  USE_TC_TILING = True


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
