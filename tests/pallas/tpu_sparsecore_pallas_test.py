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
import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic import sc_core
from jax._src.pallas.mosaic import sc_lowering
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class PallasSCTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest("SparseCore only supported on TPU v5+")

    super().setUp()


class VectorSubcoreTest(PallasSCTest):

  # Used for testing masked loads and stores below
  MASK_FNS = [lambda x: x < 4, lambda x: x >= 4, lambda x: x % 2 == 0]

  @parameterized.product(
      dtype=[jnp.int32, jnp.float32], op=[jnp.add, jnp.subtract]
  )
  def test_add_sub_one(self, dtype, op):
    x = jnp.arange(8, dtype=dtype)

    @plsc.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = op(x, 1)

    np.testing.assert_array_equal(kernel(x), op(x, 1))

  @parameterized.product(dtype=[jnp.int32, jnp.float32])
  def test_debug_print(self, dtype):
    x = jnp.arange(16, dtype=dtype)
    debug_int = 1234552
    debug_float = 12344.625

    @plsc.vector_subcore_kernel(out_shape=x)
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

  def test_add_one_block_specs(self):
    x = jnp.arange(32, dtype=jnp.int32)

    @plsc.vector_subcore_kernel(
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
      for dtype, shapes in sc_lowering._SUPPORTED_VECTOR_SHAPES.items()
      for shape in shapes
      for minor_scale in [1, 2, 4]
  ))
  def test_slicing(self, dtype, out_shape, minor_scale):
    if jtu.is_device_tpu(6, "e"):
      # TODO(b/433704850): Remove this once the bug is fixed.
      self.skipTest("Crashes")

    crashing = {
        "int16": [(2, 8), (2, 16)],
        "uint16": [(2, 8), (2, 16)],
        "float16": [(2, 8), (2, 16), (4, 8), (4, 16)],
        "bfloat16": [(2, 8), (2, 16), (4, 8), (4, 16)],
        "int8": [(4, 8), (4, 16)],
        "uint8": [(4, 8), (4, 16)],
    }
    if out_shape in crashing.get(dtype.name, []):
      self.skipTest("Crashes")

    out_minor = out_shape[-1]
    in_minor = out_minor * minor_scale
    in_shape = (*out_shape[:-1], in_minor)
    indices = [
        slice(i * out_minor, (i + 1) * out_minor) for i in range(minor_scale)
    ]

    @plsc.vector_subcore_kernel(
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
    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = lax.fori_loop(
          1, major_dim, lambda i, acc: acc + x_ref[i], x_ref[0]
      )

    x = jnp.arange(8 * major_dim).reshape(major_dim, 8)
    np.testing.assert_array_equal(kernel(x), x.sum(axis=0))

  def test_get_multi_index(self):
    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      for i, j in itertools.product(*map(range, x_ref.shape[:-1])):
        o_ref[...] += x_ref.at[i][j]

    x = jnp.arange(3 * 4 * 8).reshape(3, 4, 8)
    np.testing.assert_array_equal(kernel(x), x.sum(axis=(0, 1)))

  @parameterized.product(major_dim=[2, 3, 4])
  def test_swap_index(self, major_dim):
    @plsc.vector_subcore_kernel(
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
    x = jnp.arange(math.prod(shape)).reshape(shape)
    major_dim, *_ = shape
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(major_dim))

    @plsc.vector_subcore_kernel(
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

    @plsc.vector_subcore_kernel(
        out_shape=x, out_specs=pl.BlockSpec(memory_space=pltpu.HBM)
    )
    def kernel(x_ref, indices_ref, o_hbm_ref):
      @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.DMA)
      def _(sem):
        pltpu.async_copy(x_ref, o_hbm_ref.at[indices_ref[...]], sem).wait()

    np.testing.assert_array_equal(
        kernel(x, indices), jnp.empty_like(x).at[indices].set(x)
    )

  @parameterized.product(kind=["ref", "array"])
  def test_gather_1d(self, kind):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), x)

    @plsc.vector_subcore_kernel(
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

  def test_large_gather_1d(self):
    x = jnp.arange(1024)
    indices = jax.random.permutation(jax.random.key(42), x)

    @plsc.vector_subcore_kernel(
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
    x = jnp.arange(4 * 4 * 8).reshape(4, 4, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
    )
    def kernel(x_hbm_ref, indices_ref, o_ref):
      pltpu.sync_copy(x_hbm_ref.at[1, 2].at[indices_ref], o_ref)

    np.testing.assert_array_equal(kernel(x, indices), x[1, 2, indices])

  def test_implicit_gather_1d(self):
    num_steps = 4
    x = jnp.arange(num_steps * 8).reshape(num_steps, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(num_steps))

    @plsc.vector_subcore_kernel(
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

    @plsc.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      o_ref[...] = plsc.load_gather(x_ref, [indices_ref[...]])

    np.testing.assert_array_equal(kernel(x, indices), x[indices])

  def test_load_gather_2d(self):
    x = jnp.arange(8 * 8).reshape(8, -1)
    indices0 = indices1 = jax.random.permutation(
        jax.random.key(42), jnp.arange(8)
    )

    @plsc.vector_subcore_kernel(out_shape=jax.ShapeDtypeStruct((8,), x.dtype))
    def kernel(x_ref, indices0_ref, indices1_ref, o_ref):
      o_ref[...] = plsc.load_gather(
          x_ref, [indices0_ref[...], indices1_ref[...]]
      )

    np.testing.assert_array_equal(
        kernel(x, indices0, indices1), x[indices0, indices1]
    )

  def test_load_gather_with_indexing(self):
    num_steps = 4
    x = jnp.arange(num_steps * 8).reshape(num_steps, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @plsc.vector_subcore_kernel(out_shape=x)
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

    @plsc.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      o_ref[...] = plsc.load_gather(
          x_ref, [indices_ref[...]], mask=mask_fn(x_ref[...])
      )

    mask = mask_fn(x)
    np.testing.assert_array_equal(kernel(x, indices)[mask], x[indices][mask])

  def test_store_scatter(self):
    num_steps = 4
    x = jnp.arange(num_steps * 8).reshape(num_steps, 8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @plsc.vector_subcore_kernel(out_shape=x)
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

    @plsc.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, indices_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = jnp.zeros_like(o_ref)
      plsc.store_scatter(o_ref, [indices_ref[...]], x, mask=mask_fn(x))

    mask = mask_fn(x)
    np.testing.assert_array_equal(
        kernel(x, indices),
        jnp.zeros_like(x).at[indices[mask]].set(x[mask]),
    )

  @parameterized.parameters(*MASK_FNS)
  def test_addupdate_scatter(self, mask_fn):
    x = jnp.arange(8)
    indices = jax.random.permutation(jax.random.key(42), jnp.arange(8))

    @plsc.vector_subcore_kernel(out_shape=x)
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
    @plsc.vector_subcore_kernel(
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
    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      x = x_ref[...]
      plsc.store_compressed(o_ref.at[...], x, mask=mask_fn(x))

    x = jnp.arange(8)
    mask = mask_fn(x)
    np.testing.assert_array_equal(kernel(x)[: mask.sum()], x[mask])

  def test_addupdate(self):
    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.int32)
    )
    def kernel(o_ref):
      o_ref[...] = jnp.zeros_like(o_ref)
      for i in range(8):
        plsc.addupdate(o_ref.at[...], lax.broadcast(i, o_ref.shape))

    np.testing.assert_array_equal(kernel(), jnp.full(8, jnp.arange(8).sum()))

  @parameterized.parameters(*MASK_FNS)
  def test_addupdate_compressed(self, mask_fn):
    @plsc.vector_subcore_kernel(
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
    new_shape = (
        8 * jnp.dtype(dtype).itemsize // jnp.dtype(new_dtype).itemsize,
    )

    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=new_shape, dtype=new_dtype)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plsc.bitcast(x_ref[...], o_ref.dtype)

    x = jnp.arange(8, dtype=dtype)
    np.testing.assert_array_equal(kernel(x), x.view(new_dtype))

  def test_bitcast_invalid(self):
    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct(shape=[1], dtype=jnp.int32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plsc.bitcast(x_ref[...], o_ref.dtype)

    x = jnp.arange(2, dtype=jnp.int8)
    with self.assertRaisesRegex(ValueError, "is not divisible"):
      kernel(x)

  @parameterized.parameters(*plsc.PackFormat)
  def test_pack_unpack(self, format):
    shape = (8,)
    dtype = jnp.float32

    @plsc.vector_subcore_kernel(
        out_shape=(jax.ShapeDtypeStruct((8,), dtype),) * 2
    )
    def kernel(a_ref, b_ref, oa_ref, ob_ref):
      ab = plsc.pack(a_ref[...], b_ref[...], format=format)
      oa_ref[...], ob_ref[...] = plsc.unpack(ab, format=format)

    a = jnp.arange(math.prod(shape), dtype=dtype).reshape(shape)
    b = a * a
    out_a, out_b = kernel(a, b)
    np.testing.assert_array_equal(out_a, a)
    np.testing.assert_array_equal(out_b, b)

  def test_scan_count(self):
    shape = [8]

    @plsc.vector_subcore_kernel(
        out_shape=(
            jax.ShapeDtypeStruct(shape, jnp.int32),
            jax.ShapeDtypeStruct(shape, jnp.int32),
        ),
    )
    def kernel(x_ref, counts_ref, mask_ref):
      counts_ref[...], mask = plsc.scan_count(x_ref[...])
      mask_ref[...] = mask.astype(jnp.int32)

    key = jax.random.key(42)
    x = jax.random.randint(key, shape, 0, 10, dtype=jnp.int32)
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

    @plsc.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      mask = x_ref[...] < 50
      # TODO: b/434208146 - Test with reduce!=1 when we support v6e packed masks
      o_ref[...] = plsc.all_reduce_population_count(mask)

    np.testing.assert_array_equal(
        kernel(x), np.broadcast_to(np.count_nonzero(x < 50), x.shape)
    )

  @parameterized.named_parameters(
      ("mixed", [0, 0, 1, 0, 1, 0, 0, 0], 2),
      ("all_zero", [0, 0, 0, 0, 0, 0, 0, 0], 8),
      ("all_one", [1, 1, 1, 1, 1, 1, 1, 1], 0))
  def test_ffs(self, data, expected):
    x = jnp.array(data)

    @plsc.vector_subcore_kernel(out_shape=x)
    def kernel(x_ref, o_ref):
      mask = x_ref[...] == 1
      # TODO: b/434208146 - Test with reduce!=1 when we support v6e packed masks
      o_ref[...] = plsc.all_reduce_ffs(mask)

    np.testing.assert_array_equal(kernel(x), np.broadcast_to(expected, x.shape))

  def test_run_scoped(self):
    x = jnp.arange(8)

    @plsc.vector_subcore_kernel(
        out_shape=x, out_specs=pl.BlockSpec(memory_space=pltpu.HBM)
    )
    def kernel(x_ref, o_hbm_ref):
      pltpu.sync_copy(x_ref, o_hbm_ref)

    np.testing.assert_array_equal(kernel(x), x)

  def test_concatenate(self):
    x = jnp.arange(2 * 8).reshape(-1, 8)

    @plsc.vector_subcore_kernel(
        out_shape=jax.ShapeDtypeStruct([2 * x.shape[0], x.shape[1]], x.dtype)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = lax.concatenate([x_ref[...], x_ref[...]], 0)

    np.testing.assert_array_equal(kernel(x), np.concatenate([x, x], 0))

  def test_scratch(self):
    x = jnp.arange(8)

    @plsc.vector_subcore_kernel(
        out_shape=x,
        scratch_shapes=(pltpu.VMEM([8], jnp.int32),),
    )
    def kernel(x_ref, scratch_ref, o_ref):
      scratch_ref[...] = x_ref[...]
      o_ref[...] = scratch_ref[...]

    np.testing.assert_array_equal(kernel(x), x)

  def test_implicit_padding_unsupported(self):
    x = jnp.arange(8, dtype=jnp.int32).reshape((8, 1))

    @plsc.vector_subcore_kernel(out_shape=x, in_specs=(pl.BlockSpec((8, 1)),))
    def kernel(*args):
      del args  # Unused.

    with self.assertRaisesRegex(ValueError, "must be a multiple of 8"):
      kernel(x)


class ScalarSubcoreTest(PallasSCTest):

  @property
  def num_cores(self):
    return sc_core._num_available_cores()

  def test_debug_print(self):
    int32s = jnp.arange(512, dtype=jnp.int32).reshape(64, 8)
    int16s = jnp.arange(512, dtype=jnp.int16).reshape(32, 16)
    int8s = jnp.arange(512, dtype=jnp.int8).reshape(16, 32)
    debug_int = 1234552
    debug_float = 12344.625

    @plsc.scalar_subcore_kernel(
        out_shape=int32s,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=self.num_cores),
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

  def test_copy(self):
    x = jnp.arange(16)

    @plsc.scalar_subcore_kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=self.num_cores),
    )
    def kernel(x_ref, o_ref):
      lax.cond(
          lax.axis_index("core") == lax.axis_size("core") - 1,
          lambda: pltpu.sync_copy(x_ref, o_ref),
          lambda: None,
      )

    np.testing.assert_array_equal(kernel(x), x)

  def test_sliced_copy(self):
    x = jnp.arange(self.num_cores * 8).reshape(self.num_cores, -1)

    @plsc.scalar_subcore_kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=self.num_cores),
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

    @plsc.scalar_subcore_kernel(
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

  def test_parallel_loop(self):
    x = jnp.arange(8*8).reshape(8, 8)

    @plsc.scalar_subcore_kernel(
        out_shape=x,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
        scratch_shapes=(
            pltpu.SMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ),
    )
    def kernel(x_ref, o_ref, tmp_ref, sem):
      pltpu.async_copy(x_ref, tmp_ref, sem).wait()

      @plsc.parallel_loop(0, tmp_ref.shape[0])
      def _(i):
        @plsc.parallel_loop(0, tmp_ref.shape[1], unroll=2)
        def _(j):
          tmp_ref[i, j] += 1

      pltpu.async_copy(tmp_ref, o_ref, sem).wait()

    np.testing.assert_array_equal(kernel(x), x + 1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
