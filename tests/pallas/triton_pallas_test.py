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

import functools
import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import triton as plgpu
else:
  plgpu = None

config.parse_flags_with_absl()

intx = dtypes.default_int_dtype()
floatx = dtypes.default_float_dtype()


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jtu.test_device_matches(["cpu"]):
      if not self.INTERPRET:
        self.skipTest("On CPU the test works only in interpret mode")
    elif jtu.test_device_matches(["gpu"]):
      if not jtu.is_cuda_compute_capability_at_least("9.0"):
        self.skipTest("Only works on GPU with capability >= sm90")
    else:
      self.skipTest("Test only works on CPU and GPU")

    super().setUp()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


DTYPE_LIST = [jnp.float32, jnp.float16, jnp.bfloat16,
              jnp.float8_e4m3fn, jnp.float8_e5m2]


class TritonPallasTest(PallasBaseTest):
  INTERPRET = False

  def setUp(self):
    super().setUp()
    # Force tests to use Triton.
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(False))

  @parameterized.product(src_dtype=DTYPE_LIST, dst_dtype=DTYPE_LIST)
  def test_fp_dtype_cast(self, src_dtype, dst_dtype):
    if src_dtype == dst_dtype:
      self.skipTest("No need to test the same dtype")
    if dtypes.itemsize_bits(src_dtype) == 8 and dtypes.itemsize_bits(dst_dtype) == 8:
      self.skipTest("Not casting between 8-bit types")

    def body(x_ref, y_ref):
      y_ref[...] = x_ref[...].astype(dst_dtype)

    x = 10 * jax.random.normal(jax.random.key(0), (64, 64), dtype=src_dtype)
    y = self.pallas_call(body,
        in_specs=[pl.BlockSpec((64, 64), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((64, 64), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 64), dst_dtype),
        grid=(1,),
    )(x)
    self.assertEqual(y.dtype, dst_dtype)
    self.assertArraysEqual(y, x.astype(dst_dtype))

  @parameterized.named_parameters(
      ("add_i32", "atomic_add", np.array([1, 2, 3, 4], np.int32), np.sum),
      ("max_i32", "atomic_max", np.array([1, 2, 3, 4], np.int32), np.max),
      ("min_i32", "atomic_min", np.array([1, 2, 3, 4], np.int32), np.min),
      ("add_f16", "atomic_add", np.array([1, 2, 3, 4], np.float16), np.sum),
      ("add_f32", "atomic_add", np.array([1, 2, 3, 4], np.float32), np.sum),
      ("max_f32", "atomic_max", np.array([1, 2, 3, 4], np.float32), np.max),
      ("min_f32", "atomic_min", np.array([1, 2, 3, 4], np.float32), np.min),
  )
  def test_scalar_atomic(self, op, value, numpy_op):
    if plgpu is None:
      self.skipTest("plgpu not available on this platform.")

    op = getattr(plgpu, op)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), value.dtype),
        grid=value.shape[0],
        input_output_aliases={1: 0},
    )
    def atomic_kernel(x_ref, _, o_ref):
      pid = pl.program_id(axis=0)
      op(o_ref, (), x_ref[pid])

    if op == plgpu.atomic_add:
      neutral = np.array(0, dtype=value.dtype)
    elif op == plgpu.atomic_max:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).min, value.dtype)
      else:
        neutral = np.array(-float("inf"), value.dtype)
    elif op == plgpu.atomic_min:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).max, value.dtype)
      else:
        neutral = np.array(float("inf"), value.dtype)
    elif op == plgpu.atomic_or:
      neutral = np.array(False, value.dtype)
    else:
      raise NotImplementedError()
    out = atomic_kernel(value, neutral)
    np.testing.assert_allclose(out, numpy_op(value))

  @parameterized.parameters((0,), (1,))
  def test_array_atomic_add(self, axis):
    m, n = 32, 8
    if axis == 0:
      grid = m
    else:
      grid = n
    out_shape = jax.ShapeDtypeStruct((n if axis == 0 else m,), floatx)

    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=grid,
        input_output_aliases={1: 0},
    )
    def reduce(x_ref, _, y_ref):
      i = pl.program_id(axis=0)
      if axis == 0:
        idx = (i, jnp.arange(n))
      else:
        idx = (jnp.arange(m), i)
      x = x_ref[idx]
      plgpu.atomic_add(y_ref, (jnp.arange(y.shape[0]),), x)

    x = jax.random.normal(jax.random.key(0), (m, n))
    y = jnp.zeros(out_shape.shape, out_shape.dtype)
    y = reduce(x, y)
    y_ref = np.sum(x, axis=axis)
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  @parameterized.parameters(
      (0, 0, 1),
      (0, 1, 1),
      (1, 0, 1),
      (1, 1, 1),
      (2, 1, 1),
      (2, 1, 1),
  )
  def test_atomic_cas(self, init_value, cmp, new_value):
    if jax.config.x64_enabled:
      self.skipTest("Not supported in 64-bit mode")

    @functools.partial(
        self.pallas_call,
        out_shape=(
            jax.ShapeDtypeStruct((), intx),
            jax.ShapeDtypeStruct((), intx),
        ),
        input_output_aliases={0: 0},
    )
    def swap(_, lock_ref, out_ref):
      out_ref[()] = plgpu.atomic_cas(lock_ref, cmp, new_value)

    lock, out = swap(init_value)
    np.testing.assert_allclose(
        lock, new_value if cmp == init_value else init_value
    )
    np.testing.assert_allclose(out, init_value)

  @parameterized.parameters(1, 2, 3, 4, 8)
  def test_atomic_counter(self, num_threads):
    if self.INTERPRET:
      self.skipTest("While loop not supported in interpret mode.")
    if jax.config.x64_enabled:
      self.skipTest("Not supported in 64-bit mode")

    @functools.partial(
        self.pallas_call,
        out_shape=(
            jax.ShapeDtypeStruct((), intx),
            jax.ShapeDtypeStruct((), intx),
        ),
        input_output_aliases={0: 0, 1: 1},
        grid=(num_threads,),
    )
    def increment(_, __, lock_ref, counter_ref):
      def _cond(_):
        return plgpu.atomic_cas(lock_ref, 0, 1) == 1

      lax.while_loop(_cond, lambda a: a, 0)
      counter_ref[...] += 1
      plgpu.atomic_xchg(lock_ref, (), 0)

    lock, count = increment(0, 0)
    np.testing.assert_allclose(lock, 0)
    np.testing.assert_allclose(count, num_threads)

  @parameterized.product(
      size=[1, 2, 64, 129, 1021],
      block_size=[1, 2, 32, 64, 128],
  )
  def test_masked_load_store(self, size, block_size):
    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((size,), floatx)),
        grid=pl.cdiv(size, block_size),
    )
    def kernel(x_ref, o_ref):
      idx = pl.program_id(0) * block_size + jnp.arange(
          block_size, dtype=jnp.int32
      )
      mask = idx < x_ref.shape[0]
      x = plgpu.load(x_ref.at[idx], mask=mask)
      plgpu.store(o_ref.at[idx], x + 1.0, mask=mask)

    key = jax.random.key(0)
    x = jax.random.normal(key, (size,))
    np.testing.assert_allclose(kernel(x), x + 1.0, atol=1e-5, rtol=1e-5)

  def test_masked_oob_load_store_slice(self):
    n = 16

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((n,), floatx)),
    )
    def masked_oob_load_store_slice(x_ref, mask_ref, start_idx_ref, o_ref):
      x = plgpu.load(
          x_ref.at[pl.ds(start_idx_ref[()], n)], mask=mask_ref[:], other=-1.0
      )
      o_ref[...] = x

    x = jax.random.normal(jax.random.key(0), (n,))
    slice_start = jax.random.randint(jax.random.key(2), (), 1, n)
    indices = jnp.arange(n) + slice_start
    mask = indices < n
    out = masked_oob_load_store_slice(x, mask, slice_start)
    o_new = jnp.where(mask, x[indices], jnp.full_like(x, -1.0))
    np.testing.assert_array_equal(out, o_new)

  @parameterized.parameters(
      ((16, 32), (16,)),
      ((16, 32), (32,)),
      ((16, 32), (16, 16)),
  )
  def test_invalid_broadcasted_load(self, x_shape, mask_shape):
    if self.INTERPRET:
      self.skipTest("No broadcasting checks in pl.load in interpret mode")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32)
    )
    def kernel(x_ref, mask_ref, o_ref):
      del o_ref  # Unused.
      plgpu.load(x_ref, mask=mask_ref[:])

    x = jnp.ones(x_shape, dtype=jnp.float32)
    mask = jnp.ones(mask_shape, dtype=jnp.bool_)
    # assertRaises* methods do not support inspecting the __cause__, so
    # we have to check it manually.
    try:
      kernel(x, mask)
    except Exception as e:
      self.assertIn("Cannot broadcast", str(e.__cause__))
    else:
      self.fail("Expected exception due to invalid broadcasting")

  @parameterized.parameters("float16", "bfloat16", "float32")
  def test_approx_tanh(self, dtype):
    # Skip approx_tanh tests on ROCm due to missing tanh.approx instruction.
    # TODO(GulsumGudukbay): Will be unskipped once PR 34598 is merged. Issue #34711.
    if jtu.is_device_rocm() and dtype in ("float16", "float32"):
      self.skipTest("Skipped on ROCm due to missing tanh.approx instruction.")  # test_approx_tanh0 and test_approx_tanh2
    if self.INTERPRET:
      self.skipTest("approx_tanh is not supported in interpret mode")

    if (dtype == "bfloat16" and
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("tanh.approx.bf16 requires a GPU with capability >= sm90")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), dtype),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plgpu.approx_tanh(x_ref[...])

    x = jnp.asarray([-1, 0.42, 0.24, 1]).astype(dtype)
    # We upcast to float32 because NumPy <2.0 does not handle custom dtypes
    # properly. See https://github.com/jax-ml/jax/issues/11014.
    np.testing.assert_allclose(
        kernel(x).astype(jnp.float32),
        jnp.tanh(x).astype(jnp.float32),
        atol=5e-3,
        rtol=5e-3,
    )

  def test_elementwise_inline_asm(self):
    if self.INTERPRET:
      self.skipTest(
          "elementwise_inline_asm is not supported in interpret mode"
      )

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float16),
    )
    def kernel(x_ref, o_ref):
      [o_ref[...]] = plgpu.elementwise_inline_asm(
          "tanh.approx.f16x2 $0, $1;",
          args=[x_ref[...]],
          constraints="=r,r",
          pack=2,
          result_shape_dtypes=[jax.ShapeDtypeStruct(x_ref.shape, x_ref.dtype)],
      )

    x = jnp.arange(256).astype(jnp.float16)
    np.testing.assert_allclose(kernel(x), jnp.tanh(x), atol=5e-3, rtol=5e-3)

  def test_debug_barrier(self):
    if self.INTERPRET:
      self.skipTest("debug_barrier is not supported in interpret mode")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]
      plgpu.debug_barrier()

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x)

  @unittest.skipIf(
      sys.platform == "win32",
      "plgpu.CompilerParams unavailable on Windows",
  )
  def test_debug_print(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test flakes on gpu")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        compiler_params=plgpu.CompilerParams(
            num_warps=1, num_stages=1
        ),
    )
    def kernel(x_ref, o_ref):
      pl.debug_print("It works!")

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
      jax.effects_barrier()

    self.assertIn("It works!", output())

  @unittest.skipIf(
      sys.platform == "win32",
      "plgpu.CompilerParams unavailable on Windows",
  )
  def test_debug_print_with_values(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test flakes on gpu")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        compiler_params=plgpu.CompilerParams(
            num_warps=1, num_stages=1
        ),
    )
    def kernel(x_ref, o_ref):
      pl.debug_print("x[0] =", x_ref[0])

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
      jax.effects_barrier()

    self.assertIn("x[0] = 4.2", output())

  @parameterized.named_parameters(*[
    (f"m_{m}_n_{n}_k_{k}_dtype_{dtype}_bm_{block_size_m}_"
     f"bn_{block_size_n}_bk_{block_size_k}_gm_{group_size_m}", m, n, k, dtype,
     block_size_m, block_size_n, block_size_k, group_size_m)
      for m in [512, 1024]
      for k in [512]
      for n in [512, 1024]
      for dtype in ["float32", "float16"]
      for block_size_m in [64, 128]
      for block_size_n in [64, 128]
      for block_size_k in [32]
      for group_size_m in [8]
      if block_size_m <= m and block_size_n <= n and block_size_k <= k
    ])
  def test_matmul(self, m, n, k, dtype, bm, bn, bk, gm):
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      self.skipTest("On TPU the test works only in interpret mode")
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (m, k), dtype=dtype)
    y = jax.random.normal(k2, (k, n), dtype=dtype)
    out = matmul(x, y, bm=bm, bn=bn, bk=bk, gm=gm,
                 interpret=self.INTERPRET)
    expected = jnp.matmul(
            x, y, preferred_element_type=jnp.float32).astype(dtype)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.named_parameters(*(
      dict(
          testcase_name=f"{batch_size}_{size}_{block_size}_{dtype}",
          batch_size=batch_size,
          size=size,
          block_size=block_size,
          dtype=dtype,
      )
      for batch_size in [1, 2, 4, 23]
      for size in [1, 2, 129, 255, 256]
      for block_size in [1, 2, 32, 64, 128, 256]
      for dtype in ["float32"]
      if size < block_size
  ))
  def test_softmax(self, batch_size, size, block_size, dtype):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((batch_size, size), dtype),
        grid=batch_size,
    )
    def softmax(x_ref, o_ref):
      row_idx = pl.program_id(0)
      x_idx = jnp.arange(block_size)
      row_idxs = (row_idx, x_idx)
      mask = x_idx < x_ref.shape[1]
      row = plgpu.load(x_ref.at[row_idxs], mask=mask, other=-float("inf"))
      row_minus_max = row - jnp.max(row, axis=0)
      numerator = jnp.exp(row_minus_max)
      denominator = jnp.sum(numerator, axis=0)
      softmax_output = numerator / denominator
      plgpu.store(o_ref.at[row_idxs], softmax_output, mask=mask)

    key = jax.random.key(0)
    x = jax.random.normal(key, [batch_size, size], dtype=dtype)
    np.testing.assert_allclose(
        softmax(x), jax.nn.softmax(x, axis=-1), atol=1e-5, rtol=1e-5
    )


@functools.partial(
    jax.jit, static_argnames=["bm", "bn", "gm", "bk", "interpret", "debug"]
)
def matmul(x, y, *, bm, bn, gm, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]

  @functools.partial(
      pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      grid=pl.cdiv(m, bm) * pl.cdiv(n, bn),
  )
  def matmul_kernel(x_ref, y_ref, o_ref):
    pid = pl.program_id(axis=0).astype(intx)
    num_pid_m = m // bm
    num_pid_n = n // bn
    num_pid_in_group = gm * num_pid_n
    group_id = lax.div(pid, num_pid_in_group)
    first_pid_m = group_id * gm
    group_size_m = jnp.minimum(num_pid_m - first_pid_m, gm)
    pid_m = first_pid_m + lax.rem(pid, group_size_m)
    pid_n = lax.div(lax.rem(pid, num_pid_in_group), group_size_m)
    idx_m = pid_m * bm + jnp.arange(bm)
    idx_n = pid_n * bn + jnp.arange(bn)
    idx_m = plgpu.max_contiguous(pl.multiple_of(idx_m, bm), bm)
    idx_n = plgpu.max_contiguous(pl.multiple_of(idx_n, bn), bn)
    acc = jnp.zeros((bm, bn), dtype=jnp.float32)

    def body(i, acc):
      idx_k = i * bk + jnp.arange(bk)
      x_idx = (
          lax.broadcast_in_dim(idx_m, (bm, bk), (0,)),
          lax.broadcast_in_dim(idx_k, (bm, bk), (1,)),
      )
      y_idx = (
          lax.broadcast_in_dim(idx_k, (bk, bn), (0,)),
          lax.broadcast_in_dim(idx_n, (bk, bn), (1,)),
      )
      x_block, y_block = x_ref[x_idx], y_ref[y_idx]
      out = pl.dot(x_block, y_block)
      return acc + out

    acc = lax.fori_loop(0, k // bk, body, acc).astype(o_ref.dtype)
    o_idx = (
        lax.broadcast_in_dim(idx_m, (bm, bn), (0,)),
        lax.broadcast_in_dim(idx_n, (bm, bn), (1,)),
    )
    o_ref[o_idx] = acc

  return matmul_kernel(x, y)


if __name__ == "__main__":
  absltest.main()
