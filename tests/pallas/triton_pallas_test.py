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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
import jax.lax
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

  @parameterized.product(src_dtype=DTYPE_LIST, dst_dtype=DTYPE_LIST)
  def test_fp_dtype_cast(self, src_dtype, dst_dtype):
    if src_dtype == dst_dtype:
      self.skipTest("No need to test the same dtype")
    if dtypes.bit_width(src_dtype) == 8 and dtypes.bit_width(dst_dtype) == 8:
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

      jax.lax.while_loop(_cond, lambda a: a, 0)
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


if __name__ == "__main__":
  absltest.main()
