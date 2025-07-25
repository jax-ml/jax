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

intx = dtypes.canonicalize_dtype(jnp.int64)
floatx = dtypes.canonicalize_dtype(jnp.float64)


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
      ("add_i32", plgpu.atomic_add, np.array([1, 2, 3, 4], np.int32), np.sum),
      ("max_i32", plgpu.atomic_max, np.array([1, 2, 3, 4], np.int32), np.max),
      ("min_i32", plgpu.atomic_min, np.array([1, 2, 3, 4], np.int32), np.min),
      ("add_f16", plgpu.atomic_add, np.array([1, 2, 3, 4], np.float16), np.sum),
      ("add_f32", plgpu.atomic_add, np.array([1, 2, 3, 4], np.float32), np.sum),
      ("max_f32", plgpu.atomic_max, np.array([1, 2, 3, 4], np.float32), np.max),
      ("min_f32", plgpu.atomic_min, np.array([1, 2, 3, 4], np.float32), np.min),
  )
  def test_scalar_atomic(self, op, value, numpy_op):
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
      x = pl.load(x_ref, idx)
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


if __name__ == "__main__":
  absltest.main()
