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

import sys
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.typing import ArrayLike, DTypeLike
import jax.numpy as jnp


config.parse_flags_with_absl()


def _e8m0fnu(shape):
  key = jax.random.key(42)
  return jax.random.normal(key, shape, dtype=jnp.float8_e8m0fnu)


def _e4m3fn(shape):
  key = jax.random.key(42)
  return jax.random.normal(key, shape, dtype=jnp.float8_e4m3fn)


def _e5m2(shape):
  key = jax.random.key(42)
  return jax.random.normal(key, shape, dtype=jnp.float8_e5m2)


def _bf16(shape):
  key = jax.random.key(42)
  return jax.random.normal(key, shape, dtype=jnp.bfloat16)


def _scaled_dot_2d(
    *args,
    lhs: ArrayLike | None = None,
    rhs: ArrayLike | None = None,
    lhs_scales: ArrayLike | None = None,
    rhs_scales: ArrayLike | None = None,
    preferred_element_type: DTypeLike | None = None,
):
  """Helper function to create a scaled dot 2d.

  If the arguments are not provided, the default values are:

  lhs = _e4m3fn((1, 32), jnp.float8_e4m3fn),

  rhs = _e4m3fn((32, 4), jnp.float8_e4m3fn),

  lhs_scales = _e8m0fnu((1, 1), jnp.float8_e8m0fnu),

  rhs_scales = _e8m0fnu((1, 4), jnp.float8_e8m0fnu),
  """
  if args:
    raise ValueError(
        "Wrong test setup: all the arguments must be passed as keyword"
        " arguments."
    )
  if lhs is None:
    lhs = _e4m3fn((1, 32))
  if rhs is None:
    rhs = _e4m3fn((32, 4))
  if lhs_scales is None:
    lhs_scales = _e8m0fnu((1, 1))
  if rhs_scales is None:
    rhs_scales = _e8m0fnu((1, 4))
  jax.lax.scaled_dot(
      lhs,
      rhs,
      lhs_scale=lhs_scales,
      rhs_scale=rhs_scales,
      preferred_element_type=preferred_element_type,
  )


def _scaled_dot_3d(
    *args,
    lhs: ArrayLike | None = None,
    rhs: ArrayLike | None = None,
    lhs_scales: ArrayLike | None = None,
    rhs_scales: ArrayLike | None = None,
    preferred_element_type: DTypeLike | None = None,
):
  """Helper function to create a scaled dot 3d.

  If the arguments are not provided, the default values are:

    lhs = _e4m3fn((1, 1, 32))

    rhs = _e4m3fn((1, 32, 4))

    lhs_scales = _e8m0fnu((1, 1, 1))

    rhs_scales = _e8m0fnu((1, 1, 4))
  """
  if args:
    raise ValueError(
        "Wrong test setup: all the arguments must be passed as keyword"
        " arguments."
    )
  if lhs is None:
    lhs = _e4m3fn((1, 1, 32))
  if rhs is None:
    rhs = _e4m3fn((1, 32, 4))
  if lhs_scales is None:
    lhs_scales = _e8m0fnu((1, 1, 1))
  if rhs_scales is None:
    rhs_scales = _e8m0fnu((1, 1, 4))
  jax.lax.scaled_dot(
      lhs,
      rhs,
      lhs_scale=lhs_scales,
      rhs_scale=rhs_scales,
      preferred_element_type=preferred_element_type,
  )


def _quantize_to_fp8(x: jnp.ndarray, subchannel_size: int = 32):
  """Quantizes a bfloat16 tensor to fp8e4m3fn and returns fp8e8m0fnu scales."""
  assert x.dtype == jnp.bfloat16
  assert x.ndim == 3
  B, M, K = x.shape
  assert K % subchannel_size == 0
  num_subchannels = K // subchannel_size

  # Reshape for subchannel quantization
  x_reshaped = x.reshape(B, M, num_subchannels, subchannel_size)

  # Find maximum absolute value for scaling
  scales = jnp.max(jnp.abs(x_reshaped), axis=-1)
  scales = jnp.where(scales == 0.0, 1.0, scales)
  scales = scales.astype(jnp.float8_e8m0fnu)
  # Apply scales and quantize
  inv_scales = 1.0 / scales.astype(jnp.bfloat16)
  x_quantized = (x_reshaped * jnp.expand_dims(inv_scales, axis=-1)).astype(
      jnp.float8_e4m3fn
  )

  x_quantized = x_quantized.reshape(B, M, K)
  return x_quantized, scales


class ScaledDotTest(jtu.JaxTestCase):

  @parameterized.product(
      dtype=[jnp.float8_e4m3fn, jnp.float8_e5m2],
      B=[1],
      M=[1024],
      N=[256],
      K=[128],
  )
  def test_scaled_dot_fp6(self, dtype, B, M, N, K):
    a = self.rng().randn(B, M, K).astype(dtype)
    b = self.rng().randn(B, K, N).astype(dtype)

    a_scales = jnp.ones((B, M, K // 32), dtype=jnp.float8_e8m0fnu)
    b_scales = jnp.ones((B, K // 32, N), dtype=jnp.float8_e8m0fnu)

    @jax.jit
    def my_dot(a, b, a_scales, b_scales):
      return jax.lax.scaled_dot(
          a,
          b,
          lhs_scale=a_scales,
          rhs_scale=b_scales,
          preferred_element_type=jnp.bfloat16,
      )

    r = my_dot(a, b, a_scales, b_scales)
    self.assertEqual(r.dtype, jnp.bfloat16)

  @parameterized.product(
      S=[1, 2, 3, 4, 5],
      dtype=[jnp.float8_e4m3fn, jnp.float8_e5m2],
      B=[1],
      M=[256],
      N=[256],
      K=[128],
  )
  def test_scaled_dot_s_values(self, S, dtype, B, M, N, K):
    a = self.rng().randn(B, M, K * 32 * S).astype(dtype)
    b = self.rng().randn(B, K * 32 * S, N).astype(dtype)

    a_scales = jnp.ones((B, M, K), dtype=jnp.float8_e8m0fnu)
    b_scales = jnp.ones((B, K, N), dtype=jnp.float8_e8m0fnu)

    @jax.jit
    def my_dot(a, b, a_scales, b_scales):
      return jax.lax.scaled_dot(
          a,
          b,
          lhs_scale=a_scales,
          rhs_scale=b_scales,
          preferred_element_type=jnp.bfloat16,
      )

    r = my_dot(a, b, a_scales, b_scales)
    self.assertEqual(r.dtype, jnp.bfloat16)

  def test_same_rank_error(self):
    with self.assertRaisesRegex(TypeError, "must have the same rank."):
      _scaled_dot_2d(lhs=_e4m3fn((1, 1, 1)))

  def test_rank_2_or_3_error(self):
    with self.assertRaisesRegex(TypeError, "must have a rank 2 or 3."):
      _scaled_dot_2d(
          lhs=_e4m3fn((1)),
          rhs=_e4m3fn((1)),
          lhs_scales=_e8m0fnu((1)),
          rhs_scales=_e8m0fnu((1)),
      )

  def test_batch_dim_mismatch_error(self):
    with self.assertRaisesRegex(TypeError, "same batch dimension size"):
      _scaled_dot_3d(lhs=_e4m3fn((2, 1, 1)))

  def test_contracting_dim_mismatch_error(self):
    with self.assertRaisesRegex(
        TypeError,
        "LHS contracting dim .* of size .* does not match RHS contracting dim"
        " .* of size .*.",
    ):
      _scaled_dot_2d(lhs=_e4m3fn((1, 16)))

  def test_lhs_contracting_dim_too_small_error(self):
    with self.assertRaisesRegex(
        TypeError,
        "The ratio of LHS contracting dim .* to its scale's dim size .* must be"
        " a multiple of 32.",
    ):
      _scaled_dot_2d(lhs_scales=_e8m0fnu((1, 2)))

  def test_rhs_contracting_dim_too_small_error(self):
    with self.assertRaisesRegex(
        TypeError,
        "The ratio of RHS contracting dim .* to its scale's dim size .* must be"
        " a multiple of 32.",
    ):
      _scaled_dot_2d(rhs_scales=_e8m0fnu((2, 1)))

  def test_lhs_scale_dim_mismatch_error(self):
    with self.assertRaisesRegex(
        TypeError, "LHS dim .* of size .* does not match scale dim size .*."
    ):
      _scaled_dot_2d(lhs=_e4m3fn((4, 32)), lhs_scales=_e8m0fnu((1, 1)))

  def test_rhs_scale_dim_mismatch_error(self):
    with self.assertRaisesRegex(
        TypeError, "RHS dim .* of size .* does not match scale dim size .*."
    ):
      _scaled_dot_2d(rhs=_e4m3fn((32, 4)), rhs_scales=_e8m0fnu((1, 1)))

  def test_too_many_args(self):
    with self.assertRaisesRegex(
        TypeError, "takes 2 positional arguments but 3 were given"
    ):
      jax.lax.scaled_dot(_e4m3fn((1, 32)), _e4m3fn((32, 4)), _e8m0fnu((1, 1)))

  def test_scale_presence(self):
    with self.assertRaisesRegex(
        ValueError, "lhs_scale must be provided if lhs dtype is float8."
    ):
      jax.lax.scaled_dot(
          _e4m3fn((1, 32)), _e4m3fn((32, 4)), rhs_scale=_e8m0fnu((1, 4))
      )

  def test_scale_dtype_lhs(self):
    with self.assertRaisesRegex(
        TypeError,
        "lhs dtype must be float8_e4m3fn, float8_e5m2 or bfloat16, got"
        " float8_e8m0fnu",
    ):
      _scaled_dot_2d(lhs=_e8m0fnu((1, 32)))

  def test_scale_dtype_rhs(self):
    with self.assertRaisesRegex(
        TypeError,
        "rhs dtype must be float8_e4m3fn, float8_e5m2 or bfloat16, got"
        " float8_e8m0fnu",
    ):
      _scaled_dot_2d(rhs=_e8m0fnu((32, 4)))

  def test_preferred_element_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        "preferred_element_type must be one of bfloat16 or float32, got",
    ):
      _scaled_dot_2d(preferred_element_type=jnp.int32)

  def test_lhs_bf16(self):
    jax.lax.scaled_dot(
        _bf16((1, 32)),
        _e4m3fn((32, 4)),
        rhs_scale=_e8m0fnu((1, 4)),
    )

  def test_rhs_bf16(self):
    jax.lax.scaled_dot(
        _e4m3fn((1, 32)),
        _bf16((32, 4)),
        lhs_scale=_e8m0fnu((1, 1)),
    )

  def test_lhs_bf16_jit(self):
    @jax.jit
    def f(lhs, rhs, rhs_scale):
      return jax.lax.scaled_dot(lhs, rhs, rhs_scale=rhs_scale)

    r = f(_bf16((1, 32)), _e4m3fn((32, 4)), rhs_scale=_e8m0fnu((1, 4)))
    self.assertEqual(r.dtype, jnp.bfloat16)

  def test_rhs_bf16_jit(self):
    @jax.jit
    def f(lhs, rhs, lhs_scale):
      return jax.lax.scaled_dot(lhs, rhs, lhs_scale=lhs_scale)

    r = f(_e4m3fn((1, 32)), _bf16((32, 4)), lhs_scale=_e8m0fnu((1, 1)))
    self.assertEqual(r.dtype, jnp.bfloat16)

  def test_lhs_scale_bf16(self):
    with self.assertRaisesRegex(
        ValueError, "lhs_scale must be None if lhs dtype is bfloat16."
    ):
      jax.lax.scaled_dot(
          lhs=_bf16((1, 32)),
          rhs=_e4m3fn((32, 4)),
          lhs_scale=_e8m0fnu((1, 1)),
          rhs_scale=_e8m0fnu((1, 4)),
      )

  def test_rhs_scale_bf16(self):
    with self.assertRaisesRegex(
        ValueError, "rhs_scale must be None if rhs dtype is bfloat16."
    ):
      jax.lax.scaled_dot(
          _e4m3fn((1, 32)),
          _bf16((32, 4)),
          lhs_scale=_e8m0fnu((1, 1)),
          rhs_scale=_e8m0fnu((1, 4)),
      )

  def test_jit_passes(self):
    @jax.jit
    def scaled_dot_with_jit(lhs, rhs, lhs_scale, rhs_scale):
      return jax.lax.scaled_dot(
          lhs,
          rhs,
          lhs_scale=lhs_scale,
          rhs_scale=rhs_scale,
          preferred_element_type=jnp.bfloat16,
      )

    r = scaled_dot_with_jit(
        _e4m3fn((1, 32)),
        _e4m3fn((32, 4)),
        lhs_scale=_e8m0fnu((1, 1)),
        rhs_scale=_e8m0fnu((1, 4)),
    )
    self.assertEqual(r.dtype, jnp.bfloat16)

  def test_multiple_contracting_dims_error(self):
    with self.assertRaisesRegex(
        TypeError, "Only one contracting dimension is supported."
    ):
      jax.lax.scaled_dot(
          _e4m3fn((1, 32, 32)),
          _e4m3fn((32, 32, 4)),
          lhs_scale=_e8m0fnu((1, 1, 1)),
          rhs_scale=_e8m0fnu((1, 1, 4)),
          dimension_numbers=(((1, 2), (0, 1)), ((), ())),
      )

  def test_multiple_batch_dims_error(self):
    with self.assertRaisesRegex(
        TypeError, "Only one batch dimension is supported."
    ):
      jax.lax.scaled_dot(
          _e4m3fn((2, 2, 32)),
          _e4m3fn((2, 2, 32, 4)),
          lhs_scale=_e8m0fnu((2, 2, 1)),
          rhs_scale=_e8m0fnu((2, 2, 4)),
          dimension_numbers=(((2,), (2,)), ((0, 1), (0, 1))),
      )

  def test_bf16_dot_vs_scaled_dot_numeric_equivalence(self):
    B, M, N, K = 1, 16, 64, 4096
    x = jnp.abs(self.rng().randn(B, M, K).astype(jnp.bfloat16))
    print(f"x: {x.shape}", flush=True, file=sys.stderr)
    y = jnp.abs(self.rng().randn(B, K, N).astype(jnp.bfloat16))
    print(f"y: {y.shape}", flush=True, file=sys.stderr)
    y = y.transpose((0, 2, 1))
    x_fp8, x_scales = _quantize_to_fp8(x)
    y_fp8, y_scales = _quantize_to_fp8(y)

    @jax.jit
    def scaled_dot(a, b, a_scales, b_scales):
      return jax.lax.scaled_dot(
          a,
          b,
          lhs_scale=a_scales,
          rhs_scale=b_scales,
          preferred_element_type=jnp.float32,
          dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      )

    scaled_dot_result = scaled_dot(x_fp8, y_fp8, x_scales, y_scales)

    @jax.jit
    def bf16_dot(a, b):
      return jax.lax.dot_general(
          a,
          b,
          preferred_element_type=jnp.float32,
          dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      )

    original_bf16_dot_result = bf16_dot(x, y)
    self.assertAllClose(
        scaled_dot_result, original_bf16_dot_result, atol=1e0, rtol=1e0
    )

  def test_working_example(self):
    B = 32
    M = 16384
    N = 16
    K = 4096
    subchannel_size = 32

    lhs_shape = (B, M, K)
    rhs_shape = (B, K, N)
    lhs_scales_shape = (B, M, K // subchannel_size)
    rhs_scales_shape = (B, K // subchannel_size, N)

    key = jax.random.key(42)

    lhs = jax.random.normal(key, lhs_shape, dtype=jnp.float8_e4m3fn)
    rhs = jax.random.normal(key, rhs_shape, dtype=jnp.float8_e4m3fn)
    lhs_scales = jax.random.normal(
        key, lhs_scales_shape, dtype=jnp.float8_e8m0fnu
    )
    rhs_scales = jax.random.normal(
        key, rhs_scales_shape, dtype=jnp.float8_e8m0fnu
    )

    @jax.jit
    def scaled_dot_fn(lhs, rhs, lhs_scale, rhs_scale):
      return jax.lax.scaled_dot(
          lhs,
          rhs,
          lhs_scale=lhs_scale,
          rhs_scale=rhs_scale,
          preferred_element_type=jnp.bfloat16,
      )

    result = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales,
    )
    self.assertEqual(result.dtype, jnp.bfloat16)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
