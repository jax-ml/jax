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

from functools import partial
import os
import sys
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.typing import ArrayLike, DTypeLike
import jax.numpy as jnp

if "pytest" not in sys.modules:
  os.environ["XLA_FLAGS"] = (
      "--xla_gpu_experimental_scaled_dot_with_triton=true "
  )


config.parse_flags_with_absl()


def _e8m0fnu(shape):
  key = jax.random.key(42)
  return jax.random.randint(
      key, shape, minval=0, maxval=256, dtype=jnp.int8
  ).astype(jnp.float8_e8m0fnu)


def _e4m3fn(shape):
  key = jax.random.key(42)
  return jax.random.normal(key, shape, dtype=jnp.float8_e4m3fn)


def _e5m2(shape):
  key = jax.random.key(42)
  return jax.random.normal(key, shape, dtype=jnp.float8_e5m2)


def _bf16(shape):
  key = jax.random.key(42)
  return jax.random.normal(key, shape, dtype=jnp.bfloat16)


def _scale(shape, dtype=jnp.float8_e8m0fnu):
  key = jax.random.key(42)
  if dtype == jnp.float8_e8m0fnu:
    return jax.random.randint(
        key, shape, minval=0, maxval=256, dtype=jnp.int8
    ).astype(jnp.float8_e8m0fnu)
  elif dtype == jnp.bfloat16:
    return jax.random.normal(key, shape, dtype=jnp.bfloat16)
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")


def _scaled_dot_2d(
    *args,
    lhs: ArrayLike | None = None,
    rhs: ArrayLike | None = None,
    lhs_scales: ArrayLike | None = None,
    rhs_scales: ArrayLike | None = None,
    preferred_element_type: DTypeLike | None = None,
    scale_dtype: DTypeLike = jnp.float8_e8m0fnu,
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
    lhs_scales = _scale((1, 1), scale_dtype)
  if rhs_scales is None:
    rhs_scales = _scale((1, 4), scale_dtype)
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
    scale_dtype: DTypeLike = jnp.float8_e8m0fnu,
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
    lhs_scales = _scale((1, 1, 1), scale_dtype)
  if rhs_scales is None:
    rhs_scales = _scale((1, 1, 4), scale_dtype)
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


@jax.jit(static_argnames=["dimension_numbers", "preferred_element_type"])
def scaled_dot_fn(
    lhs,
    rhs,
    *,
    lhs_scale=None,
    rhs_scale=None,
    dimension_numbers=None,
    preferred_element_type=None,
):
  return jax.lax.scaled_dot(
      lhs,
      rhs,
      lhs_scale=lhs_scale,
      rhs_scale=rhs_scale,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )


class ScaledDotTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if "pytest" in sys.modules:
      self.skipTest(
          "Skip scaled_dot_test when running under pytest. We don't want to set"
          " XLA_FLAGS in pytest because it could affect other tests."
      )

  def _should_skip_test(self, scale_dtype):
    if jtu.device_under_test() == "tpu" and scale_dtype == jnp.float8_e8m0fnu:
      self.skipTest("Skip. TPU does not support f8e8m0fnu.")
    if (
        jtu.device_under_test() == "gpu"
        and scale_dtype == jnp.float8_e8m0fnu
        and not jtu.is_cuda_compute_capability_at_least("8.9")
    ):
      self.skipTest("Skip. Old GPU does not support f8e8m0fnu.")

  @parameterized.product(
      scale_dtype=[jnp.float8_e8m0fnu, jnp.bfloat16],
  )
  def test_working_example(self, scale_dtype):
    self._should_skip_test(scale_dtype)
    B = 32
    M = 1024
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
    lhs_scales = jax.random.normal(key, lhs_scales_shape, dtype=scale_dtype)
    rhs_scales = jax.random.normal(key, rhs_scales_shape, dtype=scale_dtype)

    @jax.jit
    def scaled_dot_fn(lhs, rhs, lhs_scale, rhs_scale):
      return jax.lax.scaled_dot(
          lhs,
          rhs,
          lhs_scale=lhs_scale,
          rhs_scale=rhs_scale,
      )

    result_jit = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales,
    )
    self.assertEqual(result_jit.dtype, jnp.bfloat16)

  @parameterized.product(
      dtype=[jnp.float8_e4m3fn, jnp.float8_e5m2],
      scale_dtype=[jnp.float8_e8m0fnu, jnp.bfloat16],
      B=[1],
      M=[1024],
      N=[256],
      K=[128],
  )
  def test_fp8_types(self, dtype, scale_dtype, B, M, N, K):
    self._should_skip_test(scale_dtype)
    a = self.rng().randn(B, M, K).astype(dtype)
    b = self.rng().randn(B, K, N).astype(dtype)

    a_scales = jnp.ones((B, M, K // 32), dtype=scale_dtype)
    b_scales = jnp.ones((B, K // 32, N), dtype=scale_dtype)

    r = scaled_dot_fn(a, b, lhs_scale=a_scales, rhs_scale=b_scales)
    self.assertEqual(r.dtype, jnp.bfloat16)

  @parameterized.product(
      S=[1, 2, 3, 4, 5],
      dtype=[jnp.float8_e4m3fn, jnp.float8_e5m2],
      scale_dtype=[jnp.float8_e8m0fnu, jnp.bfloat16],
      B=[1],
      M=[256],
      N=[256],
      K=[128],
  )
  def test_different_subchannel_sizes(self, S, dtype, scale_dtype, B, M, N, K):
    self._should_skip_test(scale_dtype)
    a = self.rng().randn(B, M, K * 32 * S).astype(dtype)
    b = self.rng().randn(B, K * 32 * S, N).astype(dtype)

    a_scales = jnp.ones((B, M, K), dtype=scale_dtype)
    b_scales = jnp.ones((B, K, N), dtype=scale_dtype)

    result_jit = scaled_dot_fn(a, b, lhs_scale=a_scales, rhs_scale=b_scales)
    result = jax.lax.scaled_dot(a, b, lhs_scale=a_scales, rhs_scale=b_scales)
    self.assertAllClose(result_jit, result)

  def test_same_rank_error(self):
    with self.assertRaisesRegex(TypeError, "must have the same rank."):
      _scaled_dot_2d(lhs=_e4m3fn((1, 1, 1)), scale_dtype=jnp.bfloat16)

  def test_batch_dim_mismatch_error(self):
    with self.assertRaisesRegex(TypeError, "same batch dimension size"):
      _scaled_dot_3d(lhs=_e4m3fn((2, 1, 1)), scale_dtype=jnp.bfloat16)

  def test_contracting_dim_mismatch_error(self):
    with self.assertRaisesRegex(
        TypeError,
        "LHS contracting dim .* of size .* does not match RHS contracting dim"
        " .* of size .*.",
    ):
      _scaled_dot_2d(lhs=_e4m3fn((1, 16)), scale_dtype=jnp.bfloat16)

  def test_lhs_contracting_dim_too_small_error(self):
    with self.assertRaisesRegex(
        TypeError,
        "The ratio of LHS contracting dim .* to its scale's dim size .* must be"
        " at least 2.",
    ):
      _scaled_dot_2d(lhs_scales=_bf16((1, 32)), scale_dtype=jnp.bfloat16)

  def test_rhs_contracting_dim_too_small_error(self):
    with self.assertRaisesRegex(
        TypeError,
        "The ratio of RHS contracting dim .* to its scale's dim size .* must be"
        " at least 2.",
    ):
      _scaled_dot_2d(rhs_scales=_bf16((32, 4)), scale_dtype=jnp.bfloat16)

  def test_lhs_scale_dim_mismatch_error(self):
    with self.assertRaisesRegex(
        TypeError, "LHS dim .* of size .* does not match scale dim size .*."
    ):
      _scaled_dot_2d(
          lhs=_e4m3fn((4, 32)),
          lhs_scales=_bf16((1, 1)),
          scale_dtype=jnp.bfloat16,
      )

  def test_rhs_scale_dim_mismatch_error(self):
    with self.assertRaisesRegex(
        TypeError, "RHS dim .* of size .* does not match scale dim size .*."
    ):
      _scaled_dot_2d(
          rhs=_e4m3fn((32, 4)),
          rhs_scales=_bf16((1, 1)),
          scale_dtype=jnp.bfloat16,
      )

  def test_too_many_args(self):
    with self.assertRaisesRegex(
        TypeError, "takes 2 positional arguments but 3 were given"
    ):
      jax.lax.scaled_dot(_e4m3fn((1, 32)), _e4m3fn((32, 4)), _bf16((1, 1)))

  def test_lhs_bf16(self):
    """Mix precision case.

    Here we check the mix precison case where the lhs is bf16
    and the rhs is e4m3fn with the rhs scale being e8m0fnu.
    """
    self._should_skip_test(jnp.float8_e8m0fnu)
    lhs = _bf16((1, 32))
    rhs = _e4m3fn((32, 4))
    rhs_scale = _e8m0fnu((1, 4))
    result = jax.lax.scaled_dot(
        lhs,
        rhs,
        rhs_scale=rhs_scale,
    )
    result_jit = scaled_dot_fn(lhs, rhs, rhs_scale=rhs_scale)
    self.assertAllClose(result, result_jit)

  def test_rhs_bf16(self):
    """Mix precision case.

    Here we check the mix precison case where the rhs is bf16
    and the lhs is e4m3fn with the lhs scale being e8m0fnu.
    """
    if jtu.device_under_test() == "tpu":
      self.skipTest("Skip. TPU does not support f8e8m0fnu.")
    lhs = _e4m3fn((1, 32))
    rhs = _bf16((32, 4))
    lhs_scale = _e8m0fnu((1, 1))
    result = jax.lax.scaled_dot(lhs, rhs, lhs_scale=lhs_scale)
    result_jit = scaled_dot_fn(lhs, rhs, lhs_scale=lhs_scale)
    self.assertAllClose(result, result_jit)

  def test_f16_and_f32_operand_types(self):
    self._should_skip_test(jnp.float8_e8m0fnu)
    B, M, N, K = 1, 32, 32, 32
    lhs_scale = _e8m0fnu((B, M, 1))
    rhs_scale = _e8m0fnu((B, 1, N))

    # float16
    lhs = self.rng().randn(B, M, K).astype(jnp.float16)
    rhs = self.rng().randn(B, K, N).astype(jnp.float16)
    result = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        preferred_element_type=jnp.float32,
    )
    self.assertEqual(result.dtype, jnp.float32)

    # float32
    lhs = self.rng().randn(B, M, K).astype(jnp.float32)
    rhs = self.rng().randn(B, K, N).astype(jnp.float32)
    result = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        preferred_element_type=jnp.float32,
    )
    self.assertEqual(result.dtype, jnp.float32)

  def test_mixed_precision_scales(self):
    B, M, N, K = 1, 32, 32, 32
    # Case 1: BF16 operands with BF16 scales
    lhs = self.rng().randn(B, M, K).astype(jnp.bfloat16)
    rhs = self.rng().randn(B, K, N).astype(jnp.bfloat16)
    lhs_scale = jnp.full((B, M, 1), 2.0, dtype=jnp.bfloat16)
    rhs_scale = jnp.full((B, 1, N), 0.5, dtype=jnp.bfloat16)

    # Reference
    lhs_ref = lhs * lhs_scale
    rhs_ref = rhs * rhs_scale
    ref = jax.lax.dot_general(
        lhs_ref,
        rhs_ref,
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    res = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        preferred_element_type=jnp.float32,
    )

    self.assertAllClose(res, ref, atol=1e-1, rtol=1e-1)

    # Case 2: F32 operands with BF16 scales
    lhs = self.rng().randn(B, M, K).astype(jnp.float32)
    rhs = self.rng().randn(B, K, N).astype(jnp.float32)
    lhs_scale = jnp.full((B, M, 1), 2.0, dtype=jnp.bfloat16)
    rhs_scale = jnp.full((B, 1, N), 0.5, dtype=jnp.bfloat16)

    lhs_ref = lhs * lhs_scale.astype(jnp.float32)
    rhs_ref = rhs * rhs_scale.astype(jnp.float32)
    ref = jax.lax.dot_general(
        lhs_ref,
        rhs_ref,
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

    res = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        preferred_element_type=jnp.float32,
    )
    self.assertAllClose(res, ref, atol=1e-1, rtol=1e-1)

  def test_jit_passes(self):
    self._should_skip_test(jnp.float8_e8m0fnu)
    result_jit = scaled_dot_fn(
        _e4m3fn((1, 32)),
        _e4m3fn((32, 4)),
        lhs_scale=_e8m0fnu((1, 1)),
        rhs_scale=_e8m0fnu((1, 4)),
    )
    self.assertEqual(result_jit.dtype, jnp.bfloat16)

  @parameterized.product(
      scale_dtype=[jnp.float8_e8m0fnu, jnp.bfloat16],
  )
  def test_multiple_contracting_dims_jit(self, scale_dtype):
    self._should_skip_test(scale_dtype)
    lhs = _e4m3fn((1, 32, 32))
    rhs = _e4m3fn((32, 32, 4))
    lhs_scale = _scale((1, 1, 1), dtype=scale_dtype)
    rhs_scale = _scale((1, 1, 4), dtype=scale_dtype)
    dimension_numbers = (((1, 2), (0, 1)), ((), ()))

    result = jax.lax.scaled_dot(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        dimension_numbers=dimension_numbers,
    )
    self.assertEqual(result.dtype, jnp.bfloat16)

    result_jit = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        dimension_numbers=dimension_numbers,
    )
    self.assertAllClose(result, result_jit)

  @parameterized.product(
      scale_dtype=[jnp.float8_e8m0fnu, jnp.bfloat16],
  )
  def test_multiple_batch_dims_jit(self, scale_dtype):
    self._should_skip_test(scale_dtype)
    lhs = _e4m3fn((2, 2, 1, 32))
    rhs = _e4m3fn((2, 2, 4, 32))
    lhs_scale = _scale((2, 2, 1, 1), dtype=scale_dtype)
    rhs_scale = _scale((2, 2, 4, 1), dtype=scale_dtype)

    dimension_numbers = (((3,), (3,)), ((0, 1), (0, 1)))

    result_jit = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        dimension_numbers=dimension_numbers,
    )
    self.assertEqual(result_jit.dtype, jnp.bfloat16)

    result = jax.lax.scaled_dot(
        lhs,
        rhs,
        lhs_scale=lhs_scale,
        rhs_scale=rhs_scale,
        dimension_numbers=dimension_numbers,
    )
    self.assertEqual(result.dtype, jnp.bfloat16)
    self.assertAllClose(result_jit, result)

  @parameterized.product(
      scale_dtype=[jnp.float8_e8m0fnu, jnp.bfloat16],
  )
  def test_broadcast_less_than_32(self, scale_dtype):
    self._should_skip_test(scale_dtype)
    B, M, N, K = 32, 1024, 16, 128
    subchannel_size = 16  # Ratio < 32

    lhs = _e4m3fn((B, M, K))
    rhs = _e4m3fn((B, K, N))
    lhs_scales = _scale((B, M, K // subchannel_size), dtype=scale_dtype)
    rhs_scales = _scale((B, K // subchannel_size, N), dtype=scale_dtype)

    preferred_type = (
        jnp.float32 if scale_dtype == jnp.bfloat16 else jnp.bfloat16
    )
    result = jax.lax.scaled_dot(
        lhs,
        rhs,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales,
        preferred_element_type=preferred_type,
    )

    result_jit = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales,
        preferred_element_type=preferred_type,
    )
    self.assertEqual(result_jit.dtype, preferred_type)
    self.assertAllClose(result, result_jit, atol=1)

  def test_bf16_dot_vs_scaled_dot_numeric_equivalence(self):
    self._should_skip_test(jnp.float8_e8m0fnu)
    B, M, N, K = 32, 256, 16, 512
    x = jnp.abs(self.rng().randn(B, M, K).astype(jnp.bfloat16))
    y = jnp.abs(self.rng().randn(B, K, N).astype(jnp.bfloat16))
    x_fp8, x_scales = _quantize_to_fp8(x)
    y_fp8, y_scales = _quantize_to_fp8(y.transpose((0, 2, 1)))
    y_fp8 = y_fp8.transpose((0, 2, 1))
    y_scales = jnp.transpose(y_scales, (0, 2, 1))
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))

    scaled_dot_result_jit = scaled_dot_fn(
        x_fp8,
        y_fp8,
        lhs_scale=x_scales,
        rhs_scale=y_scales,
        preferred_element_type=jnp.float32,
        dimension_numbers=dimension_numbers,
    )

    @jax.jit
    def bf16_dot(a, b):
      return jax.lax.dot_general(
          a,
          b,
          preferred_element_type=jnp.float32,
          dimension_numbers=dimension_numbers,
      )

    original_dot_result = bf16_dot(x, y)
    self.assertAllClose(
        scaled_dot_result_jit, original_dot_result, atol=1e0, rtol=1e0
    )

  def test_batching_3d_vs_vmap_equivalence(self):
    self._should_skip_test(jnp.float8_e8m0fnu)
    B, M, N, K = 4, 64, 32, 128
    subchannel_size = 32
    lhs = _e4m3fn((B, M, K))
    rhs = _e4m3fn((B, K, N))
    lhs_scales = _e8m0fnu((B, M, K // subchannel_size))
    rhs_scales = _e8m0fnu((B, K // subchannel_size, N))

    def scaled_dot_batched_fn(lhs, rhs, lhs_scales, rhs_scales):
      scaled_dot_2d = partial(
          jax.lax.scaled_dot,
          dimension_numbers=(((1,), (0,)), ((), ())),
          preferred_element_type=jnp.bfloat16,
      )
      return jax.vmap(scaled_dot_2d, in_axes=(0, 0))(
          lhs, rhs, lhs_scale=lhs_scales, rhs_scale=rhs_scales
      )

    result_vmap = scaled_dot_batched_fn(lhs, rhs, lhs_scales, rhs_scales)

    result_batch_dims = jax.lax.scaled_dot(
        lhs,
        rhs,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.bfloat16,
    )
    self.assertAllClose(result_vmap, result_batch_dims, atol=1e-6, rtol=1e-6)

  def test_batching_2d_vs_vmap_equivalence(self):
    self._should_skip_test(jnp.float8_e8m0fnu)
    B, M, N, K = 4, 64, 32, 128
    subchannel_size = 32
    lhs = _e4m3fn((B, M, K))
    rhs = _e4m3fn((K, N))
    lhs_scales = _e8m0fnu((B, M, K // subchannel_size))
    rhs_scales = _e8m0fnu((K // subchannel_size, N))

    def scaled_dot_batched_fn(lhs, rhs, lhs_scales, rhs_scales):
      def scaled_dot_2d(lhs, rhs, lhs_scale, rhs_scale):
        return jax.lax.scaled_dot(
            lhs,
            rhs,
            lhs_scale=lhs_scale,
            rhs_scale=rhs_scale,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16,
        )

      return jax.vmap(scaled_dot_2d, in_axes=(0, None, 0, None))(
          lhs, rhs, lhs_scales, rhs_scales
      )

    result_vmap = scaled_dot_batched_fn(lhs, rhs, lhs_scales, rhs_scales)

    rhs_repeated = jnp.broadcast_to(rhs, (B, K, N))
    rhs_scales_repeated = jnp.broadcast_to(
        rhs_scales, (B, K // subchannel_size, N)
    )
    result_batch_dims = jax.lax.scaled_dot(
        lhs,
        rhs_repeated,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales_repeated,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.bfloat16,
    )
    self.assertAllClose(result_vmap, result_batch_dims, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
