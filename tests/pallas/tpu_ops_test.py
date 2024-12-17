# Copyright 2024 The JAX Authors.
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
"""Tests for TPU specific operations within pallas_call."""

import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src.pallas import utils as pallas_utils
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import tpu as pltpu
else:
  pltpu = None

try:
  import hypothesis as hp
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("tests depend on hypothesis library")

import hypothesis.strategies as hps

jax.config.parse_flags_with_absl()
jtu.setup_hypothesis(max_examples=100)

_JAX_DTYPES = (
    jnp.float32,
    jnp.bfloat16,
    jnp.int32,
    jnp.int16,
    jnp.int8,
    jnp.bool_,
)


class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test only supported on TPU.")

    super().setUp()

  @classmethod
  def pallas_call(cls, *args, **kwargs):
    return pl.pallas_call(*args, interpret=cls.INTERPRET, **kwargs)


class OpsTest(PallasBaseTest):

  @parameterized.product(
      from_dtype=_JAX_DTYPES, to_dtype=_JAX_DTYPES, is_ref_bitcast=[False, True]
  )
  def test_bitcast(self, from_dtype, to_dtype, is_ref_bitcast):
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Run on TPUv4+ to have expected memory layout")
    if from_dtype == to_dtype:
      self.skipTest("No bitcast needed")
    if from_dtype == jnp.bool_ or to_dtype == jnp.bool_:
      self.skipTest("Bitcasting with bool is not supported")

    def kernel(x_ref, y_ref):
      if is_ref_bitcast:
        y_ref[...] = x_ref.bitcast(to_dtype)[...]
      else:
        y_ref[...] = pltpu.bitcast(x_ref[...], to_dtype)

    m, n = 1, 256
    in_packing = 32 // pallas_utils.dtype_bitwidth(from_dtype)
    out_packing = 32 // pallas_utils.dtype_bitwidth(to_dtype)
    in_shape = (m * in_packing, n)
    out_shape = (m * out_packing, n)
    inp = np.arange(np.prod(in_shape), dtype=from_dtype).reshape(in_shape)
    out = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, to_dtype),
    )(inp)
    if not self.INTERPRET:
      out_interpret = pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_shape, to_dtype),
          interpret=True,
      )(inp)
      self.assertAllClose(out, out_interpret)

  @parameterized.product(is_dynamic=(False, True))
  @hp.given(
      axis=hps.integers(0, 3),
      shift=hps.integers(0, 3),
      stride=hps.one_of(hps.just(None), hps.integers(0, 2)),
      # Stride dimension on the minor most is not supported.
      stride_axis=hps.one_of(hps.just(None), hps.integers(0, 2)),
  )
  @hp.example(3, 9, 1, 2)
  @hp.example(3, 9, 2, 2)
  @hp.example(0, 9, 0, 1)
  @hp.example(0, 9, 1, 1)
  def test_roll(self, is_dynamic, axis, shift, stride, stride_axis):
    if (stride is None) != (stride_axis is None):
      self.skipTest(
          "Roll op requires both stride and stride_axis to be either specified"
          " or not specified."
      )
    if (not jtu.is_device_tpu(version=5)) and stride_axis == 2:
      self.skipTest(
          "Roll op with stride axis on 2nd minor requires at least TPU v5"
      )
    shape = (4, 4, 32, 512)

    def kernel(s_ref, x_ref, y_ref):
      amt = s_ref[0] if is_dynamic else shift
      y_ref[...] = pltpu.roll(
          x_ref[...], amt, axis, stride=stride, stride_axis=stride_axis
      )

    def roll(x, shift, axis, stride=None, stride_axis=None):
      assert (stride is None) == (stride_axis is None)
      if stride is None:
        return np.roll(x, shift, axis)
      outputs = [
          np.roll(xs, shift + i * stride, axis)
          for i, xs in enumerate(np.split(x, x.shape[stride_axis], stride_axis))
      ]
      return np.concatenate(outputs, stride_axis)

    inp = np.arange(np.prod(shape), dtype=jnp.int32).reshape(shape)
    ref = roll(inp, shift, axis, stride, stride_axis)
    dynamic_shift = jnp.array([abs(shift)], jnp.int32)
    for interpret in [False, True]:
      out = pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(shape, jnp.int32),
          grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1),
          interpret=interpret,
      )(dynamic_shift, inp)
      np.testing.assert_array_equal(out, ref, err_msg=f"{interpret=}")

  def test_interleave_vectors(self):
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Expect TPUv4+")

    def kernel(x_ref, y_ref, out_ref):
      x = pltpu.bitcast(x_ref[...].astype(jnp.float32), jnp.int32)
      y = pltpu.bitcast(y_ref[...].astype(jnp.float32), jnp.int32)
      shift = jax.lax.broadcast(16, x.shape)
      out_ref[...] = pltpu.bitcast(
          y | jax.lax.shift_right_logical(x, shift), jnp.bfloat16
      )

    m, n = 16, 128
    inp = np.arange(m * n * 2, dtype=jnp.bfloat16).reshape(m, n * 2)
    x, y = np.split(inp, 2, axis=1)
    out = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m * 2, n), jnp.bfloat16),
    )(x, y)
    np.testing.assert_array_equal(out, inp.reshape(m * 2, n))

  def test_tpu_unsigned_int(self):
    def body(x_ref, o_ref):
      # Test cast from uint16 -> uint32
      ux = lax.convert_element_type(x_ref[...], jnp.uint32)
      res = ux + 1
      # Test cast from uint32 -> float32
      o_ref[...] = res.astype(jnp.float32)
    out = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    x = jnp.arange(8 * 128, dtype=jnp.uint16).reshape((8, 128))
    result = self.pallas_call(body, out_shape=out)(x)
    np.testing.assert_array_equal(result, x.astype(jnp.float32) + 1.0)

  def test_tpu_signed_int_upcast(self):
    if not jtu.is_device_tpu_at_least(version=5):
      self.skipTest("TPUv5+ needed for integer matmuls")

    def body(x_ref, o_ref):
      # Test cast from int4 -> int8
      ux = lax.convert_element_type(x_ref[...], jnp.int8)
      o_ref[...] = jax.lax.dot(ux, ux, preferred_element_type=jnp.int32)

    out = jax.ShapeDtypeStruct((128, 128), jnp.int32)
    x = jnp.arange(128 * 128, dtype=jnp.int4).reshape((128, 128))
    result = self.pallas_call(body, out_shape=out)(x)
    np.testing.assert_array_equal(
        result,
        jax.lax.dot(
            x.astype(jnp.int8),
            x.astype(jnp.int8),
            preferred_element_type=jnp.int32,
        ),
    )

  def test_select_with_scalar_condition(self):
    def kernel(cond, lhs, rhs, out):
      out[:] = jax.lax.select(cond[0] != 0, lhs[:], rhs[:])

    def run(cond, lhs, rhs):
      return pl.pallas_call(
          kernel,
          out_shape=lhs,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[
                  pl.BlockSpec(memory_space=pltpu.SMEM),
                  pl.BlockSpec(memory_space=pltpu.VMEM),
                  pl.BlockSpec(memory_space=pltpu.VMEM),
              ],
          ),
          name="select_kernel",
      )(cond, lhs, rhs)

    cond = jnp.array([1], dtype=jnp.int32)
    lhs = jnp.zeros((8, 128), dtype=jnp.float32)
    rhs = jnp.ones((8, 128), dtype=jnp.float32)

    assert (run(cond, lhs, rhs) == lhs).all()

  def test_logical_and_relayouted_mask(self):
    def get_mask(x_ref):
      x = x_ref[...] == 1
      iota = jax.lax.broadcasted_iota(jnp.int32, x_ref.shape, 1)
      iota = iota > 7
      return jnp.logical_and(x, iota)

    def body(x_ref, y_ref):
      y_ref[...] = jnp.where(get_mask(x_ref), 0.0, -1.0)

    shape = (2, 512)
    out = jax.ShapeDtypeStruct(shape, jnp.float32)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(shape)
    result = self.pallas_call(body, out_shape=out)(x)
    expected = jnp.ones(x.shape, dtype=jnp.float32)
    expected = expected.at[...].set(jnp.where(get_mask(x), 0.0, -1.0))
    np.testing.assert_array_equal(result, expected)


class OpsInterpretTest(OpsTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main()
