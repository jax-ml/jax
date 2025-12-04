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

import functools
import math
import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.pallas import pallas_test_util as ptu
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import tpu as pltpu
else:
  pltpu = None


jax.config.parse_flags_with_absl()
jtu.setup_hypothesis(max_examples=100)

_JAX_DTYPES_NO_BOOL = (
    jnp.float32,
    jnp.bfloat16,
    jnp.int32,
    jnp.int16,
    jnp.int8,
    jnp.int4,
    jnp.float8_e5m2,
)

_JAX_DTYPES = (
    *_JAX_DTYPES_NO_BOOL,
    jnp.bool_,
)

_JAX_INT_DTYPES = (
    jnp.int32,
    jnp.int16,
    jnp.int8,
    jnp.int4,
    jnp.uint32,
    jnp.uint16,
    jnp.uint8,
    jnp.uint4,
)


def rand(
    shape: tuple[int, ...], dtype: np.dtype | jnp.dtype, seed: int = 1234
) -> np.ndarray:
  """A helper function to generate random data for testing."""
  rng = np.random.Generator(np.random.Philox(counter=0, key=seed))
  if jnp.issubdtype(dtype, jnp.floating):
    return rng.normal(size=shape).astype(dtype)
  if jnp.issubdtype(dtype, jnp.integer):
    return rng.integers(
        jnp.iinfo(dtype).min, jnp.iinfo(dtype).max, shape, dtype=np.int32
    ).astype(dtype)
  raise NotImplementedError(f"Unsupported random data generation for {dtype=}")


@jtu.thread_unsafe_test_class(condition=not jtu.hypothesis_is_thread_safe())
class OpsTest(ptu.PallasTPUTest):

  @parameterized.product(
      from_dtype=_JAX_DTYPES,
      to_dtype=_JAX_DTYPES,
      is_ref_bitcast=[False, True],
      use_primitive_io_op=[False, True],
  )
  def test_bitcast(
      self, from_dtype, to_dtype, is_ref_bitcast, use_primitive_io_op
  ):
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Run on TPUv4+ to have expected memory layout")
    if from_dtype == to_dtype:
      self.skipTest("No bitcast needed")
    if from_dtype == jnp.bool_ or to_dtype == jnp.bool_:
      self.skipTest("Bitcasting with bool is not supported")

    def kernel(x_ref, y_ref):
      if is_ref_bitcast:
        if use_primitive_io_op:
          pltpu.store(y_ref, pltpu.load(x_ref.bitcast(to_dtype)))
        else:
          y_ref[...] = x_ref.bitcast(to_dtype)[...]
      else:
        if use_primitive_io_op:
          pltpu.store(y_ref, pltpu.bitcast(pltpu.load(x_ref), to_dtype))
        else:
          y_ref[...] = pltpu.bitcast(x_ref[...], to_dtype)

    m, n = 1, 256
    in_packing = 32 // dtypes.itemsize_bits(from_dtype)
    out_packing = 32 // dtypes.itemsize_bits(to_dtype)
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

  def test_stop_gradient(self):
    def kernel(x_ref, y_ref):
      y_ref[...] = jax.lax.stop_gradient(x_ref[...] + 1)

    x = jnp.arange(1024, dtype=jnp.float32)
    y = pl.pallas_call(kernel, out_shape=x)(x)
    self.assertAllClose(y, x + 1)

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

  @parameterized.parameters([jnp.int32, jnp.int16, jnp.int8, jnp.int4])
  def test_row_broadcast(self, dtype):
    bitwidth = dtypes.itemsize_bits(dtype)
    if not self.INTERPRET and jtu.get_tpu_version() < 4 and bitwidth < 8:
      self.skipTest("Requires TPUv4+ for sub-byte types")
    def kernel(x_ref, y_ref):
      y_ref[...] = jnp.broadcast_to(x_ref[pl.ds(3, 1)], y_ref.shape).astype(y_ref.dtype)
    m, n = 4, 1152
    x = jax.random.randint(
        jax.random.key(12), (m, n), minval=-1000, maxval=1000, dtype=jnp.int32
    ).astype(dtype)
    y = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((m, n), jnp.int32)
    )(x)
    np.testing.assert_array_equal(y, jnp.broadcast_to(x[3:4], y.shape))

  @parameterized.parameters([jnp.uint4, jnp.int4])
  def test_tpu_int4_upcast_and_matmul(self, dtype):
    if not jtu.is_device_tpu_at_least(version=5):
      self.skipTest("TPUv5+ needed for integer matmuls")

    def body(x_ref, o_ref):
      # Test cast from (u)int4 -> int8
      ux = lax.convert_element_type(x_ref[...], jnp.int8)
      o_ref[...] = jax.lax.dot(ux, ux, preferred_element_type=jnp.int32)

    out = jax.ShapeDtypeStruct((128, 128), jnp.int32)
    x = jnp.arange(128 * 128, dtype=dtype).reshape((128, 128))
    result = self.pallas_call(body, out_shape=out)(x)
    np.testing.assert_array_equal(
        result,
        jax.lax.dot(
            x.astype(jnp.int8),
            x.astype(jnp.int8),
            preferred_element_type=jnp.int32,
        ),
    )

  def test_sum_of_two_matmuls(self):
    if not jtu.if_cloud_tpu_at_least(2025, 11, 15):
      self.skipTest("Test requires libtpu from 2025/11/15 or later")
    if not jtu.is_device_tpu_at_least(version=5):
      self.skipTest("Test requires TPUv5+")

    M, K = 8, 8
    k1, k2, k3, k4 = jax.random.split(jax.random.key(42), 4)
    a_val = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    b_val = jax.random.normal(k2, (K,), dtype=jnp.float32)
    c_val = jax.random.normal(k3, (M, K), dtype=jnp.float32)
    d_val = jax.random.normal(k4, (K,), dtype=jnp.float32)

    def kernel(a_ref, b_ref, c_ref, d_ref, o_ref):
      a = a_ref[:]
      b = b_ref[:]
      c = c_ref[:]
      d = d_ref[:]
      res1 = jnp.dot(a, b)
      res2 = jnp.dot(c, d)

      o_ref[:] = res1 + res2

    @jax.jit
    def pallas_fn(a, b, c, d):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((M,), np.float32),
          grid=(1,),
      )(a, b, c, d)

    result_pallas = pallas_fn(a_val, b_val, c_val, d_val)
    expected = jnp.dot(a_val, b_val) + jnp.dot(c_val, d_val)
    self.assertAllClose(result_pallas, expected, atol=1e-5, rtol=1e-5)

  @parameterized.product(from_dtype=_JAX_INT_DTYPES,
                         to_dtype=_JAX_INT_DTYPES)
  def test_integer_cast(self, from_dtype, to_dtype):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Expect TPUv4+")
    # Generate both low and high values to better cover the entire range
    # of the source dtype.
    min_val = from_dtype(jnp.iinfo(from_dtype).min)
    max_val = from_dtype(jnp.iinfo(from_dtype).max)
    if jnp.iinfo(from_dtype).bits > 4:
      x_random = jax.random.randint(jax.random.key(0), shape=(112, 256),
                                    minval=min_val, maxval=max_val, dtype=from_dtype)
    else:
      # randint does not support sub-byte types.
      x_random = jnp.arange(112 * 256, dtype=from_dtype).reshape((112, 256))
    arange = jnp.arange(8 * 256, dtype=from_dtype).reshape((8, 256))
    x = jnp.concatenate([min_val + arange, x_random, max_val - arange], axis=0)

    def body(x_ref, o_ref):
      o_ref[...] = lax.convert_element_type(x_ref[...], to_dtype)

    out = jax.ShapeDtypeStruct(x.shape, to_dtype)
    expected = x.astype(to_dtype)
    result = self.pallas_call(body, out_shape=out)(x)
    np.testing.assert_array_equal(result, expected)

  def test_select_with_scalar_condition(self):
    def kernel(cond, lhs, rhs, out):
      out[:] = jax.lax.select(cond[0] != 0, lhs[:], rhs[:])

    def run(cond, lhs, rhs):
      return self.pallas_call(
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

  @parameterized.product(dtype=[jnp.float32, jnp.bfloat16, jnp.int16, jnp.int8])
  def test_cast_vector_to_mask(self, dtype):
    shape = (128, 128)
    bitwidth = dtypes.itemsize_bits(dtype)
    if jtu.get_tpu_version() < 5 and bitwidth < 32:
      self.skipTest(
          f"Not implemented: cast vector to mask with bitwidth == {bitwidth}"
      )

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
    )
    def kernel(x_ref, mask_ref, o_ref):
      zeros = jnp.zeros_like(x_ref)
      o_ref[...] = jnp.where(mask_ref[...], x_ref[...], zeros)

    mask = jax.random.bernoulli(jax.random.key(1234), 0.5, shape).astype(dtype)
    x = jnp.arange(np.prod(shape), dtype=dtype).reshape(shape) + 1

    out = kernel(x, mask)
    expected = jnp.where(mask, x, jnp.zeros_like(x))
    self.assertArraysEqual(out, expected)

  @parameterized.product(
      dtype = [jnp.float32, jnp.bfloat16, jnp.int32],
      axis = [0, 1, 2],
      reduce_func = [jnp.sum, jnp.max, jnp.min]
  )
  def test_reduction(self, dtype, axis, reduce_func):
    in_shape = (2, 16, 128)
    out_shape = list(in_shape)
    out_shape[axis] = 1

    def kernel(x, out):
      out[:] = reduce_func(x[:], axis, keepdims=True)

    x = jnp.arange(np.prod(in_shape), dtype=dtype).reshape(in_shape)
    result = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
    )(x)
    expected = reduce_func(x, axis, keepdims=True)
    np.testing.assert_array_equal(result, expected)

  @parameterized.product(
      axis=[0, 1, 2],
      in_shape=[(2, 29, 206), (12, 28), (12,)],
      reduce_func = [jnp.argmax, jnp.argmin],
      keepdims=[False, True],
  )
  def test_reduce_index(self, axis, in_shape, reduce_func, keepdims):
    if not keepdims and not jtu.if_cloud_tpu_at_least(2025, 11, 24):
      self.skipTest("Requires libtpu built after 2025-11-24")
    dtype = jnp.float32
    rank = len(in_shape)
    if axis >= rank:
      self.skipTest("Requires axis < rank")
    if axis == rank - 1:
      if keepdims and not jtu.is_device_tpu_at_least(version=4):
        self.skipTest("Requires TPUv4+ for axis=rank-1 and keepdims=True")
      if not keepdims and not jtu.is_device_tpu_at_least(version=5):
        self.skipTest("Requires TPUv5+ for axis=rank-1 and keepdims=False")
    if rank == 1 and not keepdims:
      self.skipTest("Scalar output not supported")

    out_shape = list(in_shape)
    if keepdims:
      out_shape[axis] = 1
    else:
      del out_shape[axis]

    def kernel(x, out):
      out[:] = reduce_func(x[:], axis, keepdims=keepdims)

    x = jax.random.permutation(
        jax.random.key(22),
        jnp.arange(np.prod(in_shape), dtype=dtype)
    ).reshape(in_shape)
    result = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.int32),
    )(x)
    expected = reduce_func(x, axis, keepdims=keepdims)
    np.testing.assert_array_equal(result, expected)

  @parameterized.product(
      shape=[(129, 129), (1, 129), (2, 129), (4, 129)],
      msk_dtype=[jnp.float32, jnp.bfloat16, jnp.int8],
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_i1_relayout_bw(self, shape, msk_dtype, dtype):
    if shape[0] < 8 and not jtu.if_cloud_tpu_at_least(2025, 11, 9):
      self.skipTest("Requires libtpu built after 2025-11-09")
    msk_bitwidth = dtypes.itemsize_bits(msk_dtype)
    bitwidth = dtypes.itemsize_bits(dtype)
    if jtu.get_tpu_version() < 5 and msk_bitwidth < 32:
      self.skipTest(
          "Not implemented: cast vector to mask with bitwidth =="
          f" {msk_bitwidth}"
      )
    if jtu.get_tpu_version() < 5 and bitwidth < 32:
      self.skipTest(f"Not implemented: comparison with bitwidth == {bitwidth}")

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
    )
    def kernel(x_ref, mask_ref, o_ref):
      zeros = jnp.zeros_like(x_ref)
      o_ref[...] = jnp.where(mask_ref[...], x_ref[...], zeros)

    mask = jax.random.bernoulli(jax.random.key(1234), 0.5, shape).astype(
        msk_dtype
    )
    x = jnp.arange(np.prod(shape), dtype=dtype).reshape(shape) + 1

    out = kernel(x, mask)
    expected = jnp.where(mask, x, jnp.zeros_like(x))
    self.assertArraysEqual(out, expected)

  @parameterized.product(
      msk_dtype=[jnp.float32, jnp.bfloat16, jnp.int8],
      dtype=[jnp.float32, jnp.bfloat16, jnp.int8],
  )
  def test_i1_relayout_bw_tiling(self, msk_dtype, dtype):
    self.skipTest("TODO: jevinjiang - Enable once presubmits pass.")
    if not jtu.if_cloud_tpu_at_least(2025, 10, 7):
      self.skipTest("Requires libtpu built after 2025-10-07")
    shape = (256, 256)
    bitwidth = dtypes.itemsize_bits(dtype)
    msk_bitwidth = dtypes.itemsize_bits(msk_dtype)
    msk_packing = 32 // msk_bitwidth
    if jtu.get_tpu_version() < 5 and msk_bitwidth < 32:
      self.skipTest(
          "Not implemented: cast vector to mask with bitwidth =="
          f" {msk_bitwidth}"
      )
    if jtu.get_tpu_version() < 5 and bitwidth < 32:
      self.skipTest(f"Not implemented: comparison with bitwidth == {bitwidth}")

    # Creating large tiling for masks by passing i32 vector first and
    # then bitcast to msk_dtype so the tiling is also bitcasted from
    #  T(8, 128) to T(8 * msk_packing, 128).
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
    )
    def kernel(x_ref, msk_ref, o_ref):
      zeros = jnp.zeros_like(x_ref)
      msk = pltpu.bitcast(msk_ref[...], msk_dtype)
      o_ref[...] = jnp.where(msk, x_ref[...], zeros)

    mask = jax.random.bernoulli(jax.random.key(1234), 0.5, shape).astype(
        msk_dtype
    )
    if msk_bitwidth < 32:
      mask_for_bitcast = mask.reshape(
          shape[0] // msk_packing, msk_packing, shape[1]
      ).swapaxes(-1, -2)
    else:
      mask_for_bitcast = mask

    mask_i32 = lax.bitcast_convert_type(
        mask_for_bitcast,
        jnp.int32,
    )
    x = jnp.arange(np.prod(shape), dtype=dtype).reshape(shape) + 1

    out = kernel(x, mask_i32)
    expected = jnp.where(mask, x, jnp.zeros_like(x))
    self.assertArraysEqual(out, expected)

  @parameterized.product(
      target=(jnp.int8,),  # TODO(apaszke): Add int4.
      round=(False, True),
  )
  def test_quantize(self, target, round):
    if not jtu.is_device_tpu_at_least(version=6):
      self.skipTest("Requires TPUv6+")
    shape = (256, 256)
    # NOTE: 256 * 256 == 2 ** 16, so those are all bf16 values.
    x = lax.bitcast_convert_type(
        np.arange(math.prod(shape), dtype=jnp.uint16).reshape(shape),
        jnp.bfloat16,
    )

    round_fn = jnp.rint if round else lambda x: x

    def kernel(x_ref, o_ref):
      o_ref[...] = round_fn(x_ref[...]).astype(target)
    out = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(shape, target)
    )(x)

    ref = jax.jit(lambda x: round_fn(x).astype(target))(x)
    np.testing.assert_array_equal(out, ref)

  @parameterized.product(axis=[0, 1], mode=["promise_in_bounds", None])
  def test_dynamic_gather_along_axis(self, axis, mode):
    if (axis == 0 and not jtu.is_device_tpu_at_least(version=5)) or (
        axis == 1 and not jtu.is_device_tpu_at_least(version=4)
    ):
      self.skipTest("Requires TPUv5+ for axis=0 and TPUv4+ for axis=1")
    dtype = jnp.int32
    shape = (8, 128)

    def kernel(x, indices, out):
      out[...] = jnp.take_along_axis(x[...], indices[...], axis, mode=mode)

    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    idx = jax.random.randint(
        key=jax.random.key(1234),
        shape=shape,
        minval=0,
        maxval=shape[axis],
        dtype=jnp.int32,
    )
    actual = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(shape, dtype)
    )(x, idx)
    expected = np.take_along_axis(x, idx, axis=axis)
    np.testing.assert_array_equal(actual, expected)

  @parameterized.product(dtype=[jnp.float32, jnp.bfloat16])
  def test_float_div(self, dtype):
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Requires TPUv4+")
    kwargs = {}
    if jtu.is_device_tpu_at_least(version=6):
      kwargs.update(dict(rtol=1e-2))
    def kernel(x, y, out):
      out[:] = jax.lax.div(x[:], y[:])

    run = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), dtype),
    )
    k1, k2 = jax.random.split(jax.random.key(1234), 2)
    x = jax.random.normal(k1, (8, 128), dtype=dtype)
    y = jax.random.normal(k2, (8, 128), dtype=dtype)
    np.testing.assert_allclose(run(x, y), jax.lax.div(x, y), **kwargs)

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16, jnp.int8],
  )
  def test_concat_mask(self, dtype):
    bitwidth = dtypes.itemsize_bits(dtype)
    if jtu.get_tpu_version() < 5 and bitwidth < 32:
      self.skipTest(
          f"Not implemented: cast vector to mask with bitwidth == {bitwidth}"
      )
    shape = (128, 128)

    def kernel(x, out):
      mask = x[...] != 0
      concated_mask = jnp.concatenate([mask, mask], axis=0)
      concated_x = jnp.concatenate([x[:], x[:]], axis=0)
      out[:] = lax.select(concated_mask, concated_x, jnp.zeros_like(concated_x))

    x = jax.random.normal(jax.random.key(1234), shape, dtype=jnp.float32)
    if dtype == jnp.int8:
      x = (x * 100).astype(jnp.int8)
    else:
      x = x.astype(dtype)
    out = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((shape[0] * 2, shape[1]), dtype)
    )(x)
    concated_mask = jnp.concatenate([x != 0, x != 0], axis=0)
    concated_x = jnp.concatenate([x, x], axis=0)
    expected = lax.select(concated_mask, concated_x, jnp.zeros_like(concated_x))
    np.testing.assert_array_equal(out, expected)

  def test_reduce_with_const(self):
    m = 1
    d = 1024
    x = jnp.ones((m, d), jnp.bfloat16)

    def dot(x, y):
      return jax.lax.dot_general(
          x,
          y,
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    def kernel(x, out):
      out[:] = dot(x[:], jnp.ones((1, d), jnp.bfloat16))

    run = pl.pallas_call(kernel, jax.ShapeDtypeStruct((m, 1), jnp.float32))
    output = run(x)
    expected = dot(x[:], jnp.ones((1, d), jnp.bfloat16))
    np.testing.assert_array_equal(output, expected)

  # We need to manually run the test with the env variable
  # `export LIBTPU_INIT_ARGS="--xla_jf_bounds_check=true"`
  def test_disable_bounds_check(self):
    if jtu.get_tpu_version() < 4:
      self.skipTest("Requires TPUv4+")
    src_shape = (8, 128)
    tgt_shape = (16, 256)

    def kernel(src, tgt):
      tgt[:] = src[tuple(pl.ds(d) for d in tgt.shape)]

    x = jnp.arange(np.prod(src_shape), dtype=jnp.float32).reshape(src_shape)
    run = pl.pallas_call(
        kernel,
        jax.ShapeDtypeStruct(tgt_shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )
    output = run(x)
    np.testing.assert_array_equal(
        output[tuple(slice(0, d) for d in src_shape)], x
    )

  def test_while_loop_arg_num_change(self):
    # This kernel will generate a while loop that will be CSEd by MLIR to have
    # the different number of argments in before region and after region.
    def kernel(
        out_ref,
        a,
    ):
      def loop_cond(state):
        _, y = state
        return y

      def loop_body(state):
        x, y = state

        def then_0():
          def then_1():
            return jnp.int32(0)

          def else_1():
            a[0] = a[0] + 1

            return jnp.int32(1)

          z = lax.cond(x == 0, then_1, else_1)
          new_x = z
          new_y = z != 0
          return new_x, new_y

        def else_0():
          return x, jnp.bool_(False)

        new_x, new_y = lax.cond(y, then_0, else_0)

        return (new_x, new_y)

      out_ref[0] = lax.while_loop(
          loop_cond, loop_body, (jnp.int32(0), jnp.bool_(True))
      )[0]

    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        in_specs=(),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        scratch_shapes=(pltpu.SMEM((1,), jnp.int32),),
    )()[0]
    self.assertEqual(output, 0)

  def test_produce_predicate_phi(self):
    def kernel(
        out_ref,
        a,
    ):
      def loop_cond(state):
        x, y = state
        return jnp.logical_or(y, (x == 1))

      def loop_body(state):
        x, y = state

        def then_0():
          def then_1():
            return jnp.int32(0)

          def else_1():
            a[0] = a[0] + 1

            return jnp.int32(1)

          z = lax.cond(x == 0, then_1, else_1)
          new_x = z
          new_y = z != 0
          return new_x, new_y

        def else_0():
          return x, jnp.bool_(False)

        new_x, new_y = lax.cond(y, then_0, else_0)

        return (new_x, new_y)

      out_ref[0] = lax.while_loop(
          loop_cond, loop_body, (jnp.int32(0), jnp.bool_(True))
      )[0]

    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        in_specs=(),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        scratch_shapes=(pltpu.SMEM((1,), jnp.int32),),
    )()[0]
    self.assertEqual(output, 0)

  def test_retiling_with_replicated_lane(self):
    if not jtu.if_cloud_tpu_at_least(2025, 11, 5):
      self.skipTest("Test requires libtpu from 2025/11/5 or later")
    shape = (32, 1)
    broadcast_shape = (32, 256)

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 4, 256), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.broadcast_to(
          x_ref[...], broadcast_shape
      ).reshape(o_ref.shape)

    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    out = kernel(x).reshape(broadcast_shape)
    expected = jnp.broadcast_to(x, broadcast_shape)
    np.testing.assert_array_equal(out, expected)

  @parameterized.parameters(
      [jnp.bfloat16, jnp.float8_e5m2, jnp.float8_e4m3fn, jnp.float8_e4m3b11fnuz]
  )
  def test_stochastic_round(self, target_dtype):
    if not jtu.is_device_tpu_at_least(version=5):
      self.skipTest("Requires TPU v5+")
    if not jtu.if_cloud_tpu_at_least(2025, 10, 29):
      self.skipTest("Test requires libtpu from 2025/10/29 or later")

    def kernel(x_ref, b_ref, o_ref):
      o_ref[...] = pltpu.stochastic_round(
          x_ref[...], b_ref[...], target_dtype=target_dtype
      )

    shape = (8, 128)
    k1, k2 = jax.random.split(jax.random.key(4242), 2)
    x = jax.random.normal(k1, shape, dtype=jnp.float32)
    bits = jax.random.bits(k2, shape, dtype=jnp.uint32)
    x_cast = x.astype(target_dtype)
    x_cast_as_f32 = x_cast.astype(jnp.float32)
    max_val = jnp.finfo(target_dtype).max
    min_val = jnp.finfo(target_dtype).min
    lower = jnp.where(x_cast_as_f32 > x, jnp.nextafter(x_cast, min_val), x_cast)
    upper = jnp.where(x_cast_as_f32 < x, jnp.nextafter(x_cast, max_val), x_cast)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, target_dtype),
    )(x, bits)

    int_dtype = getattr(jnp, f"uint{dtypes.itemsize_bits(target_dtype)}")
    is_correct_bitwise = (
        (result.view(int_dtype) == lower.view(int_dtype)) |
        (result.view(int_dtype) == upper.view(int_dtype))
    )
    is_correct = jnp.where(
        jnp.isnan(x_cast), jnp.isnan(result), is_correct_bitwise
    )
    self.assertTrue(jnp.all(is_correct))

  def _pack_unpack_elementwise_test_data(
      self, shape, unpacked_dtype, packed_dtype):
    """Generates data for test_pack_elementwise and test_unpack_elementwise."""
    bitwidth = dtypes.itemsize_bits(packed_dtype)
    num_sources = 32 // bitwidth
    if unpacked_dtype == jnp.int32:
      stacked_sources = jax.random.randint(
          jax.random.key(0),
          (num_sources, *shape),
          minval=-1000,
          maxval=1000,
          dtype=unpacked_dtype,
      )
    else:
      stacked_sources = jax.random.uniform(
          jax.random.key(0), (num_sources, *shape), dtype=unpacked_dtype
      )
    stacked_results = (
        stacked_sources.astype(packed_dtype)
        .view(getattr(jnp, f"uint{bitwidth}"))
        .astype(jnp.uint32)
    )
    shifts = jnp.arange(num_sources, dtype=jnp.uint32) * bitwidth
    shifts = jnp.expand_dims(shifts, axis=tuple(range(1, stacked_results.ndim)))
    packed_data = jnp.bitwise_or.reduce(stacked_results << shifts, axis=0)
    return stacked_sources, packed_data

  @parameterized.product(
      config=[
          (jnp.float32, jnp.bfloat16),
          (jnp.int32, jnp.int16),
          (jnp.int32, jnp.int8),
          (jnp.int32, jnp.int4),
      ],
      shape=[(8, 128), (2, 15, 300)],
  )
  def test_pack_elementwise(self, config, shape):
    unpacked_dtype, packed_dtype = config
    if not jtu.is_device_tpu_at_least(version=5):
      self.skipTest("Requires TPU v5+")
    if not jtu.if_cloud_tpu_at_least(2025, 11, 7):
      self.skipTest("Test requires libtpu from 2025/11/7 or later")

    bitwidth = dtypes.itemsize_bits(packed_dtype)
    num_sources = 32 // bitwidth

    def kernel(xs_ref, o_ref):
      xs = [xs_ref[i] for i in range(num_sources)]
      o_ref[...] = pltpu.pack_elementwise(xs, packed_dtype=packed_dtype)

    stacked_sources, expected = self._pack_unpack_elementwise_test_data(
        shape, unpacked_dtype, packed_dtype
    )

    result = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.uint32),
    )(stacked_sources)

    np.testing.assert_array_equal(result, expected)

  @parameterized.product(
      config=[
          (jnp.float32, jnp.bfloat16),
          (jnp.int32, jnp.int16),
          (jnp.int32, jnp.int8),
          (jnp.int32, jnp.int4),
      ],
      index=[0, 1, 3],
      shape=[(8, 128), (2, 15, 300)],
  )
  def test_unpack_elementwise(self, config, index, shape):
    unpacked_dtype, packed_dtype = config
    if not jtu.is_device_tpu_at_least(version=5):
      self.skipTest("Requires TPU v5+")
    if not jtu.if_cloud_tpu_at_least(2025, 11, 7):
      self.skipTest("Test requires libtpu from 2025/11/7 or later")

    bitwidth = dtypes.itemsize_bits(packed_dtype)
    packing_factor = 32 // bitwidth

    if index >= packing_factor:
      self.skipTest(
          f"Index {index} out of bounds for packing factor {packing_factor}")

    def kernel(x_ref, o_ref):
      o_ref[...] = pltpu.unpack_elementwise(
          x_ref[...], index=index,
          packed_dtype=packed_dtype, unpacked_dtype=unpacked_dtype
      )

    sources, packed = self._pack_unpack_elementwise_test_data(
        shape, unpacked_dtype, packed_dtype
    )
    expected = sources[index].astype(packed_dtype).astype(unpacked_dtype)

    result = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, unpacked_dtype),
    )(packed)

    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
  absltest.main()
