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
from jax._src import test_util as jtu
from jax._src.pallas import utils as pallas_utils
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas import tpu as pltpu
else:
  pltpu = None

import hypothesis as hp
import hypothesis.strategies as hps


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


class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test only supported on TPU.")

    super().setUp()

  @classmethod
  def pallas_call(cls, *args, **kwargs):
    return pl.pallas_call(*args, interpret=cls.INTERPRET, **kwargs)


@jtu.thread_unsafe_test_class()  # hypothesis is not thread safe
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

  @parameterized.parameters([jnp.int32, jnp.int16, jnp.int8, jnp.int4])
  def test_row_broadcast(self, dtype):
    if not jtu.if_cloud_tpu_at_least(2025, 1, 10):
      self.skipTest("Requires libtpu built after 2025-01-10")
    if not self.INTERPRET and jtu.get_tpu_version() < 5:
      self.skipTest("Requires TPUv5+")
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

  def test_tpu_unsigned_int(self):
    self.skipTest("TODO(apaszke): Unsigned upcasts were implemented incorrectly")
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
    if not jtu.if_cloud_tpu_at_least(2025, 1, 22):
      self.skipTest("Requires libtpu built after 2025-01-22")
    shape = (128, 128)
    bitwidth = pallas_utils.dtype_bitwidth(dtype)
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
    if dtype == jnp.int32:
      # TODO(apaszke): Remove after 12 weeks have passed.
      if not jtu.if_cloud_tpu_at_least(2024, 12, 19):
        self.skipTest("Requires libtpu built after 2024-12-19")
      if axis == 2:
        self.skipTest("Int32 reduction on minor is not supported.")
    # TODO(b/384127570): fix bfloat16 reduction.
    if dtype == jnp.bfloat16 and reduce_func != jnp.sum:
      self.skipTest("b/384127570")
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
      msk_dtype=[jnp.float32, jnp.bfloat16, jnp.int8],
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_i1_relayout_with_bitwidth_change(self, msk_dtype, dtype):
    if not jtu.if_cloud_tpu_at_least(2025, 1, 25):
      self.skipTest("Requires libtpu built after 2025-01-25")
    shape = (129, 129)
    msk_bitwidth = pallas_utils.dtype_bitwidth(msk_dtype)
    bitwidth = pallas_utils.dtype_bitwidth(dtype)
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
      target=(jnp.int8,),  # TODO(apaszke): Add int4.
      round=(False, True),
  )
  def test_quantize(self, target, round):
    if not jtu.if_cloud_tpu_at_least(2025, 1, 15):
      self.skipTest("Requires libtpu built after 2025-01-15")
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
    if not jtu.if_cloud_tpu_at_least(2025, 2, 5):
      self.skipTest("Requires libtpu built after 2025-02-05")
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
    if not jtu.if_cloud_tpu_at_least(2025, 2, 13):
      self.skipTest("Requires libtpu built after 2025-02-13")
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Requires TPUv4+")
    kwargs = {}
    if jtu.get_tpu_version() == 6:
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
    if not jtu.if_cloud_tpu_at_least(2025, 2, 19):
      self.skipTest("Requires libtpu built after 2025-02-19")
    bitwidth = pallas_utils.dtype_bitwidth(dtype)
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
    if not jtu.if_cloud_tpu_at_least(2025, 4, 16):
      self.skipTest("Requires libtpu built after 2025-04-16")
    if jtu.get_tpu_version() < 4:
      self.skipTest("Requires TPUv4+")
    src_shape = (8, 128)
    tgt_shape = (16, 256)

    def kernel(src, tgt):
      tgt[:] = pl.load(src, tuple(pl.ds(0, d) for d in tgt.shape))

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

  # TODO(jevinjiang): we need to support strided load for bool.
  @parameterized.product(dtype=_JAX_DTYPES_NO_BOOL)
  @hp.given(
      slice_start=hps.integers(0, 3),
      slice_size=hps.integers(1, 3),
      m=hps.integers(1, 32),
      # Need to make sure the 2nd minor has no padding.
      n=hps.sampled_from([1, 2, 4, 8, 16, 24, 32]),
  )
  @hp.settings(max_examples=20)  # 20 examples for each dtype.
  def test_load_to_reshape(self, dtype, slice_start, slice_size, m, n):
    if not jtu.if_cloud_tpu_at_least(2025, 5, 15):
      self.skipTest("Requires libtpu built after 2025-05-15")
    bitwidth = pallas_utils.dtype_bitwidth(dtype)
    if jtu.get_tpu_version() < 4 and bitwidth != 32:
      self.skipTest("Requires TPUv4+ for non-32-bit types")
    if jtu.get_tpu_version() == 4 and bitwidth <= 8:
      self.skipTest("Int8 is not supported on this target")
    packing = 32 // bitwidth
    n *= packing
    slices = (
        slice(slice_start, slice_start + slice_size),
        slice(slice_start, slice_start + m),
        slice(None),
        slice(None),
    )
    inp_shape = (8, 64, n, 128)
    out_shape = (slice_size, m, n * 128)

    def kernel(inp_ref, out_ref):
      inp = inp_ref[slices]
      out_ref[...] = inp.reshape(out_shape)

    inp = rand(inp_shape, dtype, seed=1234)
    run = pl.pallas_call(kernel, jax.ShapeDtypeStruct(out_shape, dtype))
    output = run(inp)
    expected = inp[slices].reshape(out_shape)
    np.testing.assert_array_equal(output, expected)


@jtu.thread_unsafe_test_class()  # hypothesis is not thread safe
class OpsInterpretTest(OpsTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main()
