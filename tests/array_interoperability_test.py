# Copyright 2020 The JAX Authors.
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

import unittest
import warnings

from absl.testing import absltest
import numpy as np

import jax
import jax.dlpack
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax._src import config
from jax._src import dlpack as dlpack_src
from jax._src import test_util as jtu
from jax._src.util import cache

config.parse_flags_with_absl()

try:
  import cupy
except ImportError:
  cupy = None

try:
  # TODO(b/470156950): Remove this once a proper fix is in place
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore",
                            category=FutureWarning,
                            message=".*np.object.*")
    import tensorflow as tf

  tf_version = tuple(
    int(x) for x in tf.version.VERSION.split("-")[0].split("."))
except ImportError:
  tf = None


dlpack_dtypes = sorted([dt.type for dt in  dlpack_src.SUPPORTED_DTYPES_SET],
                       key=lambda x: x.__name__)

# These dtypes are not supported by neither NumPy nor TensorFlow, therefore
# we list them separately from ``jax.dlpack.SUPPORTED_DTYPES``.
extra_dlpack_dtypes = [
    jnp.float8_e4m3b11fnuz,
    jnp.float8_e4m3fn,
    jnp.float8_e4m3fnuz,
    jnp.float8_e5m2,
    jnp.float8_e5m2fnuz,
] + [
    dtype
    for name in [
        "float4_e2m1fn",
        "float8_e3m4",
        "float8_e4m3",
        "float8_e8m0fnu",
    ]
    if (dtype := getattr(jnp, name, None))
]

numpy_dtypes = [dt for dt in dlpack_dtypes if dt != jnp.bfloat16]
cuda_array_interface_dtypes = [dt for dt in dlpack_dtypes if dt != jnp.bfloat16]
nonempty_nonscalar_array_shapes = [(4,), (3, 4), (2, 3, 4)]
empty_array_shapes = []
empty_array_shapes += [(0,), (0, 4), (3, 0),]
nonempty_nonscalar_array_shapes += [(3, 1), (1, 4), (2, 1, 4)]

nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
all_shapes = nonempty_array_shapes + empty_array_shapes


def _get_alignment(x: int):
  """Return alignment of x.
  """
  return x & ((~x) + 1)


def _get_alignment_offset(ptr: int, alignment: int):
  """Return minimal positive offset such that
    _get_alignment(ptr + offset) == alignment

  Note that 0 <= offset < 2 * alignment.
  """
  if _get_alignment(ptr) == alignment:
    return 0
  offset = alignment - (ptr & (alignment - 1))
  if _get_alignment(ptr + offset) == alignment:
    return offset
  return offset + alignment


def _ensure_alignment(arr, desired_alignment):
  """Return a copy of numpy array such that its data pointer has the
  desired alignment exactly. The desired alignment must be power of
  two.
  """
  assert desired_alignment > 1, desired_alignment
  buf = np.empty(2 * desired_alignment + arr.nbytes, dtype=np.int8)
  ptr = buf.__array_interface__['data'][0]
  start = _get_alignment_offset(ptr, desired_alignment)
  # if arr.nbytes == 0 and start > 0 then buf[start:start+arr.nbytes]
  # incorrectly returns the original buffer, so we must use
  # buf[start:][:arr.nbytes]:
  new = buf[start:][:arr.nbytes].view(arr.dtype).reshape(arr.shape)
  np.copyto(new, arr, casting='unsafe')
  new_ptr = new.__array_interface__['data'][0]
  assert new_ptr & (desired_alignment - 1) == 0  # sanity check
  assert new_ptr & (desired_alignment * 2 - 1) != 0  # sanity check
  return new


def test_ensure_alignment():

  def reference(ptr, alignment):
    start = 0
    while _get_alignment(ptr + start) != alignment:
      start += 1
    return start

  for alignment in [2, 4, 8, 16, 32, 64, 128]:
    max_start = 1
    for ptr in range(1000):
      start = _get_alignment_offset(ptr, alignment)
      expected = reference(ptr, alignment)
      max_start = max(max_start, start)
      assert start == expected
    assert max_start == alignment * 2 - 1


@cache()
def _get_max_align_bits(dtype, device):
  max_align_bits = 64
  if device.platform == "cpu":
    from jax._src.lib import _jax

    # We determine the max_align_bits value from the error that is
    # raised by dlpack_managed_tensor_to_buffer when using a buffer
    # with a very small data alignment (=2).
    x_np = _ensure_alignment(np.zeros(5, dtype=dtype), desired_alignment=2)
    try:
      _jax.dlpack_managed_tensor_to_buffer(x_np.__dlpack__(), device, None, False)
      raise RuntimeError("unexpected success")
    except Exception as e:
      msg = str(e)
      m = "is not aligned to"
      if m in msg:
        i = msg.index(m) + len(m)
        max_align_bits = int(msg[i:].split(None, 1)[0])
      else:
        raise
  return max_align_bits


class DLPackTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest(f"DLPack not supported on {jtu.device_under_test()}")

  @jtu.sample_product(
      shape=all_shapes,
      dtype=dlpack_dtypes + extra_dlpack_dtypes,
      copy=[False, True, None],
      use_stream=[False, True],
  )
  @jtu.run_on_devices("gpu")
  def testJaxRoundTrip(self, shape, dtype, copy, use_stream):
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)

    # Check if the source device is preserved
    x = jax.device_put(np, jax.devices("cpu")[0])
    device = jax.devices("gpu")[0]
    y = jax.device_put(x, device)
    # TODO(parkers): Remove after setting 'stream' properly below.
    jax.block_until_ready(y)
    z = jax.dlpack.from_dlpack(y)

    self.assertEqual(z.devices(), {device})
    self.assertAllClose(np.astype(x.dtype), z)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=dlpack_dtypes,
    gpu=[False, True],
  )
  def testJaxArrayRoundTrip(self, shape, dtype, gpu):
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    if gpu and jax.default_backend() == "cpu":
      raise unittest.SkipTest("Skipping GPU test case on CPU")
    device = jax.devices("gpu" if gpu else "cpu")[0]
    x = jax.device_put(np, device)
    # TODO(parkers): Remove after setting 'stream' properly.
    jax.block_until_ready(x)
    y = jax.dlpack.from_dlpack(x)
    self.assertEqual(y.devices(), {device})
    self.assertAllClose(np.astype(x.dtype), y)
    # Test we can create multiple arrays
    z = jax.dlpack.from_dlpack(x)
    self.assertEqual(z.devices(), {device})
    self.assertAllClose(np.astype(x.dtype), z)

  @jtu.sample_product(shape=all_shapes, dtype=dlpack_dtypes)
  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testTensorFlowToJax(self, shape, dtype):
    if (not config.enable_x64.value and
        dtype in [jnp.int64, jnp.uint64, jnp.float64]):
      raise self.skipTest("x64 types are disabled by jax_enable_x64")
    if (jtu.test_device_matches(["gpu"]) and
        not tf.config.list_physical_devices("GPU")):
      raise self.skipTest("TensorFlow not configured with GPU support")

    if jtu.test_device_matches(["gpu"]) and dtype == jnp.int32:
      raise self.skipTest("TensorFlow does not place int32 tensors on GPU")

    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    with tf.device("/GPU:0" if jtu.test_device_matches(["gpu"]) else "/CPU:0"):
      x = tf.identity(tf.constant(np))
    y = jax.dlpack.from_dlpack(x)
    self.assertAllClose(np, y)

  @jtu.sample_product(
      shape=all_shapes,
      dtype=dlpack_dtypes,
  )
  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testJaxToTensorFlow(self, shape, dtype):
    if (not config.enable_x64.value and
        dtype in [jnp.int64, jnp.uint64, jnp.float64]):
      self.skipTest("x64 types are disabled by jax_enable_x64")
    if (jtu.test_device_matches(["gpu"]) and
        not tf.config.list_physical_devices("GPU")):
      raise self.skipTest("TensorFlow not configured with GPU support")
    rng = jtu.rand_default(self.rng())
    np = rng(shape, dtype)
    x = jnp.array(np)
    # TODO(parkers): Remove after setting 'stream' properly.
    jax.block_until_ready(x)
    # TODO(b/171320191): this line works around a missing context initialization
    # bug in TensorFlow.
    _ = tf.add(1, 1)
    y = tf.experimental.dlpack.from_dlpack(x.__dlpack__())
    self.assertAllClose(np, y.numpy())

  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testTensorFlowToJaxInt64(self):
    # See https://github.com/jax-ml/jax/issues/11895
    x = jax.dlpack.from_dlpack(tf.ones((2, 3), tf.int64))
    dtype_expected = jnp.int64 if config.enable_x64.value else jnp.int32
    self.assertEqual(x.dtype, dtype_expected)

  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testTensorFlowToJaxNondefaultLayout(self):
    x = tf.transpose(np.arange(4).reshape(2, 2))
    self.assertAllClose(x.numpy(), jax.dlpack.from_dlpack(x))

  @jtu.sample_product(
    shape=all_shapes,
    dtype=numpy_dtypes,
    copy=[False, True, None],
    aligned=[False, True],
  )
  def testNumpyToJax(self, shape, dtype, copy, aligned):
    rng = jtu.rand_default(self.rng())
    x_np = rng(shape, dtype)
    device = jax.devices()[0]

    alignment = _get_max_align_bits(dtype, device) if aligned else 2
    x_np = _ensure_alignment(x_np, desired_alignment=alignment)

    _from_dlpack = lambda: jnp.from_dlpack(x_np, device=device, copy=copy)
    if copy is not None and not copy and (jax.default_backend() != "cpu"
                                          or not aligned):
      self.assertRaisesRegex(
          ValueError, "Specified .* which requires a copy", _from_dlpack
      )
    else:
      self.assertAllClose(x_np, _from_dlpack())

  def testNumpyToJaxNondefaultLayout(self):
    x = np.arange(4).reshape(2, 2).T
    self.assertAllClose(x, jax.dlpack.from_dlpack(x))

  @jtu.sample_product(shape=all_shapes, dtype=numpy_dtypes)
  @jtu.run_on_devices("cpu")  # NumPy only accepts cpu DLPacks
  def testJaxToNumpy(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x_jax = jnp.array(rng(shape, dtype))
    x_np = np.from_dlpack(x_jax)
    self.assertAllClose(x_np, x_jax)


class CudaArrayInterfaceTest(jtu.JaxTestCase):

  @jtu.skip_on_devices("cuda", "rocm")
  def testCudaArrayInterfaceOnNonCudaFails(self):
    x = jnp.arange(5)
    self.assertFalse(hasattr(x, "__cuda_array_interface__"))
    with self.assertRaisesRegex(
        AttributeError,
        "__cuda_array_interface__ is only defined for .*GPU buffers.",
    ):
      _ = x.__cuda_array_interface__

  @jtu.run_on_devices("gpu")
  def testCudaArrayInterfaceOnShardedArrayFails(self):
    devices = jax.local_devices()
    if len(devices) <= 1:
      raise unittest.SkipTest("Test requires 2 or more devices")
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    x = jnp.arange(16)
    x = jax.device_put(x, sharding)
    self.assertFalse(hasattr(x, "__cuda_array_interface__"))
    with self.assertRaisesRegex(
        AttributeError,
        "__cuda_array_interface__ is only supported for unsharded arrays.",
    ):
      _ = x.__cuda_array_interface__


  @jtu.sample_product(
    shape=all_shapes,
    dtype=cuda_array_interface_dtypes,
  )
  @jtu.run_on_devices("gpu")
  def testCudaArrayInterfaceWorks(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    y = jnp.array(x)
    z = np.asarray(y)
    a = y.__cuda_array_interface__
    self.assertEqual(shape, a["shape"])
    self.assertEqual(z.__array_interface__["typestr"], a["typestr"])

  @jtu.run_on_devices("gpu")
  def testCudaArrayInterfaceBfloat16Fails(self):
    rng = jtu.rand_default(self.rng())
    x = rng((2, 2), jnp.bfloat16)
    y = jnp.array(x)
    with self.assertRaisesRegex(AttributeError, ".*not supported for BF16.*"):
      _ = y.__cuda_array_interface__

  @jtu.sample_product(
    shape=all_shapes,
    dtype=cuda_array_interface_dtypes,
  )
  @unittest.skipIf(not cupy, "Test requires CuPy")
  @jtu.run_on_devices("cuda")
  def testJaxToCuPy(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    y = jnp.array(x)
    # TODO(parkers): Remove after setting 'stream' properly.
    jax.block_until_ready(y)
    z = cupy.asarray(y)
    self.assertEqual(y.__cuda_array_interface__["data"][0],
                     z.__cuda_array_interface__["data"][0])
    self.assertAllClose(x, cupy.asnumpy(z))

  @jtu.sample_product(
    shape=all_shapes,
    dtype=jtu.dtypes.supported(cuda_array_interface_dtypes),
  )
  @unittest.skipIf(not cupy, "Test requires CuPy")
  @jtu.run_on_devices("cuda")
  def testCuPyToJax(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    y = cupy.asarray(x)
    z = jnp.array(y, copy=False)  # this conversion uses dlpack protocol
    self.assertEqual(z.dtype, dtype)
    self.assertEqual(y.__cuda_array_interface__["data"][0],
                     z.__cuda_array_interface__["data"][0])
    self.assertAllClose(np.asarray(z), cupy.asnumpy(y))

  @jtu.sample_product(
    shape=all_shapes,
    dtype=jtu.dtypes.supported(cuda_array_interface_dtypes),
  )
  @jtu.run_on_devices("gpu")
  def testCaiToJax(self, shape, dtype):
    dtype = np.dtype(dtype)

    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)

    # using device with highest device_id for testing the correctness
    # of detecting the device id from a pointer value
    device = jax.devices('gpu')[-1]
    with jax.default_device(device):
      y = jnp.array(x, dtype=dtype)
      # TODO(parkers): Remove after setting 'stream' properly below.
      jax.block_until_ready(y)
    self.assertEqual(y.dtype, dtype)

    # Using a jax array CAI provider support to construct an object
    # that implements the CUDA Array Interface, versions 2 and 3.
    cai = y.__cuda_array_interface__
    stream = tuple(y.devices())[0].get_stream_for_external_ready_events()

    class CAIWithoutStridesV2:
      __cuda_array_interface__ = cai.copy()
      __cuda_array_interface__["version"] = 2
      # CAI version 2 may not define strides and does not define stream
      __cuda_array_interface__.pop("strides", None)
      __cuda_array_interface__.pop("stream", None)

    class CAIWithoutStrides:
      __cuda_array_interface__ = cai.copy()
      __cuda_array_interface__["version"] = 3
      __cuda_array_interface__["strides"] = None
      __cuda_array_interface__["stream"] = None  # default stream

    class CAIWithStrides:
      __cuda_array_interface__ = cai.copy()
      __cuda_array_interface__["version"] = 3
      strides = (dtype.itemsize,) if shape else ()
      for s in reversed(shape[1:]):
        strides = (strides[0] * s, *strides)
      __cuda_array_interface__['strides'] = strides
      __cuda_array_interface__["stream"] = stream

    for CAIObject in [CAIWithoutStridesV2, CAIWithoutStrides,
                      CAIWithStrides]:
      z = jnp.array(CAIObject(), copy=False)
      self.assertEqual(y.__cuda_array_interface__["data"][0],
                       z.__cuda_array_interface__["data"][0])
      self.assertAllClose(x, z)
      if 0 in shape:
        # the device id detection from a zero pointer value is not
        # possible
        pass
      else:
        self.assertEqual(y.devices(), z.devices())

      z = jnp.array(CAIObject(), copy=True)
      if 0 not in shape:
        self.assertNotEqual(y.__cuda_array_interface__["data"][0],
                            z.__cuda_array_interface__["data"][0])
      self.assertAllClose(x, z)

class Bfloat16Test(jtu.JaxTestCase):

  @unittest.skipIf((not tf or tf_version < (2, 5, 0)),
                   "Test requires TensorFlow 2.5.0 or newer")
  def testJaxAndTfHaveTheSameBfloat16Type(self):
    self.assertEqual(np.dtype(jnp.bfloat16).num,
                     np.dtype(tf.dtypes.bfloat16.as_numpy_dtype).num)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
