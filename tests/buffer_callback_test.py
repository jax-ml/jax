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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version
from jax.experimental import buffer_callback

jax.config.parse_flags_with_absl()


class BufferCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jaxlib_extension_version < 334:
      self.skipTest(
          "Requires a version of jaxlib with buffer callback support."
      )
    if jtu.test_device_matches(["tpu"]):
      self.skipTest("Not supported on TPU.")

  @parameterized.parameters(jtu.dtypes.all)
  @jtu.run_on_devices("cpu")
  def test_numpy(self, dtype):
    def callback(ctx, out, arg):
      with self.assertRaisesRegex(
          jax.errors.JaxRuntimeError, "XLA FFI GPU context is not available"
      ):
        ctx.stream

      self.assertEqual(ctx.stage, buffer_callback.ExecutionStage.EXECUTE)
      self.assertEqual(arg.shape, shape)
      self.assertEqual(arg.dtype, dtype)
      self.assertEqual(out.shape, shape)
      self.assertEqual(out.dtype, dtype)

      self.assertFalse(arg.writeable)
      self.assertTrue(out.writeable)

      x = np.asarray(arg)
      self.assertArraysEqual(x, data)

      y = np.asarray(out)
      self.assertEqual(x.dtype, y.dtype)
      self.assertEqual(x.shape, y.shape)
      y[...] = x

    rng = jtu.rand_default(self.rng())
    shape = (3, 4)
    data = rng(shape, dtype)
    fun = buffer_callback.buffer_callback(
        callback, jax.ShapeDtypeStruct(data.shape, data.dtype)
    )
    self.assertArraysEqual(fun(data), data)

  @parameterized.parameters(jtu.dtypes.all)
  @jtu.run_on_devices("cpu")
  def test_dlpack(self, dtype):
    if dtype == jnp.bfloat16:
      self.skipTest("Numpy's DLPack implementation does not support bfloat16")

    def callback(ctx, out, arg):
      del ctx  # unused

      x = np.from_dlpack(arg)
      self.assertArraysEqual(x, data)

      y = np.from_dlpack(out)
      self.assertEqual(x.dtype, y.dtype)
      self.assertEqual(x.shape, y.shape)

    rng = jtu.rand_default(self.rng())
    shape = (3, 4)
    data = rng(shape, dtype)
    fun = buffer_callback.buffer_callback(
        callback, jax.ShapeDtypeStruct(data.shape, data.dtype)
    )

    # We can't actually test the output because numpy doesn't support writable
    # DLPack tensors.
    jax.block_until_ready(fun(data))

  @parameterized.parameters(jtu.dtypes.all)
  @jtu.run_on_devices("cuda")
  def test_cuda_array_interface(self, dtype):
    def callback(ctx, out, arg):
      ctx.stream  # doesn't crash

      self.assertEqual(ctx.stage, buffer_callback.ExecutionStage.EXECUTE)
      self.assertEqual(arg.shape, shape)
      self.assertEqual(arg.dtype, dtype)
      self.assertEqual(out.shape, shape)
      self.assertEqual(out.dtype, dtype)

      obj = arg.__cuda_array_interface__
      self.assertEqual(obj["shape"], data.shape)
      self.assertEqual(obj["typestr"], data.dtype.str)

      obj = out.__cuda_array_interface__
      self.assertEqual(obj["shape"], data.shape)
      self.assertEqual(obj["typestr"], data.dtype.str)

    rng = jtu.rand_default(self.rng())
    shape = (3, 4)
    data = rng(shape, dtype)
    fun = buffer_callback.buffer_callback(
        callback, jax.ShapeDtypeStruct(data.shape, data.dtype)
    )
    jax.block_until_ready(fun(data))

  @parameterized.parameters([
      "sequential", "sequential_unrolled", "expand_dims", "broadcast_all"
  ])
  @jtu.run_on_devices("cpu")
  def test_batching(self, vmap_method):
    def callback(ctx, out, *args):
      del ctx  # unused
      x = np.asarray(args[0])
      y = np.asarray(args[1])
      z = np.asarray(out)
      z[...] = x
      z[...] += y

    rng = jtu.rand_default(self.rng())
    shape = (3, 4)
    x = rng(shape, jnp.float32)
    y = rng(shape, jnp.float32)
    fun = buffer_callback.buffer_callback(
        callback,
        jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
        vmap_method=vmap_method,
    )
    self.assertArraysEqual(jax.vmap(fun)(x, y), x + y)

  @jtu.run_on_devices("cpu")
  def test_input_output_aliases(self):
    def callback(ctx, out, arg):
      del ctx  # unused
      x = np.asarray(arg)
      y = np.asarray(out)
      self.assertEqual(x.ctypes.data, y.ctypes.data)

    rng = jtu.rand_default(self.rng())
    shape = (3, 4)
    data = rng(shape, jnp.float32)
    fun = buffer_callback.buffer_callback(
        callback, jax.ShapeDtypeStruct(data.shape, data.dtype),
        input_output_aliases={0: 0},
    )
    jax.block_until_ready(fun(data))

  def test_side_effect(self):
    def callback(*_):
      nonlocal called
      called = True

    called = False
    fun = buffer_callback.buffer_callback(
        callback, jax.ShapeDtypeStruct((), jnp.float32), has_side_effect=True)
    jax.block_until_ready(fun())
    self.assertTrue(called)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
