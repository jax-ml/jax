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
from jax._src import test_util as jtu
from jax.experimental import buffer_callback

jax.config.parse_flags_with_absl()


class BufferCallbackTest(jtu.JaxTestCase):

  @parameterized.parameters(jtu.dtypes.all)
  @jtu.run_on_devices("cpu")
  def test_numpy(self, dtype):
    def callback(ctx, out, arg):
      with self.assertRaisesRegex(
          jax.errors.JaxRuntimeError, "XLA FFI GPU context is not available"
      ):
        ctx.stream

      self.assertEqual(ctx.stage, buffer_callback.ExecutionStage.EXECUTE)
      self.assertEqual(ctx.device_ordinal, 0)
      self.assertEqual(arg.shape, shape)
      self.assertEqual(arg.dtype, dtype)
      self.assertEqual(out.shape, shape)
      self.assertEqual(out.dtype, dtype)

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
      self.assertTrue(obj["data"][1])

      obj = out.__cuda_array_interface__
      self.assertEqual(obj["shape"], data.shape)
      self.assertEqual(obj["typestr"], data.dtype.str)
      self.assertFalse(obj["data"][1])

    rng = jtu.rand_default(self.rng())
    shape = (3, 4)
    data = rng(shape, dtype)
    fun = buffer_callback.buffer_callback(
        callback, jax.ShapeDtypeStruct(data.shape, data.dtype)
    )
    fun(data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
