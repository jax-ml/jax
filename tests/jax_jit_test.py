# Copyright 2020 Google LLC
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

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import api
from jax import dtypes
from jax import lib as jaxlib
from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import config
import numpy as np


# It covers all JAX numpy types types except bfloat16 and numpy array.
# TODO(jblespiau): Add support for float0 in the C++ path.
_EXCLUDED_TYPES = [np.ndarray]

_SCALAR_NUMPY_TYPES = [
    x for x in jax._src.abstract_arrays.array_types if x not in _EXCLUDED_TYPES
]


def _cpp_device_put(value, device):
  return jaxlib.jax_jit.device_put(value, config.x64_enabled, device)


class JaxJitTest(parameterized.TestCase):

  def test_is_float_0(self):
    self.assertTrue(
        jaxlib.jax_jit._is_float0(np.zeros((5, 5), dtype=jax.float0)))
    self.assertFalse(jaxlib.jax_jit._is_float0(np.zeros((5, 5))))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_numpy_scalars(self, device_put_function):

    device = jax.devices()[0]
    for dtype in _SCALAR_NUMPY_TYPES:
      value = dtype(0)

      output_buffer = device_put_function(value, device=device)

      self.assertFalse(output_buffer.aval.weak_type)
      self.assertEqual(output_buffer.aval, jax.core.ShapedArray((), dtype))
      self.assertEqual(output_buffer.dtype, dtypes.canonicalize_dtype(dtype))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_numpy_arrays(self, device_put_function):

    device = jax.devices()[0]
    for dtype in _SCALAR_NUMPY_TYPES:
      value = np.zeros((3, 4), dtype=dtype)
      output_buffer = device_put_function(value, device=device)

      self.assertFalse(output_buffer.aval.weak_type)
      self.assertEqual(output_buffer.aval, jax.core.ShapedArray((3, 4), dtype))
      self.assertEqual(output_buffer.dtype, dtypes.canonicalize_dtype(dtype))
      np.testing.assert_array_equal(output_buffer, np.zeros((3, 4),
                                                            dtype=dtype))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_buffers(self, device_put_function):
    device = jax.devices()[0]
    jitted_f = jax.jit(lambda x: x + 1)

    # We run it twice, to cover `_DeviceArray` and the C++ `Buffer`.
    for value in range(2):
      buffer = jitted_f(value)
      output_buffer = device_put_function(buffer, device=device)

      self.assertEqual(output_buffer.dtype, buffer.dtype)
      self.assertEqual(output_buffer.aval, buffer.aval)
      np.testing.assert_array_equal(output_buffer, np.array(value + 1))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_sharded_device_array(self, device_put_function):
    device = jax.devices()[0]

    pmaped_f = jax.pmap(lambda x: x + 1)
    for _ in range(2):
      sda = pmaped_f(np.asarray([[1]]))
      output_buffer = device_put_function(sda, device=device)

      self.assertNotIsInstance(output_buffer,
                               jax.interpreters.pxla.ShardedDeviceArray)
      self.assertEqual(output_buffer.dtype, sda.dtype)
      self.assertEqual(output_buffer.aval, sda.aval)
      np.testing.assert_array_equal(output_buffer, np.asarray(sda))

  def test_device_put_on_python_scalars(self):
    device = jax.devices()[0]
    int_type = dtypes.canonicalize_dtype(np.int64)
    float_type = dtypes.canonicalize_dtype(np.float64)
    complex_type = dtypes.canonicalize_dtype(np.complex128)

    # int
    res = _cpp_device_put(1, device).to_py()
    self.assertEqual(res, 1)
    self.assertEqual(res.dtype, int_type)
    # We also compare to the Python Jax API, to make sure we have the exact
    # same behavior. When Jax removes the flag and removes this feature, this
    # test will fail.
    self.assertEqual(jnp.asarray(1).dtype, res.dtype)

    # float
    res = _cpp_device_put(1.0, device).to_py()
    self.assertEqual(res, 1.0)
    self.assertEqual(res.dtype, float_type)
    self.assertEqual(jnp.asarray(1.0).dtype, res.dtype)

    # bool
    for bool_value in [True, False]:
      res = _cpp_device_put(bool_value, device).to_py()
      self.assertEqual(res, np.asarray(bool_value))
      self.assertEqual(res.dtype, np.bool_)
      self.assertEqual(jnp.asarray(bool_value).dtype, res.dtype)

    # Complex
    res = _cpp_device_put(1 + 1j, device).to_py()
    self.assertEqual(res, 1 + 1j)
    self.assertEqual(res.dtype, complex_type)
    self.assertEqual(jnp.asarray(1 + 1j).dtype, res.dtype)

  def test_convert_int_overflow(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "(Python int too large|Unable to convert Python scalar).*"):
      jaxlib.jax_jit.device_put(int(1e100), True, jax.devices()[0])

  def test_arg_signature_of_value(self):
    """Tests the C++ code-path."""
    jax_enable_x64 = config.x64_enabled

    # 1. Numpy scalar types
    for dtype in _SCALAR_NUMPY_TYPES:
      value = dtype(0)

      signature = jaxlib.jax_jit._ArgSignatureOfValue(value, jax_enable_x64)
      self.assertEqual(signature.dtype, jax.device_put(value).dtype)
      self.assertEqual(signature.shape, ())
      self.assertFalse(signature.weak_type)

    # 2. Numpy arrays
    for dtype in _SCALAR_NUMPY_TYPES:
      value = np.zeros((3, 4), dtype=dtype)

      signature = jaxlib.jax_jit._ArgSignatureOfValue(value, jax_enable_x64)
      self.assertEqual(signature.dtype, jax.device_put(value).dtype)
      self.assertEqual(signature.shape, (3, 4))
      self.assertFalse(signature.weak_type)

    int_type = dtypes.canonicalize_dtype(np.int64)
    float_type = dtypes.canonicalize_dtype(np.float64)
    complex_type = dtypes.canonicalize_dtype(np.complex128)

    # 3. Python scalar types
    # int
    signature = jaxlib.jax_jit._ArgSignatureOfValue(1, jax_enable_x64)
    self.assertEqual(signature.dtype, jax.device_put(1).dtype)
    self.assertEqual(signature.dtype, int_type)
    self.assertEqual(signature.shape, ())
    self.assertTrue(signature.weak_type)
    # float
    signature = jaxlib.jax_jit._ArgSignatureOfValue(1.0, jax_enable_x64)
    self.assertEqual(signature.dtype, jax.device_put(1.0).dtype)
    self.assertEqual(signature.dtype, float_type)
    self.assertEqual(signature.shape, ())
    self.assertTrue(signature.weak_type)
    # bool
    for bool_value in [True, False]:
      signature = jaxlib.jax_jit._ArgSignatureOfValue(bool_value,
                                                      jax_enable_x64)
      self.assertEqual(signature.dtype, jax.device_put(bool_value).dtype)
      self.assertEqual(signature.dtype, np.bool_)
      self.assertEqual(signature.shape, ())
      self.assertTrue(signature.weak_type)
    # Complex
    signature = jaxlib.jax_jit._ArgSignatureOfValue(1 + 1j, jax_enable_x64)
    self.assertEqual(signature.dtype, jax.device_put(1 + 1j).dtype)
    self.assertEqual(signature.dtype, complex_type)
    self.assertEqual(signature.shape, ())
    self.assertTrue(signature.weak_type)

  def test_signature_support(self):
    def f(a, b, c):
      return a + b + c

    jitted_f = api._cpp_jit(f)
    self.assertEqual(inspect.signature(f), inspect.signature(jitted_f))


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
