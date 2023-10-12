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

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import dtypes
from jax import numpy as jnp
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import lib as jaxlib
from jax._src import test_util as jtu
from jax._src.interpreters import pxla
import numpy as np

config.parse_flags_with_absl()

def _cpp_device_put(value, device):
  aval = api_util.shaped_abstractify(value)
  return pxla.batched_device_put(
      aval, jax.sharding.SingleDeviceSharding(device), [value], [device])


class JaxJitTest(jtu.JaxTestCase):

  def test_is_float_0(self):
    self.assertTrue(
        jaxlib.jax_jit._is_float0(np.zeros((5, 5), dtype=jax.float0)))
    self.assertFalse(jaxlib.jax_jit._is_float0(np.zeros((5, 5))))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_numpy_masked_array(self, device_put_function):
    # TODO(jakevdp): add appropriate logic to jaxlib device_put and update this test.
    if device_put_function is _cpp_device_put:
      self.skipTest("cpp device_put does not yet reject masked arrays.")
    device = jax.devices()[0]
    value = np.ma.array([1, 2, 3], mask=[True, False, True])
    with self.assertRaisesRegex(ValueError, "numpy masked arrays are not supported"):
      device_put_function(value, device=device)

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_numpy_scalars(self, device_put_function):

    device = jax.devices()[0]
    for dtype in jtu.supported_dtypes():
      value = dtype(0)

      output_buffer = device_put_function(value, device=device)

      self.assertFalse(output_buffer.aval.weak_type)
      dtype = dtypes.canonicalize_dtype(dtype)
      self.assertEqual(output_buffer.aval, core.ShapedArray((), dtype))
      self.assertEqual(output_buffer.dtype, dtype)

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_numpy_arrays(self, device_put_function):

    device = jax.devices()[0]
    for dtype in jtu.supported_dtypes():
      value = np.zeros((3, 4), dtype=dtype)
      output_buffer = device_put_function(value, device=device)

      self.assertFalse(output_buffer.aval.weak_type)
      dtype = dtypes.canonicalize_dtype(dtype)
      self.assertEqual(output_buffer.aval, core.ShapedArray((3, 4), dtype))
      self.assertEqual(output_buffer.dtype, dtype)
      np.testing.assert_array_equal(output_buffer, np.zeros((3, 4),
                                                            dtype=dtype))

  @parameterized.parameters([jax.device_put, _cpp_device_put])
  def test_device_put_on_buffers(self, device_put_function):
    device = jax.devices()[0]
    jitted_f = jax.jit(lambda x: x + 1)

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

      self.assertEqual(output_buffer.dtype, sda.dtype)
      self.assertEqual(output_buffer.aval, sda.aval)
      np.testing.assert_array_equal(output_buffer, np.asarray(sda))

  def test_device_put_on_python_scalars(self):
    device = jax.devices()[0]
    int_type = dtypes.canonicalize_dtype(np.int64)
    float_type = dtypes.canonicalize_dtype(np.float64)
    complex_type = dtypes.canonicalize_dtype(np.complex128)

    # int
    res = np.asarray(_cpp_device_put(1, device))
    self.assertEqual(res, 1)
    self.assertEqual(res.dtype, int_type)
    # We also compare to the Python Jax API, to make sure we have the exact
    # same behavior. When Jax removes the flag and removes this feature, this
    # test will fail.
    self.assertEqual(jnp.asarray(1).dtype, res.dtype)

    # float
    res = np.asarray(_cpp_device_put(1.0, device))
    self.assertEqual(res, 1.0)
    self.assertEqual(res.dtype, float_type)
    self.assertEqual(jnp.asarray(1.0).dtype, res.dtype)

    # bool
    for bool_value in [True, False]:
      res = np.asarray(_cpp_device_put(bool_value, device))
      self.assertEqual(res, np.asarray(bool_value))
      self.assertEqual(res.dtype, np.bool_)
      self.assertEqual(jnp.asarray(bool_value).dtype, res.dtype)

    # Complex
    if not (config.enable_x64.value and jtu.test_device_matches(["tpu"])):
      # No TPU support for complex128.
      res = np.asarray(_cpp_device_put(1 + 1j, device))
      self.assertEqual(res, 1 + 1j)
      self.assertEqual(res.dtype, complex_type)
      self.assertEqual(jnp.asarray(1 + 1j).dtype, res.dtype)

  def test_arg_signature_of_value(self):
    """Tests the C++ code-path."""
    jax_enable_x64 = config.enable_x64.value

    # 1. Numpy scalar types
    for dtype in jtu.supported_dtypes():
      value = dtype(0)

      signature = jaxlib.jax_jit._ArgSignatureOfValue(value, jax_enable_x64)
      self.assertEqual(signature.dtype, jax.device_put(value).dtype)
      self.assertEqual(signature.shape, ())
      self.assertFalse(signature.weak_type)

    # 2. Numpy arrays
    for dtype in jtu.supported_dtypes():
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
    if not (jax_enable_x64 and jtu.test_device_matches(["tpu"])):
      # No TPU support for complex128.
      signature = jaxlib.jax_jit._ArgSignatureOfValue(1 + 1j, jax_enable_x64)
      self.assertEqual(signature.dtype, jax.device_put(1 + 1j).dtype)
      self.assertEqual(signature.dtype, complex_type)
      self.assertEqual(signature.shape, ())
      self.assertTrue(signature.weak_type)

  def test_signature_support(self):
    def f(a, b, c):
      return a + b + c

    jitted_f = jax.jit(f)
    self.assertEqual(inspect.signature(f), inspect.signature(jitted_f))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
