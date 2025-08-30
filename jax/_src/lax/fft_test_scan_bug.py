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

"""Tests for FFT operations in scan contexts, specifically for issue #31374."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import test_util as jtu


class FftScanBugTest(jtu.JaxTestCase):
  """Test cases for FFT bug in scan operations (issue #31374)."""

  def test_small_fft_in_scan_basic(self):
    """Test the basic failing case: small FFT in scan with multiple steps."""
    n_arr = 30
    arr = jnp.linspace(0., 1., n_arr) * 1.j
    
    def step_fun(arr, _):
      arr = jnp.fft.fftn(arr)
      return arr, None
    
    # This should not crash with LegacyDuccFft error
    result, _ = jax.lax.scan(
        step_fun,
        init=arr,
        length=10,
    )
    
    # Verify the result has the correct shape and type
    self.assertEqual(result.shape, arr.shape)
    self.assertEqual(result.dtype, arr.dtype)

  @parameterized.parameters([
    (28, 2),
    (29, 2), 
    (30, 2),
    (31, 2),
    (32, 2),
    (30, 1),  # Single step should work
    (30, 10), # Multiple steps should work
  ])
  def test_fft_scan_size_threshold(self, array_size, scan_length):
    """Test FFT in scan around the problematic size threshold."""
    arr = jnp.linspace(0., 1., array_size) * 1.j
    
    def step_fun(arr, _):
      return jnp.fft.fftn(arr), None
    
    result, _ = jax.lax.scan(step_fun, init=arr, length=scan_length)
    
    # Verify the result
    self.assertEqual(result.shape, arr.shape)
    self.assertEqual(result.dtype, arr.dtype)

  def test_fft_scan_with_workaround(self):
    """Test that the workaround case still works after the fix."""
    n_fft = 30
    n_arithmetic = 128
    
    def step_fun(carry, _):
      arr_arithmetic, arr_fft = carry
      arr_arithmetic = jnp.exp(arr_arithmetic)
      arr_fft = jnp.fft.fftn(arr_fft)
      return (arr_arithmetic, arr_fft), None
    
    arr_arithmetic = jnp.linspace(0., 1., n_arithmetic)
    arr_fft = jnp.linspace(0., 1., n_fft) * 1.j
    
    (result_arithmetic, result_fft), _ = jax.lax.scan(
        step_fun,
        init=(arr_arithmetic, arr_fft),
        length=2,
    )
    
    # Verify results
    self.assertEqual(result_arithmetic.shape, arr_arithmetic.shape)
    self.assertEqual(result_fft.shape, arr_fft.shape)

  def test_fft_immediate_mode_still_works(self):
    """Test that immediate mode FFT still works."""
    n_arr = 30
    arr = jnp.linspace(0., 1., n_arr) * 1.j
    
    result = jnp.fft.fftn(arr)
    
    self.assertEqual(result.shape, arr.shape)
    self.assertEqual(result.dtype, arr.dtype)

  def test_fft_jit_mode_still_works(self):
    """Test that JIT mode FFT still works."""
    n_arr = 30
    arr = jnp.linspace(0., 1., n_arr) * 1.j
    
    @jax.jit
    def fft_func(x):
      return jnp.fft.fftn(x)
    
    result = fft_func(arr)
    
    self.assertEqual(result.shape, arr.shape)
    self.assertEqual(result.dtype, arr.dtype)

  @parameterized.parameters([
    jnp.fft.fft,
    jnp.fft.ifft,
    jnp.fft.fftn,
    jnp.fft.ifftn,
  ])
  def test_various_fft_types_in_scan(self, fft_func):
    """Test various FFT function types in scan operations."""
    n_arr = 30
    arr = jnp.linspace(0., 1., n_arr) * 1.j
    
    def step_fun(arr, _):
      if fft_func in [jnp.fft.fftn, jnp.fft.ifftn]:
        return fft_func(arr), None
      else:
        # For 1D functions, use the last axis
        return fft_func(arr, axis=-1), None
    
    result, _ = jax.lax.scan(step_fun, init=arr, length=2)
    
    self.assertEqual(result.shape, arr.shape)
    self.assertEqual(result.dtype, arr.dtype)

  def test_multidimensional_fft_in_scan(self):
    """Test multidimensional FFT in scan operations."""
    arr = jnp.linspace(0., 1., 30).reshape(5, 6) * 1.j
    
    def step_fun(arr, _):
      return jnp.fft.fftn(arr), None
    
    result, _ = jax.lax.scan(step_fun, init=arr, length=3)
    
    self.assertEqual(result.shape, arr.shape)
    self.assertEqual(result.dtype, arr.dtype)

  def test_real_input_fft_in_scan(self):
    """Test real input FFT in scan operations."""
    n_arr = 30
    arr = jnp.linspace(0., 1., n_arr)  # Real input
    
    def step_fun(arr, _):
      # Use rfft for real input
      result = jnp.fft.rfft(arr)
      # Convert back to real for next iteration
      return jnp.fft.irfft(result, n=len(arr)), None
    
    result, _ = jax.lax.scan(step_fun, init=arr, length=2)
    
    self.assertEqual(result.shape, arr.shape)
    self.assertEqual(result.dtype, arr.dtype)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
