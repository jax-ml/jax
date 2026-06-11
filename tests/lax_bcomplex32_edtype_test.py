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

"""Tests for bcomplex32 ExtendedDType support in JAX."""

import unittest

import numpy as np
import ml_dtypes

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax._src import core

jax.config.update("jax_enable_x64", False)

# Reference bf16 and bcomplex32 types
bf16 = ml_dtypes.bfloat16
_bfloat16_dtype = dtypes._bfloat16_dtype
_bcomplex32_edtype = dtypes.bcomplex32_edtype


class BComplex32DTypeTest(unittest.TestCase):
  """Test bcomplex32 ExtendedDType infrastructure."""

  def test_edtype_exists(self):
    """bcomplex32_edtype is an ExtendedDType singleton."""
    self.assertIsInstance(_bcomplex32_edtype, dtypes.ExtendedDType)
    self.assertIs(_bcomplex32_edtype, dtypes.bcomplex32_edtype)

  def test_edtype_properties(self):
    """Basic properties of bcomplex32_edtype."""
    self.assertEqual(_bcomplex32_edtype.name, "bcomplex32")
    self.assertEqual(_bcomplex32_edtype.itemsize, 4)
    self.assertEqual(repr(_bcomplex32_edtype), "bcomplex32")

  def test_edtype_equality(self):
    """bcomplex32_edtype equality and hash."""
    self.assertEqual(_bcomplex32_edtype, _bcomplex32_edtype)
    self.assertNotEqual(_bcomplex32_edtype, _bfloat16_dtype)
    # Different instances of the same type should be equal
    self.assertEqual(hash(_bcomplex32_edtype), hash(dtypes.bcomplex32_edtype))

  def test_issubdtype(self):
    """issubdtype works correctly for bcomplex32_edtype."""
    self.assertTrue(dtypes.issubdtype(_bcomplex32_edtype, dtypes.extended))
    self.assertTrue(
      dtypes.issubdtype(_bcomplex32_edtype, dtypes.bcomplex32_scalar)
    )
    self.assertTrue(dtypes.issubdtype(_bcomplex32_edtype, np.complexfloating))
    self.assertTrue(dtypes.issubdtype(_bcomplex32_edtype, np.inexact))
    self.assertTrue(dtypes.issubdtype(_bcomplex32_edtype, np.number))
    self.assertFalse(dtypes.issubdtype(_bcomplex32_edtype, np.floating))
    self.assertFalse(dtypes.issubdtype(_bcomplex32_edtype, dtypes.prng_key))

  def test_physical_element_aval(self):
    """Physical element aval is ShapedArray((2,), bf16)."""
    rules = _bcomplex32_edtype._rules
    self.assertIsNotNone(rules)
    aval = rules.physical_element_aval(_bcomplex32_edtype)
    self.assertIsInstance(aval, core.ShapedArray)
    self.assertEqual(aval.shape, (2,))
    self.assertEqual(aval.dtype, _bfloat16_dtype)

  def test_physical_aval(self):
    """physical_aval converts logical bcomplex32 to physical bf16."""
    logical = core.ShapedArray((3, 4), _bcomplex32_edtype)
    physical = core.physical_aval(logical)
    self.assertEqual(physical.shape, (3, 4, 2))
    self.assertEqual(physical.dtype, _bfloat16_dtype)

  def test_tangent_dtype(self):
    """Tangent dtype for bcomplex32 is bcomplex32 itself (complex tangents)."""
    rules = _bcomplex32_edtype._rules
    self.assertIs(rules.tangent_dtype(_bcomplex32_edtype), _bcomplex32_edtype)

  def test_dtype_resolution(self):
    """dtype() correctly maps bcomplex32 inputs to bcomplex32_edtype."""
    # From np.dtype
    dt = dtypes.dtype(np.dtype(ml_dtypes.bcomplex32))
    self.assertIs(dt, _bcomplex32_edtype)

    # From scalar type
    dt = dtypes.dtype(ml_dtypes.bcomplex32)
    self.assertIs(dt, _bcomplex32_edtype)

  def test_check_and_canonicalize_user_dtype(self):
    """check_and_canonicalize_user_dtype maps bcomplex32 to edtype."""
    dt = dtypes.check_and_canonicalize_user_dtype(
      np.dtype(ml_dtypes.bcomplex32)
    )
    self.assertIs(dt, _bcomplex32_edtype)

    dt = dtypes.check_and_canonicalize_user_dtype(_bcomplex32_edtype)
    self.assertIs(dt, _bcomplex32_edtype)


class BComplex32ArrayCreationTest(unittest.TestCase):
  """Test array creation with bcomplex32 ExtendedDType."""

  def test_zeros(self):
    """Create bcomplex32 zeros array."""
    x = jnp.zeros(3, dtype=_bcomplex32_edtype)
    self.assertEqual(x.dtype, _bcomplex32_edtype)
    self.assertEqual(x.shape, (3,))

  def test_ones(self):
    """Create bcomplex32 ones array."""
    x = jnp.ones(3, dtype=_bcomplex32_edtype)
    self.assertEqual(x.dtype, _bcomplex32_edtype)
    self.assertEqual(x.shape, (3,))

  def test_full(self):
    """Create bcomplex32 full array."""
    x = jnp.full((2, 3), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    self.assertEqual(x.dtype, _bcomplex32_edtype)
    self.assertEqual(x.shape, (2, 3))

  def test_scalar_type_of(self):
    """scalar_type_of returns complex for bcomplex32."""
    x = jnp.zeros(1, dtype=_bcomplex32_edtype)
    self.assertEqual(dtypes.scalar_type_of(x), complex)

  def test_cross_dtype_promotion_blocked(self):
    """bcomplex32 must not implicitly promote with other dtypes (cf. PRNG keys).

    ExtendedDTypes are kept out of the global type promotion lattice so that
    cross-dtype operations require explicit decomposition via `lax.real` /
    `lax.imag`. This preserves the memory advantage of bcomplex32 by
    preventing silent promotion to complex64.
    """
    x = jnp.zeros(2, dtype=_bcomplex32_edtype)
    with self.assertRaises((TypeError, ValueError)):
      _ = x + jnp.ones(2, dtype=jnp.complex64)
    with self.assertRaises((TypeError, ValueError)):
      _ = x + jnp.ones(2, dtype=jnp.bfloat16)
    with self.assertRaises((TypeError, ValueError)):
      _ = x + 1.0  # weak float must not promote
    with self.assertRaises((TypeError, ValueError)):
      _ = x + 1j  # weak complex must not promote

  def test_same_dtype_arithmetic_allowed(self):
    """Same-dtype arithmetic on bcomplex32 must work without promotion."""
    x = jnp.full((2,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    y = jnp.full((2,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    z = x + y
    self.assertEqual(z.dtype, _bcomplex32_edtype)

  def test_indexing_and_iteration(self):
    """Indexing slices the trailing physical dim; iteration yields 0-d arrays."""
    x = jnp.full((3,), 0.0, dtype=_bcomplex32_edtype)
    x0 = jnp.full((1,), 1.0 + 10.0j, dtype=_bcomplex32_edtype)
    x1 = jnp.full((1,), 2.0 + 20.0j, dtype=_bcomplex32_edtype)
    x2 = jnp.full((1,), 3.0 + 30.0j, dtype=_bcomplex32_edtype)
    x = lax.concatenate([x0, x1, x2], dimension=0)

    # Indexing
    self.assertEqual(x[1].dtype, _bcomplex32_edtype)
    self.assertEqual(x[1].shape, ())
    self.assertEqual(x[1:].shape, (2,))

    # Iteration yields 0-d bcomplex32 arrays
    items = list(x)
    self.assertEqual(len(items), 3)
    self.assertEqual(items[0].dtype, _bcomplex32_edtype)
    self.assertEqual(items[0].shape, ())
    np.testing.assert_allclose(
      np.array(jnp.real(items[1]), dtype=np.float32),
      np.array([2.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_iter_on_zero_d_raises(self):
    """Iterating a 0-d bcomplex32 array must raise TypeError."""
    x = jnp.full((), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    with self.assertRaises(TypeError):
      list(x)


class BComplex32OpsTest(unittest.TestCase):
  """Test basic operations on bcomplex32 arrays."""

  def test_real(self):
    """Extract real part of bcomplex32 array."""
    x = jnp.full((2,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    # Override: create with distinct values via separate full calls
    x0 = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    x1 = jnp.full((1,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    from jax._src.lax import lax

    x = lax.concatenate([x0, x1], 0)
    r = jnp.real(x)
    self.assertEqual(r.dtype, _bfloat16_dtype)
    np.testing.assert_allclose(
      np.array(r, dtype=np.float32),
      np.array([1.0, 3.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_imag(self):
    """Extract imaginary part of bcomplex32 array."""
    x0 = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    x1 = jnp.full((1,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    from jax._src.lax import lax

    x = lax.concatenate([x0, x1], 0)
    i = jnp.imag(x)
    self.assertEqual(i.dtype, _bfloat16_dtype)
    np.testing.assert_allclose(
      np.array(i, dtype=np.float32),
      np.array([2.0, 4.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_complex_construction(self):
    """Construct bcomplex32 from bf16 real and imag."""
    from jax._src.lax import lax

    re = jnp.array([1.0, 3.0], dtype=_bfloat16_dtype)
    im = jnp.array([2.0, 4.0], dtype=_bfloat16_dtype)
    z = lax.complex(re, im)
    self.assertEqual(z.dtype, _bcomplex32_edtype)
    self.assertEqual(z.shape, (2,))

  def test_neg(self):
    """Negate bcomplex32 array."""
    x = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    y = -x
    self.assertEqual(y.dtype, _bcomplex32_edtype)
    np.testing.assert_allclose(
      np.array(jnp.real(y), dtype=np.float32),
      np.array([-1.0], dtype=np.float32),
      rtol=0.02,
    )
    np.testing.assert_allclose(
      np.array(jnp.imag(y), dtype=np.float32),
      np.array([-2.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_add(self):
    """Add two bcomplex32 arrays."""
    x = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    y = jnp.full((1,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    z = x + y
    self.assertEqual(z.dtype, _bcomplex32_edtype)
    np.testing.assert_allclose(
      np.array(jnp.real(z), dtype=np.float32),
      np.array([4.0], dtype=np.float32),
      rtol=0.02,
    )
    np.testing.assert_allclose(
      np.array(jnp.imag(z), dtype=np.float32),
      np.array([6.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_sub(self):
    """Subtract two bcomplex32 arrays."""
    x = jnp.full((1,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    y = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    z = x - y
    self.assertEqual(z.dtype, _bcomplex32_edtype)
    np.testing.assert_allclose(
      np.array(jnp.real(z), dtype=np.float32),
      np.array([2.0], dtype=np.float32),
      rtol=0.02,
    )
    np.testing.assert_allclose(
      np.array(jnp.imag(z), dtype=np.float32),
      np.array([2.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_mul(self):
    """Multiply two bcomplex32 arrays."""
    # (1+2j) * (3+4j) = 3 + 4j + 6j + 8j^2 = 3 - 8 + 10j = -5 + 10j
    x = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    y = jnp.full((1,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    z = x * y
    self.assertEqual(z.dtype, _bcomplex32_edtype)
    np.testing.assert_allclose(
      np.array(jnp.real(z), dtype=np.float32),
      np.array([-5.0], dtype=np.float32),
      rtol=0.1,
    )
    np.testing.assert_allclose(
      np.array(jnp.imag(z), dtype=np.float32),
      np.array([10.0], dtype=np.float32),
      rtol=0.1,
    )

  def test_conj(self):
    """Conjugate bcomplex32 array."""
    x = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    y = jnp.conj(x)
    self.assertEqual(y.dtype, _bcomplex32_edtype)
    np.testing.assert_allclose(
      np.array(jnp.real(y), dtype=np.float32),
      np.array([1.0], dtype=np.float32),
      rtol=0.02,
    )
    np.testing.assert_allclose(
      np.array(jnp.imag(y), dtype=np.float32),
      np.array([-2.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_abs(self):
    """Absolute value of bcomplex32 array."""
    x = jnp.full((1,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    y = jnp.abs(x)
    self.assertEqual(y.dtype, _bfloat16_dtype)
    np.testing.assert_allclose(
      np.array(y, dtype=np.float32),
      np.array([5.0], dtype=np.float32),
      rtol=0.05,
    )


class BComplex32TypePromotionTest(unittest.TestCase):
  """Test type promotion with bcomplex32."""

  def test_promote_bf16_bcomplex32(self):
    """bf16 + bcomplex32 -> bcomplex32 via explicit construction."""
    # Mixed-type promotion between bf16 and bcomplex32_edtype requires
    # explicitly constructing the complex value from real bf16 + zero imag.
    from jax._src.lax import lax as lax_mod

    x_re = jnp.full((1,), 1.0, dtype=_bfloat16_dtype)
    x = lax_mod.complex(x_re, jnp.zeros_like(x_re))
    y = jnp.full((1,), 2.0 + 3.0j, dtype=_bcomplex32_edtype)
    z = x + y
    self.assertEqual(z.dtype, _bcomplex32_edtype)

  def test_to_complex_dtype(self):
    """to_complex_dtype maps bf16 to bcomplex32_edtype."""
    result = dtypes.to_complex_dtype(_bfloat16_dtype)
    self.assertIs(result, _bcomplex32_edtype)


class BComplex32JitTest(unittest.TestCase):
  """Test JIT compilation with bcomplex32."""

  def test_jit_real(self):
    """JIT-compiled real extraction."""

    @jax.jit
    def f(x):
      return jnp.real(x)

    x = jnp.full((2,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    r = f(x)
    self.assertEqual(r.dtype, _bfloat16_dtype)
    np.testing.assert_allclose(
      np.array(r, dtype=np.float32),
      np.array([1.0, 1.0], dtype=np.float32),
      rtol=0.02,
    )

  def test_jit_add(self):
    """JIT-compiled addition."""

    @jax.jit
    def f(x, y):
      return x + y

    x = jnp.full((1,), 1.0 + 2.0j, dtype=_bcomplex32_edtype)
    y = jnp.full((1,), 3.0 + 4.0j, dtype=_bcomplex32_edtype)
    z = f(x, y)
    self.assertEqual(z.dtype, _bcomplex32_edtype)
    np.testing.assert_allclose(
      np.array(jnp.real(z), dtype=np.float32),
      np.array([4.0], dtype=np.float32),
      rtol=0.02,
    )


class BComplex32DivTest(unittest.TestCase):
  """Test division on bcomplex32 arrays."""

  def test_div_basic(self):
    """Basic complex division: (6+8j) / (3+4j) = 2.0."""
    from jax._src.lax import lax

    re_x = jnp.array([6.0], dtype=_bfloat16_dtype)
    im_x = jnp.array([8.0], dtype=_bfloat16_dtype)
    x = lax.complex(re_x, im_x)

    re_y = jnp.array([3.0], dtype=_bfloat16_dtype)
    im_y = jnp.array([4.0], dtype=_bfloat16_dtype)
    y = lax.complex(re_y, im_y)

    z = x / y
    self.assertEqual(z.dtype, _bcomplex32_edtype)
    z_re = np.array(jnp.real(z), dtype=np.float32)
    z_im = np.array(jnp.imag(z), dtype=np.float32)
    # (6+8j)/(3+4j) = (6*3+8*4 + (8*3-6*4)j) / (9+16) = (50+0j)/25 = 2.0
    np.testing.assert_allclose(z_re, [2.0], rtol=0.1, atol=0.1)
    np.testing.assert_allclose(z_im, [0.0], rtol=0.1, atol=0.1)

  def test_div_general(self):
    """General complex division: (1+2j) / (3+4j)."""
    from jax._src.lax import lax

    re_x = jnp.array([1.0], dtype=_bfloat16_dtype)
    im_x = jnp.array([2.0], dtype=_bfloat16_dtype)
    x = lax.complex(re_x, im_x)

    re_y = jnp.array([3.0], dtype=_bfloat16_dtype)
    im_y = jnp.array([4.0], dtype=_bfloat16_dtype)
    y = lax.complex(re_y, im_y)

    z = x / y
    self.assertEqual(z.dtype, _bcomplex32_edtype)
    z_re = np.array(jnp.real(z), dtype=np.float32)
    z_im = np.array(jnp.imag(z), dtype=np.float32)
    # (1+2j)/(3+4j) = (3+8 + (6-4)j) / (9+16) = (11+2j)/25
    np.testing.assert_allclose(z_re, [0.44], rtol=0.1, atol=0.1)
    np.testing.assert_allclose(z_im, [0.08], rtol=0.1, atol=0.1)

  def test_div_jit(self):
    """JIT-compiled complex division."""

    @jax.jit
    def f(x, y):
      return x / y

    from jax._src.lax import lax

    x = lax.complex(
      jnp.array([6.0], dtype=_bfloat16_dtype),
      jnp.array([8.0], dtype=_bfloat16_dtype),
    )
    y = lax.complex(
      jnp.array([3.0], dtype=_bfloat16_dtype),
      jnp.array([4.0], dtype=_bfloat16_dtype),
    )

    z = f(x, y)
    z_re = np.array(jnp.real(z), dtype=np.float32)
    np.testing.assert_allclose(z_re, [2.0], rtol=0.1, atol=0.1)


class BComplex32MatmulTest(unittest.TestCase):
  """Test matrix multiplication with bcomplex32."""

  def _make_bcomplex32_matrix(self, complex64_arr):
    """Convert a complex64 array to bcomplex32 via lax.complex."""
    from jax._src.lax import lax

    re_bf16 = jnp.array(complex64_arr.real, dtype=_bfloat16_dtype)
    im_bf16 = jnp.array(complex64_arr.imag, dtype=_bfloat16_dtype)
    return lax.complex(re_bf16, im_bf16)

  def _to_complex64(self, bcomplex32_arr):
    """Convert a bcomplex32 array back to complex64 for comparison."""
    from jax._src.lax import lax

    return lax.complex(
      jnp.array(jnp.real(bcomplex32_arr), dtype=jnp.float32),
      jnp.array(jnp.imag(bcomplex32_arr), dtype=jnp.float32),
    )

  def test_2x2_matmul(self):
    """2x2 complex matrix multiplication."""
    a64 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    b64 = np.array([[2 + 1j, 4 + 3j], [6 + 5j, 8 + 7j]], dtype=np.complex64)
    expected = a64 @ b64

    a32 = self._make_bcomplex32_matrix(a64)
    b32 = self._make_bcomplex32_matrix(b64)
    result = a32 @ b32
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, expected, rtol=0.15, atol=0.15)

  def test_vector_matrix_mul(self):
    """Vector-matrix multiplication."""
    a64 = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    b64 = np.array([[2 + 1j, 4 + 3j], [6 + 5j, 8 + 7j]], dtype=np.complex64)
    expected = a64 @ b64

    a32 = self._make_bcomplex32_matrix(a64)
    b32 = self._make_bcomplex32_matrix(b64)
    result = a32 @ b32
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, expected, rtol=0.15, atol=0.15)

  def test_matmul_jit(self):
    """JIT-compiled matmul."""

    @jax.jit
    def f(a, b):
      return a @ b

    a64 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    b64 = np.array([[2 + 1j, 4 + 3j], [6 + 5j, 8 + 7j]], dtype=np.complex64)
    expected = a64 @ b64

    a32 = self._make_bcomplex32_matrix(a64)
    b32 = self._make_bcomplex32_matrix(b64)
    result = f(a32, b32)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, expected, rtol=0.15, atol=0.15)


class BComplex32TranscendentalTest(unittest.TestCase):
  """Test transcendental functions on bcomplex32."""

  def _make_bcomplex32_value(self, re, im):
    """Create a scalar bcomplex32 value."""
    from jax._src.lax import lax

    return lax.complex(
      jnp.array([re], dtype=_bfloat16_dtype),
      jnp.array([im], dtype=_bfloat16_dtype),
    )

  def _to_complex64(self, arr):
    """Convert bcomplex32 to complex64."""
    from jax._src.lax import lax

    return lax.complex(
      jnp.array(jnp.real(arr), dtype=jnp.float32),
      jnp.array(jnp.imag(arr), dtype=jnp.float32),
    )

  def test_exp(self):
    """exp of bcomplex32 value."""
    x = self._make_bcomplex32_value(0.0, 1.0)
    # exp(i) = cos(1) + i*sin(1) ~ 0.5403 + 0.8415i
    result = jnp.exp(x)
    result64 = self._to_complex64(result)
    expected = np.exp(1j)
    np.testing.assert_allclose(result64, [expected], rtol=0.1, atol=0.1)

  def test_exp_real(self):
    """exp of a real-valued bcomplex32."""
    x = self._make_bcomplex32_value(1.0, 0.0)
    result = jnp.exp(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [np.exp(1.0)], rtol=0.05, atol=0.05)

  def test_log(self):
    """log of bcomplex32 value."""
    # log(1) = 0
    x = self._make_bcomplex32_value(1.0, 0.0)
    result = jnp.log(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [0.0], rtol=0.05, atol=0.05)

  def test_log_complex(self):
    """log of complex bcomplex32 value."""
    x = self._make_bcomplex32_value(1.0, 1.0)
    result = jnp.log(x)
    result64 = self._to_complex64(result)
    expected = np.log(1.0 + 1.0j)
    np.testing.assert_allclose(result64, [expected], rtol=0.1, atol=0.1)

  def test_sin(self):
    """sin of bcomplex32 value."""
    x = self._make_bcomplex32_value(1.0, 0.0)
    result = jnp.sin(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [np.sin(1.0)], rtol=0.05, atol=0.05)

  def test_cos(self):
    """cos of bcomplex32 value."""
    x = self._make_bcomplex32_value(1.0, 0.0)
    result = jnp.cos(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [np.cos(1.0)], rtol=0.05, atol=0.05)

  def test_sqrt(self):
    """sqrt of bcomplex32 value."""
    x = self._make_bcomplex32_value(4.0, 0.0)
    result = jnp.sqrt(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [2.0], rtol=0.05, atol=0.05)

  def test_sqrt_complex(self):
    """sqrt of a complex bcomplex32 value."""
    x = self._make_bcomplex32_value(-1.0, 0.0)
    result = jnp.sqrt(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [1j], rtol=0.1, atol=0.1)

  def test_tanh(self):
    """tanh of bcomplex32 value."""
    x = self._make_bcomplex32_value(0.0, 0.0)
    result = jnp.tanh(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [0.0], rtol=0.05, atol=0.05)

  def test_tanh_real(self):
    """tanh of a real-valued bcomplex32."""
    x = self._make_bcomplex32_value(1.0, 0.0)
    result = jnp.tanh(x)
    result64 = self._to_complex64(result)
    np.testing.assert_allclose(result64, [np.tanh(1.0)], rtol=0.05, atol=0.05)

  def test_exp_jit(self):
    """JIT-compiled exp."""

    @jax.jit
    def f(x):
      return jnp.exp(x)

    x = self._make_bcomplex32_value(0.0, 1.0)
    result = f(x)
    result64 = self._to_complex64(result)
    expected = np.exp(1j)
    np.testing.assert_allclose(result64, [expected], rtol=0.1, atol=0.1)


class BComplex32AutodiffTest(unittest.TestCase):
  """Test autodiff (grad, jvp, vjp) with bcomplex32."""

  def test_grad_mul(self):
    """grad of |x|^2 at x=1+2j matches complex64 behavior."""
    from jax._src.lax import lax

    def f(x):
      return lax.real(lax.mul(x, lax.conj(x)))

    # |x|^2 = x * conj(x)
    # With JAX's grad convention for complex, grad(|x|^2) = 2*conj(x)
    x = lax.complex(
      jnp.array(1.0, dtype=_bfloat16_dtype),
      jnp.array(2.0, dtype=_bfloat16_dtype),
    )
    g = jax.grad(f)(x)
    # grad should be approximately 2*conj(x) = 2-4j (matches complex64 behavior)
    re_grad = lax.convert_element_type(lax.real(g), np.float32)
    im_grad = lax.convert_element_type(lax.imag(g), np.float32)
    np.testing.assert_allclose(re_grad, 2.0, rtol=0.1)
    np.testing.assert_allclose(im_grad, -4.0, rtol=0.1)

  def test_grad_real_function(self):
    """grad of Re(x) for complex input matches complex64 behavior."""
    from jax._src.lax import lax

    def f(x):
      return lax.real(x)

    x = lax.complex(
      jnp.array(3.0, dtype=_bfloat16_dtype),
      jnp.array(4.0, dtype=_bfloat16_dtype),
    )
    g = jax.grad(f)(x)
    # With JAX's grad convention for complex, grad(Re(x)) = 1+0j
    re_grad = lax.convert_element_type(lax.real(g), np.float32)
    im_grad = lax.convert_element_type(lax.imag(g), np.float32)
    np.testing.assert_allclose(re_grad, 1.0, rtol=0.1)
    np.testing.assert_allclose(im_grad, 0.0, atol=0.1)

  def test_jvp_add(self):
    """JVP of addition works"""
    from jax._src.lax import lax

    x = lax.complex(
      jnp.array([1.0], dtype=_bfloat16_dtype),
      jnp.array([2.0], dtype=_bfloat16_dtype),
    )
    tx = lax.complex(
      jnp.array([0.1], dtype=_bfloat16_dtype),
      jnp.array([0.2], dtype=_bfloat16_dtype),
    )
    primals, tangents = jax.jvp(lambda x: lax.add(x, x), [x], [tx])
    self.assertEqual(primals.dtype, _bcomplex32_edtype)

  def test_vjp_matmul(self):
    """VJP of matmul works with bcomplex32"""
    from jax._src.lax import lax

    a = lax.complex(
      jnp.eye(3, dtype=_bfloat16_dtype),
      jnp.zeros((3, 3), dtype=_bfloat16_dtype),
    )
    b = lax.complex(
      jnp.ones((3, 3), dtype=_bfloat16_dtype),
      jnp.zeros((3, 3), dtype=_bfloat16_dtype),
    )
    _, vjp_fn = jax.vjp(jnp.matmul, a, b)
    g_out = lax.complex(
      jnp.ones((3, 3), dtype=_bfloat16_dtype),
      jnp.zeros((3, 3), dtype=_bfloat16_dtype),
    )
    ga, gb = vjp_fn(g_out)
    self.assertEqual(ga.dtype, _bcomplex32_edtype)
    self.assertEqual(gb.dtype, _bcomplex32_edtype)

  def test_grad_jit(self):
    """Grad + JIT works"""
    from jax._src.lax import lax

    @jax.jit
    def f(x):
      return lax.real(lax.mul(x, x))

    x = lax.complex(
      jnp.array(2.0, dtype=_bfloat16_dtype),
      jnp.array(0.0, dtype=_bfloat16_dtype),
    )
    g = jax.grad(f)(x)
    re_grad = lax.convert_element_type(lax.real(g), np.float32)
    np.testing.assert_allclose(re_grad, 4.0, rtol=0.1)  # d/dx(x^2) = 2x = 4


if __name__ == "__main__":
  unittest.main()
