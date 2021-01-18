# Copyright 2019 Google LLC
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


import enum
import itertools
import operator

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import dtypes
from jax import numpy as jnp
from jax import test_util as jtu
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

bool_dtypes = [np.dtype('bool')]

signed_dtypes = [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
                 np.dtype('int64'), np.dtype('longlong'), np.dtype('intc')]

unsigned_dtypes = [np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32'),
                   np.dtype('uint64')]

np_float_dtypes = [np.dtype('float16'), np.dtype('float32'),
                   np.dtype('float64')]

float_dtypes = [np.dtype(dtypes.bfloat16)] + np_float_dtypes

complex_dtypes = [np.dtype('complex64'), np.dtype('complex128')]


all_dtypes = (bool_dtypes + signed_dtypes + unsigned_dtypes + float_dtypes +
              complex_dtypes)

scalar_types = [jnp.bool_, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64,
                jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64,
                jnp.complex64, jnp.complex128]

_EXPECTED_CANONICALIZE_X64 = {value: value for value in scalar_types}

_EXPECTED_CANONICALIZE_X32 = {value: value for value in scalar_types}
_EXPECTED_CANONICALIZE_X32[np.int64] = np.int32
_EXPECTED_CANONICALIZE_X32[np.uint64] = np.uint32
_EXPECTED_CANONICALIZE_X32[np.float64] = np.float32
_EXPECTED_CANONICALIZE_X32[np.complex128] = np.complex64
_EXPECTED_CANONICALIZE_X32[np.longlong] = np.int32


class DtypesTest(jtu.JaxTestCase):

  def test_canonicalize_type(self):
    expected = {
        True: _EXPECTED_CANONICALIZE_X64,
        False: _EXPECTED_CANONICALIZE_X32,
    }
    for in_dtype, expected_dtype in expected[FLAGS.jax_enable_x64].items():
      self.assertEqual(dtypes.canonicalize_dtype(in_dtype), expected_dtype)

  @parameterized.named_parameters(
    {"testcase_name": "_type={}".format(type.__name__), "type": type,
     "dtype": dtype}
    for type, dtype in [(bool, jnp.bool_), (int, jnp.int_), (float, jnp.float_),
                        (complex, jnp.complex_)])
  def testDefaultTypes(self, type, dtype):
    for f in [jnp.array, jax.jit(jnp.array), jax.jit(lambda x: x)]:
      y = f(type(0))
      self.assertTrue(isinstance(y, jnp.ndarray), msg=(f, y))
      self.assertEqual(y.dtype, dtypes.canonicalize_dtype(dtype), msg=(f, y))

  def testUnsupportedType(self):
    with self.assertRaisesRegex(TypeError, "nonsense.* not understood"):
      dtypes.canonicalize_dtype("nonsense")

  @parameterized.named_parameters(
    {"testcase_name": "_swap={}_jit={}".format(swap, jit),
     "swap": swap, "jit": jit}
    for swap in [False, True] for jit in [False, True])
  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype.*")
  def testBinaryPromotion(self, swap, jit):
    testcases = [
      (jnp.array(1.), 0., jnp.float_),
      (jnp.array(1.), jnp.array(0.), jnp.float_),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float16), jnp.float_),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float32), jnp.float_),
      # (jnp.array(1.), jnp.array(0., dtype=jnp.float16), jnp.float16),
      # (jnp.array(1.), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array(1., dtype=jnp.float16), 0., jnp.float16),
      (jnp.array(1., dtype=jnp.float32), 0., jnp.float32),
      (jnp.array(1., dtype=jnp.float64), 0., jnp.float64),
      (jnp.array(1., dtype=jnp.float16), jnp.array(0., dtype=jnp.float16), jnp.float16),
      (jnp.array(1., dtype=jnp.float16), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array(1., dtype=jnp.float16), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array(1., dtype=jnp.float32), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array(1., dtype=jnp.float32), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array(1., dtype=jnp.float64), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array([1.]), 0., jnp.float_),
      (jnp.array([1.]), jnp.array(0.), jnp.float_),
      (jnp.array([1.]), jnp.array(0., dtype=jnp.float16), jnp.float_),
      (jnp.array([1.]), jnp.array(0., dtype=jnp.float32), jnp.float_),
      (jnp.array([1.]), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array([1.], dtype=jnp.float32), jnp.array(0., dtype=jnp.float16), jnp.float32),
      (jnp.array([1.], dtype=jnp.float16), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array([1.], dtype=jnp.float16), 0., jnp.float16),
    ]
    op = jax.jit(operator.add) if jit else operator.add
    for x, y, dtype in testcases:
      x, y = (y, x) if swap else (x, y)
      z = op(x, y)
      self.assertTrue(isinstance(z, jnp.ndarray), msg=(x, y, z))
      self.assertEqual(z.dtype, dtypes.canonicalize_dtype(dtype), msg=(x, y, z))

  def testPromoteDtypes(self):
    for t1 in all_dtypes:
      self.assertEqual(t1, dtypes.promote_types(t1, t1))

      self.assertEqual(t1, dtypes.promote_types(t1, np.bool_))
      self.assertEqual(np.dtype(np.complex128),
                       dtypes.promote_types(t1, np.complex128))

      for t2 in all_dtypes:
        # Symmetry
        self.assertEqual(dtypes.promote_types(t1, t2),
                         dtypes.promote_types(t2, t1))

    self.assertEqual(np.dtype(np.float32),
                     dtypes.promote_types(np.float16, dtypes.bfloat16))

    # Promotions of non-inexact types against inexact types always prefer
    # the inexact types.
    for t in float_dtypes + complex_dtypes:
      for i in bool_dtypes + signed_dtypes + unsigned_dtypes:
        self.assertEqual(t, dtypes.promote_types(t, i))

    # Promotions between exact types, or between inexact types, match NumPy.
    for groups in [bool_dtypes + signed_dtypes + unsigned_dtypes,
                   np_float_dtypes + complex_dtypes]:
      for t1, t2 in itertools.combinations(groups, 2):
        self.assertEqual(np.promote_types(t1, t2),
                         dtypes.promote_types(t1, t2))

  def testScalarInstantiation(self):
    for t in [jnp.bool_, jnp.int32, jnp.bfloat16, jnp.float32, jnp.complex64]:
      a = t(1)
      self.assertEqual(a.dtype, jnp.dtype(t))
      self.assertIsInstance(a, xla.DeviceArray)
      self.assertEqual(0, jnp.ndim(a))

  def testIsSubdtype(self):
    for t in scalar_types:
      self.assertTrue(dtypes.issubdtype(t, t))
      self.assertTrue(dtypes.issubdtype(np.dtype(t).type, t))
      self.assertTrue(dtypes.issubdtype(t, np.dtype(t).type))
      self.assertTrue(dtypes.issubdtype(t, np.dtype(t)))
      if t != jnp.bfloat16:
        for category in [np.generic, jnp.inexact, jnp.integer, jnp.signedinteger,
                         jnp.unsignedinteger, jnp.floating, jnp.complexfloating]:
          self.assertEqual(dtypes.issubdtype(t, category),
                           np.issubdtype(np.dtype(t).type, category))
          self.assertEqual(dtypes.issubdtype(t, category),
                           np.issubdtype(np.dtype(t).type, category))

  def testArrayCasts(self):
    for t in [jnp.bool_, jnp.int32, jnp.bfloat16, jnp.float32, jnp.complex64]:
      a = np.array([1, 2.5, -3.7])
      self.assertEqual(a.astype(t).dtype, jnp.dtype(t))
      self.assertEqual(jnp.array(a).astype(t).dtype, jnp.dtype(t))

  def testEnumPromotion(self):
    class AnEnum(enum.IntEnum):
      A = 42
      B = 101
    np.testing.assert_equal(np.array(42), np.array(AnEnum.A))
    np.testing.assert_equal(jnp.array(42), jnp.array(AnEnum.A))
    np.testing.assert_equal(np.int32(101), np.int32(AnEnum.B))
    np.testing.assert_equal(jnp.int32(101), jnp.int32(AnEnum.B))

  def testScalarCastInsideJitWorks(self):
    # jnp.int32(tracer) should work.
    self.assertEqual(jnp.int32(101),
                     jax.jit(lambda x: jnp.int32(x))(jnp.float32(101.4)))


class TestPromotionTables(jtu.JaxTestCase):

  @parameterized.named_parameters(
    {"testcase_name": "_jaxtype={}".format(jaxtype),
     "jaxtype": jaxtype}
     for jaxtype in dtypes._jax_types)
  def testJaxTypeFromType(self, jaxtype):
    self.assertIs(dtypes._jax_type(jaxtype), jaxtype)

  @parameterized.named_parameters(
    {"testcase_name": "_jaxtype={}".format(jaxtype),
     "jaxtype": jaxtype}
     for jaxtype in dtypes._jax_types)
  def testJaxTypeFromVal(self, jaxtype):
    try:
      val = jaxtype(0)
    except TypeError:
      val = jaxtype.type(0)
    self.assertIs(dtypes._jax_type(val), jaxtype)

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype.*")
  def testObservedPromotionTable(self):
    """Test that the weak & strong dtype promotion table does not change over time."""
    # Note: * here refers to weakly-typed values
    typecodes = \
        ['b1','u1','u2','u4','u8','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i*','f*','c*']
    if FLAGS.jax_enable_x64:
      expected = [
        ['b1','u1','u2','u4','u8','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i8','f8','c8'],
        ['u1','u1','u2','u4','u8','i2','i2','i4','i8','bf','f2','f4','f8','c4','c8','u1','f8','c8'],
        ['u2','u2','u2','u4','u8','i4','i4','i4','i8','bf','f2','f4','f8','c4','c8','u2','f8','c8'],
        ['u4','u4','u4','u4','u8','i8','i8','i8','i8','bf','f2','f4','f8','c4','c8','u4','f8','c8'],
        ['u8','u8','u8','u8','u8','f8','f8','f8','f8','bf','f2','f4','f8','c4','c8','u8','f8','c8'],
        ['i1','i2','i4','i8','f8','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i1','f8','c8'],
        ['i2','i2','i4','i8','f8','i2','i2','i4','i8','bf','f2','f4','f8','c4','c8','i2','f8','c8'],
        ['i4','i4','i4','i8','f8','i4','i4','i4','i8','bf','f2','f4','f8','c4','c8','i4','f8','c8'],
        ['i8','i8','i8','i8','f8','i8','i8','i8','i8','bf','f2','f4','f8','c4','c8','i8','f8','c8'],
        ['bf','bf','bf','bf','bf','bf','bf','bf','bf','bf','f4','f4','f8','c4','c8','bf','bf','c4'],
        ['f2','f2','f2','f2','f2','f2','f2','f2','f2','f4','f2','f4','f8','c4','c8','f2','f2','c4'],
        ['f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f8','c4','c8','f4','f4','c4'],
        ['f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','c8','c8','f8','f8','c8'],
        ['c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c8','c4','c8','c4','c4','c4'],
        ['c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8'],
        ['i8','u1','u2','u4','u8','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i*','f*','c*'],
        ['f8','f8','f8','f8','f8','f8','f8','f8','f8','bf','f2','f4','f8','c4','c8','f*','f*','c*'],
        ['c8','c8','c8','c8','c8','c8','c8','c8','c8','c4','c4','c4','c8','c4','c8','c*','c*','c*'],
      ]
    else:
      expected = [
        ['b1','u1','u2','u4','u4','i1','i2','i4','i4','bf','f2','f4','f4','c4','c4','i4','f4','c4'],
        ['u1','u1','u2','u4','u4','i2','i2','i4','i4','bf','f2','f4','f4','c4','c4','u1','f4','c4'],
        ['u2','u2','u2','u4','u4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','u2','f4','c4'],
        ['u4','u4','u4','u4','u4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','u4','f4','c4'],
        ['u4','u4','u4','u4','u4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','u4','f4','c4'],
        ['i1','i2','i4','i4','i4','i1','i2','i4','i4','bf','f2','f4','f4','c4','c4','i1','f4','c4'],
        ['i2','i2','i4','i4','i4','i2','i2','i4','i4','bf','f2','f4','f4','c4','c4','i2','f4','c4'],
        ['i4','i4','i4','i4','i4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','i4','f4','c4'],
        ['i4','i4','i4','i4','i4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','i4','f4','c4'],
        ['bf','bf','bf','bf','bf','bf','bf','bf','bf','bf','f4','f4','f4','c4','c4','bf','bf','c4'],
        ['f2','f2','f2','f2','f2','f2','f2','f2','f2','f4','f2','f4','f4','c4','c4','f2','f2','c4'],
        ['f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','c4','c4','f4','f4','c4'],
        ['f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','c4','c4','f4','f4','c4'],
        ['c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4'],
        ['c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4'],
        ['i4','u1','u2','u4','u4','i1','i2','i4','i4','bf','f2','f4','f4','c4','c4','i*','f*','c*'],
        ['f4','f4','f4','f4','f4','f4','f4','f4','f4','bf','f2','f4','f4','c4','c4','f*','f*','c*'],
        ['c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c*','c*','c*'],
      ]
    typecode_to_dtype = {
      'b1': jnp.bool_,
      'u1': jnp.uint8, 'u2': jnp.uint16, 'u4': jnp.uint32, 'u8': jnp.uint64,
      'i1': jnp.int8, 'i2': jnp.int16, 'i4': jnp.int32, 'i8': jnp.int64,
      'bf': jnp.bfloat16, 'f2': jnp.float16, 'f4': jnp.float32, 'f8': jnp.float64,
      'c4': jnp.complex64, 'c8': jnp.complex128,
      'i*': jnp.int64, 'f*': jnp.float64, 'c*': jnp.complex128,
    }
    dtype_to_typecode = {jnp.dtype(v): k for k, v in typecode_to_dtype.items()
                        if not k.endswith('*')}

    def typecode_to_val(typecode):
      weak_type = typecode.endswith('*')
      dtype = typecode_to_dtype[typecode]
      val = dtype(0)
      if weak_type:
        val = val.item()
      return val

    def val_to_typecode(val):
      dtype = dtypes.result_type(val)
      weak_type = dtypes.is_weakly_typed(val)
      typecode = dtype_to_typecode[dtype]
      if weak_type:
        typecode = typecode[:-1] + '*'
      return typecode

    vals = [typecode_to_val(t) for t in typecodes]
    table = [[val_to_typecode(v1 + v2) for v1 in vals] for v2 in vals]

    def show_differences(epected, actual):
      diffs = ""
      for i, t1 in enumerate(typecodes):
        for j, t2 in enumerate(typecodes):
          if expected[i][j] != actual[i][j]:
            diffs += f"\n{t1}, {t2} -> want {expected[i][j]}, got {actual[i][j]}"
      return diffs

    self.assertEqual(table, expected, show_differences(expected, table))

# TODO(jakevdp): re-apply #4850 after rollback
#   @parameterized.named_parameters(
#     {"testcase_name": "_xtype={}_ytype={}_xfun={}_yfun={}".format(
#       xtype.__name__, ytype.__name__, xfun.__name__, yfun.__name__),
#      "xtype": xtype, "ytype": ytype, "xfun": xfun, "yfun": yfun}
#     for xtype, ytype in itertools.product(
#       [int, float, jnp.int16, jnp.int32, jnp.float16, jnp.float32], repeat=2)
#     for xfun, yfun in itertools.product(
#       [identity, abs, jnp.array], repeat=2)
#     )
#   def testBinaryPromotionJitInvariance(self, xtype, ytype, xfun, yfun):
#     """Test jit invariance of simple binary promotion rules with and without weak types."""
#     f = lambda x, y: xfun(x) + yfun(y)
#     args_maker = lambda: [xtype(1), ytype(1)]
#     self._CompileAndCheck(f, args_maker, check_dtypes=True)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
