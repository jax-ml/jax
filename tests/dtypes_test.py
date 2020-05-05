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
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import core
from jax import dtypes
from jax import numpy as jnp
from jax import test_util as jtu
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

bool_dtypes = [np.dtype('bool')]

signed_dtypes = [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
                 np.dtype('int64')]

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

class DtypesTest(jtu.JaxTestCase):

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

  @parameterized.named_parameters(
    {"testcase_name": "_swap={}_jit={}".format(swap, jit),
     "swap": swap, "jit": jit} 
    for swap in [False, True] for jit in [False, True])
  @jtu.skip_on_devices("tpu")  # F16 not supported on TPU
  def testBinaryPromotion(self, swap, jit):
    testcases = [
      (jnp.array(1.), 0., jnp.float_),
      (jnp.array(1.), jnp.array(0.), jnp.float_),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float16), jnp.float_),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float32), jnp.float_),
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
      z = x + y
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
    with core.skipping_checks():
      # Passing AnEnum.A to jnp.array fails the type check in bind
      np.testing.assert_equal(jnp.array(42), jnp.array(AnEnum.A))
    np.testing.assert_equal(np.int32(101), np.int32(AnEnum.B))
    np.testing.assert_equal(jnp.int32(101), jnp.int32(AnEnum.B))

if __name__ == "__main__":
  absltest.main()
