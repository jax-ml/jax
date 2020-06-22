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

import operator

from absl.testing import absltest
from absl.testing import parameterized

from jax import numpy as jnp
from jax import test_util as jtu
from jax.experimental.doubledouble import doubledouble, _DoubleDouble

from jax.config import config, flags
config.parse_flags_with_absl()

FLAGS = flags.FLAGS

class DoubleDoubleTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_{}".format(
        op.__name__, jtu.format_shape_dtype_string(shape, dtype)),
        "dtype": dtype, "shape": shape, "op": op}
    for dtype in (jnp.float16, jnp.float32, jnp.float64)
    for shape in ((), (5,), (2, 3), (2, 3, 4))
    for op in (abs, operator.neg, operator.pos, jnp.sqrt)))
  def testUnaryOp(self, dtype, shape, op):
    rng = jtu.rand_default(self.rng())
    op_doubled = doubledouble(op)
    args = (rng(shape, dtype),)
    self.assertAllClose(op(*args), op_doubled(*args))
  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_{}".format(
        op.__name__, jtu.format_shape_dtype_string(shape, dtype)),
        "dtype": dtype, "shape": shape, "op": op}
    for dtype in (jnp.float16, jnp.float32, jnp.float64)
    for shape in ((), (5,), (2, 3), (2, 3, 4))
    for op in (operator.add, operator.sub, operator.mul, operator.truediv,
               operator.gt, operator.ge, operator.lt, operator.le,
               operator.eq, operator.ne)))
  def testBinaryOp(self, dtype, shape, op):
    rng = jtu.rand_default(self.rng())
    op_doubled = doubledouble(op)
    args = rng(shape, dtype), rng(shape, dtype)
    self.assertAllClose(op(*args), op_doubled(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_{}".format(
      jtu.format_shape_dtype_string(shape, dtype), label),
      "shape": shape, "dtype": dtype, "op1": op1, "op2": op2
    }
    for dtype in (jnp.float32, jnp.float64)
    for shape in ((), (5,), (2, 3), (2, 3, 4))
    for label, op1, op2 in [
      ('add_sub', lambda x, y: x + y - x, lambda x, y: y),
      ("add_neg_add", lambda x, y: -(x + y) + x, lambda x, y: -y),
      ("add_mul_sub", lambda x, y: 2 * (x + y) - 2 * x, lambda x, y: 2 * y),
      ("add_div_sub", lambda x, y: (x + y) / 2 - x / 2, lambda x, y: y / 2),
    ]))
  def testDoubledPrecision(self, shape, dtype, op1, op2):
    """Test operations that would lose precision without doubling."""
    rng = jtu.rand_default(self.rng())
    double_op1 = doubledouble(op1)
    args = 1E20 * rng(shape, dtype), rng(shape, dtype)
    check_dtypes = not FLAGS.jax_enable_x64

    self.assertAllClose(double_op1(*args), op2(*args), check_dtypes=check_dtypes)

    # Sanity check: make sure test fails for regular precision.
    with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
      self.assertAllClose(op1(*args), op2(*args), check_dtypes=check_dtypes)
  def testTypeConversion(self):
    x = jnp.arange(10, dtype='float16')
    f = lambda x, y: (x + y).astype('float32')
    g = doubledouble(f)
    self.assertAllClose(f(1E2 * x, 1E-2 * x), 1E2 * x.astype('float32'))
    self.assertAllClose(g(1E2 * x, 1E-2 * x), 100.01 * x.astype('float32'))

  def testRepeatedDoubling(self):
    def f(x, y, z):
      return x + y + z - x - y
    f2 = doubledouble(f)
    f4 = doubledouble(f2)
    dtype = jnp.float32
    x, y, z = dtype(1E20), dtype(1.0), dtype(1E-20)

    self.assertEqual(f(x, y, z), -y)
    self.assertEqual(f2(x, y, z), 0)
    self.assertEqual(f4(x, y, z), z)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_{}".format(dtype, val), "dtype": dtype, "val": val}
    for dtype in ["float16", "float32", "float64"]
    for val in ["6.0221409e23", "3.14159265358", "0", 123456789]
  ))
  def testClassInstantiation(self, dtype, val):
    dtype = jnp.dtype(dtype).type
    self.assertEqual(dtype(val), _DoubleDouble(val, dtype).to_array())

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_{}".format(
      jtu.format_shape_dtype_string(shape, dtype), op.__name__),
      "shape": shape, "dtype": dtype, "op": op
    }
    for dtype in (jnp.float32, jnp.float64)
    for shape in ((), (5,), (2, 3), (2, 3, 4))
    for op in (operator.neg, operator.abs)
  ))
  def testClassUnaryOp(self, dtype, shape, op):
    rng = jtu.rand_default(self.rng())
    args = (rng(shape, dtype),)
<<<<<<< HEAD
    class_op = lambda x: op(_DoubleDouble(x)).to_array()
=======
    class_op = lambda x: op(DoubleDouble(x)).to_array()
>>>>>>> b66fdf70... Add class wrapper for double-double arithmetic
    self.assertAllClose(op(*args), class_op(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_{}".format(
      jtu.format_shape_dtype_string(shape, dtype), op.__name__),
      "shape": shape, "dtype": dtype, "op": op
    }
    for dtype in (jnp.float32, jnp.float64)
    for shape in ((), (5,), (2, 3), (2, 3, 4))
    for op in (operator.add, operator.sub, operator.mul, operator.truediv,
               operator.gt, operator.ge, operator.lt, operator.le,
               operator.eq, operator.ne)
  ))
  def testClassBinaryOp(self, dtype, shape, op):
    rng = jtu.rand_default(self.rng())
    args = rng(shape, dtype), rng(shape, dtype)
    def class_op(x, y):
      result = op(_DoubleDouble(x), _DoubleDouble(y))
      if isinstance(result, _DoubleDouble):
        result = result.to_array()
      return result
    self.assertAllClose(op(*args), class_op(*args))


if __name__ == '__main__':
  absltest.main()
