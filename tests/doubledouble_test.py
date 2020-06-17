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
from jax.experimental import doubledouble

from jax.config import config
config.parse_flags_with_absl()

class DoubleDoubleTest(jtu.JaxTestCase):
    @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_{}_{}".format(
            op.__name__, jtu.format_shape_dtype_string(shape, dtype)),
          "dtype": dtype, "shape": shape, "op": op}
        for dtype in (jnp.float16, jnp.float32, jnp.float64)
        for shape in ((), (5,), (2, 3), (2, 3, 4))
        for op in (abs, operator.neg, jnp.sqrt)))
    def testUnaryOp(self, dtype, shape, op):
        rng = jtu.rand_default(self.rng())
        op_doubled = doubledouble.doubledouble(op)
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
        op_doubled = doubledouble.doubledouble(op)
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
            ("add_lt", lambda x, y: x + y < x, lambda x, y: y < 0)
        ]))
    def testDoubledPrecision(self, shape, dtype, op1, op2):
        """Test operations that would lose precision without doubling."""
        rng = jtu.rand_default(self.rng())
        double_op1 = doubledouble.doubledouble(op1)
        args = 1E20 * rng(shape, dtype), rng(shape, dtype)

        self.assertAllClose(double_op1(*args), op2(*args))

        # Sanity check: make sure test fails for regular precision.
        with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
            self.assertAllClose(op1(*args), op2(*args))


if __name__ == '__main__':
  absltest.main()
