# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
from absl.testing import absltest, parameterized

from jax import numpy as np, test_util as jtu, lax, api, random, vmap, jit, \
  shapecheck

from jax.abstract_arrays import Poly, Mon
from jax.api import _parse_shape_spec, ShapeError

from jax.scipy.special import expit

from jax.config import config
config.parse_flags_with_absl()


# These are 'manual' tests for shape checking. The more exhaustive,
# more systematic tests should live in lax_test.py.

def const_poly(c):
  return Poly({Mon(): c})

class ShapesTest(jtu.JaxTestCase):

  @parameterized.parameters([
      ['(m, n)', 'ShapeSpec(m, n)'],
      ['(m * n)', 'ShapeSpec(m n)'],
      ['m * n', 'ShapeSpec(m n)'],
      ['(m * n,)', 'ShapeSpec(m n)'],
      ['(3, m)', 'ShapeSpec(3, m)'],
      ['(10, m)', 'ShapeSpec(10, m)'],
      ['(-10, m)', 'ShapeSpec(-10, m)'],
      ['(3 * m)', 'ShapeSpec(3 m)'],
      ['m', 'ShapeSpec(m)'],
      ['', 'ShapeSpec()'],
      ['m + n', 'ShapeSpec(m + n)'],
      ['m + n * k', 'ShapeSpec(m + k n)'],
      ['m + 3 * k', 'ShapeSpec(3 k + m)'],
      ['', 'ShapeSpec()'],
      ['_', 'ShapeSpec(_)'],
  ])
  def test_parse_spec(self, spec, ans):
    self.assertEqual(str(_parse_shape_spec(spec)), ans)

  def test_Poly_equal(self):
    assert const_poly(3) == 3
    assert onp.array(3, onp.int64) == const_poly(3)
    assert onp.array(3, onp.int64)[()] == const_poly(3)
    assert not onp.array(3, onp.int64) != const_poly(3)
    assert const_poly(4) != 3
    assert 3 == const_poly(3)
    assert 4 != const_poly(3)
    assert const_poly(4) == const_poly(4)
    assert const_poly(3) != const_poly(4)
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) == Poly({Mon({'n': 1}): 4, Mon(): 3})
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 2}): 4})
    assert Poly({Mon(): 3, Mon({'m': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 1}): 4})

  def test_Poly_hash(self):
    assert not len(set(hash(Poly({Mon(): i})) for i in range(10))) == 1
    assert hash(Poly({Mon(): 3, Mon({'n': 1}): 4})) == hash(Poly({Mon({'n': 1}): 4, Mon(): 3}))

  def test_Poly_compare(self):
    poly = Poly({Mon(): 3, Mon({'n': 1}): 4})
    # Assume poly > 0 to make various shape rules work with polymorphic shapes:
    assert poly >= 0
    assert poly >= 1
    assert poly > 0

    assert 0 <= poly
    assert 0 < poly
    assert const_poly(3) >= 1
    assert const_poly(3) > 1
    self.assertRaisesRegex(ValueError, "", lambda: poly >= 2)
    self.assertRaisesRegex(ValueError, "", lambda: poly > 1)

  def test_Poly_divmod(self):
    n = Poly({Mon({'n': 1}): 1})
    assert (n, 1) == divmod(2*n+1, 2)
    assert (2*n, 0) == divmod(10*n, 5)
    assert (2*n+4, 3) == divmod(10*n+23, 5)

  def test_add_broadcast(self):
     @shapecheck(['(m, n)', 'n'], '(m, n)')
     @shapecheck(['n', ''], 'n')
     def add(a, b):
       return a + b

  def test_sum(self):
    @shapecheck(['(m, n)'], '')
    def sum(x):
      return np.sum(x)

  def test_prod(self):
    @shapecheck(['(m, n)'], '')
    def prod(x):
      return np.prod(x)

  def test_max(self):
    @shapecheck(['(m, n)'], '')
    def prod(x):
      return np.max(x)

  def test_min(self):
    @shapecheck(['(m, n)'], '')
    def prod(x):
      return np.min(x)

  def test_dot(self):
    @shapecheck(['(m, n)', 'n'], 'm')
    def matvec(A, b):
      return np.dot(A, b)

    def thunk():
      @shapecheck(['(m, n)', 'n'], 'm')
      def matvec(A, b):
        return lax.dot_general(A, b, [((0,), (0,)), ((), ())])
    self.assertRaisesRegex(TypeError, "", thunk)

  def test_flatten(self):
    @shapecheck(['(m, n)'], 'm * n')
    def flatten(x):
      return lax.reshape(x, (x.shape[0] * x.shape[1],))

  def test_concatenate(self):
    @shapecheck(['m', 'n', 'm'], '3*m + n')
    def cat(x, y, z):
      return lax.concatenate([x, y, x, z], 0)

    def thunk():
      @shapecheck(['m', 'n', 'm'], '3*m + n')
      def cat(x, y, z):
        return lax.concatenate([x, y, x], 0)
    self.assertRaisesRegex(ShapeError, "", thunk)

  def test_device_put(self):
    @shapecheck(['n'], 'n')
    def d_put(x):
      return api.device_put(x)

  def test_broadcast_in_dim(self):
    x = np.zeros((7, 1))
    lax.broadcast_in_dim(x, shape=(3, x.shape[0], 4), broadcast_dimensions=(1, 2))
    @shapecheck(['(n, 1)'], '(3, n, 4)')
    def broadcast_in_dim(x):
      return lax.broadcast_in_dim(x, shape=(3, x.shape[0], 4), broadcast_dimensions=(1, 2))

  def test_jit(self):
    @shapecheck(['n'], '2*n')
    @jit
    def concat(x):
      return lax.concatenate([x, x], 0)

    # TODO:
    # @shapecheck(['n'], 'n')
    # @jit
    # @grad
    # def sum_square(x):
    #   return np.sum(x ** 2)

  def test_pad(self):
    @shapecheck(['n'], '2*n+1')
    def p(x):
      return lax.pad(x, np.array(0., x.dtype), [(1, 1, 1)])

  def test_numpy_pad(self):
    @shapecheck(['n'], 'n+1')
    def p(x):
      return np.pad(x, (0, 1))

  @parameterized.named_parameters(jtu.cases_from_list(
    {
      'testcase_name': "strides={}_padding={}_lhs_dilation={}_dimension_numbers"
                       "={}_lhs_perm={}_rhs_perm={}_out_perm={}".format(
        strides, padding, lhs_dilation, dimension_numbers, lhs_perm, rhs_perm, out_perm),
      'strides': strides, 'padding': padding, 'lhs_dilation': lhs_dilation,
      'dimension_numbers': dimension_numbers, 'lhs_perm': lhs_perm,
      'rhs_perm': rhs_perm, 'out_perm': out_perm}
    for strides in [(1, 1), (2, 1)]
    for padding in ['SAME', 'VALID', ((1, 0), (2, 0))]
    for lhs_dilation in (None, (1, 2))
    for dimension_numbers, (lhs_perm, rhs_perm, out_perm) in (
            (("NCHW", "OIHW", "NCHW"), ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))),
            (("NHWC", "HWIO", "NHWC"), ((0, 2, 3, 1), (2, 3, 1, 0), (0, 2, 3, 1))),
            (("NCHW", "HWIO", "NHWC"), ((0, 1, 2, 3), (2, 3, 1, 0), (0, 2, 3, 1)))
    )
    # String padding is not implemented for transposed convolution, see conv_general_dilated implementation:
    if (lhs_dilation is None or not isinstance(padding, str)) and
    # only test strides with same padding:
    (strides[0] == 1 or padding == 'SAME')))
  def test_conv(self, strides, padding, lhs_dilation,
                           dimension_numbers, lhs_perm, rhs_perm, out_perm):
    valid = padding == 'VALID'
    is_strided = strides[0] != 1
    lhs_shape = '({}, {}, {}, {})'.format(*onp.take(['n', 'i', '2*h' if is_strided else 'h', 'w'], lhs_perm))
    rhs_shape = '({}, {}, {}, {})'.format(*onp.take(['o', 'i', '2', '3'], rhs_perm))
    out_shape = '({}, {}, {}, {})'.format(*onp.take([
      'n', 'o', 'h+-1' if valid and not is_strided else 'h',
      ('w+-2' if valid else 'w') if lhs_dilation is None else '2*w+-1'], out_perm))

    @shapecheck([lhs_shape, rhs_shape], out_shape)
    def conv(lhs, rhs):
      return lax.conv_general_dilated(
        lhs, rhs, strides, padding,
        lhs_dilation=lhs_dilation, dimension_numbers=dimension_numbers)

  def test_indexing(self):
    @shapecheck(['n'], '')
    def first(x):
      return x[0]

    @shapecheck(['n'], '')
    def last(x):
      return x[-1]

    @shapecheck(['(n,m,a)'], 'n,m')
    @vmap
    @shapecheck(['(n,a)'], 'n')
    def last_column(x):
      return x[..., -1]

  def test_slicing(self):
    @shapecheck(['n'], 'n+-1')
    def slice(x):
      return x[1:]

    @shapecheck(['n'], 'n+-1')
    def slice(x):
      return x[:-1]

  def test_split(self):
    @shapecheck(['2*n'], ['n', 'n'])
    def split_half(x):
      return np.split(x, 2)

    @shapecheck(['n'], ['10', 'n+-10'])
    def split_after_ten(x):
      return np.split(x, [10])

  def test_iota(self):
    @shapecheck(['n'], 'n')
    def range_like(x):
      return lax.iota(np.int32, x.shape[0])

  def test_arange(self):
    @shapecheck(['n'], 'n')
    def arange_like(x):
      return np.arange(x.shape[0], dtype=np.int32)

  def test_expit(self):
    @shapecheck(['n'], 'n')
    def expit_(x):
      return expit(x)

  def test_reshape(self):
    @shapecheck(['n, a, b'], 'n, a*b')
    def flatten(x):
      return np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

  def test_uniform(self):
    @shapecheck(['2', 'n'], 'n')
    def uniform_like(key, x):
      return random.uniform(key, x.shape)

  def test_ravel(self):
    a = np.array(1)

    @shapecheck(['n'], '')
    def thunk(n):
      return -(a + n.ravel()[0] * 0)


if __name__ == '__main__':
  absltest.main()
