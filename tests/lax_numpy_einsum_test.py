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

from collections import defaultdict
import itertools

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
import jax.test_util as jtu

from jax.config import config
config.parse_flags_with_absl()


def rng():
  return onp.random.RandomState(0)


class EinsumTest(jtu.JaxTestCase):

  def _check(self, s, *ops):
    a = onp.einsum(s, *ops)
    b = np.einsum(s, *ops)

    atol = 2e-1 if jtu.device_under_test() == "tpu" else 1e-4
    self.assertAllClose(a, b, atol=atol, rtol=1e-4, check_dtypes=True)

  def test_three_operands_1(self):
    r = rng()
    x = r.randn(3)
    y = r.randn(4)
    z = r.randn(5)
    s = 'i,j,k->ijk'
    self._check(s, x, y, z)

  def test_three_operands_2(self):
    r = rng()
    x = r.randn(3)
    y = r.randn(4)
    z = r.randn(5)
    s = 'i,j,k->ijk'
    self._check(s, x, y, z)

  def test_two_operands_1(self):
    r = rng()
    x = r.randn(3, 4)
    y = r.randn(4)
    s = 'ij,j->i'
    self._check(s, x, y)

  def test_two_operands_2(self):
    r = rng()
    x = r.randn(3, 4, 5)
    y = r.randn(4)
    s = 'ijk,j->i'
    self._check(s, x, y)

  def test_two_operands_3(self):
    r = rng()
    x = r.randn(3, 4, 3)
    y = r.randn(3)
    s = 'iji,i->j'
    self._check(s, x, y)

  def test_two_operands_4(self):
    r = rng()
    x = r.randn(3, 4)
    y = r.randn(3, 4)
    s = 'ij,ij->'
    self._check(s, x, y)

  def test_two_operands_5(self):
    r = rng()
    x = r.randn(10, 2, 3)
    y = r.randn(3, 4)
    s = 'nij,jk->nik'
    self._check(s, x, y)

  def test_two_operands_6(self):
    # based on https://github.com/google/jax/issues/37#issuecomment-448572187
    r = rng()
    x = r.randn(2, 1)
    y = r.randn(2, 3, 4)
    s = 'sa,shb->shab'
    self._check(s, x, y)

  def test_one_operand_1(self):
    r = rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->j'
    self._check(s, x)

  def test_one_operand_2(self):
    r = rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->kij'
    self._check(s, x)

  def test_one_operand_3(self):
    r = rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->ki'
    self._check(s, x)

  def test_one_operand_4(self):
    r = rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->ki'
    self._check(s, x)

  def test_one_operand_5(self):
    r = rng()
    x = r.randn(2, 3, 4, 5)
    s = '...ijk->...ki'
    self._check(s, x)

  def test_one_operand_6(self):
    r = rng()
    x = r.randn(3, 4, 5)
    s = '...ijk->ki'
    self._check(s, x)

  def test_one_operand_7(self):
    r = rng()
    x = r.randn(3, 3)
    s = 'ii->'
    self._check(s, x)

  def test_one_operand_8(self):
    r = rng()
    x = r.randn(3, 3)
    s = 'ij->'
    self._check(s, x)

  def test_one_operand_9(self):
    r = rng()
    x = r.randn(3, 3, 3)
    s = 'iii->'
    self._check(s, x)

  def test_one_operand_10(self):
    r = rng()
    x = r.randn(3, 3)
    s = 'ii->i'
    self._check(s, x)

  def test_one_operand_11(self):
    r = rng()
    x = r.randn(3, 3, 4)
    s = 'iij->i'
    self._check(s, x)

  def test_one_operand_12(self):
    r = rng()
    x = r.randn(3, 3, 3)
    s = 'iii->i'
    self._check(s, x)

  def test_one_operand_13(self):
    r = rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkk->i'
    self._check(s, x)

  def test_one_operand_14(self):
    r = rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkk->ik'
    self._check(s, x)

  def test_one_operand_15(self):
    r = rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkl->il'
    self._check(s, x)

  def test_one_operand_16(self):
    r = rng()
    x = r.randn(3, 3)
    s = 'ij->ij'
    self._check(s, x)

  def test_tf_unsupported_1(self):
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = rng()
    x = r.randn(2, 3, 5, 1)
    y = r.randn(3, 4, 5, 1)
    s = 'ij...,jk...->ik...'
    self._check(s, x, y)

  def test_tf_unsupported_2(self):
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = rng()
    x = r.randn(2, 3, 3)
    y = r.randn(4)
    s = 'ijj,k->ik'
    self._check(s, x, y)

  def test_tf_unsupported_3(self):
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = rng()
    x = r.randn(2, 3)
    y = r.randn(2, 3)
    z = r.randn(3, 4)
    s = 'ij,ij,jk->ik'
    self._check(s, x, y, z)

  # these tests are based on https://github.com/dask/dask/pull/3412/files
  @parameterized.named_parameters(
      {"testcase_name": "_{}_dtype={}".format(einstr, dtype.__name__),
      "einstr": einstr, "dtype": dtype}
      for einstr in [
          'abc,bad->abcd',
          'abcdef,bcdfg->abcdeg',
          'ea,fb,abcd,gc,hd->efgh',
          'ab,b',
          'aa',
          'a,a->',
          'a,a->a',
          'a,a',
          'a,b',
          'a,b,c',
          'a',
          'ba,b',
          'ba,b->',
          'defab,fedbc->defac',
          'ab...,bc...->ac...',
          'a...a',
          'abc...->cba...',
          '...ab->...a',
          'a...a->a...',
          # Following 2 from # https://stackoverflow.com/a/19203475/1611416
          '...abc,...abcd->...d',
          'ab...,b->ab...',
          # https://github.com/dask/dask/pull/3412#discussion_r182413444
          'aa->a',
          'ab,ab,c->c',
          'aab,bc->ac',
          'aab,bcc->ac',
          'fdf,cdd,ccd,afe->ae',
          'fff,fae,bef,def->abd',
      ]
      for dtype in [np.float32, np.int32, np.complex64, np.bool_])
  def test_from_dask(self, einstr, dtype):
    r = jtu.rand_default()
    if '->' in einstr:
      input_str, result_names = einstr.split('->')
    else:
      input_str = einstr
    input_names = input_str.split(',')

    dims = itertools.cycle([2, 3, 4])
    shapes = defaultdict(lambda: next(dims))
    input_shapes = [tuple(shapes[c] for c in names.replace('...', '01'))
                    for names in input_names]
    operands = [r(shape, dtype) for shape in input_shapes]

    self._check(einstr, *operands)

  def test_ordered_front_batch_dim_case(self):
    x = onp.ones((1,8,20,4))
    y = onp.ones((1,8,20,4))
    s = 'ijkl,ijml->ijkm'
    self._check(s, x, y)

  def test_einsum_path(self):
    # just check examples from onp.einsum_path docstring
    a = onp.random.rand(2, 2)
    b = onp.random.rand(2, 5)
    c = onp.random.rand(5, 2)

    path_info = onp.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
    self.assertEqual(str(path_info[0]), "['einsum_path', (1, 2), (0, 1)]")
    self.assertEqual(path_info[1].split('\n')[0],
                     '  Complete contraction:  ij,jk,kl->il')

    # check this doesn't crash
    I = onp.random.rand(10, 10, 10, 10)
    C = onp.random.rand(10, 10)
    onp.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C, optimize='greedy')

  def test_einsum_kpmurphy_example(self):
    # code from an email with @murphyk
    N = 2; C = 3; D = 4; K = 5; T = 6;
    r = rng()
    S = r.randn(N, T, K)
    W = r.randn(K, D)
    V = r.randn(D, C)
    L = onp.zeros((N, C))
    for n in range(N):
      for c in range(C):
        s = 0
        for d in range(D):
          for k in range(K):
            for t in range(T):
                s += S[n,t,k] * W[k,d] * V[d,c]
        L[n,c] = s

    path = np.einsum_path('ntk,kd,dc->nc', S, W, V, optimize='optimal')[0]
    rtol = 1e-2 if jtu.device_under_test() == "tpu" else None
    self.assertAllClose(L, np.einsum('ntk,kd,dc->nc', S, W, V, optimize=path),
                        check_dtypes=False, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
