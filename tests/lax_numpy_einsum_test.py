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
from absl.testing import absltest

import jax.numpy as np
import jax.test_util as jtu

from jax.config import config
config.parse_flags_with_absl()


def rng():
  return onp.random.RandomState(0)

def check(s, *ops):
  a = onp.einsum(s, *ops)
  b = np.einsum(s, *ops)
  assert onp.allclose(a, b, atol=1e-4, rtol=1e-4)


class EinsumTest(jtu.JaxTestCase):

  def test_three_operands_1(self):
    x = rng().randn(3)
    y = rng().randn(4)
    z = rng().randn(5)
    s = 'i,j,k->ijk'
    check(s, x, y, z)

  def test_three_operands_2(self):
    x = rng().randn(3)
    y = rng().randn(4)
    z = rng().randn(5)
    s = 'i,j,k->ijk'
    check(s, x, y, z)

  def test_two_operands_1(self):
    x = rng().randn(3, 4)
    y = rng().randn(4)
    s = 'ij,j->i'
    check(s, x, y)

  def test_two_operands_2(self):
    x = rng().randn(3, 4, 5)
    y = rng().randn(4)
    s = 'ijk,j->i'
    check(s, x, y)

  def test_two_operands_3(self):
    x = rng().randn(3, 4, 3)
    y = rng().randn(3)
    s = 'iji,i->j'
    check(s, x, y)

  def test_two_operands_4(self):
    x = rng().randn(3, 4)
    y = rng().randn(3, 4)
    s = 'ij,ij->'
    check(s, x, y)

  def test_one_operand_1(self):
    x = rng().randn(3, 4, 5)
    s = 'ijk->j'
    check(s, x)

  def test_one_operand_2(self):
    x = rng().randn(3, 4, 5)
    s = 'ijk->kij'
    check(s, x)

  def test_one_operand_3(self):
    x = rng().randn(3, 4, 5)
    s = 'ijk->ki'
    check(s, x)

  def test_one_operand_4(self):
    x = rng().randn(3, 4, 5)
    s = 'ijk->ki'
    check(s, x)

  def test_one_operand_5(self):
    x = rng().randn(2, 3, 4, 5)
    s = '...ijk->...ki'
    check(s, x)

  def test_one_operand_6(self):
    x = rng().randn(3, 4, 5)
    s = '...ijk->ki'
    check(s, x)

  def test_one_operand_7(self):
    x = rng().randn(3, 3)
    s = 'ii->'
    check(s, x)

  def test_one_operand_8(self):
    x = rng().randn(3, 3)
    s = 'ij->'
    check(s, x)

  def test_one_operand_9(self):
    x = rng().randn(3, 3, 3)
    s = 'iii->'
    check(s, x)

  def test_one_operand_10(self):
    x = rng().randn(3, 3)
    s = 'ii->i'
    check(s, x)

  def test_one_operand_11(self):
    x = rng().randn(3, 3, 4)
    s = 'iij->i'
    check(s, x)

  def test_one_operand_12(self):
    x = rng().randn(3, 3, 3)
    s = 'iii->i'
    check(s, x)

  def test_one_operand_13(self):
    x = rng().randn(3, 3, 5, 4, 4)
    s = 'iijkk->i'
    check(s, x)

  def test_one_operand_14(self):
    x = rng().randn(3, 3, 5, 4, 4)
    s = 'iijkk->ik'
    check(s, x)

  def test_one_operand_15(self):
    x = rng().randn(3, 3, 5, 4, 4)
    s = 'iijkl->il'
    check(s, x)

  def test_one_operand_16(self):
    x = rng().randn(3, 3)
    s = 'ij->ij'
    check(s, x)


if __name__ == '__main__':
  absltest.main()
