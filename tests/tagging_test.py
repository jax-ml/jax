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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from jax import test_util as jtu
from jax import collect, inject, grad_intermediates, vmap, jit, grad
from jax import lax, random
import jax.numpy as np

from jax.config import config
config.parse_flags_with_absl()

class TaggingTest(jtu.JaxTestCase):

  def test_collect(self):
    def foo(x):
      y = lax.tag(x ** 2, "y")
      z = y + 1
      return z
    val, tree = collect(foo)(2.)
    self.assertAllClose(val, foo(2.), check_dtypes=True)
    self.assertAllClose(tree, {"y": 4.}, check_dtypes=True)
  
  def test_collect_jit(self):
    def foo(x):
      y = lax.tag(x ** 2, "y")
      z = y + 1
      return z
    self.skipTest("collect(jit) doesn't work yet")
    val, tree = collect(jit(foo))(2.)
    self.assertAllClose(val, foo(2.), check_dtypes=True)
    self.assertAllClose(tree, {"y": 4.}, check_dtypes=True)
  
  def test_jit_collect(self):
    def foo(x):
      y = lax.tag(x ** 2, "y")
      z = y + 1
      return z
    val, tree = jit(collect(foo))(2.)
    self.assertAllClose(val, foo(2.), check_dtypes=True)
    self.assertAllClose(tree, {"y": 4.}, check_dtypes=True)

  def test_collect_nested(self):
    def bar(key, x):
      y = lax.tag(random.normal(random.fold_in(key, "y")))
      return x + y
    def baz(key, x):
      a = bar(random.fold_in(key, 1), x)
      b = bar(random.fold_in(key, 2), x)
      return a + b
    key = random.PRNGKey(0)
    val, tree = collect(baz)(key, 2.)
    self.assertAllClose(val, baz(key, 2.), check_dtypes=True)
    expected = {1: {"y": random.normal(random.fold_in(random.fold_in(key, 1), "y"))},
                2: {"y": random.normal(random.fold_in(random.fold_in(key, 2), "y"))}}
    self.assertAllClose(tree, expected, check_dtypes=True)

  def test_inject(self):
    def foo(x):
      y = lax.tag(x ** 2, "y")
      z = y + 1
      return z
    val = inject(foo, {"y": 1.})(2.)
    self.assertAllClose(val, 2., check_dtypes=True)

  def test_inject_jit(self):
    def foo(x):
      y = lax.tag(x ** 2, "y")
      z = y + 1
      return z
    val = inject(jit(foo), {"y": 1.})(2.)
    self.assertAllClose(val, 2., check_dtypes=True)
  
  def test_jit_inject(self):
    def foo(x):
      y = lax.tag(x ** 2, "y")
      z = y + 1
      return z
    val = jit(inject(foo, {"y": 1.}))(2.)
    self.assertAllClose(val, 2., check_dtypes=True)

  def test_inject_nested(self):
    def bar(key, x):
      y = lax.tag(random.normal(random.fold_in(key, "y")))
      return x + y
    def baz(key, x):
      a = bar(random.fold_in(key, 1), x)
      b = bar(random.fold_in(key, 2), x)
      return a + b
    key = random.PRNGKey(0)
    val = inject(baz, {1: {"y": 0.3}, 2: {"y": 0.5}})(key, 2.)
    self.assertAllClose(val, 4.8, check_dtypes=True)

  def test_grad_intermediates(self):
    def foo(x):
      y = lax.tag(x ** 2, "y")
      z = y + 1
      return z
    val = grad_intermediates(foo)(2.)
    expected = {"y": 1.}
    self.assertAllClose(val, expected, check_dtypes=True)

if __name__ == "__main__":
  absltest.main()
