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
from jax import collect, inject, grad_intermediates, vmap, jit, grad, Scope
from jax import lax, random
import jax.numpy as np

from jax.config import config
config.parse_flags_with_absl()

class TaggingTest(jtu.JaxTestCase):

  def test_collect(self):
    def foo(scope, x):
      y = lax.tag(x ** 2, scope, "y")
      z = y + 1
      return z
    scope = Scope()
    val, tree = collect(foo)(scope, 2.)
    self.assertAllClose(val, foo(scope, 2.), check_dtypes=True)
    self.assertAllClose(tree, {"y": 4.}, check_dtypes=True)
  
  def test_collect_jit(self):
    def foo(scope, x):
      y = lax.tag(x ** 2, scope, "y")
      z = y + 1
      return z
    scope = Scope()
    val, tree = collect(jit(foo))(scope, 2.)
    self.assertAllClose(val, foo(scope, 2.), check_dtypes=True)
    self.assertAllClose(tree, {"y": 4.}, check_dtypes=True)
  
  def test_jit_collect(self):
    def foo(scope, x):
      y = lax.tag(x ** 2, scope, "y")
      z = y + 1
      return z
    scope = Scope()
    val, tree = jit(collect(foo))(scope, 2.)
    self.assertAllClose(val, foo(scope, 2.), check_dtypes=True)
    self.assertAllClose(tree, {"y": 4.}, check_dtypes=True)

  def test_collect_nested(self):
    def bar(scope, key, x):
      scope, key = lax.push_scope(scope, "y"), random.fold_in(key, "y")
      y = lax.tag(random.normal(key), scope)
      return x + y
    def baz(scope, key, x):
      scope_1, key_1 = lax.push_scope(scope, 1), random.fold_in(key, 1)
      a = bar(scope_1, key_1, x)
      scope_2, key_2 = lax.push_scope(scope, 2), random.fold_in(key, 2)
      b = bar(scope_2, key_2, x)
      return a + b
    scope = Scope()
    key = random.PRNGKey(0)
    val, tree = collect(baz)(scope, key, 2.)
    self.assertAllClose(val, baz(scope, key, 2.), check_dtypes=True)
    expected = {1: {"y": random.normal(random.fold_in(random.fold_in(key, 1), "y"))},
                2: {"y": random.normal(random.fold_in(random.fold_in(key, 2), "y"))}}
    self.assertAllClose(tree, expected, check_dtypes=True)

  def test_inject(self):
    def foo(scope, x):
      y = lax.tag(x ** 2, scope, "y")
      z = y + 1
      return z
    scope = Scope()
    val = inject(foo, {"y": 1.})(scope, 2.)
    self.assertAllClose(val, 2., check_dtypes=True)

  def test_inject_jit(self):
    def foo(scope, x):
      y = lax.tag(x ** 2, scope, "y")
      z = y + 1
      return z
    scope = Scope()
    val = inject(jit(foo), {"y": 1.})(scope, 2.)
    self.assertAllClose(val, 2., check_dtypes=True)
  
  def test_jit_inject(self):
    def foo(scope, x):
      y = lax.tag(x ** 2, scope, "y")
      z = y + 1
      return z
    scope = Scope()
    val = jit(inject(foo, {"y": 1.}))(scope, 2.)
    self.assertAllClose(val, 2., check_dtypes=True)

  def test_inject_nested(self):
    def bar(scope, key, x):
      scope, key = lax.push_scope(scope, "y"), random.fold_in(key, "y")
      y = lax.tag(random.normal(key), scope)
      return x + y
    def baz(scope, key, x):
      scope_1, key_1 = lax.push_scope(scope, 1), random.fold_in(key, 1)
      a = bar(scope_1, key_1, x)
      scope_2, key_2 = lax.push_scope(scope, 2), random.fold_in(key, 2)
      b = bar(scope_2, key_2, x)
      return a + b
    scope = Scope()
    key = random.PRNGKey(0)
    val = inject(baz, {1: {"y": 0.3}, 2: {"y": 0.5}})(scope, key, 2.)
    self.assertAllClose(val, 4.8, check_dtypes=True)

  def test_grad_intermediates(self):
    def foo(scope, x):
      y = lax.tag(x ** 2, scope, "y")
      z = y + 1
      return z
    scope = Scope() 
    val = grad_intermediates(foo)(scope, 2.)
    expected = {"y": 1.}
    self.assertAllClose(val, expected, check_dtypes=True)

if __name__ == "__main__":
  absltest.main()
