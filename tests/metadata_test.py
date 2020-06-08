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

from unittest import SkipTest

from absl.testing import absltest
from jax import test_util as jtu

import jax
from jax import numpy as jnp

from jax.config import config
config.parse_flags_with_absl()

class MetadataTest(jtu.JaxTestCase):

  def test_jit_metadata(self):
    hlo = jax.xla_computation(jnp.sin)(1.).get_hlo_module().to_string()
    self.assertRegex(hlo, 'op_type="sin"')
    self.assertRegex(hlo, 'op_name="xla_computation\\(sin\\)/sin"')
    def foo(x):
      return jnp.sin(x)
    hlo = jax.xla_computation(foo)(1.).get_hlo_module().to_string()
    self.assertRegex(hlo, 'op_type="sin"')
    self.assertRegex(hlo, 'op_name="xla_computation\\(foo\\)/sin"')

  def test_nested_jit_metadata(self):
    raise SkipTest              # TODO(jekbradbury)
    @jax.jit
    def foo(x):
      return jnp.sin(x)
    def bar(x):
      return jnp.cos(foo(x))
    _ = bar(1.)
    assert self.op_types[-2] == 'sin'
    assert self.op_names[-2] == 'jit(foo)/sin'
    assert self.op_types[-1] == 'cos'
    assert self.op_names[-1] == 'cos'
    _ = jax.jit(bar)(1.)
    assert self.op_types[-3] == 'xla_call'
    assert self.op_names[-3] == 'jit(bar)/xla_call[ backend=None\n' \
                                '                   device=None\n' \
                                '                   name=foo ]'
    assert self.op_types[-2] == 'sin'
    assert self.op_names[-2] == 'jit(bar)/jit(foo)/sin'
    assert self.op_types[-1] == 'cos'
    assert self.op_names[-1] == 'jit(bar)/cos'

  def test_grad_jit_metadata(self):
    @jax.jit
    def foo(x):
      return jnp.sin(x)
    hlo = jax.xla_computation(jax.grad(foo))(1.).get_hlo_module().to_string()
    self.assertRegex(hlo, 'op_type="sin"')
    self.assertRegex(hlo, 'op_type="cos"')
    self.assertRegex(hlo, 'op_type="mul"')
    # TODO(mattjj,jekbradbury): update these tests post-omnistaging
    # self.assertRegex(hlo, 'op_name=".*jit\\(jvp\\(foo\\)\\)/sin"')
    # self.assertRegex(hlo, 'op_name=".*jit\\(jvp\\(foo\\)\\)/cos"')
    # self.assertRegex(hlo, 'op_name=".*jit\\(transpose\\('
    #                       'jvp\\(foo\\)\\)\\)/mul"')

  def test_cond_metadata(self):
    def true_fun(x):
      return jnp.sin(x)
    def false_fun(x):
      return jnp.cos(x)
    def f(x):
      return jax.lax.cond(True, x, true_fun, x, false_fun)
    hlo = jax.xla_computation(f)(1.).get_hlo_module().to_string()
    self.assertRegex(hlo, 'op_type="cond"')
    self.assertRegex(hlo, 'op_name=".*cond\\[ linear=\\(False, False\\) \\]"')
    self.assertRegex(hlo, 'op_type="cos"')
    self.assertRegex(hlo, 'op_name=".*cond/branch_0_fun/cos"')
    self.assertRegex(hlo, 'op_type="sin"')
    self.assertRegex(hlo, 'op_name=".*cond/branch_1_fun/sin"')


if __name__ == "__main__":
  absltest.main()
