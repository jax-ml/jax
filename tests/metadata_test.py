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


from absl.testing import absltest
from jax import test_util as jtu
from jax.interpreters import xla
from jax.lib import xla_bridge as xb

import jax
from jax import numpy as jnp

from jax.config import config
config.parse_flags_with_absl()

class MetadataTest(jtu.JaxTestCase):

  def setUp(self):
    xla.xla_primitive_callable.cache_clear()
    xla._xla_callable.cache_clear()
    self.op_names = []
    self.op_types = []
    def SetOpMetadata(builder, metadata):
      self.op_names.append(metadata.op_name)
      self.op_types.append(metadata.op_type)
      return super(xb._JaxComputationBuilder, builder).SetOpMetadata(metadata)
    xb._JaxComputationBuilder.SetOpMetadata = SetOpMetadata

  def tearDown(self):
    self.op_names = []
    self.op_types = []
    del xb._JaxComputationBuilder.SetOpMetadata

  def test_primitive_metadata(self):
    _ = jnp.sin(1.)
    assert self.op_types[-1] == 'sin'
    assert self.op_names[-1] == 'sin'
    _ = jnp.reshape(1., (1,))
    assert self.op_types[-1] == 'reshape'
    assert self.op_names[-1] == 'reshape[ dimensions=None\n' \
                                '         new_sizes=(1,)\n' \
                                '         old_sizes=() ]'

  def test_jit_metadata(self):
    _ = jax.jit(jnp.sin)(1.)
    assert self.op_types[-1] == 'sin'
    assert self.op_names[-1] == 'jit(sin)/sin'
    def foo(x):
      return jnp.sin(x)
    _ = jax.jit(foo)(1.)
    assert self.op_types[-1] == 'sin'
    assert self.op_names[-1] == 'jit(foo)/sin'

  def test_nested_jit_metadata(self):
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
    _ = jax.grad(foo)(1.)
    assert self.op_types[-3] == 'sin'
    assert self.op_names[-3] == 'jit(pe(jvp(foo)))/sin'
    assert self.op_types[-2] == 'cos'
    assert self.op_names[-2] == 'jit(pe(jvp(foo)))/cos'
    assert self.op_types[-1] == 'mul'
    assert self.op_names[-1] == 'jit(transpose(pe(jvp(foo))))/mul'

  def test_cond_metadata(self):
    def true_fun(x):
      return jnp.sin(x)
    def false_fun(x):
      return jnp.cos(x)
    _ = jax.lax.cond(True, 1., true_fun, 1., false_fun)
    assert self.op_types[-3] == 'cond'
    assert self.op_names[-3] == 'cond[ false_nconsts=0\n' \
                                '      true_nconsts=0 ]'
    assert self.op_types[-2] == 'sin'
    assert self.op_names[-2] == 'cond/true_fun/sin'
    assert self.op_types[-1] == 'cos'
    assert self.op_names[-1] == 'cond/false_fun/cos'


if __name__ == "__main__":
  absltest.main()
