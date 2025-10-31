# Copyright 2025 The JAX Authors.
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

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

from jax.experimental.scheduling_groups import (
    scheduling_group, xla_metadata_call)

jax.config.parse_flags_with_absl()


class SchedulingGroupsTest(jtu.JaxTestCase):

  def test_basic(self):
    a = 1.
    b = 2.
    x = 3.
    y = 4.

    @scheduling_group(name="grp0:sub_grp0")
    def fn0(a, b):
      c = jnp.add(a, b)
      return c

    @scheduling_group(name="grp0:sub_grp1")
    def fn1(x, y):
      z = jnp.multiply(x, y)
      return z

    @scheduling_group(name="grp0")
    def fn(a, b, x, y):
      c = fn0(a, b)
      z = fn1(x, y)
      return c, z

    lowered = jax.jit(fn).lower(a, b, x, y)
    self.assertIn('scheduling_group = "grp0"', lowered.as_text())

  def test_transforms(self):
    @scheduling_group(name='yash')
    def f(x):
      return 2 * x

    ans = jax.vmap(f)(jnp.arange(3.))
    self.assertAllClose(ans, 2. * jnp.arange(3.))

    ans = jax.grad(f)(3.)
    self.assertAllClose(ans, 2., check_dtypes=False)

  # TODO(yashkatariya): Enable this on TPU once XLA:TPU knows about inlineable
  @jtu.run_on_devices('cpu')
  def test_xla_metadata_call_inlineable(self):
    inp = jnp.arange(8.)

    @xla_metadata_call(inlineable="false")
    def g(x):
      return x * 2

    @jax.jit
    def f(x):
      y = g(x)
      return jnp.sin(y).sum()

    f(inp)  # doesn't crash

    lowered = jax.jit(jax.grad(f)).lower(inp)
    self.assertIn('inlineable = "false"', lowered.as_text())
    compiled = lowered.compile()
    self.assertIn('inlineable="false"', compiled.as_text())
    compiled(inp)  # doesn't crash


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
