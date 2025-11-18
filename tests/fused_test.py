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

from jax.experimental.fused import fused

jax.config.parse_flags_with_absl()

@fused(out_spaces=(jax.memory.Space.Host, jax.memory.Space.Device))
def f(x, y):
  z = x + y
  w = x * y
  return z, w

class FusedTest(jtu.JaxTestCase):

  def test_basic(self):
    x = jnp.arange(3.)
    x_host = jax.device_put(x, jax.memory.Space.Host)
    y_device = jnp.arange(3.)
    low = jax.jit(f).trace(x_host, y_device).lower(lowering_platforms=('cuda',))
    txt = low._lowering.hlo().as_hlo_module().to_string()
    self.assertIn('custom_call', txt)
    self.assertIn('inlineable', txt)
    self.assertIn('MUST_FUSE', txt)
    self.assertIn('out_spaces', txt)

  def test_vmap_basic(self):
    x = jnp.arange(3.)
    x_host = jax.device_put(x, jax.memory.Space.Host)
    y_device = jnp.arange(3.)
    f_ = jax.jit(jax.vmap(f))
    f_.trace(x_host, y_device).lower(lowering_platforms=('cuda',)) # don't crash

  def test_jvp_basic(self):
    x = jnp.arange(3.)
    x_host = jax.device_put(x, jax.memory.Space.Host)
    y_device = jnp.arange(3.)
    f_ = jax.jit(lambda x, y: jax.jvp(f, (x, y), (x, y)))
    f_.trace(x_host, y_device).lower(lowering_platforms=('cuda',)) # don't crash

  def test_grad_basic(self):
    x = jnp.arange(3.)
    x_host = jax.device_put(x, jax.memory.Space.Host)
    y_device = jnp.arange(3.)
    f_ = jax.jit(jax.grad(lambda x, y: f(x, y)[1].sum()))
    f_.trace(x_host, y_device).lower(lowering_platforms=('cuda',)) # don't crash


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
