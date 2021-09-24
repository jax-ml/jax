# Copyright 2021 Google LLC
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

from unittest import skipIf

from absl.testing import absltest
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax.util import safe_map, safe_zip

from jax.experimental import djax
from jax.experimental.djax import (
    bbarray, ones_like, sin, add, iota, nonzero, reduce_sum, broadcast)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

jax.config.parse_flags_with_absl()


class DJaxTests(jtu.JaxTestCase):

  def test_identity_typechecks(self):
    def f(x):
      return x
    x = jnp.array([0, 1])
    jaxpr, _, _ = djax.make_djaxpr(f, x)
    djax.typecheck_jaxpr(jaxpr)

  def test_sin_typechecks(self):
    def f(x):
      return sin(x)
    x = bbarray((5,), jnp.arange(3.))
    jaxpr, _, _ = djax.make_djaxpr(f, x)
    djax.typecheck_jaxpr(jaxpr)

  def test_sin_and_add_typechecks(self):
    def f(x):
      y = sin(x)
      z = sin(y)
      return add(y, z)
    x = bbarray((5,), jnp.arange(3.))
    jaxpr, _, _ = djax.make_djaxpr(f, x)
    djax.typecheck_jaxpr(jaxpr)

  def test_iota_typechecks(self):
    def f():
      return iota(3)
    jaxpr, _, _ = djax.make_djaxpr(f)
    djax.typecheck_jaxpr(jaxpr)

  def test_nonzero_typechecks(self):
    def f(x):
      return nonzero(x)
    x = jnp.array([1, 0, -2, 0, 3, 0])
    jaxpr, _, _ = djax.make_djaxpr(f, x)
    djax.typecheck_jaxpr(jaxpr)

  def test_sum_of_nonzero_typechecks(self):
    def f(x):
      return reduce_sum(nonzero(x), tuple(range(len(x.shape))))
    x = jnp.array([1, 0, -2, 0, 3, 0])
    jaxpr, _, _ = djax.make_djaxpr(f, x)
    djax.typecheck_jaxpr(jaxpr)


@skipIf(jax.config.x64_enabled, "only 32bit for now")
class DJaxXLATests(jtu.JaxTestCase):

  def test_reduce_sum_of_nonzero(self):
    @djax.djit
    def f(x):
      nonzero_idx = nonzero(x)
      return reduce_sum(nonzero_idx)

    x = jnp.array([0, 1, 0, 1, 0, 1])
    ans = f(x)
    expected = np.sum(np.nonzero(x)[0])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nonzero(self):
    @djax.djit
    def f(x):
      return nonzero(x)
    x = jnp.array([0, 1, 0, 1, 0, 1])
    ans = f(x)
    expected, = np.nonzero(x)
    self.assertAllClose(np.array(ans), expected, check_dtypes=False)

  def test_iota(self):
    @djax.djit
    def f(i):
      return iota(i)
    ans = f(djax.BoundedInt(3, 5))
    expected = np.arange(3)
    self.assertAllClose(np.array(ans), expected, check_dtypes=False)

  def test_broadcast(self):
    @djax.djit
    def f(x, n):
      y = nonzero(x)
      return broadcast(y, n)
    x = np.arange(3)
    n = djax.BoundedInt(4, 5)
    ans = f(x, n)
    expected = np.broadcast_to(np.nonzero(x)[0], (4, 2))
    self.assertAllClose(np.array(ans), expected, check_dtypes=False)


@skipIf(jax.config.x64_enabled, "only 32bit for now")
class DJaxADTests(jtu.JaxTestCase):

  def test_jvp(self):
    @djax.djit
    def f(x):
      y = sin(x)
      return reduce_sum(y, axes=(0,))
    x = bbarray((5,), jnp.arange(2.))
    z, z_dot = jax.jvp(f, (x,), (ones_like(x),))

    def g(x):
      return jnp.sin(x).sum()
    expected_z, expected_z_dot = jax.jvp(g, (np.arange(2.),), (np.ones(2),))

    self.assertAllClose(np.array(z), expected_z, check_dtypes=False)
    self.assertAllClose(np.array(z_dot), expected_z_dot, check_dtypes=False)

  def test_linearize(self):
    @djax.djit
    def f(x):
      y = sin(x)
      return reduce_sum(y, axes=(0,))
    x = bbarray((5,), jnp.arange(2.))
    with jax.enable_checks(False):  # TODO implement dxla_call abs eval rule
      z, f_lin = jax.linearize(f, x)
    z_dot = f_lin(ones_like(x))

    def g(x):
      return jnp.sin(x).sum()
    expected_z, expected_z_dot = jax.jvp(g, (np.arange(2.),), (np.ones(2),))

    self.assertAllClose(np.array(z), expected_z, check_dtypes=False)
    self.assertAllClose(np.array(z_dot), expected_z_dot, check_dtypes=False)


@skipIf(jax.config.x64_enabled, "only 32bit for now")
class DJaxBatchingTests(jtu.JaxTestCase):

  def test_nonzero(self):
    raise absltest.SkipTest("TODO")  # TODO broke this somehow
    @djax.djit
    def f(x):
      return nonzero(x)
    xs = jnp.array([[0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 0, 1]])
    jax.vmap(f)(xs)  # doesn't crash
    # TODO check value


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
