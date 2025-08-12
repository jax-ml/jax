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
from jax._src import accumulator
from jax._src import test_util as jtu
import jax
import jax.numpy as jnp
import numpy as np

class AccumulatorTest(jtu.JaxTestCase):
  def test_accumulate(self):
    x = jnp.array([0., 0., 0.])
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    self.assertAllClose(acc[...], x)

    acc.at[...].accumulate(x)
    self.assertAllClose(acc[...], x + x)

    acc.at[...].accumulate(x)
    self.assertAllClose(acc[...], x + x + x)

  def test_vmap_batched_accumulator_batched_data(self):
    def f(acc, x):
      acc.at[...].accumulate(x)
      acc.at[...].accumulate(x)
    x = jnp.arange(3.0)
    acc = accumulator.accumulator(x, 0, jax.lax.add)
    jax.vmap(f)(acc, x)
    self.assertAllClose(acc[...], x + x + x)

  def test_vmap_batched_accumulator_unbatched_data(self):
    def f(acc, y):
      acc.at[...].accumulate(y)
      acc.at[...].accumulate(y)
    x = jnp.arange(3.0)
    y = jnp.float32(1.0)
    acc = accumulator.accumulator(x, 0, jax.lax.add)
    jax.vmap(f, in_axes=[0, None])(acc, y)
    self.assertAllClose(acc[...], x + y + y)

  def test_vmap_unbatched_accumulator_batched_data(self):
    x = jnp.arange(3.0)
    acc = accumulator.accumulator(x, 0, jax.lax.add)
    def f(y):
      acc.at[...].accumulate(y)
      acc.at[...].accumulate(y)
    y = jnp.arange(9.0).reshape(3, 3)
    jax.vmap(f)(y)
    self.assertAllClose(acc[...], x + jnp.sum(y, axis=0) + jnp.sum(y, axis=0))

  def test_vmap_unbatched_accumulator_unbatched_data(self):
    x = jnp.arange(3.0)
    y = jnp.arange(3.0)
    acc = accumulator.accumulator(x, 0, jax.lax.add)
    def f(_):
      acc.at[...].accumulate(y)
      acc.at[...].accumulate(y)
    jax.vmap(f)(x)
    self.assertAllClose(acc[...], x + 6 * y)

  def test_indexing_batched_accumulator_unbatched_val(self):
    x = jnp.arange(27.0).reshape(3, 3, 3)
    y = jnp.arange(9.0).reshape(3, 3)
    i = jnp.array([2, 1, 0])
    j = jnp.array([[2, 1, 0]] * 3)

    # Basic index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc):
      acc.at[1, 1].accumulate(y[1, 1])
    jax.vmap(f)(acc)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, 1, 1] += y[1, 1]
    self.assertAllClose(acc[...], want)

    # Unbatched index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc):
      acc.at[i, 0].accumulate(y[i, 0])
    jax.vmap(f)(acc)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, i, 0] += y[i, 0]
    self.assertAllClose(acc[...], want)

    # Batched index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc, j):
      acc.at[j, 0].accumulate(y[j, 0])
    jax.vmap(f)(acc, j)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, j[k], 0] += y[j[k], 0]
    self.assertAllClose(acc[...], want)

    # Mixed index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc, j):
      acc.at[i, j].accumulate(y[i, j])
    jax.vmap(f)(acc, j)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, i, j[k]] += y[i, j[k]]
    self.assertAllClose(acc[...], want)

  def test_indexing_unbatched_accumulator_unbatched_val(self):
    x = jnp.arange(27.0).reshape(3, 3, 3)
    y = jnp.arange(9.0).reshape(3, 3)
    i = jnp.array([2, 1, 0])
    j = jnp.array([[2, 1, 0]] * 3)

    # Basic index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x):
      acc.at[1, 1].accumulate(y[1, 1])
    jax.vmap(f)(x)

    want = np.array(y)
    for _ in range(x.shape[0]):
      want[1, 1] += y[1, 1]
    self.assertAllClose(acc[...], want)

    # Unbatched index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x):
      acc.at[i, 0].accumulate(y[i, 0])
    jax.vmap(f)(x)

    want = np.array(y)
    for _ in range(x.shape[0]):
      want[i, 0] += y[i, 0]
    self.assertAllClose(acc[...], want)

    # Batched index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x, j):
      acc.at[j, 0].accumulate(y[j, 0])
    jax.vmap(f)(x, j)

    want = np.array(y)
    for k in range(x.shape[0]):
      want[j[k], 0] += y[j[k], 0]
    self.assertAllClose(acc[...], want)

    # Mixed index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x, j):
      acc.at[i, j].accumulate(y[i, j])
    jax.vmap(f)(x, j)

    want = np.array(y)
    for k in range(x.shape[0]):
      want[i, j[k]] += y[i, j[k]]
    self.assertAllClose(acc[...], want)

  def test_indexing_batched_accumulator_batched_val(self):
    x = jnp.arange(27.0).reshape(3, 3, 3)
    i = jnp.array([2, 1, 0])
    j = jnp.array([[2, 1, 0]] * 3)

    # Basic index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc, x):
      acc.at[1, 1].accumulate(x[1, 1])
    jax.vmap(f)(acc, x)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, 1, 1] += x[k, 1, 1]
    self.assertAllClose(acc[...], want)

    # Unbatched index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc, x):
      acc.at[i, 0].accumulate(x[i, 0])
    jax.vmap(f)(acc, x)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, i, 0] += x[k, i, 0]
    self.assertAllClose(acc[...], want)

    # Batched index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc, x, j):
      acc.at[j, 0].accumulate(x[j, 0])
    jax.vmap(f)(acc, x, j)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, j[k], 0] += x[k, j[k], 0]
    self.assertAllClose(acc[...], want)

    # Mixed index.
    acc = accumulator.accumulator(x, 0.0, jax.lax.add)
    def f(acc, x, j):
      acc.at[i, j].accumulate(x[i, j])
    jax.vmap(f)(acc, x, j)

    want = np.array(x)
    for k in range(x.shape[0]):
      want[k, i, j[k]] += x[k, i, j[k]]
    self.assertAllClose(acc[...], want)

  def test_indexing_unbatched_accumulator_batched_val(self):
    x = jnp.arange(27.0).reshape(3, 3, 3)
    y = jnp.arange(9.0).reshape(3, 3)
    i = jnp.array([2, 1, 0])
    j = jnp.array([[2, 1, 0]] * 3)

    # Basic index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x):
      acc.at[1, 1].accumulate(x[1, 1])
    jax.vmap(f)(x)

    want = np.array(y)
    for k in range(x.shape[0]):
      want[1, 1] += x[k, 1, 1]
    self.assertAllClose(acc[...], want)

    # Unbatched index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x):
      acc.at[i, 0].accumulate(x[i, 0])
    jax.vmap(f)(x)

    want = np.array(y)
    for k in range(x.shape[0]):
      want[i, 0] += x[k, i, 0]
    self.assertAllClose(acc[...], want)

    # Batched index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x, j):
      acc.at[j, 0].accumulate(x[j, 0])
    jax.vmap(f)(x, j)

    want = np.array(y)
    for k in range(x.shape[0]):
      want[j[k], 0] += x[k, j[k], 0]
    self.assertAllClose(acc[...], want)

    # Mixed index.
    acc = accumulator.accumulator(y, 0.0, jax.lax.add)
    def f(x, j):
      acc.at[i, j].accumulate(x[i, j])
    jax.vmap(f)(x, j)

    want = np.array(y)
    for k in range(x.shape[0]):
      want[i, j[k]] += x[k, i, j[k]]
    self.assertAllClose(acc[...], want)

  def _identity(self, f, dtype):
    match f:
      case jax.lax.add:
        return jax._src.lax.lax._get_sum_identity(dtype)
      case jax.lax.mul:
        return jax._src.lax.lax._get_prod_identity(dtype)
      case jax.lax.bitwise_or:
        return jax._src.lax.lax._get_bitwise_or_identity(dtype)
      case jax.lax.bitwise_and:
        return jax._src.lax.lax._get_bitwise_and_identity(dtype)
      case jax.lax.bitwise_xor:
        return jax._src.lax.lax._get_bitwise_or_identity(dtype)
      case jax.lax.max:
        return jax._src.lax.lax._get_max_identity(dtype)
      case jax.lax.min:
        return jax._src.lax.lax._get_min_identity(dtype)
      case _:
        raise ValueError(f'Unexpected function {f}')

  def test_monoids(self):
    x = jnp.arange(3.0)
    y = x + x
    for f in [jax.lax.add, jax.lax.mul, jax.lax.max, jax.lax.min]:
      acc = accumulator.accumulator(x, self._identity(f, x.dtype), f)
      acc.at[...].accumulate(y)
      self.assertAllClose(acc[...], f(x, y))

    x = jnp.arange(3)
    y = x + x
    for f in [jax.lax.bitwise_or, jax.lax.bitwise_and, jax.lax.bitwise_xor]:
      acc = accumulator.accumulator(x, self._identity(f, x.dtype), f)
      acc.at[...].accumulate(y)
      self.assertAllClose(acc[...], f(x, y))

  def test_monoids_in_vmap(self):
    x = jnp.arange(3.0)
    y = jnp.arange(9.0).reshape(3, 3)
    for f in [jax.lax.add, jax.lax.mul, jax.lax.max, jax.lax.min]:
      acc = accumulator.accumulator(x, self._identity(f, x.dtype), f)
      def g(y):
        acc.at[...].accumulate(y)
      jax.vmap(g)(y)
      self.assertAllClose(acc[...], f(f(f(x, y[0]), y[1]), y[2]))

    x = jnp.arange(3)
    y = jnp.arange(9).reshape(3, 3)
    for f in [jax.lax.bitwise_or, jax.lax.bitwise_and, jax.lax.bitwise_xor]:
      acc = accumulator.accumulator(x, self._identity(f, x.dtype), f)
      def g(y):
        acc.at[...].accumulate(y)
      jax.vmap(g)(y)
      self.assertAllClose(acc[...], f(f(f(x, y[0]), y[1]), y[2]))

  def test_custom_monoid(self):
    x = jnp.arange(1, 10).reshape(3, 3)
    y = jnp.arange(1, 4)
    def take_right(x, y):
      return jnp.where(jnp.logical_or(x == 0, y == 0), x + y, y)
    acc = accumulator.accumulator(y, 0, take_right)
    acc.at[...].accumulate(y)
    def f(x):
      acc.at[...].accumulate(x)
      acc.at[...].accumulate(100*x)
      acc.at[...].accumulate(10*x)
    jax.vmap(f)(x)
    self.assertAllClose(acc[...], 10 * x[2])


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
