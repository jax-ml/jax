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

from absl.testing import parameterized
import jax
from jax import lax
from jax import numpy as jnp
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
import numpy as np


def randint_sample(shape):
  return jax.random.randint(jax.random.PRNGKey(42), shape, -100, 100)


class AllReduceTest(jt_multiprocess.MultiProcessTest):

  def test_psum_simple(self):
    f = jax.pmap(lambda x: lax.psum(x, "i"), "i", devices=jax.devices())
    np.testing.assert_array_equal(
        np.array([jax.device_count()] * jax.local_device_count()),
        f(jnp.array([1] * jax.local_device_count())),
    )

  @parameterized.parameters(
      (np.int32,), (jnp.float32,), (jnp.float16,), (jnp.bfloat16,)
  )
  def test_psum(self, dtype):
    f = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")
    xs = randint_sample(
        [jax.process_count(), jax.local_device_count(), 100]
    ).astype(dtype)
    out = f(xs[jax.process_index()])
    expected = sum(sum(xs))
    for actual in out:
      jtu.check_close(actual, expected)

  def test_psum_subset_devices(self):
    f = jax.pmap(
        lambda x: lax.psum(x, "i"), axis_name="i", devices=jax.local_devices()
    )
    xs = randint_sample([jax.local_device_count(), 100])
    out = f(xs)
    expected = sum(xs)
    for actual in out:
      np.testing.assert_array_equal(actual, expected)

  def test_psum_del(self):  # b/171945402
    f = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")
    g = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")
    xs = randint_sample([jax.process_count(), jax.local_device_count(), 100])
    expected = sum(sum(xs))

    out = f(xs[jax.process_index()])
    for actual in out:
      np.testing.assert_array_equal(actual, expected)

    del f

    out = g(xs[jax.process_index()])
    for actual in out:
      np.testing.assert_array_equal(actual, expected)

  def test_psum_multiple_operands(self):
    f = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")
    xs = randint_sample([jax.process_count(), jax.local_device_count(), 100])
    ys = randint_sample([jax.process_count(), jax.local_device_count(), 200])
    out_xs, out_ys = f((xs[jax.process_index()], ys[jax.process_index()]))
    expected_xs = sum(sum(xs))
    expected_ys = sum(sum(ys))
    for actual in out_xs:
      np.testing.assert_array_equal(actual, expected_xs)
    for actual in out_ys:
      np.testing.assert_array_equal(actual, expected_ys)

  def test_psum_axis_index_groups(self):
    devices = list(range(jax.device_count()))
    axis_index_groups = [devices[0::2], devices[1::2]]
    print(axis_index_groups, jax.devices())
    f = jax.pmap(
        lambda x: lax.psum(x, "i", axis_index_groups=axis_index_groups),
        axis_name="i",
    )
    xs = randint_sample([jax.process_count(), jax.local_device_count(), 100])
    out = f(xs[jax.process_index()])

    xs = xs.reshape([jax.device_count(), 100])
    group0_expected = sum(xs[0::2, :])
    group1_expected = sum(xs[1::2, :])
    for i, actual in enumerate(out):
      device_id = i + jax.process_index() * jax.local_device_count()
      expected = group0_expected if device_id % 2 == 0 else group1_expected
      np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  jt_multiprocess.main()
