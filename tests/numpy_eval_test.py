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


import numpy as np

from jax import (config, jit, test_util as jtu, numpy as jnp, lax, device_put,
                 random, ad_util, tree_flatten)
from jax.interpreters.numpy_eval import numpy_eval
from jax.interpreters.xla import DeviceArray
from tests.lax_test import all_dtypes

config.parse_flags_with_absl()


class NumpyEvalTest(jtu.JaxTestCase):
  def check(self, lax_fun, *args, allow_device_array_output=False):
    if not config.omnistaging_enabled: return
    expected_flat, expected_tree = tree_flatten(lax_fun(*args))
    with numpy_eval():
      actual_flat, actual_tree = tree_flatten(lax_fun(*args))
    assert expected_tree == actual_tree
    for expected, actual in zip(expected_flat, actual_flat):
      is_np = (type(actual) in {np.ndarray, np.int64}.union(all_dtypes).union(
                 {DeviceArray} if allow_device_array_output else {}) or
               type(actual).__name__ == 'int64')  # needed for NumPy 1.16.4
      assert is_np, f"{type(actual).__name__} not a valid NumPy type."
    self.assertAllClose(expected, actual)

  def test_jit(self):
    self.check(jit(lax.square), np.array(1))

  def test_jit_of_numpy_eval(self):
    if not config.omnistaging_enabled: return

    @jit
    @numpy_eval()
    def f(v):
      return jnp.full((), v)

    a = f(1)
    self.assertAllClose(np.full((), 1), a)

  def test_lax_ops_on_device_arrays(self):
    self.check(lambda: jnp.array(1))
    self.check(lambda: jnp.full((), 1))
    self.check(lambda: jnp.zeros(()))
    self.check(lambda: jnp.ones(()))
    self.check(lax.transpose, jnp.array([[1, 2]]), (1, 0))
    self.check(lax.reshape, jnp.array([[1, 2]]), (2, 1, 1))
    self.check(lax.slice, jnp.array([1, 2]), (0,), (1,))
    self.check(lax.dynamic_slice, jnp.array([1, 2]), (0,), (1,))
    self.check(lax.dynamic_update_slice, jnp.array([1, 2]), jnp.array([0]), [0])
    self.check(lax.gather, jnp.arange(5,), jnp.array([[0], [2]]),
               lax.GatherDimensionNumbers((1,), (), (0,)), (3,))
    self.check(lax.sort, jnp.array([2, 3, 1]))
    self.check(lax.reduce, jnp.array([2, 3, 1]), jnp.array(0), lax.add, (0,))
    self.check(lax.broadcast_in_dim, jnp.array([1, 2]), (2, 2), (0,))
    self.check(lax.conv_general_dilated, jnp.array([[1.]]), jnp.array([[1.]]),
               (), 'VALID')

  def test_random_ops(self):
    self.check(random.PRNGKey, jnp.array(0))
    self.check(random.threefry_2x32, random.PRNGKey(0), lax.iota(np.uint32, 4))
    self.check(random.fold_in, random.PRNGKey(0), 0)
    self.check(random.split, random.PRNGKey(0))
    self.check(random.split, random.PRNGKey(0), 3)

  def test_ad_ops(self):
    self.check(ad_util.zeros_like_jaxval, jnp.array(0))
    self.check(lax.stop_gradient, {'a': [jnp.array(0)], 'b': jnp.array(1)},
               allow_device_array_output=True)

  def test_xla_ops(self):
    self.check(device_put, jnp.array(1), allow_device_array_output=True)
