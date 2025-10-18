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
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np


def randint_sample(shape):
  return jax.random.randint(jax.random.PRNGKey(42), shape, -100, 100)


class AllGatherTest(jt_multiprocess.MultiProcessTest):

  @parameterized.parameters(
      (np.int32,), (jnp.float32,), (jnp.float16,), (jnp.bfloat16,)
  )
  def test_all_gather(self, dtype):
    f = jax.pmap(lambda x: lax.all_gather(x, "i"), axis_name="i")
    xs = randint_sample(
        [jax.process_count(), jax.local_device_count(), 100]
    ).astype(dtype)
    out = f(xs[jax.process_index()])
    expected = np.reshape(xs, [jax.device_count(), 100])
    for actual in out:
      jtu.check_close(actual, expected)


if __name__ == "__main__":
  jt_multiprocess.main()
