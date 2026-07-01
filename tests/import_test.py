# Copyright 2026 The JAX Authors.
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
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import parallel
import jax.numpy as jnp

hlo_pb2 = None
try:
  from tensorflow.compiler.xla.service import hlo_pb2
except ImportError:
  pass


config.parse_flags_with_absl()


class ImportTest(jtu.JaxTestCase):

  def test_lower_async_all_gather(self):
    def f(x):
      return x + x

    x = jnp.ones(67)
    jax.jit(f).lower(x).compile()
    print(hlo_pb2)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
