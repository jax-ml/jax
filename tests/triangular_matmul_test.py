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
from jax._src import test_util as jtu
import jax.numpy as jnp

class TriangularMatmulTest(jtu.JaxTestCase):

  def test_lower_triangular_matrix_attribute(self):
    # b/168891041
    key = jax.random.key(4352)
    a = jax.random.normal(key, (64, 64), jnp.float32)

    # from instruction: %select.9 = f32[64,64]{1,0}
    #   select(%compare.8, %Arg_0.1, %broadcast.3),
    #     frontend_attributes={matrix_type="hlo.lower_triangular_matrix"}, ...

    hlo = jax.jit(jnp.tril).lower(a).as_text()
    # Check that matrix is lower triangular.
    self.assertIn('hlo.lower_triangular_matrix', hlo)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
