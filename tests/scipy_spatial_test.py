# Copyright 2023 The JAX Authors.
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

from jax._src import test_util as jtu
from jax.scipy.spatial.transform import Rotation as jsp_Rotation
from scipy.spatial.transform import Rotation as osp_Rotation

from jax.config import config

config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.floating
real_dtypes = float_dtypes + jtu.dtypes.integer + jtu.dtypes.boolean


class LaxBackedScipySpatialTransformTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.spatial implementations"""

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3, 3)],
  )
  def testRotationFromMatrix(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_Rotation.from_matrix(a).as_rotvec()
    np_fn = lambda a: osp_Rotation.from_matrix(a).as_rotvec()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
