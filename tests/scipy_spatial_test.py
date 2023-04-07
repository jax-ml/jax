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
    shape=[(4,)],
    vector_shape=[(3,)],
    inverse=[True, False],
  )
  def testRotationApply(self, shape, vector_shape, dtype, inverse):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(vector_shape, dtype),)
    jnp_fn = lambda q, v: jsp_Rotation.from_quat(q).apply(v, inverse=inverse)
    np_fn = lambda q, v: osp_Rotation.from_quat(q).apply(v, inverse=inverse)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],
    seq=['xyz'],
    degrees=[True, False],
  )
  def testRotationAsEuler(self, shape, dtype, seq, degrees):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_euler(seq=seq, degrees=degrees)
    np_fn = lambda q: osp_Rotation.from_quat(q).as_euler(seq=seq, degrees=degrees)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],
  )
  def testRotationAsMatrix(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_matrix()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_matrix()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],
  )
  def testRotationAsRotvec(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_rotvec()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_rotvec()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],
  )
  def testRotationAsQuat(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  # @jtu.sample_product(
  #   dtype=float_dtypes,
  #   shape=[(3, 3)],
  # )
  # def testRotationFromMatrix(self, shape, dtype):
  #   rng = jtu.rand_default(self.rng())
  #   args_maker = lambda: (rng(shape, dtype),)
  #   jnp_fn = lambda q: jsp_Rotation.from_matrix(a).as_rotvec()
  #   np_fn = lambda q: osp_Rotation.from_matrix(a).as_rotvec()
  #   self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
  #                           tol=1e-4)
  #   self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],
  )
  def testRotationFromRotvec(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],
  )
  def testRotationMultiply(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype))
    jnp_fn = lambda q, o: (jsp_Rotation.from_quat(q) * jsp_Rotation.from_quat(o)).as_quat()
    np_fn = lambda q, o: (osp_Rotation.from_quat(q) * osp_Rotation.from_quat(o)).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],
  )
  def testRotationInv(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).inv().as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(q).inv().as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
