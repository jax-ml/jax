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
# from jax.scipy.spatial.transform import Slerp as jsp_Slerp
# from scipy.spatial.transform import Slerp as osp_Slerp

from jax.config import config

config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.floating
real_dtypes = float_dtypes + jtu.dtypes.integer + jtu.dtypes.boolean


class LaxBackedScipySpatialTransformTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.spatial implementations"""

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (2, 4)],
    vector_shape=[(3,), (2, 3)],
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
    shape=[(4,), (2, 4)],
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
    shape=[(4,), (2, 4)],
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
    shape=[(4,), (2, 4)],
  )
  def testRotationAsMrp(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_mrp()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_mrp()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (2, 4)],
    degrees=[True, False],
  )
  def testRotationAsRotvec(self, shape, dtype, degrees):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_rotvec(degrees=degrees)
    np_fn = lambda q: osp_Rotation.from_quat(q).as_rotvec(degrees=degrees)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (2, 4)],
  )
  def testRotationAsQuat(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(2, 4)],
    other_shape=[(2, 4)],
  )
  def testRotationConcatenate(self, shape, other_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(other_shape, dtype),)
    jnp_fn = lambda q, o: jsp_Rotation.concatenate([jsp_Rotation.from_quat(q), jsp_Rotation.from_quat(o)]).as_quat()
    np_fn = lambda q, o: osp_Rotation.concatenate([osp_Rotation.from_quat(q), osp_Rotation.from_quat(o)]).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(10, 4)],
    indexer=[slice(1, 5), slice(0), slice(-5, -3)],
  )
  def testRotationGetItem(self, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q)[indexer].as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(q)[indexer].as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3,), (2, 3)],
    seq=['xyz'],
    degrees=[True, False],
  )
  def testRotationFromEuler(self, shape, dtype, seq, degrees):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_Rotation.from_euler(seq, a, degrees).as_rotvec()
    np_fn = lambda a: osp_Rotation.from_euler(seq, a, degrees).as_rotvec()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3, 3), (2, 3, 3)],
  )
  def testRotationFromMatrix(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda m: jsp_Rotation.from_matrix(m).as_quat()
    np_fn = lambda m: osp_Rotation.from_matrix(m).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3,), (2, 3)],
  )
  def testRotationFromMrp(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda m: jsp_Rotation.from_mrp(m).as_quat()
    np_fn = lambda m: osp_Rotation.from_mrp(m).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3,), (2, 3)],
  )
  def testRotationFromRotvec(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda r: jsp_Rotation.from_rotvec(r).as_quat()
    np_fn = lambda r: osp_Rotation.from_rotvec(r).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    num=[None],
  )
  def testRotationIdentity(self, num):
    args_maker = lambda: (num,)
    jnp_fn = lambda n: jsp_Rotation.identity(n).as_quat()
    np_fn = lambda n: osp_Rotation.identity(n).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (2, 4)],
  )
  def testRotationMagnitude(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).magnitude()
    np_fn = lambda q: osp_Rotation.from_quat(q).magnitude()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(2, 4)],
    is_weights=[True, False],
  )
  def testRotationMean(self, shape, dtype, is_weights):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape[0], dtype) if is_weights else None)
    jnp_fn = lambda q, w: jsp_Rotation.from_quat(q).mean(w).as_quat()
    np_fn = lambda q, w: osp_Rotation.from_quat(q).mean(w).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (2, 4)],
    other_shape=[(4,), (2, 4)],
  )
  def testRotationMultiply(self, shape, other_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(other_shape, dtype))
    jnp_fn = lambda q, o: (jsp_Rotation.from_quat(q) * jsp_Rotation.from_quat(o)).as_quat()
    np_fn = lambda q, o: (osp_Rotation.from_quat(q) * osp_Rotation.from_quat(o)).as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (2, 4)],
  )
  def testRotationInv(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).inv().as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(q).inv().as_quat()
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(2, 4)],
  )
  def testRotationLen(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: len(jsp_Rotation.from_quat(q))
    np_fn = lambda q: len(osp_Rotation.from_quat(q))
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (2, 4)],
  )
  def testRotationSingle(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).single
    np_fn = lambda q: osp_Rotation.from_quat(q).single
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  # @jtu.sample_product(
  #   dtype=float_dtypes,
  #   shape=[(2, 4)],
  #   times=[[0.]],
  # )
  # def testSlerp(self, shape, dtype, times):
  #   rng = jtu.rand_default(self.rng())
  #   args_maker = lambda: (list(range(shape[0])), rng(shape, dtype),)
  #   jnp_fn = lambda t, q: jsp_Slerp.init(t, jsp_Rotation.from_quat(q))(times).as_quat()
  #   np_fn = lambda t, q: osp_Slerp(t, osp_Rotation.from_quat(q))(times).as_quat()
  #   self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
  #                           tol=1e-4)
  #   self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
