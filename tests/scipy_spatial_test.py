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

import jax

import scipy.version
from jax._src import test_util as jtu
from jax.scipy.spatial.transform import Rotation as jsp_Rotation
from scipy.spatial.transform import Rotation as osp_Rotation
from jax.scipy.spatial.transform import Slerp as jsp_Slerp
from scipy.spatial.transform import Slerp as osp_Slerp

import jax.numpy as jnp
import numpy as onp
from jax.config import config

config.parse_flags_with_absl()

scipy_version = tuple(map(int, scipy.version.version.split('.')[:3]))

float_dtypes = jtu.dtypes.floating
real_dtypes = float_dtypes + jtu.dtypes.integer + jtu.dtypes.boolean

num_samples = 2

class LaxBackedScipySpatialTransformTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.spatial implementations"""

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(2, 3)],
    use_weights=[False, True],
    return_sensitivity=[False, True],
  )
  def testRotationAlignVectors(self, shape, dtype, use_weights, return_sensitivity):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype), onp.abs(rng(shape[-2], dtype)) if use_weights else None)
    def jnp_fn(a, b, weights):
      result = jsp_Rotation.align_vectors(a, b, weights, return_sensitivity)
      return (result[0].as_rotvec(), *result[1:])
    def np_fn(a, b, weights):
      result = osp_Rotation.align_vectors(a, b, weights, return_sensitivity)
      return (result[0].as_rotvec(), *result[1:])
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
    vector_shape=[(3,), (num_samples, 3)],
    inverse=[True, False],
  )
  @jax.default_matmul_precision("float32")
  def testRotationApply(self, shape, vector_shape, dtype, inverse):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(vector_shape, dtype),)
    jnp_fn = lambda q, v: jsp_Rotation.from_quat(q).apply(v, inverse=inverse)
    np_fn = lambda q, v: osp_Rotation.from_quat(q).apply(v, inverse=inverse).astype(dtype)  # HACK
    tol = 5e-2 if jtu.device_under_test() == 'tpu' else 1e-4
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  @jtu.sample_product(
    dtype=float_dtypes,
    group=['I', 'O', 'T'],
  )
  def testRotationCreateGroup(self, group, dtype):
    args_maker = lambda: (None,)
    jnp_fn = lambda x: jsp_Rotation.create_group(group, dtype=dtype).as_quat()
    np_fn = lambda x: osp_Rotation.create_group(group).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    group=['C1', 'D1', 'C2', 'D2', 'C3', 'D3'],
    axis=['Z', 'Y', 'X'],
  )
  def testRotationCreateGroupWithAxis(self, group, axis, dtype):
    args_maker = lambda: (None,)
    jnp_fn = lambda x: jsp_Rotation.create_group(group, axis, dtype).as_quat()
    np_fn = lambda x: osp_Rotation.create_group(group, axis).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, tol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
    seq=['xyz', 'zyx', 'XYZ', 'ZYX'],
    degrees=[True, False],
  )
  def testRotationAsEuler(self, shape, dtype, seq, degrees):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_euler(seq=seq, degrees=degrees)
    np_fn = lambda q: osp_Rotation.from_quat(q).as_euler(seq=seq, degrees=degrees).astype(dtype)  # HACK
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
  )
  def testRotationAsMatrix(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_matrix()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_matrix().astype(dtype)  # HACK
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
  )
  def testRotationAsMrp(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_mrp()
    np_fn = lambda q: osp_Rotation.from_quat(q).as_mrp().astype(dtype)  # HACK
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
    degrees=[True, False],
  )
  def testRotationAsRotvec(self, shape, dtype, degrees):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).as_rotvec(degrees=degrees)
    np_fn = lambda q: osp_Rotation.from_quat(q).as_rotvec(degrees=degrees).astype(dtype)  # HACK
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
  )
  def testRotationAsQuat(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(jnp.where(jnp.sum(q, axis=0) > 0, q, -q)).as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(onp.where(jnp.sum(q, axis=0) > 0, q, -q)).as_quat().astype(dtype)  # HACK
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(num_samples, 4)],
    other_shape=[(num_samples, 4)],
  )
  def testRotationConcatenate(self, shape, other_shape, dtype):
    if scipy_version < (1, 8, 0):
      self.skipTest("Scipy 1.8.0 needed for concatenate.")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(other_shape, dtype),)
    jnp_fn = lambda q, o: jsp_Rotation.concatenate([jsp_Rotation.from_quat(q), jsp_Rotation.from_quat(o)]).as_quat()
    np_fn = lambda q, o: osp_Rotation.concatenate([osp_Rotation.from_quat(q), osp_Rotation.from_quat(o)]).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(10, 4)],
    indexer=[slice(1, 5), slice(0), slice(-5, -3)],
  )
  def testRotationGetItem(self, shape, dtype, indexer):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(jnp.where(jnp.sum(q, axis=0) > 0, q, -q))[indexer].as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(onp.where(onp.sum(q, axis=0) > 0, q, -q))[indexer].as_quat().astype(dtype)  # HACK
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    size=[1, num_samples],
    seq=['xy', 'xyz', 'XYZ'],
    degrees=[True, False],
  )
  def testRotationFromEuler(self, size, dtype, seq, degrees):
    rng = jtu.rand_default(self.rng())
    shape = (size, len(seq))
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_Rotation.from_euler(seq, a, degrees).as_quat()
    np_fn = lambda a: osp_Rotation.from_euler(seq, a, degrees).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    size=[1, num_samples],
    seq=['x', 'y', 'z'],
    degrees=[True, False],
  )
  def testRotationFromSingleEuler(self, size, dtype, seq, degrees):
    assert len(seq) == 1
    rng = jtu.rand_default(self.rng())
    shape = (size,)
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_Rotation.from_euler(seq, a, degrees).as_quat()
    np_fn = lambda a: osp_Rotation.from_euler(seq, a, degrees).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3, 3), (num_samples, 3, 3)],
  )
  def testRotationFromMatrix(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda m: jsp_Rotation.from_matrix(m).as_quat()
    np_fn = lambda m: osp_Rotation.from_matrix(m).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3,), (num_samples, 3)],
  )
  def testRotationFromMrp(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda m: jsp_Rotation.from_mrp(m).as_quat()
    np_fn = lambda m: osp_Rotation.from_mrp(m).as_quat()
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(3,), (num_samples, 3)],
  )
  def testRotationFromRotvec(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda r: jsp_Rotation.from_rotvec(r).as_quat()
    np_fn = lambda r: osp_Rotation.from_rotvec(r).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    num=[None],
  )
  def testRotationIdentity(self, num, dtype):
    args_maker = lambda: (num,)
    jnp_fn = lambda n: jsp_Rotation.identity(n, dtype).as_quat()
    np_fn = lambda n: osp_Rotation.identity(n).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    num=[None, num_samples],
  )
  def testRotationRandom(self, num, dtype):
    args_maker = lambda: (num,)
    key = jax.random.PRNGKey(0)
    jnp_fn = lambda n: jsp_Rotation.random(key, num, dtype).as_quat()
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
  )
  def testRotationMagnitude(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).magnitude()
    np_fn = lambda q: jnp.array(osp_Rotation.from_quat(q).magnitude(), dtype=dtype)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(num_samples, 4)],
    rng_weights =[True, False],
  )
  def testRotationMean(self, shape, dtype, rng_weights):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), jnp.abs(rng(shape[0], dtype)) if rng_weights else None)
    jnp_fn = lambda q, w: jsp_Rotation.from_quat(q).mean(w).as_quat()
    np_fn = lambda q, w: osp_Rotation.from_quat(q).mean(w).as_quat().astype(dtype)  # HACK
    tol = 5e-3 if jtu.device_under_test() == 'tpu' else 1e-4
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
    other_shape=[(4,), (num_samples, 4)],
  )
  def testRotationMultiply(self, shape, other_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(other_shape, dtype))
    jnp_fn = lambda q, o: (jsp_Rotation.from_quat(q) * jsp_Rotation.from_quat(o)).as_quat()
    np_fn = lambda q, o: (osp_Rotation.from_quat(q) * osp_Rotation.from_quat(o)).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
  )
  def testRotationInv(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).inv().as_quat()
    np_fn = lambda q: osp_Rotation.from_quat(q).inv().as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(num_samples, 4)],
  )
  def testRotationLen(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: len(jsp_Rotation.from_quat(q))
    np_fn = lambda q: len(osp_Rotation.from_quat(q))
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,)],  #, (num_samples, 4)],
    use_left=[False],
    use_right=[False],
    return_indices=[False],
  )
  def testRotationReduce(self, use_left, use_right, return_indices, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), rng(shape, dtype), rng(shape, dtype))
    jnp_fn = lambda p, l, r: jsp_Rotation.from_quat(p).reduce(jsp_Rotation.from_quat(l) if use_left else None, jsp_Rotation.from_quat(r) if use_right else None, return_indices).as_quat()
    np_fn = lambda p, l, r: osp_Rotation.from_quat(p).reduce(osp_Rotation.from_quat(l) if use_left else None, osp_Rotation.from_quat(r) if use_right else None, return_indices).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(4,), (num_samples, 4)],
  )
  def testRotationSingle(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda q: jsp_Rotation.from_quat(q).single
    np_fn = lambda q: osp_Rotation.from_quat(q).single
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=float_dtypes,
    shape=[(num_samples, 4)],
    compute_times=[0., onp.zeros(1), onp.zeros(2)],
  )
  def testSlerp(self, shape, dtype, compute_times):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    times = jnp.arange(shape[0], dtype=dtype)
    jnp_fn = lambda q: jsp_Slerp.init(times, jsp_Rotation.from_quat(q))(compute_times).as_quat()
    np_fn = lambda q: osp_Slerp(times, osp_Rotation.from_quat(q))(compute_times).as_quat().astype(dtype)  # HACK
    self._CheckQuaternionAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  def _CheckQuaternionAgainstNumpy(self, numpy_reference_op, lax_op, args_maker,
                                   check_dtypes=True, tol=None, atol=None, rtol=None,
                                   canonicalize_dtypes=True):
    args = args_maker()
    lax_quat = lax_op(*args)
    numpy_quat = numpy_reference_op(*args)
    if numpy_quat.ndim == 1:
      value = jnp.abs(jnp.dot(lax_quat, numpy_quat))
    elif numpy_quat.ndim == 2:
      value = jnp.abs(jnp.einsum('ij,ij->i', lax_quat, numpy_quat))
    expected_value = jnp.ones_like(value)
    self.assertAllClose(value, expected_value, check_dtypes=check_dtypes,
                        atol=atol or tol, rtol=rtol or tol,
                        canonicalize_dtypes=canonicalize_dtypes)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
