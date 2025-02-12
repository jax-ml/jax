# Copyright 2024 The JAX Authors.
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

from functools import partial

import numpy as np

from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import linalg as lax_linalg
from jax._src.lib import gpu_solver

config.parse_flags_with_absl()

float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex


class MagmaLinalgTest(jtu.JaxTestCase):

  @jtu.sample_product(
    shape=[(0, 0), (4, 4), (5, 5), (50, 50), (2, 6, 6)],
    dtype=float_types + complex_types,
    compute_left_eigenvectors=[False, True],
    compute_right_eigenvectors=[False, True],
  )
  @jtu.run_on_devices("gpu")
  def testEig(self, shape, dtype, compute_left_eigenvectors,
              compute_right_eigenvectors):
    if not gpu_solver.has_magma():
      self.skipTest("MAGMA is not installed or can't be loaded.")
    # TODO(b/377907938), TODO(danfm): Debug issues MAGMA support for
    # complex128 in some configurations.
    if dtype == np.complex128:
      self.skipTest("MAGMA support for complex128 types is flaky.")
    rng = jtu.rand_default(self.rng())
    n = shape[-1]
    args_maker = lambda: [rng(shape, dtype)]

    # Norm, adjusted for dimension and type.
    def norm(x):
      norm = np.linalg.norm(x, axis=(-2, -1))
      return norm / ((n + 1) * jnp.finfo(dtype).eps)

    def check_right_eigenvectors(a, w, vr):
      self.assertTrue(
        np.all(norm(np.matmul(a, vr) - w[..., None, :] * vr) < 100))

    def check_left_eigenvectors(a, w, vl):
      rank = len(a.shape)
      aH = jnp.conj(a.transpose(list(range(rank - 2)) + [rank - 1, rank - 2]))
      wC = jnp.conj(w)
      check_right_eigenvectors(aH, wC, vl)

    a, = args_maker()
    results = lax_linalg.eig(
        a, compute_left_eigenvectors=compute_left_eigenvectors,
        compute_right_eigenvectors=compute_right_eigenvectors,
        use_magma=True)
    w = results[0]

    if compute_left_eigenvectors:
      check_left_eigenvectors(a, w, results[1])
    if compute_right_eigenvectors:
      check_right_eigenvectors(a, w, results[1 + compute_left_eigenvectors])

    self._CompileAndCheck(jnp.linalg.eig, args_maker, rtol=1e-3)

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (50, 50), (2, 6, 6)],
    dtype=float_types + complex_types,
    compute_left_eigenvectors=[False, True],
    compute_right_eigenvectors=[False, True],
  )
  @jtu.run_on_devices("gpu")
  def testEigHandlesNanInputs(self, shape, dtype, compute_left_eigenvectors,
                              compute_right_eigenvectors):
    """Verifies that `eig` fails gracefully if given non-finite inputs."""
    if not gpu_solver.has_magma():
      self.skipTest("MAGMA is not installed or can't be loaded.")
    # TODO(b/377907938), TODO(danfm): Debug issues MAGMA support for
    # complex128 in some configurations.
    if dtype == np.complex128:
      self.skipTest("MAGMA support for complex128 types is flaky.")
    a = jnp.full(shape, jnp.nan, dtype)
    results = lax_linalg.eig(
        a, compute_left_eigenvectors=compute_left_eigenvectors,
        compute_right_eigenvectors=compute_right_eigenvectors,
        use_magma=True)
    for result in results:
      self.assertTrue(np.all(np.isnan(result)))

  def testEigMagmaConfig(self):
    if not gpu_solver.has_magma():
      self.skipTest("MAGMA is not installed or can't be loaded.")
    rng = jtu.rand_default(self.rng())
    a = rng((5, 5), np.float32)
    with config.gpu_use_magma("on"):
      hlo = jax.jit(partial(lax_linalg.eig, use_magma=True)).lower(a).as_text()
      self.assertIn('magma = "on"', hlo)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
