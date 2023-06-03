# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial, reduce
import numpy as np
from numpy.polynomial import chebyshev as ncs
import numpy.polynomial.polyutils as npu
from numpy.polynomial.polyutils import RankWarning

from absl.testing import absltest

from jax import numpy as jnp
from jax._src import test_util as jtu
import jax.numpy.polynomials as pu
from jax.numpy.polynomials import chebyshev as cs

from jax.lax import Precision
from jax.config import config
config.parse_flags_with_absl()
config.update('jax_default_matmul_precision', Precision.HIGHEST)

all_dtypes = jtu.dtypes.floating + jtu.dtypes.integer + jtu.dtypes.complex

class TestChebyshev(jtu.JaxTestCase):

  @jtu.sample_product(
    dtype_off = all_dtypes,
    dtype_scl = all_dtypes,
    mul = [0,1],
  )
  def testChebline(self, dtype_off, dtype_scl, mul):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      off = rng(1, dtype_off)
      scl = rng(1, dtype_scl)
      scl  = (scl*mul).astype(dtype_scl)
      return (off, scl)

    jnp_fn = partial(cs.chebline, trim=False)

    with jtu.strict_promotion_if_dtypes_match([dtype_off, dtype_scl]):
      self._CheckAgainstNumpy(ncs.chebline, cs.chebline, args_maker, check_dtypes=False)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype1 = all_dtypes,
    dtype2 = all_dtypes,
    length1 = [1, 5, 10],
    length2 = [1, 5, 10],
    trailing1 = [0, 1],
    trailing2 = [0, 1],
  )
  def testChebadd(self, dtype1, dtype2, length1, length2, trailing1, trailing2):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c1 = rng((length1,), dtype1)
      c2 = rng((length2,), dtype2)
      return (jnp.concatenate([c1, jnp.zeros(trailing1, c1.dtype)]),
              jnp.concatenate([c2, jnp.zeros(trailing2, c2.dtype)]))

    jnp_fn = partial(cs.chebadd, trim=False)

    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(ncs.chebadd, cs.chebadd, args_maker, check_dtypes=False)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype1 = all_dtypes,
    dtype2 = all_dtypes,
    length1 = [1, 5, 10],
    length2 = [1, 5, 10],
    trailing1 = [0, 1],
    trailing2 = [0, 1],
  )
  def testChebsub(self, dtype1, dtype2, length1, length2, trailing1, trailing2):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c1 = rng((length1,), dtype1)
      c2 = rng((length2,), dtype2)
      return (jnp.concatenate([c1, jnp.zeros(trailing1, c1.dtype)]),
              jnp.concatenate([c2, jnp.zeros(trailing2, c2.dtype)]))

    jnp_fn = partial(cs.chebsub, trim=False)

    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(ncs.chebsub, cs.chebsub, args_maker, check_dtypes=False)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
  )
  def testChebmulx(self, dtype, length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype)
      return jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]),

    jnp_fn = partial(cs.chebmulx, trim=False)

    self._CheckAgainstNumpy(ncs.chebmulx, cs.chebmulx, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype1 = all_dtypes,
    dtype2 = all_dtypes,
    length1 = [1, 5, 10],
    length2 = [1, 5, 10],
    trailing1 = [0, 1],
    trailing2 = [0, 1],
  )
  def testChebmul(self, dtype1, dtype2, length1, length2, trailing1, trailing2):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c1 = rng((length1,), dtype1)
      c2 = rng((length2,), dtype2)
      return (jnp.concatenate([c1, jnp.zeros(trailing1, c1.dtype)]),
              jnp.concatenate([c2, jnp.zeros(trailing2, c2.dtype)]))

    jnp_fn = partial(cs.chebmul, trim=False)

    tolerance = {np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype1, dtype2])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(ncs.chebmul, cs.chebmul, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype1 = all_dtypes,
    dtype2 = all_dtypes,
    length1 = [1, 5, 10],
    length2 = [1, 5, 10],
    trailing1 = [0, 1],
    trailing2 = [0, 1],
  )
  def testChebdiv(self, dtype1, dtype2, length1, length2, trailing1, trailing2):
    rng = jtu.rand_nonzero(self.rng())

    def args_maker():
      c1 = rng((length1,), dtype1)
      c2 = rng((length2,), dtype2)

      return (jnp.concatenate([c1, jnp.zeros(trailing1, c1.dtype)]),
              jnp.concatenate([c2, jnp.zeros(trailing2, c2.dtype)]))

    jnp_fn = partial(cs.chebdiv, trim=False)

    tolerance = {np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype1, dtype2])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
      self._CheckAgainstNumpy(ncs.chebdiv, cs.chebdiv, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
    pow = [0, 1, 2, 10],
  )
  def testChebpow(self, dtype, length, trailing, pow):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype)
      return jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]),

    np_fn = partial(ncs.chebpow, pow=pow)
    jnp_fn = partial(cs.chebpow, pow=pow)
    jnp_fn_jit = partial(cs.chebpow, pow=pow, trim=False)

    tolerance = {np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = jtu.tolerance(dtype, tolerance)
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fn_jit, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_c = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
    pts = [0, 1, 2, 5],
    tensor = [False, True],
  )
  def testChebval(self, dtype_x, dtype_c, length, trailing, pts, tensor):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype_c)
      if pts == 0:
        p, = rng(1, dtype_x)
      else:
        p = rng((pts,), dtype_x)
      return (p, jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]))

    np_fn = partial(ncs.chebval, tensor=tensor)
    jnp_fn = partial(cs.chebval, tensor=tensor)

    tolerance = {np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_c])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_c]):
      self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
    m = [0, 1, 2, 5],
  )
  def testChebder(self, dtype, length, trailing, m):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype)
      return jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]),

    np_fn = partial(ncs.chebder, m=m)
    jnp_fn = partial(cs.chebder, m=m)

    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
    m = [0, 1, 2, 5],
    k = [0, 1],
    lbnd = [-1, 0, 1],
  )
  def testChebint(self, dtype, length, trailing, m, k, lbnd):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype)
      k_vals = rng((m,), dtype)
      k_vals *= k
      return (jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]), k_vals)

    np_fn = lambda c, k_vals: ncs.chebint(c, m=m, k=k_vals, lbnd=lbnd)
    jnp_fn = lambda c, k_vals: cs.chebint(c, m=m, k=k_vals, lbnd=lbnd)
    jnp_fn_jit = lambda c, k_vals: cs.chebint(c, m=m, k=k_vals, lbnd=lbnd, trim=False)

    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn_jit, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
  )
  def testCheb2poly(self, dtype, length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype)
      return jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]),

    jnp_fn = partial(cs.cheb2poly, trim=False)

    self._CheckAgainstNumpy(ncs.cheb2poly, cs.cheb2poly, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
  )
  def testPoly2cheb(self, dtype, length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      p = rng((length,), dtype)
      return jnp.concatenate([p, jnp.zeros(trailing, p.dtype)]),

    jnp_fn = partial(cs.poly2cheb, trim=False)

    self._CheckAgainstNumpy(ncs.poly2cheb, cs.poly2cheb, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [0, 1, 5, 10],
  )
  def testChebfromroots(self, dtype, length):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      p = rng((length,), dtype)
      return p,

    self._CheckAgainstNumpy(ncs.chebfromroots, cs.chebfromroots, args_maker, check_dtypes=False)
    self._CompileAndCheck(cs.chebfromroots, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
    degree = [0, 1, 5, 10],
  )
  def testChebvander(self, dtype, length, trailing, degree):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      p = rng((length,), dtype)
      return jnp.concatenate([p, jnp.zeros(trailing, p.dtype)]),

    np_fn = partial(ncs.chebvander, deg=degree)
    jnp_fn = partial(cs.chebvander, deg=degree)

    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_y = all_dtypes,
    length = [1, 5, 10],
    degree = [0, 1, 5],
    rcond = [None, 0.00001, 0.0001],
    full = [True, False],
    usew = [True, False],
  )
  def testChebfit(self, dtype_x, dtype_y, length, degree, rcond, full, usew):
    rng = jtu.rand_nonzero(self.rng())

    def args_maker():
      x = rng((length,), dtype_x)
      y = rng((length,), dtype_y)
      if usew:
        w = rng((length,), dtype_x)
      else:
        w = None
      return (x, y, w)

    jnp_fn = lambda x, y, w: cs.chebfit(x, y, degree, rcond=rcond, full=full, w=w)
    jnp_fn_jit = lambda x, y, w: cs.chebfit(x, y, degree, rcond=rcond, full=full, w=w, numpy_resid=False)
    np_fn = lambda x, y, w: ncs.chebfit(x, y, degree, rcond=rcond, full=full, w=w)
    np_fn = jtu.ignore_warning(category=RankWarning)(np_fn)

    tolerance = {np.float32: 1e-3, np.complex64: 1e-3, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_y])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_y]):
      self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_fn_jit, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
  )
  @jtu.skip_on_devices("gpu", "tpu")
  def testChebroots(self, dtype, length):
    rng = jtu.rand_nonzero(self.rng())

    def args_maker():
      p = rng((length,), dtype)
      return p,

    tolerance = {np.int32: 1e-5, np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = jtu.tolerance(dtype, tolerance)
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    jnp_fn = partial(cs.chebroots, trim=False)
    self._CheckAgainstNumpy(ncs.chebroots, cs.chebroots, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    npts = [1, 2, 3, 7, 10, 17],
  )
  def testChebpts1(self, npts):
    def args_maker():
      return npts,

    tolerance = {np.float64: 1e-12, np.complex128: 1e-12}
    tol = reduce(jtu.join_tolerance, [tolerance, jtu.default_tolerance()])

    self._CheckAgainstNumpy(ncs.chebpts1, cs.chebpts1, args_maker, check_dtypes=False, tol=tol)

  @jtu.sample_product(
    npts = [2, 3, 7, 10, 17],
  )
  def testChebpts2(self, npts):
    def args_maker():
      return npts,

    tolerance = {np.float64: 1e-12, np.complex128: 1e-12}
    tol = reduce(jtu.join_tolerance, [tolerance, jtu.default_tolerance()])

    self._CheckAgainstNumpy(ncs.chebpts2, cs.chebpts2, args_maker, check_dtypes=False, tol=tol)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_c = all_dtypes,
    val_length = [1, 5, 10],
    c_length = [1, 5, 10],
    trailing =[0, 1],
  )
  def testChebval2d(self, dtype_x, dtype_c, val_length, c_length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((val_length,), dtype_x)
      y = rng((val_length,), dtype_x)
      c = rng((c_length,), dtype_c)
      return(x, y, jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]))

    tolerance = {np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_c])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_c]):
      self._CheckAgainstNumpy(ncs.chebval2d, cs.chebval2d, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(cs.chebval2d, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_c = all_dtypes,
    val_length = [1, 5, 10],
    c_length = [1, 5, 10],
    trailing = [0, 1],
  )
  def testChebval3d(self, dtype_x, dtype_c, val_length, c_length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((val_length,), dtype_x)
      y = rng((val_length,), dtype_x)
      z = rng((val_length,), dtype_x)
      c = rng((c_length,), dtype_c)
      return(x, y, z, jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]))

    tolerance = {np.float32: 1e-3, np.complex64: 1e-3, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_c])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_c]):
      self._CheckAgainstNumpy(ncs.chebval3d, cs.chebval3d, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(cs.chebval3d, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_c = all_dtypes,
    val_length = [1, 5, 10],
    c_length = [1, 5, 10],
    trailing = [0, 1],
  )
  def testChebgrid2d(self, dtype_x, dtype_c, val_length, c_length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((val_length,), dtype_x)
      y = rng((val_length,), dtype_x)
      c = rng((c_length,), dtype_c)
      return(x, y, jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]))

    tolerance = {np.int32: 1e-5, np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_c])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_c]):
      self._CheckAgainstNumpy(ncs.chebgrid2d, cs.chebgrid2d, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(cs.chebgrid2d, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_c = all_dtypes,
    val_length = [1, 5, 10],
    c_length = [1, 5, 10],
    trailing = [0, 1],
  )
  def testChebgrid3d(self, dtype_x, dtype_c, val_length, c_length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((val_length,), dtype_x)
      y = rng((val_length,), dtype_x)
      z = rng((val_length,), dtype_x)
      c = rng((c_length,), dtype_c)
      return(x, y, z, jnp.concatenate([c, jnp.zeros(trailing, c.dtype)]))

    tolerance = {np.int32: 1e-5, np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_c])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_c]):
      self._CheckAgainstNumpy(ncs.chebgrid3d, cs.chebgrid3d, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(cs.chebgrid3d, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_y = all_dtypes,
    length = [1, 5, 10],
    x_degree = [0, 1, 5],
    y_degree = [0, 1, 5],
  )
  def testChebvander2d(self, dtype_x, dtype_y, length, x_degree, y_degree):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((length,), dtype_x)
      y = rng((length,), dtype_y)
      return (x, y)

    np_fn = partial(ncs.chebvander2d, deg=[x_degree, y_degree])
    jnp_fn = partial(cs.chebvander2d, deg_x=x_degree, deg_y=y_degree)

    tolerance = {np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_y])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_y]):
      self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_y = all_dtypes,
    dtype_z = all_dtypes,
    length = [1, 5],
    x_degree = [0, 5],
    y_degree = [0, 5],
    z_degree = [0, 5],
  )
  def testChebvander3d(self, dtype_x, dtype_y, dtype_z, length, x_degree, y_degree, z_degree):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((length,), dtype_x)
      y = rng((length,), dtype_y)
      z = rng((length,), dtype_z)
      return (x, y, z)

    np_fn = partial(ncs.chebvander3d, deg=[x_degree, y_degree, z_degree])
    jnp_fn = partial(cs.chebvander3d, deg_x=x_degree, deg_y=y_degree, deg_z=z_degree)

    tolerance = {np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_y, dtype_z])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_y, dtype_z]):
      self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [2, 5, 10],
  )
  def testChebcompanion(self, dtype, length):
    rng = jtu.rand_nonzero(self.rng())

    def args_maker():
      p = rng((length,), dtype)

      return p,

    jnp_fn = partial(cs.chebcompanion, trim=False)

    self._CheckAgainstNumpy(ncs.chebcompanion, cs.chebcompanion, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    degree = [1, 2, 3, 5, 7, 10],
  )
  def testChebgauss(self, degree):
    def args_maker():
      return degree,

    tolerance = {np.float64: 1e-12, np.complex128: 1e-12}
    tol = reduce(jtu.join_tolerance, [tolerance, jtu.default_tolerance()])

    self._CheckAgainstNumpy(ncs.chebgauss, cs.chebgauss, args_maker, check_dtypes=False, tol=tol)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
  )
  def testChebweight(self, dtype, length):
    rng = jtu.rand_uniform(self.rng(), low=-0.999, high=0.999)

    def args_maker():
      x = rng((length,), dtype)
      return x,

    self._CheckAgainstNumpy(ncs.chebweight, cs.chebweight, args_maker, check_dtypes=False)
    self._CompileAndCheck(cs.chebweight, args_maker)

  @jtu.sample_product(
    func_id = [0, 1, 2, 3],
    degree = [0, 1, 5, 10],
  )
  def testChebinterpolate(self, func_id, degree):
    funcs = [(lambda x: x+1, lambda x: x+1), (jnp.sin, np.sin), (lambda x: x**2, lambda x: x**2), (jnp.tanh, np.tanh)]
    def args_maker():
      func = funcs[func_id]
      return (func, degree)

    jnp_fn = lambda func, deg: cs.chebinterpolate(func[0], deg)
    np_fn = lambda func, deg: ncs.chebinterpolate(func[1], deg)

    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False)

class TestPolyutils(jtu.JaxTestCase):

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    count = [1, 5, 10],
  )
  def testAsSeries(self, dtype, length, count):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      return [rng((length,), dtype) for i in range(count)],

    jnp_fn = partial(pu.as_series, trim=False)

    self._CheckAgainstNumpy(npu.as_series, pu.as_series, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    trailing = [0, 1],
  )
  def testTrimseq(self, dtype, length, trailing):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype)
      return jnp.concatenate([c, jnp.zeros((trailing,), dtype=c.dtype)]),

    self._CheckAgainstNumpy(npu.trimseq, pu.trimseq, args_maker, check_dtypes=False)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
    tol = [0, 0.1, 1],
  )
  def testTrimcoef(self, dtype, length, tol):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      c = rng((length,), dtype)
      return (c, tol)

    self._CheckAgainstNumpy(npu.trimcoef, pu.trimcoef, args_maker, check_dtypes=False)

  @jtu.sample_product(
    dtype = all_dtypes,
    length = [1, 5, 10],
  )
  def testGetdomain(self, dtype, length):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((length,), dtype)
      return x,

    self._CheckAgainstNumpy(npu.getdomain, pu.getdomain, args_maker, check_dtypes=False)
    self._CompileAndCheck(pu.getdomain, args_maker)

  @jtu.sample_product(
    dtype_old = all_dtypes,
    dtype_new = all_dtypes,
  )
  def testMapparms(self, dtype_old, dtype_new):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      old = rng((2,), dtype_old)
      new = rng((2,), dtype_new)

      if old[1] == old[0]:
        old[1] += 1
      if new[1] == new[0]:
        new[1] += 1

      old.sort()
      new.sort()
      return (old, new)

    np_fn = npu.mapparms
    np_fn = jtu.ignore_warning(category=RuntimeWarning)(np_fn)

    with jtu.strict_promotion_if_dtypes_match([dtype_old, dtype_new]):
      self._CheckAgainstNumpy(np_fn, pu.mapparms, args_maker, check_dtypes=False)
      self._CompileAndCheck(pu.mapparms, args_maker)

  @jtu.sample_product(
    dtype_x = all_dtypes,
    dtype_d = all_dtypes,
    length = [1, 5, 10],
  )
  def testMapdomain(self, dtype_x, dtype_d, length):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      x = rng((length,), dtype_x)
      old = rng((2,), dtype_d)
      new = rng((2,), dtype_d)
      old.sort()
      new.sort()
      return (x, old, new)

    np_fn = npu.mapdomain
    np_fn = jtu.ignore_warning(category=RuntimeWarning)(np_fn)

    tolerance = {np.float32: 1e-4, np.complex64: 1e-4, np.float64: 1e-12, np.complex128: 1e-12}
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in [dtype_x, dtype_d])
    tol = reduce(jtu.join_tolerance, [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match([dtype_x, dtype_d]):
      self._CheckAgainstNumpy(np_fn, pu.mapdomain, args_maker, check_dtypes=False,
                              tol=tol)
      self._CompileAndCheck(pu.mapdomain, args_maker)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
