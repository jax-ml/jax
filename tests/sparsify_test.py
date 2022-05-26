# Copyright 2021 Google LLC
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
import operator

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from jax import config, jit, lax
import jax.numpy as jnp
import jax._src.test_util as jtu
from jax.experimental.sparse import BCOO, sparsify, todense, SparseTracer
from jax.experimental.sparse.transform import (
  arrays_to_spvalues, spvalues_to_arrays, sparsify_raw, SparsifyValue, SparsifyEnv)
from jax.experimental.sparse.util import CuSparseEfficiencyWarning

config.parse_flags_with_absl()


class SparsifyTest(jtu.JaxTestCase):
  @classmethod
  def sparsify(cls, f):
    return sparsify(f, use_tracer=False)

  def testNotImplementedMessages(self):
    x = BCOO.fromdense(jnp.arange(5.0))
    # Test a densifying primitive
    with self.assertRaisesRegex(NotImplementedError,
        r"^sparse rule for cos is not implemented because it would result in dense output\."):
      self.sparsify(lax.cos)(x)

    # Test a generic not implemented primitive.
    with self.assertRaisesRegex(NotImplementedError,
        r"^sparse rule for complex is not implemented\.$"):
      self.sparsify(lax.complex)(x, x)

  def testTracerIsInstanceCheck(self):
    @self.sparsify
    def f(x):
      self.assertNotIsInstance(x, SparseTracer)
    f(jnp.arange(5))

  def assertBcooIdentical(self, x, y):
    self.assertIsInstance(x, BCOO)
    self.assertIsInstance(y, BCOO)
    self.assertEqual(x.shape, y.shape)
    self.assertArraysEqual(x.data, y.data)
    self.assertArraysEqual(x.indices, y.indices)

  def testSparsifyValue(self):
    X = jnp.arange(5)
    X_BCOO = BCOO.fromdense(X)

    args = (X, X_BCOO, X_BCOO)

    # Independent index
    spenv = SparsifyEnv()
    spvalues = arrays_to_spvalues(spenv, args)
    self.assertEqual(len(spvalues), len(args))
    self.assertLen(spenv._buffers, 5)
    self.assertEqual(spvalues,
        (SparsifyValue(X.shape, 0, None, False),
         SparsifyValue(X.shape, 1, 2, True),
         SparsifyValue(X.shape, 3, 4, True)))

    args_out = spvalues_to_arrays(spenv, spvalues)
    self.assertEqual(len(args_out), len(args))
    self.assertArraysEqual(args[0], args_out[0])
    self.assertBcooIdentical(args[1], args_out[1])
    self.assertBcooIdentical(args[2], args_out[2])

    # Shared index
    spvalues = (SparsifyValue(X.shape, 0, None), SparsifyValue(X.shape, 1, 2), SparsifyValue(X.shape, 3, 2))
    spenv = SparsifyEnv([X, X_BCOO.data, X_BCOO.indices, X_BCOO.data])

    args_out = spvalues_to_arrays(spenv, spvalues)
    self.assertEqual(len(args_out), len(args))
    self.assertArraysEqual(args[0], args_out[0])
    self.assertBcooIdentical(args[1], args_out[1])
    self.assertBcooIdentical(args[2], args_out[2])

  def testDropvar(self):
    def inner(x):
      return x * 2, x * 3

    def f(x):
      _, y = jit(inner)(x)
      return y * 4

    x_dense = jnp.arange(5)
    x_sparse = BCOO.fromdense(x_dense)
    self.assertArraysEqual(self.sparsify(f)(x_sparse).todense(), f(x_dense))

  def testPytreeInput(self):
    f = self.sparsify(lambda x: x)
    args = (jnp.arange(4), BCOO.fromdense(jnp.arange(4)))
    out = f(args)
    self.assertLen(out, 2)
    self.assertArraysEqual(args[0], out[0])
    self.assertBcooIdentical(args[1], out[1])

  def testSparsify(self):
    M_dense = jnp.arange(24).reshape(4, 6)
    M_sparse = BCOO.fromdense(M_dense)
    v = jnp.arange(M_dense.shape[0])

    @self.sparsify
    def func(x, v):
      return -jnp.sin(jnp.pi * x).T @ (v + 1)

    with jtu.ignore_warning(
        category=CuSparseEfficiencyWarning,
        message="bcoo_dot_general GPU lowering requires matrices with sorted indices*"):
      result_sparse = func(M_sparse, v)
    result_dense = func(M_dense, v)
    self.assertAllClose(result_sparse, result_dense)

  def testSparsifyWithConsts(self):
    M_dense = jnp.arange(24).reshape(4, 6)
    M_sparse = BCOO.fromdense(M_dense)

    @self.sparsify
    def func(x):
      return jit(lambda x: jnp.sum(x, 1))(x)

    result_dense = func(M_dense)
    result_sparse = func(M_sparse)

    self.assertAllClose(result_sparse.todense(), result_dense)

  def testSparseMatmul(self):
    X = jnp.arange(16).reshape(4, 4)
    Xsp = BCOO.fromdense(X)
    Y = jnp.ones(4)
    Ysp = BCOO.fromdense(Y)

    func = self.sparsify(operator.matmul)

    # dot_general
    result_sparse = func(Xsp, Y)
    result_dense = operator.matmul(X, Y)
    self.assertAllClose(result_sparse, result_dense)

    # rdot_general
    result_sparse = func(Y, Xsp)
    result_dense = operator.matmul(Y, X)
    self.assertAllClose(result_sparse, result_dense)

    # spdot_general
    result_sparse = self.sparsify(operator.matmul)(Xsp, Ysp)
    result_dense = operator.matmul(X, Y)
    self.assertAllClose(result_sparse.todense(), result_dense)

  def testSparseAdd(self):
    x = BCOO.fromdense(jnp.arange(5))
    y = BCOO.fromdense(2 * jnp.arange(5))

    # Distinct indices
    out = self.sparsify(operator.add)(x, y)
    self.assertEqual(out.nse, 8)  # uses concatenation.
    self.assertArraysEqual(out.todense(), 3 * jnp.arange(5))

    # Shared indices – requires lower level call
    spenv = SparsifyEnv([x.indices, x.data, y.data])
    spvalues = [
      spenv.sparse(x.shape, data_ref=1, indices_ref=0),
      spenv.sparse(y.shape, data_ref=2, indices_ref=0)
    ]

    result = sparsify_raw(operator.add)(spenv, *spvalues)
    args_out, _ = result
    out, = spvalues_to_arrays(spenv, args_out)

    self.assertAllClose(out.todense(), x.todense() + y.todense())

  def testSparseMul(self):
    x = BCOO.fromdense(jnp.arange(5))
    y = BCOO.fromdense(2 * jnp.arange(5))

    # Scalar multiplication
    out = self.sparsify(operator.mul)(x, 2.5)
    self.assertArraysEqual(out.todense(), x.todense() * 2.5)

    # Shared indices – requires lower level call
    spenv = SparsifyEnv([x.indices, x.data, y.data])
    spvalues = [
      spenv.sparse(x.shape, data_ref=1, indices_ref=0),
      spenv.sparse(y.shape, data_ref=2, indices_ref=0)
    ]

    result = sparsify_raw(operator.mul)(spenv, *spvalues)
    args_out, _ = result
    out, = spvalues_to_arrays(spenv, args_out)

    self.assertAllClose(out.todense(), x.todense() * y.todense())

  def testSparseSubtract(self):
    x = BCOO.fromdense(3 * jnp.arange(5))
    y = BCOO.fromdense(jnp.arange(5))

    # Distinct indices
    out = self.sparsify(operator.sub)(x, y)
    self.assertEqual(out.nse, 8)  # uses concatenation.
    self.assertArraysEqual(out.todense(), 2 * jnp.arange(5))

    # Shared indices – requires lower level call
    spenv = SparsifyEnv([x.indices, x.data, y.data])
    spvalues = [
      spenv.sparse(x.shape, data_ref=1, indices_ref=0),
      spenv.sparse(y.shape, data_ref=2, indices_ref=0)
    ]

    result = sparsify_raw(operator.sub)(spenv, *spvalues)
    args_out, _ = result
    out, = spvalues_to_arrays(spenv, args_out)

    self.assertAllClose(out.todense(), x.todense() - y.todense())

  def testSparseSum(self):
    x = jnp.arange(20).reshape(4, 5)
    xsp = BCOO.fromdense(x)

    def f(x):
      return x.sum(), x.sum(0), x.sum(1), x.sum((0, 1))

    result_dense = f(x)
    result_sparse = self.sparsify(f)(xsp)

    assert len(result_dense) == len(result_sparse)

    for res_dense, res_sparse in zip(result_dense, result_sparse):
      if isinstance(res_sparse, BCOO):
        res_sparse = res_sparse.todense()
      self.assertArraysAllClose(res_dense, res_sparse)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dimensions={}_nbatch={}_ndense={}".format(
          jtu.format_shape_dtype_string(shape, np.float32), dimensions, n_batch, n_dense),
       "shape": shape, "dimensions": dimensions, "n_batch": n_batch, "n_dense": n_dense}
      for shape, dimensions in [
          [(1,), (0,)],
          [(1,), (-1,)],
          [(2, 1, 4), (1,)],
          [(2, 1, 3, 1), (1,)],
          [(2, 1, 3, 1), (1, 3)],
          [(2, 1, 3, 1), (3,)],
      ]
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) - n_batch + 1)))
  def testSparseSqueeze(self, shape, dimensions, n_batch, n_dense):
    rng = jtu.rand_default(self.rng())

    M_dense = rng(shape, np.float32)
    M_sparse = BCOO.fromdense(M_dense, n_batch=n_batch, n_dense=n_dense)
    func = self.sparsify(partial(lax.squeeze, dimensions=dimensions))

    result_dense = func(M_dense)
    result_sparse = func(M_sparse).todense()

    self.assertAllClose(result_sparse, result_dense)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_shapes={shapes}_func={func}_nbatch={n_batch}",
       "shapes": shapes, "func": func, "n_batch": n_batch}
      for shapes, func, n_batch in [
          ([(4,), (4,)], "concatenate", 0),
          ([(4,), (4,)], "stack", 0),
          ([(4,), (4,)], "hstack", 0),
          ([(4,), (4,)], "vstack", 0),
          ([(4,), (4,)], "concatenate", 1),
          ([(4,), (4,)], "stack", 1),
          ([(4,), (4,)], "hstack", 1),
          ([(4,), (4,)], "vstack", 1),
          ([(2, 4), (2, 4)], "stack", 0),
          ([(2, 4), (3, 4)], "vstack", 0),
          ([(2, 4), (2, 5)], "hstack", 0),
          ([(2, 4), (3, 4)], "vstack", 1),
          ([(2, 4), (2, 5)], "hstack", 1),
          ([(2, 4), (3, 4)], "vstack", 2),
          ([(2, 4), (2, 5)], "hstack", 2),
          ([(2, 4), (4,), (3, 4)], "vstack", 0),
          ([(1, 4), (4,), (1, 4)], "vstack", 0),
      ]))
  def testSparseConcatenate(self, shapes, func, n_batch):
    f = self.sparsify(getattr(jnp, func))
    rng = jtu.rand_some_zero(self.rng())
    arrs = [rng(shape, 'int32') for shape in shapes]
    sparrs = [BCOO.fromdense(arr, n_batch=n_batch) for arr in arrs]
    self.assertArraysEqual(f(arrs), f(sparrs).todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{shape}->{new_shape}_n_batch={n_batch}_n_dense={n_dense}",
       "shape": shape, "new_shape": new_shape, "n_batch": n_batch, "n_dense": n_dense}
      for shape, new_shape, n_batch, n_dense in [
        [(6,), (2, 3), 0, 0],
        [(1, 4), (2, 2), 0, 0],
        [(12, 2), (2, 3, 4), 0, 0],
        [(1, 3, 2), (2, 3), 0, 0],
        [(1, 6), (2, 3, 1), 0, 0],
        [(2, 3, 4), (3, 8), 0, 0],
        [(2, 3, 4), (1, 2, 12), 1, 0],
        [(2, 3, 4), (6, 2, 2), 2, 0],
      ]))
  def testSparseReshapeMethod(self, shape, new_shape, n_batch, n_dense):
    rng = jtu.rand_some_zero(self.rng())
    arr = rng(shape, 'int32')
    arr_sparse = BCOO.fromdense(arr, n_batch=n_batch, n_dense=n_dense)

    arr2 = arr.reshape(new_shape)
    arr2_sparse = arr_sparse.reshape(new_shape)

    self.assertArraysEqual(arr2, arr2_sparse.todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{shape}->{new_shape}_n_batch={n_batch}_n_dense={n_dense}_dimensions={dimensions}",
       "shape": shape, "new_shape": new_shape, "n_batch": n_batch, "n_dense": n_dense,
       "dimensions": dimensions}
      for shape, new_shape, n_batch, n_dense, dimensions in [
        [(2, 3, 4), (24,), 0, 0, None],
        [(2, 3, 4), (24,), 0, 0, (0, 1, 2)],
        [(2, 3, 4), (24,), 0, 0, (0, 2, 1)],
        [(2, 3, 4), (24,), 0, 0, (1, 0, 2)],
        [(2, 3, 4), (24,), 0, 0, (1, 2, 0)],
        [(2, 3, 4), (24,), 0, 0, (2, 0, 1)],
        [(2, 3, 4), (24,), 0, 0, (2, 1, 0)],
        [(4, 2, 3), (2, 2, 6), 1, 0, (0, 1, 2)],
        [(4, 2, 3), (2, 2, 6), 1, 0, (0, 2, 1)],
        [(2, 3, 4), (6, 4), 2, 0, (0, 1, 2)],
        [(2, 3, 4), (6, 4), 2, 0, (1, 0, 2)],
      ]))
  def testSparseReshapeWithDimensions(self, shape, new_shape, n_batch, n_dense, dimensions):
    rng = jtu.rand_some_zero(self.rng())
    arr = rng(shape, 'int32')
    arr_sparse = BCOO.fromdense(arr, n_batch=n_batch, n_dense=n_dense)

    f = self.sparsify(lambda x: lax.reshape(x, new_shape, dimensions=dimensions))

    arr2 = f(arr)
    arr2_sparse = f(arr_sparse)

    self.assertArraysEqual(arr2, arr2_sparse.todense())

  def testSparseWhileLoop(self):
    def cond_fun(params):
      i, A = params
      return i < 5

    def body_fun(params):
      i, A = params
      return i + 1, 2 * A

    def f(A):
      return lax.while_loop(cond_fun, body_fun, (0, A))

    A = jnp.arange(4)
    out_dense = f(A)

    Asp = BCOO.fromdense(A)
    out_sparse = self.sparsify(f)(Asp)

    self.assertEqual(len(out_dense), 2)
    self.assertEqual(len(out_sparse), 2)
    self.assertArraysEqual(out_dense[0], out_dense[0])
    self.assertArraysEqual(out_dense[1], out_sparse[1].todense())

  def testSparseWhileLoopDuplicateIndices(self):
    def cond_fun(params):
      i, A, B = params
      return i < 5

    def body_fun(params):
      i, A, B = params
      # TODO(jakevdp): track shared indices through while loop & use this
      #   version of the test, which requires shared indices in order for
      #   the nse of the result to remain the same.
      # return i + 1, A, A + B

      # This version is fine without shared indices, and tests that we're
      # flattening non-shared indices consistently.
      return i + 1, B, A

    def f(A):
      return lax.while_loop(cond_fun, body_fun, (0, A, A))

    A = jnp.arange(4).reshape((2, 2))
    out_dense = f(A)

    Asp = BCOO.fromdense(A)
    out_sparse = self.sparsify(f)(Asp)

    self.assertEqual(len(out_dense), 3)
    self.assertEqual(len(out_sparse), 3)
    self.assertArraysEqual(out_dense[0], out_dense[0])
    self.assertArraysEqual(out_dense[1], out_sparse[1].todense())
    self.assertArraysEqual(out_dense[2], out_sparse[2].todense())

  def testSparsifyDenseXlaCall(self):
    # Test handling of dense xla_call within jaxpr interpreter.
    out = self.sparsify(jit(lambda x: x + 1))(0.0)
    self.assertEqual(out, 1.0)

  def testSparsifySparseXlaCall(self):
    # Test sparse lowering of XLA call
    def func(M):
      return 2 * M

    M = jnp.arange(6).reshape(2, 3)
    Msp = BCOO.fromdense(M)

    out_dense = func(M)
    out_sparse = self.sparsify(jit(func))(Msp)
    self.assertArraysEqual(out_dense, out_sparse.todense())

  def testSparseForiLoop(self):
    def func(M, x):
      body_fun = lambda i, val: (M @ val) / M.shape[1]
      return lax.fori_loop(0, 2, body_fun, x)

    x = jnp.arange(5.0)
    M = jnp.arange(25).reshape(5, 5)
    M_bcoo = BCOO.fromdense(M)

    result_dense = func(M, x)
    result_sparse = self.sparsify(func)(M_bcoo, x)

    self.assertArraysAllClose(result_dense, result_sparse)

  def testSparseCondSimple(self):
    def func(x):
      return lax.cond(False, lambda x: x, lambda x: 2 * x, x)

    x = jnp.arange(5.0)
    result_dense = func(x)

    x_bcoo = BCOO.fromdense(x)
    result_sparse = self.sparsify(func)(x_bcoo)

    self.assertArraysAllClose(result_dense, result_sparse.todense())

  def testSparseCondMismatchError(self):
    @self.sparsify
    def func(x, y):
      return lax.cond(False, lambda x: x[0], lambda x: x[1], (x, y))

    x = jnp.arange(5.0)
    y = jnp.arange(5.0)

    x_bcoo = BCOO.fromdense(x)
    y_bcoo = BCOO.fromdense(y)

    func(x, y)  # No error
    func(x_bcoo, y_bcoo)  # No error

    with self.assertRaisesRegex(TypeError, "sparsified true_fun and false_fun output.*"):
      func(x_bcoo, y)

  def testToDense(self):
    M = jnp.arange(4)
    Msp = BCOO.fromdense(M)
    @self.sparsify
    def func(M):
      return todense(M) + 1
    self.assertArraysEqual(func(M), M + 1)
    self.assertArraysEqual(func(Msp), M + 1)
    self.assertArraysEqual(jit(func)(M), M + 1)
    self.assertArraysEqual(jit(func)(Msp), M + 1)

  def testWeakTypes(self):
    # Regression test for https://github.com/google/jax/issues/8267
    M = jnp.arange(12, dtype='int32').reshape(3, 4)
    Msp = BCOO.fromdense(M)
    self.assertArraysEqual(
      operator.mul(2, M),
      self.sparsify(operator.mul)(2, Msp).todense(),
      check_dtypes=True,
    )

class SparsifyTracerTest(SparsifyTest):
  @classmethod
  def sparsify(cls, f):
    return sparsify(f, use_tracer=True)

  def testTracerIsInstanceCheck(self):
    @self.sparsify
    def f(x):
      self.assertIsInstance(x, SparseTracer)
    f(jnp.arange(5))

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
