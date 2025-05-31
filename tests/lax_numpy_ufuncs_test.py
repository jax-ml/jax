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

import itertools
from functools import partial

from absl.testing import absltest

import numpy as np
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


def scalar_add(x, y):
  assert np.shape(x) == np.shape(y) == ()
  return x + y


def scalar_div(x, y):
  assert np.shape(x) == np.shape(y) == ()
  return x / y


def scalar_mul(x, y):
  assert np.shape(x) == np.shape(y) == ()
  return x * y


def scalar_sub(x, y):
  assert np.shape(x) == np.shape(y) == ()
  return x - y


SCALAR_FUNCS = [
  {'func': scalar_add, 'nin': 2, 'nout': 1, 'identity': 0},
  {'func': scalar_div, 'nin': 2, 'nout': 1, 'identity': None},
  {'func': scalar_mul, 'nin': 2, 'nout': 1, 'identity': 1},
  {'func': scalar_sub, 'nin': 2, 'nout': 1, 'identity': None},
]

def _jnp_ufunc_props(name):
  jnp_func = getattr(jnp, name)
  assert isinstance(jnp_func, jnp.ufunc)
  np_func = getattr(np, name)
  dtypes = [np.dtype(c) for c in "FfIi?" if f"{c}{c}->{c}" in np_func.types or f"{c}->{c}" in np_func.types]
  return [dict(name=name, dtype=dtype) for dtype in dtypes]


JAX_NUMPY_UFUNCS = [
  name for name in dir(jnp) if isinstance(getattr(jnp, name), jnp.ufunc)
]

BINARY_UFUNCS = [
  name for name in JAX_NUMPY_UFUNCS if getattr(jnp, name).nin == 2
]

UNARY_UFUNCS = [
  name for name in JAX_NUMPY_UFUNCS if getattr(jnp, name).nin == 1
]

JAX_NUMPY_UFUNCS_WITH_DTYPES = list(itertools.chain.from_iterable(
  _jnp_ufunc_props(name) for name in JAX_NUMPY_UFUNCS
))

BINARY_UFUNCS_WITH_DTYPES = list(itertools.chain.from_iterable(
  _jnp_ufunc_props(name) for name in BINARY_UFUNCS
))

UNARY_UFUNCS_WITH_DTYPES = list(itertools.chain.from_iterable(
  _jnp_ufunc_props(name) for name in UNARY_UFUNCS
))


broadcast_compatible_shapes = [(), (1,), (3,), (1, 3), (4, 1), (4, 3)]
nonscalar_shapes = [(3,), (4,), (4, 3)]

def cast_outputs(fun):
  def wrapped(*args, **kwargs):
    dtype = np.asarray(args[0]).dtype
    return jax.tree.map(lambda x: np.asarray(x, dtype=dtype), fun(*args, **kwargs))
  return wrapped


class LaxNumpyUfuncTests(jtu.JaxTestCase):

  @jtu.sample_product(SCALAR_FUNCS)
  def test_frompyfunc_properties(self, func, nin, nout, identity):
    jnp_fun = jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity)
    self.assertEqual(jnp_fun.identity, identity)
    self.assertEqual(jnp_fun.nin, nin)
    self.assertEqual(jnp_fun.nout, nout)
    self.assertEqual(jnp_fun.nargs, nin)

  @jtu.sample_product(name=JAX_NUMPY_UFUNCS)
  def test_ufunc_properties(self, name):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)
    self.assertEqual(jnp_fun.identity, np_fun.identity)
    self.assertEqual(jnp_fun.nin, np_fun.nin)
    self.assertEqual(jnp_fun.nout, np_fun.nout)
    self.assertEqual(jnp_fun.nargs, np_fun.nargs - 1)  # -1 because NumPy accepts `out`

  @jtu.sample_product(SCALAR_FUNCS)
  def test_frompyfunc_properties_readonly(self, func, nin, nout, identity):
    jnp_fun = jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity)
    for attr in ['nargs', 'nin', 'nout', 'identity', '_func']:
      getattr(jnp_fun, attr)  # no error on attribute access.
      with self.assertRaises(AttributeError):
        setattr(jnp_fun, attr, None)  # error when trying to mutate.

  @jtu.sample_product(name=JAX_NUMPY_UFUNCS)
  def test_ufunc_properties_readonly(self, name):
    jnp_fun = getattr(jnp, name)
    for attr in ['nargs', 'nin', 'nout', 'identity', '_func']:
      getattr(jnp_fun, attr)  # no error on attribute access.
      with self.assertRaises(AttributeError):
        setattr(jnp_fun, attr, None)  # error when trying to mutate.

  @jtu.sample_product(SCALAR_FUNCS)
  def test_frompyfunc_hash(self, func, nin, nout, identity):
    jnp_fun = jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity)
    jnp_fun_2 = jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity)
    self.assertEqual(jnp_fun, jnp_fun_2)
    self.assertEqual(hash(jnp_fun), hash(jnp_fun_2))

    other_fun = jnp.frompyfunc(jnp.add, nin=2, nout=1, identity=0)
    self.assertNotEqual(jnp_fun, other_fun)
    # Note: don't test hash for non-equality because it may collide.

  @jtu.sample_product(
      SCALAR_FUNCS,
      lhs_shape=broadcast_compatible_shapes,
      rhs_shape=broadcast_compatible_shapes,
      dtype=jtu.dtypes.floating,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def test_frompyfunc_call(self, func, nin, nout, identity, lhs_shape, rhs_shape, dtype):
    jnp_fun = jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity)
    np_fun = cast_outputs(np.frompyfunc(func, nin=nin, nout=nout, identity=identity))

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      UNARY_UFUNCS_WITH_DTYPES,
      shape=broadcast_compatible_shapes,
  )
  def test_unary_ufunc_call(self, name, dtype, shape):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      lhs_shape=broadcast_compatible_shapes,
      rhs_shape=broadcast_compatible_shapes,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def test_binary_ufunc_call(self, name, dtype, lhs_shape, rhs_shape):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      SCALAR_FUNCS,
      lhs_shape=broadcast_compatible_shapes,
      rhs_shape=broadcast_compatible_shapes,
      dtype=jtu.dtypes.floating,
  )
  def test_frompyfunc_outer(self, func, nin, nout, identity, lhs_shape, rhs_shape, dtype):
    if (nin, nout) != (2, 1):
      self.skipTest(f"outer requires (nin, nout)=(2, 1); got {(nin, nout)=}")
    jnp_fun = jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity).outer
    np_fun = cast_outputs(np.frompyfunc(func, nin=nin, nout=nout, identity=identity).outer)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      lhs_shape=broadcast_compatible_shapes,
      rhs_shape=broadcast_compatible_shapes,
  )
  def test_binary_ufunc_outer(self, name, lhs_shape, rhs_shape, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun.outer, np_fun.outer, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun.outer, args_maker)

  @jtu.sample_product(
      SCALAR_FUNCS,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in [None, *range(-len(shape), len(shape))]],
      dtype=jtu.dtypes.floating,
  )
  def test_frompyfunc_reduce(self, func, nin, nout, identity, shape, axis, dtype):
    if (nin, nout) != (2, 1):
      self.skipTest(f"reduce requires (nin, nout)=(2, 1); got {(nin, nout)=}")
    jnp_fun = partial(jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity).reduce, axis=axis)
    np_fun = cast_outputs(partial(np.frompyfunc(func, nin=nin, nout=nout, identity=identity).reduce, axis=axis))

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in [None, *range(-len(shape), len(shape))]],
  )
  def test_binary_ufunc_reduce(self, name, shape, axis, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)

    if jnp_fun.identity is None and axis is None and len(shape) > 1:
      self.skipTest("Multiple-axis reduction over non-reorderable ufunc.")

    jnp_fun_reduce = partial(jnp_fun.reduce, axis=axis)
    np_fun_reduce = partial(np_fun.reduce, axis=axis)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun_reduce, np_fun_reduce, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun_reduce, args_maker)

  @jtu.sample_product(
      SCALAR_FUNCS,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in [None, *range(-len(shape), len(shape))]],
      dtype=jtu.dtypes.floating,
  )
  def test_frompyfunc_reduce_where(self, func, nin, nout, identity, shape, axis, dtype):
    if (nin, nout) != (2, 1):
      self.skipTest(f"reduce requires (nin, nout)=(2, 1); got {(nin, nout)=}")

    # Need initial if identity is None
    initial = 1 if identity is None else None

    def jnp_fun(arr, where):
      return jnp.frompyfunc(func, nin, nout, identity=identity).reduce(
          arr, where=where, axis=axis, initial=initial)

    @cast_outputs
    def np_fun(arr, where):
      # Workaround for https://github.com/numpy/numpy/issues/24530
      # TODO(jakevdp): remove this when possible.
      initial_workaround = identity if initial is None else initial
      return np.frompyfunc(func, nin=nin, nout=nout, identity=identity).reduce(
          arr, where=where, axis=axis, initial=initial_workaround)

    rng = jtu.rand_default(self.rng())
    rng_where = jtu.rand_bool(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng_where(shape, bool)]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in [None, *range(-len(shape), len(shape))]],
  )
  def test_binary_ufunc_reduce_where(self, name, shape, axis, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)

    if jnp_fun.identity is None:
      self.skipTest("reduce with where requires identity")

    jnp_fun_reduce = lambda a, where: jnp_fun.reduce(a, axis=axis, where=where)
    np_fun_reduce = lambda a, where: np_fun.reduce(a, axis=axis, where=where)

    rng = jtu.rand_default(self.rng())
    rng_where = jtu.rand_bool(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng_where(shape, bool)]

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun_reduce, np_fun_reduce, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun_reduce, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in [None, *range(-len(shape), len(shape))]],
  )
  def test_binary_ufunc_reduce_initial(self, name, shape, axis, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)

    if jnp_fun.identity is None and axis is None and len(shape) > 1:
      self.skipTest("Multiple-axis reduction over non-reorderable ufunc.")

    jnp_fun_reduce = lambda a, initial: jnp_fun.reduce(a, axis=axis, initial=initial)
    np_fun_reduce = lambda a, initial: np_fun.reduce(a, axis=axis, initial=initial)

    rng = jtu.rand_default(self.rng())
    rng_initial = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng_initial((), dtype)]

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun_reduce, np_fun_reduce, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun_reduce, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      [{'shape': shape, 'axis': axis}
      for shape in nonscalar_shapes
      for axis in [None, *range(-len(shape), len(shape))]],
  )
  def test_binary_ufunc_reduce_where_initial(self, name, shape, axis, dtype):
      jnp_fun = getattr(jnp, name)
      np_fun = getattr(np, name)

      # Skip if the ufunc doesn't have an identity and we're doing a multi-axis reduction
      if jnp_fun.identity is None and axis is None and len(shape) > 1:
          self.skipTest("Multiple-axis reduction over non-reorderable ufunc.")

      jnp_fun_reduce = lambda a, where, initial: jnp_fun.reduce(
          a, axis=axis, where=where, initial=initial)
      np_fun_reduce = lambda a, where, initial: np_fun.reduce(
          a, axis=axis, where=where, initial=initial)

      rng = jtu.rand_default(self.rng())
      rng_where = jtu.rand_bool(self.rng())
      rng_initial = jtu.rand_default(self.rng())
      args_maker = lambda: [
          rng(shape, dtype),
          rng_where(shape, bool),
          rng_initial((), dtype)
      ]

      tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

      self._CheckAgainstNumpy(jnp_fun_reduce, np_fun_reduce, args_maker, tol=tol)
      self._CompileAndCheck(jnp_fun_reduce, args_maker)

  @jtu.sample_product(
      SCALAR_FUNCS,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in range(-len(shape), len(shape))],
      dtype=jtu.dtypes.floating,
  )
  def test_frompyfunc_accumulate(self, func, nin, nout, identity, shape, axis, dtype):
    if (nin, nout) != (2, 1):
      self.skipTest(f"accumulate requires (nin, nout)=(2, 1); got {(nin, nout)=}")
    jnp_fun = partial(jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity).accumulate, axis=axis)
    np_fun = cast_outputs(partial(np.frompyfunc(func, nin=nin, nout=nout, identity=identity).accumulate, axis=axis))

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in range(-len(shape), len(shape))]
  )
  def test_binary_ufunc_accumulate(self, name, shape, axis, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    jnp_fun_accumulate = partial(jnp_fun.accumulate, axis=axis)
    def np_fun_accumulate(x):
      # numpy accumulate has different dtype casting behavior.
      result = np_fun.accumulate(x, axis=axis)
      return result if x.dtype == bool else result.astype(x.dtype)

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun_accumulate, np_fun_accumulate, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun_accumulate, args_maker, tol=tol)

  @jtu.sample_product(
      SCALAR_FUNCS,
      shape=nonscalar_shapes,
      idx_shape=[(), (2,)],
      dtype=jtu.dtypes.floating,
  )
  def test_frompyfunc_at(self, func, nin, nout, identity, shape, idx_shape, dtype):
    if (nin, nout) != (2, 1):
      self.skipTest(f"accumulate requires (nin, nout)=(2, 1); got {(nin, nout)=}")
    jnp_fun = partial(jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity).at, inplace=False)
    def np_fun(x, idx, y):
      x_copy = x.copy()
      np.frompyfunc(func, nin=nin, nout=nout, identity=identity).at(x_copy, idx, y)
      return x_copy

    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), low=-shape[0], high=shape[0])
    args_maker = lambda: [rng(shape, dtype), idx_rng(idx_shape, 'int32'), rng(idx_shape[1:], dtype)]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      UNARY_UFUNCS_WITH_DTYPES,
      shape=nonscalar_shapes,
      idx_shape=[(), (2,)],
  )
  def test_unary_ufunc_at(self, name, shape, idx_shape, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)

    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), low=-shape[0], high=shape[0])
    args_maker = lambda: [rng(shape, dtype), idx_rng(idx_shape, 'int32')]

    jnp_fun_at = partial(jnp_fun.at, inplace=False)
    def np_fun_at(x, idx):
      x_copy = x.copy()
      np_fun.at(x_copy, idx)
      return x_copy

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun_at, np_fun_at, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun_at, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      shape=nonscalar_shapes,
      idx_shape=[(), (2,)],
  )
  def test_binary_ufunc_at(self, name, shape, idx_shape, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)

    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), low=-shape[0], high=shape[0])
    args_maker = lambda: [rng(shape, dtype), idx_rng(idx_shape, 'int32'), rng(idx_shape[1:], dtype)]

    jnp_fun_at = partial(jnp_fun.at, inplace=False)
    def np_fun_at(x, idx, y):
      x_copy = x.copy()
      np_fun.at(x_copy, idx, y)
      return x_copy

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun_at, np_fun_at, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun_at, args_maker)

  def test_frompyfunc_at_broadcasting(self):
    # Regression test for https://github.com/jax-ml/jax/issues/18004
    args_maker = lambda: [np.ones((5, 3)), np.array([0, 4, 2]),
                          np.arange(9.0).reshape(3, 3)]
    def np_fun(x, idx, y):
      x_copy = np.copy(x)
      np.add.at(x_copy, idx, y)
      return x_copy
    jnp_fun = partial(jnp.frompyfunc(jnp.add, nin=2, nout=1, identity=0).at, inplace=False)

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      SCALAR_FUNCS,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in [*range(-len(shape), len(shape))]],
      idx_shape=[(0,), (3,), (5,)],
      dtype=jtu.dtypes.floating,
  )
  def test_frompyfunc_reduceat(self, func, nin, nout, identity, shape, axis, idx_shape, dtype):
    if (nin, nout) != (2, 1):
      self.skipTest(f"accumulate requires (nin, nout)=(2, 1); got {(nin, nout)=}")
    jnp_fun = partial(jnp.frompyfunc(func, nin=nin, nout=nout, identity=identity).reduceat, axis=axis)
    np_fun = cast_outputs(partial(np.frompyfunc(func, nin=nin, nout=nout, identity=identity).reduceat, axis=axis))

    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), low=0, high=shape[axis])
    args_maker = lambda: [rng(shape, dtype), idx_rng(idx_shape, 'int32')]

    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      BINARY_UFUNCS_WITH_DTYPES,
      [{'shape': shape, 'axis': axis}
       for shape in nonscalar_shapes
       for axis in [*range(-len(shape), len(shape))]],
      idx_shape=[(0,), (3,), (5,)],
  )
  def test_binary_ufunc_reduceat(self, name, shape, axis, idx_shape, dtype):
    jnp_fun = getattr(jnp, name)
    np_fun = getattr(np, name)
    if (jnp_fun.nin, jnp_fun.nout) != (2, 1):
      self.skipTest(f"accumulate requires (nin, nout)=(2, 1); got {(jnp_fun.nin, jnp_fun.nout)=}")
    if name in ['add', 'multiply'] and dtype == bool:
      # TODO(jakevdp): figure out how to fix test cases.
      self.skipTest(f"known failure for {name}.reduceat with {dtype=}")

    rng = jtu.rand_default(self.rng())
    idx_rng = jtu.rand_int(self.rng(), low=0, high=shape[axis])
    args_maker = lambda: [rng(shape, dtype), idx_rng(idx_shape, 'int32')]

    def np_fun_reduceat(x, i):
      # Numpy has different casting behavior.
      return np_fun.reduceat(x, i).astype(x.dtype)

    tol = {np.float32: 1E-4} if jtu.test_device_matches(['tpu']) else None

    self._CheckAgainstNumpy(jnp_fun.reduceat, np_fun_reduceat, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun.reduceat, args_maker)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
