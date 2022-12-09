# Copyright 2021 The JAX Authors.
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
"""Sparse test utilities."""
import functools

from typing import Any, Callable, Sequence, Union

import numpy as np

from jax._src import api
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src.typing import DTypeLike
from jax import tree_util
from jax.util import split_list
from jax.experimental import sparse
from jax.experimental.sparse import bcoo as sparse_bcoo
import jax.numpy as jnp


class SparseTestCase(jtu.JaxTestCase):
  def assertSparseArraysEquivalent(self, x, y, *, check_dtypes=True, atol=None,
                                   rtol=None, canonicalize_dtypes=True, err_msg=''):
    x_bufs, x_tree = tree_util.tree_flatten(x)
    y_bufs, y_tree = tree_util.tree_flatten(y)

    self.assertEqual(x_tree, y_tree)
    self.assertAllClose(x_bufs, y_bufs, check_dtypes=check_dtypes, atol=atol, rtol=rtol,
                        canonicalize_dtypes=canonicalize_dtypes, err_msg=err_msg)

  def _CheckAgainstDense(self, dense_op, sparse_op, args_maker,
                         check_dtypes=True, tol=None, atol=None, rtol=None,
                         canonicalize_dtypes=True):
    """Check an operation against a dense equivalent"""
    is_sparse = lambda x: isinstance(x, sparse.JAXSparse)

    sparse_args = args_maker()
    dense_args = tree_util.tree_map(sparse.todense, sparse_args, is_leaf=is_sparse)

    sparse_ans = sparse_op(*sparse_args)
    actual = tree_util.tree_map(sparse.todense, sparse_ans, is_leaf=is_sparse)
    expected = dense_op(*dense_args)

    self.assertAllClose(expected, actual, check_dtypes=check_dtypes,
                        atol=atol or tol, rtol=rtol or tol,
                        canonicalize_dtypes=canonicalize_dtypes)

  def _CompileAndCheckSparse(self, fun, args_maker, check_dtypes=True,
                             rtol=None, atol=None, check_cache_misses=True):
    args = args_maker()

    def wrapped_fun(*args):
      self.assertTrue(python_should_be_executing)
      return fun(*args)

    python_should_be_executing = True
    python_ans = fun(*args)

    cache_misses = dispatch.xla_primitive_callable.cache_info().misses
    python_ans = fun(*args)
    if check_cache_misses:
      self.assertEqual(
          cache_misses, dispatch.xla_primitive_callable.cache_info().misses,
          "Compilation detected during second call of {} in op-by-op "
          "mode.".format(fun))

    cfun = api.jit(wrapped_fun)
    python_should_be_executing = True
    monitored_ans = cfun(*args)

    python_should_be_executing = False
    compiled_ans = cfun(*args)

    self.assertSparseArraysEquivalent(python_ans, monitored_ans, check_dtypes=check_dtypes,
                                      atol=atol, rtol=rtol)
    self.assertSparseArraysEquivalent(python_ans, compiled_ans, check_dtypes=check_dtypes,
                                      atol=atol, rtol=rtol)

def _rand_sparse(shape: Sequence[int], dtype: DTypeLike, *,
                 rng: np.random.RandomState, rand_method: Callable[..., Any],
                 nse: Union[int, float], n_batch: int, n_dense: int,
                 sparse_format: str) -> Union[sparse.BCOO, sparse.BCSR]:
  if sparse_format not in ['bcoo', 'bcsr']:
    raise ValueError(f"Sparse format {sparse_format} not supported.")

  n_sparse = len(shape) - n_batch - n_dense

  if n_sparse < 0 or n_batch < 0 or n_dense < 0:
    raise ValueError(f"Invalid parameters: {shape=} {n_batch=} {n_sparse=}")

  if sparse_format == 'bcsr' and n_sparse != 2:
    raise ValueError("bcsr array must have 2 sparse dimensions; "
                     f"{n_sparse} is given.")

  batch_shape, sparse_shape, dense_shape = split_list(shape,
                                                      [n_batch, n_sparse])
  if 0 <= nse < 1:
    nse = int(np.ceil(nse * np.prod(sparse_shape)))
  data_rng = rand_method(rng)
  index_shape = (*batch_shape, nse, n_sparse)
  data_shape = (*batch_shape, nse, *dense_shape)
  bcoo_indices = jnp.array(
      rng.randint(0, sparse_shape, size=index_shape, dtype=np.int32))  # type: ignore[arg-type]
  data = jnp.array(data_rng(data_shape, dtype))

  if sparse_format == 'bcoo':
    return sparse.BCOO((data, bcoo_indices), shape=shape)

  bcsr_indices, bcsr_indptr = sparse_bcoo._bcoo_to_bcsr(
      bcoo_indices, shape=shape)
  return sparse.BCSR((data, bcsr_indices, bcsr_indptr), shape=shape)

def rand_bcoo(rng: np.random.RandomState,
              rand_method: Callable[..., Any]=jtu.rand_default,
              nse: Union[int, float]=0.5, n_batch: int=0, n_dense: int=0):
  """Generates a random BCOO array."""
  return functools.partial(_rand_sparse, rng=rng, rand_method=rand_method,
                           nse=nse, n_batch=n_batch, n_dense=n_dense,
                           sparse_format='bcoo')

def rand_bcsr(rng: np.random.RandomState,
              rand_method: Callable[..., Any]=jtu.rand_default,
              nse: Union[int, float]=0.5, n_batch: int=0, n_dense: int=0):
  """Generates a random BCSR array."""
  return functools.partial(_rand_sparse, rng=rng, rand_method=rand_method,
                           nse=nse, n_batch=n_batch, n_dense=n_dense,
                           sparse_format='bcsr')
