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

"""JAX primitives related to sparse operations.

This is experimental work to explore sparse support in JAX.

The primitives defined here are deliberately low-level: each primitive implements
a common sparse operation (sparse to dense, dense to sparse, sparse matrix/vector
product, sparse matrix/matrix product) for two common sparse representations
(CSR and COO).

These routines have reference implementations defined via XLA scatter/gather
operations that will work on any backend, although they are not particularly
performant. On GPU runtimes built against CUDA 11.0/ROCm 5.0 or newer, each operation is
computed efficiently via cusparse/hipsparse.

Further down are some examples of potential high-level wrappers for sparse objects.
(API should be considered unstable and subject to change).
"""
from functools import partial
import operator
from typing import Callable, Optional, Union
import numpy as np

import jax
from jax import core
from jax import tree_util
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.bcoo import BCOO, _bcoo_extract
from jax.experimental.sparse.bcsr import BCSR
from jax.experimental.sparse.coo import COO
from jax.experimental.sparse.csr import CSR, CSC
from jax.experimental.sparse.util import _coo_extract
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax._src import dtypes
from jax._src.typing import Array, ArrayLike, DTypeLike, Shape
from jax._src import util
from jax._src.lax.control_flow import _initial_style_jaxpr


#----------------------------------------------------------------------
# todense – function to convert sparse matrices to dense while letting
#           dense matrices pass through.
todense_p = core.Primitive('todense')
todense_p.multiple_results = False

def todense(arr: Union[JAXSparse, Array]) -> Array:
  """Convert input to a dense matrix. If input is already dense, pass through."""
  bufs, tree = tree_util.tree_flatten(arr)
  return todense_p.bind(*bufs, tree=tree)

@todense_p.def_impl
def _todense_impl(*bufs, tree):
  arr = tree_util.tree_unflatten(tree, bufs)
  return arr.todense() if isinstance(arr, JAXSparse) else arr

@todense_p.def_abstract_eval
def _todense_abstract_eval(*bufs, tree):
  arr = tree_util.tree_unflatten(tree, bufs)
  if isinstance(arr, core.ShapedArray):
    return arr
  return core.ShapedArray(arr.shape, arr.dtype, weak_type=dtypes.is_weakly_typed(arr.data))

def _todense_jvp(primals, tangents, *, tree):
  assert not isinstance(tangents[0], ad.Zero)
  assert all(isinstance(t, ad.Zero) for t in tangents[1:])
  primals_out = todense_p.bind(*primals, tree=tree)
  tangents_out = todense_p.bind(tangents[0], *primals[1:], tree=tree)
  return primals_out, tangents_out

def _todense_transpose(ct, *bufs, tree):
  assert ad.is_undefined_primal(bufs[0])
  assert not any(ad.is_undefined_primal(buf) for buf in bufs[1:])

  standin = object()
  obj = tree_util.tree_unflatten(tree, [standin] * len(bufs))
  if obj is standin:
    return (ct,)
  elif isinstance(obj, BCOO):
    _, indices = bufs
    return _bcoo_extract(indices, ct), indices
  elif isinstance(obj, COO):
    _, row, col = bufs
    return _coo_extract(row, col, ct), row, col
  else:
    raise NotImplementedError(f"todense_transpose for {type(obj)}")

def _todense_batching_rule(batched_args, batch_dims, *, tree):
  return jax.vmap(partial(_todense_impl, tree=tree), batch_dims)(*batched_args), 0

ad.primitive_jvps[todense_p] = _todense_jvp
ad.primitive_transposes[todense_p] = _todense_transpose
batching.primitive_batchers[todense_p] = _todense_batching_rule
mlir.register_lowering(todense_p, mlir.lower_fun(
    _todense_impl, multiple_results=False))

#----------------------------------------------------------------------
# map_specified_elements – function to map a scalar function to specified elements
map_specified_elements_p = core.Primitive('map_specified_elements')
map_specified_elements_p.multiple_results = True

def map_specified_elements(fun: Callable[[ArrayLike], ArrayLike], arg: Union[ArrayLike, JAXSparse]) -> Union[ArrayLike, JAXSparse]:
  """Map a given function across the specified elements of an array

  If the input is a dense array, ``fun`` will be applied to every array element.
  If the input is a sparse array, ``fun`` will be applied only to the data buffer.

  Args:
    fun : a function that accepts a single scalar value and returns a
      single scalar value.
    arg : a dense or sparse array.

  Returns:
    out : a dense or sparse array; same format as the input.
  """
  args = (arg,)
  bufs, tree = tree_util.tree_flatten(args)
  # TODO(jakevdp): check that arg is a sparse array or dense array, rather than
  # a more general PyTree.
  jaxpr, consts = _create_jaxpr(fun, bufs[0])
  bufs_out = map_specified_elements_p.bind(*consts, *bufs, jaxpr=jaxpr)
  return tree_util.tree_unflatten(tree, bufs_out)[0]

def _create_jaxpr(fun, flat_val):
  aval = core.raise_to_shaped(core.get_aval(flat_val))
  aval_scalar = core.ShapedArray((), aval.dtype, aval.weak_type)
  _, trivial_tree = tree_util.tree_flatten([1])
  jaxpr, consts, out_tree = _initial_style_jaxpr(
    fun, trivial_tree, (aval_scalar,), "map_specified_elements_fun")
  # TODO(jakevdp) convert asserts into comprehensible user errors
  assert out_tree == tree_util.tree_flatten(1)[1]
  assert len(jaxpr.out_avals) == 1
  assert jaxpr.out_avals[0].shape == ()
  return jaxpr, consts

@map_specified_elements_p.def_impl
def _map_specified_elements_impl(*args, jaxpr):
  # Note: tree here is a way to pass along sparse matrix metadata; it's not a general tree.
  # TODO(jakevdp): use metadata to check for duplicate entries & handle appropriately
  num_consts = len(jaxpr.jaxpr.constvars)
  num_args = len(jaxpr.jaxpr.invars)
  consts, args, bufs = util.split_list(args, [num_consts, num_args])
  fun = core.jaxpr_as_fun(jaxpr, *consts)

  if not all(arg.shape == args[0].shape for arg in args[1:]):
    raise ValueError(f"arg shapes must match; got {args}")

  for _ in range(args[0].ndim):
    fun = jax.vmap(fun)
  return (*fun(*args), *bufs)

@map_specified_elements_p.def_abstract_eval
def _map_specified_elements_abstract_eval(*args, jaxpr):
  num_consts = len(jaxpr.jaxpr.constvars)
  num_args = len(jaxpr.jaxpr.invars)
  assert len(jaxpr.in_avals) == len(jaxpr.out_avals)
  _, args, bufs = util.split_list(args, [num_consts, num_args])
  out_avals = (core.ShapedArray(np.shape(arg), out.dtype, out.weak_type)
               for (arg, out) in zip(args, jaxpr.out_avals))
  return (*out_avals, *bufs)

def _map_specified_elements_batching_rule(batched_args, batch_dims, *, jaxpr):
  bdims_out = batch_dims[len(jaxpr.jaxpr.constvars):]
  mapped = jax.vmap(partial(_map_specified_elements_impl, jaxpr=jaxpr),
                    in_axes=batch_dims, out_axes=bdims_out)
  return mapped(*batched_args), bdims_out

def _map_specified_elements_jvp(primals, tangents, jaxpr):
  num_consts = len(jaxpr.jaxpr.constvars)
  num_args = len(jaxpr.jaxpr.invars)
  const_p, arg_p, buf_p = util.split_list(primals, [num_consts, num_args])
  const_t, arg_t, buf_t = util.split_list(tangents, [num_consts, num_args])
  assert all(isinstance(t, ad.Zero) for t in const_t)
  arg_nz = [not isinstance(t, ad.Zero) for t in arg_t]
  assert all(isinstance(t, ad.Zero) for t in buf_t)

  jaxpr_jvp, _ = ad.jvp_jaxpr(jaxpr, arg_nz, instantiate=False)
  outs = map_specified_elements_p.bind(*const_p, *const_t, *arg_p, *arg_t, jaxpr=jaxpr_jvp)
  primals_out, tangents_out = util.split_list(outs, [len(arg_p)])
  return (*primals_out, *buf_p), (*tangents_out, *buf_t)

ad.primitive_jvps[map_specified_elements_p] = _map_specified_elements_jvp
batching.primitive_batchers[map_specified_elements_p] = _map_specified_elements_batching_rule
mlir.register_lowering(map_specified_elements_p, mlir.lower_fun(
    _map_specified_elements_impl, multiple_results=True))

def empty(shape: Shape, dtype: Optional[DTypeLike]=None, index_dtype: DTypeLike = 'int32',
          sparse_format: str = 'bcoo', **kwds) -> JAXSparse:
  """Create an empty sparse array.

  Args:
    shape: sequence of integers giving the array shape.
    dtype: (optional) dtype of the array.
    index_dtype: (optional) dtype of the index arrays.
    format: string specifying the matrix format (e.g. ['bcoo']).
    **kwds: additional keywords passed to the format-specific _empty constructor.
  Returns:
    mat: empty sparse matrix.
  """
  formats = {'bcsr': BCSR, 'bcoo': BCOO, 'coo': COO, 'csr': CSR, 'csc': CSC}
  if sparse_format not in formats:
    raise ValueError(f"sparse_format={sparse_format!r} not recognized; "
                     f"must be one of {list(formats.keys())}")
  cls = formats[sparse_format]
  return cls._empty(shape, dtype=dtype, index_dtype=index_dtype, **kwds)


def eye(N: int, M: Optional[int] = None, k: int = 0, dtype: Optional[DTypeLike] = None,
        index_dtype: DTypeLike = 'int32', sparse_format: str = 'bcoo', **kwds) -> JAXSparse:
  """Create 2D sparse identity matrix.

  Args:
    N: int. Number of rows in the output.
    M: int, optional. Number of columns in the output. If None, defaults to `N`.
    k: int, optional. Index of the diagonal: 0 (the default) refers to the main
       diagonal, a positive value refers to an upper diagonal, and a negative value
       to a lower diagonal.
    dtype: data-type, optional. Data-type of the returned array.
    index_dtype: (optional) dtype of the index arrays.
    format: string specifying the matrix format (e.g. ['bcoo']).
    **kwds: additional keywords passed to the format-specific _empty constructor.

  Returns:
    I: two-dimensional sparse matrix with ones along the k-th diagonal.
  """
  formats = {'bcoo': BCOO, 'coo': COO, 'csr': CSR, 'csc': CSC}
  if M is None:
    M = N
  N = core.concrete_or_error(operator.index, N)
  M = core.concrete_or_error(operator.index, M)
  k = core.concrete_or_error(operator.index, k)

  cls = formats[sparse_format]
  return cls._eye(M=M, N=N, k=k, dtype=dtype, index_dtype=index_dtype, **kwds)
