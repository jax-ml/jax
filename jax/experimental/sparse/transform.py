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

"""
Sparsify transform
==================

This is an experimental JAX transform that will allow arbitrary JAX functions to accept
sparse matrices as inputs, so long as sparse rules are implemented for the primitives
called by the function.

For example:

>>> import jax.numpy as jnp
>>> from jax import random
>>> from jax.experimental.sparse import BCOO, sparsify

>>> mat = random.uniform(random.PRNGKey(1701), (5, 5))
>>> mat = mat.at[mat < 0.5].set(0)
>>> vec = random.uniform(random.PRNGKey(42), (5,))

>>> def f(mat, vec):
...   return -(jnp.sin(mat) @ vec)
...
>>> f(mat, vec)
DeviceArray([-1.2655463 , -0.52060574, -0.14522289, -0.10817424,
             -0.15574613], dtype=float32)

>>> mat_sparse = BCOO.fromdense(mat)
>>> mat_sparse
BCOO(float32[5, 5], nse=8)

>>> sparsify(f)(mat_sparse, vec)
DeviceArray([-1.2655463 , -0.52060574, -0.14522289, -0.10817424,
             -0.15574613], dtype=float32)
"""

import functools
from typing import (
  Any, Callable, Dict, NamedTuple, List, Optional, Sequence, Tuple, Union)

import numpy as np

from jax import core
from jax import lax
from jax import linear_util as lu
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_unflatten
from jax.util import safe_map
from jax._src.util import canonicalize_axis
from jax.experimental import sparse
from jax.experimental.sparse import BCOO

sparse_rules : Dict[core.Primitive, Callable] = {}

Array = Any
AnyArray = Union[Array, BCOO]


class SparseEnv:
  """Environment for sparse jaxpr evaluation."""
  _buffers : List[Array]

  def __init__(self, bufs=()):
    self._buffers = list(bufs)

  def push(self, arr: Array) -> int:
    self._buffers.append(np.array(arr) if np.isscalar(arr) else arr)  # type: ignore
    return len(self._buffers) - 1

  def get(self, ind: int) -> Array:
    return self._buffers[ind]

  def size(self):
    return len(self._buffers)


class ArgSpec(NamedTuple):
  shape: Tuple[int, ...]
  data_ref: int
  indices_ref: Optional[int]

  @property
  def ndim(self):
    return len(self.shape)

  def is_sparse(self):
    return self.indices_ref is not None

  def data(self, spenv: SparseEnv):
    return spenv.get(self.data_ref)

  def indices(self, spenv: SparseEnv):
    assert self.indices_ref is not None
    return spenv.get(self.indices_ref)


def arrays_to_argspecs(
    spenv: SparseEnv,
    args: Sequence[AnyArray]
    ) -> Sequence[ArgSpec]:
  argspecs: List[ArgSpec] = []
  for arg in args:
    if isinstance(arg, BCOO):
      argspecs.append(ArgSpec(arg.shape, spenv.push(arg.data), spenv.push(arg.indices)))  # type: ignore
    else:
      argspecs.append(ArgSpec(np.shape(arg), spenv.push(arg), None))  # type: ignore
  return argspecs


def argspecs_to_arrays(
    spenv: SparseEnv,
    argspecs: Sequence[ArgSpec],
    ) -> Sequence[AnyArray]:
  args = []
  for argspec in argspecs:
    if argspec.is_sparse():
      assert argspec.indices_ref is not None
      args.append(BCOO((argspec.data(spenv), argspec.indices(spenv)), shape=argspec.shape))
    else:
      args.append(argspec.data(spenv))
    assert args[-1].shape == argspec.shape
  return tuple(args)


def argspecs_to_avals(
    spenv: SparseEnv,
    argspecs: Sequence[ArgSpec],
    ) -> Sequence[core.ShapedArray]:
  return [core.ShapedArray(a.shape, a.data(spenv).dtype) for a in argspecs]


def eval_sparse(
    jaxpr: core.Jaxpr,
    consts: Sequence[Array],  # all consts are dense
    argspecs: Sequence[ArgSpec],  # mix of sparse and dense pointers into spenv
    spenv: SparseEnv,
) -> Sequence[ArgSpec]:
  env : Dict[core.Var, ArgSpec] = {}

  def read(var: core.Var) -> Union[Array, ArgSpec]:
    # all literals are dense
    if isinstance(var, core.Literal):
      return ArgSpec(np.shape(var.val), spenv.push(var.val), None)
    else:
      return env[var]

  def write_buffer(var: core.Var, a: Array) -> None:
    if var is core.dropvar:
      return
    env[var] = ArgSpec(a.shape, spenv.push(a), None)

  def write(var: core.Var, a: ArgSpec) -> None:
    if var is core.dropvar:
      return
    env[var] = a

  # TODO: handle unitvar at all?
  #write_buffer(core.unitvar, core.unit)
  safe_map(write_buffer, jaxpr.constvars, consts)
  safe_map(write, jaxpr.invars, argspecs)

  for eqn in jaxpr.eqns:
    prim = eqn.primitive
    invals = safe_map(read, eqn.invars)

    if any(val.is_sparse() for val in invals):
      if prim not in sparse_rules:
        raise NotImplementedError(f"sparse rule for {prim}")
      out = sparse_rules[prim](spenv, *invals, **eqn.params)
    else:
      out_bufs = prim.bind(*(val.data(spenv) for val in invals), **eqn.params)
      out_bufs = out_bufs if prim.multiple_results else [out_bufs]
      out = []
      for buf in out_bufs:
        out.append(ArgSpec(buf.shape, spenv.push(buf), None))
    safe_map(write, eqn.outvars, out)

  return safe_map(read, jaxpr.outvars)

def sparsify_raw(f):
  def wrapped(spenv: SparseEnv, *argspecs: ArgSpec, **params: Any) -> Tuple[Sequence[ArgSpec], bool]:
    in_avals = argspecs_to_avals(spenv, argspecs)
    in_avals_flat, in_tree = tree_flatten(in_avals)
    wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
    jaxpr, out_avals_flat, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)
    result = eval_sparse(jaxpr, consts, argspecs, spenv)
    if len(out_avals_flat) != len(result):
      raise Exception("Internal: eval_sparse does not return expected number of arguments. "
                      "Got {result} for avals {out_avals_flat}")
    return result, out_tree()
  return wrapped

def sparsify(f):
  f_raw = sparsify_raw(f)
  @functools.wraps(f)
  def wrapped(*args, **params):
    spenv = SparseEnv()
    argspecs = arrays_to_argspecs(spenv, args)
    argspecs_out, out_tree = f_raw(spenv, *argspecs, **params)
    out = argspecs_to_arrays(spenv, argspecs_out)
    return tree_unflatten(out_tree, out)
  return wrapped

def _zero_preserving_unary_op(prim):
  def func(spenv, *argspecs, **kwargs):
    assert len(argspecs) == 1
    buf = argspecs[0].data(spenv)
    buf_out = prim.bind(buf, **kwargs)
    out_argspec = ArgSpec(argspecs[0].shape, spenv.push(buf_out), argspecs[0].indices_ref)
    return (out_argspec,)
  return func

# TODO(jakevdp): some of these will give incorrect results when there are duplicated indices.
#                how should we handle this?
for _prim in [
    lax.abs_p, lax.expm1_p, lax.log1p_p, lax.neg_p, lax.sign_p, lax.sin_p,
    lax.sinh_p, lax.sqrt_p, lax.tan_p, lax.tanh_p, lax.convert_element_type_p
  ]:
  sparse_rules[_prim] = _zero_preserving_unary_op(_prim)

def _dot_general_sparse(spenv, *argspecs, dimension_numbers, precision, preferred_element_type):
  assert len(argspecs) == 2
  assert argspecs[0].is_sparse() and not argspecs[1].is_sparse()
  args = argspecs_to_arrays(spenv, argspecs)

  result = sparse.bcoo_dot_general(args[0].data, args[0].indices, args[1],
                                   lhs_shape=args[0].shape,
                                   dimension_numbers=dimension_numbers)
  argspec = ArgSpec(result.shape, spenv.push(result), None)
  return [argspec]

sparse_rules[lax.dot_general_p] = _dot_general_sparse

def _transpose_sparse(spenv, *argspecs, permutation):
  permutation = tuple(permutation)
  args = argspecs_to_arrays(spenv, argspecs)
  shape = args[0].shape
  data, indices = sparse.bcoo_transpose(args[0].data, args[0].indices,
                                        permutation=permutation,
                                        shape=shape)
  out_shape = tuple(shape[i] for i in permutation)

  n_batch = args[0].indices.ndim - 2
  n_sparse = args[0].indices.shape[-2]
  batch_dims_unchanged = (permutation[:n_batch] == tuple(range(n_batch)))
  dense_dims_unchanged = (permutation[n_batch + n_sparse:] == tuple(range(n_batch + n_sparse, len(shape))))
  sparse_dims_unchanged = (permutation[n_batch:n_batch + n_sparse] == tuple(range(n_batch, n_batch + n_sparse)))

  # Data is unchanged if batch & dense dims are not permuted
  if batch_dims_unchanged and dense_dims_unchanged:
    data_ref = argspecs[0].data_ref
  else:
    data_ref = spenv.push(data)

  # Indices unchanged if batch & sparse dims are not permuted
  if batch_dims_unchanged and sparse_dims_unchanged:
    indices_ref = argspecs[0].indices_ref
  else:
    indices_ref = spenv.push(indices)

  argspec = ArgSpec(out_shape, data_ref, indices_ref)
  return (argspec,)

sparse_rules[lax.transpose_p] = _transpose_sparse

def _add_sparse(spenv, *argspecs):
  X, Y = argspecs
  if X.is_sparse() and Y.is_sparse():
    if X.shape != Y.shape:
      raise NotImplementedError("Addition between sparse matrices of different shapes.")
    if X.indices_ref == Y.indices_ref:
      out_data = lax.add(X.data(spenv), Y.data(spenv))
      out_argspec = ArgSpec(X.shape, spenv.push(out_data), X.indices_ref)
    elif X.indices(spenv).ndim != Y.indices(spenv).ndim or X.data(spenv).ndim != Y.data(spenv).ndim:
      raise NotImplementedError("Addition between sparse matrices with different batch/dense dimensions.")
    else:
      out_indices = lax.concatenate([X.indices(spenv), Y.indices(spenv)],
                                    dimension=X.indices(spenv).ndim - 1)
      out_data = lax.concatenate([X.data(spenv), Y.data(spenv)],
                                 dimension=X.indices(spenv).ndim - 2)
      out_argspec = ArgSpec(X.shape, spenv.push(out_data), spenv.push(out_indices))
  else:
    raise NotImplementedError("Addition between sparse and dense matrix.")

  return (out_argspec,)

sparse_rules[lax.add_p] = _add_sparse

def _mul_sparse(spenv, *argspecs):
  X, Y = argspecs
  if X.is_sparse() and Y.is_sparse():
    if X.shape != Y.shape:
      raise NotImplementedError("Multiplication between sparse matrices of different shapes.")
    if X.indices_ref == Y.indices_ref:
      out_data = lax.mul(X.data(spenv), Y.data(spenv))
      out_argspec = ArgSpec(X.shape, spenv.push(out_data), X.indices_ref)
    elif X.indices(spenv).ndim != Y.indices(spenv).ndim or X.data(spenv).ndim != Y.data(spenv).ndim:
      raise NotImplementedError("Multiplication between sparse matrices with different batch/dense dimensions.")
    else:
      raise NotImplementedError("Multiplication between sparse matrices with different sparsity patterns.")
  else:
    if Y.is_sparse():
      X, Y = Y, X
    Ydata = Y.data(spenv)
    if Ydata.ndim == 0:
      out_data = lax.mul(X.data(spenv), Ydata)
    elif Ydata.shape == X.shape:
      out_data = lax.mul(X.data(spenv), sparse.bcoo_extract(X.indices(spenv), Ydata))
    else:
      raise NotImplementedError("Multiplication between sparse and dense matrices of different shape.")
    out_argspec = ArgSpec(X.shape, spenv.push(out_data), X.indices_ref)

  return (out_argspec,)

sparse_rules[lax.mul_p] = _mul_sparse

def _reduce_sum_sparse(spenv, *argspecs, axes):
  X, = argspecs
  data, indices, out_shape = sparse.bcoo_reduce_sum(
      X.data(spenv), X.indices(spenv), shape=X.shape, axes=axes)
  if out_shape == ():
    out_argspec = ArgSpec(out_shape, spenv.push(data.sum()), None)
  else:
    out_argspec = ArgSpec(out_shape, spenv.push(data), spenv.push(indices))
  return (out_argspec,)

sparse_rules[lax.reduce_sum_p] = _reduce_sum_sparse


def _squeeze_sparse(spenv, *argspecs, dimensions):
  arr, = argspecs
  dimensions = tuple(canonicalize_axis(dim, arr.ndim) for dim in dimensions)
  if any(arr.shape[dim] != 1 for dim in dimensions):
    raise ValueError("cannot select an axis to squeeze out which has size not equal to one, "
                     f"got shape={arr.shape} and dimensions={dimensions}")
  data = arr.data(spenv)
  indices = arr.indices(spenv)
  n_sparse = indices.shape[-2]
  n_batch = indices.ndim - 2
  batch_dims = tuple(d for d in dimensions if d < n_batch)
  sparse_dims = np.array([i for i in range(n_sparse) if i + n_batch not in dimensions], dtype=int)
  dense_dims = tuple(d - n_sparse + 1 for d in dimensions if d >= n_batch + n_sparse)
  data_out = lax.squeeze(data, batch_dims + dense_dims)
  indices_out = lax.squeeze(indices[..., sparse_dims, :], batch_dims)
  out_shape = tuple(s for i, s in enumerate(arr.shape) if i not in dimensions)
  return (ArgSpec(out_shape, spenv.push(data_out), spenv.push(indices_out)),)

sparse_rules[lax.squeeze_p] = _squeeze_sparse
