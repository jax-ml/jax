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
from jax.interpreters import xla
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jax.util import safe_map, safe_zip, split_list
from jax._src.lax.control_flow import _check_tree_and_avals
from jax._src.util import canonicalize_axis
from jax.experimental import sparse
from jax.experimental.sparse import BCOO

sparse_rules : Dict[core.Primitive, Callable] = {}

Array = Any


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
  data_ref: Optional[int]
  indices_ref: Optional[int]

  @property
  def ndim(self):
    return len(self.shape)

  def is_sparse(self):
    return self.indices_ref is not None

  def is_unit(self):
    return self.data_ref is None

  def data(self, spenv: SparseEnv):
    assert self.data_ref is not None
    return spenv.get(self.data_ref)

  def indices(self, spenv: SparseEnv):
    assert self.indices_ref is not None
    return spenv.get(self.indices_ref)

_is_bcoo = lambda arg: isinstance(arg, BCOO)
_is_argspec = lambda arg: isinstance(arg, ArgSpec)


def arrays_to_argspecs(
    spenv: SparseEnv,
    args: Any
    ) -> Any:
  """Convert a pytree of (sparse) arrays to an equivalent pytree of argspecs."""
  def array_to_argspec(arg):
    if isinstance(arg, BCOO):
      return ArgSpec(arg.shape, spenv.push(arg.data), spenv.push(arg.indices))
    elif core.get_aval(arg) is core.abstract_unit:
      return ArgSpec((), None, None)
    else:
      return ArgSpec(np.shape(arg), spenv.push(arg), None)
  return tree_map(array_to_argspec, args, is_leaf=_is_bcoo)


def argspecs_to_arrays(
    spenv: SparseEnv,
    argspecs: Any,
    ) -> Any:
  """Convert a pytree of argspecs to an equivalent pytree of (sparse) arrays."""
  def argspec_to_array(argspec):
    if argspec.is_sparse():
      assert argspec.indices_ref is not None
      return BCOO((argspec.data(spenv), argspec.indices(spenv)), shape=argspec.shape)
    elif argspec.is_unit():
      return core.unit
    else:
      return argspec.data(spenv)
  return tree_map(argspec_to_array, argspecs, is_leaf=_is_argspec)


def argspecs_to_avals(
    spenv: SparseEnv,
    argspecs: Any,
    ) -> Any:
  """Convert a pytree of argspecs to an equivalent pytree of abstract values."""
  def argspec_to_aval(argspec):
    if argspec.is_unit():
      return core.abstract_unit
    else:
      return core.ShapedArray(argspec.shape, argspec.data(spenv).dtype)
  return tree_map(argspec_to_aval, argspecs, is_leaf=_is_argspec)


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
      if prim is xla.xla_call_p:
        # TODO(vanderplas,frostig): workaround for binding call primitives
        # within a jaxpr interpreter
        params = eqn.params.copy()
        fun = lu.wrap_init(core.jaxpr_as_fun(pe.ClosedJaxpr(params.pop('call_jaxpr'), ())))
        out_bufs = prim.bind(fun, *(val.data(spenv) for val in invals), **params)
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
    argspecs_flat, in_tree = tree_flatten(argspecs, is_leaf=_is_argspec)
    in_avals_flat = argspecs_to_avals(spenv, argspecs_flat)
    wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(f, params), in_tree)
    jaxpr, out_avals_flat, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)
    result = eval_sparse(jaxpr, consts, argspecs_flat, spenv)
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
  if argspecs[0].is_sparse() and argspecs[1].is_sparse():
    raise NotImplementedError("dot_general between two sparse matrices.")
  A, B = argspecs_to_arrays(spenv, argspecs)
  if argspecs[0].is_sparse():
    result = sparse.bcoo_dot_general(A.data, A.indices, B, lhs_shape=A.shape,
                                    dimension_numbers=dimension_numbers)
  else:
    result = sparse.bcoo_rdot_general(A, B.data, B.indices, rhs_shape=B.shape,
                                      dimension_numbers=dimension_numbers)
  return [ArgSpec(result.shape, spenv.push(result), None)]

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

def _sparsify_jaxpr(spenv, jaxpr, *argspecs):
  # TODO(jakevdp): currently this approach discards all information about
  #   shared data & indices when generating the sparsified jaxpr. The
  #   current approach produces valid sparsified while loops, but they
  #   don't work in corner cases (see associated TODO in sparsify_test.py)
  out_tree = None

  @lu.wrap_init
  def wrapped(*args_flat):
    nonlocal out_tree
    args = tree_unflatten(in_tree, args_flat)
    argspecs = arrays_to_argspecs(spenv, args)
    result = eval_sparse(jaxpr.jaxpr, jaxpr.consts, argspecs, spenv)
    out = argspecs_to_arrays(spenv, result)
    out_flat, out_tree = tree_flatten(out)
    return out_flat

  args = argspecs_to_arrays(spenv, argspecs)
  args_flat, in_tree = tree_flatten(args)
  avals_flat = [core.raise_to_shaped(core.get_aval(arg)) for arg in args_flat]
  sp_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped, avals_flat)
  sp_jaxpr = pe.ClosedJaxpr(pe.convert_constvars_jaxpr(sp_jaxpr), consts)
  return sp_jaxpr, out_tree

def _while_sparse(spenv, *argspecs, cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts):
  cond_const_argspecs, body_const_argspecs, init_val_argspecs = split_list(
    argspecs, [cond_nconsts, body_nconsts])

  cond_sp_jaxpr, _ = _sparsify_jaxpr(spenv, cond_jaxpr, *cond_const_argspecs, *init_val_argspecs)
  body_sp_jaxpr, out_tree = _sparsify_jaxpr(spenv, body_jaxpr, *body_const_argspecs, *init_val_argspecs)

  cond_consts, _ = tree_flatten(argspecs_to_arrays(spenv, cond_const_argspecs))
  body_consts, _ = tree_flatten(argspecs_to_arrays(spenv, body_const_argspecs))
  init_vals, _ = tree_flatten(argspecs_to_arrays(spenv, init_val_argspecs))

  out_flat = lax.while_p.bind(*cond_consts, *body_consts, *init_vals,
                              cond_nconsts=len(cond_consts), cond_jaxpr=cond_sp_jaxpr,
                              body_nconsts=len(body_consts), body_jaxpr=body_sp_jaxpr)
  return arrays_to_argspecs(spenv, tree_unflatten(out_tree, out_flat))

sparse_rules[lax.while_p] = _while_sparse

def _xla_call_sparse(spenv, *argspecs, call_jaxpr, donated_invars, **params):
  if any(donated_invars):
    raise NotImplementedError("sparse xla_call with donated_invars")
  sp_call_jaxpr, out_tree = _sparsify_jaxpr(spenv, pe.ClosedJaxpr(call_jaxpr, ()), *argspecs)
  fun = lu.wrap_init(core.jaxpr_as_fun(sp_call_jaxpr))
  args_flat, _ = tree_flatten(argspecs_to_arrays(spenv, argspecs))
  donated_invars = tuple(False for arg in args_flat)
  out_flat = xla.xla_call_p.bind(fun, *args_flat, donated_invars=donated_invars, **params)
  return arrays_to_argspecs(spenv, tree_unflatten(out_tree, out_flat))

sparse_rules[xla.xla_call_p] = _xla_call_sparse


def _duplicate_for_sparse_argspecs(argspecs, params):
  for argspec, param in safe_zip(argspecs, params):
      yield from [param, param] if argspec.is_sparse() else [param]


def _scan_sparse(spenv, *argspecs, jaxpr, num_consts, num_carry, **params):
  const_argspecs, carry_argspecs, xs_argspecs = split_list(
    argspecs, [num_consts, num_carry])
  if xs_argspecs:
    # TODO(jakevdp): we don't want to pass xs_argspecs, we want to pass one row
    # of xs argspecs. How to do this?
    raise NotImplementedError("sparse rule for scan with x values.")
  sp_jaxpr, _ = _sparsify_jaxpr(spenv, jaxpr, *const_argspecs, *carry_argspecs, *xs_argspecs)

  consts, _ = tree_flatten(argspecs_to_arrays(spenv, const_argspecs))
  carry, carry_tree = tree_flatten(argspecs_to_arrays(spenv, carry_argspecs))
  xs, xs_tree = tree_flatten(argspecs_to_arrays(spenv, xs_argspecs))

  # params['linear'] has one entry per arg; expand it to match the sparsified args.
  const_linear, carry_linear, xs_linear = split_list(
    params.pop('linear'), [num_consts, num_carry])
  sp_linear = tuple([
    *_duplicate_for_sparse_argspecs(const_argspecs, const_linear),
    *_duplicate_for_sparse_argspecs(carry_argspecs, carry_linear),
    *_duplicate_for_sparse_argspecs(xs_argspecs, xs_linear)])

  out = lax.scan_p.bind(*consts, *carry, *xs, jaxpr=sp_jaxpr, linear=sp_linear,
                        num_consts=len(consts), num_carry=len(carry), **params)
  carry_out = tree_unflatten(carry_tree, out[:len(carry)])
  xs_out = tree_unflatten(xs_tree, out[len(carry):])
  return arrays_to_argspecs(spenv, carry_out + xs_out)

sparse_rules[lax.scan_p] = _scan_sparse

def _cond_sparse(spenv, pred, *operands, branches, linear, **params):
  sp_branches, treedefs = zip(*(_sparsify_jaxpr(spenv, jaxpr, *operands)
                                for jaxpr in branches))
  _check_tree_and_avals("sparsified true_fun and false_fun output",
                        treedefs[0], sp_branches[0].out_avals,
                        treedefs[1], sp_branches[1].out_avals)
  sp_linear = tuple(_duplicate_for_sparse_argspecs(operands, linear))
  args, _ = tree_flatten(argspecs_to_arrays(spenv, (pred, *operands)))
  out_flat = lax.cond_p.bind(*args, branches=sp_branches, linear=sp_linear, **params)
  out = tree_unflatten(treedefs[0], out_flat)
  return arrays_to_argspecs(spenv, out)

sparse_rules[lax.cond_p] = _cond_sparse
