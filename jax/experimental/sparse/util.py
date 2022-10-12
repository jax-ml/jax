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

"""Sparse utilities."""

import functools

import numpy as np
import jax
from jax import core
from jax import lax
from jax import tree_util
from jax import vmap
from jax._src import dtypes
from jax._src import stages
from jax._src.api_util import flatten_axes
import jax.numpy as jnp
from jax.util import safe_zip

class SparseEfficiencyError(ValueError):
  pass

class SparseEfficiencyWarning(UserWarning):
  pass

class CuSparseEfficiencyWarning(SparseEfficiencyWarning):
  pass

#--------------------------------------------------------------------
# utilities
# TODO: possibly make these primitives, targeting cusparse rountines
#       csr2coo/coo2csr/SPDDMM

def _asarray_or_float0(arg):
  if isinstance(arg, np.ndarray) and arg.dtype == dtypes.float0:
    return arg
  return jnp.asarray(arg)

def _broadcasting_vmap(fun, in_axes=0, out_axes=0):
  @functools.wraps(fun)
  def batched_fun(*args):
    args_flat, in_tree  = tree_util.tree_flatten(args)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, in_axes, kws=False)
    size = max(arg.shape[i] for arg, i in safe_zip(args_flat, in_axes_flat) if i is not None)
    if size > 1:
      if any(i is not None and arg.shape[i] not in (1, size)
             for arg, i in safe_zip(args_flat, in_axes_flat)):
        raise ValueError("broadcasting_vmap: mismatched input shapes")
      args_flat, in_axes_flat = zip(*(
          (arg, None) if i is None else (lax.squeeze(arg, (i,)), None) if arg.shape[i] == 1 else (arg, i)
          for arg, i in zip(args_flat, in_axes_flat)
      ))
    new_args = tree_util.tree_unflatten(in_tree, args_flat)
    new_in_axes = tree_util.tree_unflatten(in_tree, in_axes_flat)
    return vmap(fun, in_axes=new_in_axes, out_axes=out_axes)(*new_args)
  return batched_fun

@jax.jit
def _csr_to_coo(indices, indptr):
  """Given CSR (indices, indptr) return COO (row, col)"""
  return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices

def _csr_extract(indices, indptr, mat):
  """Extract values of dense matrix mat at given CSR indices."""
  return _coo_extract(*_csr_to_coo(indices, indptr), mat)

def _coo_extract(row, col, mat):
  """Extract values of dense matrix mat at given COO indices."""
  return mat[row, col]

def _count_stored_elements_per_batch(mat, n_batch=0, n_dense=0):
  """Return per-batch number of stored elements (nse) of a dense matrix."""
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  mask = mask.sum(list(range(n_batch, mask.ndim)))
  return mask

def _count_stored_elements(mat, n_batch=0, n_dense=0):
  """Return the number of stored elements (nse) of the given dense matrix."""
  return int(_count_stored_elements_per_batch(mat, n_batch, n_dense).max())

def _is_pytree_placeholder(*args):
  # Returns True if the arguments are consistent with being a placeholder within
  # pytree validation.
  return all(type(arg) is object for arg in args) or all(arg is None for arg in args)

def _is_aval(*args):
  return all(isinstance(arg, core.AbstractValue) for arg in args)

def _is_arginfo(*args):
  return all(isinstance(arg, stages.ArgInfo) for arg in args)

def _safe_asarray(args):
  if _is_pytree_placeholder(*args) or _is_aval(*args) or _is_arginfo(*args):
    return args
  return map(_asarray_or_float0, args)
