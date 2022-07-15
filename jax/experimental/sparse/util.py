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

"""Sparse utilities."""

import numpy as np
import jax
from jax import core
from jax._src import dtypes
from jax._src import stages
import jax.numpy as jnp

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

def _is_pytree_placeholder(*args):
  # Returns True if the arguments are consistent with being a placeholder within
  # pytree validation.
  return all(type(arg) is object for arg in args) or all(arg is None for arg in args)

def _is_aval(*args):
  return all(isinstance(arg, core.AbstractValue) for arg in args)

def _is_arginfo(*args):
  return all(isinstance(arg, stages.ArgInfo) for arg in args)

def _asarray_or_float0(arg):
  if isinstance(arg, np.ndarray) and arg.dtype == dtypes.float0:
    return arg
  return jnp.asarray(arg)

def _safe_asarray(args):
  if _is_pytree_placeholder(*args) or _is_aval(*args) or _is_arginfo(*args):
    return args
  return map(_asarray_or_float0, args)
