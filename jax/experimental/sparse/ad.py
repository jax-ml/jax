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

from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np

import jax
from jax import core
from jax import tree_util
from jax._src.api_util import _ensure_index, _ensure_index_tuple
from jax.util import safe_zip
from jax._src.util import wraps
from jax._src.traceback_util import api_boundary
from jax.experimental.sparse.bcoo import BCOO


def value_and_grad(fun: Callable,
                   argnums: Union[int, Sequence[int]] = 0,
                   **kwargs) -> Callable[..., Tuple[Any, Any]]:
  """Sparse-aware version of :func:`jax.value_and_grad`

  Arguments and return values are the same as :func:`jax.value_and_grad`, but when
  taking the gradient with respect to a BCOO matrix, the matrix indices are ignored.
  """
  # The approach here is to set allow_int=True (so that gradients of indices don't raise an error)
  # and then at the end replace the float0 outputs with the input indices.
  allow_int = kwargs.pop('allow_int', False)
  kwargs['allow_int'] = True
  raw_value_and_grad_fun = jax.value_and_grad(fun, argnums=argnums, **kwargs)
  argnums = core.concrete_or_error(_ensure_index, argnums)

  def maybe_copy_index(arg_in, arg_out):
    if isinstance(arg_in, BCOO) and isinstance(arg_out, BCOO):
      assert arg_in.indices.shape == arg_out.indices.shape
      return BCOO((arg_out.data, arg_in.indices), shape=arg_out.shape)
    else:
      return arg_out

  @wraps(fun, docstr=raw_value_and_grad_fun.__doc__, argnums=argnums)
  @api_boundary
  def value_and_grad_fun(*args, **kwargs):
    if not allow_int:
      dyn_args = [args[i] for i in _ensure_index_tuple(argnums)]
      dyn_args_flat, _ = tree_util.tree_flatten(dyn_args, is_leaf=lambda arg: isinstance(arg, BCOO))
      for arg in dyn_args_flat:
        dtype = np.dtype(arg)
        if not (np.issubdtype(arg, np.floating) or np.issubdtype(arg, np.complexfloating)):
          raise TypeError("grad requires real- or complex-valued inputs (input dtype that "
                          "is a sub-dtype of np.floating or np.complexfloating), "
                          f"but got {dtype.name}. If you want to use integer-valued "
                          "inputs, set allow_int to True.")
    value, grad = raw_value_and_grad_fun(*args, **kwargs)
    if isinstance(argnums, int):
      grad = maybe_copy_index(args[argnums], grad)
    else:
      grad = tuple(maybe_copy_index(args[argnum], g) for argnum, g in safe_zip(argnums, grad))
    return value, grad

  return value_and_grad_fun


def grad(fun: Callable,
         argnums: Union[int, Sequence[int]] = 0,
         has_aux=False, **kwargs) -> Callable:
  """Sparse-aware version of jax.grad

  Arguments and return values are the same as :func:`jax.grad`, but when taking
  the gradient with respect to a BCOO matrix, the matrix indices are ignored.
  """
  value_and_grad_f = value_and_grad(fun, argnums, has_aux=has_aux, **kwargs)

  docstr = ("Gradient of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "gradient, which has the same shape as the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f(*args, **kwargs):
    _, g = value_and_grad_f(*args, **kwargs)
    return g

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f_aux(*args, **kwargs):
    (_, aux), g = value_and_grad_f(*args, **kwargs)
    return g, aux

  return grad_f_aux if has_aux else grad_f
