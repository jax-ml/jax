# Copyright 2020 Google LLC
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
import functools
import re
import textwrap
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as onp

from .. import api
from .. import lax
from .. import linear_util as lu
from . import lax_numpy as np
from ..util import safe_map as map, safe_zip as zip
from .lax_numpy import _wraps


# See http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)


CoreDims = Tuple[str, ...]
NDArray = Any


def _parse_gufunc_signature(
    signature: str,
) -> Tuple[List[CoreDims], List[CoreDims]]:
  """Parse string signatures for a generalized universal function.

  Args:
    signature: generalized universal function signature, e.g.,
      ``(m,n),(n,p)->(m,p)`` for ``np.matmul``.

  Returns:
    Tuple of input and output core dimensions parsed from the signature.
  """
  if not re.match(_SIGNATURE, signature):
    raise ValueError(
        'not a valid gufunc signature: {}'.format(signature))
  return tuple([tuple(re.findall(_DIMENSION_NAME, arg))
                for arg in re.findall(_ARGUMENT, arg_list)]
               for arg_list in signature.split('->'))


def _update_dim_sizes(
    dim_sizes: Dict[str, int],
    shape: Tuple[int, ...],
    core_dims: CoreDims,
    error_context: str,
    *,
    is_input: bool,
):
  """Incrementally check and update core dimension sizes for a single argument.

  Args:
    dim_sizes: sizes of existing core dimensions. Will be updated in-place.
    shape: shape of this argument.
    core_dims: core dimensions for this argument.

  """
  num_core_dims = len(core_dims)
  if is_input:
    if len(shape) < num_core_dims:
      raise ValueError(
          'input with shape %r does not have enough dimensions for all core '
          'dimensions %r %s' % (shape, core_dims, error_context))
  else:
    if len(shape) != num_core_dims:
      raise ValueError(
          'output shape %r does not match core dimensions %r %s'
          % (shape, core_dims, error_context))

  core_shape = shape[-num_core_dims:] if core_dims else ()
  for dim, size in zip(core_dims, core_shape):
    if dim not in dim_sizes:
      dim_sizes[dim] = size
    elif size != dim_sizes[dim]:
      raise ValueError(
          'inconsistent size for core dimension %r: %r vs %r %s'
          % (dim, size, dim_sizes[dim], error_context))


def _parse_input_dimensions(
    args: Tuple[NDArray, ...],
    input_core_dims: List[CoreDims],
    error_context: str,
) -> Tuple[Tuple[int, ...], Dict[str, int]]:
  """Parse broadcast and core dimensions for vectorize with a signature.

  Args:
    args: tuple of input arguments to examine.
    input_core_dims: list of core dimensions corresponding to each input.

  Returns:
    broadcast_shape: common shape to broadcast all non-core dimensions to.
    dim_sizes: common sizes for named core dimensions.
  """
  shapes = []
  dim_sizes = {}
  for arg, core_dims in zip(args, input_core_dims):
    _update_dim_sizes(dim_sizes, arg.shape, core_dims, error_context,
                      is_input=True)
    ndim = arg.ndim - len(core_dims)
    shapes.append(arg.shape[:ndim])
  broadcast_shape = lax.broadcast_shapes(*shapes)
  return broadcast_shape, dim_sizes


def _broadcast_with_core_dims(
    args: Tuple[NDArray, ...],
    input_core_dims: List[CoreDims],
    error_context: str,
) -> Tuple[Tuple[NDArray, ...], Dict[str, int]]:
  if len(args) != len(input_core_dims):
    raise TypeError(
        'wrong number of positional arguments: expected %r, got %r %s'
        % (len(input_core_dims), len(args), error_context))

  broadcast_shape, dim_sizes = _parse_input_dimensions(
      args, input_core_dims, error_context)
  input_shapes = [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
                  for core_dims in input_core_dims]
  args = tuple(map(np.broadcast_to, args, input_shapes))
  return args, dim_sizes


def _check_output_dims(
    func: Callable,
    dim_sizes: Dict[str, int],
    expected_output_core_dims: List[CoreDims],
    error_context: str):
  """Check that output core dimensions match the signature."""
  def wrapped(*args):
    out = func(*args)
    out_shapes = map(np.shape, out if isinstance(out, tuple) else [out])

    if expected_output_core_dims is None:
      output_core_dims = [()] * len(out_shapes)
    else:
      output_core_dims = expected_output_core_dims
      if len(output_core_dims) > 1 and not isinstance(out, tuple):
        raise TypeError(
            "output must be a tuple when multiple outputs are expected, "
            "got: {!r}\n{}".format(out, error_context))
      if len(out_shapes) != len(output_core_dims):
        raise TypeError(
            'wrong number of output arguments: expected %r, got %r %s'
            % (len(output_core_dims), len(out_shapes), error_context))

    sizes = dict(dim_sizes)
    for shape, core_dims in zip(out_shapes, output_core_dims):
      _update_dim_sizes(sizes, shape, core_dims, error_context,
                        is_input=False)

    return out
  return wrapped


def _apply_excluded(func, excluded, args):
  if not excluded:
    return func, args

  if max(excluded) >= len(args):
    raise ValueError("excluded={!r} is invalid for {!r} argument(s)"
                     .format(excluded, len(args)))

  dynamic_args = [arg for i, arg in enumerate(args) if i not in excluded]
  static_args = [(i, args[i]) for i in sorted(excluded)]

  def new_func(*args):
    args = list(args)
    for i, arg in static_args:
      args.insert(i, arg)
    return func(*args)

  return new_func, dynamic_args


@_wraps(onp.vectorize, lax_description=textwrap.dedent("""
    JAX's implementation of vectorize should be considerably more efficient
    than NumPy's, because it uses a batching transformation rather than an
    explicit "for" loop.

    Note that JAX only supports the optional ``excluded`` (integer only) and
    ``signature`` arguments, both of which must be specified with keywords.
    """))
def vectorize(pyfunc, *, excluded=frozenset(), signature=None):

  if any(not isinstance(exclude, int) for exclude in excluded):
    raise TypeError("jax.numpy.vectorize can only exclude integer arguments, "
                    "but excluded={!r}".format(excluded))
  if excluded and min(excluded) < 0:
    raise ValueError("excluded={!r} contains negative numbers".format(excluded))

  @functools.wraps(pyfunc)
  def vectorized(*args):
    error_context = ("on vectorized function with excluded={!r} and "
                     "signature={!r}".format(excluded, signature))
    excluded_func, args = _apply_excluded(pyfunc, excluded, args)
    args = tuple(map(np.asarray, args))

    if signature is not None:
      input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
      broadcast_args, dim_sizes = _broadcast_with_core_dims(
          args, input_core_dims, error_context)
    else:
      input_core_dims = [()] * len(args)
      broadcast_args = np.broadcast_arrays(*args)
      output_core_dims = None
      dim_sizes = {}

    checked_func = _check_output_dims(
        excluded_func, dim_sizes, output_core_dims, error_context)

    num_batch_dims = broadcast_args[0].ndim - len(input_core_dims[0])
    vectorized_func = checked_func
    for _ in range(num_batch_dims):
      vectorized_func = api.vmap(vectorized_func)
    return vectorized_func(*broadcast_args)
  return vectorized
