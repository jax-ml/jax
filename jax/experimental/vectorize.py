# Copyright 2018 Google LLC
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
"""Extending JAX's vmap to work like NumPy's gufuncs.

By `Stephan Hoyer <https://github.com/shoyer>`_

What is a gufunc?
=================

`Generalized universal functions
<https://docs.scipy.org/doc/numpy-1.15.0/reference/c-api.generalized-ufuncs.html>`_
("gufuncs") are one of my favorite abstractions from NumPy. They generalize
NumPy's `broadcasting rules
<https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html>`_ to
handle non-scalar operations. When a gufuncs is applied to arrays, there are: 

* "core dimensions" over which an operation is defined.  
* "broadcast dimensions" over which operations can be automatically vectorized.

A string `signature <https://docs.scipy.org/doc/numpy-1.15.0/reference/c-api.generalized-ufuncs.html#details-of-signature>`_
associated with each gufunc controls how this happens by indicating how core
dimensions are mapped between inputs and outputs. The syntax is easiest to
understand by looking at a few examples:

* Addition: `(),()->()`
* 1D inner product: `(i),(i)->()`
* 1D sum: `(i)->()`
* Matrix multiplcation: `(m,n),(n,k)->(m,k)`

Why write gufuncs?
=====================

From a user perspective, gufuncs are nice because they're guaranteed to
vectorize in a consistent and general fashion. For example, by default gufuncs
use the last dimensions of arrays as core dimensions, but you can control that
explicitly with the ``axis`` or ``axes`` arguments.

From a developer perspective, gufuncs are nice because they simplify your work:
you only need to think about the core logic of your function, not how it
handles arbitrary dimensional input. You can just write that down in a simple,
declarative way.

JAX makes it easy to write high-level performant code
=====================================================

Unfortunately, writing NumPy gufuncs today is somewhat non-trivial. Your
options today are:

1. Write the inner loops yourself in C.
2. `np.vectorize <https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html>`_ creates something kind of like a gufunc, but it's painfully slow: the outer loop is performed in Python.
3. `numba.guvectorize <https://numba.pydata.org/numba-doc/dev/user/vectorize.html>`_ can work well, if you don't need further code transformations like automatic differentiation.

JAX's ``vmap`` contains all the core functionality we need to write functions that work like gufuncs. JAX gufuncs play nicely with other transformations like ``grad`` and ``jit``.

A simple example
================

Consider a simple example from data preprocessing, centering an array.

Here's how we might write a vectorized version using NumPy::

  def center(array, axis=-1):
    # array can have any number of dimensions
    bias = np.mean(array, axis=axis)
    debiased = array - np.expand_dims(bias, axis)
    return bias, debiased

And here's how we could write a vectorized version using JAX gufuncs::

  @vectorize('(n)->(),(n)')
  def center(array):
    # array is always a 1D vector
    bias = np.mean(array)
    debiased = array - bias
    return bias, debiased

See the difference?

* Instead of needing to think about broadcasting while writing the entire function, we can write the function assuming the input is always a vector.
* We get the ``axis`` argument automatically, without needing to write it ourselves.
* As a bonus, the decorator makes the function self-documenting: a reader immediately knows that it handles higher dimensional input and output correctly.

"""

from jax import grad, jit, vmap
import jax.numpy as jnp
import numpy as np
import re

# See http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)


def _parse_gufunc_signature(signature):
    """Parse string signatures for a generalized universal function.

    Args:
      signature : string
	  Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
	  for ``np.matmul``.

    Returns:
      Tuple of input and output core dimensions parsed from the signature, each
      of the form List[Tuple[str, ...]].
    """
    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid gufunc signature: {}'.format(signature))
    return tuple([tuple(re.findall(_DIMENSION_NAME, arg))
                  for arg in re.findall(_ARGUMENT, arg_list)]
                 for arg_list in signature.split('->'))



def _update_dim_sizes(dim_sizes, arg, core_dims):
    """Incrementally check and update core dimension sizes for a single argument.

    Args:
      dim_sizes : Dict[str, int]
	  Sizes of existing core dimensions. Will be updated in-place.
      arg : ndarray
	  Argument to examine.
      core_dims : Tuple[str, ...]
	  Core dimensions for this argument.
    """
    if not core_dims:
        return

    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            '%d-dimensional argument does not have enough '
            'dimensions for all core dimensions %r'
            % (arg.ndim, core_dims))

    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError(
                    'inconsistent size for core dimension %r: %r vs %r'
                    % (dim, size, dim_sizes[dim]))
        else:
            dim_sizes[dim] = size


def _parse_input_dimensions(args, input_core_dims):
    """Parse broadcast and core dimensions for vectorize with a signature.

    Args:
      args : Tuple[ndarray, ...]
	  Tuple of input arguments to examine.
      input_core_dims : List[Tuple[str, ...]]
	  List of core dimensions corresponding to each input.

    Returns:
      broadcast_shape : Tuple[int, ...]
	  Common shape to broadcast all non-core dimensions to.
      dim_sizes : Dict[str, int]
	  Common sizes for named core dimensions.
    """
    broadcast_args = []
    dim_sizes = {}
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, arg.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib.stride_tricks._broadcast_shape(*broadcast_args)
    return broadcast_shape, dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    """Helper for calculating broadcast shapes with core dimensions."""
    return [broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
            for core_dims in list_of_core_dims]

  
# adapted from np.vectorize (again authored by shoyer@)
def broadcast_with_core_dims(args, input_core_dims, output_core_dims):
  if len(args) != len(input_core_dims):
    raise TypeError('wrong number of positional arguments: '
                    'expected %r, got %r'
                    % (len(input_core_dims), len(args)))

  broadcast_shape, dim_sizes = _parse_input_dimensions(
      args, input_core_dims)
  input_shapes = _calculate_shapes(broadcast_shape, dim_sizes,
                                   input_core_dims)
  args = [jnp.broadcast_to(arg, shape)
          for arg, shape in zip(args, input_shapes)]
  return args

def verify_axis_is_supported(input_core_dims, output_core_dims):
  all_core_dims = set()
  for input_or_output_core_dims in [input_core_dims, output_core_dims]:
    for core_dims in input_or_output_core_dims:
      all_core_dims.update(core_dims)
  if len(core_dims) > 1:
    raise ValueError('only one gufuncs with one core dim support axis')


def reorder_inputs(args, axis, input_core_dims):
  return tuple(jnp.moveaxis(arg, axis, -1) if core_dims else arg
               for arg, core_dims in zip(args, input_core_dims))


def reorder_outputs(result, axis, output_core_dims):
  if not isinstance(result, tuple):
    result = (result,)
  result = tuple(jnp.moveaxis(res, -1, axis) if core_dims else res
                 for res, core_dims in zip(result, output_core_dims))
  if len(result) == 1:
    (result,) = result
  return result

import functools


def vectorize(signature):
  """Vectorize a function using JAX.

  Turns an abritrary function into a numpy style "gufunc". Once
  you specify the behavior of the core axis, the rest will be 
  broadcast naturally.

  Args:
    signature: an einsum style signature that defines how the core dimensions are mapped between inputs and outputs.

  Returns:
	The vectorized 'gufunc' that will automatically broadcast
	while maintaining the specified core logic, the returned
	function also has a new ``axis`` parameter for specifying
	which axis should be treated as the core one.
  """
  input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
  
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      axis = kwargs.get('axis')  # for python2 compat.

      if axis is not None:
        verify_axis_is_supported(input_core_dims, output_core_dims)
        args = reorder_inputs(args, axis, input_core_dims)

      broadcast_args = broadcast_with_core_dims(
          args, input_core_dims, output_core_dims)
      num_batch_dims = len(broadcast_args[0].shape) - len(input_core_dims[0])

      vectorized_func = func
      for _ in range(num_batch_dims):
        vectorized_func = vmap(vectorized_func)
      result = vectorized_func(*broadcast_args)

      if axis is not None:
        result = reorder_outputs(result, axis, output_core_dims)

      return result
    return wrapper
  return decorator
