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
    """
    Parse string signatures for a generalized universal function.

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
        for ``np.matmul``.

    Returns
    -------
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
    """
    Incrementally check and update core dimension sizes for a single argument.

    Arguments
    ---------
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
    """
    Parse broadcast and core dimensions for vectorize with a signature.

    Arguments
    ---------
    args : Tuple[ndarray, ...]
        Tuple of input arguments to examine.
    input_core_dims : List[Tuple[str, ...]]
        List of core dimensions corresponding to each input.

    Returns
    -------
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
  """Vectorize a function using JAX."""
  input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
  
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, axis=None):

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
