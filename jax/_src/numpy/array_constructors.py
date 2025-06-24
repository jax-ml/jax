# Copyright 2025 The JAX Authors.
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

import importlib
from typing import Any

import numpy as np

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import tree_util
from jax._src import xla_bridge
from jax._src.lax import lax
from jax._src.lib import xla_client as xc
from jax._src.numpy import util
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.sharding import Sharding


export = util.set_module('jax.numpy')

for pkg_name in ['jax_cuda12_plugin', 'jax.jaxlib.cuda']:
  try:
    cuda_plugin_extension = importlib.import_module(
        f'{pkg_name}.cuda_plugin_extension'
    )
  except ImportError:
    cuda_plugin_extension = None  # type: ignore
  else:
    break


def _supports_buffer_protocol(obj):
  try:
    view = memoryview(obj)
  except TypeError:
    return False
  else:
    return True


def _make_string_array(
    object: np.ndarray,
    dtype: DTypeLike | None = None,
    ndmin: int = 0,
    device: xc.Device | Sharding | None = None,
) -> Array:
  if not isinstance(object, np.ndarray):
    raise TypeError(
        "Currently, string arrays can only be made from NumPy"
        f" arrays. Got:  {type(object)}."
    )
  if dtype is not None and (
      dtypes.is_string_dtype(object.dtype) != dtypes.is_string_dtype(dtype)
  ):
    raise TypeError(
        f"Cannot make an array with dtype {dtype} from an object with dtype"
        f" {object.dtype}."
    )
  if ndmin > object.ndim:
    raise TypeError(
        f"ndmin {ndmin} cannot be greater than object's ndims"
        f" {object.ndim} for string arrays."
    )

  # Just do a device_put since XLA does not support string as a data type.
  return api.device_put(x=object, device=device)


@export
def array(object: Any, dtype: DTypeLike | None = None, copy: bool = True,
          order: str | None = "K", ndmin: int = 0,
          *, device: xc.Device | Sharding | None = None) -> Array:
  """Convert an object to a JAX array.

  JAX implementation of :func:`numpy.array`.

  Args:
    object: an object that is convertible to an array. This includes JAX
      arrays, NumPy arrays, Python scalars, Python collections like lists
      and tuples, objects with an ``__array__`` method, and objects
      supporting the Python buffer protocol.
    dtype: optionally specify the dtype of the output array. If not
      specified it will be inferred from the input.
    copy: specify whether to force a copy of the input. Default: True.
    order: not implemented in JAX
    ndmin: integer specifying the minimum number of dimensions in the
      output array.
    device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    A JAX array constructed from the input.

  See also:
    - :func:`jax.numpy.asarray`: like `array`, but by default only copies
      when necessary.
    - :func:`jax.numpy.from_dlpack`: construct a JAX array from an object
      that implements the dlpack interface.
    - :func:`jax.numpy.frombuffer`: construct a JAX array from an object
      that implements the buffer interface.

  Examples:
    Constructing JAX arrays from Python scalars:

    >>> jnp.array(True)
    Array(True, dtype=bool)
    >>> jnp.array(42)
    Array(42, dtype=int32, weak_type=True)
    >>> jnp.array(3.5)
    Array(3.5, dtype=float32, weak_type=True)
    >>> jnp.array(1 + 1j)
    Array(1.+1.j, dtype=complex64, weak_type=True)

    Constructing JAX arrays from Python collections:

    >>> jnp.array([1, 2, 3])  # list of ints -> 1D array
    Array([1, 2, 3], dtype=int32)
    >>> jnp.array([(1, 2, 3), (4, 5, 6)])  # list of tuples of ints -> 2D array
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)
    >>> jnp.array(range(5))
    Array([0, 1, 2, 3, 4], dtype=int32)

    Constructing JAX arrays from NumPy arrays:

    >>> jnp.array(np.linspace(0, 2, 5))
    Array([0. , 0.5, 1. , 1.5, 2. ], dtype=float32)

    Constructing a JAX array via the Python buffer interface, using Python's
    built-in :mod:`array` module.

    >>> from array import array
    >>> pybuffer = array('i', [2, 3, 5, 7])
    >>> jnp.array(pybuffer)
    Array([2, 3, 5, 7], dtype=int32)
  """
  if order is not None and order != "K":
    raise NotImplementedError("Only implemented for order='K'")

  # check if the given dtype is compatible with JAX
  dtypes.check_user_dtype_supported(dtype, "array")

  # Here we make a judgment call: we only return a weakly-typed array when the
  # input object itself is weakly typed. That ensures asarray(x) is a no-op
  # whenever x is weak, but avoids introducing weak types with something like
  # array([1, 2, 3])
  weak_type = dtype is None and dtypes.is_weakly_typed(object)
  if device is None and isinstance(object, core.Tracer):
    sharding = object.aval.sharding
    sharding = None if sharding.mesh.empty else sharding
  else:
    sharding = util.canonicalize_device_to_sharding(device)

  # Use device_put to avoid a copy for ndarray inputs.
  if (not copy and isinstance(object, np.ndarray) and
      (dtype is None or dtype == object.dtype) and (ndmin <= object.ndim) and
      device is None):
    # Keep the output uncommitted.
    return api.device_put(object)

  # String arrays need separate handling because XLA does not support string
  # as a data type.
  if dtypes.is_string_dtype(dtype) or (
      hasattr(object, "dtype") and dtypes.is_string_dtype(object.dtype)
  ):
    return _make_string_array(
        object=object, dtype=dtype, ndmin=ndmin, device=device
    )

  # For Python scalar literals, call coerce_to_array to catch any overflow
  # errors. We don't use dtypes.is_python_scalar because we don't want this
  # triggering for traced values. We do this here because it matters whether or
  # not dtype is None. We don't assign the result because we want the raw object
  # to be used for type inference below.
  if isinstance(object, (bool, int, float, complex)):
    _ = dtypes.coerce_to_array(object, dtype)
  elif not isinstance(object, Array):
    # Check if object supports any of the data exchange protocols
    # (except dlpack, see data-apis/array-api#301). If it does,
    # consume the object as jax array and continue (but not return) so
    # that other array() arguments get processed against the input
    # object.
    #
    # Notice that data exchange protocols define dtype in the
    # corresponding data structures and it may not be available as
    # object.dtype. So, we'll resolve the protocols here before
    # evaluating object.dtype.
    if hasattr(object, '__jax_array__'):
      object = object.__jax_array__()
    elif hasattr(object, '__cuda_array_interface__'):
      cai = object.__cuda_array_interface__
      backend = xla_bridge.get_backend("cuda")
      if cuda_plugin_extension is None:
        device_id = None
      else:
        device_id = cuda_plugin_extension.get_device_ordinal(cai["data"][0])
      object = xc._xla.cuda_array_interface_to_buffer(
          cai=cai, gpu_backend=backend, device_id=device_id)

  leaves, treedef = tree_util.tree_flatten(object, is_leaf=lambda x: x is None)
  if any(leaf is None for leaf in leaves):
    raise ValueError("None is not a valid value for jnp.array")
  leaves = [
      leaf
      if (leaf_jax_array := getattr(leaf, "__jax_array__", None)) is None
      else leaf_jax_array()
      for leaf in leaves
  ]
  if dtype is None:
    # Use lattice_result_type rather than result_type to avoid canonicalization.
    # Otherwise, weakly-typed inputs would have their dtypes canonicalized.
    try:
      dtype = dtypes._lattice_result_type(*leaves)[0] if leaves else dtypes.float_
    except TypeError:
      # This happens if, e.g. one of the entries is a memoryview object.
      # This is rare, so we only handle it if the normal path fails.
      leaves = [_convert_to_array_if_dtype_fails(leaf) for leaf in leaves]
      dtype = dtypes._lattice_result_type(*leaves)[0]

  if not weak_type:
    dtype = dtypes.canonicalize_dtype(dtype, allow_extended_dtype=True)  # type: ignore[assignment]

  object = treedef.unflatten(leaves)
  out: ArrayLike
  if all(not isinstance(leaf, Array) for leaf in leaves):
    # TODO(jakevdp): falling back to numpy here fails to overflow for lists
    # containing large integers; see discussion in
    # https://github.com/jax-ml/jax/pull/6047. More correct would be to call
    # coerce_to_array on each leaf, but this may have performance implications.
    out = np.asarray(object, dtype=dtype)
  elif isinstance(object, Array):
    assert object.aval is not None
    out = lax._array_copy(object) if copy else object
  elif isinstance(object, (list, tuple)):
    if object:
      arrs = (array(elt, dtype=dtype, copy=False) for elt in object)
      arrays_out = [lax.expand_dims(arr, [0]) for arr in arrs]
      # lax.concatenate can be slow to compile for wide concatenations, so form a
      # tree of concatenations as a workaround especially for op-by-op mode.
      # (https://github.com/jax-ml/jax/issues/653).
      k = 16
      while len(arrays_out) > k:
        arrays_out = [lax.concatenate(arrays_out[i:i+k], 0)
                      for i in range(0, len(arrays_out), k)]
      out = lax.concatenate(arrays_out, 0)
    else:
      out = np.array([], dtype=dtype)
  elif _supports_buffer_protocol(object):
    object = memoryview(object)
    # TODO(jakevdp): update this once we support NumPy 2.0 semantics for the copy arg.
    out = np.array(object) if copy else np.asarray(object)
  else:
    raise TypeError(f"Unexpected input type for array: {type(object)}")
  out_array: Array = lax._convert_element_type(
      out, dtype, weak_type=weak_type, sharding=sharding)
  if ndmin > np.ndim(out_array):
    out_array = lax.expand_dims(out_array, range(ndmin - np.ndim(out_array)))
  return out_array


def _get_platform(
    device_or_sharding: xc.Device | Sharding | None | str) -> str:
  """Get device_or_sharding platform or look up config.default_device.value."""
  if isinstance(device_or_sharding, xc.Device):
    return device_or_sharding.platform
  elif isinstance(device_or_sharding, Sharding):
    return list(device_or_sharding.device_set)[0].platform
  elif isinstance(device_or_sharding, str):
    return device_or_sharding
  elif device_or_sharding is None:
    if config.default_device.value is None:
      return xla_bridge.default_backend()
    else:
      return _get_platform(config.default_device.value)
  else:
    raise ValueError(f"`{device_or_sharding = }` was passed to"
                     "`canonicalize_or_get_default_platform`, only xc.Device,"
                     " Sharding, None or str values are supported.")


def _convert_to_array_if_dtype_fails(x: ArrayLike) -> ArrayLike:
  try:
    dtypes.dtype(x)
  except TypeError:
    return np.asarray(x)
  else:
    return x


@export
def asarray(a: Any, dtype: DTypeLike | None = None, order: str | None = None,
            *, copy: bool | None = None,
            device: xc.Device | Sharding | None = None) -> Array:
  """Convert an object to a JAX array.

  JAX implementation of :func:`numpy.asarray`.

  Args:
    a: an object that is convertible to an array. This includes JAX
      arrays, NumPy arrays, Python scalars, Python collections like lists
      and tuples, objects with an ``__array__`` method, and objects
      supporting the Python buffer protocol.
    dtype: optionally specify the dtype of the output array. If not
      specified it will be inferred from the input.
    order: not implemented in JAX
    copy: optional boolean specifying the copy mode. If True, then always
      return a copy. If False, then error if a copy is necessary. Default is
      None, which will only copy when necessary.
    device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    A JAX array constructed from the input.

  See also:
    - :func:`jax.numpy.array`: like `asarray`, but defaults to `copy=True`.
    - :func:`jax.numpy.from_dlpack`: construct a JAX array from an object
      that implements the dlpack interface.
    - :func:`jax.numpy.frombuffer`: construct a JAX array from an object
      that implements the buffer interface.

  Examples:
    Constructing JAX arrays from Python scalars:

    >>> jnp.asarray(True)
    Array(True, dtype=bool)
    >>> jnp.asarray(42)
    Array(42, dtype=int32, weak_type=True)
    >>> jnp.asarray(3.5)
    Array(3.5, dtype=float32, weak_type=True)
    >>> jnp.asarray(1 + 1j)
    Array(1.+1.j, dtype=complex64, weak_type=True)

    Constructing JAX arrays from Python collections:

    >>> jnp.asarray([1, 2, 3])  # list of ints -> 1D array
    Array([1, 2, 3], dtype=int32)
    >>> jnp.asarray([(1, 2, 3), (4, 5, 6)])  # list of tuples of ints -> 2D array
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)
    >>> jnp.asarray(range(5))
    Array([0, 1, 2, 3, 4], dtype=int32)

    Constructing JAX arrays from NumPy arrays:

    >>> jnp.asarray(np.linspace(0, 2, 5))
    Array([0. , 0.5, 1. , 1.5, 2. ], dtype=float32)

    Constructing a JAX array via the Python buffer interface, using Python's
    built-in :mod:`array` module.

    >>> from array import array
    >>> pybuffer = array('i', [2, 3, 5, 7])
    >>> jnp.asarray(pybuffer)
    Array([2, 3, 5, 7], dtype=int32)
  """
  # For copy=False, the array API specifies that we raise a ValueError if the input supports
  # the buffer protocol but a copy is required. Since array() supports the buffer protocol
  # via numpy, this is only the case when the default device is not 'cpu'
  if (copy is False and not isinstance(a, Array)
      and _get_platform(device) != "cpu"
      and _supports_buffer_protocol(a)):
    raise ValueError(f"jnp.asarray: cannot convert object of type {type(a)} to JAX Array "
                     f"on platform={_get_platform(device)} with "
                     "copy=False. Consider using copy=None or copy=True instead.")
  dtypes.check_user_dtype_supported(dtype, "asarray")
  if dtype is not None:
    dtype = dtypes.canonicalize_dtype(dtype, allow_extended_dtype=True)  # type: ignore[assignment]
  return array(a, dtype=dtype, copy=bool(copy), order=order, device=device)
