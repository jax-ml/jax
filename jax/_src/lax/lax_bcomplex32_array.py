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

"""BComplex32Array: Python wrapper for bcomplex32 extended dtype arrays.

bcomplex32 is physically stored as bfloat16 array with trailing dim of size 2.
This wrapper provides the logical complex-number view over the physical data.
"""

from __future__ import annotations

import math


from jax._src import basearray
from jax._src import core
from jax._src import dtypes
from jax._src import tree_util
from jax._src import sharding_impls
from jax._src.interpreters import pxla
from jax._src.interpreters import mlir


class BComplex32Array(basearray.Array):
  """Array wrapper for bcomplex32 (complex<bfloat16>) dtype.

  Stores data physically as bfloat16 array with trailing dimension of size 2.
  The trailing dim holds [real_part, imaginary_part].
  """

  __slots__ = ["_aval", "_base_array"]
  __hash__ = None
  __array_priority__ = 100

  def __init__(self, aval, base_array):
    assert isinstance(aval, core.ShapedArray), aval
    assert aval.dtype is dtypes.bcomplex32_edtype, aval.dtype
    assert not isinstance(base_array, core.Tracer), type(base_array)
    self._aval = aval
    self._base_array = base_array

  # --- Logical view properties ---

  @property
  def aval(self):
    return self._aval

  @property
  def dtype(self):
    return self._aval.dtype

  @property
  def shape(self):
    return self._aval.shape

  @property
  def ndim(self):
    return len(self._aval.shape)

  @property
  def size(self):
    return math.prod(self._aval.shape) if self._aval.shape else 1

  @property
  def itemsize(self):
    return 4  # 2 * bfloat16

  @property
  def nbytes(self):
    return self.size * self.itemsize

  def __len__(self):
    if self.ndim == 0:
      raise TypeError("len() of unsized object")
    return self.shape[0]

  def __repr__(self):
    return f"BComplex32Array(shape={self.shape}, dtype={self.dtype})"

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")
    for i in range(self.shape[0]):
      yield self[i]

  def reshape(self, *args, order="C"):
    """Reshape the bcomplex32 array via lax.reshape."""
    from jax._src.lax import lax as lax_mod

    # Normalize shape argument: reshape((2,2)) or reshape(2,2) or reshape(-1)
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
      shape = tuple(args[0])
    else:
      shape = tuple(int(a) for a in args)
    # Handle -1 in shape: compute the inferred dimension
    if -1 in shape:
      known = 1
      unknown_idx = None
      for i, s in enumerate(shape):
        if s == -1:
          if unknown_idx is not None:
            raise ValueError("can only specify one unknown dimension")
          unknown_idx = i
        else:
          known *= s
      if unknown_idx is not None:
        total = self.size
        inferred = total // known
        shape = shape[:unknown_idx] + (inferred,) + shape[unknown_idx + 1 :]
    return lax_mod.reshape(self, shape)

  # --- Forward to self._base_array ---

  @property
  def _device(self):
    return self._base_array._device

  @property
  def _committed(self):
    return self._base_array._committed

  @property
  def committed(self):
    return self._base_array.committed

  @property
  def device(self):
    return self._base_array.device

  @property
  def is_fully_addressable(self):
    return self._base_array.is_fully_addressable

  @property
  def is_fully_replicated(self):
    return self._base_array.is_fully_replicated

  def devices(self):
    return self._base_array.devices()

  def delete(self):
    self._base_array.delete()

  def is_deleted(self):
    return self._base_array.is_deleted()

  def on_device_size_in_bytes(self):
    return self._base_array.on_device_size_in_bytes()

  def block_until_ready(self):
    _ = self._base_array.block_until_ready()
    return self

  def copy_to_host_async(self):
    self._base_array.copy_to_host_async()

  def copy(self):
    return BComplex32Array(self._aval, self._base_array.copy())

  # --- Defer to extended dtype rules for sharding ---

  @property
  def sharding(self):
    phys_sharding = self._base_array.sharding
    return sharding_impls.logical_sharding(
      self.shape, self.dtype, phys_sharding
    )

  def addressable_data(self, index: int):
    return BComplex32Array(self._aval, self._base_array.addressable_data(index))

  @property
  def addressable_shards(self):
    return [
      type(s)(
        device=s._device,
        sharding=s._sharding,
        global_shape=s._global_shape,
        data=BComplex32Array(self._aval, s._data),
      )
      for s in self._base_array.addressable_shards
    ]

  @property
  def global_shards(self):
    return [
      type(s)(
        device=s._device,
        sharding=s._sharding,
        global_shape=s._global_shape,
        data=BComplex32Array(self._aval, s._data),
      )
      for s in self._base_array.global_shards
    ]


# ============================================================
# Registrations
# ============================================================


# Pytree: flatten/unflatten
def _bcomplex32_flatten(x):
  return (x._base_array,), x._aval


def _bcomplex32_unflatten(aval, children):
  return BComplex32Array(aval, children[0])


tree_util.dispatch_registry.register_node(
  BComplex32Array, _bcomplex32_flatten, _bcomplex32_unflatten
)

# Type -> abstract value mapping
core.pytype_aval_mappings[BComplex32Array] = lambda x: x.aval

# Don't convert to numpy
dtypes.register_canonicalize_value_handler(BComplex32Array, None)


# Shard argument handler (unwrap to physical for dispatch)
def _bcomplex32_shard_arg_handler(xs, shardings, layouts, copy_semantics):
  arrs = [x._base_array for x in xs]
  phys_shardings = [
    sharding_impls.physical_sharding(x.aval, sharding)
    for x, sharding in zip(xs, shardings)
  ]
  # TODO(yashkatariya): `layouts` should be converted to physical layouts.
  return pxla.shard_args(phys_shardings, layouts, copy_semantics, arrs)


pxla.shard_arg_handlers[BComplex32Array] = _bcomplex32_shard_arg_handler


# MLIR constant handler (unwrap to physical)
def _bcomplex32_constant_handler(val, aval):
  arr = val._base_array
  return mlir.get_constant_handler(type(arr))(arr, aval)


mlir.register_constant_handler(BComplex32Array, _bcomplex32_constant_handler)

# Operator forwarding: set array operators so that __add__, __mul__, etc.
# dispatch through JAX's lax_numpy functions which handle ExtendedDType.
from jax._src.numpy.array_methods import (
  _array_operators,
  _set_array_base_attributes,
)

_set_array_base_attributes(
  BComplex32Array,
  include=[
    *(f"__{op}__" for op in _array_operators),
  ],
)

# AD utility: register adder for gradient accumulation
from jax._src import ad_util
from jax._src.lax import lax as _lax_module


def _bcomplex32_add(x, y):
  return _lax_module.add(x, y)


ad_util.raw_jaxval_adders[BComplex32Array] = _bcomplex32_add
