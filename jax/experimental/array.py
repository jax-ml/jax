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

from __future__ import annotations

import operator as op
import numpy as np
from typing import Sequence, Tuple, Callable, Union, Optional, cast, List

from jax import core
from jax._src import abstract_arrays
from jax._src import ad_util
from jax._src import api_util
from jax._src import basearray
from jax._src import dispatch
from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax._src.config import config
from jax._src.util import prod, safe_zip
from jax._src.lib import xla_client as xc
from jax._src.api import device_put
from jax._src.typing import ArrayLike
from jax.interpreters import pxla, xla, mlir
from jax.experimental.sharding import (
    Sharding, SingleDeviceSharding, XLACompatibleSharding, PmapSharding,
    device_replica_id_map)

Shape = Tuple[int, ...]
Device = xc.Device
DeviceArray = xc.Buffer
Index = Tuple[slice, ...]


class Shard:
  """A single data shard of an Array.

  Attributes:
    device : Which device this shard resides on.
    index : The index into the global array of this shard.
    replica_id : Integer id indicating which replica of the global array this
      shard is part of. Always 0 for fully sharded data
      (i.e. when there’s only 1 replica).
    data : The data of this shard. None if ``device`` is non-local.
  """

  def __init__(self, device: Device, sharding: Sharding, global_shape: Shape,
               data: Optional[Array] = None):
    self.device = device
    self._sharding = sharding
    self._global_shape = global_shape
    self.data = data

  def __repr__(self):
    try:
      return (f'Shard(device={repr(self.device)}, index={self.index}, '
              f'replica_id={self.replica_id}, data={self.data})')
    except ValueError:
      return f'Shard(device={repr(self.device)}, data={self.data})'

  @property
  def index(self) -> Index:
    try:
      device_indices_map_fn = self._sharding.devices_indices_map
    except AttributeError:
      raise ValueError('Cannot calculate indices from sharding: '
                       f'{self._sharding}. Please create a device to index '
                       'mapping for your sharding.') from None
    index = device_indices_map_fn(self._global_shape)[self.device]
    assert index is not None
    return index

  @property
  def replica_id(self) -> int:
    return device_replica_id_map(self._sharding, self._global_shape)[self.device]


def _reconstruct_array(fun, args, arr_state, aval_state):
  """Method to reconstruct a device array from a serialized state."""
  np_value = fun(*args)
  np_value.__setstate__(arr_state)
  jnp_value = device_put(np_value)
  jnp_value.aval = jnp_value.aval.update(**aval_state)
  return jnp_value


def _single_device_array_from_buf(buf, committed):
  db = pxla._set_aval(buf)
  return Array(db.aval, SingleDeviceSharding(db.device()), [db],
               committed=committed, _skip_checks=True)


@pxla.use_cpp_class(xc.Array if xc._version >= 92 else None)
class Array(basearray.Array):
  """Experimental unified Array type.

  This Python implementation will eventually be replaced by a C++ implementation.
  """
  # TODO(yashkatariya): Add __slots__ here.

  aval: core.ShapedArray
  _sharding: Sharding
  _arrays: List[DeviceArray]
  _committed: bool
  _skip_checks: bool
  _npy_value: Optional[np.ndarray]

  @pxla.use_cpp_method
  def __init__(self, aval: core.ShapedArray, sharding: Sharding,
               arrays: Union[Sequence[DeviceArray], Sequence[Array]],
               committed: bool, _skip_checks: bool = False):
    # NOTE: the actual implementation of the constructor is moved to C++.

    self.aval = aval
    self._sharding = sharding
    # Extract DeviceArrays from arrays with `SingleDeviceSharding` to keep the
    # code handling `self._arrays` simpler.
    # TODO(yashkatariya): This will be slower as it will happen during
    # `__init__` on single controller environment. Make it lazy.
    self._arrays = [a if isinstance(a, DeviceArray) else a._arrays[0] for a in arrays]
    # See https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
    # for what committed means.
    self._committed = committed
    self._npy_value = None

    if not _skip_checks or config.jax_enable_checks:
      self._check()

    # Don't rearrange if skip_checks is enabled because this assumes that the
    # input buffers are already arranged properly. This usually happens when
    # Array's are created as output of a JAX transformation
    # (like pjit, xmap, etc).
    if not _skip_checks:
      self._rearrange()

  def _check(self):
    ss = self.sharding.shard_shape(self.shape)
    for db in self._arrays:
      if db.shape != ss:
        raise ValueError(
            f"Expected shard shape {ss} doesn't match the buffer "
            f"shape {db.shape} for buffer: {db}")

    for db in self._arrays:
      if db.dtype != self.dtype:
        raise ValueError(
            "Input buffers to `Array` must have matching dtypes. "
            f"Got {db.dtype}, expected {self.dtype} for buffer: {db}")

  def _rearrange(self):
    # Rearrange arrays based on the device assignment.
    # TODO(yashkatariya): Add a similar check for shardings that are not
    # XLACompatibleSharding. But leave the rearragement to XLACompatibleSharding
    # only.
    if isinstance(self.sharding, XLACompatibleSharding):
      addressable_da = self.sharding._addressable_device_assignment
      if len(self._arrays) != len(addressable_da):
        raise ValueError(
            f"Expected {len(addressable_da)} per-device arrays "
            "(this is how many devices are addressable by the sharding), but "
            f"got {len(self._arrays)}")
      device_to_buffer = {db.device().id: db for db in self._arrays}
      try:
        self._arrays = [device_to_buffer[device.id]
                        for device in addressable_da]
      except KeyError as e:
        array_device_ids = set(a.device().id for a in self._arrays)
        addressable_device_ids = set(d.id for d in addressable_da)
        diff = set(array_device_ids) - set(addressable_device_ids)
        raise ValueError(
            f"Some per-device arrays are placed on devices {diff}, which are "
            f"not used in the specified sharding {self.sharding}") from e

  @property
  def shape(self) -> Shape:
    return self.aval.shape

  @property
  def dtype(self):
    return self.aval.dtype

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def size(self):
    return prod(self.shape)

  @property
  def sharding(self):
    return self._sharding

  def __str__(self):
    return str(self._value)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error

  def __bool__(self):
    return bool(self._value)

  def __nonzero__(self):
    return bool(self._value)

  def __float__(self):
    return self._value.__float__()

  def __int__(self):
    return self._value.__int__()

  def __complex__(self):
    return self._value.__complex__()

  def __hex__(self):
    assert self.ndim == 0, 'hex only works on scalar values'
    return hex(self._value)  # type: ignore

  def __oct__(self):
    assert self.ndim == 0, 'oct only works on scalar values'
    return oct(self._value)  # type: ignore

  def __index__(self):
    return op.index(self._value)

  def tobytes(self, order="C"):
    return self._value.tobytes(order)

  def tolist(self):
    return self._value.tolist()

  def __format__(self, format_spec):
    # Simulates behavior of https://github.com/numpy/numpy/pull/9883
    if self.ndim == 0:
      return format(self._value[()], format_spec)
    else:
      return format(self._value, format_spec)

  def __getitem__(self, idx):
    from jax._src.numpy import lax_numpy
    self._check_if_deleted()

    if dispatch.is_single_device_sharding(self.sharding):
      return lax_numpy._rewriting_take(self, idx)
    # TODO(yashkatariya): Make it work for other Shardings too wherever its
    # possible to not do data movement.
    elif isinstance(self.sharding, PmapSharding):
      if not isinstance(idx, tuple):
        cidx = (idx,) + (slice(None),) * (len(self.shape) - 1)
      else:
        cidx = idx + (slice(None),) * (len(self.shape) - len(idx))
      if self._npy_value is None:
        indices = tuple(self.sharding.devices_indices_map(self.shape).values())
        try:
          buf_idx = indices.index(cidx)
        except ValueError:
          buf_idx = None
        if buf_idx is not None:
          buf = self._arrays[buf_idx]
          aval = core.ShapedArray(buf.shape, self.dtype)
          return Array(aval, SingleDeviceSharding(buf.device()), [buf],
                       committed=False, _skip_checks=True)
      return lax_numpy._rewriting_take(self, idx)
    else:
      # TODO(yashkatariya): Don't bounce to host and use `_rewriting_take` or
      # the fast path (see PmapSharding branch above) after b/245667823 is
      # fixed.
      return self._value[idx]

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    else:
      assert self.is_fully_replicated() or self.is_fully_addressable()
      if dispatch.is_single_device_sharding(self.sharding):
        return (sl for chunk in self._chunk_iter(100) for sl in chunk._unstack())  # type: ignore
      elif isinstance(self.sharding, PmapSharding):
        return (self[i] for i in range(self.shape[0]))  # type: ignore
      else:
        # TODO(yashkatariya): Don't bounce to host and use `_chunk_iter` path
        # here after b/245667823 is fixed.
        return (self._value[i] for i in range(self.shape[0]))

  def item(self):
    if dtypes.issubdtype(self.dtype, np.complexfloating):
      return complex(self)
    elif dtypes.issubdtype(self.dtype, np.floating):
      return float(self)
    elif dtypes.issubdtype(self.dtype, np.integer):
      return int(self)
    elif dtypes.issubdtype(self.dtype, np.bool_):
      return bool(self)
    else:
      raise TypeError(self.dtype)

  def is_fully_replicated(self) -> bool:
    return self.shape == self._arrays[0].shape

  def __repr__(self):
    prefix = '{}('.format(self.__class__.__name__.lstrip('_'))
    if self.aval is not None and self.aval.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.dtype.name})'

    if self.is_fully_addressable() or self.is_fully_replicated():
      line_width = np.get_printoptions()["linewidth"]
      s = np.array2string(self._value, prefix=prefix, suffix=',',
                          separator=', ', max_line_width=line_width)
      last_line_len = len(s) - s.rfind('\n') + 1
      sep = ' '
      if last_line_len + len(dtype_str) + 1 > line_width:
        sep = ' ' * len(prefix)
      return f"{prefix}{s},{sep}{dtype_str}"
    else:
      return f"{prefix}{self.shape}, {dtype_str}"

  def is_fully_addressable(self) -> bool:
    return self.sharding.is_fully_addressable()

  def __array__(self, dtype=None, context=None):
    return np.asarray(self._value, dtype=dtype)

  def __dlpack__(self):
    from jax.dlpack import to_dlpack  # pylint: disable=g-import-not-at-top
    return to_dlpack(self)

  def __reduce__(self):
    fun, args, arr_state = self._value.__reduce__()  # type: ignore
    aval_state = {'weak_type': self.aval.weak_type,
                  'named_shape': self.aval.named_shape}
    return (_reconstruct_array, (fun, args, arr_state, aval_state))

  def unsafe_buffer_pointer(self):
    assert len(self._arrays) == 1
    return self._arrays[0].unsafe_buffer_pointer()

  @property
  def __cuda_array_interface__(self):
    assert len(self._arrays) == 1
    return self._arrays[0].__cuda_array_interface__  # pytype: disable=attribute-error  # bind-properties

  # TODO(yashkatariya): Remove this method when everyone is using devices().
  def device(self) -> Device:
    self._check_if_deleted()
    device_set = self.sharding.device_set
    if len(device_set) == 1:
      single_device, = device_set
      return single_device
    raise ValueError('Length of devices is greater than 1. '
                     'Please use `.devices()`.')

  def devices(self) -> List[Device]:
    self._check_if_deleted()
    return list(self.sharding.device_set)

  # TODO(https://github.com/google/jax/issues/12380): Remove this when DA is
  # deleted.
  @property
  def device_buffer(self) -> DeviceArray:
    self._check_if_deleted()
    if len(self._arrays) == 1:
      return _single_device_array_from_buf(self._arrays[0], self._committed)
    raise ValueError('Length of buffers is greater than 1. Please use '
                     '`.device_buffers` instead.')

  # TODO(https://github.com/google/jax/issues/12380): Remove this when SDA is
  # deleted.
  @property
  def device_buffers(self) -> Sequence[DeviceArray]:
    self._check_if_deleted()
    return [_single_device_array_from_buf(a, self._committed)
            for a in self._arrays]

  def addressable_data(self, index: int) -> Array:
    self._check_if_deleted()
    return _single_device_array_from_buf(self._arrays[index], self._committed)

  @pxla.maybe_cached_property
  def addressable_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    out = []
    for db in self._arrays:
      # Wrap the device arrays in `Array` until C++ returns an Array instead
      # of a DA.
      array = _single_device_array_from_buf(db, self._committed)
      out.append(Shard(db.device(), self.sharding, self.shape, array))
    return out

  def delete(self):
    if self._arrays is None:
      return
    for buf in self._arrays:
      buf.delete()
    self._arrays = None
    self._npy_value = None

  def is_deleted(self):
    if self._arrays is None:
      return True
    # This path is taken when a view of `Array` is created and the original
    # Array is deleted. In that case, the buffers the view represents also get
    # deleted.
    return any(buf.is_deleted() for buf in self._arrays)

  def _check_if_deleted(self):
    if self._arrays is None:
      raise RuntimeError("Array has been deleted.")

  @pxla.use_cpp_method
  def block_until_ready(self):
    self._check_if_deleted()
    for db in self._arrays:
      db.block_until_ready()
    return self

  def copy_to_host_async(self):
    self._check_if_deleted()
    if self._npy_value is None:
      try:
        self.addressable_shards[0].replica_id
        replica_id_exists = True
      except ValueError:
        replica_id_exists = False

      for s in self.addressable_shards:
        if not replica_id_exists or s.replica_id == 0:
          s.data._arrays[0].copy_to_host_async()  # pytype: disable=attribute-error

  @property
  def _value(self) -> np.ndarray:
    self._check_if_deleted()

    if self._npy_value is None:
      if self.is_fully_replicated():
        self._npy_value = np.asarray(self._arrays[0])  # type: ignore
        self._npy_value.flags.writeable = False
        return cast(np.ndarray, self._npy_value)

      if not self.is_fully_addressable():
        raise RuntimeError("Fetching value for `jax.Array` that spans "
                           "non-addressable devices is not possible. You can use "
                           "`jax.experimental.multihost_utils.process_allgather` "
                           "for this use case.")

      self.copy_to_host_async()
      npy_value = np.empty(self.shape, self.dtype)

      try:
        self.addressable_shards[0].replica_id
        replica_id_exists = True
      except ValueError:
        replica_id_exists = False

      for s in self.addressable_shards:
        if not replica_id_exists or s.replica_id == 0:
          npy_value[s.index] = np.asarray(s.data._arrays[0])  # type: ignore  # [union-attr]
      self._npy_value = npy_value  # type: ignore
      self._npy_value.flags.writeable = False
    # https://docs.python.org/3/library/typing.html#typing.cast
    return cast(np.ndarray, self._npy_value)

# explicitly set to be unhashable. Same as what device_array.py does.
setattr(Array, "__hash__", None)
setattr(Array, "__array_priority__", 100)

def make_array_from_callback(shape: Shape, sharding: Sharding,
                             data_callback: Callable[[Optional[Index]], ArrayLike]) -> Array:
  device_to_index_map = sharding.devices_indices_map(shape)
  # Use addressable_devices here instead of `_addressable_device_assignment`
  # because `_addressable_device_assignment` is only available on
  # `XLACompatibleSharding` and this function is supposed to work for every
  # `Sharding`.
  arrays = [
      device_put(data_callback(device_to_index_map[device]), device)
      for device in sharding.addressable_devices
  ]
  aval = core.ShapedArray(shape, arrays[0].dtype, weak_type=False)
  return Array(aval, sharding, arrays, committed=True)


core.pytype_aval_mappings[Array] = abstract_arrays.canonical_concrete_aval
xla.pytype_aval_mappings[Array] = op.attrgetter('aval')
xla.canonicalize_dtype_handlers[Array] = pxla.identity
api_util._shaped_abstractify_handlers[Array] = op.attrgetter('aval')
ad_util.jaxval_adders[Array] = lax_internal.add
ad_util.jaxval_zeros_likers[Array] = lax_internal.zeros_like_array
if xc._version >= 92:
  # TODO(jakevdp) replace this with true inheritance at the C++ level.
  basearray.Array.register(Array)


def _array_mlir_constant_handler(val, canonicalize_types=True):
  return mlir.ir_constants(val._value,
                           canonicalize_types=canonicalize_types)
mlir.register_constant_handler(Array, _array_mlir_constant_handler)


def _device_put_array(x, device: Optional[Device]):
  # TODO(yashkatariya): Remove this restriction and the round trip via host
  # once lowering to XLA goes through `lower_mesh_computation`.
  assert x.is_fully_addressable()
  if dispatch.is_single_device_sharding(x.sharding):
    x = dispatch._copy_device_array_to_device(pxla._set_aval(x._arrays[0]), device)
    return (x,)
  else:
    # Round trip via host if x is sharded. SDA also does a round trip via host.
    return dispatch._device_put_array(x._value, device)

dispatch.device_put_handlers[Array] = _device_put_array


def _array_pmap_shard_arg(x, devices, indices, mode):
  if dispatch.is_single_device_sharding(x.sharding):
    return pxla._shard_device_array(x, devices, indices, mode)

  # If the sharding of Array does not match pmap's sharding then take the slow
  # path which is similar to what SDA does. This slow path reroute only happens
  # for `pmap`.
  x_indices = tuple(x.sharding.addressable_devices_indices_map(x.shape).values())
  if indices == x_indices:
    return [buf if buf.device() == d else buf.copy_to_device(d)
            for buf, d in safe_zip(x._arrays, devices)]
  else:
    return pxla._shard_sharded_device_array_slow_path(x, devices, indices, mode)


def _array_rest_shard_arg(x, devices, indices, mode):
  if not x._committed:
    if dispatch.is_single_device_sharding(x.sharding):
      # This condition is to break the recursion that happens when only
      # `pxla._shard_device_array` is used since it has `_multi_slice` in the
      # implementation which is jitted. Eventually it calls back here and the
      # recursion happens.
      x_indices = tuple(x.sharding.addressable_devices_indices_map(x.shape).values())
      if x_indices == indices:
        return [buf if buf.device() == d else buf.copy_to_device(d)
                for buf, d in safe_zip(x._arrays, devices)]
      return pxla._shard_device_array(x, devices, indices, mode)
    else:
      raise NotImplementedError('Resharding uncommitted arrays sharded over '
                                'multiple devices is not supported.')
  # TODO(yashkatariya): Remove the special case here and don't move to another
  # device if its already committed. There is a TODO in dispatch.py already
  # for this.
  if dispatch.is_single_device_sharding(x.sharding):
    return [buf if buf.device() == d else buf.copy_to_device(d)
            for buf, d in safe_zip(x._arrays, devices)]
  # If PmapSharding exists, then do a round trip via host. This will happen
  # if the input Array containing PmapSharding takes the jit path
  # i.e. `apply_primitive` or `xla_callable_uncached`. `jit(pmap)` is the most
  # common case where this will happen.
  # TODO(yashkatariya): Remove the special case here and don't move to another
  # device if its already committed. There is a TODO in dispatch.py already
  # for this.
  elif isinstance(x.sharding, PmapSharding):
    return pxla.device_put(x._value, devices, replicate=True)
  else:
    return x._arrays


def _array_shard_arg(x, devices, indices, mode):
  if mode == pxla.InputsHandlerMode.pmap:
    return _array_pmap_shard_arg(x, devices, indices, mode)
  else:
    return _array_rest_shard_arg(x, devices, indices, mode)
pxla.shard_arg_handlers[Array] = _array_shard_arg


def _array_global_result_handler(global_aval, out_sharding, committed,
                                 is_out_sharding_from_xla):
  if global_aval.dtype == dtypes.float0:
    return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
  if core.is_opaque_dtype(global_aval.dtype):
    return global_aval.dtype._rules.global_sharded_result_handler(
        global_aval, out_sharding, committed, is_out_sharding_from_xla)
  return lambda bufs: Array(global_aval, out_sharding, bufs,
                            committed=committed, _skip_checks=True)
pxla.global_result_handlers[(core.ShapedArray, pxla.OutputType.Array)] = _array_global_result_handler
pxla.global_result_handlers[(core.ConcreteArray, pxla.OutputType.Array)] = _array_global_result_handler
pxla.global_result_handlers[(core.AbstractToken, pxla.OutputType.Array)] = lambda *_: lambda *_: core.token


# Only used for Arrays that come out of pmap.
def _array_local_result_handler(aval, sharding, indices):
  if core.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.local_sharded_result_handler(
        aval, sharding, indices)
  return lambda bufs: Array(aval, sharding, bufs, committed=True,
                            _skip_checks=True)
pxla.local_result_handlers[(core.ShapedArray, pxla.OutputType.Array)] = _array_local_result_handler
pxla.local_result_handlers[(core.ConcreteArray, pxla.OutputType.Array)] = _array_local_result_handler
