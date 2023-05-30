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

import math
import operator as op
import numpy as np
import functools
from typing import (Sequence, Tuple, Callable, Optional, List, cast, Set,
                    TYPE_CHECKING)

from jax._src import abstract_arrays
from jax._src import api
from jax._src import api_util
from jax._src import basearray
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import profiler
from jax._src import xla_bridge
from jax._src.config import config
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.interpreters import xla
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    SingleDeviceSharding, XLACompatibleSharding, PmapSharding,
    device_replica_id_map, hashed_index)
from jax._src.typing import ArrayLike
from jax._src.util import use_cpp_class, use_cpp_method

Shape = Tuple[int, ...]
Device = xc.Device
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
               data: Optional[ArrayImpl] = None):
    self._device = device
    self._sharding = sharding
    self._global_shape = global_shape
    self._data = data

  def __repr__(self):
    try:
      return (f'Shard(device={repr(self.device)}, index={self.index}, '
              f'replica_id={self.replica_id}, data={self.data})')
    except ValueError:
      return f'Shard(device={repr(self.device)}, data={self.data})'

  @functools.cached_property
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

  @functools.cached_property
  def replica_id(self) -> int:
    return device_replica_id_map(self._sharding, self._global_shape)[self.device]

  @property
  def device(self):
    return self._device

  @property
  def data(self):
    return self._data


def _reconstruct_array(fun, args, arr_state, aval_state):
  """Method to reconstruct a device array from a serialized state."""
  np_value = fun(*args)
  np_value.__setstate__(arr_state)
  jnp_value = api.device_put(np_value)
  jnp_value.aval = jnp_value.aval.update(**aval_state)
  return jnp_value


@functools.lru_cache(maxsize=4096)
def _cached_index_calc(s, shape):
  map_ = s.addressable_devices_indices_map(shape)
  seen_h_indices = set()
  m = {}
  for d, index in map_.items():
    h_index = hashed_index(index)
    if h_index not in seen_h_indices:
      seen_h_indices.add(h_index)
      m[d] = index
  return m


def _create_copy_plan(arrays, s: Sharding, shape: Shape):
  di_map = _cached_index_calc(s, shape)
  copy_plan = []
  for a in arrays:
    ind = di_map.get(a.device(), None)
    if ind is not None:
      copy_plan.append((ind, a))
  return copy_plan


@functools.lru_cache(maxsize=4096)
def _process_has_full_value_in_mcjax(s, shape):
  # Return False for single host as a fast path.
  if xla_bridge.process_count() == 1:
    return False

  num_unique_indices = len(
      set(hashed_index(v) for v in s.devices_indices_map(shape).values()))
  num_addressable_unique_indices = len(
      set(hashed_index(v) for v in s.addressable_devices_indices_map(shape).values()))
  return num_unique_indices == num_addressable_unique_indices


class ArrayImpl(basearray.Array):
  # TODO(yashkatariya): Add __slots__ here.

  aval: core.ShapedArray
  _sharding: Sharding
  _arrays: List[ArrayImpl]
  _committed: bool
  _skip_checks: bool
  _npy_value: Optional[np.ndarray]

  @use_cpp_method()
  def __init__(self, aval: core.ShapedArray, sharding: Sharding,
               arrays: Sequence[ArrayImpl],
               committed: bool, _skip_checks: bool = False):
    # NOTE: the actual implementation of the constructor is moved to C++.

    self.aval = aval
    self._sharding = sharding
    self._arrays = [a._arrays[0] for a in arrays]
    self._committed = committed
    self._npy_value = None

    # Don't rearrange if skip_checks is enabled because this assumes that the
    # input buffers are already arranged properly. This usually happens when
    # Array's are created as output of a JAX transformation
    # (like pjit, xmap, etc).
    if not _skip_checks or config.jax_enable_checks:
      self._check_and_rearrange()

  def _check_and_rearrange(self):
    for db in self._arrays:
      if db.dtype != self.dtype:
        raise ValueError(
            "Input buffers to `Array` must have matching dtypes. "
            f"Got {db.dtype}, expected {self.dtype} for buffer: {db}")

    device_id_to_buffer = {db.device().id: db for db in self._arrays}

    addressable_dev = self.sharding.addressable_devices
    if len(self._arrays) != len(addressable_dev):
      raise ValueError(
          f"Expected {len(addressable_dev)} per-device arrays "
          "(this is how many devices are addressable by the sharding), but "
          f"got {len(self._arrays)}")

    array_device_ids = set(device_id_to_buffer.keys())
    addressable_device_ids = {d.id for d in addressable_dev}
    # Calculate a symmetric difference because the device ids between sharding
    # and _arrays should match.
    diff = array_device_ids ^ addressable_device_ids
    if diff:
      dev_in_sharding_not_in_arrays = addressable_device_ids - array_device_ids
      dev_in_arrays_not_in_sharding = array_device_ids - addressable_device_ids
      err_msg = (
          "Addressable devices and per-device arrays devices do not match.")
      if dev_in_sharding_not_in_arrays:
        err_msg += (f" Sharding contains devices {dev_in_sharding_not_in_arrays} "
                    "that are not present in per-device arrays.")
      if dev_in_arrays_not_in_sharding:
        err_msg += (f" Per-device arrays contain devices {dev_in_arrays_not_in_sharding} "
                    "that are not present in the sharding.")
      raise ValueError(err_msg)

    ss = self.sharding.shard_shape(self.shape)
    for db in self._arrays:
      if db.shape != ss:
        raise ValueError(
            f"Expected shard shape {ss} doesn't match the single device array "
            f"shape {db.shape}. Shape of Array is "
            f"{self.aval.str_short()} with sharding {self.sharding}")

    # Rearrange arrays based on the device assignment.
    if isinstance(self.sharding, XLACompatibleSharding):
      addressable_da = self.sharding._addressable_device_assignment
      self._arrays = [device_id_to_buffer[device.id] for device in addressable_da]

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
    return math.prod(self.shape)

  @property
  def sharding(self):
    return self._sharding

  @property
  def weak_type(self):
    return self.aval.weak_type

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

    if isinstance(self.sharding, PmapSharding):
      if not isinstance(idx, tuple):
        cidx = (idx,) + (slice(None),) * (len(self.shape) - 1)
      else:
        cidx = idx + (slice(None),) * (len(self.shape) - len(idx))
      if self._npy_value is None:
        indices = tuple(self.sharding.devices_indices_map(self.shape).values())
        try:
          arr_idx = indices.index(cidx)
        except ValueError:
          arr_idx = None
        if arr_idx is not None:
          a = self._arrays[arr_idx]
          return ArrayImpl(
              a.aval, SingleDeviceSharding(a.device()), [a], committed=False,
              _skip_checks=True)
      return lax_numpy._rewriting_take(self, idx)
    else:
      return lax_numpy._rewriting_take(self, idx)

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    else:
      assert self.is_fully_replicated or self.is_fully_addressable
      if dispatch.is_single_device_sharding(self.sharding) or self.is_fully_replicated:
        return (sl for chunk in self._chunk_iter(100) for sl in chunk._unstack())  # type: ignore
      elif isinstance(self.sharding, PmapSharding):
        return (self[i] for i in range(self.shape[0]))  # type: ignore
      else:
        # TODO(yashkatariya): Don't bounce to host and use `_chunk_iter` path
        # here after uneven partitioning support is added.
        return (api.device_put(self._value[i]) for i in range(self.shape[0]))

  @property
  def is_fully_replicated(self) -> bool:
    return self.sharding.is_fully_replicated

  def __repr__(self):
    prefix = 'Array('
    if self.aval is not None and self.aval.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.dtype.name})'

    if self.is_fully_addressable or self.is_fully_replicated:
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

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self.sharding.is_fully_addressable

  def __array__(self, dtype=None, context=None):
    return np.asarray(self._value, dtype=dtype)

  def __dlpack__(self):
    from jax._src.dlpack import to_dlpack  # pylint: disable=g-import-not-at-top
    return to_dlpack(self)

  def __reduce__(self):
    fun, args, arr_state = self._value.__reduce__()  # type: ignore
    aval_state = {'weak_type': self.aval.weak_type,
                  'named_shape': self.aval.named_shape}
    return (_reconstruct_array, (fun, args, arr_state, aval_state))

  @use_cpp_method()
  def unsafe_buffer_pointer(self):
    if len(self._arrays) != 1:
      raise ValueError("unsafe_buffer_pointer() is supported only for unsharded"
                       " arrays.")
    return self._arrays[0].unsafe_buffer_pointer()

  @property
  @use_cpp_method()
  def __cuda_array_interface__(self):
    if len(self._arrays) != 1:
      raise ValueError("__cuda_array_interface__() is supported only for "
                       "unsharded arrays.")
    return self._arrays[0].__cuda_array_interface__  # pytype: disable=attribute-error  # bind-properties

  @use_cpp_method()
  def on_device_size_in_bytes(self):
    """Returns the total global on-device size of the array in bytes."""
    arr = self._arrays[0]
    per_shard_size = arr.on_device_size_in_bytes()  # type: ignore
    return per_shard_size * len(self.sharding.device_set)

  # TODO(yashkatariya): Remove this method when everyone is using devices().
  def device(self) -> Device:
    self._check_if_deleted()
    device_set = self.sharding.device_set
    if len(device_set) == 1:
      single_device, = device_set
      return single_device
    raise ValueError('Length of devices is greater than 1. '
                     'Please use `.devices()`.')

  def devices(self) -> Set[Device]:
    self._check_if_deleted()
    return self.sharding.device_set

  # TODO(https://github.com/google/jax/issues/12380): Remove this when DA is
  # deleted.
  @property
  def device_buffer(self) -> ArrayImpl:
    self._check_if_deleted()
    if len(self._arrays) == 1:
      return self._arrays[0]
    raise ValueError('Length of buffers is greater than 1. Please use '
                     '`.device_buffers` instead.')

  # TODO(https://github.com/google/jax/issues/12380): Remove this when SDA is
  # deleted.
  @property
  def device_buffers(self) -> Sequence[ArrayImpl]:
    self._check_if_deleted()
    return self._arrays

  def addressable_data(self, index: int) -> ArrayImpl:
    self._check_if_deleted()
    if self.is_fully_replicated and xla_extension_version >= 148:
      return self._fully_replicated_shard()
    return self._arrays[index]

  @functools.cached_property
  def addressable_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    out = []
    for a in self._arrays:
      out.append(Shard(a.device(), self.sharding, self.shape, a))
    return out

  @property
  def global_shards(self) -> Sequence[Shard]:
    """Returns list of all `Shard`s of the Array across all devices.

    The result includes shards that are not addressable by the current process.
    If a `Shard` is not addressable, then its `data` will be `None`.
    """
    self._check_if_deleted()
    if self.is_fully_addressable:  # pylint: disable=using-constant-test
      return self.addressable_shards

    out = []
    device_id_to_buffer = {a.device().id: a for a in self._arrays}
    for global_d in self.sharding.device_set:
      if device_id_to_buffer.get(global_d.id, None) is not None:
        array = device_id_to_buffer[global_d.id]
      else:
        array = None
      out.append(Shard(global_d, self.sharding, self.shape, array))
    return out

  @use_cpp_method()
  def delete(self):
    if self._arrays is None:
      return
    for buf in self._arrays:
      buf.delete()
    self._arrays = None
    self._npy_value = None

  @use_cpp_method()
  def is_deleted(self):
    if self._arrays is None:
      return True
    # This path is taken when a view of `Array` is created and the original
    # Array is deleted. In that case, the buffers the view represents also get
    # deleted.
    return any(buf.is_deleted() for buf in self._arrays)

  def _check_if_deleted(self):
    if self.is_deleted():
      raise RuntimeError("Array has been deleted.")

  @use_cpp_method()
  def block_until_ready(self):
    self._check_if_deleted()
    for db in self._arrays:
      db.block_until_ready()
    return self

  @use_cpp_method()
  def _single_device_array_to_np_array(self):
    return np.asarray(self._arrays[0])

  @use_cpp_method()
  def _copy_single_device_array_to_host_async(self):
    self._arrays[0].copy_to_host_async()

  @profiler.annotate_function
  def copy_to_host_async(self):
    self._check_if_deleted()
    if self._npy_value is None:
      if self.is_fully_replicated:
        self._copy_single_device_array_to_host_async()
        return
      copy_plan = _create_copy_plan(self._arrays, self.sharding, self.shape)
      for _, arr in copy_plan:
        arr._copy_single_device_array_to_host_async()

  @property
  @functools.partial(profiler.annotate_function, name="np.asarray(jax.Array)")
  def _value(self) -> np.ndarray:
    self._check_if_deleted()

    if self._npy_value is None:
      if self.is_fully_replicated:
        self._npy_value = self._single_device_array_to_np_array()  # type: ignore
        self._npy_value.flags.writeable = False
        return cast(np.ndarray, self._npy_value)

      # TODO(yashkatariya): Merge `_process_has_full_value_in_mcjax` with
      # is_fully_addressable.
      if (not self.is_fully_addressable and
          not _process_has_full_value_in_mcjax(self.sharding, self.shape)):
        raise RuntimeError("Fetching value for `jax.Array` that spans "
                           "non-addressable devices is not possible. You can use "
                           "`jax.experimental.multihost_utils.process_allgather` "
                           "for this use case.")

      copy_plan = _create_copy_plan(self._arrays, self.sharding, self.shape)
      for _, arr in copy_plan:
        arr._copy_single_device_array_to_host_async()

      npy_value = np.empty(self.shape, self.dtype)
      for ind, arr in copy_plan:
        npy_value[ind] = arr._single_device_array_to_np_array()
      self._npy_value = npy_value  # type: ignore
      self._npy_value.flags.writeable = False
    # https://docs.python.org/3/library/typing.html#typing.cast
    return cast(np.ndarray, self._npy_value)


# TODO(b/273265390): ideally we would write this as a decorator on the ArrayImpl
# class, however this triggers a pytype bug. Workaround: apply the decorator
# after the fact.
if not TYPE_CHECKING:
  ArrayImpl = use_cpp_class(xc.ArrayImpl)(ArrayImpl)


# explicitly set to be unhashable. Same as what device_array.py does.
setattr(ArrayImpl, "__hash__", None)
setattr(ArrayImpl, "__array_priority__", 100)

def make_array_from_callback(
    shape: Shape, sharding: Sharding,
    data_callback: Callable[[Optional[Index]], ArrayLike]) -> ArrayImpl:
  """Returns a ``jax.Array`` via data fetched from ``data_callback``.

  ``data_callback`` is used to fetch the data for each addressable shard of the
  returned ``jax.Array``.

  Args:
    shape : Shape of the ``jax.Array``.
    sharding: A ``Sharding`` instance which describes how the ``jax.Array`` is
      laid out across devices.
    data_callback : Callback that takes indices into the global array value as
      input and returns the corresponding data of the global array value.
      The data can be returned as any array-like object, e.g. a ``numpy.ndarray``.

  Returns:
    A ``jax.Array`` via data fetched from ``data_callback``.

  Example:

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> input_shape = (8, 8)
    >>> global_input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    >>> global_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> inp_sharding = jax.sharding.NamedSharding(global_mesh, P('x', 'y'))
    ...
    >>> def cb(index):
    ...  return global_input_data[index]
    ...
    >>> arr = jax.make_array_from_callback(input_shape, inp_sharding, cb)
    >>> arr.addressable_data(0).shape
    (4, 2)
  """
  device_to_index_map = sharding.devices_indices_map(shape)
  # Use addressable_devices here instead of `_addressable_device_assignment`
  # because `_addressable_device_assignment` is only available on
  # `XLACompatibleSharding` and this function is supposed to work for every
  # `Sharding`.
  arrays = [
      api.device_put(data_callback(device_to_index_map[device]), device)
      for device in sharding.addressable_devices
  ]
  aval = core.ShapedArray(shape, arrays[0].dtype, weak_type=False)
  if dtypes.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.make_sharded_array(aval, sharding, arrays, committed=True)
  return ArrayImpl(aval, sharding, arrays, committed=True)


def make_array_from_single_device_arrays(
    shape: Shape, sharding: Sharding, arrays: Sequence[basearray.Array]
) -> ArrayImpl:
  r"""Returns a ``jax.Array`` from a sequence of ``jax.Array``\s on a single device.

  ``jax.Array`` on a single device is analogous to a ``DeviceArray``. You can use
  this function if you have already ``jax.device_put`` the value on a single
  device and want to create a global Array. The smaller ``jax.Array``\s should be
  addressable and belong to the current process.

  Args:
    shape : Shape of the ``jax.Array``.
    sharding: A ``Sharding`` instance which describes how the ``jax.Array`` is
      laid out across devices.
    arrays: Sequence of ``jax.Array``\s that are on a single device.

  Returns:
    A ``jax.Array`` from a sequence of ``jax.Array``\s on a single device.

  Example:

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> global_shape = (8, 8)
    >>> global_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> sharding = jax.sharding.NamedSharding(global_mesh, P('x', 'y'))
    >>> inp_data = np.arange(math.prod(global_shape)).reshape(global_shape)
    ...
    >>> arrays = [
    ...     jax.device_put(inp_data[index], d)
    ...     for d, index in sharding.addressable_devices_indices_map(global_shape).items()]
    ...
    >>> arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
    >>> arr.addressable_data(0).shape
    (4, 2)

    In multi-process case, if the input is process local and data parallel
    i.e. each process receives a different part of the data, then you can use
    `make_array_from_single_device_arrays` to create a global jax.Array

    >>> local_shape = (8, 2)
    >>> global_shape = (jax.process_count() * local_shape[0], ) + local_shape[1:]
    >>> local_array = np.arange(math.prod(local_shape)).reshape(local_shape)
    >>> arrays = jax.device_put(
    ...   np.split(local_array, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)
    >>> sharding = jax.sharding.NamedSharding(global_mesh, P(('x', 'y'), ))
    >>> arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
    >>> arr.addressable_data(0).shape
    (1, 2)
  """
  # All input arrays should be committed. Checking it is expensive on
  # single-controller systems.
  aval = core.ShapedArray(shape, arrays[0].dtype, weak_type=False)
  if dtypes.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.make_sharded_array(aval, sharding, arrays, committed=True)
  # TODO(phawkins): ideally the cast() could be checked. Revisit this after
  # removing DeviceArray.
  return ArrayImpl(aval, sharding, cast(Sequence[ArrayImpl], arrays),
                   committed=True)


core.pytype_aval_mappings[ArrayImpl] = abstract_arrays.canonical_concrete_aval
xla.pytype_aval_mappings[ArrayImpl] = op.attrgetter('aval')
xla.canonicalize_dtype_handlers[ArrayImpl] = pxla.identity
api_util._shaped_abstractify_handlers[ArrayImpl] = op.attrgetter('aval')
# TODO(jakevdp) replace this with true inheritance at the C++ level.
basearray.Array.register(ArrayImpl)


def _array_mlir_constant_handler(val, canonicalize_types=True):
  return mlir.ir_constants(val._value,
                           canonicalize_types=canonicalize_types)
mlir.register_constant_handler(ArrayImpl, _array_mlir_constant_handler)


def _array_shard_arg(x, devices, indices, sharding):
  x._check_if_deleted()

  x_indices = x.sharding.addressable_devices_indices_map(x.shape).values()
  if not x.is_fully_addressable:
    if tuple(x_indices) == tuple(indices):
      return x
    else:
      raise NotImplementedError(
          "Cannot reshard an input that is not fully addressable")
  else:
    if tuple(x_indices) == tuple(indices):
      return xc.copy_array_to_devices_with_sharding(
          x, list(devices), sharding)
    # Resharding starts here:
    if dispatch.is_single_device_sharding(x.sharding):
      return pxla.shard_device_array(x, devices, indices, sharding)
    else:
      return pxla.shard_sharded_device_array_slow_path(
          x, devices, indices, sharding)


pxla.shard_arg_handlers[ArrayImpl] = _array_shard_arg


def _array_global_result_handler(global_aval, out_sharding, committed,
                                 is_out_sharding_from_xla):
  if global_aval.dtype == dtypes.float0:
    return lambda _: np.zeros(global_aval.shape, dtypes.float0)  # type: ignore
  if dtypes.is_opaque_dtype(global_aval.dtype):
    return global_aval.dtype._rules.global_sharded_result_handler(
        global_aval, out_sharding, committed, is_out_sharding_from_xla)
  return xc.array_result_handler(
      global_aval, out_sharding, committed=committed, _skip_checks=True
  )
pxla.global_result_handlers[core.ShapedArray] = _array_global_result_handler
pxla.global_result_handlers[core.ConcreteArray] = _array_global_result_handler
pxla.global_result_handlers[core.AbstractToken] = lambda *_: lambda *_: core.token


# Only used for Arrays that come out of pmap.
def _array_local_result_handler(aval, sharding, indices):
  if aval.dtype == dtypes.float0:
    return lambda _: np.zeros(aval.shape, dtypes.float0)  # type: ignore
  if dtypes.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.local_sharded_result_handler(
        aval, sharding, indices)
  return xc.array_result_handler(
      aval, sharding, committed=True, _skip_checks=True
  )
pxla.local_result_handlers[core.ShapedArray] = _array_local_result_handler
pxla.local_result_handlers[core.ConcreteArray] = _array_local_result_handler
