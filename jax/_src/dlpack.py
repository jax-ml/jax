# Copyright 2020 The JAX Authors.
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

from collections.abc import Callable
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jax._src import array
from jax._src import core
from jax._src import util
from jax._src import xla_bridge
from jax._src.api import device_put
from jax._src.lax.lax import _array_copy
from jax._src.lib import gpu_dlpack
from jax._src.lib import xla_client
from jax._src.lib import xla_extension_version
from jax._src.sharding import Sharding
from jax._src.typing import Array
from jax._src.typing import DLDeviceType
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla


DLPACK_VERSION = (0, 8)
MIN_DLPACK_VERSION = (0, 5)

# A set of dtypes that dlpack supports.
# Note: Make sure to use a "type", not a dtype instance, when looking up this set
# because their hashes are different.
# For example,
# hash(jnp.float32) != hash(jnp.dtype(jnp.float32))
# hash(jnp.float32) == hash(jnp.dtype(jnp.float32).type)
# TODO(phawkins): Migrate to using dtypes instead of the scalar type objects.
SUPPORTED_DTYPES = frozenset({
    jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16,
    jnp.uint32, jnp.uint64, jnp.float16, jnp.bfloat16, jnp.float32,
    jnp.float64, jnp.complex64, jnp.complex128})

if xla_extension_version >= 231:
  SUPPORTED_DTYPES = SUPPORTED_DTYPES | frozenset({jnp.bool_})


def _to_dlpack(x: Array, stream: int | Any | None,
               src_device: xla_client.Device | None = None,
               device: xla_client.Device | None = None,
               copy: bool | None = None):

  if src_device is None:
    src_device, = x.devices()
  if device and (src_device is None or device != src_device):
    if copy is not None and not copy:
      raise ValueError(
        f"Specified {device=} which requires a copy since the source device "
        f"is {repr(src_device)}, however copy=False. Set copy=True or "
        "copy=None to perform the requested operation."
      )
    else:
      arr = device_put(x, device)
  else:
    arr = _array_copy(x) if copy else x
  return xla_client._xla.buffer_to_dlpack_managed_tensor(
    arr.addressable_data(0), stream=stream
  )

def to_dlpack(x: Array, stream: int | Any | None = None,
              src_device: xla_client.Device | None = None,
              dl_device: tuple[DLDeviceType, int] | None = None,
              max_version: tuple[int, int] | None = None,
              copy : bool | None = None):
  """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``.

  Args:
    x: a :class:`~jax.Array`, on either CPU or GPU.
    stream: optional platform-dependent stream to wait on until the buffer is
      ready. This corresponds to the `stream` argument to ``__dlpack__``
      documented in https://dmlc.github.io/dlpack/latest/python_spec.html.
    src_device: either a CPU or GPU :class:`~jax.Device`.
    dl_device: a tuple of ``(dl_device_type, local_hardware_id)`` in DLPack
      format e.g. as produced by ``__dlpack_device__``.
    max_version: the maximum DLPack version that the consumer (i.e. caller of
      ``__dlpack__``) supports in the form of a 2-tuple of ``(major, minor)``.
      This function is not guaranteed to return a capsule of version
      ``max_version``.
    copy: a boolean indicating whether or not to copy the input. If
      ``copy=True`` then the function must always copy. When
      ``copy=False`` then the function must never copy, and must raise an error
      when a copy is deemed necessary. If ``copy=None`` then the function must
      avoid a copy if possible but may copy if needed.

  Returns:
    A DLPack PyCapsule object.

  Note:
    While JAX arrays are always immutable, ``DLPackManagedTensor`` buffers
    cannot be marked as immutable, and it is possible for processes external
    to JAX to mutate them in-place. If a DLPack buffer derived from a JAX array
    is mutated, it may lead to undefined behavior when using the associated JAX
    array. When JAX eventually supports ``DLManagedTensorVersioned``
    (DLPack 1.0), it will be possible to specify that a buffer is read-only.
  """
  if not isinstance(x, array.ArrayImpl):
    raise TypeError("Argument to to_dlpack must be a jax.Array, "
                    f"got {type(x)}")

  device = None
  dl_device_type, local_hardware_id = dl_device if dl_device else (None, None)
  if dl_device_type:
    try:
      dl_device_platform = {
          DLDeviceType.kDLCPU: "cpu",
          DLDeviceType.kDLCUDA: "cuda",
          DLDeviceType.kDLROCM: "rocm",
      }[dl_device_type]
      backend = xla_bridge.get_backend(dl_device_platform)
      device = backend.device_from_local_hardware_id(local_hardware_id)
    except TypeError:
      # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
      # recommends using BufferError.
      raise BufferError(
          "The device specification passed to to_dlpack contains an unsupported "
          f"device type (DLDeviceType: {dl_device_type})")

  # As new versions are adopted over time, we can maintain some legacy paths
  # for compatability mediated through the max_version parameter.
  # TODO(micky774): Deprecate default usage of DLPackManagedTensor when XLA
  # supports DLManagedTensorVersioned (DLPack version 1.0) and repurpose the
  # current _to_dlpack as a legacy path for (0,5) <= max_version < (1,0).
  if max_version is None or max_version >= DLPACK_VERSION:
    # Latest
    return _to_dlpack(
      x, stream=stream,
      src_device=src_device,
      device=device,
      copy=copy
    )
  elif max_version >= MIN_DLPACK_VERSION:
    # Oldest supported
    return _to_dlpack(
      x, stream=stream,
      src_device=src_device,
      device=device,
      copy=copy
    )
  else:
    raise BufferError(
      f"JAX does not support any version below {MIN_DLPACK_VERSION} but "
      f"version ({max_version}) was requested."
    )

def _place_array(_arr, device, dlpack_device, copy):
  if device and dlpack_device != device:
    if copy is not None and not copy:
      raise ValueError(
        f"Specified {device=} which requires a copy since the source device "
        f"is {repr(dlpack_device)}, however copy=False. Set copy=True or "
        "copy=None to perform the requested operation."
      )
    else:
      return device_put(_arr, device)
  if copy:
    return jnp.array(_arr, copy=True)
  return _arr

def _legacy_from_dlpack(dlpack, device: xla_client.Device | None = None,
                        copy: bool | None = None):
  preferred_platform = getattr(device, "platform", None)
  if device and preferred_platform == "gpu":
    preferred_platform = "cuda" if "cuda" in device.client.platform_version else "rocm"

  cpu_backend = xla_bridge.get_backend("cpu")
  gpu_backend = None

  if preferred_platform in {"cuda", "rocm"}:
    try:
      gpu_backend = xla_bridge.get_backend(preferred_platform)
    except RuntimeError:
      raise TypeError(
        f"A {str.upper(preferred_platform)} device was specified, however no "
        f"{str.upper(preferred_platform)} backend was found."
      )

  if preferred_platform is None:
    try:
      gpu_backend = xla_bridge.get_backend("cuda")
    except RuntimeError:
      pass
    # Try ROCm if CUDA backend not found
    if gpu_backend is None:
      try:
        gpu_backend = xla_bridge.get_backend("rocm")
      except RuntimeError:
        pass

  _arr = jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
      dlpack, cpu_backend, gpu_backend)) # type: ignore
  dlpack_device, = _arr.devices()
  return _place_array(_arr, device, dlpack_device, copy)

def _from_dlpack(external_array, device: xla_client.Device | None = None,
                 copy: bool | None = None):
  dl_device_type, device_id = external_array.__dlpack_device__()
  try:
    dl_device_platform = {
        DLDeviceType.kDLCPU: "cpu",
        DLDeviceType.kDLCUDA: "cuda",
        DLDeviceType.kDLROCM: "rocm",
    }[dl_device_type]
  except TypeError:
    # https://dmlc.github.io/dlpack/latest/python_spec.html recommends using
    # TypeError.
    raise TypeError(
        "Array passed to from_dlpack is on unsupported device type "
        f"(DLDeviceType: {dl_device_type}, array: {external_array}")

  backend = xla_bridge.get_backend(dl_device_platform)
  dlpack_device = backend.device_from_local_hardware_id(device_id)
  try:
    stream = dlpack_device.get_stream_for_external_ready_events()
  except xla_client.XlaRuntimeError as err:  # type: ignore
    if "UNIMPLEMENTED" in str(err):
      stream = None
    else:
      raise
  dlpack = external_array.__dlpack__(stream=stream)

  _arr = jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
      dlpack, dlpack_device, stream))
  return _place_array(_arr, device, dlpack_device, copy)

def from_dlpack(external_array,
                device: xla_client.Device | Sharding | None = None,
                copy: bool | None = None):
  """Returns a :class:`~jax.Array` representation of a DLPack tensor.

  The returned :class:`~jax.Array` shares memory with ``external_array`` if no
  device transfer or copy was requested.

  Args:
    external_array: An array object that has __dlpack__ and __dlpack_device__
      methods, or a DLPack tensor on either CPU or GPU (legacy API).

    device: The (optional) :py:class:`Device`, representing the device on which
      the returned array should be placed. If given, then the result is committed
      to the device. If unspecified, the resulting array will be unpacked onto the
      same device it originated from. Setting ``device`` to a device different from
      the source of ``external_array`` will require a copy, meaning ``copy`` must be
      set to either ``True`` or ``None``.

    copy: An (optional) boolean, controlling whether or not a copy is performed.
      If ``copy=True`` then a copy is always performed, even if unpacked onto the
      same device. If ``copy=False`` then the copy is never performed and will raise
      an error if necessary. When ``copy=None`` then a copy may be performed if
      needed for a device transfer.

  Returns:
    A jax.Array

  Note:
    While JAX arrays are always immutable, dlpack buffers cannot be marked as
    immutable, and it is possible for processes external to JAX to mutate them
    in-place. If a jax Array is constructed from a dlpack buffer and the buffer
    is later modified in-place, it may lead to undefined behavior when using
    the associated JAX array.
  """
  if isinstance(device, Sharding):
    device_set = device.device_set
    if len(device_set) > 1:
      raise ValueError(
        "from_dlpack can only unpack a dlpack tensor onto a singular device, but "
        f"a Sharding with {len(device_set)} devices was provided."
      )
    device, = device_set
  if hasattr(external_array, "__dlpack__"):
    return _from_dlpack(external_array, device, copy)

  # Legacy path
  return _legacy_from_dlpack(external_array, device, copy)


def callback(
    callback: Callable[..., Any],
    *args: Any,
    result_shape_dtypes: Any,
    vectorized: bool = False,
    mutable_results: bool = False,
    **kwargs: Any,
) -> Any:
  """Calls a pure Python callback with DLPack tensors.

  This is a variant of `jax.pure_callback` which uses DLPack tensors to
  keep the data on the device.

  Args:
    callback: function to execute. The callback is assumed to be a pure
      function (i.e. one without side-effects): if an impure function is passed,
      it may behave in unexpected ways, particularly under transformations.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype``
      attributes, whose structure matches the expected output of the callback
      function at runtime. :class:`jax.ShapeDtypeStruct` is often used to define
      leaf values.
    *args: arguments to be passed to the callback function
    vectorized: boolean specifying whether the callback function can operate in
      a vectorized manner.
    mutable_results: if True the callback is called with the ``results=``
      keyword argument, which is a pytree of mutable DLPack tensors to store the
      results.
    **kwargs: keyword arguments to be passed to the callback function.

  Returns:
    * If ``mutable_results`` is False: a pytree of :class:`jax.Array` objects
      whose structure matches that of ``result_shape_dtypes``.
    * If ``mutable_results`` is True: None.
  """

  if mutable_results:
    _, results_tree = jax.tree.flatten(result_shape_dtypes)

    def _flat_callback(*flat_args_results):
      args, kwargs = jax.tree.unflatten(
          in_tree, flat_args_results[: in_tree.num_leaves]
      )
      results = jax.tree.unflatten(
          results_tree, flat_args_results[in_tree.num_leaves :]
      )
      callback(*args, results, **kwargs)
      return None

  else:

    def _flat_callback(*flat_args):
      args, kwargs = jax.tree.unflatten(in_tree, flat_args)
      return jax.tree.leaves(callback(*args, **kwargs))

  flat_args, in_tree = jax.tree.flatten((args, kwargs))
  result_avals = jax.tree.map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes
  )
  flat_result_avals, out_tree = jax.tree.flatten(result_avals)
  out_flat = dlpack_callback_p.bind(
      *flat_args,
      callback=_flat_callback,
      result_avals=tuple(flat_result_avals),
      vectorized=vectorized,
      mutable_results=mutable_results,
  )
  return jax.tree.unflatten(out_tree, out_flat)


dlpack_callback_p = core.Primitive("dlpack_callback")
dlpack_callback_p.multiple_results = True
dlpack_callback_p.def_impl(
    functools.partial(xla.apply_primitive, dlpack_callback_p)
)


@dlpack_callback_p.def_abstract_eval
def dlpack_callback_abstract_eval(*args, result_avals: Any, **kwargs: Any):
  del args, kwargs
  return result_avals


def dlpack_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "DLPack callbacks do not support JVP. "
      "Please use `jax.custom_jvp` to use callbacks while taking gradients."
  )


ad.primitive_jvps[dlpack_callback_p] = dlpack_callback_jvp_rule


def dlpack_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "DLPack callbacks do not support transpose. "
      "Please use `jax.custom_vjp` to use callbacks while taking gradients."
  )


ad.primitive_transposes[dlpack_callback_p] = dlpack_callback_transpose_rule


if gpu_dlpack:
  mlir.register_lowering(
      dlpack_callback_p, gpu_dlpack.cuda_dlpack_callback, platform="cuda"
  )


def dlpack_callback_batching(
    args,
    dims,
    *,
    callback,
    result_avals,
    vectorized,
    mutable_results,
):
  axis_size = next(
      a.shape[d] for a, d in zip(args, dims) if d is not batching.not_mapped
  )
  new_args = [
      arg if dim is batching.not_mapped else batching.moveaxis(arg, dim, 0)
      for arg, dim in zip(args, dims)
  ]
  if vectorized:
    result_avals = tuple(
        core.unmapped_aval(axis_size, core.no_axis_name, 0, aval)
        for aval in result_avals
    )
    outvals = dlpack_callback_p.bind(
        *new_args,
        callback=callback,
        result_avals=result_avals,
        vectorized=vectorized,
        mutable_results=mutable_results,
    )
  else:
    is_batched = [d is not batching.not_mapped for d in dims]
    unbatched_args, batched_args = util.partition_list(is_batched, new_args)

    def _batch_fun(batched_args):
      return dlpack_callback_p.bind(
          *util.merge_lists(is_batched, unbatched_args, batched_args),
          callback=callback,
          result_avals=result_avals,
          vectorized=vectorized,
          mutable_results=mutable_results,
      )

    outvals = jax.lax.map(_batch_fun, batched_args)
  return tuple(outvals), (0,) * len(outvals)


batching.primitive_batchers[dlpack_callback_p] = dlpack_callback_batching
