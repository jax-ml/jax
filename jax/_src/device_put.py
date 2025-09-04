# Copyright 2018 The JAX Authors.
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

# Primitive dispatch and jit dispatch.
from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
from functools import partial
from typing import Any, Callable

from jax._src import api
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import profiler
from jax._src import sharding_impls
from jax._src import traceback_util
from jax._src import typing
from jax._src import util

from jax._src import xla_bridge
from jax._src.abstract_arrays import array_types
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.layout import Layout, Format
from jax._src.lib import xla_client as xc
from jax._src.mesh import Mesh
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    NamedSharding, SingleDeviceSharding,
    is_single_device_sharding)
import numpy as np


zip, unsafe_zip = util.safe_zip, zip

ArrayCopySemantics = xc.ArrayCopySemantics
Device = xc.Device


traceback_util.register_exclusion(__file__)

def _identity_fn(x):
  return x


def _different_device_order_reshard(
    x: array.ArrayImpl, target_sharding: NamedSharding, copy: ArrayCopySemantics
) -> array.ArrayImpl:
  x._check_if_deleted()
  inp_sharding = x.sharding
  assert isinstance(inp_sharding, NamedSharding)

  donate_argnums = 0 if copy == ArrayCopySemantics.DONATE_INPUT else None
  if inp_sharding._device_assignment == target_sharding._device_assignment:
    return api.jit(_identity_fn, out_shardings=target_sharding,
                   donate_argnums=donate_argnums)(x)

  if inp_sharding.is_fully_replicated:
    permute_order = None
  else:
    permute_order = np.vectorize(target_sharding._device_assignment.index,
                                  otypes=[int])(inp_sharding._device_assignment)
  new_mesh = Mesh(
      target_sharding.mesh.devices.reshape(inp_sharding.mesh.axis_sizes),
      inp_sharding.mesh.axis_names)
  new_s = NamedSharding(
      new_mesh, inp_sharding.spec, memory_kind=target_sharding.memory_kind,
      _logical_device_ids=(None if permute_order is None else
                            tuple(permute_order.tolist())))
  new_x = xc.reorder_shards(x, new_s, ArrayCopySemantics.REUSE_INPUT)  # type: ignore
  return api.jit(_identity_fn, out_shardings=target_sharding,
                donate_argnums=donate_argnums)(new_x)


@util.cache(max_size=2048, trace_context_in_key=False)
def _is_supported_cross_host_transfer(ndim, src_sharding, dst_sharding):
  """Returns True if src->dst is a supported cross-host transfer."""
  if (src_sharding._internal_device_list.device_kind !=
      dst_sharding._internal_device_list.device_kind):
    return False
  if (src_sharding._to_xla_hlo_sharding(ndim) !=
      dst_sharding._to_xla_hlo_sharding(ndim)):
    return False
  # This check excludes the case where the source and destination shardings
  # have the same process index sets but there are shards that require
  # cross-host transfers. This case is supportable but expensive to check for.
  different_process_inds = (
      src_sharding._internal_device_list.process_indices !=
      dst_sharding._internal_device_list.process_indices)
  backend = xla_bridge.get_backend()
  # If a cross-host device transfer is requested but the backend does not
  # support it, then the user must set the flags to enable DCN-based transfers.
  if (different_process_inds and
      not getattr(backend, 'supports_cross_host_transfers', False) and
      not xla_bridge.CROSS_HOST_TRANSFER_SOCKET_ADDRESS.value):
    raise ValueError(
        f"The backend ({backend.platform}, {backend.platform_version}) does "
        "not support cross-host device transfers via ICI/NCCL. Please set "
        "jax_cross_host_transfer_socket_address and (optionally) "
        "jax_cross_host_transport_addresses flags to enable DCN-based cross "
        "host device transfers.")
  return different_process_inds

@dataclasses.dataclass(frozen=True)
class _DeferredShardArg:
  """Deferred call to `shard_args`.

  Per-array impls return this object instead of a result array to indicate a
  deferred `shard_args` call. `_batched_device_put_impl` then batches all
  `_DeferredShardArg` objects into a single `shard_args` call.
  """

  x: Any
  s: Sharding
  aval: core.AbstractValue
  committed: bool
  copy_semantics: ArrayCopySemantics

  def result_handler(self, shard_arg_result):
    return pxla.global_aval_to_result_handler(
        self.aval, self.s, self.committed)(shard_arg_result)


def _device_put_sharding_impl(
    x: Any,
    aval: core.ShapedArray,
    device: Device | Sharding | None,
    copy: ArrayCopySemantics,
):
  from jax.experimental import multihost_utils  # pytype: disable=import-error

  if isinstance(x, array.ArrayImpl):
    x_is_jax_array = True
    x_is_fully_addressable, x_sharding = x.is_fully_addressable, x.sharding
  else:
    x_is_jax_array = False
    x_is_fully_addressable, x_sharding = None, None

  if isinstance(device, Sharding):
    s = device
    s_is_fully_addressable = s.is_fully_addressable
    if (getattr(x, 'sharding', None) == s and getattr(x, '_committed', False)
        and copy == ArrayCopySemantics.REUSE_INPUT):
      return x

    if (not s_is_fully_addressable and
        x_is_jax_array and not x_is_fully_addressable and
        s.device_set == x_sharding.device_set):
      assert isinstance(s, NamedSharding), s
      return _different_device_order_reshard(x, s, copy)

    if (s_is_fully_addressable and x_is_jax_array and
        x_is_fully_addressable and s.num_devices > 1 and
        s._internal_device_list != x_sharding._internal_device_list and  # pytype: disable=attribute-error
        s.device_set == x_sharding.device_set):
      assert isinstance(s, NamedSharding), s
      return _different_device_order_reshard(x, s, copy)

    if (x_is_jax_array and x._committed and xla_bridge.process_count() > 1
        and _is_supported_cross_host_transfer(x.ndim, x_sharding, s)):
      return xc.batched_copy_array_to_devices_with_sharding(
          [x], [s._internal_device_list], [s],  # pytype: disable=attribute-error
          [copy])[0]

    if not s_is_fully_addressable:
      # If both the source and target shardings are not fully addressable and
      # one of the above conditions has not been met, then assume that the user
      # is attempting a different device order reshard.
      if (x_is_jax_array and not x_is_fully_addressable
          and s.device_set != x_sharding.device_set):
        inp_ids = [d.id for d in x_sharding._device_assignment]
        inp_plat = x_sharding._device_assignment[0].platform.upper()
        target_ids = [d.id for d in s._device_assignment]
        target_plat = s._device_assignment[0].platform.upper()
        raise ValueError(
            "For a cross-host reshard in multi-controller JAX, input and target"
            " sharding should have the same set of devices. Got input's device"
            f" set ids: {inp_ids} on platform {inp_plat} and target sharding's"
            f" device set ids: {target_ids} on platform {target_plat}.\n\n"
            "There is experimental support for cross-host transfers with "
            "different device sets, when input/output shardings have the same "
            "indices and layouts, in the TFRT TPU runtime only.")

      if ((x_is_jax_array and not x._committed) or
          type(x) in array_types or type(x) in dtypes.python_scalar_types):
        # If all hosts participate in the sharding, assert that the input is the
        # same on all hosts. If some hosts have no addressable devices in the
        # sharding, bypass the check, since we can't easily distinguish between
        # these two cases: (1) the sharding contains the same subset of global
        # devices on all hosts (and hosts with no addressable devices in the
        # sharding do not transfer data) or (2) the sharding contains a
        # different subset of devices on each host. For (1), the input should be
        # the same on all hosts, but for (2) it need not be.
        if xla_bridge.process_count() == len(s._internal_device_list.process_indices):  # pytype: disable=attribute-error
          multihost_utils.assert_equal(
              x, fail_message=(
                  f"{type(x)} passed to device_put is not the same on each"
                  " process. Make sure you are passing the same value of"
                  f" {type(x)} on each process."))
        return _DeferredShardArg(x, s, aval, True, copy)
      # TODO(yashkatariya,mattjj): Link to a doc about McJAX and jax.Array.
      raise ValueError(
          "device_put's second argument must be a Device or a Sharding which"
          f" represents addressable devices, but got {s}. Please pass device or"
          " Sharding which represents addressable devices.")
    return _DeferredShardArg(x, s, aval, True, copy)

  # Only `Device` exists below. `Sharding` instance is handled above.
  if x_is_jax_array:
    if not x_is_fully_addressable:
      raise ValueError(
          "device_put's first argument must be a fully addressable array, but "
          f"got value with devices {x.devices()}")
    if device is None:
      if copy == ArrayCopySemantics.REUSE_INPUT:
        return x
      else:
        return _DeferredShardArg(x, x_sharding, aval, x.committed, copy)
    elif is_single_device_sharding(x_sharding):
      device = x_sharding._device_assignment[0] if device is None else device
      if copy == ArrayCopySemantics.ALWAYS_COPY:
        return xc.batched_device_put(aval, SingleDeviceSharding(device), [x],
                                     [device], True, True)
      return batched_device_put(aval, SingleDeviceSharding(device), [x],
                                [device])

  sh = SingleDeviceSharding(pxla.get_default_device()
                            if device is None else device)
  return _DeferredShardArg(x, sh, aval, device is not None, copy)


def _device_put_impl(
    x, *, device: Device | Sharding | Format | None,
    src: Device | Sharding | Format | None, copy: ArrayCopySemantics, aval):
  if aval is None:
    try:
      aval = core.abstractify(x)
    except TypeError as err:
      raise TypeError(
          f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err

  if isinstance(device, core.MemorySpace):
    return dispatch.apply_primitive(device_put_p, x, devices=(device,),
                                    srcs=(src,), copy_semantics=(copy,))[0]

  if isinstance(device, Format):
    l = device
    dll = l.layout
    x_dll = x.format.layout if hasattr(x, 'format') else None
    if dll is None and l.sharding is None:
      return _device_put_sharding_impl(x, aval, l.sharding, copy)
    if (not isinstance(l.sharding, Sharding) or
        not isinstance(dll, (Layout, type(None)))):
      raise ValueError(
          "sharding and layout in `Layout` instance should be"
          f" concrete. Got layout: {l} for input {aval.str_short()}")
    if (getattr(x, 'format', None) == l and getattr(x, '_committed', False) and
        copy == ArrayCopySemantics.REUSE_INPUT):
      return x
    if x_dll is None and dll is None:
      return _device_put_sharding_impl(x, aval, l.sharding, copy)
    return api.jit(
        _identity_fn,
        out_shardings=l,
        donate_argnums=(0 if copy == ArrayCopySemantics.DONATE_INPUT else None),
    )(x)

  return _device_put_sharding_impl(x, aval, device, copy)


def _batched_device_put_impl(
    *xs,
    devices: Sequence[Device | Sharding | Format | None],
    srcs: Sequence[Device | Sharding | Format | None],
    copy_semantics: Sequence[ArrayCopySemantics],
    x_avals: Sequence[core.ShapedArray | None]):
  ys = []
  dsa_indices, dsa_xs, dsa_shardings, dsa_copy_semantics = [], [], [], []
  for i, (x, device, src, cp, aval) in enumerate(
      zip(xs, devices, srcs, copy_semantics, x_avals)):
    y = _device_put_impl(x, device=device, src=src, copy=cp, aval=aval)
    if isinstance(y, _DeferredShardArg):
      dsa_indices.append(i)
      dsa_xs.append(y.x)
      dsa_shardings.append(y.s)
      dsa_copy_semantics.append(y.copy_semantics)
    ys.append(y)

  if dsa_xs:
    # Batch shard_arg calls. Helps improve efficiency for backends that support
    # efficient batch transfer.
    # device_put handles `Format` via a different path, so just pass `None` as
    # the layout here.
    shard_arg_results = shard_args(
        dsa_shardings, [None] * len(dsa_xs), dsa_copy_semantics, dsa_xs)
    for i, shard_arg_result in zip(dsa_indices, shard_arg_results):
      assert isinstance(ys[i], _DeferredShardArg)
      ys[i] = ys[i].result_handler(shard_arg_result)
  return ys

def batched_device_put_impl(
    *xs,
    devices: Sequence[Device | Sharding | Format | None],
    srcs: Sequence[Device | Sharding | Format | None],
    copy_semantics: Sequence[ArrayCopySemantics]):
  return _batched_device_put_impl(
      *xs, devices=devices, srcs=srcs, copy_semantics=copy_semantics,
      x_avals=[None] * len(devices))


device_put_p = core.Primitive('device_put')
device_put_p.multiple_results = True
device_put_p.def_impl(batched_device_put_impl)


def _device_put_abstract_eval(*xs, devices, srcs, copy_semantics):
  out = []
  for x, d in zip(xs, devices):
    if isinstance(d, Sharding) and d.memory_kind is not None:
      out.append(x.update(memory_space=core.mem_kind_to_space(d.memory_kind)))
    elif isinstance(d, core.MemorySpace):
      out.append(x.update(memory_space=d))
    else:
      out.append(x)
  return out
device_put_p.def_abstract_eval(_device_put_abstract_eval)

def _device_put_transpose(cts, *_, devices, srcs, copy_semantics):
  results = [None] * len(cts)
  dp_args = []
  for i, (ct, device, src, cp) in enumerate(zip(cts, devices, srcs, copy_semantics)):
    if type(ct) is not ad.Zero:
      dp_args.append((i, ct, device, src, cp))
  if dp_args:
    indices, args, devices, srcs, copy_semantics = list(zip(*dp_args))
    new_copy_semantics = []
    for cp in copy_semantics:
      if cp == ArrayCopySemantics.DONATE_INPUT:
        raise ValueError(
            "donate=True is not allowed during tranposition of device_put."
            " Please file an issue if you want this to be supported.")
      elif cp == ArrayCopySemantics.REUSE_INPUT:
        new_copy_semantics.append(ArrayCopySemantics.ALWAYS_COPY)
      else:
        assert cp == ArrayCopySemantics.ALWAYS_COPY
        new_copy_semantics.append(ArrayCopySemantics.ALWAYS_COPY)
    ys = device_put_p.bind(*args, devices=srcs, srcs=devices,
                           copy_semantics=tuple(new_copy_semantics))
    for i, y in zip(indices, ys):
      results[i] = y
  return results
ad.primitive_jvps[device_put_p] = partial(ad.linear_jvp, device_put_p)
ad.primitive_transposes[device_put_p] = _device_put_transpose

def _device_put_batcher(batched_args, batch_dims, **params):
  mapped_batch_dims = [bd for bd in batch_dims if bd is not batching.not_mapped]
  assert not mapped_batch_dims or all(
      mapped_batch_dims[0] == bd for bd in mapped_batch_dims[1:]
  ), batch_dims
  return device_put_p.bind(*batched_args, **params), batch_dims
batching.primitive_batchers[device_put_p] = _device_put_batcher

def _tpu_gpu_device_put_lowering(ctx, *xs, devices, srcs, copy_semantics):
  # TODO(yashkatariya): Maybe we should add the custom calls anyways if it's
  # being used inside jit? Atleast for now, this preserves the old behavior.
  if ctx.module_context.all_default_mem_kind:
    return xs
  def lower(x, device, aval, out_aval):
    if ((isinstance(device, Sharding) and device.memory_kind is not None) or
        isinstance(device, core.MemorySpace)):
      if isinstance(device, Sharding):
        if config.use_shardy_partitioner.value:
          x = mlir.wrap_with_sharding_op(
              ctx, x, out_aval,
              device._to_sdy_sharding(aval.ndim))
        else:
          x = mlir.wrap_with_sharding_op(
              ctx, x, out_aval,
              device._to_xla_hlo_sharding(aval.ndim).to_proto())
      mem_kind = (core.mem_space_to_kind(device)
                  if isinstance(device, core.MemorySpace) else device.memory_kind)
      x = mlir.wrap_with_memory_kind(x, mem_kind, out_aval)
      return x
    return x
  return list(map(lower, xs, devices, ctx.avals_in, ctx.avals_out))

mlir.register_lowering(
  device_put_p, _tpu_gpu_device_put_lowering, platform='tpu')
mlir.register_lowering(
  device_put_p, _tpu_gpu_device_put_lowering, platform='gpu')


def _common_device_put_lowering(ctx, *xs, devices, srcs, copy_semantics):
  return xs
mlir.register_lowering(device_put_p, _common_device_put_lowering)



@profiler.annotate_function
def shard_args(
    shardings: Sequence[Sharding],
    layouts: Sequence[Any | None],
    copy_semantics: Sequence[xc.ArrayCopySemantics],
    args: Sequence[Any],
    canonicalize: bool = True,
) -> Sequence[xc.ArrayImpl]:
  # Fast path for one argument.
  if len(args) == 1:
    arg = args[0]
    if canonicalize:
      arg = dtypes.canonicalize_value(arg)
    return shard_arg_handlers[type(arg)]([arg], shardings, layouts,
                                         copy_semantics)

  # type(arg) -> (list[indices], list[args], list[shardings], list[layouts],
  #               list[copy_semantics])
  batches = collections.defaultdict(lambda: ([], [], [], [], []))  # type: ignore
  for i, (arg, sharding, layout, cs) in enumerate(
      zip(args, shardings, layouts, copy_semantics)):
    if canonicalize:
      arg = dtypes.canonicalize_value(arg)
    batch = batches[type(arg)]
    batch[0].append(i)
    batch[1].append(arg)
    batch[2].append(sharding)
    batch[3].append(layout)
    batch[4].append(cs)

  # Call `shard_arg_handlers` per batch and build a flat list of arrays returned
  # from each call in the same order as `args`. Since `batches` is grouped by
  # types, we cannot simply flatten the results and we have to use the original
  # indices to put each array back to its original position.
  results: list[typing.Array | None] = [None] * len(args)
  for t, (indices, a, s, l, xcs) in batches.items():
    outs = shard_arg_handlers[t](a, s, l, xcs)
    for i, out in zip(indices, outs):
      results[i] = out
  assert all(result is not None for result in results)
  return results


shard_arg_handlers: dict[
    Any,
    Callable[
      [Sequence[Any], Sequence[Any], Sequence[Any],
       Sequence[xc.ArrayCopySemantics]],
      Sequence[Any],
    ],
] = {}

shard_arg_handlers[array.ArrayImpl] = array.array_shard_arg
shard_arg_handlers[core.Token] = array.token_shard_arg

def _masked_array_error(xs, shardings, layouts, copy_semantics):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")
shard_arg_handlers[np.ma.MaskedArray] = _masked_array_error

def _shard_np_array(xs, shardings, layouts, copy_semantics):
  results = []
  for x, sharding, layout in zip(xs, shardings, layouts):
    devices = sharding._addressable_device_assignment
    if x.dtype == dtypes.float0:
      x = np.zeros(x.shape, dtype=np.dtype(bool))
    aval = core.shaped_abstractify(x)
    if layout is not None:
      results.append(api.device_put(x, Format(layout, sharding)))
    else:
      if sharding.is_fully_replicated:
        shards = [x] * len(devices)
      else:
        indices = tuple(sharding.addressable_devices_indices_map(x.shape).values())
        shards = [x[i] for i in indices]
      results.append(batched_device_put(aval, sharding, shards, devices))
  return results
for _t in array_types:
  shard_arg_handlers[_t] = _shard_np_array

def _shard_python_scalar(xs, shardings, layouts, copy_semantics):
  return shard_args(shardings, layouts, copy_semantics,
                    [np.array(x) for x in xs])
for _t in dtypes.python_scalar_types:
  shard_arg_handlers[_t] = _shard_python_scalar

def _shard_darray(xs, shardings, layouts, copy_semantics):
  return shard_args(shardings, layouts, copy_semantics, [x._data for x in xs])
shard_arg_handlers[core.DArray] = _shard_darray

def _shard_mutable_array(xs, shardings, layouts, copy_semantics):
  return shard_args(shardings, layouts, copy_semantics, [x._buf for x in xs])
shard_arg_handlers[core.MutableArray] = _shard_mutable_array



def batched_device_put(aval: core.ShapedArray,
                       sharding: Sharding, xs: Sequence[Any],
                       devices: Sequence[xc.Device], committed: bool = True):
  util.test_event("batched_device_put_start")
  try:
    bufs = [x for x, d in zip(xs, devices)
            if (isinstance(x, array.ArrayImpl) and
                sharding_impls.is_single_device_sharding(x.sharding) and
                x.devices() == {d})]
    if len(bufs) == len(xs) > 0:
      return array.ArrayImpl(
          aval, sharding, bufs, committed=committed, _skip_checks=True)
    return xc.batched_device_put(aval, sharding, xs, list(devices), committed)
  finally:
    util.test_event("batched_device_put_end")

