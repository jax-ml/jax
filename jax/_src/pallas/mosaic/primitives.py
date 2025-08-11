# Copyright 2023 The JAX Authors.
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

"""Module for Pallas:TPU-specific JAX primitives and functions."""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import jax
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import pretty_printer as pp
from jax._src import prng as jax_prng
from jax._src import random as jax_random
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.pallas import core as pl_core
from jax._src.pallas import primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax._src.state.types import Transform
from jax._src.typing import DTypeLike
import jax.numpy as jnp

Slice = indexing.Slice

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

IntDeviceId = int | jax.Array
MultiDimDeviceId = tuple[IntDeviceId, ...] | dict[str | tuple[str, ...], IntDeviceId]
Ref = state.AbstractRef | state.TransformedRef

repeat_p = jax_core.Primitive('repeat')

def repeat(x, repeats, axis):
  axis = util.canonicalize_axis(axis, x.ndim)
  return repeat_p.bind(x, repeats=repeats, axis=axis)

@repeat_p.def_abstract_eval
def _repeat_abstract_eval(x, *, repeats, axis):
  if axis < 0 or axis >= len(x.shape):
    raise ValueError(f"axis: {axis} is out of range [0, {len(x.shape)})")
  shape = list(x.shape)
  shape[axis] *= repeats
  return jax_core.ShapedArray(shape, x.dtype)


def _repeat_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, repeats, axis):
  def _repeat(x):
    return jnp.repeat(x, repeats, axis)
  return mlir.lower_fun(_repeat, multiple_results=False)(ctx, x)
mlir.register_lowering(repeat_p, _repeat_lowering_rule)

bitcast_p = jax_core.Primitive("bitcast")


def bitcast(x, ty: DTypeLike):
  ty = dtypes.canonicalize_dtype(ty)
  if len(x.shape) < 2:
    raise ValueError("Not implemented: bitcast 1D")
  src_bitwidth = dtypes.bit_width(x.dtype)
  dst_bitwidth = dtypes.bit_width(ty)
  if x.shape[-2] * src_bitwidth % dst_bitwidth:
    raise ValueError(
        "Not implemented: the 2nd minor dim can not be perfectly packed or"
        " unpacked"
    )
  return bitcast_p.bind(x, ty=ty)


@bitcast_p.def_abstract_eval
def _bitcast_abstract_eval(x, *, ty):
  shape = list(x.shape)
  src_bitwidth = dtypes.bit_width(x.dtype)
  dst_bitwidth = dtypes.bit_width(ty)
  shape[-2] = shape[-2] * src_bitwidth // dst_bitwidth
  return jax_core.ShapedArray(shape, ty)


def _bitcast_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, ty):
  def _bitcast(x):
    src_bitwidth = dtypes.bit_width(x.dtype)
    dst_bitwidth = dtypes.bit_width(ty)
    if src_bitwidth < dst_bitwidth:
      *leading, m, n = x.shape
      packing = dst_bitwidth // src_bitwidth
      x = x.reshape(*leading, m // packing, packing, n)
      x = jnp.swapaxes(x, -1, -2)
      return jax.lax.bitcast_convert_type(x, ty)
    if src_bitwidth > dst_bitwidth:
      y = jax.lax.bitcast_convert_type(x, ty)
      *leading, m, n, packing = y.shape
      return jnp.swapaxes(y, -1, -2).reshape(*leading, m * packing, n)
    return jax.lax.bitcast_convert_type(x, ty)

  return mlir.lower_fun(_bitcast, multiple_results=False)(ctx, x)


mlir.register_lowering(bitcast_p, _bitcast_lowering_rule)

roll_p = jax_core.Primitive("roll")


def roll(
    x,
    shift,
    axis: int,
    *,
    stride: int | None = None,
    stride_axis: int | None = None,
):
  if isinstance(shift, int) and shift < 0:
    raise ValueError("shift must be non-negative.")
  if axis < 0 or axis >= len(x.shape):
    raise ValueError("axis is out of range.")
  if (stride is None) != (stride_axis is None):
    raise ValueError("stride and stride_axis must be both specified or not.")
  if stride is not None and stride_axis is not None:
    if stride < 0:
      raise ValueError("stride must be non-negative.")
    if stride_axis < 0 or stride_axis >= len(x.shape):
      raise ValueError("stride_axis is out of range")
    if axis == stride_axis:
      raise ValueError("expected axis and stride_axis are different.")
  return roll_p.bind(
      x, shift, axis=axis, stride=stride, stride_axis=stride_axis
  )


@roll_p.def_abstract_eval
def _roll_abstract_eval(x, shift, **_):
  del shift
  return x


def _roll_lowering_rule(
    ctx: mlir.LoweringRuleContext, x, shift, *, axis, stride, stride_axis
):
  def _roll(x, shift):
    if stride is None:
      return jnp.roll(x, shift, axis)
    outputs = [
        jnp.roll(xs, shift + i * stride, axis)
        for i, xs in enumerate(jnp.split(x, x.shape[stride_axis], stride_axis))
    ]
    return jnp.concatenate(outputs, stride_axis)

  return mlir.lower_fun(_roll, multiple_results=False)(ctx, x, shift)


mlir.register_lowering(roll_p, _roll_lowering_rule)


@dataclasses.dataclass
class AsyncCopyDescriptor:
  src_ref: Any
  src_transforms: tuple[Transform, ...]
  dst_ref: Any
  dst_transforms: tuple[Transform, ...]
  dst_sem: int | jax.Array
  dst_sem_transforms: tuple[Transform, ...]
  src_sem: int | jax.Array | None
  src_sem_transforms: tuple[Transform, ...] | None
  device_id: MultiDimDeviceId | IntDeviceId | None
  device_id_type: primitives.DeviceIdType = primitives.DeviceIdType.MESH
  _used: bool = dataclasses.field(
      default=False, init=False, compare=False, hash=False
  )

  def __post_init__(self):
    if (self.src_sem is None) ^ (self.device_id is None):
      raise ValueError("Either both or neither `src_sem` and `device_id` "
                       "can be set.")

  def __del__(self):
    if not self._used:
      # Exceptions in ``__del__`` are ignored, so logging is our only option.
      logging.error(
          "AsyncCopyDescriptor was not used."
          " Did you mean to call `start` or `wait` on it?"
      )

  @property
  def is_remote(self):
    return self.src_sem is not None

  def _get_args_and_tree(self, swap_src_and_dst: bool = False):
    if swap_src_and_dst:
      return tree_util.tree_flatten((
          self.dst_ref,
          self.dst_transforms,
          self.src_ref,
          self.src_transforms,
          self.src_sem,
          self.src_sem_transforms,
          self.dst_sem,
          self.dst_sem_transforms,
          self.device_id,
      ))
    else:
      return tree_util.tree_flatten((
          self.src_ref,
          self.src_transforms,
          self.dst_ref,
          self.dst_transforms,
          self.dst_sem,
          self.dst_sem_transforms,
          self.src_sem,
          self.src_sem_transforms,
          self.device_id,
      ))

  def start(self, priority: int = 0):
    self._used = True
    flat_args, tree = self._get_args_and_tree()
    dma_start_p.bind(
        *flat_args,
        tree=tree,
        device_id_type=self.device_id_type,
        priority=priority,
    )

  def wait(self):
    if self.is_remote:
      self.wait_send()
    self.wait_recv()

  def wait_recv(self):
    self._used = True
    flat_args, tree = self._get_args_and_tree()
    dma_wait_p.bind(
        *flat_args, tree=tree, device_id_type=self.device_id_type
    )

  def wait_send(self):
    self._used = True
    if not self.is_remote:
      raise ValueError("Cannot `wait_send` on a local copy.")
    # We swap src and dst since by default dma_wait_p waits on the dst_sem
    # As a clean up, maybe we could modify the primitive to have a
    # `wait_on_send` bool.
    flat_args, tree = self._get_args_and_tree(swap_src_and_dst=True)
    dma_wait_p.bind(
        *flat_args, tree=tree, device_id_type=self.device_id_type
    )


def _get_dma_effects(
    src_transforms_avals,
    dst_transforms_avals,
    dst_sem_transforms_avals,
    src_sem_aval,
):
  n_src_transforms = len(tree_util.tree_leaves(src_transforms_avals))
  n_dst_transforms = len(tree_util.tree_leaves(dst_transforms_avals))
  n_dst_sem_transforms = len(tree_util.tree_leaves(dst_sem_transforms_avals))
  dst_sem_index = 1 + n_src_transforms + 1 + n_dst_transforms
  effs = {
      state.ReadEffect(0),  # Read from src ref
      state.WriteEffect(n_src_transforms + 1),  # Write to dst ref
      state.WriteEffect(dst_sem_index),  # Write to dst sem
  }
  if src_sem_aval is not None:
    src_sem_index = (
        1 + n_src_transforms + 1 + n_dst_transforms + 1 + n_dst_sem_transforms
    )
    effs.add(state.WriteEffect(src_sem_index))
  return effs


dma_start_p = jax_core.Primitive('dma_start')
dma_start_p.multiple_results = True

@dma_start_p.def_effectful_abstract_eval
def _dma_start_abstract_eval(*args, tree, device_id_type, priority):
  if priority < 0:
    raise ValueError(f"DMA start priority must be non-negative: {priority}")
  (
      src_ref_aval,
      src_transforms_avals,
      dst_ref_aval,
      dst_transforms_avals,
      dst_sem_aval,
      dst_sem_transforms_avals,
      src_sem_aval,
      src_sem_transforms_avals,
      device_id_aval,
  ) = tree_util.tree_unflatten(tree, args)
  dst_sem_shape = dst_sem_aval.shape
  if dst_sem_transforms_avals:
    dst_sem_shape = dst_sem_transforms_avals[-1].get_indexer_shape()
  if dst_sem_shape:
    raise ValueError(
        f"Cannot signal on a non-()-shaped semaphore: {dst_sem_shape}"
    )
  if src_sem_aval is not None:
    src_sem_shape = src_sem_aval.shape
    if src_sem_transforms_avals:
      src_sem_shape = src_sem_transforms_avals[-1].get_indexer_shape()
    if src_sem_shape:
      raise ValueError(
          f"Cannot signal on a non-()-shaped semaphore: {src_sem_shape}"
      )
  return [], _get_dma_effects(
      src_transforms_avals,
      dst_transforms_avals,
      dst_sem_transforms_avals,
      src_sem_aval,
  )

def _dma_start_pp_eqn(eqn: jax_core.JaxprEqn,
                      context: jax_core.JaxprPpContext,
                      settings: jax_core.JaxprPpSettings):
  invars = eqn.invars
  tree = eqn.params["tree"]
  priority = eqn.params["priority"]
  (
      src_ref,
      src_transforms,
      dst_ref,
      dst_transforms,
      dst_sem,
      dst_sem_transforms,
      src_sem,
      src_sem_transforms,
      device_id,
  ) = tree_util.tree_unflatten(tree, invars)
  del src_sem_transforms
  # TODO(sharadmv): pretty print source semaphores and device id
  if src_sem or device_id:
    return jax_core._pp_eqn(eqn, context, settings)
  return pp.concat([
      pp.text(f"dma_start(p{priority})"),
      pp.text(" "),
      sp.pp_ref_transforms(context, src_ref, src_transforms),
      pp.text(" -> "),
      sp.pp_ref_transforms(context, dst_ref, dst_transforms),
      pp.text(" "),
      sp.pp_ref_transforms(context, dst_sem, dst_sem_transforms),
  ])

jax_core.pp_eqn_rules[dma_start_p] = _dma_start_pp_eqn


def dma_start_partial_discharge_rule(
    should_discharge, in_avals, out_avals, *args, tree, device_id_type, priority
):
  # Note: we ignore the DMA priority in discharge rules.
  del priority
  (
      src_ref,
      src_transforms,
      dst_ref,
      dst_transforms,
      dst_sem,
      dst_sem_transforms,
      src_sem,
      src_sem_transforms,
      device_id,
  ) = tree_util.tree_unflatten(tree, args)
  (
      _,
      src_transforms_avals,
      _,
      dst_transforms_avals,
      dst_sem_aval,
      dst_sem_transforms_avals,
      src_sem_aval,
      src_sem_transforms_avals,
      _,
  ) = tree_util.tree_unflatten(tree, in_avals)
  del out_avals

  (
      _,
      _,
      dst_discharge,
      _,
      dst_sem_discharge,
      _,
      *maybe_src_sem_discharge,
  ) = tree_util.tree_unflatten(tree, should_discharge)
  is_remote = device_id is not None
  src_sem_discharge = None

  if is_remote:
    src_sem_discharge = maybe_src_sem_discharge[0]

  if not is_remote:
    # Local async copies only use one semaphore.
    assert src_sem is None
    assert src_sem_transforms is None

  num_src_sem_transforms = len(tree_util.tree_leaves(src_sem_transforms_avals))
  num_dst_sem_transforms = len(tree_util.tree_leaves(dst_sem_transforms_avals))
  num_src_transform_vals = len(tree_util.tree_leaves(src_transforms_avals))
  num_dst_transform_vals = len(tree_util.tree_leaves(dst_transforms_avals))

  updates = state_discharge.transform_array(src_ref[...], src_transforms)
  local_src = updates

  if is_remote:
    # Note that this code only works in SPMD mode. If not all devices execute
    # the DMA then the devices that do will hang.
    # TODO(justinfu): Verify that code only works in SPMD mode.
    axis_env = jax_core.get_axis_env()
    nonempty_axes = [name for name in axis_env.axis_sizes if name is not None]
    if isinstance(device_id, dict):
      if device_id_type is not primitives.DeviceIdType.MESH:
        raise ValueError(
            "`device_id_type` must be MESH if `device_id` is a dict,"
            f" got: {device_id_type = }."
        )
      device_id_list = []
      for axis in nonempty_axes:
        device_id_list.append(device_id.get(axis, jax.lax.axis_index(axis)))
      device_id = tuple(device_id_list)
    if device_id_type == primitives.DeviceIdType.LOGICAL:
      if len(nonempty_axes) > 1:
        raise NotImplementedError("Sharding with more than one named axis not "
                                  "implemented in dma_start_p for LOGICAL "
                                  "device_id_type.")
      shard_axis = nonempty_axes[0]
      my_axis = jax.lax.axis_index(shard_axis)
    elif device_id_type == primitives.DeviceIdType.MESH:
      device_id_len = 1
      if isinstance(device_id, jax.Array):
        device_id_len = device_id.size
      elif hasattr(device_id, '__len__'):
        device_id_len = len(device_id)
      if device_id_len != len(axis_env.axis_sizes):
        raise ValueError(
            f"device_id ({device_id_len}) and mesh ({len(axis_env.axis_sizes)}) "
            "must have same length.")
      if device_id_len > 1 or len(nonempty_axes) > 1:
        raise NotImplementedError("Meshes with more than 1 named dimension not "
                                  "implemented in dma_start_p")
      shard_axis = nonempty_axes[0]
      my_axis = jax.lax.axis_index(shard_axis)
    else:
      raise ValueError(f"Unknown device_id_type: {device_id_type}")
    # Compute the update that is being sent to the current device.
    who_copy_to_me = jax.lax.all_gather(device_id, shard_axis) == my_axis
    # TODO(justinfu): Add a checkify for verifying there is at most one source.
    # TODO(justinfu): Handle the case where no other device is copying to
    # this device.
    index = jnp.argmax(who_copy_to_me, axis=0)
    global_updates = jax.lax.all_gather(updates, shard_axis)
    updates = jax.lax.dynamic_index_in_dim(
        global_updates, index, axis=0, keepdims=False)

    # Handle asymmetrical indexing when devices do not share the same
    # dst_transform.
    global_dst_transforms = tree_util.tree_map(
        lambda x: jax.lax.all_gather(x, shard_axis), dst_transforms
    )
    dst_transforms = tree_util.tree_map(
        lambda x: jax.lax.dynamic_index_in_dim(
            x, index, axis=0, keepdims=False
        ),
        global_dst_transforms,
    )

  def do_discharge_dst(dst_ref=dst_ref):
    _, ret = state_discharge.transform_swap_array(
        dst_ref, dst_transforms, updates
    )
    return ret

  # Update semaphore values.
  # TODO(justinfu): Potentially handle asymmetric copy sizes.
  def do_discharge_dst_sem(dst_sem=dst_sem):
    recv_size = jnp.minimum(updates.size, pl_core.SEMAPHORE_MAX_VALUE)
    recv_size = jnp.array(recv_size, dtype=pl_core.SEMAPHORE_INTERPRET_DTYPE)
    dst_sem_value = primitives._transform_semaphore(
        dst_sem, dst_sem_transforms, dst_sem_aval
    )
    _, ret = state_discharge.transform_swap_array(
        dst_sem, dst_sem_transforms, dst_sem_value[...] + recv_size
    )
    return ret

  def do_discharge_src_sem(src_sem=src_sem):
    send_size = jnp.minimum(local_src.size, pl_core.SEMAPHORE_MAX_VALUE)
    send_size = jnp.array(send_size, dtype=pl_core.SEMAPHORE_INTERPRET_DTYPE)
    src_sem_value = primitives._transform_semaphore(
        src_sem, src_sem_transforms, src_sem_aval
    )
    _, ret = state_discharge.transform_swap_array(
        src_sem, src_sem_transforms, src_sem_value[...] + send_size
    )
    return ret

  new_vals = (None,)  # src_val
  new_vals += (None,) * num_src_transform_vals
  new_vals += (do_discharge_dst() if dst_discharge else None,)  # dst_val
  new_vals += (None,) * num_dst_transform_vals
  new_vals += (do_discharge_dst_sem() if dst_sem_discharge else None,)  # dst_sem
  new_vals += (None,) * num_dst_sem_transforms
  if is_remote:
    new_vals += (do_discharge_src_sem() if src_sem_discharge else None,) # src_sem
    new_vals += (None,) * num_src_sem_transforms
    new_vals += (None,)  # device_id
  assert (len(new_vals) ==
          len(in_avals)), f"{len(new_vals), new_vals} != {len(in_avals)}"

  # If we didn't discharge everything we could we should keep writes
  # to the references that are left over.
  if not dst_discharge:
    sp.ref_set(dst_ref, None, do_discharge_dst(dst_ref=dst_ref[...]))
  if not dst_sem_discharge:
    sp.ref_set(dst_sem, None, do_discharge_dst_sem(dst_sem=dst_sem[...]))
  if is_remote and not src_sem_discharge:
    sp.ref_set(src_sem, None, do_discharge_src_sem(src_sem=src_sem[...]))

  return new_vals, []


state_discharge.register_partial_discharge_rule(dma_start_p)(dma_start_partial_discharge_rule)


dma_wait_p = jax_core.Primitive('dma_wait')
dma_wait_p.multiple_results = True

@dma_wait_p.def_effectful_abstract_eval
def _dma_wait_abstract_eval(*args, tree, device_id_type):
  del device_id_type
  (
      src_ref_aval,
      src_transforms_avals,
      dst_ref_aval,
      dst_transforms_avals,
      dst_sem_aval,
      dst_sem_transforms_avals,
      src_sem_aval,
      src_sem_transforms_avals,
      device_id_aval,
  ) = tree_util.tree_unflatten(tree, args)
  return [], _get_dma_effects(
      src_transforms_avals,
      dst_transforms_avals,
      dst_sem_transforms_avals,
      src_sem_aval,
  )

def _dma_wait_pp_eqn(eqn: jax_core.JaxprEqn,
                     context: jax_core.JaxprPpContext,
                     settings: jax_core.JaxprPpSettings):
  del settings
  invars = eqn.invars
  tree = eqn.params["tree"]
  (
      _,
      _,
      ref,
      transforms,
      sem,
      sem_transforms,
      _,
      _,
      _,
  ) = tree_util.tree_unflatten(tree, invars)
  return pp.concat([
      pp.text("dma_wait"),
      pp.text(" "),
      sp.pp_ref_transforms(context, ref, transforms),
      pp.text(" "),
      sp.pp_ref_transforms(context, sem, sem_transforms),
  ])

jax_core.pp_eqn_rules[dma_wait_p] = _dma_wait_pp_eqn

def dma_wait_partial_discharge_rule(should_discharge,
                                    in_avals, out_avals,
                                    *args, tree, device_id_type):
  # TODO(b/370563115): perform ref update in dma_wait discharge rule instead of dma_start
  del out_avals, device_id_type
  _, _, dst_ref, dst_ref_transforms, dst_sem, dst_sem_transforms, _, _, _ = (
      tree_util.tree_unflatten(tree, args))
  (_,
      src_ref_transforms_avals,
      _,
      dst_ref_transforms_avals,
      dst_sem_aval,
      dst_sem_transforms_avals,
      src_sem_aval,
      src_sem_transforms_avals,
      device_id_aval,
  ) = tree_util.tree_unflatten(tree, in_avals)

  # The only one we can discharge is the dst semaphore. The provided
  # buffers are only specified for their types and not their value so
  # it's completely irrelevant for us here if they are discharged.
  should_discharge_unflattened = tree_util.tree_unflatten(tree, should_discharge)
  if not should_discharge_unflattened[4]:
    return (None,) * len(in_avals), []

  num_sem_transforms = len(tree_util.tree_leaves(dst_sem_transforms_avals))
  num_transforms = len(tree_util.tree_leaves(dst_ref_transforms_avals))
  updates = state_discharge.transform_array(dst_ref, dst_ref_transforms)
  copy_size = jnp.minimum(updates.size, pl_core.SEMAPHORE_MAX_VALUE)
  copy_size = jnp.array(copy_size, dtype=pl_core.SEMAPHORE_INTERPRET_DTYPE)
  sem_value = primitives._transform_semaphore(dst_sem, dst_sem_transforms, dst_sem_aval)
  _, new_sem = state_discharge.transform_swap_array(
      dst_sem, dst_sem_transforms, sem_value - copy_size
  )
  new_vals = (None,)  # src_ref
  new_vals += (None,) * len(tree_util.tree_leaves(src_ref_transforms_avals))
  new_vals += (None,)  # ref
  new_vals += (None,) * num_transforms  # ref_transforms
  new_vals += (new_sem,)  # sem
  new_vals += (None,) * num_sem_transforms
  new_vals += (None,) * len(tree_util.tree_leaves(src_sem_aval))  # src_sem
  new_vals += (None,) * len(tree_util.tree_leaves(src_sem_transforms_avals))
  new_vals += (None,) * len(tree_util.tree_leaves(device_id_aval)) # device_id
  return new_vals, []
state_discharge.register_partial_discharge_rule(dma_wait_p)(dma_wait_partial_discharge_rule)

def _get_ref_and_transforms(ref):
  if isinstance(ref, state.TransformedRef):
    return ref.ref, ref.transforms
  return ref, ()


def make_async_copy(src_ref, dst_ref, sem) -> AsyncCopyDescriptor:
  """Issues a DMA copying from src_ref to dst_ref."""
  src_ref, src_transforms = _get_ref_and_transforms(src_ref)
  dst_ref, dst_transforms = _get_ref_and_transforms(dst_ref)
  sem, sem_transforms = _get_ref_and_transforms(sem)
  return AsyncCopyDescriptor(
      src_ref,
      src_transforms,
      dst_ref,
      dst_transforms,
      sem,
      sem_transforms,
      None,
      None,
      None,
      primitives.DeviceIdType.MESH,
  )


def async_copy(
    src_ref, dst_ref, sem, *, priority: int = 0
) -> AsyncCopyDescriptor:
  """Issues a DMA copying from src_ref to dst_ref."""
  copy_descriptor = make_async_copy(src_ref, dst_ref, sem)
  copy_descriptor.start(priority=priority)
  return copy_descriptor


def make_async_remote_copy(
    src_ref,
    dst_ref,
    send_sem,
    recv_sem,
    device_id: MultiDimDeviceId | IntDeviceId | None,
    device_id_type: primitives.DeviceIdType = primitives.DeviceIdType.MESH,
) -> AsyncCopyDescriptor:
  """Creates a description of a remote copy operation.

  Copies data from src_ref on the current device to dst_ref on the device
  specified by device_id. Both semaphores should be waited on using the
  descriptor on both source and target devices.

  Note that device_id can also refer to the current device.

  Args:
    src_ref: The source Reference.
    dst_ref: The destination Reference.
    send_sem: The semaphore on the source device.
    recv_sem: The semaphore on the destination device.
    device_id: The device id of the destination device. It could be a tuple, or
      a dictionary specifying the communication axis and destination index.
    device_id_type: The type of the device id.

  Returns:
    An AsyncCopyDescriptor.
  """
  src_ref, src_transforms = _get_ref_and_transforms(src_ref)
  send_sem, send_sem_transforms = _get_ref_and_transforms(send_sem)
  dst_ref, dst_transforms = _get_ref_and_transforms(dst_ref)
  recv_sem, recv_sem_transforms = _get_ref_and_transforms(recv_sem)
  if device_id_type == primitives.DeviceIdType.LOGICAL:
    assert not isinstance(
        device_id, tuple | dict
    ), "LOGICAL device_id_type does not support device_id as a tuple or dict."

  return AsyncCopyDescriptor(
      src_ref,
      src_transforms,
      dst_ref,
      dst_transforms,
      recv_sem,
      recv_sem_transforms,
      send_sem,
      send_sem_transforms,
      device_id,
      device_id_type=device_id_type,
  )


def async_remote_copy(
    src_ref,
    dst_ref,
    send_sem,
    recv_sem,
    device_id,
    device_id_type: primitives.DeviceIdType = primitives.DeviceIdType.MESH,
) -> AsyncCopyDescriptor:
  copy_descriptor = make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem,
                                           device_id, device_id_type)
  copy_descriptor.start()
  return copy_descriptor


get_barrier_semaphore_p = jax_core.Primitive('get_barrier_semaphore')

@get_barrier_semaphore_p.def_abstract_eval
def _get_barrier_semaphore_abstract_eval():
  return state.AbstractRef(
      jax_core.ShapedArray((), pl_core.BarrierSemaphore()),
      tpu_core.MemorySpace.SEMAPHORE,
  )

def get_barrier_semaphore():
  """Returns a barrier semaphore.

  This function returns a barrier semaphore based on the collective_id of the
  current pallas kernel.

  It's very important that the semaphore is wait-ed back down to 0, or else the
  semaphores will become corrupted.

  It's also very important that the collective_id is different for each pallas
  kernel with communication. E.g. if you have two pallas kernels, one that syncs
  across the X axis of the device mesh and the second that syncs across the Y
  axis, they must have different collective_ids.
  However it is legal for two kernels that perform the same synchronization
  pattern (e.g. only communicating with neighbours on the same mesh axis)
  to share a collective_id. However, if in doubt, prefer not sharing
  collective_ids, as doing so incorrectly can lead to silent data corruption or
  crashes.
  Note that reusing the same collective_id doesn't guarantee that the same
  semaphore is provided by XLA.
  """
  return get_barrier_semaphore_p.bind()

delay_p = jax_core.Primitive("delay")
delay_p.multiple_results = True


@delay_p.def_abstract_eval
def _delay_abstract_eval(nanos):
  del nanos
  return []


def delay(nanos):
  """Delays vector execution for the given number of nanosconds."""
  delay_p.bind(nanos)


# RNG Ops
prng_seed_p = jax_core.Primitive("prng_seed")
prng_seed_p.multiple_results = True


@prng_seed_p.def_abstract_eval
def _prng_seed_abstract_eval(*_):
  return []


def prng_seed(*seeds: int | jax.Array) -> None:
  """Sets the seed for PRNG.

  Args:
    seeds: One or more integer seeds for setting the PRNG seed. If
      more than one seed is passed in, the seed material will be
      mixed before setting the internal PRNG state.
  """
  prng_seed_p.bind(*seeds)

prng_random_bits_p = jax_core.Primitive(
    'prng_random_bits')


@prng_random_bits_p.def_abstract_eval
def _prng_random_bits_abstract_eval(*, shape):
  return jax_core.ShapedArray(shape, jnp.dtype("int32"))


def prng_random_bits(shape):
  return prng_random_bits_p.bind(shape=shape)

# PRNG wrap/unwrap ops.
# We cannot use JAX's key_data and wrap_key_data because they return
# vectors, and Pallas keys are represented as lists of scalars.

split_key_p = jax_core.Primitive("prng_split")
split_key_p.multiple_results = True


@split_key_p.def_abstract_eval
def _split_key_scalar_abstract_eval(seed):
  key_shape = seed.dtype._impl.key_shape
  if len(key_shape) != 2 or key_shape[0] != 1:
    raise ValueError(f"Key shape must be (1, N), got {key_shape}")
  return [jax_core.ShapedArray((), jnp.dtype("uint32"))] * key_shape[1]


def unwrap_pallas_seed(seed):
  """Splits a PRNG key into it's scalar components."""
  return split_key_p.bind(seed)


join_key_p = jax_core.Primitive("prng_join")


@join_key_p.def_abstract_eval
def _join_key_scalar_abstract_eval(*seeds, impl):
  if len(impl.key_shape) != 2 or impl.key_shape[0] != 1:
    raise ValueError(f"Key shape must be (1, N), got {impl.key_shape}")
  if len(seeds) != impl.key_shape[1]:
    raise ValueError(
        f"Number of seeds must match key shape, got {len(seeds)}"
        f" != {impl.key_shape[1]}."
    )
  return jax_core.ShapedArray((), dtype=jax_prng.KeyTy(impl))


def wrap_pallas_seed(*seeds, impl):
  """Joins scalar into a single PRNG key."""
  impl = jax_random.resolve_prng_impl(impl)
  return join_key_p.bind(*seeds, impl=impl)


def with_memory_space_constraint(
    x: jax.Array, memory_space: Any
) -> jax.Array:
  """Constrains the memory space of an array.

  This primitive does not change the value of `x`, but it constrains the
  memory space where it should be allocated. This is useful to force
  Pallas to allocate an array in a specific memory space.

  As of now, this only operates on the inputs pallas_calls, as in you can
  apply this to the arguments of a pallas_call and it will constrain them, but
  other operations will not respect this constraint.

  Args:
    x: The array to constrain.
    memory_space: The memory space to constrain to.

  Returns:
    The array `x` with the memory space constraint.
  """
  if memory_space in {tpu_core.MemorySpace.ANY, pl_core.MemorySpace.ANY}:
    return x
  if memory_space not in {
      tpu_core.MemorySpace.HBM,
      tpu_core.MemorySpace.VMEM,
      tpu_core.MemorySpace.SMEM,
  }:
    raise NotImplementedError(
        "with_memory_space_constraint only supports HBM, VMEM and SMEM."
    )
  return pl_core.with_memory_space_constraint_p.bind(
      x, memory_space=memory_space)


def load(ref: Ref, *, mask: jax.Array | None = None) -> jax.Array:
  """Loads an array from the given ref.

  If ``mask`` is not specified, this function has the same semantics as
  ``ref[idx]`` in JAX.

  Args:
    ref: The ref to load from.
    mask: An optional boolean mask specifying which indices to load.

  Returns:
    The loaded array.
  """
  return primitives.load(ref, None, mask=mask)


def store(ref: Ref, val: jax.Array, *, mask: jax.Array | None = None) -> None:
  """Stores a value to the given ref.

  If ``mask`` is not specified, this function has the same semantics as
  ``ref[idx] = val`` in JAX.

  Args:
    ref: The ref to store to.
    val: The value to store.
    mask: An optional boolean mask specifying which indices to store.
  """
  return primitives.store(ref, None, val, mask=mask)
