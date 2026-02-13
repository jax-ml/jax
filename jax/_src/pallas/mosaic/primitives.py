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
from jax._src import effects
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


def repeat(x: jax.Array, repeats: int, axis: int) -> jax.Array:
  axis = util.canonicalize_axis(axis, x.ndim)
  reps = [repeats if i == axis else 1 for i in range(x.ndim)]
  return jnp.tile(x, reps)


bitcast_p = jax_core.Primitive("bitcast")


def bitcast(x: jax.Array, ty: DTypeLike) -> jax.Array:
  ty = dtypes.check_and_canonicalize_user_dtype(ty)
  if len(x.shape) < 2:
    raise ValueError("Not implemented: bitcast 1D")
  src_bitwidth = dtypes.itemsize_bits(x.dtype)
  dst_bitwidth = dtypes.itemsize_bits(ty)
  if x.shape[-2] * src_bitwidth % dst_bitwidth:
    raise ValueError(
        "Not implemented: the 2nd minor dim can not be perfectly packed or"
        " unpacked"
    )
  return bitcast_p.bind(x, ty=ty)


@bitcast_p.def_abstract_eval
def _bitcast_abstract_eval(x, *, ty):
  shape = list(x.shape)
  src_bitwidth = dtypes.itemsize_bits(x.dtype)
  dst_bitwidth = dtypes.itemsize_bits(ty)
  shape[-2] = shape[-2] * src_bitwidth // dst_bitwidth
  return jax_core.ShapedArray(shape, ty)


def _bitcast_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, ty):
  def _bitcast(x):
    src_bitwidth = dtypes.itemsize_bits(x.dtype)
    dst_bitwidth = dtypes.itemsize_bits(ty)
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
    x: jax.Array,
    shift: jax.Array | int,
    axis: int,
    *,
    stride: int | None = None,
    stride_axis: int | None = None,
) -> jax.Array:
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
      return _dma_flatten(
          self.dst_ref,
          self.dst_transforms,
          self.src_ref,
          self.src_transforms,
          self.src_sem,
          self.src_sem_transforms,
          self.dst_sem,
          self.dst_sem_transforms,
          self.device_id,
      )
    else:
      return _dma_flatten(
          self.src_ref,
          self.src_transforms,
          self.dst_ref,
          self.dst_transforms,
          self.dst_sem,
          self.dst_sem_transforms,
          self.src_sem,
          self.src_sem_transforms,
          self.device_id,
      )

  def start(self, priority: int = 0, *, add: bool = False):
    self._used = True
    flat_args, tree = self._get_args_and_tree()
    dma_start_p.bind(
        *flat_args,
        tree=tree,
        device_id_type=self.device_id_type,
        priority=priority,
        add=add,
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


def _dma_flatten(
    src_ref,
    src_transforms,
    dst_ref,
    dst_transforms,
    dst_sem,
    dst_sem_transforms,
    src_sem,
    src_sem_transforms,
    device_id,
):
  return tree_util.tree_flatten((
      src_ref,
      _maybe_wrap_transformed_refs(src_transforms),
      dst_ref,
      _maybe_wrap_transformed_refs(dst_transforms),
      dst_sem,
      dst_sem_transforms,
      src_sem,
      src_sem_transforms,
      device_id,
  ))


def _dma_unflatten(tree, flat_args):
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
  ) = tree_util.tree_unflatten(tree, flat_args)
  return (
      src_ref,
      _maybe_unwrap_transformed_refs(src_transforms),
      dst_ref,
      _maybe_unwrap_transformed_refs(dst_transforms),
      dst_sem,
      dst_sem_transforms,
      src_sem,
      src_sem_transforms,
      device_id,
  )


def _maybe_wrap_transformed_refs(transforms: Any) -> Any:
  return jax.tree.map(
      lambda obj: _maybe_wrap_transformed_refs(TransformedRefTree.wrap(obj))
      if isinstance(obj, state.TransformedRef)
      else obj,
      transforms,
  )


def _maybe_unwrap_transformed_refs(transforms: Any) -> Any:
  return jax.tree.map(
      lambda obj: _maybe_unwrap_transformed_refs(obj.unwrap())
      if isinstance(obj, TransformedRefTree)
      else obj,
      transforms,
      is_leaf=lambda obj: isinstance(obj, TransformedRefTree),
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TransformedRefTree(state.TransformedRef):
  """A PyTree wrapper for a ``TransformedRef``.

  The wrapper is necessary to support the case when a ``TransformedRef`` is
  indexed with other ``TransformedRef``s.
  """

  @classmethod
  def wrap(cls, ref: state.TransformedRef) -> TransformedRefTree:
    return cls(ref.ref, ref.transforms)

  def unwrap(self) -> state.TransformedRef:
    return state.TransformedRef(self.ref, self.transforms)


def _get_dma_effects(
    src_transforms_avals,
    dst_transforms_avals,
    dst_sem_transforms_avals,
    src_sem_aval,
    device_id_aval,
    device_id_type,
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
  if device_id_aval is not None:
    if device_id_type is primitives.DeviceIdType.MESH and isinstance(
        device_id_aval, dict
    ):
      for k in device_id_aval:
        if not isinstance(k, tuple):
          k = (k,)
        for k_ in k:
          effs.add(jax_core.NamedAxisEffect(k_))
  return effs


dma_start_p = jax_core.Primitive('dma_start')
dma_start_p.multiple_results = True

def _dma_is_high(*avals, **params):
  return any(aval.is_high for aval in avals)

dma_start_p.is_high = _dma_is_high  # type: ignore[method-assign]

def _dma_start_to_lojax(*args, tree, device_id_type, priority, add):
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
  src_ref_aval = jax_core.get_aval(src_ref)
  dst_ref_aval = jax_core.get_aval(dst_ref)
  if not (src_ref_aval.is_high and dst_ref_aval.is_high):
    raise NotImplementedError("dma_start not implemented in LoJAX yet.")
  dst_sem_aval = jax_core.get_aval(dst_sem)
  if dst_sem_aval.is_high:
    raise NotImplementedError("dma_start not implemented in LoJAX yet.")
  if src_sem is not None:
    if jax_core.get_aval(src_sem).is_high:
      raise NotImplementedError("dma_start not implemented in LoJAX yet.")
  src_transformed_ref = state.TransformedRef(src_ref, src_transforms)
  dst_transformed_ref = state.TransformedRef(dst_ref, dst_transforms)
  if src_sem is not None:
    src_sem = state.TransformedRef(src_sem, src_sem_transforms)
  dst_sem = state.TransformedRef(dst_sem, dst_sem_transforms)

  src_ref_aval.inner_aval.dma_start(
      src_transformed_ref,
      dst_transformed_ref,
      src_sem,
      dst_sem,
      device_id=device_id,
      priority=priority,
      device_id_type=device_id_type,
      add=add
  )
  return []
dma_start_p.to_lojax = _dma_start_to_lojax

@dma_start_p.def_effectful_abstract_eval
def _dma_start_abstract_eval(*args, tree, device_id_type, priority, add):
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
  ) = _dma_unflatten(tree, args)
  if not all(isinstance(x, state.AbstractRef) for x in [
      src_ref_aval, dst_ref_aval, dst_sem_aval]):
    raise ValueError(
        "DMA source/destination/semaphore arguments must be Refs.")
  dst_sem_shape = dst_sem_aval.shape
  if dst_sem_transforms_avals:
    dst_sem_shape = dst_sem_transforms_avals[-1].get_indexer_shape()
  if dst_sem_shape:
    raise ValueError(
        f"Cannot signal on a non-()-shaped semaphore: {dst_sem_shape}"
    )
  if src_sem_aval is not None:
    if not isinstance(src_sem_aval, state.AbstractRef):
      raise ValueError(
          "DMA source semaphore must be a Ref.")
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
      device_id_aval,
      device_id_type,
  )

def _dma_start_pp_eqn(eqn: jax_core.JaxprEqn,
                      context: jax_core.JaxprPpContext,
                      settings: jax_core.JaxprPpSettings):
  invars = eqn.invars
  tree = eqn.params["tree"]
  priority = eqn.params["priority"]
  add = eqn.params["add"]
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
  ) = _dma_unflatten(tree, invars)
  del src_sem_transforms
  # TODO(sharadmv): pretty print source semaphores and device id
  if src_sem or device_id:
    return jax_core._pp_eqn(eqn, context, settings)
  return pp.concat([
      pp.text(f"dma_start(p{priority}{', add' if add else ''})"),
      pp.text(" "),
      sp.pp_ref_transforms(context, src_ref, src_transforms),
      pp.text(" -> "),
      sp.pp_ref_transforms(context, dst_ref, dst_transforms),
      pp.text(" "),
      sp.pp_ref_transforms(context, dst_sem, dst_sem_transforms),
  ])

jax_core.pp_eqn_rules[dma_start_p] = _dma_start_pp_eqn


def dma_start_partial_discharge_rule(
    should_discharge, in_avals, out_avals, *args, tree, device_id_type,
    priority, add
):
  # Note: we ignore the DMA priority in discharge rules.
  del priority
  if add:
    raise NotImplementedError(
        "DMA partial discharge add=True not yet implemented.")
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
  ) = _dma_unflatten(tree, args)
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
  ) = _dma_unflatten(tree, in_avals)
  del out_avals

  (
      _,
      _,
      dst_discharge,
      _,
      dst_sem_discharge,
      _,
      *maybe_src_sem_discharge,
  ) = _dma_unflatten(tree, should_discharge)
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

dma_wait_p.is_high = _dma_is_high  # type: ignore[method-assign]

def _dma_wait_to_lojax(*args, tree, device_id_type):
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
  src_ref_aval = jax_core.get_aval(src_ref)
  dst_ref_aval = jax_core.get_aval(dst_ref)
  if not (src_ref_aval.is_high and dst_ref_aval.is_high):
    raise NotImplementedError("dma_wait not implemented in LoJAX yet.")
  dst_sem_aval = jax_core.get_aval(dst_sem)
  if dst_sem_aval.is_high:
    raise NotImplementedError("dma_wait not implemented in LoJAX yet.")
  if src_sem is not None:
    if jax_core.get_aval(src_sem).is_high:
      raise NotImplementedError("dma_wait not implemented in LoJAX yet.")
  src_transformed_ref = state.TransformedRef(src_ref, src_transforms)
  dst_transformed_ref = state.TransformedRef(dst_ref, dst_transforms)
  if src_sem is not None:
    src_sem = state.TransformedRef(src_sem, src_sem_transforms)
  dst_sem = state.TransformedRef(dst_sem, dst_sem_transforms)
  src_ref_aval.inner_aval.dma_wait(
      src_transformed_ref,
      dst_transformed_ref,
      src_sem,
      dst_sem,
      device_id=device_id,
      device_id_type=device_id_type,
  )
  return []
dma_wait_p.to_lojax = _dma_wait_to_lojax

@dma_wait_p.def_effectful_abstract_eval
def _dma_wait_abstract_eval(*args, tree, device_id_type):
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
  ) = _dma_unflatten(tree, args)
  return [], _get_dma_effects(
      src_transforms_avals,
      dst_transforms_avals,
      dst_sem_transforms_avals,
      src_sem_aval,
      device_id_aval,
      device_id_type,
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
  ) = _dma_unflatten(tree, invars)
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
      _dma_unflatten(tree, args)
  )
  (
      _,
      src_ref_transforms_avals,
      _,
      dst_ref_transforms_avals,
      dst_sem_aval,
      dst_sem_transforms_avals,
      src_sem_aval,
      src_sem_transforms_avals,
      device_id_aval,
  ) = _dma_unflatten(tree, in_avals)

  # The only one we can discharge is the dst semaphore. The provided
  # buffers are only specified for their types and not their value so
  # it's completely irrelevant for us here if they are discharged.
  should_discharge_unflattened = _dma_unflatten(tree, should_discharge)
  if not should_discharge_unflattened[4]:
    return (None,) * len(in_avals), []

  num_sem_transforms = len(tree_util.tree_leaves(dst_sem_transforms_avals))
  num_transforms = len(tree_util.tree_leaves(dst_ref_transforms_avals))
  updates = state_discharge.transform_array(dst_ref[...], dst_ref_transforms)
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
  """Creates a description of an asynchronous copy operation.

  Args:
    src_ref: The source Reference.
    dst_ref: The destination Reference.
    sem: The semaphore used to track completion of the copy.

  Returns:
    An AsyncCopyDescriptor.
  """
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
    src_ref, dst_ref, sem, *, priority: int = 0, add: bool = False,
) -> AsyncCopyDescriptor:
  """Issues a DMA copying from src_ref to dst_ref."""
  copy_descriptor = make_async_copy(src_ref, dst_ref, sem)
  copy_descriptor.start(priority=priority, add=add)
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
  """Issues a remote DMA copying from src_ref to dst_ref."""
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


# RNG Ops
prng_seed_p = jax_core.Primitive("prng_seed")
prng_seed_p.multiple_results = True


class PRNGEffect(effects.Effect):
  pass
prng_effect = PRNGEffect()
effects.control_flow_allowed_effects.add_type(PRNGEffect)
pl_core.kernel_local_effects.add_type(PRNGEffect)


@prng_seed_p.def_effectful_abstract_eval
def _prng_seed_abstract_eval(*_):
  return [], {prng_effect}


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


stochastic_round_p = jax_core.Primitive("stochastic_round")


def stochastic_round(x, random_bits, *, target_dtype):
  return stochastic_round_p.bind(x, random_bits, target_dtype=target_dtype)


@stochastic_round_p.def_abstract_eval
def _stochastic_round_abstract_eval(x, random_bits, *, target_dtype):
  if random_bits.shape != x.shape:
    raise ValueError(
        "The shape of `random_bits` must match the shape of `x` for "
        f"stochastic_round, but got {random_bits.shape} and {x.shape}"
    )
  if random_bits.dtype != jnp.dtype("uint32"):
    raise ValueError(
        "The dtype of `random_bits` must be uint32 for stochastic_round, "
        f"but got {random_bits.dtype}"
    )
  return jax_core.ShapedArray(x.shape, target_dtype)


def _get_elementwise_packing_factor(unpacked_dtype, packed_dtype):
  unpacked_bitwidth = dtypes.itemsize_bits(unpacked_dtype)
  packed_bitwidth = dtypes.itemsize_bits(packed_dtype)
  if unpacked_bitwidth % packed_bitwidth != 0:
    raise ValueError(
        "Unpacked bitwidth must be a multiple of packed bitwidth, got "
        f"{unpacked_bitwidth} and {packed_bitwidth}"
    )
  return unpacked_bitwidth // packed_bitwidth

pack_elementwise_p = jax_core.Primitive("pack_elementwise")


def pack_elementwise(xs, *, packed_dtype):
  return pack_elementwise_p.bind(*xs, packed_dtype=packed_dtype)


@pack_elementwise_p.def_abstract_eval
def _pack_elementwise_abstract_eval(*xs, packed_dtype):
  if not xs:
    raise ValueError("At least one source is required")
  first = xs[0]
  if not all(x.shape == first.shape for x in xs):
    raise ValueError("All sources must have the same shape")
  if not all(x.dtype == first.dtype for x in xs):
    raise ValueError("All sources must have the same dtype")
  if not (first.dtype == jnp.float32 and packed_dtype == jnp.bfloat16) and not (
      jnp.issubdtype(first.dtype, jnp.integer)
      and jnp.issubdtype(packed_dtype, jnp.integer)
  ):
    raise ValueError(
        "Only f32 -> bf16 and int -> int are supported. Got"
        f" {first.dtype} and {packed_dtype}"
    )
  packing_factor = _get_elementwise_packing_factor(first.dtype, packed_dtype)
  if len(xs) != packing_factor:
    raise ValueError(
        "The number of sources must match the packing factor "
        f"({packing_factor}), got {len(xs)}"
    )
  out_dtype = jnp.dtype(f"uint{dtypes.itemsize_bits(first.dtype)}")
  return jax_core.ShapedArray(first.shape, out_dtype)


unpack_elementwise_p = jax_core.Primitive("unpack_elementwise")


def unpack_elementwise(x, *, index, packed_dtype, unpacked_dtype):
  return unpack_elementwise_p.bind(
      x, index=index, packed_dtype=packed_dtype, unpacked_dtype=unpacked_dtype
  )


@unpack_elementwise_p.def_abstract_eval
def _unpack_elementwise_abstract_eval(x, *, index, packed_dtype, unpacked_dtype):
  if x.dtype != jnp.uint32:
    raise ValueError(f"Source must be uint32, got {x.dtype}")
  packing_factor = _get_elementwise_packing_factor(unpacked_dtype, packed_dtype)
  if index < 0 or index >= packing_factor:
    raise ValueError(
        f"Index {index} is out of bounds for packing factor {packing_factor}")
  return jax_core.ShapedArray(x.shape, unpacked_dtype)


def with_memory_space_constraint(
    x: jax.Array, memory_space: Any
) -> jax.Array:
  """Constrains the memory space of an array.

  This primitive does not change the value of ``x``, but it constrains the
  memory space where it should be allocated. This is useful to force
  Pallas to allocate an array in a specific memory space.

  As of now, this only operates on the inputs pallas_calls, as in you can
  apply this to the arguments of a pallas_call and it will constrain them, but
  other operations will not respect this constraint.

  Args:
    x: The array to constrain.
    memory_space: The memory space to constrain to.

  Returns:
    The array ``x`` with the memory space constraint.
  """
  if memory_space is pl_core.MemorySpace.ANY:
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


touch_p = jax_core.Primitive("add_dependency")
touch_p.multiple_results = True


def touch(ref: jax.Array | state.TransformedRef) -> None:
  """Adds a fake read-write dependency to the given ref."""
  ref_leaves = jax.tree.leaves(ref)
  ref_leaves = [ref.ref if isinstance(ref, state.TransformedRef) else ref
                for ref in ref_leaves]
  for ref in ref_leaves:
    touch_p.bind(ref)


@touch_p.def_effectful_abstract_eval
def _touch_abstract_eval(ref: jax.Array):
  return [], {state.ReadEffect(0), state.WriteEffect(0)}


trace_value_p = jax_core.Primitive("trace_value")
trace_value_p.multiple_results = True


def trace_value(label: str, value: jax.Array) -> None:
  """Emit a scalar value to the current xprof trace scope.

  This appends a dynamic scalar value to the enclosing trace region.
  The value will appear in xprof trace viewer associated with the trace event.

  Args:
    label: A string label for this value in xprof.
    value: A scalar i32 or f32 value to emit.

  Example:
    # Inside a Pallas kernel:
    x  = jnp.sum(y > 0)
    pltpu.trace_value("my_x", x)
  """
  trace_value_p.bind(value, label=label)


class TraceEffect(effects.Effect):
  pass


trace_effect = TraceEffect()
effects.control_flow_allowed_effects.add_type(TraceEffect)
pl_core.kernel_local_effects.add_type(TraceEffect)


@trace_value_p.def_effectful_abstract_eval
def _trace_value_abstract_eval(value, *, label):
  del label
  if value.shape:
    raise ValueError(
        f"trace_value requires a scalar value, got shape {value.shape}"
    )
  if value.dtype not in (jnp.int32, jnp.float32):
    raise ValueError(f"trace_value requires i32 or f32, got {value.dtype}")
  return [], {trace_effect}


class MXUEffect(effects.Effect):
  __str__ = lambda self: "MXU"
mxu_effect = MXUEffect()
effects.control_flow_allowed_effects.add_type(MXUEffect)
pl_core.kernel_local_effects.add_type(MXUEffect)


matmul_push_rhs_p = jax_core.Primitive("matmul_push_rhs")
matmul_push_rhs_p.multiple_results = True


def matmul_push_rhs(
    rhs: jax.Array, staging_register: int, mxu_index: int
) -> None:
  """Prepares the RHS for a matrix multiplication in the chosen MXU.

  Each MXU has an independent set of staging registers.

  ```{warning}
  It is not allowed to push to the same staging register twice. Once
  the RHS is prepared, it must be loaded into the MXU using `matmul_acc_lhs`
  before it can be used again.
  ```

  ```{warning}
  The kernel must not leave any data in the staging registers upon exit.
  ```

  Args:
    rhs: The right-hand side operand. Must be 256x256.
    staging_register: The staging register to use.
    mxu_index: The MXU to use.
  """
  matmul_push_rhs_p.bind(
      rhs, staging_register=staging_register, mxu_index=mxu_index
  )


@matmul_push_rhs_p.def_effectful_abstract_eval
def _matmul_push_rhs_abstract_eval(ref: jax.Array, **_):
  del ref  # Unused.
  return [], {mxu_effect}


matmul_acc_lhs_p = jax_core.Primitive("matmul_acc_lhs")
matmul_acc_lhs_p.multiple_results = True


def matmul_acc_lhs(
    acc_addr: int,
    lhs: jax.Array,
    mxu_index: int,
    load_staged_rhs: int | None = None,
) -> None:
  """Performs a matrix multiplication in the chosen MXU.

  If `load_staged_rhs` is not None, the previously pushed RHS will be loaded
  from the given staging register _before_ the matrix multiplication begins.
  The results of the multiplication are accumulated into the specified
  accumulator slice.

  Args:
    acc_addr: The base address of the accumulator slice used for results.
    lhs: The left-hand side operand. Must be M x 256. For M divisible by the
      number of sublanes multiplied by datatype packing.
    mxu_index: The MXU to use.
    load_staged_rhs: The staging register to load the RHS from. If None, the RHS
      is not loaded from staging and the matmul will reuse the existing one.
  """
  # This is a common error. You might intend to say to load the staged RHS, but
  # True is equivalent to saying "load the staged RHS FROM REGISTER 1", which is
  # probably not what you intended.
  if isinstance(load_staged_rhs, bool):
    raise TypeError("load_staged_rhs must be an integer or None.")
  matmul_acc_lhs_p.bind(
      lhs,
      acc_addr=acc_addr,
      mxu_index=mxu_index,
      load_staged_rhs=load_staged_rhs,
  )


@matmul_acc_lhs_p.def_effectful_abstract_eval
def _matmul_acc_lhs_abstract_eval(lhs: jax.Array, **_):
  del lhs  # Unused.
  return [], {mxu_effect}


matmul_pop_p = jax_core.Primitive("matmul_pop")


def matmul_pop(
    acc_addr: int,
    shape: tuple[int, int],
    dtype: jax.typing.DTypeLike,
    mxu_index: int,
):
  """Returns the result of a matrix multiplication from the chosen MXU and zeroes the accumulator.

  If the result is not ready yet (the MXU is still busy), the operation blocks.

  ```{warning}
  The kernel must not leave any data in the accumulator upon exit.
  ```

  Args:
    acc_addr: The base address of the popped accumulator slice.
    shape: The shape of the result.
    dtype: The dtype of the result.
    mxu_index: The MXU to use.
  """
  return matmul_pop_p.bind(
      acc_addr=acc_addr,
      shape=shape,
      mxu_index=mxu_index,
      dtype=jnp.dtype(dtype),
  )


@matmul_pop_p.def_effectful_abstract_eval
def _matmul_pop_abstract_eval(*, shape, dtype, **_):
  if dtype not in map(jnp.dtype, [jnp.float32, jnp.int32]):
    raise ValueError(
        f"Only float32 and int32 accumulators are supported, got {dtype}"
    )
  return jax_core.ShapedArray(shape, dtype), {mxu_effect}
