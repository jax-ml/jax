# Copyright 2026 The JAX Authors.
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

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import math
from typing import Any

import jax
from jax import lax
from jax._src import core as jax_core
from jax._src import source_info_util
from jax._src.pallas import primitives
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic_gpu import core as mosaic_gpu_core
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax._src.pallas.mosaic_gpu.interpret import gpu_callbacks
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.state import types as state_types
from jax._src.util import (safe_zip, split_list)
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp


@dataclasses.dataclass(init=False, frozen=True)
class DeviceInfo:
  """Information about the device that is being interpreted."""

  # The indices along each axis of the device being interpreted.
  axis_indices: Mapping[jax_core.AxisName, int]
  # The size of each axis in the mesh of all (SMPD) devices.
  axis_sizes: Mapping[jax_core.AxisName, int]

  def __init__(self):
    # Since this class is frozen, we must use `object.__setattr__` to set the
    # attributes.
    object.__setattr__(self, "axis_sizes", jax_core.get_axis_env().axis_sizes)
    object.__setattr__(
        self,
        "axis_indices",
        {k: lax.axis_index(k) for k in self.axis_sizes.keys()},
    )

  @functools.cached_property
  def device_id(self) -> int:
    """Computes the logical ID of the device being interpreted."""
    return interpret_utils.device_coords_to_logical_id(
        tuple(self.axis_indices.values()), self.axis_sizes, self.axis_indices
    )

  @functools.cached_property
  def num_devices(self) -> int:
    """Computes the number of (SPMD) devices."""
    return math.prod(self.axis_sizes.values())


def _raise_if_unsupported_memory_space(
    space: mosaic_gpu_core.MemorySpace | None,
):
  # TODO(nrink): Support more memory spaces.
  if space is not None and space not in [
      mosaic_gpu_core.MemorySpace.GMEM,
      mosaic_gpu_core.MemorySpace.SMEM,
  ]:
    raise NotImplementedError(f"Unsupported memory space: {space}")


def _raise_if_unsupported_collective_axes(
    mesh: plgpu.Mesh | None,
    is_collective_by_thread_cluster_axis: tuple[bool, ...],
):
  if not mesh or not mesh.thread_name:
    if any(is_collective_by_thread_cluster_axis):
      raise ValueError(
          "Requesting collective allocations, but no explicit thread axis"
          " specified."
      )
  else:
    # Note that the leading entries in `is_collective_by_thread__cluster_axis`
    # correspond to the cluster axes, while the last entry corresponds to the
    # thread axis within a block.
    *is_collective_by_cluster_axis, is_thread_axis_collective = (
        is_collective_by_thread_cluster_axis
    )
    if any(is_collective_by_cluster_axis):
      raise ValueError(
          "Collective allocations along cluster axes are not supported."
      )
    if not is_thread_axis_collective:
      raise ValueError(
          "Scoped allocation must have the thread axis in its collective axes."
      )


# TODO(nrink): Try unifying this function with `_extract_barrier_slice_base`
# from `jax._src.pallas.mosaic_gpu.primitives`.
def _get_index_for_barrier_allocation_key(
    transforms_treedef, transforms_leaves,
) -> indexing.DimIndexer | None:
  # TODO(nrink): The working out of `transforms` and the returned index below
  # may need tidying up. Specifically, GPU interpret mode should correctly
  # support legal ways to index into barriers. (Here, 'legal' is to be read as
  # 'allowed by the Pallas GPU semantics'.)
  transforms = jax.tree.unflatten(transforms_treedef, transforms_leaves)

  if not transforms:
    return None
  if not hasattr(transforms, "__len__") or len(transforms) != 1:
    raise NotImplementedError(
        f"Indexing barrier with {transforms} not supported in GPU interpret"
        " mode"
    )
  if not isinstance(transforms[0], indexing.NDIndexer):
    raise ValueError(f"Expected an `NDIndexer`, but got {transforms[0]}")
  if len(transforms[0].indices) != 1:
    raise ValueError(
        f"Expected a singleton index, but got {transforms[0].indices}"
    )
  return transforms[0].indices[0]


def _get_barrier_allocation_key_from_inval(
    inval, transforms_treedef, transforms_leaves
) -> jax.Array:
  # `inval` is expected to correspond to a barrier. Since we are interpreting,
  # `inval` will in fact contain the allocation key (which is a Jax array) for
  # the barrier.
  allocation_key_as_array = inval

  # Assert to check internal consistency: `allocation_key_as_array` should be
  # a 2-dim array (and the size of the first dimension equals the
  # `num_barriers` parameter from when the barrier was allocated).
  assert len(allocation_key_as_array.shape) == 2
  num_barriers = allocation_key_as_array.shape[0]

  index = _get_index_for_barrier_allocation_key(
      transforms_treedef, transforms_leaves
  )

  if index is None:
    if num_barriers != 1:
      raise ValueError(
          "Attempting to operate on barrier without indexing, but"
          f" `num_barriers = {num_barriers}`"
      )
    return allocation_key_as_array[0]
  else:
    return allocation_key_as_array[index]


def _get_num_threads_sharing_collective_allocation(
    axes_dims: tuple[int, ...],
    is_last_thread_axis_collective: bool,
) -> int:
  """Returns the number of threads that share a collective allocation."""
  if is_last_thread_axis_collective:
    return axes_dims[-1]
  else:
    return 1


_SENTINEL = jnp.inf


def apply_unswizzle_and_untile(
    transforms: tuple[state_types.Transform, ...],
    aval: jax_core.AbstractValue,
) -> jax_core.AbstractValue:
  if not all(isinstance(t, (mosaic_gpu_core.UnswizzleRef,
                            mosaic_gpu_core.UntilingTransform))
             for t in transforms):
    raise ValueError("Unsupported transforms:", transforms)
  return state_types.TransformedRef(aval, transforms).type


@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxprInterpreter:
  """Interprets a jaxpr by replacing memory operations with (GPU) callbacks."""

  grid_point_coords: jax.Array
  cluster_dims: tuple[int, ...]

  # The (flat) thread ID for the thread that this interpreter instance is
  # executing. On each simulated GPU, and for each grid point, we execute the
  # following number of threads concurrently (with concurrent instances of
  # `JaxprInterpreter`):
  #
  #   num_threads = math.prod(self.cluster_dims) * self.num_threads_per_block
  #
  # Hence, `thread_id` must be a number in [0, num_threads - 1].
  #
  # The `thread_id` maps to a thread's coordinates along the cluster and thread
  # axes as follows:
  #
  #   all_thread_coords = (*self.cluster_coords, self.thread_id_in_block),
  #
  # where the `self.thread_id_in_block` is the minor-most coordinate, and
  # `self.cluster_coords` are in major-to-minor order.
  thread_id: jax.Array

  mesh: plgpu.Mesh | None
  device_info: DeviceInfo
  compiler_params: Mapping[str, Any]
  interpret_params: InterpretGPUParams

  @functools.cached_property
  def num_threads_per_block(self) -> int:
    if self.mesh is None or self.mesh.num_threads is None:
      return 1
    else:
      return self.mesh.num_threads

  @functools.cached_property
  def thread_id_in_block(self) -> jax.Array:
    return lax.rem(self.thread_id, jnp.int32(self.num_threads_per_block))

  @functools.cached_property
  def cluster_coords(self) -> tuple[int, ...]:
    thread_id = self.thread_id // self.num_threads_per_block
    return interpret_utils.get_indices(self.cluster_dims, thread_id)

  @functools.cached_property
  def thread_cluster_shape(self) -> tuple[int, ...]:
    """Returns the number of threads along the cluster axes *and* within a block."""
    return self.cluster_dims + (self.num_threads_per_block,)

  @functools.cached_property
  def thread_cluster_coords(self) -> jax.Array:
    """Returns the coordinates of the thread along the cluster axes *and* within the block."""
    return jnp.array(
        list(self.cluster_coords) + [jnp.int32(self.thread_id_in_block)],
        dtype=jnp.int32,
    )

  def are_thread_cluster_axes_collective(
      self, collective_axes: tuple[jax_core.AxisName, ...]
  ) -> tuple[bool, ...]:
    """Returns a tuple of booleans indicating whether each thread cluster axis is collective.

    Args:
      collective_axes: A tuple of collective axis names. The order of the axis
        names in the tuple does not matter.

    Returns:
      A tuple of booleans, where the i-th boolean is true if the i-th axis in
      `thread_cluster_shape` is among the axes in `collective_axes`.

    Raises:
      ValueError: If `collective_axes` contains an axis name that is not a
        thread cluster axis.
    """
    thread_axis_names: list[str | object] = []
    if self.mesh is not None:
      if self.mesh.cluster_names is not None:
        thread_axis_names.extend(self.mesh.cluster_names)
      if self.mesh.thread_name is not None:
        thread_axis_names.append(self.mesh.thread_name)
      else:
        # If `thread_name` is not set, we use a sentinel value for the final
        # axis that corresponds to the (Pallas) threads in a block.
        thread_axis_names.append(object())
    else:
      # If there is no mesh, we use a sentinel value for the single thread axis
      # that corresponds to the single (Pallas) thread in a block.
      thread_axis_names.append(object())
    for axis in collective_axes:
      if not axis in thread_axis_names:
        raise ValueError(
            f"Collective axis `{axis}` not found among axes"
            f" `{thread_axis_names}`"
        )
    return tuple(axis in collective_axes for axis in thread_axis_names)

  def is_thread_block_axis_collective(
      self, collective_axes: tuple[jax_core.AxisName, ...]
  ) -> bool:
    """Returns whether the axis corresponding to the threads in a block is collective."""
    return self.are_thread_cluster_axes_collective(collective_axes)[-1]

  def _interpret_axis_index_p(self, eqn):
    assert eqn.primitive is lax.axis_index_p
    axis_name = eqn.params["axis_name"]
    if self.mesh is not None:
      if axis_name == self.mesh.thread_name:
        return jnp.int32(self.thread_id_in_block)
      elif axis_name in self.mesh.cluster_names:
        return jnp.int32(
            self.cluster_coords[self.mesh.cluster_names.index(axis_name)]
        )
      elif axis_name in self.mesh.grid_names:
        return jnp.int32(
            self.grid_point_coords[self.mesh.grid_names.index(axis_name)]
        )

    if axis_name in self.device_info.axis_indices:
      return jnp.int32(self.device_info.axis_indices[axis_name])

    raise ValueError(
          f"Unable to determine axis index for axis name {axis_name}"
      )

  def _interpret_get_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is state_primitives.get_p
    assert isinstance(eqn.outvars[0].aval, jax_core.ShapedArray)
    invals = get_invals()
    return gpu_callbacks.call_get(
        token=token,
        result_shape_and_dtype=eqn.outvars[0].aval,
        device_id=jnp.int32(self.device_info.device_id),
        grid_point_coords=self.grid_point_coords,
        thread_id=self.thread_id,
        allocation_key_as_array=invals[0],
        transforms=jax.tree.unflatten(eqn.params["tree"], invals[1:]),
        source_info=eqn.source_info,
    )

  def _interpret_swap_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is state_primitives.swap_p
    assert isinstance(eqn.outvars[0].aval, jax_core.ShapedArray)
    invals = get_invals()
    return gpu_callbacks.call_swap(
        token=token,
        result_shape_and_dtype=eqn.outvars[0].aval,
        device_id=jnp.int32(self.device_info.device_id),
        grid_point_coords=self.grid_point_coords,
        thread_id=self.thread_id,
        allocation_key_as_array=invals[0],
        transforms=jax.tree.unflatten(eqn.params["tree"], invals[2:]),
        val=invals[1],
        mask=None,
        source_info=eqn.source_info,
    )

  def _interpret_run_scoped_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is primitives.run_scoped_p

    def _allocate_for_aval(token,
                           aval,
                           transforms: tuple[state_types.Transform, ...],
                           is_thread_block_axis_collective: bool):
      _raise_if_unsupported_memory_space(aval.memory_space)
      ref_count = _get_num_threads_sharing_collective_allocation(
          self.thread_cluster_shape,
          is_thread_block_axis_collective,
      )
      match aval:
        case state_types.AbstractRef(
            inner_aval=inner, memory_space=memory_space, kind=_
        ):
          if transforms:
            # The invar/aval's shape in the jaxpr may be the tiled shape, after
            # tiling and/or swizzling transforms have been applied.  The
            # elements of `transforms` -- to undo the swizzling and/or tiling --
            # are applied any time the variable is used in the jaxpr.
            #
            # We want to allocate a buffer with the logical shape, instead of
            # the tiled shape, so we undo the swizzing and/or tiling here to get
            # the logical shape.
            inner = apply_unswizzle_and_untile(transforms, inner)
          match inner:
            case jax_core.ShapedArray(shape=shape, dtype=dtype):
              if isinstance(dtype, mosaic_gpu_core.BarrierType):
                assert len(shape) == 1
                # A barrier is shared between the threads in a block. Hence its
                # ref count, when computed based on the collective axes, should
                # equal the number of threads in a block.
                assert ref_count == self.num_threads_per_block
                # TODO(nrink): Simplify the interface to
                # `call_allocate_barriers`. Consider making it similar to
                # `call_allocate_buffer`, see below.
                return gpu_callbacks.call_allocate_barriers(
                    token=token,
                    device_id=jnp.int32(self.device_info.device_id),
                    grid_point_coords=self.grid_point_coords,
                    thread_id=self.thread_id,
                    axes_dims=self.thread_cluster_shape,
                    num_arrivals=jnp.int32(dtype.num_arrivals),
                    num_barriers=shape[0],
                    ref_count=jnp.int32(ref_count),
                    source_info=eqn.source_info,
                )
              else:
                memory_space_idx = gpu_callbacks.get_memory_space_idx(
                    memory_space
                )
                thread_id_for_allocation_key = gpu_callbacks.get_thread_id_for_collective_allocation_key(
                    thread_id=jnp.int32(self.thread_id),
                    axes_dims=self.thread_cluster_shape,
                    is_last_thread_axis_collective=is_thread_block_axis_collective,
                )
                token, allocation_request = (
                    gpu_callbacks.call_make_allocation_request_array(
                        token=token,
                        device_id=jnp.int32(self.device_info.device_id),
                        memory_space_id=memory_space_idx,
                        thread_id=jnp.int32(thread_id_for_allocation_key),
                        initial_ref_count=ref_count,
                    )
                )
                return gpu_callbacks.call_allocate_buffer(
                    token=token,
                    device_id=jnp.int32(self.device_info.device_id),
                    grid_point_coords=self.grid_point_coords,
                    thread_id=self.thread_id,
                    allocation_request_as_array=allocation_request,
                    value=interpret_utils.get_uninitialized_array(
                        shape, dtype, self.interpret_params.uninitialized_memory
                    ),
                    source_info=eqn.source_info,
                )
            case _:
              raise ValueError(f"Unsupported inner aval: {inner}")

    def _deallocate_for_aval(token, allocation, aval):
      match aval:
        case state_types.AbstractRef(inner_aval=inner, memory_space=_, kind=_):
          match inner:
            case jax_core.ShapedArray(shape=_, dtype=dtype):
              if isinstance(dtype, mosaic_gpu_core.BarrierType):
                return gpu_callbacks.call_deallocate_barrier(
                    token=token,
                    device_id=jnp.int32(self.device_info.device_id),
                    grid_point_coords=self.grid_point_coords,
                    thread_id=self.thread_id,
                    allocation_key_as_array=allocation,
                    source_info=eqn.source_info,
                )
              else:
                _raise_if_unsupported_memory_space(aval.memory_space)
                return gpu_callbacks.call_deallocate_buffer(
                    token=token,
                    device_id=jnp.int32(self.device_info.device_id),
                    grid_point_coords=self.grid_point_coords,
                    thread_id=self.thread_id,
                    allocation_key_as_array=allocation,
                    source_info=eqn.source_info,
                )

              # TODO(nrink): For sempahores, check that they have value zero at
              # the end of their lifetimes. (If semaphores are never explicitly
              # deallocated, this check could take place at the end of kernel
              # interpretation.)
            case _:
              assert False, (
                  f"Unsupported inner aval: {inner} (should have been"
                  " caught before)"
              )

    assert eqn.primitive is primitives.run_scoped_p
    collective_axes = eqn.params["collective_axes"]
    ref_transforms = eqn.params["ref_transforms"]

    # Allocate a buffer or barrier for each element of
    # `eqn.params['jaxpr'].invars`. It is assumed that each thread runs the same
    # sequence of `run_scoped`s.
    invars = eqn.params["jaxpr"].invars
    allocs = []
    _raise_if_unsupported_collective_axes(
        self.mesh, self.are_thread_cluster_axes_collective(collective_axes)
    )

    for v, transforms in safe_zip(invars, ref_transforms):
      token, alloc = _allocate_for_aval(
          token,
          v.aval,
          transforms,
          self.is_thread_block_axis_collective(collective_axes),
      )

      allocs.append(alloc)

    token, out = self.interpret(
        eqn.params["jaxpr"], token, *get_invals(), *allocs)

    for a, v in safe_zip(allocs, invars):
      token = _deallocate_for_aval(token, a, v.aval)

    return token, out

  def _interpret_cond_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is lax.cond_p
    invals = get_invals()
    return lax.switch(
        invals[0],
        [
            functools.partial(self.interpret, branch_jaxpr.jaxpr)
            for branch_jaxpr in eqn.params["branches"]
        ],
        token,
        *invals[1:],
    )

  def _interpret_scan_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is lax.scan_p
    consts, init_carry, xs = split_list(
        get_invals(),
        [eqn.params["num_consts"], eqn.params["num_carry"]],
    )

    def _scan_body(carry, a):
      token, c = carry
      token, ret = self.interpret(
          eqn.params["jaxpr"].jaxpr, token, *consts, *c, *a
      )
      new_c, b = split_list(ret, [eqn.params["num_carry"]])
      return (token, new_c), b

    (token, carry), out = lax.scan(
        _scan_body, (token, init_carry), xs=xs,
        length=eqn.params.get("length", None))
    return token, carry + out

  def _interpret_while_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    cond_consts, body_consts, init_val = split_list(
        get_invals(),
        [eqn.params["cond_nconsts"], eqn.params["body_nconsts"]],
    )
    token, first_cond = self.interpret(
        eqn.params["cond_jaxpr"].jaxpr, token, *cond_consts, *init_val
    )
    def _body(val):
      token, val, _ = val
      token, val = self.interpret(
          eqn.params["body_jaxpr"].jaxpr, token, *body_consts, *val
      )
      token, cond = self.interpret(
          eqn.params["cond_jaxpr"].jaxpr, token, *cond_consts, *val
      )
      return token, val, cond[0]

    token, out, _ = lax.while_loop(
        lambda args: args[2], _body, (token, init_val, first_cond[0])
    )
    return token, out

  def _interpret_barrier_primitive(
      self,
      eqn,
      token,
      get_invals: Callable[[], Sequence[Any]],
      barrier_callback: Callable[
          [
              jax.Array,
              jax.Array,
              jax.Array,
              jax.Array,
              jax.Array,
              source_info_util.SourceInfo | None,
          ],
          jax.Array,
      ],
  ):
    invals = get_invals()
    # `invals[0]` corresponds to the barrier this primitive operates on.
    allocation_key_as_array = _get_barrier_allocation_key_from_inval(
        invals[0], eqn.params["transforms_treedef"], invals[1:]
    )
    token = barrier_callback(
        token,
        jnp.int32(self.device_info.device_id),
        self.grid_point_coords,
        self.thread_id,
        allocation_key_as_array,
        eqn.source_info,
    )

    assert eqn.primitive.multiple_results
    return token, []

  def _interpret_barrier_arrive_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is gpu_primitives.barrier_arrive_p
    return self._interpret_barrier_primitive(
        eqn, token, get_invals, gpu_callbacks.call_barrier_arrive
    )

  def _interpret_barrier_wait_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is gpu_primitives.barrier_wait_p
    return self._interpret_barrier_primitive(
        eqn, token, get_invals, gpu_callbacks.call_barrier_wait
    )

  def _interpret_arithmetic_primitive(
      self, eqn, get_invals: Callable[[], Sequence[Any]]
  ):
    if self.interpret_params.skip_floating_point_ops and all(
        interpret_utils.is_float(ovar.aval.dtype) for ovar in eqn.outvars
    ):
      # Skip `eqn.primitive.bind` since `eqn.primitive` only produces
      # floating-point values. It is safe to populate `out` with avals
      # since mapping `env.write_many` over `out` (in `self.interpret`) below
      # only relies on the shape and dtype (for writing `Placeholder`s).
      out = [ovar.aval for ovar in eqn.outvars]
      if not eqn.primitive.multiple_results:
        out = out[0]
      return out
    else:
      bind_params = eqn.primitive.get_bind_params(eqn.params)
      for v in bind_params.values():
        if isinstance(v, jax_core.Jaxpr):
          raise NotImplementedError(f"Higher-order primitive {eqn.primitive}")
      return eqn.primitive.bind(*get_invals(), **bind_params)

  def _interpret_copy_gmem_to_smem_p(
      self, eqn, token, get_invals: Callable[[], Sequence[Any]]
  ):
    assert eqn.primitive is gpu_primitives.copy_gmem_to_smem_p
    invals = get_invals()

    if eqn.params["has_barrier"]:
      (
          (src, dst, barrier),
          src_transforms_flat, dst_transforms_flat, barrier_transforms_flat,
      ) = split_list(
          invals,
          [
              3,
              eqn.params["src_transforms_treedef"].num_leaves,
              eqn.params["dst_transforms_treedef"].num_leaves,
          ],
      )
      barrier_allocation_key_as_array = _get_barrier_allocation_key_from_inval(
          barrier, eqn.params["barrier_transforms_treedef"],
          barrier_transforms_flat)
    else:
      raise NotImplementedError(
          "copy_gmem_to_smem_p with barrier=None is not supported"
      )

    token = gpu_callbacks.call_execute_device_local_memory_transfer(
        token=token,
        device_id=jnp.int32(self.device_info.device_id),
        grid_point_coords=self.grid_point_coords,
        thread_id=self.thread_id,
        src_allocation_key_as_array=src,
        src_transforms=jax.tree.unflatten(
            eqn.params["src_transforms_treedef"], src_transforms_flat),
        dst_allocation_key_as_array=dst,
        dst_transforms=jax.tree.unflatten(
            eqn.params["dst_transforms_treedef"], dst_transforms_flat),
        barrier_allocation_key_as_array=barrier_allocation_key_as_array,
        source_info=eqn.source_info,
    )
    return token, []

  def interpret(self, jaxpr, token, *args):
    sentinel_for_floating_point_values = (
        _SENTINEL if self.interpret_params.skip_floating_point_ops else None
    )
    env = interpret_utils.JaxprEnv(
        vars=jaxpr.constvars + jaxpr.invars,
        values=args,
        sentinel_for_floating_point_values=sentinel_for_floating_point_values,
    )

    for eqn in jaxpr.eqns:
      with source_info_util.user_context(
          eqn.source_info.traceback,
          name_stack=eqn.source_info.name_stack,
      ):
        # We defer reading the values for `eqn.invars` into each of the branches
        # of the match statement below. This is because the case for arithmetic
        # primitives may not need to do any reads
        # (if `self.interpret_params.skip_floating_point_ops` is True). If this
        # is the case, we want to avoid materializing the read array into the
        # jaxpr when this function is traced.
        deferred_invals = functools.partial(env.read_many, eqn.invars)
        match eqn.primitive:
          case lax.axis_index_p:
            out = self._interpret_axis_index_p(eqn)
          case primitives.program_id_p:
            # Currently we only support grids and clusters with a single device.
            # Hence, zero is the only valid program id.
            out = jnp.int32(0)
          case state_primitives.get_p:
            token, out = self._interpret_get_p(eqn, token, deferred_invals)
          case primitives.load_p:
            raise NotImplementedError("load_p is not supported on GPU yet")
          case state_primitives.swap_p:
            token, out = self._interpret_swap_p(eqn, token, deferred_invals)
          case primitives.swap_p:
            raise NotImplementedError("swap_p is not supported on GPU yet")
          case primitives.run_scoped_p:
            token, out = self._interpret_run_scoped_p(
                eqn, token, deferred_invals)
          case lax.cond_p:
            token, out = self._interpret_cond_p(eqn, token, deferred_invals)
          case lax.scan_p:
            token, out = self._interpret_scan_p(eqn, token, deferred_invals)
          case lax.while_p:
            token, out = self._interpret_while_p(eqn, token, deferred_invals)
          case gpu_primitives.barrier_wait_p:
            token, out = self._interpret_barrier_wait_p(
                eqn, token, deferred_invals)
          case gpu_primitives.barrier_arrive_p:
            token, out = self._interpret_barrier_arrive_p(
                eqn, token, deferred_invals)
          case gpu_primitives.copy_gmem_to_smem_p:
            token, out = self._interpret_copy_gmem_to_smem_p(
                eqn, token, deferred_invals)
          case _:
            out = self._interpret_arithmetic_primitive(eqn, deferred_invals)

        out = out if eqn.primitive.multiple_results else [out]
        env.write_many(eqn.outvars, out)

    return token, env.read_many(jaxpr.outvars)
