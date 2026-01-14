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
from jax._src.pallas.mosaic_gpu.interpret import gpu_callbacks
from jax._src.state import primitives as state_primitives
from jax._src.util import safe_zip
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


_SENTINEL = jnp.inf


@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxprInterpreter:
  """Interprets a jaxpr by replacing memory operations with (GPU) callbacks."""

  thread_id: int
  mesh: plgpu.Mesh | None
  device_info: DeviceInfo
  compiler_params: Mapping[str, Any]
  interpret_params: interpret_utils.InterpretParams

  @functools.cached_property
  def num_threads(self) -> int:
    if self.mesh is None or self.mesh.num_threads is None:
      return 1
    else:
      return int(self.mesh.num_threads)

  def _interpret_axis_index_p(self, eqn):
    assert eqn.primitive is lax.axis_index_p
    axis_name = eqn.params["axis_name"]
    if (self.mesh is not None) and (axis_name == self.mesh.thread_name):
      return jnp.int32(self.thread_id)
    elif axis_name in self.device_info.axis_indices:
      return self.device_info.axis_indices[axis_name]
    else:
      raise ValueError(
          f"Unable to determine axis index for axis name {axis_name}"
      )

  def _interpret_get_p(self, eqn, get_invals: Callable[[], Sequence[Any]]):
    assert eqn.primitive is state_primitives.get_p
    assert isinstance(eqn.outvars[0].aval, jax_core.ShapedArray)
    invals = get_invals()
    return gpu_callbacks.call_get(
        result_shape_and_dtype=eqn.outvars[0].aval,
        device_id=self.device_info.device_id,
        thread_id=self.thread_id,
        allocation_key=invals[0],
        transforms=jax.tree.unflatten(eqn.params["tree"], invals[1:]),
        source_info=eqn.source_info,
    )

  def _interpret_swap_p(self, eqn, get_invals: Callable[[], Sequence[Any]]):
    assert eqn.primitive is state_primitives.swap_p
    assert isinstance(eqn.outvars[0].aval, jax_core.ShapedArray)
    invals = get_invals()
    return gpu_callbacks.call_swap(
        result_shape_and_dtype=eqn.outvars[0].aval,
        device_id=self.device_info.device_id,
        thread_id=self.thread_id,
        allocation_key=invals[0],
        transforms=jax.tree.unflatten(eqn.params["tree"], invals[2:]),
        val=invals[1],
        mask=None,
    )

  def _interpret_run_scoped_p(
      self, eqn, get_invals: Callable[[], Sequence[Any]]
  ):

    def _allocate_for_aval(aval, same_allocations_for_all_threads: bool):
      _raise_if_unsupported_memory_space(aval.memory_space)
      memory_space_idx = gpu_callbacks.get_memory_space_idx(aval.memory_space)
      allocation_request = gpu_callbacks.make_allocation_request_array(
          device_id=self.device_info.device_id,
          memory_space_id=memory_space_idx,
          thread_id=0 if same_allocations_for_all_threads else self.thread_id,
          initial_ref_count=self.num_threads
          if same_allocations_for_all_threads
          else 1,
      )
      return gpu_callbacks.call_allocate_buffer(
          self.device_info.device_id,
          self.thread_id,
          allocation_request,
          self.interpret_params.get_uninitialized_array(aval.shape, aval.dtype),
      )

    def _deallocate_for_aval(allocation, aval):
      # TODO(nrink): Check that sempahores have value zero at the end of their
      # lifetimes. (If semaphores are never explicitly deallocated, this check
      # could take place at the end of kernel interpretation.)
      _raise_if_unsupported_memory_space(aval.memory_space)
      return gpu_callbacks.call_deallocate_buffer(
          allocation,
      )

    assert eqn.primitive is primitives.run_scoped_p
    collective_axes = eqn.params["collective_axes"]
    # Note that on GPU, `SMEM` buffers and barriers can only be allocated
    # collectively (i.e. corresponding to `same_allocations=True`). In the
    # interpreter we are a little more lenient and allow non-collective
    # allocations for `SMEM` buffers.
    same_allocations = False
    if collective_axes:
      if (
          self.mesh is None
          or len(collective_axes) != 1
          or collective_axes[0] != self.mesh.thread_name
      ):
        raise NotImplementedError(
            "When interpreting `run_scoped` in a GPU kernel, non-empty"
            " `collective_axes` is currently only supported when it contains a"
            " single axis that agrees with the thread axis (i.e. `thread_name`)"
            " of the mesh."
        )
      same_allocations = True

    # Allocate a buffer or semaphore (to do, see below) for each element of
    # `eqn.params['jaxpr'].invars`. It is assumed that each thread runs the same
    # sequence of `run_scoped`s.
    vars = eqn.params["jaxpr"].invars
    allocs = []
    for v in vars:
      # TODO(nrink): Support semaphores. (Currently the call to
      # `_allocate_for_aval` will fail when trying to allocate a semaphore.)
      allocs.append(_allocate_for_aval(v.aval, same_allocations))

    out = self.interpret(eqn.params["jaxpr"], *get_invals(), *allocs)

    for a, v in safe_zip(allocs, vars):
      _deallocate_for_aval(a, v.aval)

    return out

  def _interpret_cond_p(self, eqn, get_invals: Callable[[], Sequence[Any]]):
    invals = get_invals()
    return lax.switch(
        invals[0],
        [
            functools.partial(self.interpret, branch_jaxpr.jaxpr)
            for branch_jaxpr in eqn.params["branches"]
        ],
        *invals[1:],
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
      subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
      return eqn.primitive.bind(*subfuns, *get_invals(), **bind_params)

  def interpret(self, jaxpr, *args):
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
            out = self._interpret_get_p(eqn, deferred_invals)
          case primitives.load_p:
            raise NotImplementedError("load_p is not supported on GPU yet")
          case state_primitives.swap_p:
            out = self._interpret_swap_p(eqn, deferred_invals)
          case primitives.swap_p:
            raise NotImplementedError("swap_p is not supported on GPU yet")
          case primitives.run_scoped_p:
            out = self._interpret_run_scoped_p(eqn, deferred_invals)
          case lax.cond_p:
            out = self._interpret_cond_p(eqn, deferred_invals)
          case _:
            out = self._interpret_arithmetic_primitive(eqn, deferred_invals)

        out = out if eqn.primitive.multiple_results else [out]
        env.write_many(eqn.outvars, out)

    return env.read_many(jaxpr.outvars)
