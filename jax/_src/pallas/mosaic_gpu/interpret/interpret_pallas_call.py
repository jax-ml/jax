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

from collections.abc import Mapping, Sequence, Set
import dataclasses
import math
from typing import Any

import jax
from jax._src import callback
from jax._src import core as jax_core
from jax._src import effects
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic.interpret import thread_map
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic_gpu import core as mosaic_gpu_core
from jax._src.pallas.mosaic_gpu.interpret import gpu_callbacks
from jax._src.pallas.mosaic_gpu.interpret import jaxpr_interpret
from jax._src.typing import Array
from jax._src.util import (safe_zip, split_list)
from jax.experimental.pallas import mosaic_gpu as plgpu


InterpretParams = interpret_utils.InterpretGPUParams


def get_interpret_effects() -> Set[effects.Effect]:
  return {callback._OrderedIOEffect}  # pylint: disable=protected-access


def get_races() -> gpu_callbacks.RaceDetectionState:
  return gpu_callbacks.get_races()


def reset_gpu_interpret_mode_state():
  gpu_callbacks.reset_gpu_interpret_mode_state()


def _get_grid_bounds(grid_mapping: pallas_core.GridMapping) -> tuple[int, ...]:
  if grid_mapping.num_dynamic_grid_bounds > 0:
    raise NotImplementedError(
        "Dynamic grid bounds not (yet) supported in GPU interpret mode."
    )
  result = []
  for x in grid_mapping.grid:
    # We have already tested for the absence of dynamic grid bounds. So all
    # entries in the grid should be ints.
    assert isinstance(x, int)
    result.append(x)
  return tuple(result)


def _get_grid_dims_and_num_threads(
    grid_mapping: pallas_core.GridMapping, mesh: plgpu.Mesh | None
) -> tuple[tuple[int, ...], int]:
  if not mesh:
    num_threads = 1
    grid_dims = _get_grid_bounds(grid_mapping)
  elif isinstance(mesh, plgpu.Mesh):
    if mesh.cluster is not None and math.prod(mesh.cluster) != 1:
      raise NotImplementedError(
          f"Invalid cluster {mesh.cluster} in mesh: GPU interpret mode does not"
          " support (non-trivial) clusters."
      )
    num_threads = int(mesh.num_threads or 1)
    grid_dims = tuple(mesh.grid)
  else:
    raise ValueError(f"Unsupported mesh type: {type(mesh)}")

  reconstructed_grid = grid_dims + (num_threads,)
  if math.prod(_get_grid_bounds(grid_mapping)) != math.prod(reconstructed_grid):
    raise NotImplementedError(
        f"Invalid grid {grid_mapping.grid} in grid_mapping: expected grid to"
        f" have the same size as {reconstructed_grid}"
    )

  return grid_dims, num_threads


def _allocate_buffers_for_inputs(
    device_id: int,
    invars: Sequence[Any],
    inputs: Sequence[jax.Array],
) -> list[jax.Array]:
  """Allocates `GMEM` buffers for the `inputs` of a `pallas_call`."""
  # TODO(nrink): This code is a simplified version to the corresponding TPU
  # interpreter code. Eventually, we should merge the two.
  input_buffer_keys = []
  for var, value in safe_zip(invars, inputs):
    assert var.aval.dtype == value.dtype
    allocation_request = gpu_callbacks.make_allocation_request_array(
        device_id=device_id,
        # All operands of a `pallas_call`/`core_map` that are arrays (i.e. that
        # are not sempahores, barriers etc.) are placed in `GMEM`. These arrays
        # (or slices thereof) may need to be copied into `SMEM` before executing
        # the kernel.
        memory_space_id=gpu_callbacks.get_memory_space_idx(
            mosaic_gpu_core.MemorySpace.GMEM
        ),
    )
    input_buffer_keys.append(
        gpu_callbacks.call_allocate_buffer_for_all_threads(
            device_id, allocation_request, value
        )
    )

  return input_buffer_keys


@dataclasses.dataclass(frozen=True)
class AllocationKeyAndValue:
  key: jax.Array
  value: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    return self.value.shape


def _allocate_buffers_for_outputs(
    device_id: int,
    num_threads: int,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    input_buffer_keys: Sequence[jax.Array],
    input_vals: Sequence[jax.Array],
    interpret_params: interpret_utils.InterpretGPUParams,
) -> list[AllocationKeyAndValue]:
  """Allocates `GMEM` buffers for `pallas_call` outputs, respecting aliased inputs."""
  # TODO(nrink): This code is a simplified version to the corresponding TPU
  # interpreter code. Eventually, we should merge the two.
  assert len(input_buffer_keys) == len(input_vals)

  oi_alias_map = {v: k for k, v in input_output_aliases}
  output_buffer_keys_and_values = []

  block_shapes = [
      pallas_core._get_block_shape(bm.block_shape)  # pylint: disable=protected-access
      for bm in grid_mapping.block_mappings
  ]
  num_inputs = grid_mapping.num_inputs

  num_outputs = grid_mapping.num_outputs
  output_block_shapes = block_shapes[num_inputs : num_inputs + num_outputs]
  for output_idx, bm in enumerate(grid_mapping.block_mappings_output):
    if output_idx in oi_alias_map:
      aliased_input_idx = oi_alias_map[output_idx]
      # Reuse the `GMEM` buffer for the aliased `pallas_call`/`core_map` input.
      output_buffer_keys_and_values.append(
          AllocationKeyAndValue(
              key=input_buffer_keys[aliased_input_idx],
              value=input_vals[aliased_input_idx],
          )
      )
    else:
      out_val = interpret_params.get_uninitialized_array(
          bm.array_aval.shape, bm.array_aval.dtype
      )
      padded_val = interpret_params.pad_to_block_dimension(
          out_val, output_block_shapes[output_idx]
      )
      allocation_request = gpu_callbacks.make_allocation_request_array(
          device_id=device_id,
          # All outputs of a `pallas_call`/`core_map` that are arrays (i.e. that
          # are not sempahores, barriers etc.) are placed in `GMEM`. Results
          # from executing the kernel (or slices thereof) may need to be copied
          # from `SMEM` into the `GMEM` output buffers that are allocated here.
          memory_space_id=gpu_callbacks.get_memory_space_idx(
              mosaic_gpu_core.MemorySpace.GMEM
          ),
          initial_ref_count=num_threads,
      )
      output_buffer_key = gpu_callbacks.call_allocate_buffer_for_all_threads(
          device_id, allocation_request, padded_val
      )
      output_buffer_keys_and_values.append(
          AllocationKeyAndValue(key=output_buffer_key, value=out_val)
      )

  return output_buffer_keys_and_values


def _get_kernel_buffers(
    device_id: int,
    num_threads: int,
    grid_mapping: pallas_core.GridMapping,
    invars: Sequence[Any],
    input_buffer_keys: Sequence[jax.Array],
    output_buffer_keys: Sequence[jax.Array],
    interpret_params: interpret_utils.InterpretGPUParams,
) -> list[jax.Array]:
  """Collects buffers to be passed to the kernel from `pallas_call` input/output buffers."""
  # TODO(nrink): This code is a simplified version to the corresponding TPU
  # interpreter code. Eventually, we should merge the two.
  kernel_buffer_keys = []
  for i, var in enumerate(invars):
    output_idx = i - grid_mapping.num_inputs
    is_input = i < grid_mapping.num_inputs
    is_output = (output_idx >= 0) and (output_idx < grid_mapping.num_outputs)
    aval = var.aval
    # TODO(nrink): Support allocation of semaphores.
    if gpu_callbacks.is_gmem_memory_space(aval.memory_space):
      # Use the already-allocated GMEM input or output buffer.
      #
      # TODO(jburnim): For kernel args in GMEM, check that block shape equals
      # the shape of the corresponding `pallas_call` input, and that the
      # index_map is trivial.
      assert is_input ^ is_output
      if is_input:
        kernel_buffer_keys.append(input_buffer_keys[i])
      if is_output:
        kernel_buffer_keys.append(output_buffer_keys[output_idx])
    else:
      allocation_request = gpu_callbacks.make_allocation_request_array(
          device_id=device_id,
          memory_space_id=gpu_callbacks.get_memory_space_idx(aval.memory_space),
          initial_ref_count=num_threads,
      )
      init_val = interpret_params.get_uninitialized_array(
          aval.shape, aval.dtype
      )
      kernel_buffer_keys.append(
          gpu_callbacks.call_allocate_buffer_for_all_threads(
              device_id, allocation_request, init_val
          )
      )

  return kernel_buffer_keys


def _get_outputs(
    device_id: int, output_buffers: Sequence[AllocationKeyAndValue]
) -> Sequence[Array]:
  """Reads and returns values from the allocated output buffers."""
  outputs = []
  for buffer in output_buffers:
    outputs.append(
        gpu_callbacks.call_get(
            result_shape_and_dtype=buffer.value,
            device_id=device_id,
            thread_id=0,
            allocation_key=buffer.key,
            transforms=(),  # Read the entire buffer.
        )
    )

  return outputs


def _load_and_store_between_allocation_keys(
    *,
    device_id: int,
    thread_id: int,
    share_and_dtype: Any,
    load_allocation_key: jax.Array,
    store_allocation_key: jax.Array,
    transform,
):
  loaded_value = gpu_callbacks.call_get(
      result_shape_and_dtype=share_and_dtype,
      device_id=device_id,
      thread_id=thread_id,
      allocation_key=load_allocation_key,
      transforms=transform,
  )
  gpu_callbacks.call_swap(
      result_shape_and_dtype=share_and_dtype,
      device_id=device_id,
      thread_id=thread_id,
      allocation_key=store_allocation_key,
      transforms=transform,
      val=loaded_value,
      mask=None,
  )


def _copy_from_gmem_buffers(
    device_id: int,
    thread_id: int,
    avals: Sequence[Any],
    gmem_buffer_keys: Sequence[jax.Array],
    target_buffer_keys: Sequence[jax.Array],
    transforms):
  for aval, gmem_buffer_key, target_buffer_key in zip(
      avals, gmem_buffer_keys, target_buffer_keys, strict=True
  ):
    if gpu_callbacks.is_gmem_memory_space(aval.memory_space):
      continue
    _load_and_store_between_allocation_keys(
        device_id=device_id,
        thread_id=thread_id,
        share_and_dtype=aval,
        load_allocation_key=gmem_buffer_key,
        store_allocation_key=target_buffer_key,
        transform=transforms,
    )


def _copy_to_gmem_buffers(
    device_id: int,
    thread_id: int,
    avals: Sequence[Any],
    source_buffer_keys: Sequence[jax.Array],
    gmem_buffer_keys: Sequence[jax.Array],
    transforms):
  for aval, source_buffer_key, gmem_buffer_key in zip(
      avals, source_buffer_keys, gmem_buffer_keys, strict=True
  ):
    if gpu_callbacks.is_gmem_memory_space(aval.memory_space):
      continue
    _load_and_store_between_allocation_keys(
        device_id=device_id,
        thread_id=thread_id,
        share_and_dtype=aval,
        load_allocation_key=source_buffer_key,
        store_allocation_key=gmem_buffer_key,
        transform=transforms,
    )


def interpret_pallas_call(
    *args,
    jaxpr: jax_core.Jaxpr,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    mesh: plgpu.Mesh | None,
    compiler_params: Mapping[str, Any],
    cost_estimate: pallas_core.CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
    interpret_params: interpret_utils.InterpretGPUParams,
    metadata: Mapping[str, str] | None,
    **kwargs,
) -> Sequence[Array]:
  # TODO(nrink): A more fleshed out implementation of the GPU interpreter may
  # need to use some of these `del`ed arguments.
  del debug, cost_estimate, metadata, out_avals, kwargs

  # TODO(nrink): Support non-trivial `BlockSpec`s (i.e. with non-trivial
  # `index_map`s).
  assert all(bm.has_trivial_window() for bm in grid_mapping.block_mappings)

  grid_dims, num_threads = _get_grid_dims_and_num_threads(
      grid_mapping, mesh
  )
  device_info = jaxpr_interpret.DeviceInfo()

  interpret_params = dataclasses.replace(
      interpret_params, num_cores_or_threads=num_threads
  )

  gpu_callbacks.call_initialize_shared_memory(
      num_devices=device_info.num_devices,
      num_threads=num_threads,
      interpret_params=interpret_params,
  )

  dynamic_grid_args, scalars, inputs = split_list(
      args,
      [grid_mapping.num_dynamic_grid_bounds, grid_mapping.num_index_operands],
  )
  if dynamic_grid_args:
    raise NotImplementedError("Dynamic grid bounds not (yet) supported on GPU")
  if scalars:
    raise NotImplementedError("Scalar arguments not (yet) supported on GPU")

  assert grid_mapping.num_index_operands == 0

  input_buffer_keys = _allocate_buffers_for_inputs(
      device_info.device_id,
      jaxpr.invars[: grid_mapping.num_inputs],
      inputs,
  )

  output_buffers = _allocate_buffers_for_outputs(
      device_info.device_id,
      num_threads,
      input_output_aliases,
      grid_mapping,
      input_buffer_keys,
      inputs,
      interpret_params,
  )

  kernel_buffer_keys = _get_kernel_buffers(
      device_info.device_id,
      num_threads,
      grid_mapping,
      jaxpr.invars,
      input_buffer_keys,
      [buffer.key for buffer in output_buffers],
      interpret_params,
  )

  # TODO(nrink): The two assignments below have been taken from the
  # corresponding TPU interpreter code. Confirm that they make sense here (i.e.
  # for GPU kernels).
  kernel_input_buffer_keys, kernel_output_buffer_keys, _ = split_list(
      kernel_buffer_keys, [grid_mapping.num_inputs, grid_mapping.num_outputs]
  )
  input_vars, output_vars = split_list(
      jaxpr.invars[grid_mapping.slice_block_ops], [grid_mapping.num_inputs]
  )

  def _kernel(thread_id, grid_point_coords):
    # Note that the copying from `GMEM` buffers here could introduce races when
    # multiple threads copy to the same kernel input buffer. For this to happen,
    # (a) there must be multiple threads and (b) the targeted kernel input
    # buffer must not be in `GMEM` (since we omit copies from `GMEM` to `GMEM`).
    # Currently, the ways in which a Pallas GPU kernel can be invoked do not
    # allow for (a) and (b) to be true at the same time: (a) requires that the
    # kernel is *not* invoked through a `pallas_call` but (b) can only be caused
    # if `BlockSpec`s are used when invoking the kernels, which requires that
    # the kernel be invoked through a `pallas_call`.
    #
    # TODO(nrink): Support copying of slices/blocks only, based on the
    # `BlockSpec`s. (Currently only trivial `BlockSpec`s are supported.)
    _copy_from_gmem_buffers(
        device_id=device_info.device_id,
        thread_id=thread_id,
        avals=[var.aval for var in input_vars],
        gmem_buffer_keys=input_buffer_keys,
        target_buffer_keys=kernel_input_buffer_keys,
        transforms=(),
    )

    jaxpr_interpreter = jaxpr_interpret.JaxprInterpreter(
        grid_point_coords=grid_point_coords,
        thread_id=thread_id,
        mesh=mesh,
        device_info=device_info,
        compiler_params=compiler_params,
        interpret_params=interpret_params,
    )
    jaxpr_interpreter.interpret(jaxpr, *kernel_buffer_keys)

    # Note that a comment about potential races that is analogous to the comment
    # before the call to `_copy_from_gmem_buffers` above applies here too.
    #
    # TODO(nrink): Support copying of slices/blocks only, based on the
    # `BlockSpec`s. (Currently only trivial `BlockSpec`s are supported.)
    _copy_to_gmem_buffers(
        device_id=device_info.device_id,
        thread_id=thread_id,
        avals=[var.aval for var in output_vars],
        source_buffer_keys=kernel_output_buffer_keys,
        gmem_buffer_keys=[buffer.key for buffer in output_buffers],
        transforms=(),
    )

  num_grid_loop_iterations = math.prod(grid_dims)

  def _grid_loop_body(loop_idx: int, _: None):
    grid_point_coords = interpret_utils.get_indices(
        grid_dims, loop_idx
    )
    thread_map.thread_map(_kernel, num_threads, grid_point_coords)

  # TODO(nrink): Should we only create happens-before here from thread 0 to
  # the other threads? Currently we update the vector clocks for all threads by
  # looking at the vector clock of all (other) threads. It should suffice, but
  # this needs to be confirmed, to update the vector clocks for all threads by
  # looking only at the vector clock of thread 0 (and at the vector clock for
  # the thread itself).
  gpu_callbacks.call_update_clocks_for_device_barrier(device_info.device_id)

  # TODO(nrink): For now we execute the grid by sequentially looping over the
  # points in the grid. This may need to be refined to be more faithful to the
  # semantics of grid execution on a real GPU. (The other extreme would be to
  # execute all grid points fully concurrently, e.g. in individual threads.)
  jax.lax.fori_loop(0, num_grid_loop_iterations, _grid_loop_body, None)

  # TODO(nrink): Should we only create happens-before here from the other
  # threads to thread 0? Analogous to the comment above, it should suffice, but
  # this needs to be confirmed, to update only the vector clock of thread 0 (and
  # not the vector clocks for all other threads).
  gpu_callbacks.call_update_clocks_for_device_barrier(device_info.device_id)

  outputs = _get_outputs(device_info.device_id, output_buffers)

  # We assert that no barriers remain allocated. This is an internal consistency
  # check because the interpreter should take care of deallocating all barriers
  # that it has allocated. It is important that the interpreter deallocates all
  # barriers because barrier deallocation also checks that the barrier was used
  # correctly by the kernel/threads. (Specifically, it is checked that if a
  # thread has observed any completed barrier arrival, it has in fact observed
  # all completed arrivals).
  gpu_callbacks.call_assert_no_barriers_allocated()

  gpu_callbacks.call_clean_up_shared_memory()

  return outputs
