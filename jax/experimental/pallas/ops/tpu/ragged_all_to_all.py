# Copyright 2024 The JAX Authors.
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

"""A simple ragged all-to-all kernel.

This kernel is intended to be a simple example of how to implement a ragged
all-to-all collective for TPU using Pallas.

The kernel assumes each TPU device can send data directly to each other
TPU device involved in the collective.
"""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def _get_neighbor_id(
    mesh: jax.sharding.Mesh, axis_name: str, device_id: jax.Array):
  axis_names = mesh.axis_names
  which_axis = axis_names.index(axis_name)
  return tuple(
      device_id if i == which_axis else jax.lax.axis_index(a)
      for i, a in enumerate(axis_names)
  )

def _compute_output_offsets(input_sizes: jax.Array) -> jax.Array:
  """Computes the output offsets for a ragged all-to-all.

  Arguments:
    input_sizes: The size of each group.  An int32 array of shape
      [num_devices, num_groups].

  Returns:
    The output offsets. An int32 array of shape [num_devices, num_groups].
  """
  num_devices, num_groups = input_sizes.shape
  receive_sizes = (
      input_sizes.reshape((num_devices, num_devices, num_groups // num_devices))
                 .transpose((1, 2, 0))
                 .reshape((num_devices, num_groups)))
  output_offsets_in_output_order = jnp.cumulative_sum(
      receive_sizes[:, :-1], axis=1, include_initial=True)
  return (output_offsets_in_output_order
          .reshape((num_devices, num_groups // num_devices, num_devices))
          .transpose((2, 0, 1))
          .reshape((num_devices, num_groups)))

def _ragged_all_to_all_kernel(
    input_ref,
    input_offsets_ref,   # [num_groups]
    input_sizes_ref,     # [num_groups]
    output_offsets_ref,  # [num_groups]
    total_send_amount_ref,     # [1]
    total_receive_amount_ref,  # [1]
    _,  # is also output_ref
    output_ref,
    send_sem,
    recv_sem,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
):
  device_id = jax.lax.axis_index(axis_name)
  num_groups = input_offsets_ref.shape[0]
  num_devices = mesh.shape[axis_name]
  groups_per_device = num_groups // num_devices

  # Shift group_id by device_id, so that communication in iteration `i` looks
  # like 0->i, 1->i+1, 2->i+2, ..., instead of all messages in iteration `i`
  # going to the same device (i.e., 0->i, 1->i, 2->i, ...).
  group_id = (pl.program_id(0) + device_id) % num_groups

  target_device_id = group_id // groups_per_device
  size = input_sizes_ref[group_id]
  copy = pltpu.make_async_remote_copy(
      input_ref.at[pl.ds(input_offsets_ref[group_id], size)],
      output_ref.at[pl.ds(output_offsets_ref[group_id], size)],
      send_sem,
      recv_sem,
      device_id=_get_neighbor_id(mesh, axis_name, target_device_id),
  )
  @pl.when(size > 0)  # type: ignore[no-redef]
  def _():
    copy.start()

  @pl.when(pl.program_id(0) == num_groups-1)  # type: ignore[no-redef]
  def _():
    # Create "dummy" copies so that we can wait on the send/recv semaphores
    # until all previous async copies have completed.

    num_writes = total_send_amount_ref[0]
    dummy_with_send_count = pltpu.make_async_remote_copy(
        input_ref.at[pl.ds(0, num_writes)],
        output_ref.at[pl.ds(0, num_writes)],
        send_sem,
        recv_sem,
        device_id=_get_neighbor_id(mesh, axis_name, target_device_id),
    )
    dummy_with_send_count.wait_send()

    num_reads = total_receive_amount_ref[0]
    dummy_with_receive_count = pltpu.make_async_remote_copy(
        input_ref.at[pl.ds(0, num_reads)],
        output_ref.at[pl.ds(0, num_reads)],
        send_sem,
        recv_sem,
        device_id=_get_neighbor_id(mesh, axis_name, target_device_id),
    )
    dummy_with_receive_count.wait_recv()

def ragged_all_to_all(
    input_array: jax.Array,
    input_offsets: jax.Array,
    input_sizes: jax.Array,
    output: jax.Array | None = None,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    output_size : int | None = None,
    transpose: bool = False,
) -> jax.Array:
  """All-to-all collective for ragged arrays.

  The collective operates on a sharded, ragged array:
  ```
    [ g_1,1, ..., g_1,m, g_2,1, ..., g_2,m, ..., g_n,1, ..., g_n,m ]
  ```
  where each group `g_i,j` (group `j` in shard `i`) can have a different size
  in its leading dimension, and where `m` is the number of groups (also referred
  to `num_groups`) and `n` is the number of shards (also referred to as
  `num_devices`).  The output is a sharded array:
  ```
    [ g_1,1, ..., g_1,m//n, g_2,1, ..., g_2,m//n, ..., g_n,1, ..., g_n,m//n
      g_1,m//n+1, ..., g_1,2m//n, ..., g_n,m//n+1, ..., g_n,2m//n
      ...
      g_1,(n-1)m//n+1, ..., g_1,m, ..., g_n,(n-1)m//n+1, ..., g_n,m ]
  ```
  That is, group `g_i,j` is sent from shard `i` to shard `j // (m // n) + 1`.

  Arguments:
    input_array: An array that is ragged along its leading dimension.  That is,
      each shard is a concatenation of groups `[g_1, ..., g_m]` (plus optional
      padding at the end), where each group `g_i` can have a different size in
      its leading dimension.  The array must be sharded along its leading
      dimension.  And the number of groups must be divisible by the number of
      shards (which we call `num_devices`).
    input_offsets: An int32 array of shape '[num_devices, num_groups]'.
      `input_offsets[d, i]` specifies where the collective should start reading
      the i-th group on the d-th device.
    input_sizes: An int32 array of shape '[num_devices, num_groups]'.
      `input_sizes[d, i]` specifies how many elements the collective should
      read from the i-th group on the d-th device.
    output: An optional array in which to store the result.  Either this
      argument or `output_size` must be provided.  The output array must be
      large enough to hold all of the results.
    mesh: The mesh.
    axis_name: The name of the axis over which the collective will be performed.
    output_size: If no `output` is provided, then an output array will be
      allocated with shape `[output_size] + input_array.shape[1:]`.  Either this
      argument or `output` must be provided.  The output array must be large
      enough to hold all of the results.
    transpose: If True, then the the transpose of the collective will
      be performed.  That is, for every write of element `i` from the input to
      element `j` of the output that the `transpose=False` collective would
      have performend, the collective will write element `j` of the input to
      element `i` of the output.  Thus, the `transpose=True` collective is
      the inverse of the `transpose=False` collective.

    Returns:
      The output array.
  """
  output_offsets = _compute_output_offsets(input_sizes)

  return _ragged_all_to_all(
      input_array, input_offsets, input_sizes, output_offsets,
      output, mesh, axis_name, output_size, transpose)

@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _ragged_all_to_all(
    input_array: jax.Array,
    input_offsets: jax.Array,
    input_sizes: jax.Array,
    output_offsets: jax.Array,
    output: jax.Array,
    mesh,
    axis_name,
    output_size : int | None,
    transpose: bool,
) -> jax.Array:
  num_devices, num_groups = input_sizes.shape

  if output is None:
    if output_size is None:
      raise ValueError('Neither output nor output_size was provided.')
    output = jnp.zeros((output_size,) + input_array.shape[1:],
                       input_array.dtype)
  else:
    if output_size is not None:
      pass  # TODO(jburnim): Check that the sizes match?  Or make this an error?

  if transpose:
    def _transpose(x):
      return (x.reshape((num_devices, num_devices, -1))
              .transpose((1, 0, 2))
              .reshape((num_devices, num_groups)))
    input_offsets, input_sizes, output_offsets = jax.tree.map(
        _transpose, (output_offsets, input_sizes, input_offsets))

  total_receive_sizes = (
      input_sizes.reshape(num_devices, num_devices, -1).sum((0, 2)))
  index = jax.lax.axis_index(axis_name)

  return pl.pallas_call(
      functools.partial(_ragged_all_to_all_kernel,
                        mesh=mesh, axis_name=axis_name),
      out_shape=jax.ShapeDtypeStruct(output.shape, output.dtype),
      grid=(num_groups,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.ANY),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
      scratch_shapes=[pltpu.SemaphoreType.DMA] * 2,
      input_output_aliases={6: 0},
  )(
      input_array,
      input_offsets[index],
      input_sizes[index],
      output_offsets[index],
      input_sizes[index].sum(keepdims=True),
      total_receive_sizes[index][jnp.newaxis],
      output,
  )

def _ragged_all_to_all_fwd(
  input_array: jax.Array,
  input_offsets: jax.Array,
  input_sizes: jax.Array,
  output_offsets: jax.Array,
  output: jax.Array | None,
  mesh,
  axis_name,
  output_size : int | None,
  transpose: bool):

  return _ragged_all_to_all(
      input_array,
      input_offsets,
      input_sizes,
      output_offsets,
      output,
      mesh=mesh,
      axis_name=axis_name,
      output_size=output_size,
      transpose=transpose,
  ), (input_array.shape[0],
      input_offsets,
      input_sizes,
      output_offsets,
      output)

def _ragged_all_to_all_bwd(
    mesh, axis_name, output_size, transpose, res, d_output):
  del output_size
  (input_array_length, input_offsets, input_sizes, output_offsets, output) = res

  d_input_array = _ragged_all_to_all(
      d_output,
      input_offsets,
      input_sizes,
      output_offsets,
      output,
      mesh=mesh,
      axis_name=axis_name,
      output_size=input_array_length,
      transpose=not transpose,
  )

  if output is not None:
    # What if there is an existing output?  Then we should zero-out the
    # elements of d_output corresponding to locations that were written in the
    # forward pass.
    raise NotImplementedError(
        'Gradients not yet supported when output is provided.')

  return d_input_array, None, None, None, None

_ragged_all_to_all.defvjp(_ragged_all_to_all_fwd, _ragged_all_to_all_bwd)
