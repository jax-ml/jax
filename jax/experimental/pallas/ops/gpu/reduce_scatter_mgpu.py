# Copyright 2025 The JAX Authors.
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

"""Reduce scatter kernel implemented using Mosaic GPU."""

import math
from typing import Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp


def reduce_scatter(
    x: jax.Array,
    *,
    axis_name,
    reduction: Literal["add", "min", "max", "and", "or", "xor"] = "add",
    vec_size: int = 2,
    num_blocks: int | None = None,
    rows_per_transfer: int = 1,
) -> jax.Array:
  """Performs a reduce-scatter operation across devices using multimem instructions.

  Args:
    x: Input array. Should be sharded across the specified axis.
    axis_name: Name of the mesh axis to reduce-scatter across.
    reduction: Reduction operation to perform. Supported: "add", "min", "max",
      "and", "or", "xor".
    vec_size: Vector size for the layout (default: 2).
    num_blocks: Number of blocks to use. Defaults to the device core count.
    rows_per_transfer: Number of rows to transfer per iteration (default: 1).
  """
  num_devices = lax.axis_size(axis_name)
  input_shape = x.shape
  dtype = x.dtype

  if num_blocks is None:
    num_blocks = backend.get_default_device().core_count

  # TODO(apaszke): Support other axes
  if input_shape[0] % num_devices != 0:
    raise ValueError(
        f"First dimension of input ({input_shape[0]}) must be divisible by "
        f"number of devices ({num_devices})"
    )
  output_shape = (input_shape[0] // num_devices, *input_shape[1:])
  if math.prod(output_shape) % vec_size:
    raise ValueError(
        "The total number of elements in the output"
        f" ({math.prod(output_shape)}) must be divisible by the vec_size"
        f" ({vec_size})"
    )

  if output_shape[0] % rows_per_transfer != 0:
    raise ValueError(
        f"Rows per device ({output_shape[0]}) must be divisible by "
        f"rows_per_transfer ({rows_per_transfer})"
    )

  min_transfer_elems = 128 * vec_size
  elems_per_row = math.prod(input_shape[1:])
  elems_per_transfer = rows_per_transfer * elems_per_row
  if elems_per_transfer < min_transfer_elems or elems_per_transfer % min_transfer_elems:
    raise ValueError(
        f"Transfer size ({elems_per_transfer} elements) is smaller or not"
        f" divisible by the minimum required ({min_transfer_elems} elements)."
        " Increase rows_per_transfer to at least"
        f" {min_transfer_elems // elems_per_row} or decrease {vec_size=}."
    )

  def kernel(x_ref, y_ref, scratch_ref):
    dev_idx = lax.axis_index(axis_name)
    rows_per_device = output_shape[0]
    num_transfers = rows_per_device // rows_per_transfer
    dev_base_idx = dev_idx * rows_per_device

    # TODO(apaszke): Tile other dimensions if they are too big
    @plgpu.nd_loop((num_transfers,), collective_axes="blocks")
    def _transfer_loop(loop_info: plgpu.NDLoopInfo):
      (transfer_idx,) = loop_info.index
      row_idx = transfer_idx * rows_per_transfer
      start_row = dev_base_idx + row_idx
      y_ref[pl.ds(row_idx, rows_per_transfer)] = plgpu.layout_cast(
          plgpu.multimem_load_reduce(
              x_ref.at[pl.ds(start_row, rows_per_transfer)],
              collective_axes=axis_name,
              reduction_op=reduction,
          ),
          plgpu.Layout.WG_STRIDED(
              (rows_per_transfer, *output_shape[1:]), vec_size=vec_size
          ),
      )
    # TODO(apaszke): Use multimem.red to increment the semaphore
    for d in range(num_devices):
      pl.semaphore_signal(scratch_ref, device_id=d)
    pl.semaphore_wait(scratch_ref, num_devices)

    # TODO(b/448323639): We fake modify the input to ensure that XLA:GPU copies
    # the operand into symmetric memory.
    @pl.when(dev_idx == -1)
    def _never():
      x_ref[(0,) * len(x_ref.shape)] = jnp.asarray(0, x_ref.dtype)

  return plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
      grid=(num_blocks,),
      grid_names=("blocks",),
      scratch_shapes=[plgpu.SemaphoreType.REGULAR],
  )(x)
