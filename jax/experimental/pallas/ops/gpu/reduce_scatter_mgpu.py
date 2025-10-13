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
    num_blocks: int | None = None,
    tile_size: int | None = None,
    # TODO(apaszke): Infer default from the dtype
    vec_size: int = 2,
) -> jax.Array:
  """Performs a reduce-scatter operation across devices using multimem instructions.

  Args:
    x: Input array. Should be sharded across the specified axis.
    axis_name: Name of the mesh axis to reduce-scatter across.
    reduction: Reduction operation to perform. Supported: "add", "min", "max",
      "and", "or", "xor".
    vec_size: Vector size for the layout (default: 2).
    num_blocks: Number of blocks to use. Defaults to the device core count.
    tile_size: Total tile size to split between row and minor dimensions.
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

  min_transfer_elems = 128 * vec_size
  if tile_size is None:
    # TODO(apaszke): 8 is just an arbitrary unrolling factor. Tune it!
    unroll_factor = min(math.prod(output_shape) // min_transfer_elems, 8)
    tile_size = unroll_factor * min_transfer_elems
  if tile_size < min_transfer_elems:
    raise ValueError(
        f"{tile_size=} is smaller than minimum required"
        f" {min_transfer_elems} for {vec_size=}"
    )

  minor_dims_size = math.prod(input_shape[1:])
  minor_tile = math.gcd(tile_size, minor_dims_size)
  major_tile = tile_size // minor_tile

  # TODO(apaszke): Just peel the last step if non-divisible.
  if output_shape[0] % major_tile != 0:
    raise NotImplementedError(
        f"Scattered output size ({output_shape[0]}) must be divisible by the"
        f" inferred major tile size ({major_tile}). Consider adjusting"
        " tile_size."
    )

  def kernel(x_ref, y_ref, done_barrier):
    dev_idx = lax.axis_index(axis_name)
    x_ref = x_ref.at[pl.ds(dev_idx * output_shape[0], output_shape[0])]

    x_ref_2d = x_ref.reshape((output_shape[0], minor_dims_size))
    y_ref_2d = y_ref.reshape((output_shape[0], minor_dims_size))

    minor_tiles = minor_dims_size // minor_tile
    major_tiles = output_shape[0] // major_tile
    @plgpu.nd_loop((major_tiles, minor_tiles), collective_axes="blocks")
    def _transfer_loop(loop_info: plgpu.NDLoopInfo):
      major_tile_idx, minor_tile_idx = loop_info.index
      major_idx = major_tile_idx * major_tile
      minor_idx = minor_tile_idx * minor_tile
      idxs = pl.ds(major_idx, major_tile), pl.ds(minor_idx, minor_tile)

      y_ref_2d[idxs] = plgpu.layout_cast(
          plgpu.multimem_load_reduce(
              x_ref_2d.at[idxs], collective_axes=axis_name, reduction_op=reduction
          ),
          plgpu.Layout.WG_STRIDED((major_tile, minor_tile), vec_size=vec_size)
      )
    plgpu.semaphore_signal_multicast(done_barrier, collective_axes=axis_name)
    pl.semaphore_wait(done_barrier, num_devices, decrement=False)

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
