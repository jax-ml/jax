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


def reduce_scatter(
    x: jax.Array,
    *,
    axis_name,
    reduction: Literal["add", "min", "max", "and", "or", "xor"] = "add",
    vec_size: int = 2,
) -> jax.Array:
  """Performs a reduce-scatter operation across devices using multimem instructions.

  Args:
    x: Input array. Should be sharded across the specified axis.
    axis_name: Name of the mesh axis to reduce-scatter across.
    reduction: Reduction operation to perform. Supported: "add", "min", "max",
      "and", "or", "xor".
    vec_size: Vector size for the layout (default: 2).
  """

  num_devices = lax.axis_size(axis_name)
  input_shape = x.shape
  dtype = x.dtype

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

  def kernel(x_ref, y_ref, _, scratch_ref):
    dev_idx = lax.axis_index(axis_name)
    rows_per_device = output_shape[0]
    start_row = dev_idx * rows_per_device
    y_ref[...] = plgpu.layout_cast(
        plgpu.multimem_load_reduce(
            x_ref.at[pl.ds(start_row, rows_per_device)],
            collective_axes=axis_name,
            reduction_op=reduction,
        ),
        # TODO(apaszke): Make it possible to specify the WG_STRIDED layout, but
        # auto-infer the vec_size
        plgpu.Layout.WG_STRIDED(output_shape, vec_size=vec_size),
    )
    # TODO(apaszke): Use multimem.red to increment the semaphore
    for d in range(num_devices):
      pl.semaphore_signal(scratch_ref, device_id=d)
    pl.semaphore_wait(scratch_ref, num_devices)

  result, _ = pl.pallas_call(
      kernel,
      in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
      out_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),) * 2,
      out_shape=(jax.ShapeDtypeStruct(output_shape, dtype), x),
      scratch_shapes=[plgpu.SemaphoreType.REGULAR],
      # TODO(b/448323639): Without aliasing XLA doesn't actually
      # insert the copy that puts the operand in symmetric memory,
      # which causes the kernel to crash.
      input_output_aliases={0: 1},
  )(x)
  return result
