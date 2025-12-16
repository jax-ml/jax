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

import functools
import itertools
import math
from typing import Literal

import jax
from jax import lax
from jax.experimental import multihost_utils
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp


def reduce_scatter(
    x: jax.Array,
    *,
    axis_name,
    scatter_dimension: int | None = 0,
    reduction: Literal["add", "min", "max", "and", "or", "xor"] = "add",
    num_blocks: int | None = None,
    tile_size: int | None = None,
    vec_size: int | None = None,
) -> jax.Array:
  """Performs a reduce-scatter or all-reduce operation across devices using multimem instructions.

  Args:
    x: Input array. Should be sharded across the specified axis.
    axis_name: Name of the mesh axis to reduce-scatter across.
    scatter_dimension: Axis along which to reduce-scatter. If None, performs
      all-reduce instead. Defaults to 0.
    reduction: Reduction operation to perform. Supported: "add", "min", "max",
      "and", "or", "xor".
    vec_size: Vector size for the layout. If None, automatically inferred from dtype.
    num_blocks: Number of blocks to use. Defaults to the device core count.
    tile_size: Total tile size to split across major, scatter, and minor dimensions.
  """
  num_devices = lax.axis_size(axis_name)
  input_shape = x.shape
  dtype = x.dtype
  ndim = len(input_shape)

  if num_blocks is None:
    num_blocks = backend.get_default_device().core_count

  if scatter_dimension is None:
    major_dims, scatter_dim, minor_dims = 1, math.prod(input_shape), 1
    output_scatter_dim = scatter_dim
    output_shape = input_shape
  else:
    if scatter_dimension < -ndim or scatter_dimension >= ndim:
      raise ValueError(
          f"scatter_dimension {scatter_dimension} out of bounds for array of"
          f" dimension {ndim}"
      )
    if scatter_dimension < 0:
      scatter_dimension += ndim

    scatter_dim = input_shape[scatter_dimension]
    if scatter_dim % num_devices != 0:
      raise ValueError(
          f"Scattered dimension {scatter_dimension} of input ({scatter_dim})"
          f" must be divisible by number of devices ({num_devices})"
      )

    major_dims = math.prod(input_shape[:scatter_dimension])
    minor_dims = math.prod(input_shape[scatter_dimension+1:])
    output_scatter_dim = scatter_dim // num_devices
    output_shape = (
        *input_shape[:scatter_dimension], output_scatter_dim, *input_shape[scatter_dimension + 1 :],
    )

  if (output_size := math.prod(output_shape)) % 128:
    raise ValueError("Output size must be divisible by 128")
  if jnp.issubdtype(dtype, jnp.integer):
    if vec_size is None:
      vec_size = 1  # Integer types only support unvectorized reductions
    elif vec_size != 1:
      raise ValueError("Integer types only support vec_size=1")
  elif vec_size is None:  # vec_size inference for floating point types
    dtype_bits = jnp.finfo(dtype).bits
    max_vec_size = min(128 // dtype_bits, output_size // 128)
    if tile_size is not None:
      max_vec_size_for_tile = tile_size // 128
      max_vec_size = min(max_vec_size, max_vec_size_for_tile)
    vec_size = 32 // dtype_bits  # We don't support ld_reduce below 32-bit
    while vec_size * 2 <= max_vec_size:
      vec_size *= 2
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

  minor_tile = math.gcd(tile_size, minor_dims)
  remaining_tile = tile_size // minor_tile
  scatter_tile = math.gcd(remaining_tile, output_scatter_dim)
  major_tile = remaining_tile // scatter_tile

  if major_dims % major_tile != 0:
    raise NotImplementedError(
        f"Major dimension size ({major_dims}) must be divisible by the"
        f" inferred major tile size ({major_tile}). Consider adjusting tile_size."
    )

  def kernel(x_ref, y_ref, done_barrier):
    dev_idx = lax.axis_index(axis_name)
    x_ref_3d = x_ref.reshape((major_dims, scatter_dim, minor_dims))
    y_ref_3d = y_ref.reshape((major_dims, output_scatter_dim, minor_dims))

    if scatter_dimension is not None:
      dev_slice = pl.ds(dev_idx * output_scatter_dim, output_scatter_dim)
      x_ref_3d = x_ref_3d.at[:, dev_slice, :]

    major_tiles = major_dims // major_tile
    scatter_tiles = output_scatter_dim // scatter_tile
    minor_tiles = minor_dims // minor_tile
    @plgpu.nd_loop((major_tiles, scatter_tiles, minor_tiles), collective_axes="blocks")
    def _transfer_loop(loop_info: plgpu.NDLoopInfo):
      major_tile_idx, scatter_tile_idx, minor_tile_idx = loop_info.index
      idxs = (
          pl.ds(major_tile_idx * major_tile, major_tile),
          pl.ds(scatter_tile_idx * scatter_tile, scatter_tile),
          pl.ds(minor_tile_idx * minor_tile, minor_tile)
      )

      y_ref_3d[idxs] = plgpu.layout_cast(
          plgpu.multimem_load_reduce(
              x_ref_3d.at[idxs], collective_axes=axis_name, reduction_op=reduction
          ),
          plgpu.Layout.WG_STRIDED((major_tile, scatter_tile, minor_tile), vec_size=vec_size)
      )

    # Wait for everyone to finish reading the operands before we exit and potentially free them
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


def _run_example():
  P = jax.sharding.PartitionSpec
  shape = (4 * 4096, 4 * 4096)  # This shape is global!
  dtype = jnp.bfloat16
  shards = jax.device_count()
  mesh = jax.make_mesh(
      (shards,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
  )
  jax.set_mesh(mesh)

  # We measure time per-shard and so we only need bytes per shard.
  local_in_bytes = math.prod(shape) / shards * jnp.dtype(dtype).itemsize
  # In reduce-scatter, we send (shards - 1) / shards worth of input data to the
  # switch and receive as much data as in the whole output, which is 1 / shards.
  total_bytes = local_in_bytes

  a = jax.random.normal(jax.random.key(1), shape, dtype)
  a = jax.sharding.reshard(a, P(None, "x"))

  @jax.jit
  @functools.partial(jax.shard_map, mesh=mesh, in_specs=P(None, "x"), out_specs=P(None, "x"))
  def ref_fn(x):
    return lax.psum_scatter(x, "x", scatter_dimension=1, tiled=True)
  ref_fn(a).block_until_ready()  # Warmup.
  _, ref_kernels_ms = profiler.measure(ref_fn, aggregate=False)(a)
  ref_time_us = sum(t * 1e3 for _, t in ref_kernels_ms)
  # We choose the minimum across processes to choose the runtime that didn't
  # include devices waiting for other devices.
  ref_time_us = min(multihost_utils.process_allgather(ref_time_us).tolist())
  ref_bw = total_bytes / (ref_time_us * 1e-6) / 1e9  # GB/s

  tuning_it = itertools.product(
      (4, 8, 16, 32, 64, 132),  # num_blocks
      (1024, 2048, 4096, 8192),  # tile_size
  )
  best_bw = 0.0
  best_runtime = float("inf")
  for num_blocks, tile_size in tuning_it:
    try:
      @jax.jit
      @functools.partial(
          jax.shard_map, mesh=mesh, in_specs=P(None, "x"), out_specs=P(None, "x"), check_vma=False
      )
      def kernel_fn(x):
        return reduce_scatter(x, axis_name="x", scatter_dimension=1, num_blocks=num_blocks, tile_size=tile_size)
      kernel_fn(a).block_until_ready()  # Warmup.
      _, kernels_ms = profiler.measure(kernel_fn, aggregate=False)(a)
    except ValueError as e:
      if "exceeds available shared memory" in e.args[0]:  # Ignore SMEM OOMs.
        continue
      raise
    runtime_us = sum(t * 1e3 for _, t in kernels_ms)
    runtime_us = min(multihost_utils.process_allgather(runtime_us).tolist())
    achieved_bw = total_bytes / (runtime_us * 1e-6) / 1e9  # GB/s
    if achieved_bw > best_bw:
      best_runtime = runtime_us
      best_bw = achieved_bw
    print(f"{num_blocks=}, {tile_size=}: {runtime_us:<7.1f}us = {achieved_bw:4.1f} GB/s")

  print(f"Total bytes transferred: {total_bytes / 1e9:.2f} GB")
  print(f"\tBest: {best_runtime:<7.1f}us = {best_bw:4.1f} GB/s")
  print(f"\tRef: {ref_time_us:<7.1f}us = {ref_bw:4.1f} GB/s")


if __name__ == "__main__":
  from jax._src import test_multiprocess as jt_multiprocess  # pytype: disable=import-error
  jt_multiprocess.main(shard_main=_run_example)
