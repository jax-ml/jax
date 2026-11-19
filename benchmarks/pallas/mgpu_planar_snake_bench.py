# Copyright 2026 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Planar snake Pallas-Mosaic-GPU benchmark."""

import functools
import statistics

from absl import app
import jax
from jax import numpy as jnp
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend


def planar_snake_kernel(grid_m: int, grid_n: int, tile_width: int) -> jax.Array:
  """Run a Pallas-Mosaic-GPU kernel benchmarking planar_snake."""

  def body(out_gmem):
    def loop_body(loop_info: plgpu.NDLoopInfo, carry):
      [lin_idx] = loop_info.index
      mi, ni = plgpu.planar_snake(
          lin_idx, (grid_m, grid_n), minor_dim=0, tile_width=tile_width
      )
      return carry + 1000 * mi + ni

    out = plgpu.nd_loop((grid_m * grid_n,), collective_axes="sm", init_carry=0)(
        loop_body
    )
    out_gmem[jax.lax.axis_index("sm")] = out

  num_sms = backend.get_default_device().core_count
  return plgpu.kernel(
      body,
      out_type=jax.ShapeDtypeStruct((num_sms,), jnp.int32),
      grid=(num_sms,),
      grid_names=("sm",),
  )()


def main(unused_argv):
  print("=" * 80)
  print("Pallas-MGPU planar_snake Benchmark")
  print("=" * 80)

  header = f"{'Shape':<12} {'TileW':<6} {'Latency / us':<10}"
  print(header)
  print("-" * len(header))

  for tile_width in (4, 5):
    for grid_m in (1024, 1025):
      for grid_n in (1024, 1025):
        shape = (grid_m, grid_n)
        benchmark_fn = functools.partial(
            planar_snake_kernel,
            grid_m=grid_m,
            grid_n=grid_n,
            tile_width=tile_width,
        )
        try:
          _, runtimes_ms = profiler.measure(benchmark_fn, iterations=20)()
        except Exception as e:
          print(
              f"Error benchmarking shape={shape}, tile_width={tile_width}: {e}"
          )
          continue

        latency_ns = statistics.median(runtimes_ms) * 1e3
        print(f"{f'{grid_m}x{grid_n}':<12} {tile_width:<6} {latency_ns:<10.2f}")


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
