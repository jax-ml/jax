# Copyright 2020 Google LLC
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
"""To run on CPU with 500 CPU devices:

CUDA_VISIBLE_DEVICES= XLA_FLAGS=--xla_force_host_platform_device_count=500 \
python3 pmap_benchmark.py

To make it run faster, set env var TARGET_TOTAL_SECS to a low number (e.g. 2).
"""
from absl import app

import jax
from jax import numpy as np
from jax import pmap
from jax.config import config

from benchmarks import benchmark

import numpy as onp


def pmap_shard_sharded_device_array_benchmark():
  """Pmap benchmark focusing on shard_args fast path.

  This is intended to measure how long it takes to dispatch a correctly-sharded
  ShardedDeviceArray to pmap.
  """

  def get_benchmark_fn(nargs, nshards):
    pmap_fn = pmap(lambda *args: np.sum(args))
    shape = (nshards, 4)
    args = [onp.random.random(shape) for _ in range(nargs)]
    sharded_args = pmap(lambda x: x)(args)
    assert all(isinstance(arg, jax.pxla.ShardedDeviceArray)
               for arg in sharded_args)
    def benchmark_fn():
      for _ in range(100):
        pmap_fn(*sharded_args)
    return benchmark_fn

  params = []
  for nargs in (10, 100, 101, 500, 1000, 5000):
    nshards = min(8, jax.local_device_count())
    params.append({"nargs": nargs, "nshards": nshards})
  for nshards in (2, 4, 8, 100, 500):
    if nshards > jax.local_device_count(): continue
    params.append({"nargs": 100, "nshards": nshards})
  benchmark.benchmark_suite(get_benchmark_fn, params,
                            "pmap_shard_sharded_device_array")


def pmap_shard_device_array_benchmark():
  """Pmap benchmark focusing on shard_args DeviceArray path.

  This is intended to measure how long it takes to dispatch a DeviceArray to
  pmap.
  """

  def get_benchmark_fn(nargs, nshards):
    pmap_fn = pmap(lambda *args: np.sum(args))
    shape = (nshards, 4)
    args = [np.array(onp.random.random(shape)) for _ in range(nargs)]
    assert all(isinstance(arg, jax.xla.DeviceArray) for arg in args)
    def benchmark_fn():
      for _ in range(10):
        pmap_fn(*args)
    return benchmark_fn

  params = []
  for nargs in (10, 100, 500):
    nshards = min(8, jax.local_device_count())
    params.append({"nargs": nargs, "nshards": nshards})
  for nshards in (2, 4, 8):
    if nshards > jax.local_device_count(): continue
    params.append({"nargs": 100, "nshards": nshards})
  benchmark.benchmark_suite(get_benchmark_fn, params, "pmap_shard_device_array")


def pmap_shard_outputs_benchmark():
  """Pmap benchmark focusing on array_result_handler path.

  This is intended to measure how long it takes to construct ShardedDeviceArrays
  from pmap.
  """
  def get_benchmark_fn(nouts, nshards):
    pmap_fn = pmap(lambda x: [x + i for i in range(nouts)])
    shape = (nshards, 4)
    arg = onp.random.random(shape)
    def benchmark_fn():
      for _ in range(100):
        pmap_fn(arg)
    return benchmark_fn

  params = []
  for nouts in (10, 100, 500, 1000, 5000):
    nshards = min(8, jax.local_device_count())
    params.append({"nouts": nouts, "nshards": nshards})
  for nshards in (2, 4, 8, 100, 500):
    if nshards > jax.local_device_count(): continue
    params.append({"nouts": 100, "nshards": nshards})
  benchmark.benchmark_suite(get_benchmark_fn, params, "pmap_shard_outputs")


def sharded_device_array_indexing_benchmark():
  """Benchmark focusing on ShardedDeviceArray indexing."""
  def get_benchmark_fn(indices_fn):
    nshards = min(8, jax.local_device_count())
    shape = (nshards, 8, 8)
    def benchmark_fn():
      arr = pmap(lambda x: x)(np.arange(np.prod(shape)).reshape(shape))
      indices = indices_fn()
      for idx in indices:
        arr[idx]
    return benchmark_fn

  num_internal_iters = 1000

  def integer_indices():
    return (i for _ in range(num_internal_iters) for i in range(8))

  def integer_2D_indices():
    return ((i,i) for _ in range(num_internal_iters) for i in range(8))

  params = []
  params.append({"indices_fn": integer_indices})
  params.append({"indices_fn": integer_2D_indices})
  benchmark.benchmark_suite(get_benchmark_fn, params,
                            "ShardedDeviceArray_indexing")


def run_all_benchmarks():
  pmap_shard_sharded_device_array_benchmark()
  pmap_shard_device_array_benchmark()
  pmap_shard_outputs_benchmark()
  sharded_device_array_indexing_benchmark()


def main(unused_argv):
  run_all_benchmarks()


if __name__ == "__main__":
  config.config_with_absl()
  app.run(main)
