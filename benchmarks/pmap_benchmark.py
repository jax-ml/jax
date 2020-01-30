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

CUDA_VISIBLE_DEVICES= XLA_FLAGS=--xla_force_host_platform_device_count=500 python3 pmap_benchmark.py

To make it run faster, set env var TARGET_TOTAL_SECS to a low number (e.g. 2).
"""

import os
import time

import numpy as onp
from tabulate import tabulate

import jax
import jax.numpy as np
from jax import pmap
from jax.util import safe_zip

def pstd(x):
  return x.std() / x.mean() * 100

def benchmark(f, iters=None, warmup=None, name=None, target_total_secs=None):
  if target_total_secs is None:
    target_total_secs = int(os.getenv("TARGET_TOTAL_SECS", 10))

  if warmup is None:
    if iters is None:
      warmup = 1
    else:
      warmup = onp.clip(1, iters // 10, 10)
  for _ in range(warmup):
    f()

  times = []
  count = 0
  while (count < iters if iters is not None
         else sum(times) < target_total_secs):
    start = time.time()
    f()
    end = time.time()
    times.append(end - start)
    count += 1

  times = onp.array(times)
  print("---------Benchmark results for %s---------" % (name or f.__name__))
  print("mean=%f std=%f %%std=%f total=%f" %
        (times.mean(), times.std(), pstd(times), times.sum()))
  print("#iters=%d #warmup=%d" % (count, warmup))
  print()
  return times


def benchmark_suite(funcs, params_list, param_names, name,
                    target_total_secs=None):
  times = []
  for f, params in safe_zip(funcs, params_list):
    subname = name + "".join("_%s=%s" % (n, p)
                             for n, p in safe_zip(param_names, params))
    times.append(benchmark(f, name=subname,
                           target_total_secs=target_total_secs))

  print("---------Benchmark summary for %s---------" % name)
  print(tabulate([tuple(params) +
                  (t.mean(), pstd(t), t.mean() / times[0].mean())
                  for params, t in safe_zip(params_list, times)],
                 param_names + ["mean", "%std", "relative"]))


def pmap_shard_args_benchmark():
  """Pmap benchmark focusing on shard_args fast path.

  This is intended to measure how long it takes to dispatch a correctly-sharded
  ShardedDeviceArray to pmap.
  """

  def get_benchmark_fn(nargs, nshards):
    shape = (nshards, 4)
    args = [onp.random.random(shape) for _ in range(nargs)]
    sharded_args = pmap(lambda x: x)(args)
    assert all(type(arg) == jax.pxla.ShardedDeviceArray for arg in sharded_args)
    pmap_fn = pmap(lambda *args: np.sum(args))
    def benchmark_fn():
      for _ in range(100):
        pmap_fn(*sharded_args)
    return benchmark_fn

  params = []
  for nargs in (10, 100, 101, 500):
    nshards = min(4, jax.local_device_count())
    params.append((nargs, nshards))
  for nshards in (2, 4, 8, 100, 500):
    if nshards > jax.local_device_count(): continue
    params.append((10, nshards))
  funcs = [get_benchmark_fn(nargs, nshards) for nargs, nshards in params]
  benchmark_suite(funcs, params, ["nargs", "nshards"], "pmap_shard_args")


def run_all_benchmarks():
  pmap_shard_args_benchmark()

if __name__ == "__main__":
  run_all_benchmarks()
