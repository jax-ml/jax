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
"""A simple Python microbenchmarking library."""

from collections import OrderedDict
from numbers import Number
import os
import time
from typing import Any, Optional, Union, Callable, List, Dict

import numpy as onp
from tabulate import tabulate

from jax.util import safe_zip


def benchmark(f: Callable[[], Any], iters: Optional[int] = None,
              warmup: Optional[int] = None, name: Optional[str] = None,
              target_total_secs: Optional[Union[int, float]] = None):
  """Benchmarks ``f``. Prints the results and returns the raw times.

  Args:
    f: The function to be benchmarked. Should take no arguments.
    iters: The number of iterations to run for. If none, runs until
      ``target_total_secs`` has elapsed.
    warmup: The number of warmup (untimed) iterations to run for.
    name: The name of the benchmark. Defaults to f.__name__.
    target_total_secs: If ``iters`` isn't specified, the minimum number of
      seconds to run for. Defaults to the env var TARGET_TOTAL_SECS or 10 if
      not set.

  Returns:
    An ndarray containing the number of seconds each iteration ran for.
  """
  if target_total_secs is None:
    target_total_secs = int(os.getenv("TARGET_TOTAL_SECS", "10"))

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
        (times.mean(), times.std(), _pstd(times), times.sum()))
  print("#iters=%d #warmup=%d" % (count, warmup))
  print()
  return times


def benchmark_suite(prepare: Callable[..., Callable], params_list: List[Dict],
                    name: str, target_total_secs: int = None):
  """Benchmarks a function for several combinations of parameters.

  Prints the summarized results in a table..

  Args:
    prepare: given kwargs returns a benchmark function specialized to the kwargs.
    params_list: a list of kwargs on which to run the benchmark.
    name: the name of this benchmark suite
    target_total_secs: the ``target_total_secs`` to pass to ``benchmark``.
 """
  # Sort parameters alphabetically so benchmark results print consistently.
  params_list = [OrderedDict(sorted(p.items())) for p in params_list]
  assert all(p.keys() == params_list[0].keys() for p in params_list)

  times = []
  for params in params_list:
    f = prepare(**params)
    subname = name + "".join("_%s=%s" % (n, p) for n, p in params.items())
    times.append(benchmark(f, name=subname,
                           target_total_secs=target_total_secs))

  print("---------Benchmark summary for %s---------" % name)
  param_names = list(params_list[0].keys())
  print(tabulate([tuple(params.values()) +
                  (t.mean(), _pstd(t), t.mean() / times[0].mean())
                  for params, t in safe_zip(params_list, times)],
                 param_names + ["mean", "%std", "relative"]))
  print()


def _pstd(x):
  return x.std() / x.mean() * 100
