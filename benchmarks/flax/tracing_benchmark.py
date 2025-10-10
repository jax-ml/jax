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
"""Benchmarks for Jax tracing flax examples."""
import sys
from types import ModuleType
from typing import Any

from absl import app
from absl import flags
from absl import logging
import google_benchmark
import jax

flags.DEFINE_string(
    "example",
    None,
    "Example to benchmark. If unset, use google_benchmark to benchmark all.",
)

flags.DEFINE_enum(
    "mode",
    "trace_and_lower",
    ["trace", "lower", "trace_and_lower"],
    "Measure trace, lower, or trace_and_lower.",
)

# pylint: disable=unused-import
from flax.examples.gemma import train as gemma
from flax.examples.gemma.configs import default as gemma_config
from flax.examples.mnist import train as mnist
from flax.examples.mnist.configs import default as mnist_config
from flax.examples.wmt import train as wmt
from flax.examples.wmt.configs import default as wmt_config
# pylint: enable=unused-import


def clear_caches(state):
  state.pause_timing()
  jax.clear_caches()
  state.resume_timing()


def benchmark_tracing(
    module: ModuleType, config_module: ModuleType, state: Any
) -> None:
  """Benchmark for tracing a flax example."""
  config = config_module.get_config()  # pytype: disable=attribute-error
  apply_fn, args = module.get_apply_fn_and_args(config)  # pytype: disable=attribute-error
  while state:
    if flags.FLAGS.mode == "trace" or flags.FLAGS.mode == "trace_and_lower":
      _ = apply_fn.trace(*args)
      clear_caches(state)


def benchmark_lowering(
    module: ModuleType,
    state: Any,
    config_module: ModuleType,
    platform: str = "tpu",
) -> None:
  """Benchmark for lowering a flax example."""
  config = config_module.get_config()  # pytype: disable=attribute-error
  apply_fn, args = module.get_apply_fn_and_args(config)  # pytype: disable=attribute-error
  traced = apply_fn.trace(*args)
  while state:
    if flags.FLAGS.mode == "lower" or flags.FLAGS.mode == "trace_and_lower":
      _ = traced.lower(lowering_platforms=(platform,))
      clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_gemma_trace(state):
  benchmark_tracing(gemma, gemma_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_gemma_lower(state):
  benchmark_lowering(gemma, gemma_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_mnist_trace(state):
  benchmark_tracing(mnist, mnist_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_mnist_lower(state):
  benchmark_lowering(mnist, mnist_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_wmt_trace(state):
  benchmark_tracing(wmt, wmt_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_wmt_lower(state):
  benchmark_lowering(wmt, wmt_config, state)


def main(argv):
  del argv

  if flags.FLAGS.mode == "lower":
    raise ValueError(
        "`--mode=lower` is not supported when profiling a single example."
    )

  module = globals()[flags.FLAGS.example]
  config = globals()[flags.FLAGS.example + "_config"].get_config()
  apply_fn, args, kwargs, *_ = module.get_apply_fn_and_args(config)
  traced = apply_fn.trace(*args, **kwargs)
  lowered = traced.lower(lowering_platforms=("tpu",))

  logging.info("lowered: %s", lowered.as_text("hlo"))


if __name__ == "__main__":
  flags.FLAGS(sys.argv)
  flags.FLAGS.mark_as_parsed()

  if flags.FLAGS.example is None:
    google_benchmark.main()
  else:
    app.run(main)
