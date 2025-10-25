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

r"""Tool to reduce the size of a JAX repro

"""
import pathlib

from absl import app
from absl import flags

import os

from jax._src.repro import reducer


FLAGS = flags.FLAGS

_START_FILE = flags.DEFINE_string(
    "start_file", None, "The repro file to reduce."
)
_STRATEGY = flags.DEFINE_string(
    "strategy", "",
    "The reduction strategy to use."
)
_GRANULARITY = flags.DEFINE_integer(
    "granularity", 2,
    "The number of chunks to divide the set of all candidates."
    "The algorithm starts with a granularity of 2, and then doubles it "
    "and splits them in half as it tries to find smaller reductions to apply. "
)
_START_INDEX = flags.DEFINE_integer(
    "start_index", 0,
    "The index into the all candidates to start at"
)

strategies = dict(
    drop_calls=reducer.DropFunctionCallsStrategy,
    drop_expressions=reducer.DropExpressionsStrategy,
)

def main(*args):
  input_repro_path = FLAGS.start_file
  if not FLAGS.start_file:
    raise ValueError("Must pass --start_file=")
  input_repro_path = pathlib.Path(os.path.abspath(os.path.expanduser(input_repro_path)))
  with open(input_repro_path, "r") as f:
    input_repro_source = f.read()

  if not FLAGS.strategy:
    raise ValueError("Must pass --strategy=")
  if FLAGS.strategy not in strategies:
    raise ValueError(f"Unrecognized --strategy={FLAGS.strategy}. Known values: {list(strategies.keys())}")
  strategy = strategies[FLAGS.strategy]

  # Add code only at the end, to preserve locations
  extra_postamble = """
import functools

import flax
from jax._src import repro
from jax._src import traceback_util
from jax._src import config

config.enable_checks.set_local(True)

# flax.core.axes_scan.scan = repro.repro_boundary(
#   repro.bypass_repro_wrapper(flax.core.axes_scan.scan),
#   api_name="flax.core.axes_scan.scan")

orig_flax_core_axes_scan = flax.core.axes_scan.scan
@functools.partial(traceback_util.api_boundary,
                   repro_api_name="flax.core.axes_scan.scan")
def new_flax_axes_scan(*args, **kwargs):
  return orig_flax_core_axes_scan(*args, **kwargs)
flax.core.axes_scan.scan = new_flax_axes_scan

"""
  expect_error = (AssertionError, "DO_NOT_SUBMIT")
  r = reducer.Repro.make(input_repro_path, input_repro_source,
                         expect_error, strategy)

  r_min = reducer.ddmin(r, granularity=FLAGS.granularity,
                        start_index=FLAGS.start_index)


if __name__ == '__main__':
  app.run(main)
