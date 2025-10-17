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
from absl import app
from absl import flags

import os

from jax._src import repro
from jax._src.repro.tracker import (
  _thread_local_state as repro_thread_local_state
)
from jax._src.repro import reducer


FLAGS = flags.FLAGS

_START_FILE = flags.DEFINE_string(
    "start_file", None, "The repro file to reduce."
)


def main(*args):
  input_repro_path = FLAGS.start_file
  if not input_repro_path:
    raise ValueError("Must pass --start_file=")
  with open(os.path.abspath(os.path.expanduser(input_repro_path)), "r") as f:
    input_repro_source = f.read()

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
  repro_thread_local_state.initialize_state()
  repro_thread_local_state.emit_call_preprocessor = reducer.emit_call_preprocessor
  repro_thread_local_state.undefined_value_handler = reducer.undefined_value_handler

  res = repro.eval_repro(input_repro_path, input_repro_source + extra_postamble)
  stop = 1

if __name__ == '__main__':
  app.run(main)
