# Copyright 2020 The JAX Authors.
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

import faulthandler
import logging
import os
import signal
import time
from contextlib import contextmanager

_stack_traces_file_obj = None

logger = logging.getLogger(__name__)


def start_stack_traces_dumping(stack_traces_dir):
  """Starts stack traces dumping.

  Enable faulthandler and register signal.SIGUSR1 to collect stack traces on the
  registered signals.

  Args:
    stack_traces_dir: directory to store stack traces
  """

  global _stack_traces_file_obj
  stack_traces_file = get_stack_traces_file(stack_traces_dir)
  _stack_traces_file_obj = open(stack_traces_file, "wb")

  # Enable fault handler for SIGSEGV, SIGFPE, SIGABRT, SIGBUS and SIGILL
  faulthandler.enable(file=_stack_traces_file_obj, all_threads=True)

  # Collect python stack traces on receiving SIGUSR1 signal
  faulthandler.register(
      signal.SIGUSR1, all_threads=True, file=_stack_traces_file_obj
  )


def stop_stack_traces_dumping():
  """Disable faulthandler and unregister user signals."""
  global _stack_traces_file_obj
  if _stack_traces_file_obj is not None:
    _stack_traces_file_obj.close()
    _stack_traces_file_obj = None

  faulthandler.unregister(signal.SIGUSR1)
  faulthandler.disable()


@contextmanager
def stack_traces_dumping(stack_traces_dir):
  """Context manager to collect stack traces on the registered signals.

  Args:
    stack_traces_dir: directory to store stack traces
  """
  start_stack_traces_dumping(stack_traces_dir)
  try:
    yield
  finally:
    stop_stack_traces_dumping()


def get_stack_traces_file(traces_dir):
  """Prefix stack traces file.

  Create a file with prefix as stack_traces_ and current local time in
  '%Y_%m_%d_%H_%M_%S' format inside debugging folder in traces_dir.

  Args:
    traces_dir: directory to store stack traces
  Returns: path of stack traces file
  """

  curr_path = os.path.abspath(traces_dir)
  root_trace_folder = os.path.join(curr_path, "debugging")
  if not os.path.exists(root_trace_folder):
    os.makedirs(root_trace_folder)

  current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
  traces_file_name = "stack_traces_" + current_time + ".txt"
  stack_traces_file = os.path.join(root_trace_folder, traces_file_name)
  logger.info("Stack traces will be written in: %s", stack_traces_file)
  return stack_traces_file
