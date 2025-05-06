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

"""Helper for running multi-process tests."""

import os
import pathlib
import re
import signal
import subprocess
import time

from absl import app
from absl import flags
import jax
from jax import config
from jax._src import distributed
try:
  import portpicker
except ImportError:
  portpicker = None

from absl.testing import absltest
from jax._src import test_util as jtu


_NUM_PROCESSES = flags.DEFINE_integer(
    "num_processes", None, "Number of processes to use."
)

_GPUS_PER_PROCESS = flags.DEFINE_integer(
    "gpus_per_process",
    0,
    "Number of GPUs per worker process.",
)

_MULTIPROCESS_TEST_WORKER_ID = flags.DEFINE_integer(
    "multiprocess_test_worker_id",
    -1,
    "Worker id. Set by main test process; should not be set by users.",
)

_MULTIPROCESS_TEST_CONTROLLER_ADDRESS = flags.DEFINE_string(
    "multiprocess_test_controller_address",
    "",
    "Address of the JAX controller. Set by the main test process; should not be"
    " set by users.",
)


expect_failures_with_regex = None


def main():
  config.config_with_absl()
  app.run(_main)


class GracefulKiller:
  """Add a signal handler that sets a flag if SIGINT or SIGTERM are caught."""

  # From https://stackoverflow.com/a/31464349
  kill_now = False

  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, sig_num, unused_stack_frame):
    print(f"Caught signal: {signal.Signals(sig_num).name} ({sig_num})")
    self.kill_now = True


def _main(argv):
  if _MULTIPROCESS_TEST_WORKER_ID.value >= 0:
    jax.distributed.initialize(
        _MULTIPROCESS_TEST_CONTROLLER_ADDRESS.value,
        num_processes=_NUM_PROCESSES.value,
        process_id=_MULTIPROCESS_TEST_WORKER_ID.value,
        initialization_timeout=10,
    )
    absltest.main(testLoader=jtu.JaxTestLoader())

  if not argv[0].endswith(".py"):  # Skip the interpreter path if present.
    argv = argv[1:]

  num_processes = _NUM_PROCESSES.value
  if num_processes is None:
    raise ValueError("num_processes must be set")
  gpus_per_process = _GPUS_PER_PROCESS.value
  if portpicker is None:
    jax_port = 9876
  else:
    jax_port = portpicker.pick_unused_port()
  subprocesses = []
  output_filenames = []
  output_files = []
  for i in range(num_processes):
    env = os.environ.copy()

    args = [
        "/proc/self/exe",
        *argv,
        f"--num_processes={num_processes}",
        f"--multiprocess_test_worker_id={i}",
        f"--multiprocess_test_controller_address=localhost:{jax_port}",
        "--logtostderr",
    ]

    if gpus_per_process > 0:
      gpus = range(i * gpus_per_process, (i + 1) * gpus_per_process)
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    undeclared_outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
    stdout_name = f"{undeclared_outputs}/jax_{i}_stdout.log"
    stderr_name = f"{undeclared_outputs}/jax_{i}_stderr.log"
    stdout = open(stdout_name, "wb")
    stderr = open(stderr_name, "wb")
    print(f"Launching process {i}:")
    print(f"  stdout: {stdout_name}")
    print(f"  stderr: {stderr_name}")
    proc = subprocess.Popen(args, env=env, stdout=stdout, stderr=stderr)
    subprocesses.append(proc)
    output_filenames.append((stdout_name, stderr_name))
    output_files.append((stdout, stderr))

  print(" All launched, running ".center(80, "="), flush=True)

  # Wait for all the children to finish or for a SIGTERM from bazel. If we get
  # SIGTERM, we still want to collect their logs, so kill them and continue.
  killer = GracefulKiller()
  running_procs = dict(enumerate(subprocesses))
  while not killer.kill_now and running_procs:
    time.sleep(0.1)
    for i, proc in list(running_procs.items()):
      if proc.poll() is not None:
        print(f"Process {i} finished.", flush=True)
        running_procs.pop(i)
  if killer.kill_now and running_procs:
    print("Caught termination, terminating remaining children.", flush=True)

    # Send a SIGTERM to each child process, to let it know it should terminate.
    for i, proc in running_procs.items():
      proc.terminate()
      print(f"Process {i} terminated.", flush=True)

    # We give the child process(es) a few seconds for their own cleanup, and
    # keep the rest (up to 15s) for copying the children logs into our own.
    time.sleep(5)

    # Send a SIGKILL (a "hard" kill) to each child process. This is CRITICAL:
    # without it, this process may end up waiting a long time on the proc.wait()
    # below, and never get to saving the children logs, making test timeouts
    # very hard to debug.
    for i, proc in running_procs.items():
      proc.kill()
      print(f"Process {i} killed.")
    print("Killed all child processes.", flush=True)

  retvals = []
  stdouts = []
  stderrs = []
  for proc, fds, (stdout, stderr) in zip(
      subprocesses, output_files, output_filenames
  ):
    retvals.append(proc.wait())
    for fd in fds:
      fd.close()
    stdouts.append(pathlib.Path(stdout).read_text(errors="replace"))
    stderrs.append(pathlib.Path(stderr).read_text(errors="replace"))

  print(" All finished ".center(80, "="), flush=True)

  print(" Summary ".center(80, "="))
  for i, (retval, stdout, stderr) in enumerate(zip(retvals, stdouts, stderrs)):
    m = re.search(r"Ran \d+ tests? in [\d.]+s\n\n.*", stderr, re.MULTILINE)
    result = m.group().replace("\n\n", "; ") if m else "Test crashed?"
    print(
        f"Process {i}, ret: {retval}, len(stdout): {len(stdout)}, "
        f"len(stderr): {len(stderr)}; {result}"
    )

  print(" Detailed logs ".center(80, "="))
  for i, (retval, stdout, stderr) in enumerate(zip(retvals, stdouts, stderrs)):
    print(f" Process {i}: return code: {retval} ".center(80, "="))
    if stdout:
      print(f" Process {i} stdout ".center(80, "-"))
      print(stdout)
    if stderr:
      print(f" Process {i} stderr ".center(80, "-"))
      print(stderr)

  print(" Done detailed logs ".center(80, "="), flush=True)
  for i, (retval, stderr) in enumerate(zip(retvals, stderrs)):
    if retval != 0:
      if expect_failures_with_regex is not None:
        assert re.search(
            expect_failures_with_regex, stderr
        ), f"process {i} failed, expected regex: {expect_failures_with_regex}"
      else:
        assert retval == 0, f"process {i} failed, return value: {retval}"


class MultiProcessTest(absltest.TestCase):

  def setUp(self):
    """Start tests together."""
    super().setUp()
    assert jax.process_count() == _NUM_PROCESSES.value, (
        jax.process_count(),
        _NUM_PROCESSES.value,
    )
    # Make sure all processes are at the same test case.
    client = distributed.global_state.client
    try:
      client.wait_at_barrier(self._testMethodName + "_start", 10000)
    except jax.errors.JaxRuntimeError as e:
      msg, *_ = e.args
      if msg.startswith("DEADLINE_EXCEEDED"):
        raise RuntimeError(
            f"Init or some test executed earlier than {self._testMethodName} "
            "failed. Check logs from earlier tests to debug further. We "
            "recommend debugging that specific failed test with "
            "`--test_filter` before running the full test suite again."
        ) from e

  def tearDown(self):
    """End tests together."""
    client = distributed.global_state.client
    # Ensure a shared fate for tests where a subset of processes run different
    # test assertions (i.e. some processes may pass and some processes fail -
    # but the overall test should fail).
    try:
      client.wait_at_barrier(self._testMethodName + "_end", 10000)
    except jax.errors.JaxRuntimeError as e:
      msg, *_ = e.args
      if msg.startswith("DEADLINE_EXCEEDED"):
        raise RuntimeError(
            f"Test {self._testMethodName} failed in another process.  We "
            "recommend debugging that specific failed test with "
            "`--test_filter` before running the full test suite again."
        ) from e
    super().tearDown()
