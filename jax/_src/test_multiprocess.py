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

import functools
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

from absl import app
import absl.flags
from absl.testing import absltest
from absl.testing import parameterized

from jax._src import distributed
from jax._src import xla_bridge as xb
from jax._src import test_util as jtu
from jax._src.config import config
from jax._src.lib import cuda_versions
from jax._src.lib import _jax

try:
  import portpicker  # pytype: disable=import-error
except ImportError:
  portpicker = None

NUM_PROCESSES = absl.flags.DEFINE_integer(
    "num_processes", None, "Number of processes to use."
)

_GPUS_PER_PROCESS = absl.flags.DEFINE_integer(
    "gpus_per_process",
    0,
    "Number of GPUs per worker process.",
)

_TPU_CHIPS_PER_PROCESS = absl.flags.DEFINE_integer(
    "tpu_chips_per_process",
    0,
    "Number of TPU chips per worker process.",
)

CPU_COLLECTIVES_IMPLEMENTATION = absl.flags.DEFINE_string(
    "cpu_collectives_implementation",
    "",
    "CPU collectives implementation to use. Uses default if empty.",
)

EXTRA_TEST_ARGS = absl.flags.DEFINE_multi_string(
    "extra_test_args", [], "Extra flags to pass to worker process."
)

# For internal use.
MULTIPROCESS_TEST_WORKER_ID = absl.flags.DEFINE_integer(
    "multiprocess_test_worker_id",
    -1,
    "Worker id. Set by main test process; should not be set by users.",
)

_MULTIPROCESS_TEST_CONTROLLER_ADDRESS = absl.flags.DEFINE_string(
    "multiprocess_test_controller_address",
    "",
    "Address of the JAX controller. Set by the main test process; should not be"
    " set by users.",
)

_DEVICE_IDS = absl.flags.DEFINE_list(
    "device_ids",
    None,
    "List of device ids to use. Set by main test process; should not be set by"
    " users.",
)

_ENABLE_MEGASCALE = absl.flags.DEFINE_bool(
    "enable_megascale", False, "If true, enable Megascale runtime."
)

_HEARTBEAT_TIMEOUT = absl.flags.DEFINE_integer(
    "heartbeat_timeout",
    5,
    "Timeout in seconds for heartbeat checks. Set to a higher number when"
    " running under sanitizers.",
)

_SHUTDOWN_TIMEOUT = absl.flags.DEFINE_integer(
    "shutdown_timeout",
    15,
    "JAX shutdown timeout duration in seconds for each subprocess worker. If "
    "your test is timing out, try increasing this value.",
)

_BARRIER_TIMEOUT = absl.flags.DEFINE_integer(
    "barrier_timeout",
    10,
    "Barrier timeout in seconds. Set to a higher number when running under"
    " sanitizers.",
)

_INITIALIZATION_TIMEOUT = absl.flags.DEFINE_integer(
    "initialization_timeout",
    10,
    "Coordination service initialization timeout in seconds. Set to a higher"
    " number when running under sanitizers.",
)

_DUMP_HLO = absl.flags.DEFINE_bool(
    "dump_hlo",
    False,
    "If true, dump per-process HLO to undeclared outputs. They will show up in"
    " sponge artifacts under the directory 'jax_%process_idx%_hlo_dump'.",
)

expect_failures_with_regex = None


def main(shard_main=None):
  config.config_with_absl()
  app.run(functools.partial(_main, shard_main=shard_main))


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


def _main(argv, shard_main):
  # TODO(emilyaf): Enable multiprocess tests on Windows.
  if sys.platform == "win32":
    print("Multiprocess tests are not supported on Windows.")
    return
  num_processes = NUM_PROCESSES.value
  if MULTIPROCESS_TEST_WORKER_ID.value >= 0:
    local_device_ids = _DEVICE_IDS.value
    if local_device_ids is not None:
      local_device_ids = [int(device_id) for device_id in local_device_ids]
    distributed.initialize(
        _MULTIPROCESS_TEST_CONTROLLER_ADDRESS.value,
        num_processes=num_processes,
        process_id=MULTIPROCESS_TEST_WORKER_ID.value,
        local_device_ids=local_device_ids,
        heartbeat_timeout_seconds=_HEARTBEAT_TIMEOUT.value,
        shutdown_timeout_seconds=_SHUTDOWN_TIMEOUT.value,
        initialization_timeout=_INITIALIZATION_TIMEOUT.value,
    )
    if shard_main is not None:
      return shard_main()
    return absltest.main(testLoader=jtu.JaxTestLoader())

  if not argv[0].endswith(".py"):  # Skip the interpreter path if present.
    argv = argv[1:]

  if num_processes is None:
    raise ValueError("num_processes must be set")
  gpus_per_process = _GPUS_PER_PROCESS.value
  tpu_chips_per_process = _TPU_CHIPS_PER_PROCESS.value
  num_tpu_chips = num_processes * tpu_chips_per_process
  if num_tpu_chips == 0:
    tpu_host_bounds = ""
    tpu_chips_per_host_bounds = ""
  elif num_tpu_chips == 1:
    assert tpu_chips_per_process == 1
    tpu_host_bounds = "1,1,1"
    tpu_chips_per_host_bounds = "1,1,1"
  elif num_tpu_chips == 4:
    if tpu_chips_per_process == 1:
      tpu_host_bounds = "2,2,1"
      tpu_chips_per_host_bounds = "1,1,1"
    elif tpu_chips_per_process == 2:
      tpu_host_bounds = "2,1,1"
      tpu_chips_per_host_bounds = "1,2,1"
    elif tpu_chips_per_process == 4:
      tpu_host_bounds = "1,1,1"
      tpu_chips_per_host_bounds = "2,2,1"
    else:
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  elif num_tpu_chips == 8:
    if tpu_chips_per_process == 1:
      tpu_host_bounds = "4,2,1"
      tpu_chips_per_host_bounds = "1,1,1"
    elif tpu_chips_per_process == 4:
      # Note: this branch assumes we are using 2x4 v6e LitePod, and will not
      # work with 4x2 v5e LitePod.
      tpu_host_bounds = "1,2,1"
      tpu_chips_per_host_bounds = "2,2,1"
    elif tpu_chips_per_process == 8:
      tpu_host_bounds = "1,1,1"
      tpu_chips_per_host_bounds = "2,4,1"
    else:
      # TODO(phawkins): implement other cases.
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  else:
    raise ValueError(f"Invalid number of TPU chips {num_tpu_chips}")

  if portpicker is None:
    slicebuilder_ports = [10000 + i for i in range(num_processes)]
  else:
    slicebuilder_ports = [
        portpicker.pick_unused_port() for _ in range(num_processes)
    ]
  slicebuilder_addresses = ",".join(
      f"localhost:{port}" for port in slicebuilder_ports
  )
  megascale_coordinator_port = None

  if gpus_per_process > 0:
    # Get the number of GPUs visible to this process without initializing the runtime
    if cuda_versions is not None:
      local_device_count = cuda_versions.cuda_device_count()
      if num_processes * gpus_per_process > local_device_count:
        print(
          f"Cannot run {num_processes} processes with {gpus_per_process} GPU(s) "
          f"each on a system with only {local_device_count} local GPU(s), "
          f"starting {local_device_count // gpus_per_process} instead - test "
          "cases will likely be skipped!"
        )
        num_processes = local_device_count // gpus_per_process

  if portpicker is None:
    jax_port = 9876
  else:
    # TODO(emilyaf): Use a port server if there are flaky port collisions due
    # to pick_unused_port() racing among tests.
    jax_port = portpicker.pick_unused_port()
  subprocesses = []
  output_filenames = []
  output_files = []
  sys_path = os.pathsep.join(sys.path)

  for i in range(num_processes):
    device_ids = None
    env = os.environ.copy()

    # Note: Fix for rules_python >= 1.7.0 (Strict Hermeticity):
    # The parent process sees dependencies via sys.path, but modern rules_python
    # does not export this to PYTHONPATH by default. We must manually propagate
    # it so child workers can locate dependencies.
    path_parts = [sys_path, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in path_parts if p)

    args = [
        "/proc/self/exe",
        *argv,
        f"--num_processes={num_processes}",
        f"--multiprocess_test_worker_id={i}",
        f"--multiprocess_test_controller_address=localhost:{jax_port}",
        f"--heartbeat_timeout={_HEARTBEAT_TIMEOUT.value}",
        f"--shutdown_timeout={_SHUTDOWN_TIMEOUT.value}",
        f"--barrier_timeout={_BARRIER_TIMEOUT.value}",
        f"--initialization_timeout={_INITIALIZATION_TIMEOUT.value}",
        "--logtostderr",
    ]

    if num_tpu_chips > 0:
      device_ids = range(
          i * tpu_chips_per_process, (i + 1) * tpu_chips_per_process)
      env["CLOUD_TPU_TASK_ID"] = str(i)
      env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = tpu_chips_per_host_bounds
      env["TPU_PROCESS_BOUNDS"] = tpu_host_bounds
      env["TPU_PROCESS_ADDRESSES"] = slicebuilder_addresses
      env["TPU_PROCESS_PORT"] = str(slicebuilder_ports[i])
      env["TPU_VISIBLE_CHIPS"] = ",".join(map(str, device_ids))
      env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

    if gpus_per_process > 0:
      device_ids = range(i * gpus_per_process, (i + 1) * gpus_per_process)
      args.append(f"--jax_cuda_visible_devices={','.join(map(str, device_ids))}")

    if device_ids is not None:
      args.append(f"--device_ids={','.join(map(str, device_ids))}")

    cpu_collectives_impl = CPU_COLLECTIVES_IMPLEMENTATION.value
    if cpu_collectives_impl:
      args.append(
          f"--jax_cpu_collectives_implementation={cpu_collectives_impl}"
      )

    if _ENABLE_MEGASCALE.value or cpu_collectives_impl == "megascale":
      if portpicker is None:
        megascale_port = 9877
      else:
        megascale_port = portpicker.pick_unused_port()
      if megascale_coordinator_port is None:
        megascale_coordinator_port = megascale_port
      args += [
          f"--megascale_coordinator_address=localhost:{megascale_coordinator_port}",
          f"--megascale_port={megascale_port}",
      ]

    args += EXTRA_TEST_ARGS.value

    undeclared_outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
    stdout_name = f"{undeclared_outputs}/jax_{i}_stdout.log"
    stderr_name = f"{undeclared_outputs}/jax_{i}_stderr.log"

    if _DUMP_HLO.value:
      hlo_dump_path = f"{undeclared_outputs}/jax_{i}_hlo_dump/"
      os.makedirs(hlo_dump_path, exist_ok=True)
      env["XLA_FLAGS"] = f"--xla_dump_to={hlo_dump_path}"

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


class MultiProcessTest(parameterized.TestCase):

  def setUp(self):
    """Start tests together."""
    super().setUp()
    if xb.process_count() == 1:
      self.skipTest("Test requires multiple processes.")
    assert xb.process_count() == NUM_PROCESSES.value, (
        xb.process_count(),
        NUM_PROCESSES.value,
    )
    # Make sure all processes are at the same test case.
    client = distributed.global_state.client
    if client is None:
      raise TypeError("client cannot be None")
    try:
      client.wait_at_barrier(
          f"{self._testMethodName}_start", _BARRIER_TIMEOUT.value * 1000)
    except _jax.JaxRuntimeError as e:
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
    if client is None:
      raise TypeError("client cannot be None")
    # Ensure a shared fate for tests where a subset of processes run different
    # test assertions (i.e. some processes may pass and some processes fail -
    # but the overall test should fail).
    try:
      client.wait_at_barrier(
          f"{self._testMethodName}_end", _BARRIER_TIMEOUT.value * 1000)
    except _jax.JaxRuntimeError as e:
      msg, *_ = e.args
      if msg.startswith("DEADLINE_EXCEEDED"):
        raise RuntimeError(
            f"Test {self._testMethodName} failed in another process.  We "
            "recommend debugging that specific failed test with "
            "`--test_filter` before running the full test suite again."
        ) from e
    super().tearDown()
