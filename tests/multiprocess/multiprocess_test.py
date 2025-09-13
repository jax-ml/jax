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
from absl.testing import absltest
import jax
from jax import config
from jax._src import distributed
import portpicker

NUM_PROCESSES = flags.DEFINE_integer(
    "num_processes", None, "Number of processes to use."
)

GPUS_PER_PROCESS = flags.DEFINE_integer(
    "gpus_per_process",
    0,
    "Number of GPUs per worker process.",
)

TPU_CHIPS_PER_PROCESS = flags.DEFINE_integer(
    "tpu_chips_per_process",
    0,
    "Number of TPU chips per worker process.",
)

# Should we plumb this through jax.distributed.initialize to
# get_distributed_runtime_service?
_WORKER_SHUTDOWN_TIMEOUT = flags.DEFINE_integer(
    "worker_shutdown_timeout",
    15,
    "JAX shutdown timeout duration in seconds for each subprocess worker. If "
    "your test is timing out, try increasing this value.",
)

_CPU_COLLECTIVES_IMPLEMENTATION = flags.DEFINE_string(
    "cpu_collectives_implementation",
    "",
    "CPU collectives implementation to use. Uses default if empty.",
)

_EXTRA_TEST_ARGS = flags.DEFINE_multi_string(
    "extra_test_args", [], "Extra flags to pass to worker process."
)

# For internal use.
MULTIPROCESS_TEST_WORKER_ID = flags.DEFINE_integer(
    "multiprocess_test_worker_id",
    -1,
    "TPU worker id. Set by main test process; should not be set by users.",
)

MULTIPROCESS_TEST_CONTROLLER_ADDRESS = flags.DEFINE_string(
    "multiprocess_test_controller_address",
    None,
    "Address of the JAX controller. Set by main test process; should not be set"
    " by users.",
)

DEVICE_IDS = flags.DEFINE_list(
    "device_ids",
    None,
    "List of device ids to use. Set by main test process; should not be set by"
    " users.",
)

_ENABLE_MEGASCALE = flags.DEFINE_bool(
    "enable_megascale", False, "If true, enable Megascale runtime."
)

HEARTBEAT_TIMEOUT = flags.DEFINE_integer(
    "heartbeat_timeout",
    3,
    "Timeout in seconds for heartbeat checks. Set to a higher number when"
    " running under sanitizers.",
)

BARRIER_TIMEOUT = flags.DEFINE_integer(
    "barrier_timeout",
    10,
    "Barrier timeout in seconds. Set to a higher number when running under"
    " sanitizers.",
)

_DUMP_HLO = flags.DEFINE_bool(
    "dump_hlo",
    False,
    "If true, dump per-process HLO to undeclared outputs. They will show up in"
    " sponge artifacts under the directory 'jax_%process_idx%_hlo_dump'.",
)

expect_failures_with_regex = None


def main():
  # We don't call absltest.main() because if we are the main process we don't
  # want to run the tests; instead we want to fork worker processes that do.
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
  if NUM_PROCESSES.value is None:
    raise ValueError("--num_processes must be specified.")

  if MULTIPROCESS_TEST_WORKER_ID.value >= 0:
    jax.distributed.initialize(
        coordinator_address=MULTIPROCESS_TEST_CONTROLLER_ADDRESS.value,
        num_processes=NUM_PROCESSES.value,
        process_id=MULTIPROCESS_TEST_WORKER_ID.value,
        local_device_ids=[int(d) for d in DEVICE_IDS.value],
        heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT.value,
    )
    absltest.run_tests(argv, [], {})
    return

  num_processes = NUM_PROCESSES.value
  gpus_per_process = GPUS_PER_PROCESS.value
  tpu_chips_per_process = TPU_CHIPS_PER_PROCESS.value
  num_tpu_chips = num_processes * tpu_chips_per_process
  if num_tpu_chips == 0:
    pass
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
      # Note: this branch assumes we are using 2x4 GLP, and will not work with
      # 4x2 VLPs.
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

  slicebuilder_ports = [
      portpicker.pick_unused_port() for _ in range(num_processes)
  ]
  slicebuilder_addresses = ",".join(
      f"localhost:{port}" for port in slicebuilder_ports
  )
  megascale_coordinator_port = None
  coordinator_port = portpicker.pick_unused_port()

  subprocesses = []
  output_filenames = []
  output_files = []
  device_ids = None
  for i in range(num_processes):
    env = os.environ.copy()

    args = [
        "/proc/self/exe",
        f"--multiprocess_test_worker_id={i}",
        "--logtostderr",
        f"--num_processes={num_processes}",
        f"--multiprocess_test_controller_address=localhost:{coordinator_port}",
        "--vmodule=client=10,service=10",
        f"--barrier_timeout={BARRIER_TIMEOUT.value}",
    ]

    if num_tpu_chips > 0:
      device_ids = [str(i * tpu_chips_per_process + j) for j in range(tpu_chips_per_process)]
      env["CLOUD_TPU_TASK_ID"] = str(i)
      env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = tpu_chips_per_host_bounds
      env["TPU_PROCESS_BOUNDS"] = tpu_host_bounds
      env["TPU_PROCESS_ADDRESSES"] = slicebuilder_addresses
      env["TPU_PROCESS_PORT"] = str(slicebuilder_ports[i])
      env["TPU_VISIBLE_CHIPS"] = ",".join(device_ids)
      env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

    if gpus_per_process > 0:
      device_ids = [str(i * gpus_per_process + j) for j in range(gpus_per_process)]
      args.append(f"--jax_cuda_visible_devices={','.join(device_ids)}")

    if device_ids is not None:
      args.append(f"--device_ids={','.join(map(str, device_ids))}")

    cpu_collectives_impl = _CPU_COLLECTIVES_IMPLEMENTATION.value
    if cpu_collectives_impl:
      args.append(
          f"--jax_cpu_collectives_implementation={cpu_collectives_impl}"
      )

    if _ENABLE_MEGASCALE.value or cpu_collectives_impl == "megascale":
      megascale_port = portpicker.pick_unused_port()
      if megascale_coordinator_port is None:
        megascale_coordinator_port = megascale_port
      args += [
          f"--megascale_coordinator_address=localhost:{megascale_coordinator_port}",
          f"--megascale_port={megascale_port}",
      ]


    args += _EXTRA_TEST_ARGS.value

    undeclared_outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
    stdout_name = f"{undeclared_outputs}/jax_{i}_stdout.log"
    stderr_name = f"{undeclared_outputs}/jax_{i}_stderr.log"

    if _DUMP_HLO.value:
      hlo_dump_path = f"{undeclared_outputs}/jax_{i}_hlo_dump/"
      os.makedirs(hlo_dump_path, exist_ok=True)
      env["XLA_FLAGS"] = f"--xla_dump_to={hlo_dump_path}"

    # Where to call jax.distributed.initialize?
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

  # Wait for all the children to finish or for a SIGTERM from TAP. If we get
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

    # On test timeout, Forge first sends a SIGTERM (a "soft" kill signal, that
    # the test can intercept, in order to do some cleanup, log flushing, etc).
    # After a grace period of 15 seconds, Forge sends a SIGKILL (a "hard" kill),
    # see http://yaqs/eng/q/4559876738121728#n4588728130600960. We give the
    # child process(es) a few seconds for their own cleanup, and keep the rest
    # (up to 15s) for copying the children logs into our own.
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
    assert jax.process_count() == NUM_PROCESSES.value, (
        jax.process_count(),
        NUM_PROCESSES.value,
    )
    # Make sure all processes are at the same test case.
    client = distributed.global_state.client
    try:
      client.wait_at_barrier(
          f"{self._testMethodName}_start", BARRIER_TIMEOUT.value * 1000
      )
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
      client.wait_at_barrier(
          f"{self._testMethodName}_end", BARRIER_TIMEOUT.value * 1000
      )
    except jax.errors.JaxRuntimeError as e:
      msg, *_ = e.args
      if msg.startswith("DEADLINE_EXCEEDED"):
        raise RuntimeError(
            f"Test {self._testMethodName} failed in another process.  We "
            "recommend debugging that specific failed test with "
            "`--test_filter` before running the full test suite again."
        ) from e
    super().tearDown()
