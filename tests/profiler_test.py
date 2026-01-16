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

import concurrent.futures
from functools import partial
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Dict, Set, Any, Tuple
import unittest
import unittest.mock
from absl.testing import absltest
import pathlib

import jax
import jax.numpy as jnp
import jax.profiler
import jax._src.test_util as jtu

from jax._src import profiler
from jax import jit


try:
    import portpicker
except ImportError:
    portpicker = None

try:
    from xprof.convert import _pywrap_profiler_plugin
    import jax.collect_profile
except ImportError:
    _pywrap_profiler_plugin = None

jax.config.parse_flags_with_absl()


# Constant for the trace viewer tool specification
_TRACE_VIEWER_TOOL_SPEC = "trace_viewer@^"
# Trace event constants
_METADATA_EVENT = "M"
_DURATION_EVENT = "X"
_PROCESS_NAME_KEY = "process_name"
_THREAD_NAME_KEY = "thread_name"
_GPU_DEVICE_MARKER = "/device:GPU"
_KERNEL_MARKER = "(Kernel)"


def _trace_viewer_json_from_xplane(pb_path: str) -> Dict[str, Any]:
    """Convert an xplane.pb into Trace Viewer JSON dict using xprof/tensorboard plugin.

    Args:
        pb_path: Path to the xplane.pb protobuf file.

    Returns:
        Dictionary containing trace viewer data parsed from the xplane protobuf.

    Raises:
        FileNotFoundError: If the pb_path does not exist.
        ImportError: If neither tensorboard_plugin_profile nor xprof is available.
        RuntimeError: If conversion or JSON decoding fails.
    """
    if not os.path.exists(pb_path):
        raise FileNotFoundError(f"XPlane file not found: {pb_path}")

    # Try xprof first (JAX's profiler tools), then tensorboard as fallback
    convert = None
    errors = []

    try:
        from xprof.convert import raw_to_tool_data as convert
    except (ImportError, AttributeError) as e:
        errors.append(f"xprof: {e}")

    if convert is None:
        try:
            from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
        except (ImportError, AttributeError) as e:
            errors.append(f"tensorboard_plugin_profile: {e}")

    if convert is None:
        raise ImportError(
            "Failed to import profiler conversion tools. " f"Tried: {', '.join(errors)}"
        )

    try:
        result, _ = convert.xspace_to_tool_data([pb_path], _TRACE_VIEWER_TOOL_SPEC, {})
    except Exception as e:
        raise RuntimeError(f"Failed to convert xplane to trace viewer data: {e}") from e

    # result is bytes in UTF-8 encoding
    try:
        return json.loads(result.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        # Show first 100 bytes of result for debugging
        preview = str(result[:100]) if isinstance(result, bytes) else str(result)[:100]
        raise RuntimeError(
            f"Failed to decode trace viewer JSON: {e}. " f"Result preview: {preview}"
        ) from e


def _count_gpu_events_from_traceevents(traceevents: List[Dict[str, Any]]) -> int:
    """
    Count GPU *kernel-stream* duration events in Trace Viewer traceEvents.

    Strategy:
    - GPU PIDs are discovered via process_name metadata that CONTAINS '/device:GPU'
    - Kernel stream TIDs are discovered via thread_name metadata containing '(Kernel)'.
    - We count only 'ph'=='X' events with 'dur' on those (pid, tid).
    - Fallback: if we can't find kernel tids, count all 'X' events on GPU pids.

    Args:
        traceevents: List of trace event dictionaries from Trace Viewer JSON.

    Returns:
        Count of GPU kernel duration events found.
    """
    gpu_pids, kernel_tids_by_pid = _extract_gpu_metadata(traceevents)

    if not gpu_pids:
        return 0

    has_kernel_tids = any(kernel_tids_by_pid.values())
    return _count_duration_events(
        traceevents, gpu_pids, kernel_tids_by_pid, has_kernel_tids
    )


def _extract_gpu_metadata(
    traceevents: List[Dict[str, Any]],
) -> Tuple[Set[int], Dict[int, Set[int]]]:
    """Extract GPU process IDs and kernel thread IDs from metadata events.

    Args:
        traceevents: List of trace event dictionaries.

    Returns:
        Tuple of (gpu_pids, kernel_tids_by_pid) where:
        - gpu_pids: Set of process IDs corresponding to GPU devices
        - kernel_tids_by_pid: Dict mapping each GPU PID to its kernel thread IDs
    """
    gpu_pids: Set[int] = set()
    kernel_tids_by_pid: Dict[int, Set[int]] = {}

    for event in traceevents:
        if event.get("ph") != _METADATA_EVENT:
            continue

        event_name = event.get("name")

        # Extract GPU process IDs
        if event_name == _PROCESS_NAME_KEY:
            process_name = event.get("args", {}).get("name", "")
            if isinstance(process_name, str) and _GPU_DEVICE_MARKER in process_name:
                pid = event.get("pid")
                if pid is not None:
                    gpu_pids.add(pid)
                    kernel_tids_by_pid.setdefault(pid, set())

        # Extract kernel thread IDs for GPU processes
        elif event_name == _THREAD_NAME_KEY:
            pid = event.get("pid")
            tid = event.get("tid")
            if pid in gpu_pids and tid is not None:
                thread_name = event.get("args", {}).get("name", "")
                if isinstance(thread_name, str) and _KERNEL_MARKER in thread_name:
                    kernel_tids_by_pid[pid].add(tid)

    return gpu_pids, kernel_tids_by_pid


def _count_duration_events(
    traceevents: List[Dict[str, Any]],
    gpu_pids: Set[int],
    kernel_tids_by_pid: Dict[int, Set[int]],
    has_kernel_tids: bool,
) -> int:
    """Count duration events on GPU processes/threads.

    Args:
        traceevents: List of trace event dictionaries.
        gpu_pids: Set of GPU process IDs.
        kernel_tids_by_pid: Mapping of GPU PIDs to kernel thread IDs.
        has_kernel_tids: Whether any kernel TIDs were found.

    Returns:
        Count of matching duration events.
    """
    count = 0

    for event in traceevents:
        # Only count duration events with a 'dur' field
        if event.get("ph") != _DURATION_EVENT or "dur" not in event:
            continue

        pid = event.get("pid")
        if pid not in gpu_pids:
            continue

        # If we have kernel TID info, filter by those; otherwise count all GPU events
        if has_kernel_tids:
            tid = event.get("tid")
            if tid in kernel_tids_by_pid.get(pid, set()):
                count += 1
        else:
            count += 1

    return count


def _find_and_read_trace_json_gz(profile_dir: str) -> Dict[str, Any]:
    """Find and read the trace.json.gz file from a profile directory.

    Args:
        profile_dir: Path to the profile directory.

    Returns:
        Parsed JSON data from the trace file.

    Raises:
        FileNotFoundError: If no trace.json.gz file is found.
    """
    import gzip

    # Search for trace.json.gz files
    trace_files = glob.glob(
        os.path.join(profile_dir, "**", "*.trace.json.gz"), recursive=True
    )
    if not trace_files:
        raise FileNotFoundError(f"No trace.json.gz found in {profile_dir}")

    # Use the first trace file found
    trace_file = trace_files[0]

    with gzip.open(trace_file, "rt", encoding="utf-8") as f:
        return json.load(f)


def _count_events_with_kernel_details(traceevents: List[Dict[str, Any]]) -> int:
    """Count trace events that contain kernel_details in their args.

    Args:
        traceevents: List of trace event dictionaries.

    Returns:
        Count of events with kernel_details.
    """
    count = 0
    for event in traceevents:
        args = event.get("args", {})
        if "kernel_details" in args:
            count += 1
    return count


def _run_child_matmul_trace_and_get_xplane(outdir: str, m: int, k: int, n: int) -> str:
    """
    Run a minimal JAX matmul trace in a fresh process. Returns path to *.xplane.pb.

    Uses (m, k) @ (k, n) -> (m, n).
    Each shape is run in its own child process for clean isolation and simpler attribution.
    """
    code = r"""
import glob, json, os
import jax, jax.numpy as jnp
from jax import jit
import jax.profiler

m = int(os.environ["MATMUL_M"])
k = int(os.environ["MATMUL_K"])
n = int(os.environ["MATMUL_N"])

@jit
def f(a, b):
  return jnp.dot(a, b)

# (m, k) @ (k, n) -> (m, n)
a = jnp.ones((m, k), dtype=jnp.float16)
b = jnp.ones((k, n), dtype=jnp.float16)

# Compile/warm-up outside trace.
f(a, b).block_until_ready()

outdir = os.environ["OUTDIR"]
with jax.profiler.trace(outdir):
  for _ in range(5):
    f(a, b).block_until_ready()

pbs = glob.glob(os.path.join(outdir, "**", "*.xplane.pb"), recursive=True)
assert len(pbs) == 1, pbs
print(json.dumps({"xplane": pbs[0]}))
"""

    env = os.environ.copy()
    env["OUTDIR"] = outdir
    env["JAX_PLATFORMS"] = "rocm"
    env["MATMUL_M"] = str(m)
    env["MATMUL_K"] = str(k)
    env["MATMUL_N"] = str(n)

    # Optional diagnostics
    # env["ROCPROFILER_LOG_LEVEL"] = "trace"
    # env["AMD_LOG_LEVEL"] = "5"

    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Child failed for shape ({m},{k})x({k},{n}).\n"
            f"STDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}"
        )

    info = json.loads(r.stdout.splitlines()[-1])
    return info["xplane"]


# We do not allow multiple concurrent profiler sessions.
@jtu.thread_unsafe_test_class()
class ProfilerTest(unittest.TestCase):
    # These tests simply test that the profiler API does not crash; they do not
    # check functional correctness.

    def setUp(self):
        if (
            sys.version_info < (3, 14)
            and hasattr(sys, "_is_gil_enabled")
            and not sys._is_gil_enabled()
        ):
            self.skipTest(
                "Profiler tests are not thread-safe under Python 3.13 free threading"
            )

        super().setUp()
        self.worker_start = threading.Event()
        self.profile_done = False

    @unittest.skipIf(not portpicker, "Test requires portpicker")
    def testStartStopServer(self):
        port = portpicker.pick_unused_port()
        jax.profiler.start_server(port=port)
        del port
        jax.profiler.stop_server()

    @unittest.skipIf(not portpicker, "Test requires portpicker")
    def testCantStartMultipleServers(self):
        port = portpicker.pick_unused_port()
        jax.profiler.start_server(port=port)
        port = portpicker.pick_unused_port()
        with self.assertRaisesRegex(
            ValueError, "Only one profiler server can be active at a time."
        ):
            jax.profiler.start_server(port=port)
        jax.profiler.stop_server()

    def testCantStopServerBeforeStartingServer(self):
        with self.assertRaisesRegex(ValueError, "No active profiler server."):
            jax.profiler.stop_server()

    def testProgrammaticProfiling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                jax.profiler.start_trace(tmpdir)
                jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                    jnp.ones(jax.local_device_count())
                )
            finally:
                jax.profiler.stop_trace()

            proto_path = glob.glob(
                os.path.join(tmpdir, "**/*.xplane.pb"), recursive=True
            )
            self.assertEqual(len(proto_path), 1)
            with open(proto_path[0], "rb") as f:
                proto = f.read()
            # Sanity check that serialized proto contains host, device, and
            # Python traces without deserializing.
            self.assertIn(b"/host:CPU", proto)
            if jtu.test_device_matches(["tpu"]):
                self.assertIn(b"/device:TPU", proto)
            self.assertIn(b"pxla.py", proto)

    def testProgrammaticProfilingConcurrency(self):
        def work():
            x = jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                jnp.ones(jax.local_device_count())
            )
            jax.block_until_ready(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                jax.profiler.start_trace(tmpdir)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    for _ in range(10):
                        executor.submit(work)
            finally:
                jax.profiler.stop_trace()

            proto_path = glob.glob(
                os.path.join(tmpdir, "**/*.xplane.pb"), recursive=True
            )
            self.assertEqual(len(proto_path), 1)
            with open(proto_path[0], "rb") as f:
                proto = f.read()
            # Sanity check that serialized proto contains host, device, and
            # Python traces without deserializing.
            self.assertIn(b"/host:CPU", proto)
            if jtu.test_device_matches(["tpu"]):
                self.assertIn(b"/device:TPU", proto)
            self.assertIn(b"pxla.py", proto)

    def testProgrammaticProfilingWithOptions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                options = jax.profiler.ProfileOptions()
                options.python_tracer_level = 0
                jax.profiler.start_trace(tmpdir, profiler_options=options)
                jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                    jnp.ones(jax.local_device_count())
                )
            finally:
                jax.profiler.stop_trace()

            proto_path = glob.glob(
                os.path.join(tmpdir, "**/*.xplane.pb"), recursive=True
            )
            self.assertEqual(len(proto_path), 1)
            with open(proto_path[0], "rb") as f:
                proto = f.read()
            # Verify that the serialized proto contains host and device traces, and
            # does not contain Python traces.
            self.assertIn(b"/host:CPU", proto)
            if jtu.test_device_matches(["tpu"]):
                self.assertIn(b"/device:TPU", proto)
            self.assertNotIn(b"pxla.py", proto)

    def testProgrammaticProfilingPathlib(self):
        with tempfile.TemporaryDirectory() as tmpdir_string:
            tmpdir = pathlib.Path(tmpdir_string)
            try:
                jax.profiler.start_trace(tmpdir)
                jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                    jnp.ones(jax.local_device_count())
                )
            finally:
                jax.profiler.stop_trace()

            proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
            self.assertEqual(len(proto_path), 1)
            proto = proto_path[0].read_bytes()
            # Sanity check that serialized proto contains host, device, and
            # Python traces without deserializing.
            self.assertIn(b"/host:CPU", proto)
            if jtu.test_device_matches(["tpu"]):
                self.assertIn(b"/device:TPU", proto)
            self.assertIn(b"pxla.py", proto)

    def testProgrammaticProfilingWithOptionsPathlib(self):
        with tempfile.TemporaryDirectory() as tmpdir_string:
            tmpdir = pathlib.Path(tmpdir_string)
            try:
                options = jax.profiler.ProfileOptions()
                options.advanced_configuration = {"tpu_trace_mode": "TRACE_ONLY_HOST"}
                jax.profiler.start_trace(tmpdir, profiler_options=options)
                jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                    jnp.ones(jax.local_device_count())
                )
            finally:
                jax.profiler.stop_trace()

            proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
            self.assertEqual(len(proto_path), 1)
            proto = proto_path[0].read_bytes()
            # Verify that the serialized proto contains host traces and does not
            # contain TPU device traces.
            self.assertIn(b"/host:CPU", proto)
            if jtu.test_device_matches(["tpu"]):
                self.assertNotIn(b"/device:TPU", proto)
            self.assertIn(b"pxla.py", proto)

    def testProfilerGetFDOProfile(self):
        # Tests stop_and_get_fod_profile could run.
        try:
            jax.profiler.start_trace("test")
            jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                jnp.ones(jax.local_device_count())
            )
        finally:
            fdo_profile = profiler.stop_and_get_fdo_profile()
        if jtu.test_device_matches(["gpu"]) and jtu.is_device_cuda():
            self.assertIn(b"copy", fdo_profile)

    def testProgrammaticProfilingErrors(self):
        with self.assertRaisesRegex(RuntimeError, "No profile started"):
            jax.profiler.stop_trace()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                jax.profiler.start_trace(tmpdir)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Profile has already been started. Only one profile may be run at a "
                    "time.",
                ):
                    jax.profiler.start_trace(tmpdir)
        finally:
            jax.profiler.stop_trace()

    def testProgrammaticProfilingContextManager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with jax.profiler.trace(tmpdir):
                jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                    jnp.ones(jax.local_device_count())
                )

            proto_path = glob.glob(
                os.path.join(tmpdir, "**/*.xplane.pb"), recursive=True
            )
            self.assertEqual(len(proto_path), 1)
            with open(proto_path[0], "rb") as f:
                proto = f.read()
            # Sanity check that serialized proto contains host and device traces
            # without deserializing.
            self.assertIn(b"/host:CPU", proto)
            if jtu.test_device_matches(["tpu"]):
                self.assertIn(b"/device:TPU", proto)

    @jtu.run_on_devices("gpu")
    @jtu.thread_unsafe_test()
    def testProgrammaticGpuCuptiTracing(self):
        @jit
        def xy_plus_z(x, y, z):
            return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z

        k = jax.random.key(0)
        s = 1, 16, 16
        jax.devices()
        x = jnp.int8(jax.random.normal(k, shape=s))
        y = jnp.bfloat16(jax.random.normal(k, shape=s))
        z = jnp.float32(jax.random.normal(k, shape=s))
        with tempfile.TemporaryDirectory() as tmpdir_string:
            tmpdir = pathlib.Path(tmpdir_string)
            with jax.profiler.trace(tmpdir):
                print(xy_plus_z(x, y, z))

            proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
            proto_bytes = proto_path[0].read_bytes()
            self.assertIn(b"/device:GPU", proto_bytes)

    @jtu.run_on_devices("gpu")
    @jtu.thread_unsafe_test()
    def testProgrammaticGpuCuptiTracingWithOptions(self):
        @jit
        def xy_plus_z(x, y, z):
            return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z

        k = jax.random.key(0)
        s = 1, 16, 16
        jax.devices()
        x = jnp.int8(jax.random.normal(k, shape=s))
        y = jnp.bfloat16(jax.random.normal(k, shape=s))
        z = jnp.float32(jax.random.normal(k, shape=s))
        with tempfile.TemporaryDirectory() as tmpdir_string:
            tmpdir = pathlib.Path(tmpdir_string)
            options = jax.profiler.ProfileOptions()
            options.advanced_configuration = {
                "gpu_max_callback_api_events": 1000000,
                "gpu_enable_nvtx_tracking": True,
            }
            with jax.profiler.trace(tmpdir):
                xy_plus_z(x, y, z).block_until_ready()

            proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
            proto_bytes = proto_path[0].read_bytes()
            self.assertIn(b"/device:GPU", proto_bytes)

    # TODO: b/443121646 - Enable PM sampling test on JAX OSS once the Github CI
    # host machine has privileged access.
    # @jtu.run_on_devices("gpu")
    # @jtu.thread_unsafe_test()
    # def testProgrammaticGpuCuptiTracingWithPmSampling(self):
    #   if not (jtu.is_cuda_compute_capability_equal("9.0")):
    #     self.skipTest("Only works on GPU with capability sm90")

    #   @jit
    #   def xy_plus_z(x, y, z):
    #     return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z

    #   k = jax.random.key(0)
    #   s = 1, 16, 16
    #   jax.devices()
    #   x = jnp.int8(jax.random.normal(k, shape=s))
    #   y = jnp.bfloat16(jax.random.normal(k, shape=s))
    #   z = jnp.float32(jax.random.normal(k, shape=s))
    #   with tempfile.TemporaryDirectory() as tmpdir_string:
    #     tmpdir = pathlib.Path(tmpdir_string)
    #     options = jax.profiler.ProfileOptions()
    #     options.advanced_configuration = {
    #         "gpu_pm_sample_counters": (
    #             "sm__cycles_active.sum"
    #         ),
    #         "gpu_pm_sample_interval_us": 500,
    #     }
    #     with jax.profiler.trace(tmpdir, profiler_options=options):
    #       xy_plus_z(x, y, z).block_until_ready()

    #     proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
    #     proto_bytes = proto_path[0].read_bytes()
    #     self.assertIn(b"/device:GPU", proto_bytes)
    #     self.assertIn(
    #         b"sm__cycles_active.sum", proto_bytes
    #     )

    def testProgrammaticProfilingContextManagerPathlib(self):
        with tempfile.TemporaryDirectory() as tmpdir_string:
            tmpdir = pathlib.Path(tmpdir_string)
            with jax.profiler.trace(tmpdir):
                jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
                    jnp.ones(jax.local_device_count())
                )

            proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
            self.assertEqual(len(proto_path), 1)
            proto = proto_path[0].read_bytes()
            # Sanity check that serialized proto contains host and device traces
            # without deserializing.
            self.assertIn(b"/host:CPU", proto)
            if jtu.test_device_matches(["tpu"]):
                self.assertIn(b"/device:TPU", proto)

    def testTraceAnnotation(self):
        x = 3
        with jax.profiler.TraceAnnotation("mycontext"):
            x = x + 2

    def testTraceFunction(self):
        @jax.profiler.annotate_function
        def f(x, *, y):
            return x + 2 * y

        self.assertEqual(f(7, y=3), 13)

        @jax.profiler.annotate_function
        def f(x, *, name):
            return x + 2 * len(name)

        self.assertEqual(f(7, name="abc"), 13)

        @partial(jax.profiler.annotate_function, name="aname")
        def g(x):
            return x + 2

        self.assertEqual(g(7), 9)

        @partial(jax.profiler.annotate_function, name="aname", akwarg="hello")
        def h(x):
            return x + 2

        self.assertEqual(h(7), 9)

    def testDeviceMemoryProfile(self):
        x = jnp.ones((20,)) + 7.0
        self.assertIsInstance(jax.profiler.device_memory_profile(), bytes)
        del x

    def _check_xspace_pb_exist(self, logdir):
        path = os.path.join(logdir, "plugins", "profile", "*", "*.xplane.pb")
        self.assertEqual(1, len(glob.glob(path)), "Expected one path match: " + path)

    @unittest.skip("Test causes OOMs")
    @unittest.skipIf(
        not (portpicker and _pywrap_profiler_plugin),
        "Test requires xprof and portpicker",
    )
    def testSingleWorkerSamplingMode(self, delay_ms=None):
        def on_worker(port, worker_start):
            jax.profiler.start_server(port)
            worker_start.set()
            x = jnp.ones((1000, 1000))
            while True:
                with jax.profiler.TraceAnnotation("atraceannotation"):
                    jnp.dot(x, x.T).block_until_ready()
                    if self.profile_done:
                        jax.profiler.stop_server()
                        break

        def on_profile(port, logdir, worker_start):
            worker_start.wait()
            options = {
                "host_tracer_level": 2,
                "python_tracer_level": 2,
                "device_tracer_level": 1,
                "delay_ms": delay_ms,
            }

            # Request for 1000 milliseconds of profile.
            duration_ms = 1000
            _pywrap_profiler_plugin.trace(
                f"localhost:{port}", logdir, "", True, duration_ms, 3, options
            )
            self.profile_done = True

        logdir = absltest.get_default_test_tmpdir()
        # Remove any existing log files.
        shutil.rmtree(logdir, ignore_errors=True)
        port = portpicker.pick_unused_port()
        thread_profiler = threading.Thread(
            target=on_profile, args=(port, logdir, self.worker_start)
        )
        thread_worker = threading.Thread(
            target=on_worker, args=(port, self.worker_start)
        )
        thread_worker.start()
        thread_profiler.start()
        thread_profiler.join()
        thread_worker.join(120)
        self._check_xspace_pb_exist(logdir)

    @unittest.skipIf(
        not (portpicker and _pywrap_profiler_plugin),
        "Test requires xprof and portpicker",
    )
    def test_remote_profiler(self):
        port = portpicker.pick_unused_port()
        jax.profiler.start_server(port)

        profile_done = threading.Event()
        logdir = absltest.get_default_test_tmpdir()
        # Remove any existing log files.
        shutil.rmtree(logdir, ignore_errors=True)

        def on_profile():
            os.system(
                f"{sys.executable} -m jax.collect_profile {port} 500 "
                f"--log_dir {logdir} --no_perfetto_link"
            )
            profile_done.set()

        thread_profiler = threading.Thread(target=on_profile, args=())
        thread_profiler.start()
        start_time = time.time()
        y = jnp.zeros((5, 5))
        while not profile_done.is_set():
            # The timeout here must be relatively high. The profiler takes a while to
            # start up on Cloud TPUs.
            if time.time() - start_time > 30:
                raise RuntimeError("Profile did not complete in 30s")
            y = jnp.dot(y, y)
        jax.profiler.stop_server()
        thread_profiler.join()
        self._check_xspace_pb_exist(logdir)

    @unittest.skip("Profiler takes >30s on Cloud TPUs")
    @unittest.skipIf(
        not (portpicker and _pywrap_profiler_plugin),
        "Test requires xprof and portpicker",
    )
    def test_remote_profiler_gcs_path(self):
        port = portpicker.pick_unused_port()
        jax.profiler.start_server(port)

        profile_done = threading.Event()
        logdir = "gs://mock-test-bucket/test-dir"
        # Mock XProf call in collect_profile.
        _pywrap_profiler_plugin.trace = unittest.mock.MagicMock()

        def on_profile():
            jax.collect_profile(port, 500, logdir, no_perfetto_link=True)
            profile_done.set()

        thread_profiler = threading.Thread(target=on_profile, args=())
        thread_profiler.start()
        start_time = time.time()
        y = jnp.zeros((5, 5))
        while not profile_done.is_set():
            # The timeout here must be relatively high. The profiler takes a while to
            # start up on Cloud TPUs.
            if time.time() - start_time > 30:
                raise RuntimeError("Profile did not complete in 30s")
            y = jnp.dot(y, y)
        jax.profiler.stop_server()
        thread_profiler.join()
        _pywrap_profiler_plugin.trace.assert_called_once_with(
            unittest.mock.ANY,
            logdir,
            unittest.mock.ANY,
            unittest.mock.ANY,
            unittest.mock.ANY,
            unittest.mock.ANY,
            unittest.mock.ANY,
        )

    # def test_advanced_configuration_getter(self):
    #     options = jax.profiler.ProfileOptions()
    #     advanced_config = {
    #         "tpu_trace_mode": "TRACE_COMPUTE",
    #         "tpu_num_sparse_cores_to_trace": 1,
    #         "enableFwThrottleEvent": True,
    #     }
    #     options.advanced_configuration = advanced_config
    #     returned_config = options.advanced_configuration
    #     self.assertDictEqual(returned_config, advanced_config)

    # test if there are GPU events when doing profiling via matmul on ROCm
    @jtu.run_on_devices("gpu")
    @jtu.thread_unsafe_test()
    def test_rocm_gpu_events_present_for_many_matmul_shapes(self):
        # ROCm-only gate using supported API
        from jax.extend import backend as jax_backend

        be = jax_backend.get_backend()
        platform_version = getattr(be, "platform_version", "") or ""
        if "rocm" not in platform_version.lower():
            self.skipTest(f"Not ROCm backend: {platform_version}")

        # test shapes:
        shapes = [
            (8, 8, 8),
            (8, 32, 32),
            (8, 128, 128),
            (8, 256, 8),
            (8, 512, 8),
            (8, 1024, 256),
            (1024, 1024, 1024),
        ]

        for m, k, n in shapes:
            with self.subTest(shape=f"{m}x{k}x{n}"):
                with tempfile.TemporaryDirectory() as td:
                    outdir = os.path.join(td, "profile")
                    xplane = _run_child_matmul_trace_and_get_xplane(outdir, m, k, n)

                    tv = _trace_viewer_json_from_xplane(xplane)
                    traceevents = tv.get("traceEvents", [])
                    gpu_events = _count_gpu_events_from_traceevents(traceevents)

                    self.assertGreater(
                        gpu_events,
                        0,
                        f"Expected >0 GPU events for matmul {m}x{n}; got {gpu_events}. xplane={xplane}",
                    )

    # Test kernel_details are present in trace.json.gz for ROCm profiling
    @jtu.run_on_devices("gpu")
    @jtu.thread_unsafe_test()
    def test_rocm_kernel_details_in_trace_json(self):
        """Test that kernel_details appear in the generated trace.json.gz file.

        This test reads the actual trace.json.gz file (not the xplane.pb conversion)
        to verify that kernel launch details (grid, block, memory) are captured.
        """
        # ROCm-only gate using supported API
        from jax.extend import backend as jax_backend

        be = jax_backend.get_backend()
        platform_version = getattr(be, "platform_version", "") or ""
        if "rocm" not in platform_version.lower():
            self.skipTest(f"Not ROCm backend: {platform_version}")

        # Use a large matmul that should definitely have kernel launches
        m, k, n = 1024, 1024, 1024

        with tempfile.TemporaryDirectory() as td:
            outdir = os.path.join(td, "profile")
            xplane = _run_child_matmul_trace_and_get_xplane(outdir, m, k, n)

            # Read the trace.json.gz file directly (not convert from xplane)
            try:
                trace_data = _find_and_read_trace_json_gz(outdir)
            except FileNotFoundError as e:
                self.fail(f"Could not find trace.json.gz file: {e}")

            traceevents = trace_data.get("traceEvents", [])
            self.assertGreater(len(traceevents), 0, "No trace events found")

            # Count events with kernel_details
            kernel_detail_count = _count_events_with_kernel_details(traceevents)

            # For a 1024x1024x1024 matmul, we should have multiple kernel launches
            # with kernel_details in their args
            self.assertGreater(
                kernel_detail_count,
                0,
                f"Expected kernel_details in trace.json.gz for matmul {m}x{k}x{n}; "
                f"found {kernel_detail_count} events with kernel_details out of {len(traceevents)} total events. "
                f"profile_dir={outdir}",
            )

            # Validate the format of kernel_details in the first few events
            events_checked = 0
            for event in traceevents:
                args = event.get("args", {})
                if "kernel_details" not in args:
                    continue

                kernel_details = args["kernel_details"]

                # Verify it's a string
                self.assertIsInstance(
                    kernel_details,
                    str,
                    f"kernel_details should be string, got {type(kernel_details)}",
                )

                # Verify it contains expected fields
                self.assertIn(
                    "grid:",
                    kernel_details,
                    f"kernel_details missing 'grid:' field: {kernel_details}",
                )
                self.assertIn(
                    "block:",
                    kernel_details,
                    f"kernel_details missing 'block:' field: {kernel_details}",
                )

                # Check at least 3 events with kernel_details
                events_checked += 1
                if events_checked >= 3:
                    break

            self.assertGreaterEqual(
                events_checked,
                1,
                "Could not find any events with kernel_details to validate",
            )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
