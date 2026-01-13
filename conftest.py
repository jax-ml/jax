# Copyright 2021 The JAX Authors.
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
"""pytest configuration"""

import os
import pytest
import json
import threading
import shutil
from datetime import datetime

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
  import jax
  import numpy

  doctest_namespace["jax"] = jax
  doctest_namespace["lax"] = jax.lax
  doctest_namespace["jnp"] = jax.numpy
  doctest_namespace["np"] = numpy


# A pytest hook that runs immediately before test collection (i.e. when pytest
# loads all the test cases to run). When running parallel tests via xdist on
# GPU or Cloud TPU, we use this hook to set the env vars needed to run multiple
# test processes across different chips.
#
# It's important that the hook runs before test collection, since jax tests end
# up initializing the TPU runtime on import (e.g. to query supported test
# types). It's also important that the hook gets called by each xdist worker
# process. Luckily each worker does its own test collection.
#
# The pytest_collection hook can be used to overwrite the collection logic, but
# we only use it to set the env vars and fall back to the default collection
# logic by always returning None. See
# https://docs.pytest.org/en/latest/how-to/writing_hook_functions.html#firstresult-stop-at-first-non-none-result
# for details.
#
# For TPU, the env var JAX_ENABLE_TPU_XDIST must be set for this hook to have an
# effect. We do this to minimize any effect on non-TPU tests, and as a pointer
# in test code to this "magic" hook. TPU tests should not specify more xdist
# workers than the number of TPU chips.
#
# For GPU, the env var JAX_ENABLE_CUDA_XDIST must be set equal to the number of
# CUDA devices. Test processes will be assigned in round robin fashion across
# the devices.
def pytest_collection() -> None:
  if os.environ.get("JAX_ENABLE_TPU_XDIST", None):
    # When running as an xdist worker, will be something like "gw0"
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    os.environ.setdefault("TPU_VISIBLE_CHIPS", str(xdist_worker_number))
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")

  elif num_cuda_devices := os.environ.get("JAX_ENABLE_CUDA_XDIST", None):
    num_cuda_devices = int(num_cuda_devices)
    # When running as an xdist worker, will be something like "gw0"
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", str(xdist_worker_number % num_cuda_devices)
    )

class ThreadSafeTestLogger:
    """Thread-safe logging for parallel test execution and abort detection"""
    def __init__(self):
        self.locks = {}
        self.global_lock = threading.Lock()
        self.base_dir = os.path.abspath("./logs")
        
        # Create logs directory (archiving is handled by test runner scripts)
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            print(f"[TestLogger] Initialized log directory: {self.base_dir}")
        except Exception as e:
            print(f"[TestLogger] ERROR: Failed to create log directory {self.base_dir}: {e}")
            # Fallback to temp directory if logs dir creation fails
            import tempfile
            self.base_dir = os.path.join(tempfile.gettempdir(), "jax_test_logs")
            os.makedirs(self.base_dir, exist_ok=True)
            print(f"[TestLogger] Using fallback directory: {self.base_dir}")

    def get_file_lock(self, test_file):
        """Get or create a lock for a specific test file"""
        with self.global_lock:
            if test_file not in self.locks:
                self.locks[test_file] = threading.Lock()
            return self.locks[test_file]

    def get_test_file_name(self, session):
        """Extract the test file name from the session"""
        # Try to get from session config args
        if hasattr(session, "config") and hasattr(session.config, "args"):
            for arg in session.config.args:
                # Handle full nodeid like "jax/tests/foo_test.py::TestClass::test_method"
                if "tests/" in arg:
                    # Split on :: to get just the file path
                    file_path = arg.split("::")[0]
                    if file_path.endswith(".py"):
                        return os.path.basename(file_path).replace(".py", "")
        
        # Try to get from invocation params
        if hasattr(session, "config") and hasattr(session.config, "invocation_params"):
            invocation_dir = getattr(session.config.invocation_params, "dir", None)
            if invocation_dir:
                dir_name = os.path.basename(str(invocation_dir))
                if dir_name:
                    print(f"[TestLogger] Using invocation directory as test name: {dir_name}")
                    return dir_name
        
        # Last resort: try to get from session items
        if hasattr(session, "items") and session.items:
            first_item = session.items[0]
            if hasattr(first_item, "fspath"):
                fspath = str(first_item.fspath)
                if ".py" in fspath:
                    return os.path.basename(fspath).replace(".py", "")
        
        print(f"[TestLogger] WARNING: Could not determine test file name, using 'unknown_test'")
        print(f"[TestLogger] Session config args: {getattr(session.config, 'args', 'N/A')}")
        return "unknown_test"

    def log_running_test(self, test_file, test_name, nodeid, start_time):
        """Log the currently running test for abort detection"""
        lock = self.get_file_lock(test_file)
        with lock:
            log_data = {
                "test_file": test_file,
                "test_name": test_name,
                "nodeid": nodeid,
                "start_time": start_time,
                "status": "running",
                "pid": os.getpid(),
                "gpu_id": os.environ.get("HIP_VISIBLE_DEVICES", "unknown"),
            }

            log_file = f"{self.base_dir}/{test_file}_last_running.json"
            try:
                # Ensure directory still exists (might have been deleted)
                os.makedirs(self.base_dir, exist_ok=True)
                with open(log_file, "w") as f:
                    json.dump(log_data, f, indent=2)
            except Exception as e:
                print(f"[TestLogger] ERROR: Failed to write running test log to {log_file}: {e}")
                print(f"[TestLogger] Current working directory: {os.getcwd()}")
                print(f"[TestLogger] Base directory: {self.base_dir}")
                print(f"[TestLogger] Base directory exists: {os.path.exists(self.base_dir)}")
                raise

    def clear_running_test(self, test_file):
        """Clear the running test log when test completes successfully"""
        lock = self.get_file_lock(test_file)
        with lock:
            log_file = f"{self.base_dir}/{test_file}_last_running.json"
            if os.path.exists(log_file):
                os.remove(log_file)


# Global logger instance
test_logger = ThreadSafeTestLogger()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Hook that wraps around each test to track running tests for crash detection.
    
    This creates a "last_running" file before each test starts and deletes it
    when the test completes successfully. If the test crashes, the file remains
    and can be detected by the test runner.
    """
    test_file = test_logger.get_test_file_name(item.session)
    test_name = item.name
    nodeid = item.nodeid
    start_time = datetime.now().isoformat()

    # Log that this test is starting
    try:
        test_logger.log_running_test(test_file, test_name, nodeid, start_time)
    except Exception as e:
        print(f"[TestLogger] WARNING: Failed to log running test: {e}")
        # Continue anyway - not critical for test execution

    test_completed = False
    try:
        outcome = yield
        # Test completed (successfully or with normal failure)
        test_completed = True
        
        # Clear the crash detection file
        try:
            test_logger.clear_running_test(test_file)
        except Exception as e:
            print(f"[TestLogger] WARNING: Failed to clear running test log: {e}")
            
    except Exception as e:
        # Test raised exception (might be crash, might be normal exception)
        print(f"[TestLogger] Test {test_name} exception: {e}")
        if not test_completed:
            # Don't clear the file - this might be a crash
            print(f"[TestLogger] Leaving crash file for detection")
        raise


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    gpu = os.environ.get('HIP_VISIBLE_DEVICES', '?')
    print(f"Test session starting on GPU {gpu}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Called after test run finished.
    
    If a crash file still exists, it means a test crashed and the runner
    will detect it. We just report it here for visibility.
    """
    test_file = test_logger.get_test_file_name(session)
    log_file = f"{test_logger.base_dir}/{test_file}_last_running.json"
    
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                abort_data = json.load(f)
            print(
                f"\n[CRASH DETECTED] {abort_data.get('nodeid', abort_data.get('test_name', 'unknown'))} "
                f"(GPU: {abort_data.get('gpu_id', '?')}, PID: {abort_data.get('pid', '?')})"
            )
            print(f"[CRASH DETECTED] Crash file will be processed by test runner")
        except Exception as e:
            print(f"[TestLogger] WARNING: Crash file exists but unreadable: {e}")
    else:
        # Normal completion - no crash
        pass
