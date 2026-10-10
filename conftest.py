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
_tpu_chip_lock_fd: int | None = None


def _acquire_tpu_chip_slot(xdist_worker_number: int, num_chips: int) -> int:
  """Acquires an exclusive lock on a free TPU chip slot in [0, num_chips).

  When a pytest-xdist worker crashes (e.g. gw0..gw7), xdist spawns a
  replacement worker with the next monotonically increasing ID (gw8, gw9, ...).
  Using non-blocking flock per chip slot ensures that replacement workers
  reclaim the crashed worker's freed TPU chip ID rather than setting
  TPU_VISIBLE_CHIPS to an out-of-range index or colliding with a live worker.
  """
  global _tpu_chip_lock_fd
  if _tpu_chip_lock_fd is not None:
    return int(os.environ.get("TPU_VISIBLE_CHIPS", xdist_worker_number))
  if num_chips <= 0:
    return xdist_worker_number
  try:
    import fcntl
  except ImportError:
    return xdist_worker_number % num_chips

  import tempfile
  import time

  uid = os.getuid() if hasattr(os, "getuid") else "shared"
  lock_dir = os.path.join(tempfile.gettempdir(), f"jax_tpu_xdist_locks_{uid}")
  os.makedirs(lock_dir, exist_ok=True)

  deadline = time.monotonic() + 30.0
  while True:
    for offset in range(num_chips):
      chip_id = (xdist_worker_number + offset) % num_chips
      lock_path = os.path.join(lock_dir, f"chip_{chip_id}.lock")
      lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
      try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _tpu_chip_lock_fd = lock_fd
        return chip_id
      except OSError:
        os.close(lock_fd)
    if time.monotonic() >= deadline:
      return xdist_worker_number % num_chips
    time.sleep(0.1)


def pytest_collection() -> None:
  if tpu_xdist := os.environ.get("JAX_ENABLE_TPU_XDIST", None):
    # When running as an xdist worker, will be something like "gw0"
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    if "TPU_VISIBLE_CHIPS" not in os.environ:
      num_chips = (
          int(tpu_xdist)
          if tpu_xdist.isdigit()
          else int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0"))
      )
      chip_id = _acquire_tpu_chip_slot(xdist_worker_number, num_chips)
      os.environ["TPU_VISIBLE_CHIPS"] = str(chip_id)
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

  elif num_rocm_devices := os.environ.get("JAX_ENABLE_ROCM_XDIST", None):
    num_rocm_devices = int(num_rocm_devices)
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    allocated = os.environ.get("ROCR_VISIBLE_DEVICES")
    allocated_tokens = (
        [t.strip() for t in allocated.split(",") if t.strip()]
        if allocated
        else []
    )
    if allocated_tokens:
      selected = allocated_tokens[xdist_worker_number % len(allocated_tokens)]
    else:
      selected = str(xdist_worker_number % num_rocm_devices)
    os.environ["ROCR_VISIBLE_DEVICES"] = selected
    # ROCR_VISIBLE_DEVICES filters HSA to a single physical device, which
    # becomes HIP index 0. The container env-file may preset
    # HIP_VISIBLE_DEVICES to all GPUs; override to "0" so HIP doesn't try to
    # enable agents that ROCr just hid.
    os.environ["HIP_VISIBLE_DEVICES"] = "0"

  elif num_oneapi_devices := os.environ.get("JAX_ENABLE_ONEAPI_XDIST", None):
    num_oneapi_devices = int(num_oneapi_devices)
    if num_oneapi_devices <= 0:
      return
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    os.environ.setdefault(
        "ZE_AFFINITY_MASK", str(xdist_worker_number % num_oneapi_devices)
    )
