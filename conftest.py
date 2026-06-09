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
import sys

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
# workers than the number of TPU chips, unless
# JAX_TPU_XDIST_VISIBILITY_MODE=devices is set for hosts where logical devices
# are a finer-grained unit than chips.
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
    tpu_visibility_mode = os.environ.get(
        "JAX_TPU_XDIST_VISIBILITY_MODE", "chips"
    )
    if tpu_visibility_mode == "devices":
      os.environ.pop("TPU_VISIBLE_CHIPS", None)
      os.environ["TPU_VISIBLE_DEVICES"] = str(xdist_worker_number)
      os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1,1"
      os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1,1"
    elif tpu_visibility_mode == "chips":
      os.environ.pop("TPU_VISIBLE_DEVICES", None)
      os.environ.pop("TPU_CHIPS_PER_PROCESS_BOUNDS", None)
      os.environ.pop("TPU_PROCESS_BOUNDS", None)
      os.environ["TPU_VISIBLE_CHIPS"] = str(xdist_worker_number)
    else:
      raise ValueError(
          "JAX_TPU_XDIST_VISIBILITY_MODE must be 'chips' or 'devices'; "
          f"got {tpu_visibility_mode!r}"
      )
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")
    if os.environ.get("JAX_TPU_XDIST_DEBUG") == "1":
      env_keys = (
          "TPU_VISIBLE_DEVICES",
          "TPU_VISIBLE_CHIPS",
          "TPU_CHIPS_PER_PROCESS_BOUNDS",
          "TPU_PROCESS_BOUNDS",
          "ALLOW_MULTIPLE_LIBTPU_LOAD",
      )
      env_summary = " ".join(
          f"{key}={os.environ.get(key, 'unset')}" for key in env_keys
      )
      print(
          "JAX TPU xdist worker "
          f"{xdist_worker_name}: mode={tpu_visibility_mode} {env_summary}",
          file=sys.stderr,
          flush=True,
      )

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
