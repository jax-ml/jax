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
