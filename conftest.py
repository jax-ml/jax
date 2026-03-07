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

import inspect
import os

import pytest

from jax._src.internal_test_util import mosaic_gpu_test_filter as _mosaic_filter


def _is_mosaic_gpu_item(
    item: pytest.Item,
    cache: dict[object, bool],
    *,
    running_on_rocm: bool,
    pallas_defaults_to_mosaic: bool,
) -> bool:
  """Returns True if this test item uses Mosaic GPU."""
  path_obj = getattr(item, "path", None) or getattr(item, "fspath", None)
  path_str = str(path_obj) if path_obj is not None else ""

  if _mosaic_filter.looks_like_mosaic_gpu_path(path_str):
    return True

  obj = getattr(item, "obj", None)
  if obj is None:
    return False
  if obj in cache:
    return cache[obj]
  try:
    src = inspect.getsource(obj)
  except (TypeError, OSError):
    cache[obj] = False
    return False

  cls_src = None
  cls_obj = getattr(item, "cls", None)
  if cls_obj is not None:
    try:
      cls_src = inspect.getsource(cls_obj)
    except (TypeError, OSError):
      cls_src = None

  is_mosaic = _mosaic_filter.is_mosaic_gpu_test_source(
      path_str=path_str,
      test_src=src,
      cls_src=cls_src,
      running_on_rocm=running_on_rocm,
      pallas_defaults_to_mosaic=pallas_defaults_to_mosaic,
  )
  cache[obj] = is_mosaic
  return is_mosaic


def pytest_configure(config: pytest.Config) -> None:
  """Register custom pytest markers."""
  config.addinivalue_line(
      "markers",
      "mosaic_gpu: tests that use Mosaic GPU (skipped on ROCm)",
  )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
  """Mark Mosaic GPU tests and skip them on ROCm."""
  running_on_rocm = _mosaic_filter.running_on_rocm()
  pallas_defaults_to_mosaic = (
      _mosaic_filter.pallas_defaults_to_mosaic_gpu() if running_on_rocm else False
  )
  cache: dict[object, bool] = {}
  for item in items:
    is_mosaic_gpu = _is_mosaic_gpu_item(
        item,
        cache,
        running_on_rocm=running_on_rocm,
        pallas_defaults_to_mosaic=pallas_defaults_to_mosaic,
    )
    if not is_mosaic_gpu:
      continue
    item.add_marker(pytest.mark.mosaic_gpu)
    if running_on_rocm:
      item.add_marker(pytest.mark.skip(
          reason="Mosaic GPU tests are not supported on ROCm"
      ))


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
