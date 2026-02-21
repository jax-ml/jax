# Copyright 2026 The JAX Authors.
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

"""
Mosaic GPU test filter used to automatically mark tests as Mosaic GPU
tests and skip them on ROCm.
"""

from __future__ import annotations

import functools
import os


# Keep these heuristics lightweight: they are used during test execution.
_MOSAIC_GPU_PATH_NEEDLES = (
    f"{os.sep}tests{os.sep}mosaic{os.sep}",
    f"{os.sep}tests{os.sep}pallas{os.sep}mgpu_",
    f"{os.sep}tests{os.sep}pallas{os.sep}mosaic_gpu",
    f"{os.sep}tests{os.sep}pallas{os.sep}mosaic",
)

_MOSAIC_GPU_SOURCE_NEEDLES = (
    # Mosaic GPU usage substrings (avoid import-only signals).
    "inline_mgpu",
    "plgpu_mgpu.",
    "mosaic_gpu_interpret",
    "mosaic_gpu_backend",
    "jax.experimental.mosaic.gpu",
    "jax.experimental.pallas.mosaic_gpu",
)


def looks_like_mosaic_gpu_path(path_str: str) -> bool:
  lowered = path_str.lower()
  return any(n.lower() in lowered for n in _MOSAIC_GPU_PATH_NEEDLES)


def source_mentions_mosaic_gpu(src_lowered: str) -> bool:
  return any(n in src_lowered for n in _MOSAIC_GPU_SOURCE_NEEDLES)


@functools.lru_cache(maxsize=None)
def class_mosaic_override_from_source(cls_src_lowered: str) -> bool | None:
  """Detects explicit class-level Mosaic enable/disable.

  Returns:
    - True if the class forces Mosaic GPU (`_PALLAS_USE_MOSAIC_GPU(True)`).
    - False if it forces Triton (`_PALLAS_USE_MOSAIC_GPU(False)`).
    - None if no explicit override is found.
  """
  if "_pallas_use_mosaic_gpu(true" in cls_src_lowered:
    return True
  if "_pallas_use_mosaic_gpu(false" in cls_src_lowered:
    return False
  return None


def is_mosaic_gpu_test_source(
    *,
    path_str: str,
    test_src: str,
    cls_src: str | None,
    running_on_rocm: bool,
    pallas_defaults_to_mosaic: bool,
) -> bool:
  """Best-effort detection of Mosaic GPU usage for a single test case."""
  if looks_like_mosaic_gpu_path(path_str):
    return True

  lowered = (test_src or "").lower()
  if source_mentions_mosaic_gpu(lowered):
    return True

  cls_src_lowered = (cls_src or "").lower()
  cls_override = class_mosaic_override_from_source(cls_src_lowered) if cls_src else None
  if cls_override is True:
    return True
  if cls_override is False:
    return False

  if running_on_rocm and pallas_defaults_to_mosaic:
    uses_pallas_call = (
        ".pallas_call" in lowered
        or "pl.pallas_call" in lowered
        or "pallas_call(" in lowered
    )
    explicitly_selects_compiler = "compiler_params=" in lowered
    if uses_pallas_call and not explicitly_selects_compiler:
      return True

  return False


def running_on_rocm() -> bool:
  """Best-effort ROCm detection.

  First tries to check rocm in jaxlib version, falls back to checking backend
  platform_version so that it works for ROCm PJRT plugin installs where jaxlib's
  version tag may not contain rocm.
  """
  try:
    import jaxlib.version as jaxlib_version  # pytype: disable=import-error
    version_str = getattr(jaxlib_version, "__version__", "")
  except ImportError:
    version_str = ""
  if "rocm" in str(version_str).lower():
    return True

  try:
    import jax  # pytype: disable=import-error  # noqa: F401
    from jax._src import xla_bridge  # pytype: disable=import-error
    backend = xla_bridge.get_backend()
    pv = getattr(backend, "platform_version", "") or ""
    return "rocm" in str(pv).lower()
  except (ImportError, RuntimeError):
    return False


def pallas_defaults_to_mosaic_gpu() -> bool:
  """Returns True if Pallas GPU lowering defaults to Mosaic GPU."""
  try:
    from jax._src.pallas import pallas_call as pallas_call_lib  # pytype: disable=import-error
    return bool(pallas_call_lib._PALLAS_USE_MOSAIC_GPU.value)  # pylint: disable=protected-access
  except (ImportError, AttributeError):
    return False
