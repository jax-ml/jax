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

"""Helpers for ROCm version strings in plugin/PJRT wheel metadata."""

from __future__ import annotations

import os
import re

DEFAULT_ROCM_PATH = "/opt/rocm"


def _rocm_version_from_wheel_suffix(suffix: str) -> str | None:
  """Extract the ROCm version from a wheel version suffix, if present."""
  if not suffix:
    return None
  local = suffix.rsplit("+", 1)[-1] if "+" in suffix else suffix.lstrip("+")
  match = re.search(r"(?i)(?:dev)?rocm(.+)$", local)
  return match.group(1) if match else None


def detect_rocm_version(path: str, tag: str | None) -> str:
  """Resolve a ROCm version string for package metadata."""
  if tag:
    return tag
  from_suffix = _rocm_version_from_wheel_suffix(
      os.getenv("WHEEL_VERSION_SUFFIX", "")
  )
  if from_suffix:
    return from_suffix
  rocm_ver_env = os.getenv("ROCM_VERSION")
  if rocm_ver_env:
    return rocm_ver_env
  for candidate in (path, os.path.realpath(path)):
    match = re.search(
        r"(\d+(?:\.\d+)+(?:[._-]?[a-zA-Z]+\d*)*)", candidate
    )
    if match:
      return match.group(1)
  return "unknown"
