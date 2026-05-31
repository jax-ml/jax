# Copyright 2024 The JAX Authors.
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

from typing import Any


def filter_nones(d: dict) -> dict:
  return {k: v for k, v in d.items() if v is not None}


def update_metadata(a: dict[str, Any] | None, b: dict[str, Any]):
  if not b:
    return a
  val = {} if a is None else a.copy()
  val.update(b)
  return filter_nones(val)
