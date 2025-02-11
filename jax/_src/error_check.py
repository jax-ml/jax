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

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src import source_info_util, traceback_util
from jax._src.source_info_util import Traceback
from jax._src.core import MutableArray, mutable_array


traceback_util.register_exclusion(__file__)


def get_traceback():
  return source_info_util.current().traceback


class JaxValueError(ValueError):
  pass


class _UniqueID:
  """A unique ID generator with reset support."""

  def __init__(self, start: int):
    self._start = start
    self._current = start

  def __call__(self) -> int:
    """Returns the next unique ID."""
    i = self._current
    self._current += 1
    return i

  def reset(self) -> None:
    """Resets the counter to the initial value."""
    self._current = self._start


_MAX_UINT32 = 4294967295


_error_code_ref: MutableArray = None  # pytype: disable=annotation-type-mismatch
_unique_id = _UniqueID(0)
_error_list: list[tuple[str, Traceback]] = []  # (error_message, traceback) pair


def _reset_error_code_ref() -> None:
  global _error_code_ref
  with core.eval_context():
    _error_code = jnp.uint32(_MAX_UINT32)
    _error_code_ref = mutable_array(_error_code)


def set_error_if(cond: jax.Array, msg: str) -> None:
  """Set error if cond is true."""
  if _error_code_ref is None:
    _reset_error_code_ref()

  if cond.shape != _error_code_ref.shape:
    raise ValueError(
        "Condition must be a scalar or have the same shape as the mesh"
    )

  error_code = _error_code_ref[...]
  should_update = jnp.logical_and(cond, error_code == _MAX_UINT32)
  new_error_code = jnp.where(should_update, _unique_id(), error_code)
  # TODO(ayx): support vmap and shard_map.
  _error_code_ref[...] = new_error_code
  traceback = get_traceback()
  _error_list.append((msg, traceback))  # pytype: disable=container-type-mismatch


def raise_if_error() -> None:
  """Raise error if there is any error set."""
  if _error_code_ref is None:
    return

  error_code = _error_code_ref[...].min().item()
  try:
    if error_code == _MAX_UINT32:
      return
    msg, traceback = _error_list[error_code]
    exc = JaxValueError(msg)
    traceback = traceback.as_python_traceback()
    filtered_traceback = traceback_util.filter_traceback(traceback)
    exc.with_traceback(filtered_traceback)
    raise exc
  finally:
    _reset_error_code_ref()
    _error_list.clear()
    _unique_id.reset()
