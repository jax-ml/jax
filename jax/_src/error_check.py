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

import threading

import jax
from jax._src import core
from jax._src import source_info_util
from jax._src import traceback_util
import jax.numpy as jnp


Traceback = source_info_util.Traceback


traceback_util.register_exclusion(__file__)


class JaxValueError(ValueError):
  """Exception raised for failed runtime error checks in JAX."""


_NO_ERROR = jnp.iinfo(jnp.uint32).max
"""The default error code for no error.

We choose this value because when performing reductions, we can use `min` to
obtain the smallest error code.
"""


_error_code_ref: core.MutableArray | None = None
_error_list_lock = threading.Lock()
_error_list: list[tuple[str, Traceback]] = []  # (error_message, traceback) pair


def _initialize_error_code_ref() -> None:
  with core.eval_context():
    global _error_code_ref
    error_code = jnp.uint32(_NO_ERROR)
    _error_code_ref = core.mutable_array(error_code)


def set_error_if(pred: jax.Array, msg: str) -> None:
  """Set error if pred is true.

  If the error is already set, the new error will be ignored. It will not
  override the existing error.
  """
  if _error_code_ref is None:
    _initialize_error_code_ref()
    assert _error_code_ref is not None

  traceback = source_info_util.current().traceback
  assert traceback is not None
  with _error_list_lock:
    new_error_code = len(_error_list)
    _error_list.append((msg, traceback))

  pred = pred.any()
  error_code = _error_code_ref[...]
  should_update = jnp.logical_and(pred, error_code == jnp.uint32(_NO_ERROR))
  error_code = jnp.where(should_update, new_error_code, error_code)
  # TODO(ayx): support vmap and shard_map.
  _error_code_ref[...] = error_code  # pytype: disable=unsupported-operands


def raise_if_error() -> None:
  """Raise error if an error is set."""
  if _error_code_ref is None:  # if not initialized, do nothing
    return

  error_code = _error_code_ref[...]
  if error_code == jnp.uint32(_NO_ERROR):
    return
  try:
    msg, traceback = _error_list[error_code]
    exc = JaxValueError(msg)
    traceback = traceback.as_python_traceback()
    filtered_traceback = traceback_util.filter_traceback(traceback)
    raise exc.with_traceback(filtered_traceback)
  finally:
    _error_code_ref[...] = jnp.uint32(_NO_ERROR)
