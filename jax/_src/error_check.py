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


#: The default error code for no error.
#:
#: This value is chosen because we can use `jnp.min()` to obtain the
#: first error when performing reductions.
_NO_ERROR = jnp.iinfo(jnp.uint32).max


_error_list_lock = threading.Lock()
_error_list: list[tuple[str, Traceback]] = []  # (error_message, traceback) pair


class _ErrorStorage(threading.local):

  def __init__(self):
    self.ref: core.MutableArray | None = None


_error_storage = _ErrorStorage()


def _initialize_error_code_ref() -> None:
  """Initialize error_code_ref in the current thread."""
  with core.eval_context():
    error_code = jnp.uint32(_NO_ERROR)
    _error_storage.ref = core.mutable_array(error_code)


def set_error_if(pred: jax.Array, msg: str) -> None:
  """Set error if any element of pred is true.

  If the error is already set, the new error will be ignored. It will not
  override the existing error.
  """
  if _error_storage.ref is None:
    _initialize_error_code_ref()
    assert _error_storage.ref is not None

  traceback = source_info_util.current().traceback
  assert traceback is not None
  with _error_list_lock:
    new_error_code = jnp.uint32(len(_error_list))
    _error_list.append((msg, traceback))

  pred = pred.any()
  error_code = _error_storage.ref[...]
  should_update = jnp.logical_and(pred, error_code == jnp.uint32(_NO_ERROR))
  error_code = jnp.where(should_update, new_error_code, error_code)
  # TODO(ayx): support vmap and shard_map.
  _error_storage.ref[...] = error_code


def raise_if_error() -> None:
  """Raise error if an error is set.

  This function should be called after the computation is finished. It should
  be used outside jit.
  """
  if _error_storage.ref is None:  # if not initialized, do nothing
    return

  error_code = _error_storage.ref[...]
  if error_code == jnp.uint32(_NO_ERROR):
    return
  _error_storage.ref[...] = jnp.uint32(_NO_ERROR)

  msg, traceback = _error_list[error_code]
  exc = JaxValueError(msg)
  traceback = traceback.as_python_traceback()
  filtered_traceback = traceback_util.filter_traceback(traceback)
  raise exc.with_traceback(filtered_traceback)
