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

import contextlib
from functools import partial
import threading

import jax
from jax._src import core
from jax._src import source_info_util
from jax._src import traceback_util
import jax._src.mesh as mesh_lib
from jax._src.sharding_impls import (
    NamedSharding,
    PartitionSpec as P,
    SingleDeviceSharding,
)
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp


Traceback = source_info_util.Traceback


traceback_util.register_exclusion(__file__)


class JaxValueError(ValueError):
  """Exception raised for failed runtime error checks in JAX."""


_NO_ERROR = jnp.iinfo(jnp.uint32).max
"""The default error code for no error.

This value is chosen because we can simply use `jnp.min()` to obtain the
smallest error code when performing reductions.
"""


_error_code_ref: core.MutableArray | None = None
_error_list_lock = threading.Lock()
_error_list: list[tuple[str, Traceback]] = []  # (error_message, traceback) pair


def _initialize_error_code_ref() -> None:
  with core.eval_context():
    global _error_code_ref
    error_code = jnp.uint32(_NO_ERROR)
    _error_code_ref = core.mutable_array(error_code)


@contextlib.contextmanager
def error_checking_context(mesh: mesh_lib.Mesh | None = None):
  """Redefine the error checking state based on the mesh.

  This contaxt manager should be used to when starting a multi-device
  computation, and whenever the mesh is changed.

  When exiting the context, the error checking state will be reset to the
  original state.
  """
  global _error_code_ref
  old_error_code_ref = _error_code_ref

  # If mesh is not provided, get the abstract mesh from the context.
  if mesh is None:
    mesh = mesh_lib.get_concrete_mesh()

  if mesh == ():  # single-device case.
    with core.eval_context():
      error_code = jnp.uint32(_NO_ERROR)
      _error_code_ref = core.mutable_array(error_code)

  else:  # multi-device case.
    sharding = NamedSharding(mesh, P(*mesh.axis_names))  # type: ignore
    with core.eval_context():
      error_code = jnp.full(
          mesh.axis_sizes,  # type: ignore
          jnp.uint32(_NO_ERROR),
          device=sharding,
      )
      _error_code_ref = core.mutable_array(error_code)

  try:
    yield
  finally:
    _error_code_ref = old_error_code_ref


def set_error_if(pred: jax.Array, msg: str) -> None:
  """Set error if pred is true.

  If the error is already set, the new error will be ignored. It will not
  override the existing error.

  In auto mode, this function does not work under jit.
  """
  if _error_code_ref is None:
    _initialize_error_code_ref()
    assert _error_code_ref is not None

  traceback = source_info_util.current().traceback
  assert traceback is not None
  with _error_list_lock:
    new_error_code = len(_error_list)
    _error_list.append((msg, traceback))

  if isinstance(_error_code_ref.sharding, SingleDeviceSharding):  # pytype: disable=attribute-error
    pred = pred.any()
  else:
    if _error_code_ref.sharding.mesh != pred.sharding.mesh:  # pytype: disable=attribute-error
      raise ValueError(
          "The error code state and the predicate must be on the same mesh. "
          "Please use `with error_checking_context()` to redefine the error "
          "code state based on the mesh."
      )
    pred = shard_map(
        partial(jnp.any, keepdims=True),
        mesh=_error_code_ref.sharding.mesh,  # pytype: disable=attribute-error
        in_specs=pred.sharding.spec,  # pytype: disable=attribute-error
        out_specs=_error_code_ref.sharding.spec,  # pytype: disable=attribute-error
    )(pred)

  error_code = _error_code_ref[...]
  should_update = jnp.logical_and(pred, error_code == jnp.uint32(_NO_ERROR))
  error_code = jnp.where(should_update, new_error_code, error_code)
  # TODO(ayx): support vmap and shard_map.
  _error_code_ref[...] = error_code  # pytype: disable=unsupported-operands


def raise_if_error() -> None:
  """Raise error if an error is set.

  This function should be called after the computation is finished. It should
  be used outside jit.
  """
  if _error_code_ref is None:  # if not initialized, do nothing
    return

  error_code = _error_code_ref[...].min()  # perform per-device reduction
  if error_code == jnp.uint32(_NO_ERROR):
    return
  try:
    msg, traceback = _error_list[error_code]
    exc = JaxValueError(msg)
    traceback = traceback.as_python_traceback()
    filtered_traceback = traceback_util.filter_traceback(traceback)
    raise exc.with_traceback(filtered_traceback)
  finally:
    _error_code_ref[...] = jnp.full(
        _error_code_ref.shape,
        jnp.uint32(_NO_ERROR),
        device=_error_code_ref.sharding,  # pytype: disable=attribute-error
    )
