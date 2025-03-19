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

from functools import partial
import threading
import warnings

import jax
from jax._src import core
from jax._src import source_info_util
from jax._src import traceback_util
import jax._src.mesh as mesh_lib
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


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
  """Initialize the error code ref in the current thread.

  The shape and size of the error code array depend on the mesh in the context.
  In single-device environments, the array is a scalar. In multi-device
  environments, its shape and size match those of the mesh.
  """
  with core.eval_context():
    # Get mesh from the context.
    mesh = mesh_lib.get_concrete_mesh()

    if mesh is None:  # single-device case.
      error_code = jnp.uint32(_NO_ERROR)

    else:  # multi-device case.
      sharding = NamedSharding(mesh, P(*mesh.axis_names))
      error_code = jnp.full(
          mesh.axis_sizes,
          jnp.uint32(_NO_ERROR),
          device=sharding,
      )

    _error_storage.ref = core.mutable_array(error_code)


class error_checking_context:
  """Redefine the internal error state based on the mesh in the context.

  When using JAX in multi-device environments in explicit mode, error tracking
  needs to be properly aligned with the device mesh. This context manager
  ensures that the internal error state is correctly initialized based on the
  current mesh configuration.

  This context manager should be used when starting a multi-device computation,
  or when switching between different device meshes.

  On entering the context, it initializes a new error state based on the mesh in
  the context. On exiting the context, it restores the previous error state.
  """

  __slots__ = ("old_ref",)

  def __init__(self):
    self.old_ref = None

  def __enter__(self):
    self.old_ref = _error_storage.ref
    _initialize_error_code_ref()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    _error_storage.ref = self.old_ref


def set_error_if(pred: jax.Array, /, msg: str) -> None:
  """Set the internal error state if any element of `pred` is `True`.

  This function is used inside JAX computations to detect runtime errors without
  immediately halting execution. When this function is traced (e.g., inside
  :func:`jax.jit`), the corresponding error message and its traceback are
  recorded. At execution time, if `pred` contains any `True` values, the error
  state is set, but execution continues without interruption. The recorded error
  can later be raised using :func:`raise_if_error`.

  If the error state has already been set, subsequent errors are ignored and
  will not override the existing error.

  For multi-device environments, in explicit mode, users must call
  :func:`error_checking_context()` to initialize a new error tracking state that
  matches the device mesh. In auto mode, implicit cross-device communication may
  occur inside this function, which could impact performance. A warning is
  issued in such cases.

  Args:
    pred: A JAX boolean array. If any element of `pred` is `True`, the internal
      error state will be set.
    msg: The corresponding error message to be raised later.
  """
  if _error_storage.ref is None:
    _initialize_error_code_ref()
    assert _error_storage.ref is not None

  traceback = source_info_util.current().traceback
  assert traceback is not None
  with _error_list_lock:
    new_error_code = jnp.uint32(len(_error_list))
    _error_list.append((msg, traceback))

  out_sharding = core.typeof(_error_storage.ref).sharding
  in_sharding: NamedSharding = core.typeof(pred).sharding

  # Reduce `pred`.
  if all(dim is None for dim in out_sharding.spec):  # single-device case.
    pred = pred.any()
  else:  # multi-device case.
    has_auto_axes = mesh_lib.AxisType.Auto in in_sharding.mesh.axis_types
    if has_auto_axes:  # auto mode.
      warnings.warn(
          "When at least one mesh axis of `pred` is in auto mode, calling"
          " `set_error_if` will cause implicit communication between devices."
          " To avoid this, consider converting the mesh axis in auto mode to"
          " explicit mode.",
          RuntimeWarning,
      )
      pred = pred.any()  # reduce to a single scalar
    else:  # explicit mode.
      if out_sharding.mesh != in_sharding.mesh:
        raise ValueError(
            "The error code state and the predicate must be on the same mesh, "
            f"but got {out_sharding.mesh} and {in_sharding.mesh} respectively. "
            "Please use `with error_checking_context()` to redefine the error "
            "code state based on the mesh."
        )
      pred = shard_map(
          partial(jnp.any, keepdims=True),
          mesh=out_sharding.mesh,
          in_specs=in_sharding.spec,
          out_specs=out_sharding.spec,
      )(pred)  # perform per-device reduction

  error_code = _error_storage.ref[...]
  should_update = jnp.logical_and(pred, error_code == jnp.uint32(_NO_ERROR))
  error_code = jnp.where(should_update, new_error_code, error_code)
  # TODO(ayx): support vmap and shard_map.
  _error_storage.ref[...] = error_code


def raise_if_error() -> None:
  """Raise an exception if the internal error state is set.

  This function should be called after a computation completes to check for any
  errors that were marked during execution via `set_error_if()`. If an error
  exists, it raises a `JaxValueError` with the corresponding error message.

  This function should not be called inside a traced function (e.g., inside
  :func:`jax.jit`). Doing so will raise a `ValueError`.

  Raises:
    JaxValueError: If the internal error state is set.
    ValueError: If called within a traced JAX function.
  """
  if _error_storage.ref is None:  # if not initialized, do nothing
    return

  error_code = _error_storage.ref[...].min()  # reduce to a single error code
  if isinstance(error_code, core.Tracer):
    raise ValueError(
        "raise_if_error() should not be called within a traced context, such as"
        " within a jitted function."
    )
  if error_code == jnp.uint32(_NO_ERROR):
    return
  _error_storage.ref[...] = jnp.full(
      _error_storage.ref.shape,
      jnp.uint32(_NO_ERROR),
      device=_error_storage.ref.sharding,
  )  # clear the error code

  msg, traceback = _error_list[error_code]
  exc = JaxValueError(msg)
  traceback = traceback.as_python_traceback()
  filtered_traceback = traceback_util.filter_traceback(traceback)
  raise exc.with_traceback(filtered_traceback)
