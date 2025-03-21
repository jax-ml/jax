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
import dataclasses
from functools import partial
import json
import threading
import traceback as tb_lib
from types import TracebackType
from typing import Literal
import warnings

import jax
from jax._src import config
from jax._src import core
from jax._src import source_info_util
from jax._src import traceback_util
import jax._src.mesh as mesh_lib
from jax.experimental import shard_map
import jax.export
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


traceback_util.register_exclusion(__file__)


class JaxValueError(ValueError):
  """Exception raised for runtime errors detected within JAX computations."""


#: The default error code for no error.
#:
#: This value is chosen because we can use `jnp.min()` to obtain the
#: first error when performing reductions.
_NO_ERROR = jnp.iinfo(jnp.uint32).max


_error_list_lock = threading.RLock()
# (error_message, traceback) pairs. Traceback is `str` when imported from AOT.
_error_list: list[tuple[str, TracebackType | str]] = []


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
    with core.eval_context():
      _initialize_error_code_ref()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    _error_storage.ref = self.old_ref


# TODO(ayx): Move all category-related logic into the jax.numpy integration once
# it is ready. This logic is specific to how jax.numpy decides when to call
# `set_error_if`, and doesn't belong in the core error-checking library itself.
# The responsibility for deciding whether to predicate an error should lie with
# the user or the higher-level library (like jax.numpy), not with
# `set_error_if`.
Category = Literal["nan", "divide", "oob"]


def _is_category_disabled(
    category: Category | None,
) -> bool:
  """Check if the error checking behavior for the given category is disabled."""
  if category is None:
    return False
  if category == "nan":
    return config.error_checking_behavior_nan.value == "ignore"
  if category == "divide":
    return config.error_checking_behavior_divide.value == "ignore"
  if category == "oob":
    return config.error_checking_behavior_oob.value == "ignore"
  raise ValueError(f"Invalid category: {category}")


def _set_error_if_with_category(
    pred: jax.Array,
    /,
    msg: str,
    category: Category | None = None,
) -> None:
  """Set the internal error state if any element of `pred` is `True`.

  This function is similar to :func:`set_error_if`, but it also takes a category
  argument. The category can be "nan", "divide", or "oob". The error checking
  behavior for each category can be configured using
  :func:`set_error_checking_behavior`. If not provided, there will be no
  category.

  This function is intended for use in JAX internal APIs (e.g., `jax.numpy`)
  to perform category-specific runtime checks tied to the operation being
  performed.
  """
  if _error_storage.ref is None:
    with core.eval_context():
      _initialize_error_code_ref()
    assert _error_storage.ref is not None

  if _is_category_disabled(category):
    return

  # Get the traceback.
  traceback = source_info_util.current().traceback
  assert traceback is not None
  traceback = traceback.as_python_traceback()
  assert isinstance(traceback, TracebackType)
  traceback = traceback_util.filter_traceback(traceback)
  assert isinstance(traceback, TracebackType)

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
      pred = shard_map.shard_map(
          partial(jnp.any, keepdims=True),
          mesh=out_sharding.mesh,
          in_specs=in_sharding.spec,
          out_specs=out_sharding.spec,
      )(pred)  # perform per-device reduction

  error_code = _error_storage.ref[...]
  should_update = jnp.logical_and(error_code == jnp.uint32(_NO_ERROR), pred)
  error_code = jnp.where(should_update, new_error_code, error_code)
  # TODO(ayx): support vmap and shard_map.
  _error_storage.ref[...] = error_code


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
  :func:`error_checking_context` to initialize a new error tracking state that
  matches the device mesh. In auto mode, implicit cross-device communication may
  occur inside this function, which could impact performance. A warning is
  issued in such cases.

  When exporting a function with `jax.export`, error checking must be explicitly
  wrapped using :func:`wrap_for_export` before export and
  :func:`unwrap_from_import` after import.

  Args:
    pred: A JAX boolean array. If any element of `pred` is `True`, the internal
      error state will be set.
    msg: The corresponding error message to be raised later.
  """
  _set_error_if_with_category(pred, msg)


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

  with _error_list_lock:
    msg, traceback = _error_list[error_code]
  if isinstance(traceback, str):  # from imported AOT functions
    exc = JaxValueError(
        f"{msg}\nThe original traceback is shown below:\n{traceback}"
    )
    raise exc
  else:
    exc = JaxValueError(msg)
    raise exc.with_traceback(traceback)


@dataclasses.dataclass(frozen=True)
class _ErrorClass:
  """A class to store error information for AOT compilation.

  This class is used internally by the wrapper functions `wrap_for_export` and
  `unwrap_from_import` to encapsulate error-related data within an exported
  function.

  Attributes:
    error_code (jax.Array): A JAX array representing the final error state of
      the function to be exported. This value is local to the wrapper function.
    error_list (list[tuple[str, str]]): A list of `(error_message, traceback)`
      pairs containing error messages and corresponding stack traces. This error
      list is local to the wrapper function, and does not contain pairs of error
      information from other functions.
  """

  error_code: jax.Array
  error_list: list[tuple[str, str]]


jax.tree_util.register_dataclass(
    _ErrorClass, data_fields=("error_code",), meta_fields=("error_list",)
)
jax.export.register_pytree_node_serialization(
    _ErrorClass,
    serialized_name=f"{_ErrorClass.__module__}.{_ErrorClass.__name__}",
    serialize_auxdata=lambda x: json.dumps(x, ensure_ascii=False).encode(
        "utf-8"
    ),
    deserialize_auxdata=lambda x: json.loads(x.decode("utf-8")),
)


def _traceback_to_str(traceback: TracebackType) -> str:
  """Convert a traceback to a string for export."""
  return "".join(tb_lib.format_list(tb_lib.extract_tb(traceback))).rstrip("\n")


def wrap_for_export(f):
  """Wrap a function with error checking to make it compatible with AOT mode.

  Error checking relies on global state, which cannot be serialized across
  processes. This wrapper ensures that the error state remains within the
  function scope, making it possible to export the function and later import in
  other processes.

  This function should only be applied once to a function; wrapping the same
  function multiple times is unnecessary.
  """

  def inner(*args, **kwargs):
    global _error_list

    # 1. Save the old state and initialize a new state.
    with core.eval_context():
      old_ref = _error_storage.ref
    _initialize_error_code_ref()
    with _error_list_lock:
      old_error_list, _error_list = _error_list, []

      # 2. Trace the function.
      out = f(*args, **kwargs)
      error_code = _error_storage.ref[...].min()

      # 3. Restore the old state.
      _error_list, new_error_list = old_error_list, _error_list
    with core.eval_context():
      _error_storage.ref = old_ref

    new_error_list = [
        (msg, _traceback_to_str(traceback)) for msg, traceback in new_error_list
    ]
    return out, _ErrorClass(error_code, new_error_list)

  return inner


def unwrap_from_import(f):
  """Unwrap a function after AOT import to restore error checking.

  When an AOT-exported function is imported in a new process, its error state is
  separate from the global error state of the current process. This wrapper
  ensures that errors detected during execution are correctly integrated into
  the global error checking mechanism of the current process.
  """
  if _error_storage.ref is None:
    with core.eval_context():
      _initialize_error_code_ref()
    assert _error_storage.ref is not None

  def inner(*args, **kwargs):
    out, error_class = f(*args, **kwargs)
    new_error_code, error_list = error_class.error_code, error_class.error_list

    # Update the global error list.
    with _error_list_lock:
      offset = len(_error_list)
      _error_list.extend(error_list)

    # Update the global error code array.
    error_code = _error_storage.ref[...]
    should_update = jnp.logical_and(
        error_code == jnp.uint32(_NO_ERROR),
        new_error_code != jnp.uint32(_NO_ERROR),
    )
    error_code = jnp.where(should_update, new_error_code + offset, error_code)
    # TODO(ayx): support vmap and shard_map.
    _error_storage.ref[...] = error_code

    return out

  return inner


Behavior = Literal["ignore", "raise"]


class error_checking_behavior:
  """A context manager to set the error checking behavior.

  If both `all` and a category are provided, the category will override the
  `all` setting.

  When the error checking behavior is set to "ignore", all errors will be
  ignored. When set to "raise", errors will be detected and recorded, but an
  exception will not be raised immediately. Users must call
  :func:`raise_if_error` to at the end of the computation to raise the
  exception.
  """

  def __init__(
      self,
      *,
      all: Behavior | None = None,
      nan: Behavior | None = None,
      divide: Behavior | None = None,
      oob: Behavior | None = None,
  ) -> None:
    new_settings = {}
    if all is not None:
      new_settings["nan"] = new_settings["divide"] = new_settings["oob"] = all
    if nan is not None:
      new_settings["nan"] = nan
    if divide is not None:
      new_settings["divide"] = divide
    if oob is not None:
      new_settings["oob"] = oob
    self.new_settings = new_settings
    self.stack = contextlib.ExitStack()

  def __enter__(self):
    config_flags = {
        "nan": config.error_checking_behavior_nan,
        "divide": config.error_checking_behavior_divide,
        "oob": config.error_checking_behavior_oob,
    }
    for key, value in self.new_settings.items():
        self.stack.enter_context(config_flags[key](value))
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.stack.close()
