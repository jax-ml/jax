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

import dataclasses
from functools import partial
import json
import threading
import traceback as tb_lib
from types import TracebackType

import jax
from jax._src import core
from jax._src import source_info_util
from jax._src import traceback_util
import jax._src.mesh as mesh_lib
from jax._src.sharding_impls import NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import jax.export
import jax.numpy as jnp


traceback_util.register_exclusion(__file__)


class JaxValueError(ValueError):
  """Exception raised for failed runtime error checks in JAX."""


#: The default error code for no error.
#:
#: This value is chosen because we can use `jnp.min()` to obtain the
#: first error when performing reductions.
_NO_ERROR = jnp.iinfo(jnp.uint32).max


_error_list_lock = threading.Lock()
# (error_message, traceback) pairs. Traceback is `str` when imported from AOT.
_error_list: list[tuple[str, TracebackType | str]] = []


class _ErrorStorage(threading.local):

  def __init__(self):
    self.ref: core.MutableArray | None = None


_error_storage = _ErrorStorage()


def _initialize_error_code_ref() -> None:
  """Initialize error_code_ref in the current thread.

  The size of the error code array is determined by the mesh in the context. In
  single-device environment, the array is a scalar. In multi-device
  environment, the array has the same shape as the mesh.
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
  """Redefine the error checking state based on the mesh in the context.

  This context manager should be used when starting a multi-device
  computation, and whenever the mesh is changed.

  When exiting the context, the error checking state will be reset to the
  original state.
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


def set_error_if(pred: jax.Array, /, msg: str) -> None:
  """Set error if any element of pred is true.

  If the error is already set, the new error will be ignored. It will not
  override the existing error.

  In auto mode, this function does not work under jit.
  """
  if _error_storage.ref is None:
    with core.eval_context():
      _initialize_error_code_ref()
    assert _error_storage.ref is not None

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

  if out_sharding.mesh.shape_tuple == ():  # single-device case.
    pred = pred.any()
  else:  # multi-device case.
    has_auto_axes = mesh_lib.AxisTypes.Auto in in_sharding.mesh.axis_types
    if has_auto_axes:
      raise NotImplementedError(
          "Error checking in auto mode is not supported yet. Please use"
          " explicit mode."
      )
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
  """Raise error if an error is set.

  This function should be called after the computation is finished. It should
  not be called within a traced context, such as within a jitted function.

  Raises:
    JaxValueError: if an error is set.
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
  if isinstance(traceback, str):  # from export
    exc.add_note(f"The original traceback is shown below:\n{traceback}")
    raise exc
  else:
    raise exc.with_traceback(traceback)


@dataclasses.dataclass(frozen=True)
class _ErrorClass:
  """A class to store error information for AOT compilation."""

  error_code: jax.Array
  error_list: list[tuple[str, str]]


jax.tree_util.register_dataclass(
    _ErrorClass, data_fields=("error_code",), meta_fields=("error_list",)
)
_serialize_auxdata = lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8")
_deserialize_auxdata = lambda x: json.loads(x.decode("utf-8"))
jax.export.register_pytree_node_serialization(
    _ErrorClass,
    serialized_name=f"{_ErrorClass.__module__}.{_ErrorClass.__name__}",
    serialize_auxdata=_serialize_auxdata,
    deserialize_auxdata=_deserialize_auxdata,
)


def _traceback_to_str(traceback: TracebackType) -> str:
  """Convert a traceback to a string."""
  return "".join(tb_lib.format_list(tb_lib.extract_tb(traceback))).rstrip()


def wrap_for_export(f):
  """Wrap a function for AOT compilation."""

  def inner(*args, **kwargs):
    global _error_list

    # Save the old state and initialize a new state.
    with _error_list_lock:
      old_error_list, _error_list = _error_list, []
    old_ref = _error_storage.ref
    _initialize_error_code_ref()

    out = f(*args, **kwargs)
    error_code = _error_storage.ref[...].min()

    # Restore the old state.
    with _error_list_lock:
      _error_list, new_error_list = old_error_list, _error_list
    _error_storage.ref = old_ref

    new_error_list = [
        (msg, _traceback_to_str(traceback)) for msg, traceback in new_error_list
    ]
    return out, _ErrorClass(error_code, new_error_list)

  return inner


def unwrap_from_import(f):
  """Unwrap a function for AOT compilation."""
  if _error_storage.ref is None:
    with core.eval_context():
      _initialize_error_code_ref()
    assert _error_storage.ref is not None

  def inner(*args, **kwargs):
    out, error_class = f(*args, **kwargs)
    error_code, error_list = error_class.error_code, error_class.error_list
    with _error_list_lock:
      offset = len(_error_list)
      _error_list.extend(error_list)
    _error_storage.ref[...] = jnp.where(
        error_code == jnp.uint32(_NO_ERROR),
        jnp.uint32(_NO_ERROR),
        error_code + offset,
    )
    return out

  return inner
