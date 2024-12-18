# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#

# # Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for representing pytreedefs in a serializable format.
"""

import os
import json
import threading
import base64
import itertools
import logging
import io
import time
import zipfile
import warnings
from types import ModuleType
from concurrent.futures import Future
from typing import Any, TypeVar, Callable

import jax
from jax.tree_util import (PyTreeDef, default_registry as _default_registry,
                           treedef_is_leaf)
from jax._src.export._export import (serialization_registry,
                                     deserialization_registry)
from jax.export import register_pytree_node_serialization

T = TypeVar("T")
PickleModule = ModuleType
logger = logging.getLogger(__name__)

_READABLE_PYTREE_SERIALIZATION = True
_TREE_REPR_KEY = "__jax_pytreedef_repr"
_LEAF_IDS_KEY = "__jax_leaf_ids"

_NOT_REGISTERED_MESSAGE = (
  "  * If you want to register a custom leaf, register it via"
  " `register_pytree_leaf_serialization` first.\n"
  "  * If you want to register a custom node, register is via"
  " `register_pytree_node_serialization`")

class AwaitableFuture(Future[Any]):
  """A wrapper around a Future that makes it look like an async function."""
  def __init__(self, future: Future[Any]):
    self._future, self.pytree = future, None

  def done(self):
    return self._future.done()

  def result(self, *args, **kw):
    return self._future.result(*args, **kw)

  def __await__(self):
    while not self.done():
      time.sleep(1e-3)
      yield
    return self.result()

class _MISSING_TYPE:
  pass
MISSING = _MISSING_TYPE()

def _cls2typerepr(cls):
  return f"{cls.__module__}.{cls.__name__}"

_SerializeLeaf = Callable[[Any], bytes | str]
_DeserializeLeaf = Callable[[bytes | str], Any]

leaf_serialization_registry: dict[type[Any], tuple[str, _SerializeLeaf]] = {}
leaf_deserialization_registry: dict[str,
                                    tuple[type[Any], _DeserializeLeaf]] = {}

def register_pytree_leaf_serialization(
    leaftype: type[T],
    *,
    serialize_leaf: _SerializeLeaf,
    deserialize_leaf: _DeserializeLeaf,
    serialized_name: str | None = None,
) -> type[T]:
  """Registers a custom PyTree leaf type for serialization and deserialization.

  This function allows you to define how custom PyTree leaf types are serialized
  and deserialized. This is necessary for leaf types that are not natively
  supported by the serialization mechanism.  Unlike
  `register_pytree_node_serialization`, this function is specifically for leaf
  types (especially potentially large leaves) and does not handle children
  (leaves don't have children by definition).

  Args:
    leaftype: The type of the PyTree leaf to register.  It is an error to
      attempt to register multiple serializations for the same `leaftype`.
    serialize_leaf: A function that takes a leaf of type `leaftype` and returns
      its serialized representation as a string or bytes.
    deserialize_leaf: A function that takes the serialized representation (string
      or bytes) of a leaf and reconstructs an instance of `leaftype`.
    serialized_name: An optional string used as the identifier during
      serialization and deserialization. If not provided, the type representation
      of `leaftype` will be used. It is an error to attempt to register
      multiple serializations with the same `serialized_name`.

  Returns:
    The same type passed as `leaftype`, so that this function can be used as a
    class decorator.

  Raises:
    ValueError: If a duplicate registration is attempted for the `leaftype` or
      `serialized_name`.

  Example:
    ```python
    @register_pytree_leaf_serialization(
        serialize_leaf=lambda leaf: leaf.to_string(),
        deserialize_leaf=lambda s: MyCustomLeafType.from_string(s))
    class MyCustomLeafType:
      ...
    ```
  """
  if serialized_name is None:
    serialized_name = _cls2typerepr(leaftype)
  if leaftype in leaf_serialization_registry:
    logger.warning("Duplicate serialization registration for type `%s`."
                   " Previous registration was with serialized_name `%s`.",
                   str(leaftype), str(leaf_serialization_registry[leaftype][0]))
  if serialized_name in leaf_deserialization_registry:
    logger.warning("Duplicate serialization registration for serialized_name"
                   " `%s`. Previous registration was for type `%s`.",
                   serialized_name,
                   leaf_deserialization_registry[serialized_name][0])
  leaf_serialization_registry[leaftype] = (serialized_name, serialize_leaf)
  leaf_deserialization_registry[serialized_name] = (leaftype, deserialize_leaf)
  return leaftype

_identity = lambda x: x

# register in-built leaf types
for t in [int, float, str, complex, bool]:
  register_pytree_leaf_serialization(t, serialize_leaf=json.dumps,
                                     deserialize_leaf=json.loads)
register_pytree_leaf_serialization(bytes, serialize_leaf=_identity,
                                   deserialize_leaf=_identity)
register_pytree_leaf_serialization(bytearray, serialize_leaf=_identity,
                                   deserialize_leaf=_identity)
# register in-built containers_identity
for t in [list, tuple, dict]:
  register_pytree_node_serialization(
      t, serialized_name=f"{t.__module__}.{t.__name__}",
      serialize_auxdata=_identity, deserialize_auxdata=_identity)

class MemKVStore:
  def __init__(self, data: bytes | None = None):
    self.buffer = io.BytesIO(data) if data is not None else io.BytesIO()
    self.buffer.seek(0)
    self.zipfile = zipfile.ZipFile(
      self.buffer, mode="a", compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    self._lock, self._closed = threading.Lock(), False

  def keys(self) -> list[str]:
    assert not self._closed
    return self.zipfile.namelist()

  def tobytes(self) -> bytes:
    assert not self._closed
    with self._lock:
      self.zipfile.close()
      self._closed = True
      return self.buffer.getvalue()

  def write(self, filename: str | os.PathLike[str], data: bytes | str) -> None:
    assert not self._closed
    # ignore warnings about writing the same object multiple times
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      with self._lock:
        self.zipfile.writestr(str(filename), data)

  def read(self, filename: str | os.PathLike[str]) -> bytes:
    assert not self._closed
    return self.zipfile.read(str(filename))

  def __del__(self):
    if not self._closed:
      self.zipfile.close()

def _node_serialize(node_type_data: tuple[type[Any], Any]):
  """Serializes the node data and returns the type repr and the name indicating
  a reference to the aux_data archive.
  """
  node_type, node_data = node_type_data
  if node_type not in serialization_registry:
    raise ValueError(f"Node of type `{node_type}` is not registered for"
                     f" serialization.\n{_NOT_REGISTERED_MESSAGE}")
  serialized_name, serialize_fn = serialization_registry[node_type]
  node_data_serialized = serialize_fn(node_data)
  return (serialized_name, node_data_serialized)

def _node_deserialize(node_type_data: tuple[str, Any], best_effort: bool = False
                      ) -> tuple[type[Any], Any]:
  """Deserializes the node data and returns the type and the data.

  If the node class is not defined in the current context
  AND best_effort == True, we return a "generic" node - a list of children. This
  can be useful for deserializing old data where the class no longer exists.
  """
  serialized_name, node_data = node_type_data
  if serialized_name not in deserialization_registry:
    if best_effort:
      logger.warning("We couldn't read the node %s, returning list of children",
                     serialized_name)
      return (list, None)
    raise ValueError(f"Node type \"{serialized_name}\" is not registered."
                     f"\n{_NOT_REGISTERED_MESSAGE}")
  node_type, deserialize_fn, _ = deserialization_registry[serialized_name]
  node_aux_data = deserialize_fn(node_data)
  return (node_type, node_aux_data)

def _serialize_pytreedef_helper(node, leaf_counter: itertools.count):
  if treedef_is_leaf(node) and node.num_leaves == 1:
    return dict(node_type="leaf", leaf_id=next(leaf_counter))
  node_repr = dict()
  type_repr, node_data = _node_serialize(node.node_data())
  if isinstance(node_data, bytes):
    node_data = ("base64", base64.b64encode(node_data).decode("utf-8"))
  else:
    node_data = (None, node_data)
  node_repr["name"], node_repr["node_aux_data"] = type_repr, node_data
  node_repr["node_type"] = "static_node" if node.num_nodes == 1 else "node"
  node_repr["children"] = [_serialize_pytreedef_helper(child, leaf_counter)
                           for child in node.children()]
  return node_repr

@jax.tree_util.register_static
class _EmptyStaticNode:
  """Static node replacement if the original can't be referenced."""

def _deserialize_pytreedef_helper(node, best_effort: bool = False):
  assert "node_type" in node
  if node["node_type"] == "leaf":
    # case 1: normal leaf node -------------------------------------------------
    node_data, pytree_children = None, ()
  else:
    node_aux_data_encoding, node_aux_data = node["node_aux_data"]
    if node_aux_data_encoding == "base64":
      node_aux_data = base64.b64decode(node_aux_data)
    node_data = _node_deserialize((node["name"], node_aux_data),
                                  best_effort=best_effort)
    pytree_children = [
      _deserialize_pytreedef_helper(child, best_effort=best_effort)
      for child in node["children"]] # type: ignore
    if (node["node_type"] == "static_node" and best_effort
        and node_data[0] is list):
      # the node failed to deserialize and was replaced at best effort with list
      return jax.tree.structure(_EmptyStaticNode())
  pt = PyTreeDef.make_from_node_data_and_children(_default_registry, node_data,
                                                  pytree_children)
  return pt

# serialize and deserialize pytree methods namespaces: permissive and strict
def _serialize_pytreedef_readable(node) -> dict[str, Any]:
  leaf_counter = itertools.count()
  root_repr = _serialize_pytreedef_helper(node, leaf_counter=leaf_counter)
  leaf_count = node.num_leaves
  assert leaf_count == next(leaf_counter)
  pytree_repr = {_TREE_REPR_KEY: root_repr,
                 _LEAF_IDS_KEY: list(range(leaf_count))}
  # gather data from the thread pool executor
  return pytree_repr

def _deserialize_pytreedef_readable(pytreedef_repr: dict[str, Any],
                                    best_effort: bool = False) -> Any:
  pt = _deserialize_pytreedef_helper(pytreedef_repr[_TREE_REPR_KEY],
                                     best_effort=best_effort)
  return pt


if _READABLE_PYTREE_SERIALIZATION:
  serialize_pytreedef = _serialize_pytreedef_readable
  deserialize_pytreedef = _deserialize_pytreedef_readable
