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

import base64
import logging
from types import ModuleType
from concurrent.futures import Future
from typing import Any, TypeVar

import jax
from jax._src.export.serialization import (flatbuffers, _serialize_pytreedef,
                                           _deserialize_pytreedef_to_pytree,
                                           ser_flatbuf)
from jax.export import register_pytree_node_serialization  # pylint: disable=unused-import

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

__all__ = ["serialize_pytreedef", "deserialize_pytreedef",
           "register_pytree_node_serialization"]

class PyTreeFuture(Future[Any]):
  """A wrapper around a Future that makes it look like an async function."""
  def __init__(self, future: Future[Any]):
    self._future, self.pytree = future, None

  def done(self):
    return self._future.done()

  def result(self, *args, **kw):
    return self._future.result(*args, **kw)

  def __await__(self):
    while not self.done():
      yield
    return self.result()


def _cls2typerepr(cls):
  return f"{cls.__module__}.{cls.__name__}"


def serialize_pytreedef(node) -> dict[str, Any]:
  builder = flatbuffers.Builder(65536)
  exported = _serialize_pytreedef(builder, node)
  builder.Finish(exported)
  root_repr = base64.b64encode(builder.Output()).decode("utf-8")
  leaf_count = node.num_leaves
  pytree_repr = {_TREE_REPR_KEY: root_repr,
                 _LEAF_IDS_KEY: list(range(leaf_count))}
  return pytree_repr


def deserialize_pytreedef(pytreedef_repr: dict[str, Any]):
  buf = base64.b64decode(pytreedef_repr[_TREE_REPR_KEY])
  exp = ser_flatbuf.PyTreeDef.GetRootAs(buf)
  treestruct = jax.tree.structure(_deserialize_pytreedef_to_pytree(exp))
  return treestruct
