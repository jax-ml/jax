# Copyright 2018 The JAX Authors.
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

# Lowering of jaxprs into XLA (HLO) computations.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Union

import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src.core import ShapedArray
from jax._src.util import safe_zip, safe_map

from jax._src.typing import Shape

from jax._src.lib import xla_client as xc

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# TODO(dfm): Remove these forwards
canonicalize_dtype = core.canonicalize_dtype
canonicalize_dtype_handlers = core.canonicalize_dtype_handlers

# Types

def _make_array_shape(aval: ShapedArray) -> Sequence[xc.Shape]:
  aval = core.physical_aval(aval)
  dtype = np.dtype('bool') if aval.dtype == dtypes.float0 else aval.dtype
  return (xc.Shape.array_shape(dtype, aval.shape),)

# Utilities

# HLO instructions optionally can be annotated to say how the output should be
# spatially partitioned (represented in XLA as OpSharding protos, see
# sharding_to_proto). For array outputs, the annotation is either an int per
# dimension specifying the number of ways that dimension divided (i.e. the total
# number of shards is the product), or None to indicate the array should be
# replicated. Tuple outputs are represented as tuples thereof. XLA supports
# arbitrary tuple nesting, but JAX only uses one level of tupling (and our type
# checkers don't support recursive types), so we only represent one level of
# nesting in this type definition.
SpatialSharding = Union[Shape, None, tuple[Union[Shape, None], ...]]


def sharding_to_proto(sharding: SpatialSharding):
  """Converts a SpatialSharding to an OpSharding.

  See
  https://github.com/tensorflow/tensorflow/blob/main/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  """
  proto = xc.OpSharding()
  if isinstance(sharding, tuple) and not isinstance(sharding[0], int):
    assert all(s is None or isinstance(s, tuple) for s in sharding)
    return tuple_sharding_proto(list(map(sharding_to_proto, sharding)))

  if sharding is None:
    proto.type = xc.OpSharding.Type.REPLICATED
  else:
    proto.type = xc.OpSharding.Type.OTHER
    proto.tile_assignment_dimensions = list(sharding)  # type: ignore
    proto.tile_assignment_devices = list(range(np.prod(sharding)))  # type: ignore
  return proto

def tuple_sharding_proto(elems):
  proto = xc.OpSharding()
  assert all(isinstance(e, type(proto)) for e in elems)
  proto.type = xc.OpSharding.Type.TUPLE
  proto.tuple_shardings = elems
  return proto


### handlers

# JAX abstract values -> XLA shapes

def aval_to_xla_shapes(aval: core.AbstractValue) -> Sequence[xc.Shape]:
  try:
    return _xla_shape_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No xla_shape_handler for type: {type(aval)}") from err

_xla_shape_handlers: dict[type[core.AbstractValue],
                         Callable[[Any], Sequence[xc.Shape]]] = {
    ShapedArray: _make_array_shape,
}
_xla_shape_handlers[core.AbstractToken] = lambda _: (xc.Shape.token_shape(),)

initial_style_primitives: set[core.Primitive] = set()

def register_initial_style_primitive(prim: core.Primitive):
  initial_style_primitives.add(prim)
