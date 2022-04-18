# Copyright 2021 Google LLC
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

import functools
import itertools
import operator

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo

import numpy as np

from jaxlib import xla_client

try:
  from . import _hip_prng
  for _name, _value in _hip_prng.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  pass

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def threefry2x32_lowering(keys, data):
  """ThreeFry2x32 kernel for GPU."""
  assert len(keys) == 2, keys
  assert len(data) == 2, data
  assert (ir.RankedTensorType(keys[0].type).element_type ==
          ir.IntegerType.get_unsigned(32)), keys[0].type
  typ = keys[0].type
  dims = ir.RankedTensorType(typ).shape

  for x in itertools.chain(keys, data):
    assert x.type == typ, (x.type, typ)
  ndims = len(dims)

  opaque = _hip_prng.hip_threefry2x32_descriptor(_prod(dims))
  layout = ir.DenseIntElementsAttr.get(np.arange(ndims - 1, -1, -1),
                                       type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  tup = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([typ, typ])],
      [keys[0], keys[1], data[0], data[1]],
      call_target_name = ir.StringAttr.get("hip_threefry2x32"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([layout] * 4),
      result_layouts=ir.ArrayAttr.get([layout] * 2)).result
  return [
    mhlo.GetTupleElementOp(tup, ir.IntegerAttr.get(i32_type, i)).result
    for i in range(2)
  ]
