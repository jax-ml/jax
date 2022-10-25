# Copyright 2019 The JAX Authors.
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
from functools import partial
import itertools
import operator

import jaxlib.mlir.ir as ir

from jaxlib import xla_client

from .mhlo_helpers import custom_call

try:
  from .cuda import _prng as _cuda_prng
  for _name, _value in _cuda_prng.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")
except ImportError:
  _cuda_prng = None

try:
  from .rocm import _prng as _hip_prng
  for _name, _value in _hip_prng.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  _hip_prng = None

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def _threefry2x32_lowering(prng, platform, keys, data):
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

  opaque = prng.threefry2x32_descriptor(_prod(dims))
  layout = tuple(range(ndims - 1, -1, -1))
  return custom_call(
      f"{platform}_threefry2x32",
      [typ, typ],
      [keys[0], keys[1], data[0], data[1]],
      backend_config=opaque,
      operand_layouts=[layout] * 4,
      result_layouts=[layout] * 2)


cuda_threefry2x32 = partial(_threefry2x32_lowering, _cuda_prng, "cu")
rocm_threefry2x32 = partial(_threefry2x32_lowering, _hip_prng, "hip")
