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
"""Shape polymorphism support for jax2tf.

For usage instructions, read the jax2tf.convert docstring, and the
[README](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md).

"""
import collections
import string
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from jax import core

DimSize = core.DimSize
Shape = core.Shape

class DimVar:
  """A shape dimension variable.
  We assume that these range over integer values >= 1.

  Implements only a minimal set of operations to allow us to construct sets and
  dictionaries.
  """
  def __init__(self, varname: str):
    self._varname = varname

  def __str__(self):
    return self._varname

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash(self._varname)

  def __eq__(self, other):
    if isinstance(other, DimVar) and self._varname == other._varname:
      return True
    else:
      raise core.InconclusiveDimensionOperation(f"Shape variable comparison {self} == {other} is inconclusive")

  def __ne__(self, other):
    return not self == other


class DimensionHandlerVar(core.DimensionHandler):
  """See core.DimensionHandler."""
  def symbolic_equal(self, d1: DimSize, d2: DimSize) -> bool:
    # We compare hashes first, to avoid InconclusiveDimensionOperation.
    return hash(d1) == hash(d2) and d1 == d2

  def greater_equal(self, d1: DimSize, d2: DimSize):
    if self.symbolic_equal(d1, d2) or (type(d2) is not DimVar and 1 >= d2):
      return True
    else:
      raise core.InconclusiveDimensionOperation(f"Shape variable comparison {d1} >= {d2} is inconclusive")

  def sum(self, *ds: DimSize):
    d_ints, d_vars = _split_shape_ints(ds)
    if len(d_vars) != 1:
      raise TypeError(f"Adding shape variables is not supported ({' + '.join(map(str, ds))})")
    if sum(d_ints) != 0:
      raise TypeError(f"Adding non-zero to shape variables is not supported {' + '.join(map(str, ds))}")
    return d_vars[0]

  def diff(self, d1: DimSize, d2: DimSize) -> DimSize:
    if self.symbolic_equal(d1, d2):
      return 0
    if d2 in {0}:
      return d1
    raise core.InconclusiveDimensionOperation(f"Subtracting shape variables is not supported ({d1} - {d2})")

  def divide_shape_sizes(self, s1: Shape, s2: Shape) -> int:
    s1_ints, s1_vars = _split_shape_ints(s1)
    s2_ints, s2_vars = _split_shape_ints(s2)
    if collections.Counter(s1_vars) != collections.Counter(s2_vars):
      msg = (f"Shapes {s1} and {s2} must have the same set of shape variables.")
      raise core.InconclusiveDimensionOperation(msg)
    return super(DimensionHandlerVar, self).divide_shape_sizes(s1_ints, s2_ints)

  def dilate(self, d: DimSize, dilation: DimSize) -> DimSize:
    """Implements `0 if d == 0 else 1 + dilation * (d - 1))`"""
    if dilation not in {1}:
      raise core.InconclusiveDimensionOperation(f"Dilation is not supported for shape variables (d = {dilation})")
    return d

  def stride(self, d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
    """Implements `(d - window_size) // window_stride + 1`"""
    if {window_size, window_stride} != {1}:
      raise core.InconclusiveDimensionOperation(f"Striding is not supported for shape variables (window_size = {window_size}, stride = {window_stride}")
    return d


core._SPECIAL_DIMENSION_HANDLERS[DimVar] = DimensionHandlerVar()

def _split_shape_ints(shape: Shape) -> Tuple[Sequence[int], Sequence[DimVar]]:
  """Splits the shape into an integer sequence and a sequence of vars."""
  shape_ints: List[int] = []
  shape_vars: List[DimVar] = []
  for d in shape:
    (shape_ints if isinstance(d, int) else shape_vars).append(d)
  return shape_ints, shape_vars


class ShapeSyntaxError(Exception): pass

_identifiers = frozenset(string.ascii_lowercase)
def parse_spec(spec: Optional[str],
               arg_shape: Sequence[Optional[int]]) -> Tuple[DimSize, ...]:
  """Parse the shape polymorphic specification for one array argument.
  Args:
    spec: a shape polymorphic specification.
    arg_shape: an actual shape, possibly containing unknown dimensions (None).

  The placeholders `_` in the specification are replaced with the values from
  the actual shape, which must be known.

  See the README.md for usage.
  """
  shape_var_map: Dict[str, Set[int]] = collections.defaultdict(set)
  def _parse_dim(dim_spec: str, dim_size: Optional[int]) -> Union[int, DimSize]:
    if dim_spec == '_':
      if dim_size is None:
        msg = (f"polymorphic_shape '{spec}' has `_` placeholders for argument shape "
               f"dimensions that are unknown: {arg_shape}")
        raise ValueError(msg)
      return dim_size
    elif dim_spec.isdigit():
      spec_size = int(dim_spec)
      if dim_size != spec_size:
        if dim_size is None:
          msg = (f"polymorphic_shape '{spec}' must contain shape variables for argument shape "
                 f"dimensions that are unknown: {arg_shape}")
        else:
          msg = (f"polymorphic_shape '{spec}' does not match argument shape {arg_shape}")
        raise ValueError(msg)
      return spec_size
    elif dim_spec[0] in _identifiers:
      if dim_size is not None:
        shape_var_map[dim_spec].add(dim_size)
      return DimVar(dim_spec)
    else:
      raise ShapeSyntaxError(dim_spec)

  if not spec:
    if any(d is None for d in arg_shape):
      msg = ("polymorphic_shape must be specified when the argument "
             f"shape {arg_shape} is partially known.")
      raise ValueError(msg)
    return tuple(arg_shape)

  if spec[0] == '(':
    if spec[-1] != ')':
      raise ShapeSyntaxError(spec)
    spec_ = spec[1:-1]
  else:
    spec_ = spec
  specs = spec_.replace(' ', '').strip(',').split(',')
  if len(specs) != len(arg_shape):
    msg = (f"polymorphic_shape '{spec}' has different rank than argument "
           f"shape {arg_shape}")
    raise ValueError(msg)
  dims = tuple(map(_parse_dim, specs, arg_shape))

  for dim_var, dim_var_values in shape_var_map.items():
    if len(dim_var_values) != 1:
      msg = (f"polymorphic shape variable '{dim_var}' corresponds to multiple "
             f"values ({sorted(dim_var_values)}), in polymorphic_shape '{spec}' and "
             f"argument shape {arg_shape}")
      raise ValueError(msg)

  return dims
