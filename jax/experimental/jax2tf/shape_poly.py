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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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
  def is_constant(self, d: DimSize) -> bool:
    assert isinstance(d, DimVar)
    return False

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
    raise core.InconclusiveDimensionOperation(
        f"Subtracting shape variables is not supported ({d1} - {d2})")

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
      raise core.InconclusiveDimensionOperation(
          f"Only dilation == 1 is supported for shape variables (var = {d}, "
          f"dilation = {dilation})")
    return d

  def stride(self, d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
    """Implements `(d - window_size) // window_stride + 1`"""
    if {window_size, window_stride} != {1}:
      raise core.InconclusiveDimensionOperation(
          "Only striding with window_size == window_stride == 1 is supported "
          f"for shape variables (var = {d}, window_size = {window_size}, "
          f"stride = {window_stride}")
    return d


core._SPECIAL_DIMENSION_HANDLERS[DimVar] = DimensionHandlerVar()

def _split_shape_ints(shape: Shape) -> Tuple[Sequence[int], Sequence[DimVar]]:
  """Splits the shape into an integer sequence and a sequence of vars."""
  shape_ints: List[int] = []
  shape_vars: List[DimVar] = []
  for d in shape:
    (shape_ints if isinstance(d, int) else shape_vars).append(d)
  return shape_ints, shape_vars


class PolyShape(tuple):
  """Tuple of polymorphic dimension specifications.

  See docstring of :func:`jax2tf.convert`.
  """
  def __new__(cls, *dim_specs):
    return tuple.__new__(PolyShape, dim_specs)


def parse_spec(spec: Optional[Union[str, PolyShape]],
               arg_shape: Sequence[Optional[int]]) -> Tuple[DimSize, ...]:
  """Parse the shape polymorphic specification for one array argument.
  Args:
    spec: a shape polymorphic specification, either a string, or a PolyShape.
    arg_shape: an actual shape, possibly containing unknown dimensions (None).

  The placeholders `_` in the specification are replaced with the values from
  the actual shape, which must be known.

  See the README.md for usage.
  """
  if spec is None:
    spec_tuple = (...,)  # type: Tuple[Any,...]
  elif isinstance(spec, PolyShape):
    spec_tuple = tuple(spec)
  elif isinstance(spec, str):
    spec_ = spec.replace(" ", "")
    if spec_[0] == "(":
      if spec_[-1] != ")":
        raise ValueError(spec)
      spec_ = spec_[1:-1]
    spec_ = spec_.rstrip(",")
    if not spec_:
      spec_tuple = ()
    else:
      specs = spec_.split(',')
      def parse_dim(ds: str):
        if ds == "...":
          return ...
        elif ds.isdigit():
          return int(ds)
        elif ds == "_" or ds.isalnum():
          return ds
        else:
          raise ValueError(f"PolyShape '{spec}' has invalid syntax")

      spec_tuple = tuple(map(parse_dim, specs))
  else:
    raise ValueError(f"PolyShape '{spec}' must be either None, a string, or PolyShape.")

  ds_ellipses = tuple(ds for ds in spec_tuple if ds == ...)
  if ds_ellipses:
    if len(ds_ellipses) > 1 or spec_tuple[-1] != ...:
      raise ValueError(f"PolyShape '{spec}' can contain Ellipsis only at the end.")
    spec_tuple = spec_tuple[0:-1]
    if len(arg_shape) >= len(spec_tuple):
      spec_tuple = spec_tuple + ("_",) * (len(arg_shape) - len(spec_tuple))

  if len(arg_shape) != len(spec_tuple):
    raise ValueError(f"PolyShape '{spec}' must match the rank of arguments {arg_shape}.")

  shape_var_map: Dict[str, Set[int]] = collections.defaultdict(set)
  def _process_dim(i: int, dim_spec):
    if not isinstance(dim_spec, (str, int)):
      raise ValueError(f"PolyShape '{spec}' in axis {i} must contain only integers, strings, or Ellipsis.")
    dim_size = arg_shape[i]
    if dim_size is None:
      if dim_spec == "_" or not isinstance(dim_spec, str):
        msg = (f"PolyShape '{spec}' in axis {i} must contain a shape variable "
               f"for unknown dimension in argument shape {arg_shape}")
        raise ValueError(msg)
      return DimVar(dim_spec)
    else:  # dim_size is known
      if dim_spec == "_":
        return dim_size
      if isinstance(dim_spec, int):
        if dim_spec != dim_size:
          msg = (f"PolyShape '{spec}' in axis {i} must contain a constant or '_' "
                 f"for known dimension in argument shape {arg_shape}")
          raise ValueError(msg)
        return dim_size
      # We have a dimension variable for a known dimension.
      shape_var_map[dim_spec].add(dim_size)
      return DimVar(dim_spec)

  dims = tuple([_process_dim(i, ds) for i, ds in enumerate(spec_tuple)])
  for dim_var, dim_var_values in shape_var_map.items():
    if len(dim_var_values) != 1:
      msg = (f"PolyShape '{spec}' has dimension variable '{dim_var}' "
             f"corresponding to multiple values ({sorted(dim_var_values)}), for "
             f"argument shape {arg_shape}")
      raise ValueError(msg)

  return dims
