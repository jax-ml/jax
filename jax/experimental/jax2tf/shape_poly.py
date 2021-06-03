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
import itertools
import functools
import operator as op
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union


import jax
from jax._src.numpy import lax_numpy
import opt_einsum
from jax import config
from jax import core
import numpy as np

DimSize = core.DimSize
Shape = core.Shape

class _DimMon(dict):
  """Represents a multivariate monomial, such as n^3 * m.

  The representation is a dictionary mapping var:exponent.
  The `var` are strings and the exponents are >= 1.
  The shape variables are assumed to range over integers >= 1.
  """
  def __hash__(self):
    return hash(frozenset(self.items()))

  def __str__(self):
    return ' '.join(f'{key}^{exponent}' if exponent != 1 else str(key)
                    for key, exponent in sorted(self.items()))

  @classmethod
  def from_var(cls, v: str) -> '_DimMon':
    return _DimMon({v: 1})

  def to_var(self) -> Optional[str]:
    """Extract the variable name "x", from a monomial "x".
     Return None, if the monomial is not a single variable."""
    items = self.items()
    if len(items) != 1:
      return None
    (v, vexp), = items
    if vexp != 1:
      return None
    return v

  def get_vars(self) -> Set[str]:
    return set(self.keys())

  @property
  def degree(self):
    return sum(self.values())

  def __lt__(self, other: '_DimMon'):
    """
    Comparison to another monomial in graded reverse lexicographic order.
    Used for sorting.
    """
    self_key = -self.degree, tuple(sorted(self))
    other_key = -other.degree, tuple(sorted(other))
    return self_key > other_key

  def mul(self, other: '_DimMon') -> '_DimMon':
    """
    Returns the product with another monomial. Example: (n^2*m) * n == n^3 * m.
    """
    return _DimMon(collections.Counter(self) + collections.Counter(other))

  def divide(self, divisor: '_DimMon') -> '_DimMon':
    """
    Divides by another monomial. Raises a core.InconclusiveDimensionOperation
    if the result is not a monomial.
    For example, (n^3 * m) // n == n^2*m, but n // m fails.
    """
    d = collections.Counter(self)
    for key, exponent in divisor.items():
      diff = self.get(key, 0) - exponent
      if diff < 0:
        raise core.InconclusiveDimensionOperation(f"Cannot divide {self} by {divisor}.")
      elif diff == 0: del d[key]
      elif diff > 0: d[key] = diff
    return _DimMon(d)


class _DimPolynomial(dict):
  """Polynomial with integer coefficients for polymorphic shapes.

  The shape variables are assumed to range over integers >= 1.

  We overload integer operations, but we do that soundly, raising
  :class:`core.InconclusiveDimensionOperation` when the result is not
  representable as a polynomial.

  The representation of a polynomial is as a dictionary mapping _DimMonomial to
  integer coefficients. The special monomial `_DimMonomial()` is mapped to the
  free integer coefficient of the polynomial. The constant result of arithmetic
  operations is represented as a Python constant.
  """

  def __init__(self, coeffs: Dict[_DimMon, int]):
    # Makes sure Polynomials are always in canonical form
    coeffs = {mon: op.index(coeff)
              for mon, coeff in coeffs.items() if coeff != 0}
    coeffs = coeffs or {_DimMon(): 0}
    super().__init__(coeffs)

  @classmethod
  def from_coeffs(cls, coeffs: Dict[_DimMon, int]) -> DimSize:
    """Constructs _DimPolynomial or an int."""
    has_non_zero_degree = False
    zero_degree_const = 0
    new_coeffs = {}
    for mon, count in coeffs.items():
      if count != 0:
        if mon.degree == 0:
          zero_degree_const = count
        else:
          has_non_zero_degree = True
        new_coeffs[mon] = count
    if not has_non_zero_degree:
      return int(zero_degree_const)
    return _DimPolynomial(new_coeffs)

  @classmethod
  def from_var(cls, v: str) -> '_DimPolynomial':
    return _DimPolynomial({_DimMon.from_var(v): 1})

  def to_var(self) -> Optional[str]:
    """Extract the variable name "x", from a polynomial "x" """
    items = self.items()
    if len(items) != 1:
      return None
    (mon, mon_count), = items
    if mon_count != 1:
      return None
    return mon.to_var()

  def get_vars(self) -> Set[str]:
    """The variables that appear in a polynomial."""
    acc = set()
    for mon, _ in self.items():
      acc.update(mon.get_vars())
    return acc

  def __hash__(self):
    return hash(tuple(sorted(self.items())))

  def __str__(self):
    return ' + '.join(f'{c} {mon}' if c != 1 or mon.degree == 0 else str(mon)
                      for mon, c in sorted(self.items(), reverse=True)).strip()

  def __repr__(self):
    return str(self)

  # We overload , -, *, because they are fully defined for _DimPolynomial.
  def __add__(self, other: DimSize) -> DimSize:
    coeffs = self.copy()
    for mon, coeff in _ensure_poly(other).items():
      coeffs[mon] = coeffs.get(mon, 0) + coeff
    return _DimPolynomial.from_coeffs(coeffs)

  def __sub__(self, other: DimSize) -> DimSize:
    return self + -other

  def __neg__(self) -> '_DimPolynomial':
    return _DimPolynomial({mon: -coeff for mon, coeff in self.items()})

  def __mul__(self, other: DimSize) -> DimSize:
    other = _ensure_poly(other)
    coeffs: Dict[_DimMon, int] = {}
    for (mon1, coeff1), (mon2, coeff2) in itertools.product(self.items(), other.items()):
      mon = mon1.mul(mon2)
      coeffs[mon] = coeffs.get(mon, 0) + coeff1 * coeff2
    return _DimPolynomial.from_coeffs(coeffs)

  def __rmul__(self, other: DimSize) -> DimSize:
    return self * other  # multiplication commutes

  def __radd__(self, other: DimSize) -> DimSize:
    return self + other  # addition commutes

  def __rsub__(self, other: DimSize) -> DimSize:
    return _ensure_poly(other) - self

  def eq(self, other: DimSize) -> bool:
    lb, ub = _ensure_poly(self - other).bounds()
    if lb == ub == 0:
      return True
    if lb is not None and lb > 0:
      return False
    if ub is not None and ub < 0:
      return False
    raise core.InconclusiveDimensionOperation(f"Dimension polynomial comparison '{self}' == '{other}' is inconclusive")

  # We must overload __eq__ and __ne__, or else we get unsound defaults.
  __eq__ = eq
  def __ne__(self, other: DimSize) -> bool:
    return not self.eq(other)

  def ge(self, other: DimSize) -> bool:
    lb, ub = _ensure_poly(self - other).bounds()
    if lb is not None and lb >= 0:
      return True
    if ub is not None and ub < 0:
      return False
    raise core.InconclusiveDimensionOperation(f"Dimension polynomial comparison '{self}' >= '{other}' is inconclusive")
  __ge__ = ge

  def __le__(self, other: DimSize):
    return _ensure_poly(other).__ge__(self)

  def __gt__(self, other: DimSize):
    return not _ensure_poly(other).__ge__(self)

  def __lt__(self, other: DimSize):
    return not self.__ge__(other)

  def divmod(self, divisor: DimSize) -> Tuple[DimSize, int]:
    """
    Floor division with remainder (divmod) generalized to polynomials.
    If the `divisor` is not a constant, the remainder must be 0.
    If the `divisor` is a constant, the remainder may be non 0, for consistency
    with integer divmod.

    :return: Quotient resulting from polynomial division and integer remainder.
    """
    divisor = _ensure_poly(divisor)
    dmon, dcount = divisor.leading_term
    dividend, quotient = self, 0
    err_msg = f"Dimension polynomial '{self}' is not a multiple of '{divisor}'"
    # invariant: self = dividend + divisor * quotient
    # the leading term of dividend decreases through the loop.
    while not (isinstance(dividend, int) or dividend.is_constant):
      mon, count = dividend.leading_term
      try:
        qmon = mon.divide(dmon)
      except core.InconclusiveDimensionOperation:
        raise core.InconclusiveDimensionOperation(err_msg)
      qcount, rcount = divmod(count, dcount)
      if rcount != 0:
        raise core.InconclusiveDimensionOperation(err_msg)

      q = _DimPolynomial.from_coeffs({qmon: qcount})
      quotient += q
      dividend -= q * divisor  # type: ignore[assignment]

    dividend = int(dividend)  # type: ignore[assignment]
    if divisor.is_constant:
      q, r = divmod(dividend, int(divisor))  # type: ignore
      quotient += q
      remainder = r
    else:
      if dividend != 0:
        raise core.InconclusiveDimensionOperation(err_msg)
      remainder = 0

    if config.jax_enable_checks:
      assert self == divisor * quotient + remainder
    return quotient, remainder

  def __floordiv__(self, divisor: DimSize) -> DimSize:
    return self.divmod(divisor)[0]

  def __rfloordiv__(self, other):
    return _ensure_poly(other).__floordiv__(self)

  def __mod__(self, divisor: DimSize) -> int:
    return self.divmod(divisor)[1]

  __divmod__ = divmod

  def __rdivmod__(self, other: DimSize) -> Tuple[DimSize, int]:
    return _ensure_poly(other).divmod(self)

  def __int__(self):
    if self.is_constant:
      return op.index(next(iter(self.values())))
    else:
      raise core.InconclusiveDimensionOperation(f"Dimension polynomial '{self}' is not constant")

  def bounds(self) -> Tuple[Optional[int], Optional[int]]:
    """Returns the lower and upper bounds, if defined."""
    lb = ub = self.get(_DimMon(), 0)  # The free coefficient
    for mon, coeff in self.items():
      if mon.degree == 0: continue
      if coeff > 0:
        ub = None
        lb = None if lb is None else lb + coeff
      else:
        lb = None
        ub = None if ub is None else ub + coeff
    return lb, ub

  def evaluate(self, env: Dict[str, Any]) -> Any:
    prod = lambda xs: functools.reduce(op.mul, xs) if xs else 1
    def mul(coeff, mon):
      try:
        coeff = int(coeff)
      except:
        return coeff * mon
      else:
        return 0 if coeff == 0 else mon if coeff == 1 else coeff * mon
    terms = [mul(coeff, prod([pow(env[id], deg) for id, deg in mon.items()]))
             for mon, coeff in self.items()]
    return sum(terms) if len(terms) > 1 else terms[0]

  @property
  def is_constant(self):
    return len(self) == 1 and next(iter(self)).degree == 0

  @property
  def leading_term(self) -> Tuple[_DimMon, int]:
    """Returns the highest degree term that comes first lexicographically."""
    return max(self.items())


def _ensure_poly(p: DimSize) -> _DimPolynomial:
  if isinstance(p, _DimPolynomial): return p
  return _DimPolynomial({_DimMon(): p})

def is_poly_dim(p: DimSize) -> bool:
  return isinstance(p, _DimPolynomial)


class DimensionHandlerPoly(core.DimensionHandler):
  """See core.DimensionHandler.

  Most methods are inherited.
  """
  def is_constant(self, d: DimSize) -> bool:
    assert isinstance(d, _DimPolynomial)
    return False

  def symbolic_equal(self, d1: core.DimSize, d2: core.DimSize) -> bool:
    try:
      return _ensure_poly(d1) == d2
    except core.InconclusiveDimensionOperation:
      return False

  def greater_equal(self, d1: DimSize, d2: DimSize):
    return _ensure_poly(d1) >= d2

  def divide_shape_sizes(self, s1: Shape, s2: Shape) -> DimSize:
    sz1 = np.prod(s1)
    sz2 = np.prod(s2)
    if core.symbolic_equal_dim(sz1, sz2):  # Takes care also of sz1 == sz2 == 0
      return 1
    err_msg = f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}"
    try:
      q, r = _ensure_poly(sz1).divmod(sz2)
    except core.InconclusiveDimensionOperation:
      raise core.InconclusiveDimensionOperation(err_msg)
    if r != 0:
      raise core.InconclusiveDimensionOperation(err_msg)
    return q  # type: ignore[return-value]

  def stride(self, d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
    """Implements `(d - window_size) // window_stride + 1`"""
    try:
      q, r = _ensure_poly(d - window_size).divmod(window_stride)
      return q + 1
    except core.InconclusiveDimensionOperation as e:
      raise core.InconclusiveDimensionOperation(
          f"Cannot compute stride for dimension '{d}', "
          f"window_size '{window_size}', stride '{window_stride}'. Reason: {e}.")
    return d

core._SPECIAL_DIMENSION_HANDLERS[_DimPolynomial] = DimensionHandlerPoly()

def _einsum_contract_path(*operands, **kwargs):
  """Like opt_einsum.contract_path, with support for DimPolynomial shapes.

  We use opt_einsum.contract_path to compute the schedule, using a fixed
  constant for all dimension variables. This is safe because we throw an
  error if there are more than 1 contractions. Essentially, we just use
  opt_einsum.contract_path to parse the specification.
  """

  # Replace the polymorphic shapes with some concrete shapes for calling
  # into opt_einsum.contract_path, because the latter wants to compute the
  # sizes of operands and intermediate results.
  fake_ops = []
  for operand in operands:
    if isinstance(operand, str):
      fake_ops.append(operand)
    else:
      shape = np.shape(operand)
      def fake_dim(d):
        if core.is_constant_dim(d):
          return d
        else:
          if not isinstance(d, _DimPolynomial):
            raise TypeError(f"Encountered unexpected shape dimension {d}")
          # It is Ok to replace all polynomials with the same value. We may miss
          # here some errors due to non-equal dimensions, but we catch them
          # later.
          return 8
      fake_ops.append(jax.ShapeDtypeStruct(tuple(map(fake_dim, shape)),
                                           operand.dtype))

  contract_fake_ops, contractions = opt_einsum.contract_path(*fake_ops,
                                                             **kwargs)
  if len(contractions) > 1:
    msg = ("Shape polymorphism is not yet supported for einsum with more than "
           f"one contraction {contractions}")
    raise ValueError(msg)
  contract_operands = []
  for operand in contract_fake_ops:
    idx = tuple(i for i, fake_op in enumerate(fake_ops) if operand is fake_op)
    assert len(idx) == 1
    contract_operands.append(operands[idx[0]])
  return contract_operands, contractions

lax_numpy._polymorphic_einsum_contract_path_handlers[_DimPolynomial] = _einsum_contract_path


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
      return _DimPolynomial.from_var(dim_spec)
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
      return _DimPolynomial.from_var(dim_spec)

  dims = tuple([_process_dim(i, ds) for i, ds in enumerate(spec_tuple)])
  for dim_var, dim_var_values in shape_var_map.items():
    if len(dim_var_values) != 1:
      msg = (f"PolyShape '{spec}' has dimension variable '{dim_var}' "
             f"corresponding to multiple values ({sorted(dim_var_values)}), for "
             f"argument shape {arg_shape}")
      raise ValueError(msg)
    elif list(dim_var_values)[0] <= 0:
      msg = (f"PolyShape '{spec}' has dimension variable '{dim_var}' "
             f"corresponding to 0, for argument shape {arg_shape}")
      raise ValueError(msg)

  return dims


def eval_shape(shape: Sequence[DimSize], shape_env: Dict[str, Any]) -> Sequence[Any]:
  return tuple(d.evaluate(shape_env) if type(d) is _DimPolynomial else d   # type: ignore
               for d in shape)
