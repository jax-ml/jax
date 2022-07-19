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
"""Shape polymorphism support.

We introduce a set of dimension variables at the top-level of a `jit` function.
They are introduced implicitly by way of specifying for each dimension of each
argument a dimension polynomial in terms of some dimension variables.
All dimension variables are assumed to range over integers greater or equal to 1.

Dimension polynomials overload some integer operations, such as
add, multiply, equality, etc. The JAX NumPy layer and the LAX layers have been
touched up to be sensitive to handling shapes that contain dimension
polynomials. This enables many JAX programs to be traced with dimension
polynomials in some dimensions. A priority has been to enable the batch
dimension in neural network examples to be polymorphic.

This was built initially for jax2tf, but it is now customizeable to be
independent of TF. The best documentation at the moment is in the
jax2tf.convert docstring, and the
[README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
"""
import collections
import dataclasses
import itertools
import functools
import operator as op
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import jax
from jax._src.numpy import lax_numpy
from jax._src import dtypes
from jax._src.lax import lax
import opt_einsum
from jax import config
from jax import core

import numpy as np

DimSize = core.DimSize
Shape = core.Shape
TfVal = Any
# A dimension environment maps dimension variables to expressions that
# compute the value of the dimension. These expressions refer to the
# function arguments.
ShapeEnv = Dict[str, Any]
DType = Any

class InconclusiveDimensionOperation(core.InconclusiveDimensionOperation):
  """Raised when we cannot conclusively compute with symbolic dimensions."""

  _help_msg = """
This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
"""

  def __init__(self, message: str):
    error_msg = f"{message}\n{InconclusiveDimensionOperation._help_msg}"
    # https://github.com/python/mypy/issues/5887
    super().__init__(error_msg)  # type: ignore


class _DimMon(dict):
  """Represents a multivariate monomial, such as n^3 * m.

  The representation is a dictionary mapping var:exponent.
  The `var` are strings and the exponents are integers >= 1.
  The dimension variables are assumed to range over integers >= 1.
  """
  def __hash__(self):
    return hash(frozenset(self.items()))

  def __str__(self):
    return "*".join(f"{key}^{exponent}" if exponent != 1 else str(key)
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
    Divides by another monomial. Raises a InconclusiveDimensionOperation
    if the result is not a monomial.
    For example, (n^3 * m) // n == n^2*m, but n // m fails.
    """
    d = collections.Counter(self)
    for key, exponent in divisor.items():
      diff = self.get(key, 0) - exponent
      if diff < 0:
        raise InconclusiveDimensionOperation(f"Cannot divide {self} by {divisor}.")
      elif diff == 0: del d[key]
      elif diff > 0: d[key] = diff
    return _DimMon(d)

  def evaluate(self, env: ShapeEnv):
    prod = lambda xs: functools.reduce(_multiply, xs) if xs else np.int32(1)
    def pow_opt(v, p: int):
      return v if p == 1 else prod([v] * p)
    return prod([pow_opt(env[id], deg) for id, deg in self.items()])


class _DimPolynomial():
  """Polynomial with integer coefficients for polymorphic shapes.

  The dimension variables are assumed to range over integers >= 1.

  We overload integer operations, but we do that soundly, raising
  :class:`InconclusiveDimensionOperation` when the result is not
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
    self._coeffs = coeffs or {_DimMon(): 0}

  def monomials(self) -> Iterable[Tuple[_DimMon, int]]:
    return self._coeffs.items()

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
    items = self.monomials()
    if len(items) != 1:  # type: ignore
      return None
    (mon, mon_count), = items
    if mon_count != 1:
      return None
    return mon.to_var()

  def get_vars(self) -> Set[str]:
    """The variables that appear in a polynomial."""
    acc = set()
    for mon, _ in self.monomials():
      acc.update(mon.get_vars())
    return acc

  def __hash__(self):
    return hash(tuple(sorted(self.monomials())))

  def __str__(self):
    def _one_monomial(mon, c):
      if mon.degree == 0:
        return str(c)
      if c == 1:
        return str(mon)
      return f"{c}*{mon}"
    return " + ".join(_one_monomial(mon, c)
                      for mon, c in sorted(self.monomials(), reverse=True))

  def __repr__(self):
    return str(self)

  # We overload , -, *, because they are fully defined for _DimPolynomial.
  def __add__(self, other: DimSize) -> DimSize:
    coeffs = self._coeffs.copy()
    for mon, coeff in _ensure_poly(other).monomials():
      coeffs[mon] = coeffs.get(mon, 0) + coeff
    return _DimPolynomial.from_coeffs(coeffs)

  def __sub__(self, other: DimSize) -> DimSize:
    return self + -other

  def __neg__(self) -> '_DimPolynomial':
    return _DimPolynomial({mon: -coeff for mon, coeff in self.monomials()})

  def __mul__(self, other: DimSize) -> DimSize:
    other = _ensure_poly(other)
    coeffs: Dict[_DimMon, int] = {}
    for (mon1, coeff1), (mon2, coeff2) in itertools.product(self.monomials(), other.monomials()):
      mon = mon1.mul(mon2)
      coeffs[mon] = coeffs.get(mon, 0) + coeff1 * coeff2
    return _DimPolynomial.from_coeffs(coeffs)

  def __pow__(self, power, modulo=None):
    assert modulo is None
    try:
      power = int(power)
    except:
      raise InconclusiveDimensionOperation(f"Dimension polynomial cannot be raised to non-integer power '{self}' ^ '{power}'")
    return functools.reduce(op.mul, [self] * power)

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
    raise InconclusiveDimensionOperation(f"Dimension polynomial comparison '{self}' == '{other}' is inconclusive")

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
    raise InconclusiveDimensionOperation(f"Dimension polynomial comparison '{self}' >= '{other}' is inconclusive")
  __ge__ = ge

  def __le__(self, other: DimSize):
    return _ensure_poly(other).__ge__(self)

  def __gt__(self, other: DimSize):
    return not _ensure_poly(other).__ge__(self)

  def __lt__(self, other: DimSize):
    return not self.__ge__(other)

  def _division_error_msg(self, dividend, divisor, details: str = "") -> str:
    msg = f"Cannot divide '{dividend}' by '{divisor}'."
    if details:
      msg += f"\nDetails: {details}."
    msg += "\nSee https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported."
    return msg

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
    # invariant: self = dividend + divisor * quotient
    # the leading term of dividend decreases through the loop.
    while is_poly_dim(dividend) and not dividend.is_constant:
      mon, count = dividend.leading_term
      try:
        qmon = mon.divide(dmon)
      except InconclusiveDimensionOperation as e:
        raise InconclusiveDimensionOperation(
            self._division_error_msg(self, divisor, str(e)))
      qcount, rcount = divmod(count, dcount)
      if rcount != 0:
        raise InconclusiveDimensionOperation(
            self._division_error_msg(self, divisor))

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
        raise InconclusiveDimensionOperation(
            self._division_error_msg(self, divisor))
      remainder = 0

    if config.jax_enable_checks:
      assert self == divisor * quotient + remainder
    return quotient, remainder

  def __floordiv__(self, divisor: DimSize) -> DimSize:
    return self.divmod(divisor)[0]

  def __rfloordiv__(self, other):
    return _ensure_poly(other).__floordiv__(self)

  def __truediv__(self, divisor: DimSize):
    # Used for "/"
    q, r = self.divmod(divisor)
    if r != 0:
      raise InconclusiveDimensionOperation(
          self._division_error_msg(self, divisor,
                                   f"Remainder is not zero: {r}"))
    return q

  def __rtruediv__(self, dividend: DimSize):
    # Used for "/", when dividend is not a _DimPolynomial
    raise InconclusiveDimensionOperation(
        self._division_error_msg(dividend, self, "Dividend must be a polynomial"))

  def __mod__(self, divisor: DimSize) -> int:
    return self.divmod(divisor)[1]

  __divmod__ = divmod

  def __rdivmod__(self, other: DimSize) -> Tuple[DimSize, int]:
    return _ensure_poly(other).divmod(self)

  def __int__(self):
    if self.is_constant:
      return op.index(next(iter(self._coeffs.values())))
    else:
      raise InconclusiveDimensionOperation(f"Dimension polynomial '{self}' used in a context that requires a constant")

  def bounds(self) -> Tuple[Optional[int], Optional[int]]:
    """Returns the lower and upper bounds, if defined."""
    lb = ub = self._coeffs.get(_DimMon(), 0)  # The free coefficient
    for mon, coeff in self.monomials():
      if mon.degree == 0: continue
      if coeff > 0:
        ub = None  # type: ignore
        lb = None if lb is None else lb + coeff
      else:
        lb = None  # type: ignore
        ub = None if ub is None else ub + coeff
    return lb, ub

  @property
  def is_constant(self):
    return len(self._coeffs) == 1 and next(iter(self._coeffs)).degree == 0

  @property
  def leading_term(self) -> Tuple[_DimMon, int]:
    """Returns the highest degree term that comes first lexicographically."""
    return max(self.monomials())

  def evaluate(self, env: ShapeEnv):
    terms = [_multiply(mon.evaluate(env), np.int32(coeff)) for mon, coeff in self.monomials()]
    return functools.reduce(_add, terms) if len(terms) > 1 else terms[0]

  @staticmethod
  def get_aval(_: "_DimPolynomial"):
    return core.ShapedArray((),
                            dtypes.canonicalize_dtype(np.int64),
                            weak_type=True)


core.pytype_aval_mappings[_DimPolynomial] = _DimPolynomial.get_aval

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
    except InconclusiveDimensionOperation:
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
    except InconclusiveDimensionOperation as e:
      raise InconclusiveDimensionOperation(err_msg + f"\nDetails: {e}")
    if r != 0:
      raise InconclusiveDimensionOperation(err_msg + f"\nRemainder is not zero: {r}")
    return q  # type: ignore[return-value]

  def stride(self, d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
    """Implements `(d - window_size) // window_stride + 1`"""
    try:
      # TODO(necula): check for d == 0 or window_size > d and return 0.
      q, r = _ensure_poly(d - window_size).divmod(window_stride)
      return q + 1
    except InconclusiveDimensionOperation as e:
      raise InconclusiveDimensionOperation(
          f"Cannot compute stride for dimension '{d}', "
          f"window_size '{window_size}', stride '{window_stride}'.\nDetails: {e}.")
    return d

  def as_value(self, d: DimSize):
    """Turns a dimension size into a Jax value that we can compute with."""
    return _dim_as_value(d)

core._SPECIAL_DIMENSION_HANDLERS[_DimPolynomial] = DimensionHandlerPoly()
dtypes.python_scalar_dtypes[_DimPolynomial] = dtypes.python_scalar_dtypes[int]

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
    # We replace only array operands
    if not hasattr(operand, "dtype"):
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
  contract_operands = []
  for operand in contract_fake_ops:
    idx = tuple(i for i, fake_op in enumerate(fake_ops) if operand is fake_op)
    assert len(idx) == 1
    contract_operands.append(operands[idx[0]])
  return contract_operands, contractions

lax_numpy._polymorphic_einsum_contract_path_handlers[_DimPolynomial] = _einsum_contract_path

# A JAX primitive with no array arguments but with a dimension parameter
# that is a DimPoly. The value of the primitive is the value of the dimension.
# This primitive is used only in the context of jax2tf, so it does not need
# XLA translation rules.
dim_as_value_p = core.Primitive("dim_as_value")
def _dim_as_value_abstract(dim: DimSize) -> core.AbstractValue:
  return core.ShapedArray((), np.int32)

dim_as_value_p.def_abstract_eval(_dim_as_value_abstract)

def _dim_as_value(dim: DimSize):
  return dim_as_value_p.bind(dim=dim)

class PolyShape(tuple):
  """Tuple of polymorphic dimension specifications.

  See docstring of :func:`jax2tf.convert`.
  """

  def __init__(self, *dim_specs):
    tuple.__init__(dim_specs)

  def __new__(cls, *dim_specs):
    for ds in dim_specs:
      if not isinstance(ds, (int, str)) and ds != ...:
        msg = (f"Invalid polymorphic shape element: {repr(ds)}; must be a string "
               "representing a dimension variable, or an integer, or ...")
        raise ValueError(msg)
    return tuple.__new__(PolyShape, dim_specs)


def _parse_spec(spec: Optional[Union[str, PolyShape]],
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
    spec_tuple = spec
  elif isinstance(spec, str):
    spec_ = spec.strip()
    if spec_[0] == "(":
      if spec_[-1] != ")":
        raise ValueError(f"polymorphic shape {repr(spec)} has invalid syntax")
      spec_ = spec_[1:-1]
      spec_ = spec_.strip()
    spec_ = spec_.rstrip(",")
    if not spec_:
      spec_tuple = ()
    else:
      spec_tuple = spec_.split(",")  # type: ignore
  else:
    raise ValueError(f"polymorphic shape {repr(spec)} must be either None, a string, or PolyShape.")

  # Process ...
  spec_tuple = tuple(map(lambda s: ... if isinstance(s, str) and s.strip() == "..." else s,
                         spec_tuple))
  ds_ellipses = tuple(ds for ds in spec_tuple if ds == ...)
  if ds_ellipses:
    if len(ds_ellipses) > 1 or spec_tuple[-1] != ...:
      raise ValueError(f"polymorphic shape {repr(spec)} can contain Ellipsis only at the end.")
    spec_tuple = spec_tuple[0:-1]
    if len(arg_shape) >= len(spec_tuple):
      spec_tuple = spec_tuple + ("_",) * (len(arg_shape) - len(spec_tuple))

  if len(arg_shape) != len(spec_tuple):
    raise ValueError(f"polymorphic shape {repr(spec)} of rank {len(spec_tuple)} must match the rank {len(arg_shape)} of argument shape {arg_shape}.")

  # The actual parsing.
  # We actually parse not just dimension variables, but polynomials.
  # This is not a supported feature of the API, but is needed when parsing the
  # polymorphic_shapes of a gradient function, when the primal function has polynomial
  # output shapes.
  def _parse_dim(dim_spec: Union[str, int]) -> DimSize:
    if isinstance(dim_spec, int):
      return dim_spec  #
    dim_spec = dim_spec.strip()
    if not dim_spec:
      raise ValueError(f"polymorphic shape {repr(spec)} has invalid syntax (empty dimension '{dim_spec}')")
    # Terms are separated by "+"
    terms = dim_spec.split("+")
    if not terms:
      raise ValueError(f"polymorphic shape {repr(spec)} has invalid syntax (empty dimension '{dim_spec}')")
    def _parse_term(term_spec: str) -> DimSize:
      term_spec = term_spec.strip()
      # Factors are separated by "*"
      factors = term_spec.split("*")
      if not factors:
        raise ValueError(f"polymorphic shape {repr(spec)} has invalid syntax (unexpected term '{term_spec}')")
      def _parse_factor(factor_spec: str) -> DimSize:
        factor_spec = factor_spec.strip()
        if re.match(r"^-?\d+$", factor_spec):
          return int(factor_spec)
        m = re.match(r"^([a-zA-Z]\w*)(\^(\d+))?$", factor_spec)
        if not m:
          raise ValueError(f"polymorphic shape {repr(spec)} has invalid syntax (unexpected term '{factor_spec}')")
        var = _DimPolynomial.from_var(m.group(1))
        if m.group(3) is None:
          return var
        return var ** int(m.group(3))

      return functools.reduce(op.mul, map(_parse_factor, factors))
    return functools.reduce(op.add, map(_parse_term, terms))

  def _process_dim(i: int, dim_spec: Union[str, int]):
    if isinstance(dim_spec, str):
      dim_spec = dim_spec.strip()
    dim_size = arg_shape[i]
    if dim_size is None:
      def need_dim_var_msg():
        msg = (f"polymorphic shape {repr(spec)} in axis {i} must contain a dimension variable "
               f"for unknown dimension in argument shape {arg_shape}")
        if spec is None:
          msg += ". Perhaps you forgot to add the polymorphic_shapes= parameter to jax2tf.convert?"
        return msg

      if dim_spec == "_":
        raise ValueError(need_dim_var_msg())
      dim_poly = _parse_dim(dim_spec)
      if not is_poly_dim(dim_poly):
        raise ValueError(need_dim_var_msg())
      return dim_poly
    else:  # dim_size is known
      dim_size = int(dim_size)
      if dim_spec == "_":
        return dim_size
      dim_poly = _parse_dim(dim_spec)
      if not is_poly_dim(dim_poly):
        if dim_poly != dim_size:
          msg = (f"polymorphic shape {repr(spec)} in axis {i} must match the "
                 f"known dimension size {dim_size} for argument shape {arg_shape}")
          raise ValueError(msg)
        return dim_size
      return dim_poly

  dims = tuple(_process_dim(i, ds) for i, ds in enumerate(spec_tuple))
  return dims


def _add(v1, v2):
  if isinstance(v1, int) and v1 == 0:
    return v2
  elif isinstance(v2, int) and v2 == 0:
    return v1
  else:
    return v1 + v2

def _multiply(v1, v2):
  if isinstance(v1, int) and v1 == 1:
    return v2
  elif isinstance(v2, int) and v2 == 1:
    return v1
  else:
    return v1 * v2

def _is_known_constant(v) -> Optional[int]:
  try:
    return int(v)
  except Exception:
    # TODO(necula): added this so that in jax2tf, in Eager mode, we can tell
    # that a tensor is a constant. We should move this dependency into some
    # jax2tf-specific area.
    if hasattr(v, "val"):
      try:
        vint = int(v.val)
        if isinstance(vint, int):  # In TF, int(tf.Tensor) is tf.Tensor!
          return vint
      except Exception:
        pass
    return None

# dimension_size(operand, dimension=i) get the operand.shape[i] as a
# JAX value.
dimension_size_p = core.Primitive("dimension_size")
def _dimension_size_abstract(aval: core.AbstractValue, **_) -> core.AbstractValue:
  return core.ShapedArray((), np.int32)

dimension_size_p.def_abstract_eval(_dimension_size_abstract)

def _dimension_size_impl(arg, *, dimension):
  return arg.shape[dimension]
dimension_size_p.def_impl(_dimension_size_impl)

_JaxValue = Any

@dataclasses.dataclass
class DimEquation:
  # Represents poly == _expr
  poly: _DimPolynomial
  dim_expr: _JaxValue


def get_shape_evaluator(dim_vars: Sequence[str], shape: Sequence[DimSize]) ->\
  Tuple[Callable, Sequence[core.AbstractValue]]:
  """Prepares a shape evaluator.

  Returns a pair with the first component a function that given the values
  for the dimension variables returns the values for the dimensions of `shape`.
  The second component of the returned pair is a tuple of AbstractValues for
  the dimension variables.
  """
  def eval_shape(*dim_values: Any) -> Sequence[Any]:
    shape_env_jax = dict(zip(dim_vars, dim_values))

    def eval_dim(d: DimSize):
      return d.evaluate(shape_env_jax)  # type: ignore[union-attr]

    return tuple(eval_dim(d) if type(d) is _DimPolynomial else np.int32(d)  # type: ignore
                 for d in shape)
  return (eval_shape,
          tuple(core.ShapedArray((), np.int32) for _ in dim_vars))

def args_avals(
    arg_shapes: Sequence[Sequence[Optional[int]]],
    arg_jax_dtypes: Sequence[DType],
    polymorphic_shapes: Sequence[Optional[Union[str, PolyShape]]]) -> \
  Sequence[core.ShapedArray]:
  """Computes abstract values.

  Args:
    arg_shapes: the shapes for the arguments, possibly having None dimensions.
    arg_dtypes: the inferred JAX dtypes for the args.
    polymorphic_shapes: the polymorphic specifications for the arguments.
  Returns: a sequence of abstract values corresponding to the arguments.
  """

  def input_aval(arg_shape: Sequence[Optional[int]],
                 arg_jax_dtype: DType,
                 polymorphic_shape: Optional[str]) -> core.ShapedArray:
    """The abstract value for an input."""
    aval_shape = _parse_spec(polymorphic_shape, arg_shape)
    return core.ShapedArray(aval_shape, arg_jax_dtype)

  avals = tuple(map(input_aval, arg_shapes, arg_jax_dtypes, polymorphic_shapes))  # type: ignore
  return avals


def prepare_dim_var_env(args_avals: Sequence[core.AbstractValue]) -> \
    Tuple[Sequence[str], Callable]:
  """Get the dimension variables and the function to compute them.

  Returns a tuple of dimension variables that appear in `args_avals` along
  with a function that given the actual arguments of the top-level function
  returns a tuple of dimension variable values, in the same order as the
  dimension variables returns in the first component.
  """
  dim_vars: Set[str] = set()
  for a in args_avals:
    for d in a.shape:
      if is_poly_dim(d):
        dim_vars = dim_vars.union(d.get_vars())

  def get_dim_var_values(*args: Any) -> Sequence[Any]:
    dim_equations: List[DimEquation] = []
    for a in args:
      for i, d in enumerate(a.shape):
        if is_poly_dim(d):
          dim_equations.append(DimEquation(
              poly=d, dim_expr=dimension_size_p.bind(a, dimension=i)))

    dim_env = _solve_dim_equations(dim_equations)
    return tuple(dim_env[dv] for dv in dim_vars)
  return tuple(dim_vars), get_dim_var_values

def _solve_dim_equations(eqns: List[DimEquation]) -> ShapeEnv:
  # Returns a shape environment if it can solve all dimension variables.
  # Raises an exception if it cannot.
  shapeenv: ShapeEnv = {}

  def _shapeenv_to_str() -> str:
    if shapeenv:
      return (" Partial solution: " +
              ", ".join([f"{var} = {val}" for var, val in shapeenv.items()]) + ".")
    else:
      return ""

  def process_one_eqn(eqn: DimEquation) -> bool:
    # Try to rewrite the equation as "var * factor_var = dim_expr" (a linear
    # uni-variate equation. Return False if this rewrite fails.
    # Otherwise, add the variable to shapeenv and return True.

    # The invariant is: var * factor_var + rest_eqn_poly = dim_expr
    var, factor_var = None, None
    dim_expr = eqn.dim_expr

    for mon, factor in eqn.poly.monomials():
      # Perhaps we can already evaluate this monomial (all vars solved)
      try:
        mon_value = mon.evaluate(shapeenv)
        dim_expr = dim_expr - _multiply(mon_value, np.int32(factor))
        continue
      except KeyError:
        # There are some indeterminate variables. We handle only the case of
        # linear remaining indeterminates.
        v = mon.to_var()
        if v is not None and var is None:
          var = v
          factor_var = factor
          continue
      return False

    if var is not None:
      if factor_var == 1:
        var_value, var_remainder = dim_expr, np.int32(0)
      else:
        var_value = lax.div(dim_expr, np.int32(factor_var))  # type: ignore
        var_remainder = lax.rem(dim_expr, np.int32(factor_var))  # type: ignore

      # Check that the division is even. Works only in eager mode.
      var_remainder_int = _is_known_constant(var_remainder)
      if var_remainder_int is not None and var_remainder_int != 0:
        # TODO(necula): check even in graph mode, by embedding the checks in
        # the graph.
        msg = (f"Dimension variable {var} must have integer value >= 1. "  # type: ignore
               f"Found value {int(_is_known_constant(dim_expr)) / factor_var} when solving "  # type: ignore
               f"{eqn.poly} == {eqn.dim_expr}.{_shapeenv_to_str()}")
        raise ValueError(msg)
      var_value_int = _is_known_constant(var_value)
      if var_value_int is not None and var_value_int <= 0:
        msg = (f"{var_value_int} Dimension variable {var} must have integer value >= 1. "
               f"Found value {int(var_value_int)} when solving "
               f"{eqn.poly} == {eqn.dim_expr}.{_shapeenv_to_str()}")
        raise ValueError(msg)

      shapeenv[var] = var_value
      return True
    else:
      # All variables are resolved for this equation
      dim_expr_int = _is_known_constant(dim_expr)
      if dim_expr_int is not None and dim_expr_int != 0:
        err_msg = (
            "Found inconsistency when solving "
            f"{eqn.poly} == {eqn.dim_expr}.{_shapeenv_to_str()}")
        raise ValueError(err_msg)
      return True

  while True:
    nr_eqns = len(eqns)
    eqns = [eqn for eqn in eqns if not process_one_eqn(eqn)]
    if not eqns:
      return shapeenv  # SUCCESS
    elif len(eqns) >= nr_eqns:
      break

  # We have some equations that we cannot solve further
  unsolved_vars: Set[str] = set()
  unsolved_polys: List[_DimPolynomial] = []
  for eqn in eqns:
    unsolved_vars = unsolved_vars.union(eqn.poly.get_vars())
    unsolved_polys.append(eqn.poly)
  unsolved_vars = unsolved_vars.difference(shapeenv.keys())
  eqns_str = "\n  ".join([str(eqn.poly) for eqn in eqns])
  err_msg = (
      f"Cannot solve for values of dimension variables {unsolved_vars} from "
      f"the remaining dimension polynomials\n  {eqns_str}.{_shapeenv_to_str()} "
      "Dimension variables can be solved only from linear polynomials.")
  raise ValueError(err_msg)
