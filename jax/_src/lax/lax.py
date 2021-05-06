# Copyright 2018 Google LLC
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

# Pytype is too slow to check this file.
# pytype: skip-file

import builtins
from enum import IntEnum
import functools
import itertools
import operator
from typing import (Any, Callable, List, NamedTuple, Optional, Sequence,\
                    Union, Tuple)
import warnings

import numpy as np

import jax
from jax import core
from jax import ad_util
from jax._src import api
from jax import api_util
from jax import linear_util as lu
from jax._src import dtypes
from jax import tree_util
from jax._src.config import config
from jax.core import (Primitive, _canonicalize_dimension, UnshapedArray,
                      ShapedArray, ConcreteArray, raise_to_shaped,
                      abstract_token, canonicalize_shape)
from jax.abstract_arrays import array_types
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters import ad
from jax.interpreters import invertible_ad as iad
from jax.interpreters import batching
from jax.interpreters import masking
from jax._src.util import (cache, safe_zip, partial, prod, safe_map,
                           canonicalize_axis, split_list)
from jax.tree_util import tree_map
from jax.lib import pytree
from jax.lib import xla_bridge
from jax.lib import xla_client

xb = xla_bridge
xc = xla_client
xops = xla_client.ops

_max = builtins.max
_min = builtins.min
_reduce = functools.reduce

Array = Any
DType = Any
Shape = core.Shape

def _try_broadcast_shapes(shapes):
  assert shapes
  if len(shapes) == 1: return shapes[0]
  rank, *others = {len(shape) for shape in shapes}
  if others: return None  # must have consistent rank
  if not rank: return ()  # scalar case
  result_shape = [None] * rank
  for i, sizes in enumerate(zip(*shapes)):
    non_1s = set([d for d in sizes if not core.symbolic_equal_dim(d, 1)])
    if len(non_1s) > 1:
      return None  # must have equal sizes other than 1-sized axes
    result_shape[i] = next(iter(non_1s), 1)

  return tuple(result_shape)

@cache()
def broadcast_shapes(*shapes):
  """Returns the shape that results from NumPy broadcasting of `shapes`."""
  if len(shapes) == 1:
    return shapes[0]
  ndim = _max(len(shape) for shape in shapes)
  shapes = [(1,) * (ndim - len(shape)) + shape for shape in shapes]
  result_shape = _try_broadcast_shapes(shapes)
  if result_shape is None:
    raise ValueError("Incompatible shapes for broadcasting: {}"
                     .format(tuple(map(tuple, shapes))))
  return result_shape

def _identity(x): return x

### traceables

def neg(x: Array) -> Array:
  r"""Elementwise negation: :math:`-x`."""
  return neg_p.bind(x)

def sign(x: Array) -> Array:
  r"""Elementwise sign.

  For floating-point inputs, returns
  :math:`\mathrm{sign}(x) = \begin{cases}
  -1 & x < 0\\
  -0 & x = -0\\
  \mathit{NaN} & x = \mathit{NaN}\\
  +0 & x = +0\\
  1 & x > 0
  \end{cases}`

  For signed integer inputs, returns
  :math:`\mathrm{sign}(x) = \begin{cases}
  -1 & x < 0\\
  0 & x = 0\\
  1 & x > 0
  \end{cases}`

  For complex inputs, returns the complex phase, i.e.
  :math:`\mathrm{sign}(x) = \frac{x}{|x|}`.
  """
  return sign_p.bind(x)

def nextafter(x1: Array, x2: Array) -> Array:
  r"""Returns the next representable value after `x1` in the direction of `x2`.

  Note that in some environments flush-denormal-to-zero semantics is used.
  This means that, around zero, this function returns strictly non-zero
  values which appear as zero in any operations. Consider this example::

    >>> jnp.nextafter(0, 1)  # denormal numbers are representable
    DeviceArray(1.e-45, dtype=float32)
    >>> jnp.nextafter(0, 1) * 1  # but are flushed to zero
    DeviceArray(0., dtype=float32)

  For the smallest usable (i.e. normal) float, use ``tiny`` of ``jnp.finfo``.
  """
  return nextafter_p.bind(x1, x2)

def floor(x: Array) -> Array:
  r"""Elementwise floor: :math:`\left\lfloor x \right\rfloor`."""
  return floor_p.bind(x)

def ceil(x: Array) -> Array:
  r"""Elementwise ceiling: :math:`\left\lceil x \right\rceil`."""
  return ceil_p.bind(x)

class RoundingMethod(IntEnum):
  AWAY_FROM_ZERO = 0
  TO_NEAREST_EVEN = 1

def round(x: Array,
          rounding_method: RoundingMethod = RoundingMethod.AWAY_FROM_ZERO
          ) -> Array:
  r"""Elementwise round.

  Rounds values to the nearest integer.

  Args:
    x: an array or scalar value to round.
    rounding_method: the method to use when rounding halfway values
      (e.g., `0.5`). See ``lax.RoundingMethod`` for the list of possible
      values.

  Returns:
    An array containing the elementwise rounding of x.
  """
  rounding_method = RoundingMethod(rounding_method)
  return round_p.bind(x, rounding_method=rounding_method)

def is_finite(x: Array) -> Array:
  r"""Elementwise :math:`\mathrm{isfinite}`.

  For each element x returns `True` if and only if x is not :math:`\pm\infty` or
  :math:`\mathit{NaN}`.
  """
  return is_finite_p.bind(x)

def exp(x: Array) -> Array:
  r"""Elementwise exponential: :math:`e^x`."""
  return exp_p.bind(x)

def expm1(x: Array) -> Array:
  r"""Elementwise :math:`e^{x} - 1`."""
  return expm1_p.bind(x)

def log(x: Array) -> Array:
  r"""Elementwise natural logarithm: :math:`\mathrm{log}(x)`."""
  return log_p.bind(x)

def log1p(x: Array) -> Array:
  r"""Elementwise :math:`\mathrm{log}(1 + x)`."""
  return log1p_p.bind(x)

def tanh(x: Array) -> Array:
  r"""Elementwise hyperbolic tangent: :math:`\mathrm{tanh}(x)`."""
  return tanh_p.bind(x)

def sin(x: Array) -> Array:
  r"""Elementwise sine: :math:`\mathrm{sin}(x)`."""
  return sin_p.bind(x)

def cos(x: Array) -> Array:
  r"""Elementwise cosine: :math:`\mathrm{cos}(x)`."""
  return cos_p.bind(x)

def atan2(x: Array, y: Array) -> Array:
  r"""Elementwise arc tangent of two variables:
    :math:`\mathrm{atan}({x \over y})`."""
  return atan2_p.bind(x, y)

def betainc(a: Array, b: Array, x: Array) -> Array:
  r"""Elementwise regularized incomplete beta integral."""
  return regularized_incomplete_beta_p.bind(a, b, x)

def lgamma(x: Array) -> Array:
  r"""Elementwise log gamma: :math:`\mathrm{log}(\Gamma(x))`."""
  return lgamma_p.bind(x)

def digamma(x: Array) -> Array:
  r"""Elementwise digamma: :math:`\psi(x)`."""
  return digamma_p.bind(x)

def igamma(a: Array, x: Array) -> Array:
  r"""Elementwise regularized incomplete gamma function."""
  return igamma_p.bind(a, x)

def igammac(a: Array, x: Array) -> Array:
  r"""Elementwise complementary regularized incomplete gamma function."""
  return igammac_p.bind(a, x)

def igamma_grad_a(a: Array, x: Array) -> Array:
  r"""Elementwise derivative of the regularized incomplete gamma function."""
  return igamma_grad_a_p.bind(a, x)

def random_gamma_grad(a: Array, x: Array) -> Array:
  r"""Elementwise derivative of samples from `Gamma(a, 1)`."""
  return random_gamma_grad_p.bind(a, x)

def bessel_i0e(x: Array) -> Array:
  r"""Exponentially scaled modified Bessel function of order 0:
  :math:`\mathrm{i0e}(x) = e^{-|x|} \mathrm{i0}(x)`
  """
  return bessel_i0e_p.bind(x)

def bessel_i1e(x: Array) -> Array:
  r"""Exponentially scaled modified Bessel function of order 1:
  :math:`\mathrm{i1e}(x) = e^{-|x|} \mathrm{i1}(x)`
  """
  return bessel_i1e_p.bind(x)

def erf(x: Array) -> Array:
  r"""Elementwise error function: :math:`\mathrm{erf}(x)`."""
  return erf_p.bind(x)

def erfc(x: Array) -> Array:
  r"""Elementwise complementary error function:
    :math:`\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)`."""
  return erfc_p.bind(x)

def erf_inv(x: Array) -> Array:
  r"""Elementwise inverse error function: :math:`\mathrm{erf}^{-1}(x)`."""
  return erf_inv_p.bind(x)

def real(x: Array) -> Array:
  r"""Elementwise extract real part: :math:`\mathrm{Re}(x)`.

  Returns the real part of a complex number.
  """
  return real_p.bind(x)

def imag(x: Array) -> Array:
  r"""Elementwise extract imaginary part: :math:`\mathrm{Im}(x)`.

  Returns the imaginary part of a complex number.
  """
  return imag_p.bind(x)

def complex(x: Array, y: Array) -> Array:
  r"""Elementwise make complex number: :math:`x + jy`.

  Builds a complex number from real and imaginary parts.
  """
  return complex_p.bind(x, y)

def conj(x: Array) -> Array:
  r"""Elementwise complex conjugate function: :math:`\overline{x}`."""
  return conj_p.bind(x, input_dtype=_dtype(x))

def abs(x: Array) -> Array:
  r"""Elementwise absolute value: :math:`|x|`."""
  return abs_p.bind(x)

def pow(x: Array, y: Array) -> Array:
  r"""Elementwise power: :math:`x^y`."""
  return pow_p.bind(x, y)

def integer_pow(x: Array, y: int) -> Array:
  r"""Elementwise power: :math:`x^y`, where :math:`y` is a fixed integer."""
  return integer_pow_p.bind(x, y=y)

def sqrt(x: Array) -> Array:
  r"""Elementwise square root: :math:`\sqrt{x}`."""
  return sqrt_p.bind(x)

def rsqrt(x: Array) -> Array:
  r"""Elementwise reciprocal square root:  :math:`1 \over \sqrt{x}`."""
  return rsqrt_p.bind(x)

def bitwise_not(x: Array) -> Array:
  r"""Elementwise NOT: :math:`\neg x`."""
  return not_p.bind(x)

def bitwise_and(x: Array, y: Array) -> Array:
  r"""Elementwise AND: :math:`x \wedge y`."""
  return and_p.bind(x, y)

def bitwise_or(x: Array, y: Array) -> Array:
  r"""Elementwise OR: :math:`x \vee y`."""
  return or_p.bind(x, y)

def bitwise_xor(x: Array, y: Array) -> Array:
  r"""Elementwise exclusive OR: :math:`x \oplus y`."""
  return xor_p.bind(x, y)

def population_count(x: Array) -> Array:
  r"""Elementwise popcount, count the number of set bits in each element."""
  return population_count_p.bind(x)

def clz(x: Array) -> Array:
  r"""Elementwise count-leading-zeros."""
  return clz_p.bind(x)

def add(x: Array, y: Array) -> Array:
  r"""Elementwise addition: :math:`x + y`."""
  return add_p.bind(x, y)

def sub(x: Array, y: Array) -> Array:
  r"""Elementwise subtraction: :math:`x - y`."""
  return sub_p.bind(x, y)

def mul(x: Array, y: Array) -> Array:
  r"""Elementwise multiplication: :math:`x \times y`."""
  return mul_p.bind(x, y)

def div(x: Array, y: Array) -> Array:
  r"""Elementwise division: :math:`x \over y`."""
  return div_p.bind(x, y)

def rem(x: Array, y: Array) -> Array:
  r"""Elementwise remainder: :math:`x \bmod y`."""
  return rem_p.bind(x, y)

def max(x: Array, y: Array) -> Array:
  r"""Elementwise maximum: :math:`\mathrm{max}(x, y)`

  For complex numbers, uses a lexicographic comparison on the
  `(real, imaginary)` pairs."""
  return max_p.bind(x, y)

def min(x: Array, y: Array) -> Array:
  r"""Elementwise minimum:  :math:`\mathrm{min}(x, y)`

  For complex numbers, uses a lexicographic comparison on the
  `(real, imaginary)` pairs."""
  return min_p.bind(x, y)

def shift_left(x: Array, y: Array) -> Array:
  r"""Elementwise left shift: :math:`x \ll y`."""
  return shift_left_p.bind(x, y)

def shift_right_arithmetic(x: Array, y: Array) -> Array:
  r"""Elementwise arithmetic right shift: :math:`x \gg y`."""
  return shift_right_arithmetic_p.bind(x, y)

def shift_right_logical(x: Array, y: Array) -> Array:
  r"""Elementwise logical right shift: :math:`x \gg y`."""
  return shift_right_logical_p.bind(x, y)

def eq(x: Array, y: Array) -> Array:
  r"""Elementwise equals: :math:`x = y`."""
  return eq_p.bind(x, y)

def ne(x: Array, y: Array) -> Array:
  r"""Elementwise not-equals: :math:`x \neq y`."""
  return ne_p.bind(x, y)

def ge(x: Array, y: Array) -> Array:
  r"""Elementwise greater-than-or-equals: :math:`x \geq y`."""
  return ge_p.bind(x, y)

def gt(x: Array, y: Array) -> Array:
  r"""Elementwise greater-than: :math:`x > y`."""
  return gt_p.bind(x, y)

def le(x: Array, y: Array) -> Array:
  r"""Elementwise less-than-or-equals: :math:`x \leq y`."""
  return le_p.bind(x, y)

def lt(x: Array, y: Array) -> Array:
  r"""Elementwise less-than: :math:`x < y`."""
  return lt_p.bind(x, y)

def convert_element_type(operand: Array, new_dtype: DType) -> Array:
  """Elementwise cast.

  Wraps XLA's `ConvertElementType
  <https://www.tensorflow.org/xla/operation_semantics#convertelementtype>`_
  operator, which performs an elementwise conversion from one type to another.
  Similar to a C++ `static_cast`.

  Args:
    operand: an array or scalar value to be cast
    new_dtype: a NumPy dtype representing the target type.

  Returns:
    An array with the same shape as `operand`, cast elementwise to `new_dtype`.
  """
  if hasattr(operand, '__jax_array__'):
    operand = operand.__jax_array__()
  return _convert_element_type(operand, new_dtype, weak_type=False)

def _convert_element_type(operand: Array, new_dtype: Optional[DType] = None,
                          weak_type: bool = False):
  # Don't canonicalize old_dtype because x64 context might cause
  # un-canonicalized operands to be passed in.
  old_dtype = np.result_type(operand)
  old_weak_type = dtypes.is_weakly_typed(operand)

  new_dtype = dtypes.canonicalize_dtype(new_dtype or old_dtype)
  new_weak_type = bool(weak_type)

  if (dtypes.issubdtype(old_dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    msg = "Casting complex values to real discards the imaginary part"
    warnings.warn(msg, np.ComplexWarning, stacklevel=2)

  # Python has big integers, but convert_element_type(2 ** 100, np.float32) need
  # not be an error since the target dtype fits the value. Handle this case by
  # converting to a NumPy array before calling bind. Without this step, we'd
  # first canonicalize the input to a value of dtype int32 or int64, leading to
  # an overflow error.
  if type(operand) is int:
    operand = np.asarray(operand, new_dtype)

  if ((old_dtype, old_weak_type) == (new_dtype, new_weak_type)
      and isinstance(operand, (core.Tracer, xla.DeviceArray))):
    return operand
  else:
    return convert_element_type_p.bind(operand, new_dtype=new_dtype,
                                       weak_type=new_weak_type)

def bitcast_convert_type(operand: Array, new_dtype: DType) -> Array:
  """Elementwise bitcast.

  Wraps XLA's `BitcastConvertType
  <https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype>`_
  operator, which performs a bit cast from one type to another. The bitwidth
  of the source and destination types must match.

  Args:
    operand: an array or scalar value to be cast
    new_dtype: the new type. Should be a NumPy type.

  Returns:
    An array with the same shape as `operand`, bitcast elementwise to
    `new_dtype`.
  """
  new_dtype = dtypes.canonicalize_dtype(new_dtype)
  return bitcast_convert_type_p.bind(operand, new_dtype=new_dtype)

def clamp(min: Array, x: Array, max: Array) -> Array:
  r"""Elementwise clamp.

  Returns :math:`\mathrm{clamp}(x) = \begin{cases}
  \mathit{min} & \text{if } x < \mathit{min},\\
  \mathit{max} & \text{if } x > \mathit{max},\\
  x & \text{otherwise}
  \end{cases}`.
  """
  return clamp_p.bind(min, x, max)

def concatenate(operands: Sequence[Array], dimension: int) -> Array:
  """Concatenates a sequence of arrays along `dimension`.

  Wraps XLA's `Concatenate
  <https://www.tensorflow.org/xla/operation_semantics#concatenate>`_
  operator.

  Args:
    operands: a sequence of arrays to concatenate. The arrays must have equal
      shapes, except in the `dimension` axis.
    dimension: the dimension along which to concatenate the arrays.

  Returns:
    An array containing the concatenation.
  """
  return concatenate_p.bind(*operands, dimension=dimension)

Precision = xla_client.PrecisionConfig.Precision
Precision.__str__ = lambda precision: precision.name
PrecisionType = Any
PrecisionLike = Union[None, str, PrecisionType, Tuple[str, str],
                      Tuple[PrecisionType, PrecisionType]]
_precision_strings = {
    'highest':       Precision.HIGHEST,
    'float32':       Precision.HIGHEST,
    'bfloat16_3x':   Precision.HIGH,
    'tensorfloat32': Precision.HIGH,
    'bfloat16':      Precision.DEFAULT,
    'fastest':       Precision.DEFAULT,
    None:            Precision.DEFAULT,
}

class ConvDimensionNumbers(NamedTuple):
  """Describes batch, spatial, and feature dimensions of a convolution.

  Args:
    lhs_spec: a tuple of nonnegative integer dimension numbers containing
      `(batch dimension, feature dimension, spatial dimensions...)`.
    rhs_spec: a tuple of nonnegative integer dimension numbers containing
      `(out feature dimension, in feature dimension, spatial dimensions...)`.
    out_spec: a tuple of nonnegative integer dimension numbers containing
      `(batch dimension, feature dimension, spatial dimensions...)`.
  """
  lhs_spec: Sequence[int]
  rhs_spec: Sequence[int]
  out_spec: Sequence[int]

ConvGeneralDilatedDimensionNumbers = Union[
  None, ConvDimensionNumbers, Tuple[str, str, str]]

def conv_general_dilated(
  lhs: Array, rhs: Array, window_strides: Sequence[int],
  padding: Union[str, Sequence[Tuple[int, int]]],
  lhs_dilation: Optional[Sequence[int]] = None,
  rhs_dilation: Optional[Sequence[int]] = None,
  dimension_numbers: ConvGeneralDilatedDimensionNumbers  = None,
  feature_group_count: int = 1, batch_group_count: int = 1,
  precision: PrecisionLike = None,
  preferred_element_type: Optional[DType] = None) -> Array:
  """General n-dimensional convolution operator, with optional dilation.

  Wraps XLA's `Conv
  <https://www.tensorflow.org/xla/operation_semantics#conv_convolution>`_
  operator.

  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    lhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `lhs`. LHS dilation
      is also known as transposed convolution.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. RHS dilation
      is also known as atrous convolution.
    dimension_numbers: either `None`, a ``ConvDimensionNumbers`` object, or
      a 3-tuple ``(lhs_spec, rhs_spec, out_spec)``, where each element is a
      string of length `n+2`.
    feature_group_count: integer, default 1. See XLA HLO docs.
    batch_group_count: integer, default 1. See XLA HLO docs.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``), a string (e.g. 'highest' or
      'fastest', see the ``jax.default_matmul_precision`` context manager), or a
      tuple of two ``lax.Precision`` enums or strings indicating precision of
      ``lhs`` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    An array containing the convolution result.

  In the string case of ``dimension_numbers``, each character identifies by
  position:

  - the batch dimensions in ``lhs``, ``rhs``, and the output with the character
    'N',
  - the feature dimensions in `lhs` and the output with the character 'C',
  - the input and output feature dimensions in rhs with the characters 'I'
    and 'O' respectively, and
  - spatial dimension correspondences between lhs, rhs, and the output using
    any distinct characters.

  For example, to indicate dimension numbers consistent with the ``conv``
  function with two spatial dimensions, one could use ``('NCHW', 'OIHW',
  'NCHW')``. As another example, to indicate dimension numbers consistent with
  the TensorFlow Conv2D operation, one could use ``('NHWC', 'HWIO', 'NHWC')``.
  When using the latter form of convolution dimension specification, window
  strides are associated with spatial dimension character labels according to
  the order in which the labels appear in the ``rhs_spec`` string, so that
  ``window_strides[0]`` is matched with the dimension corresponding to the first
  character appearing in rhs_spec that is not ``'I'`` or ``'O'``.

  If ``dimension_numbers`` is ``None``, the default is ``('NCHW', 'OIHW',
  'NCHW')`` (for a 2D convolution).
  """
  dnums = conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
  if lhs_dilation is None:
    lhs_dilation = (1,) * (lhs.ndim - 2)
  elif isinstance(padding, str) and not len(lhs_dilation) == lhs_dilation.count(1):
    raise ValueError(
        "String padding is not implemented for transposed convolution "
        "using this op. Please either exactly specify the required padding or "
        "use conv_transpose.")
  if rhs_dilation is None:
    rhs_dilation = (1,) * (rhs.ndim - 2)
  if isinstance(padding, str):
    lhs_perm, rhs_perm, _ = dnums
    rhs_shape = np.take(rhs.shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [(k-1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)]
    padding = padtype_to_pads(
        np.take(lhs.shape, lhs_perm)[2:], effective_rhs_shape,  # type: ignore[index]
        window_strides, padding)
  return conv_general_dilated_p.bind(
      lhs, rhs, window_strides=tuple(window_strides), padding=tuple(padding),
      lhs_dilation=tuple(lhs_dilation), rhs_dilation=tuple(rhs_dilation),
      dimension_numbers=dnums,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
      lhs_shape=lhs.shape, rhs_shape=rhs.shape,
      precision=_canonicalize_precision(precision),
      preferred_element_type=preferred_element_type)

def dot(lhs: Array, rhs: Array, precision: PrecisionLike = None,
        preferred_element_type: Optional[DType] = None) -> Array:
  """Vector/vector, matrix/vector, and matrix/matrix multiplication.

  Wraps XLA's `Dot
  <https://www.tensorflow.org/xla/operation_semantics#dot>`_
  operator.

  For more general contraction, see the `dot_general` operator.

  Args:
    lhs: an array of rank 1 or 2.
    rhs: an array of rank 1 or 2.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    An array containing the product.
  """
  if 1 <= lhs.ndim <= 2 and 1 <= rhs.ndim <= 2 and lhs.shape[-1] == rhs.shape[0]:
    return dot_general(lhs, rhs, (((lhs.ndim - 1,), (0,)), ((), ())),
                       precision=precision, preferred_element_type=preferred_element_type)
  else:
    raise TypeError("Incompatible shapes for dot: got {} and {}.".format(
        lhs.shape, rhs.shape))


DotDimensionNumbers = Tuple[Tuple[Sequence[int], Sequence[int]],
                            Tuple[Sequence[int], Sequence[int]]]

def dot_general(lhs: Array, rhs: Array, dimension_numbers: DotDimensionNumbers,
                precision: PrecisionLike = None,
                preferred_element_type: Optional[DType] = None) -> Array:
  """More general contraction operator.

  Wraps XLA's `DotGeneral
  <https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`_
  operator.

  Args:
    lhs: an array
    rhs: an array
    dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
      (lhs_batch_dims, rhs_batch_dims))`
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    An array containing the result.
  """
  contract_dims_seq, batch_dims_seq = dimension_numbers
  contract_dims = tuple(map(tuple, contract_dims_seq))  # type: ignore
  batch_dims = tuple(map(tuple, batch_dims_seq))  # type: ignore
  return dot_general_p.bind(lhs, rhs,
                            dimension_numbers=(contract_dims, batch_dims),
                            precision=_canonicalize_precision(precision),
                            preferred_element_type=preferred_element_type)

def broadcast(operand: Array, sizes: Sequence[int]) -> Array:
  """Broadcasts an array, adding new major dimensions.

  Wraps XLA's `Broadcast
  <https://www.tensorflow.org/xla/operation_semantics#broadcast>`_
  operator.

  Args:
    operand: an array
    sizes: a sequence of integers, giving the sizes of new major dimensions
      to add.

  Returns:
    An array containing the result.
  """
  dims = tuple(range(len(sizes), len(sizes) + np.ndim(operand)))
  return broadcast_in_dim(operand, tuple(sizes) + np.shape(operand), dims)

def broadcast_in_dim(operand: Array, shape: Shape,
                     broadcast_dimensions: Sequence[int]) -> Array:
  """Wraps XLA's `BroadcastInDim
  <https://www.tensorflow.org/xla/operation_semantics#broadcastindim>`_
  operator.
  """
  shape = _broadcast_in_dim_shape_rule(
    operand, shape=shape, broadcast_dimensions=broadcast_dimensions)
  if (np.ndim(operand) == len(shape) and not len(broadcast_dimensions)
      and isinstance(operand, (xla.DeviceArray, core.Tracer))):
    return operand
  return broadcast_in_dim_p.bind(
      operand, shape=tuple(shape),
      broadcast_dimensions=tuple(broadcast_dimensions))

def broadcast_to_rank(x: Array, rank: int) -> Array:
  """Adds leading dimensions of ``1`` to give ``x`` rank ``rank``."""
  return broadcast(x, (1,) * (rank - x.ndim))

def reshape(operand: Array, new_sizes: Shape,
            dimensions: Optional[Sequence[int]] = None) -> Array:
  """Wraps XLA's `Reshape
  <https://www.tensorflow.org/xla/operation_semantics#reshape>`_
  operator.

  For inserting/removing dimensions of size 1, prefer using ``lax.squeeze`` /
  ``lax.expand_dims``. These preserve information about axis identity that may
  be useful for advanced transformation rules.

  Args:
    operand: array to be reshaped.
    new_sizes: sequence of integers specifying the resulting shape. The size
      of the final array must match the size of the input.
    dimensions: optional sequence of integers specifying the permutation order of
      the input shape. If specified, the length must match ``operand.shape``.

  Returns:
    out: reshaped array.

  Examples:
    Simple reshaping from one to two dimensions:

    >>> x = jnp.arange(6)
    >>> y = reshape(x, (2, 3))
    >>> y
    DeviceArray([[0, 1, 2],
                 [3, 4, 5]], dtype=int32)

    Reshaping back to one dimension:

    >>> reshape(y, (6,))
    DeviceArray([0, 1, 2, 3, 4, 5], dtype=int32)

    Reshaping to one dimension with permutation of dimensions:

    >>> reshape(y, (6,), (1, 0))
    DeviceArray([0, 3, 1, 4, 2, 5], dtype=int32)
  """
  new_sizes = canonicalize_shape(new_sizes)  # TODO
  new_sizes = tuple(new_sizes)
  same_shape = np.shape(operand) == new_sizes
  same_dims = dimensions is None or tuple(dimensions) == tuple(range(np.ndim(operand)))
  if np.shape(operand) and same_shape and same_dims:
    return operand
  else:
    return reshape_p.bind(
      operand, new_sizes=new_sizes,
      dimensions=None if dimensions is None or same_dims else tuple(dimensions))

def pad(operand: Array, padding_value: Array,
        padding_config: Sequence[Tuple[int, int, int]]) -> Array:
  """Applies low, high, and/or interior padding to an array.

  Wraps XLA's `Pad
  <https://www.tensorflow.org/xla/operation_semantics#pad>`_
  operator.

  Args:
    operand: an array to be padded.
    padding_value: the value to be inserted as padding. Must have the same dtype
      as ``operand``.
    padding_config: a sequence of ``(low, high, interior)`` tuples of integers,
      giving the amount of low, high, and interior (dilation) padding to insert
      in each dimension.

  Returns:
    The ``operand`` array with padding value ``padding_value`` inserted in each
    dimension according to the ``padding_config``.
  """
  return pad_p.bind(operand, padding_value, padding_config=tuple(padding_config))

def rev(operand: Array, dimensions: Sequence[int]) -> Array:
  """Wraps XLA's `Rev
  <https://www.tensorflow.org/xla/operation_semantics#rev_reverse>`_
  operator.
  """
  return rev_p.bind(operand, dimensions=tuple(dimensions))

def select(pred: Array, on_true: Array, on_false: Array) -> Array:
  """Wraps XLA's `Select
  <https://www.tensorflow.org/xla/operation_semantics#select>`_
  operator.
  """
  return select_p.bind(pred, on_true, on_false)

def slice(operand: Array, start_indices: Sequence[int],
          limit_indices: Sequence[int],
          strides: Optional[Sequence[int]] = None) -> Array:
  """Wraps XLA's `Slice
  <https://www.tensorflow.org/xla/operation_semantics#slice>`_
  operator.
  """
  return slice_p.bind(operand, start_indices=tuple(start_indices),
                      limit_indices=tuple(limit_indices),
                      strides=None if strides is None else tuple(strides))

def dynamic_slice(operand: Array, start_indices: Sequence[Array],
                  slice_sizes: Shape) -> Array:
  """Wraps XLA's `DynamicSlice
  <https://www.tensorflow.org/xla/operation_semantics#dynamicslice>`_
  operator.

  Args:
    operand: an array to slice.
    start_indices: a list of scalar indices, one per dimension. These values
      may be dynamic.
    slice_sizes: the size of the slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`. Inside a JIT compiled
      function, only static values are supported (all JAX arrays inside JIT
      must have statically known size).

  Returns:
    An array containing the slice.

  Examples:
    Here is a simple two-dimensional dynamic slice:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    DeviceArray([[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_slice(x, (1, 1), (2, 3))
    DeviceArray([[ 5,  6,  7],
                 [ 9, 10, 11]], dtype=int32)

    Note the potentially surprising behavior for the case where the requested slice
    overruns the bounds of the array; in this case the start index is adjusted to
    return a slice of the requested size:

    >>> dynamic_slice(x, (1, 1), (2, 4))
    DeviceArray([[ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)
  """
  start_indices = _dynamic_slice_indices(operand, start_indices)
  return dynamic_slice_p.bind(operand, *start_indices,
                              slice_sizes=tuple(slice_sizes))

def dynamic_update_slice(operand: Array, update: Array,
                         start_indices: Array) -> Array:
  """Wraps XLA's `DynamicUpdateSlice
  <https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice>`_
  operator.

  Args:
    operand: an array to slice.
    update: an array containing the new values to write onto `operand`.
    start_indices: a list of scalar indices, one per dimension.

  Returns:
    An array containing the slice.

  Examples:
    Here is an example of updating a one-dimensional slice update:

    >>> x = jnp.zeros(6)
    >>> y = jnp.ones(3)
    >>> dynamic_update_slice(x, y, (2,))
    DeviceArray([0., 0., 1., 1., 1., 0.], dtype=float32)

    If the update slice is too large to fit in the array, the start
    index will be adjusted to make it fit

    >>> dynamic_update_slice(x, y, (3,))
    DeviceArray([0., 0., 0., 1., 1., 1.], dtype=float32)
    >>> dynamic_update_slice(x, y, (5,))
    DeviceArray([0., 0., 0., 1., 1., 1.], dtype=float32)

    Here is an example of a two-dimensional slice update:

    >>> x = jnp.zeros((4, 4))
    >>> y = jnp.ones((2, 2))
    >>> dynamic_update_slice(x, y, (1, 2))
    DeviceArray([[0., 0., 0., 0.],
                 [0., 0., 1., 1.],
                 [0., 0., 1., 1.],
                 [0., 0., 0., 0.]], dtype=float32)
  """
  start_indices = _dynamic_slice_indices(operand, start_indices)
  return dynamic_update_slice_p.bind(operand, update, *start_indices)


class GatherDimensionNumbers(NamedTuple):
  """
  Describes the dimension number arguments to an `XLA's Gather operator
  <https://www.tensorflow.org/xla/operation_semantics#gather>`_. See the XLA
  documentation for more details of what the dimension numbers mean.

  Args:
    offset_dims: the set of dimensions in the `gather` output that offset into
      an array sliced from `operand`. Must be a tuple of integers in ascending
      order, each representing a dimension number of the output.
    collapsed_slice_dims: the set of dimensions `i` in `operand` that have
      `slice_sizes[i] == 1` and that should not have a corresponding dimension
      in the output of the gather. Must be a tuple of integers in ascending
      order.
    start_index_map: for each dimension in `start_indices`, gives the
      corresponding dimension in `operand` that is to be sliced. Must be a
      tuple of integers with size equal to `start_indices.shape[-1]`.

  Unlike XLA's `GatherDimensionNumbers` structure, `index_vector_dim` is
  implicit; there is always an index vector dimension and it must always be the
  last dimension. To gather scalar indices, add a trailing dimension of size 1.
  """
  offset_dims: Sequence[int]
  collapsed_slice_dims: Sequence[int]
  start_index_map: Sequence[int]


def gather(operand: Array, start_indices: Array,
           dimension_numbers: GatherDimensionNumbers,
           slice_sizes: Shape) -> Array:
  """Gather operator.

  Wraps `XLA's Gather operator
  <https://www.tensorflow.org/xla/operation_semantics#gather>`_.

  The semantics of gather are complicated, and its API might change in the
  future. For most use cases, you should prefer `Numpy-style indexing
  <https://docs.scipy.org/doc/numpy-1.16.0/reference/arrays.indexing.html>`_
  (e.g., `x[:, (1,4,7), ...]`), rather than using `gather` directly.

  Args:
    operand: an array from which slices should be taken
    start_indices: the indices at which slices should be taken
    dimension_numbers: a `lax.GatherDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices` and the output relate.
    slice_sizes: the size of each slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`.

  Returns:
    An array containing the gather output.
  """
  return gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=canonicalize_shape(slice_sizes))


class ScatterDimensionNumbers(NamedTuple):
  """
  Describes the dimension number arguments to an `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_. See the XLA
  documentation for more details of what the dimension numbers mean.

  Args:
    update_window_dims: the set of dimensions in the `updates` that are window
      dimensions. Must be a tuple of integers in ascending
      order, each representing a dimension number.
    inserted_window_dims: the set of size 1 window dimensions that must be inserted
      into the shape of `updates`. Must be a tuple of integers in ascending
      order, each representing a dimension number of the output. These are the
      mirror image of `collapsed_slice_dims` in the case of `gather`.
    scatter_dims_to_operand_dims: for each dimension in `scatter_indices`, gives
      the corresponding dimension in `operand`. Must be a sequence of integers
      with size equal to indices.shape[-1].

  Unlike XLA's `ScatterDimensionNumbers` structure, `index_vector_dim` is
  implicit; there is always an index vector dimension and it must always be the
  last dimension. To scatter scalar indices, add a trailing dimension of size 1.
  """
  update_window_dims: Sequence[int]
  inserted_window_dims: Sequence[int]
  scatter_dims_to_operand_dims: Sequence[int]

def scatter_add(operand: Array, scatter_indices: Array, updates: Array,
                dimension_numbers: ScatterDimensionNumbers, *,
                indices_are_sorted: bool = False,
                unique_indices: bool = False) -> Array:
  """Scatter-add operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  addition is used to combine updates and values from `operand`.

  The semantics of scatter are complicated and its API is subject to change.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = _reduction_jaxpr(add, _abstractify(_const(operand, 0)))
  return scatter_add_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)

def scatter_mul(operand: Array, scatter_indices: Array, updates: Array,
                dimension_numbers: ScatterDimensionNumbers, *,
                indices_are_sorted: bool = False,
                unique_indices: bool = False) -> Array:
  """Scatter-multiply operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  multiplication is used to combine updates and values from `operand`.

  The semantics of scatter are complicated and its API is subject to change.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = _reduction_jaxpr(mul, _abstractify(_const(operand, 1)))
  return scatter_mul_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)

def scatter_min(operand: Array, scatter_indices: Array, updates: Array,
                dimension_numbers: ScatterDimensionNumbers, *,
                indices_are_sorted: bool = False,
                unique_indices: bool = False) -> Array:
  """Scatter-min operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  the `min` function is used to combine updates and values from `operand`.

  The semantics of scatter are complicated and its API is subject to change.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = _reduction_jaxpr(min, _abstractify(_const(operand, 0)))
  return scatter_min_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)

def scatter_max(operand: Array, scatter_indices: Array, updates: Array,
                dimension_numbers: ScatterDimensionNumbers, *,
                indices_are_sorted: bool = False,
                unique_indices: bool = False) -> Array:
  """Scatter-max operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  the `max` function is used to combine updates and values from `operand`.

  The semantics of scatter are complicated and its API is subject to change.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = _reduction_jaxpr(max, _abstractify(_const(operand, 0)))
  return scatter_max_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)

# Define this outside of scatter to ensure cache hits.
_scatter_reduction_computation = lambda x, y: y

def scatter(operand: Array, scatter_indices: Array, updates: Array,
            dimension_numbers: ScatterDimensionNumbers, *,
            indices_are_sorted: bool = False,
            unique_indices: bool = False) -> Array:
  """Scatter-update operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where updates
  replace values from `operand`.

  If multiple updates are performed to the same index of operand, they may be
  applied in any order.

  The semantics of scatter are complicated and its API is subject to change.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = _reduction_jaxpr(_scatter_reduction_computation,
                                   _abstractify(_const(operand, 0)))
  return scatter_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)

def index_take(src: Array, idxs: Array, axes: Sequence[int]) -> Array:
  indices = concatenate([expand_dims(i, (1,)) for i in idxs], 1)
  indices = indices % np.array([src.shape[ax] for ax in axes])
  slice_sizes = list(src.shape)
  for ax in axes:
    slice_sizes[ax] = 1
  offset_dims = tuple(range(1, src.ndim - indices.shape[1] + 1))
  dnums = GatherDimensionNumbers(
      offset_dims=offset_dims,
      collapsed_slice_dims=axes,
      start_index_map=axes)
  return gather(src, indices, dimension_numbers=dnums,
                slice_sizes=tuple(slice_sizes))

def transpose(operand: Array, permutation: Sequence[int]) -> Array:
  """Wraps XLA's `Transpose
  <https://www.tensorflow.org/xla/operation_semantics#transpose>`_
  operator.
  """
  permutation = tuple(permutation)
  if permutation == tuple(range(len(permutation))):
    return operand
  else:
    return transpose_p.bind(operand, permutation=permutation)

def argmin(operand: Array, axis: int,
           index_dtype: DType) -> Tuple[Array, Array]:
  """Computes the index of the minimum element along ``axis``."""
  return argmin_p.bind(operand, axes=(axis,),
                       index_dtype=dtypes.canonicalize_dtype(index_dtype))

def argmax(operand: Array, axis: int,
           index_dtype: DType) -> Tuple[Array, Array]:
  """Computes the index of the maximum element along ``axis``."""
  return argmax_p.bind(operand, axes=(axis,),
                       index_dtype=dtypes.canonicalize_dtype(index_dtype))

def reduce(operands: Array, init_values: Array, computation: Callable,
           dimensions: Sequence[int]) -> Array:
  """Wraps XLA's `Reduce
  <https://www.tensorflow.org/xla/operation_semantics#reduce>`_
  operator.

  ``init_values`` and ``computation`` together must form a `monoid
  <https://en.wikipedia.org/wiki/Monoid>`_
  for correctness. That is ``init_values`` must be an identity of
  ``computation``, and ``computation`` must be associative. XLA may exploit both
  of these properties during code generation; if either is violated the result
  is undefined.
  """
  flat_operands, operand_tree = tree_util.tree_flatten(operands)
  flat_init_values, init_value_tree = tree_util.tree_flatten(init_values)
  if operand_tree != init_value_tree:
    raise ValueError('Operands must have the same tree structure as init_values:'
                     f' {operand_tree} vs. {init_value_tree}')
  if len(flat_operands) != len(flat_init_values):
    raise ValueError('Must have same total number of operands as init_values: '
                     f' {len(flat_operands)} vs. {len(flat_init_values)}')
  monoid_reducer = _get_monoid_reducer(computation, flat_init_values)
  if monoid_reducer:
    # monoid reducers bypass the weak_type_rule, so we set it explicitly.
    weak_type = dtypes.is_weakly_typed(*flat_operands) and dtypes.is_weakly_typed(*flat_init_values)
    return _convert_element_type(monoid_reducer(*flat_operands, dimensions),
                                 weak_type=weak_type)
  else:
    flat_init_avals = safe_map(_abstractify, flat_init_values)
    jaxpr, consts, out_tree = _variadic_reduction_jaxpr(
        computation, tuple(flat_init_avals), init_value_tree)
    out = reduce_p.bind(*(flat_operands + flat_init_values), computation=computation,
                         jaxpr=jaxpr, consts=consts, dimensions=tuple(dimensions))
    return tree_util.tree_unflatten(out_tree, out)

@cache()
def _reduction_jaxpr(computation, aval):
  pval = pe.PartialVal.unknown(aval)
  @lu.wrap_init
  def comp(x, y):
    result = computation(x, y)
    if not (isinstance(result, core.Tracer) or core.valid_jaxtype(result)):
      raise ValueError(
          f"Invalid return type from reduction function: {type(result)}\n"
          f"Reduction functions should only return an array.\n"
          f"Full return value: {result}")
    return (result,)
  jaxpr, _, consts = pe.trace_to_jaxpr(comp, (pval, pval), instantiate=False)
  return jaxpr, consts

@cache()
def _variadic_reduction_jaxpr(computation, flat_avals, aval_tree):
  avals = tree_util.tree_unflatten(aval_tree, flat_avals)
  flat_in_avals, in_tree = tree_util.tree_flatten((avals, avals))
  pvals = safe_map(pe.PartialVal.unknown, flat_in_avals)
  comp = lu.wrap_init(computation)
  flat_comp, out_tree = api_util.flatten_fun_nokwargs(comp, in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr(flat_comp, tuple(pvals),
                                       instantiate=False)
  return jaxpr, consts, out_tree()

def _get_monoid_reducer(monoid_op: Callable, xs: Array) -> Optional[Callable]:
  if len(xs) != 1:
    return None
  x, = xs
  aval = core.get_aval(x)
  dtype = _dtype(x)
  if (type(aval) is ConcreteArray) and aval.shape == ():
    if monoid_op is add:
      return np.equal(aval.val, 0) and partial(_reduce_sum)
    elif monoid_op is mul:
      return np.equal(aval.val, 1) and _reduce_prod
    elif monoid_op is bitwise_or and dtype == np.bool_:
      return np.equal(aval.val, _get_max_identity(dtype)) and _reduce_or
    elif monoid_op is bitwise_and and dtype == np.bool_:
      return np.equal(aval.val, _get_min_identity(dtype)) and _reduce_and
    elif monoid_op is max:
      return np.equal(aval.val, _get_max_identity(dtype)) and _reduce_max
    elif monoid_op is min:
      return np.equal(aval.val, _get_min_identity(dtype)) and _reduce_min
  return None

def _get_max_identity(dtype: DType) -> Array:
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(-np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).min, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(False, np.bool_)

def _get_min_identity(dtype: DType) -> Array:
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).max, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(True, np.bool_)

def _reduce_sum(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_sum_p.bind(operand, axes=tuple(axes))

def _reduce_prod(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_prod_p.bind(operand, axes=tuple(axes))

def _reduce_max(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_max_p.bind(operand, axes=tuple(axes))

def _reduce_min(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_min_p.bind(operand, axes=tuple(axes))

def _reduce_or(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_or_p.bind(operand, axes=tuple(axes))

def _reduce_and(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_and_p.bind(operand, axes=tuple(axes))

def reduce_window(operand: Array, init_value: Array, computation: Callable,
                  window_dimensions: Shape, window_strides: Sequence[int],
                  padding: Union[str, Sequence[Tuple[int, int]]],
                  base_dilation: Optional[Sequence[int]] = None,
                  window_dilation: Optional[Sequence[int]] = None) -> Array:
  """Wraps XLA's `ReduceWindowWithGeneralPadding
  <https://www.tensorflow.org/xla/operation_semantics#reducewindow>`_
  operator.
  """
  if isinstance(padding, str):
    dilated_window_dims = (window_dimensions if window_dilation is None else
                           _dilate_shape(window_dimensions, window_dilation))
    padding = tuple(padtype_to_pads(operand.shape, dilated_window_dims,
                                    window_strides, padding))
  else:
    padding = tuple(padding)
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  monoid_reducer = _get_monoid_window_reducer(computation, init_value)
  if monoid_reducer:
    return monoid_reducer(operand, window_dimensions, window_strides, padding,
                          base_dilation, window_dilation)
  else:
    jaxpr, consts = _reduction_jaxpr(computation, _abstractify(init_value))
    return reduce_window_p.bind(
        operand, init_value, jaxpr=jaxpr, consts=consts,
        window_dimensions=tuple(window_dimensions),
        window_strides=tuple(window_strides), padding=padding,
        base_dilation=tuple(base_dilation),
        window_dilation=tuple(window_dilation))

def _get_monoid_window_reducer(monoid_op: Callable, x: Array) -> Optional[Callable]:
  aval = core.get_aval(x)
  if (type(aval) is ConcreteArray) and aval.shape == ():
    if monoid_op is add:
      return aval.val == 0 and _reduce_window_sum
    elif monoid_op is max:
      return aval.val == _get_max_identity(aval.dtype) and _reduce_window_max
    elif monoid_op is min:
      return aval.val == _get_min_identity(aval.dtype) and _reduce_window_min
  return None

def _reduce_window_sum(operand: Array, window_dimensions: Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[Tuple[int, int]],
                       base_dilation: Optional[Sequence[int]] = None,
                       window_dilation: Optional[Sequence[int]] = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_sum_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_prod(operand: Array, window_dimensions: Shape,
                        window_strides: Sequence[int],
                        padding: Sequence[Tuple[int, int]],
                        base_dilation: Optional[Sequence[int]] = None,
                        window_dilation: Optional[Sequence[int]] = None) -> Array:
  init_value = _const(operand, 1)
  jaxpr, consts = _reduction_jaxpr(mul, _abstractify(init_value))
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_p.bind(
      operand, init_value, jaxpr=jaxpr, consts=consts,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_max(operand: Array, window_dimensions: Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[Tuple[int, int]],
                       base_dilation: Optional[Sequence[int]] = None,
                       window_dilation: Optional[Sequence[int]] = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_max_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_min(operand: Array, window_dimensions: Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[Tuple[int, int]],
                       base_dilation: Optional[Sequence[int]] = None,
                       window_dilation: Optional[Sequence[int]] = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_min_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _select_and_scatter(operand: Array, select: Callable,
                        window_dimensions: Shape, window_strides: Sequence[int],
                        padding: Sequence[Tuple[int, int]], source: Array,
                        init_value: Array, scatter: Callable,
                        base_dilation: Sequence[int],
                        window_dilation: Sequence[int]) -> Array:
  select_jaxpr, select_consts = _reduction_jaxpr(select, _abstractify(init_value))
  scatter_jaxpr, scatter_consts = _reduction_jaxpr(scatter, _abstractify(init_value))
  return select_and_scatter_p.bind(
      operand, source, init_value, select_jaxpr=select_jaxpr,
      select_consts=select_consts, scatter_jaxpr=scatter_jaxpr,
      scatter_consts=scatter_consts, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _select_and_scatter_add(source: Array, operand: Array,
                            select_prim: core.Primitive,
                            window_dimensions: Shape,
                            window_strides: Sequence[int],
                            padding: Sequence[Tuple[int, int]]) -> Array:
  return select_and_scatter_add_p.bind(
      source, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding))

def _select_and_gather_add(tangents: Array, operand: Array,
                           select_prim: core.Primitive,
                           window_dimensions: Shape,
                           window_strides: Sequence[int],
                           padding: Sequence[Tuple[int, int]],
                           base_dilation: Sequence[int],
                           window_dilation: Sequence[int]) -> Array:
  """Extracts the tangent corresponding to the minimum or maximum element in each
  window of the `operand` array.

  Wraps XLA's `ReduceWindow
  <https://www.tensorflow.org/xla/operation_semantics#reducewindow>`_
  operator, which applies a reduction function to all elements in each window of the
  input multi-dimensional array. In this case, the input multi-dimensional array is
  built by packing each element in the `operand` array with its corresponding
  element in the `tangents` array.

  Args:
    tangents: an array
    operand: an array with the same shape as `tangents`
    select_prim: a reduction function (restricted to `ge_p` and `le_p`)
    window_dimensions: an array of integers for window dimension values
    window_strides: an array of integers for window stride values
    base_dilation: an array of integers for base dilation values
    window_dilation: an array of integers for window dilation values

  Returns:
    An array containing the elements in `tangents` corresponding to the output of the
    reduction of `operand` fin each window.
  """
  return select_and_gather_add_p.bind(
      tangents, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def sort(operand: Union[Array, Sequence[Array]], dimension: int = -1,
         is_stable: bool = True, num_keys: int = 1) -> Union[Array, Tuple[Array, ...]]:
  """Wraps XLA's `Sort
  <https://www.tensorflow.org/xla/operation_semantics#sort>`_
  operator.

  Args:
    operand : Array or sequence of arrays
    dimension : integer dimension along which to sort. Default: -1.
    is_stable : boolean specifying whether to use a stable sort. Default: True.
    num_keys : number of operands to treat as sort keys. Default: 1.
      For num_keys > 1, the sort order will be determined lexicographically using
      the first `num_keys` arrays, with the first key being primary.
      The remaining operands will be returned with the same permutation.

  Returns:
    operand : sorted version of the input or inputs.
  """
  if isinstance(operand, Sequence):
    if len(operand) == 0:
      raise TypeError("Sort requires at least one operand")
    if not (1 <= num_keys <= len(operand)):
      raise ValueError(f"num_keys={num_keys} must be between 1 and len(operand)={len(operand)}")
    dimension = canonicalize_axis(dimension, len(operand[0].shape))
    return tuple(sort_p.bind(*operand, dimension=dimension,
                             is_stable=is_stable,
                             num_keys=num_keys))
  else:
    if num_keys != 1:
      raise ValueError(f"num_keys={num_keys} must equal 1 for a single operand.")
    dimension = canonicalize_axis(dimension, len(operand.shape))
    return sort_p.bind(operand, dimension=dimension, is_stable=is_stable, num_keys=1)[0]

def sort_key_val(keys: Array, values: Array, dimension: int = -1,
                 is_stable: bool = True) -> Tuple[Array, Array]:
  """Sorts ``keys`` along ``dimension`` and applies same permutation to ``values``."""
  dimension = canonicalize_axis(dimension, len(keys.shape))
  k, v = sort_p.bind(keys, values, dimension=dimension, is_stable=is_stable, num_keys=1)
  return k, v

def top_k(operand: Array, k: int) -> Tuple[Array, Array]:
  """Returns top ``k`` values and their indices along the last axis of ``operand``."""
  k = int(k)
  if k < 0:
    raise ValueError("k argument to top_k must be nonnegative, got {}".format(k))
  return top_k_p.bind(operand, k=k)

def tie_in(x: Array, y: Array) -> Array:
  """Deprecated. Ignores ``x`` and returns ``y``."""
  return y

def full(shape: Shape, fill_value: Array, dtype: Optional[DType] = None) -> Array:
  """Returns an array of `shape` filled with `fill_value`.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    fill_value: the value to fill the new array with.
    dtype: the type of the output array, or `None`. If not `None`, `fill_value`
      will be cast to `dtype`.
  """
  shape = canonicalize_shape(shape)
  if np.shape(fill_value):
    msg = "full must be called with scalar fill_value, got fill_value.shape {}."
    raise TypeError(msg.format(np.shape(fill_value)))
  weak_type = dtype is None and dtypes.is_weakly_typed(fill_value)
  dtype = dtypes.canonicalize_dtype(dtype or _dtype(fill_value))
  fill_value = _convert_element_type(fill_value, dtype, weak_type)
  return broadcast(fill_value, shape)

def _device_put_raw(x, weak_type=None):
  if isinstance(x, xla.DeviceArray):
    return x
  else:
    aval = raise_to_shaped(core.get_aval(x), weak_type=weak_type)
    return xla.array_result_handler(None, aval)(*xla.device_put(x))

def iota(dtype: DType, size: int) -> Array:
  """Wraps XLA's `Iota
  <https://www.tensorflow.org/xla/operation_semantics#iota>`_
  operator.
  """
  dtype = dtypes.canonicalize_dtype(dtype)
  size, = canonicalize_shape((size,))
  return iota_p.bind(dtype=dtype, shape=(size,), dimension=0)

def broadcasted_iota(dtype: DType, shape: Shape, dimension: int) -> Array:
  """Convenience wrapper around ``iota``."""
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = canonicalize_shape(shape)
  dimension = core.concrete_or_error(
      int, dimension, "dimension argument of lax.broadcasted_iota")
  return iota_p.bind(dtype=dtype, shape=shape, dimension=dimension)

def _eye(dtype: DType, shape: Shape, offset: int) -> Array:
  """Like numpy.eye, create a 2D array with ones on a diagonal."""
  N, M = tuple(map(int, shape))
  offset = int(offset)
  dtype = dtypes.canonicalize_dtype(dtype)
  bool_eye = eq(add(broadcasted_iota(np.int32, (N, M), 0), np.int32(offset)),
                broadcasted_iota(np.int32, (N, M), 1))
  return convert_element_type_p.bind(bool_eye, new_dtype=dtype, weak_type=False)

def _delta(dtype: DType, shape: Shape, axes: Sequence[int]) -> Array:
  """This utility function exists for creating Kronecker delta arrays."""
  shape = tuple(map(int, shape))
  axes = tuple(map(int, axes))
  dtype = dtypes.canonicalize_dtype(dtype)
  base_shape = tuple(np.take(shape, axes))  # type: ignore[arg-type]
  iotas = [broadcasted_iota(np.uint32, base_shape, i)
           for i in range(len(base_shape))]
  eyes = [eq(i1, i2) for i1, i2 in zip(iotas[:-1], iotas[1:])]
  result = convert_element_type_p.bind(_reduce(operator.and_, eyes),
                                       new_dtype=dtype, weak_type=False)
  return broadcast_in_dim(result, shape, axes)

def _tri(dtype: DType, shape: Shape, offset: int) -> Array:
  """Like numpy.tri, create a 2D array with ones below a diagonal."""
  N, M = tuple(map(int, shape))
  offset = int(offset)
  dtype = dtypes.canonicalize_dtype(dtype)
  bool_tri = ge(add(broadcasted_iota(np.int32, (N, M), 0), np.int32(offset)),
                broadcasted_iota(np.int32, (N, M), 1))
  return convert_element_type_p.bind(bool_tri, new_dtype=dtype, weak_type=False)

def stop_gradient(x):
  """Stops gradient computation.

  Operationally ``stop_gradient`` is the identity function, that is, it returns
  argument `x` unchanged. However, ``stop_gradient`` prevents the flow of
  gradients during forward or reverse-mode automatic differentiation. If there
  are multiple nested gradient computations, ``stop_gradient`` stops gradients
  for all of them.

  For example:

  >>> jax.grad(lambda x: x**2)(3.)
  DeviceArray(6., dtype=float32)
  >>> jax.grad(lambda x: jax.lax.stop_gradient(x)**2)(3.)
  array(0., dtype=float32)
  >>> jax.grad(jax.grad(lambda x: x**2))(3.)
  DeviceArray(2., dtype=float32)
  >>> jax.grad(jax.grad(lambda x: jax.lax.stop_gradient(x)**2))(3.)
  array(0., dtype=float32)
  """
  def stop(x):
    if (dtypes.issubdtype(_dtype(x), np.floating) or
        dtypes.issubdtype(_dtype(x), np.complexfloating)):
      return ad_util.stop_gradient_p.bind(x)
    else:
      return x  # only bind primitive on inexact dtypes, to avoid some staging
  return tree_map(stop, x)


### convenience wrappers around traceables


def conv(lhs: Array, rhs: Array, window_strides: Sequence[int],
         padding: str, precision: PrecisionLike = None,
         preferred_element_type: Optional[DType] = None) -> Array:
  """Convenience wrapper around `conv_general_dilated`.

  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    An array containing the convolution result.
  """
  return conv_general_dilated(lhs, rhs, window_strides, padding,
                              precision=precision,
                              preferred_element_type=preferred_element_type)

def conv_with_general_padding(lhs: Array, rhs: Array,
                              window_strides: Sequence[int],
                              padding: Union[str, Sequence[Tuple[int, int]]],
                              lhs_dilation: Optional[Sequence[int]],
                              rhs_dilation: Optional[Sequence[int]],
                              precision: PrecisionLike = None,
                              preferred_element_type: Optional[DType] = None) -> Array:
  """Convenience wrapper around `conv_general_dilated`.

  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    lhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `lhs`. LHS dilation
      is also known as transposed convolution.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. RHS dilation
      is also known as atrous convolution.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    An array containing the convolution result.
  """
  return conv_general_dilated(
      lhs, rhs, window_strides, padding, lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation, precision=precision,
      preferred_element_type=preferred_element_type)


def _conv_transpose_padding(k, s, padding):
  """Calculate before and after padding for a dim of transposed convolution.

  Args:
    k: int: kernel dimension.
    s: int: dimension stride value.
    padding: 'same' or 'valid' padding mode for original forward conv.

  Returns:
    2-tuple: ints: before and after padding for transposed convolution.
  """
  if padding == 'SAME':
    pad_len = k + s - 2
    if s > k - 1:
      pad_a = k - 1
    else:
      pad_a = int(np.ceil(pad_len / 2))
  elif padding == 'VALID':
    pad_len = k + s - 2 + _max(k - s, 0)
    pad_a = k - 1
  else:
    raise ValueError('Padding mode must be `SAME` or `VALID`.')
  pad_b = pad_len - pad_a
  return pad_a, pad_b


def _flip_axes(x, axes):
  """Flip ndarray 'x' along each axis specified in axes tuple."""
  for axis in axes:
    x = np.flip(x, axis)
  return x


def conv_transpose(lhs: Array, rhs: Array, strides: Sequence[int],
                   padding: Union[str, Sequence[Tuple[int, int]]],
                   rhs_dilation: Optional[Sequence[int]] = None,
                   dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
                   transpose_kernel: bool = False,
                   precision: PrecisionLike = None,
                   preferred_element_type: Optional[DType] = None) -> Array:
  """Convenience wrapper for calculating the N-d convolution "transpose".

  This function directly calculates a fractionally strided conv rather than
  indirectly calculating the gradient (transpose) of a forward convolution.

  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    strides: sequence of `n` integers, sets fractional stride.
    padding: 'SAME', 'VALID' will set as transpose of corresponding forward
      conv, or a sequence of `n` integer 2-tuples describing before-and-after
      padding for each `n` spatial dimension.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. RHS dilation
      is also known as atrous convolution.
    dimension_numbers: tuple of dimension descriptors as in
      lax.conv_general_dilated. Defaults to tensorflow convention.
    transpose_kernel: if True flips spatial axes and swaps the input/output
      channel axes of the kernel. This makes the output of this function identical
      to the gradient-derived functions like keras.layers.Conv2DTranspose
      applied to the same kernel. For typical use in neural nets this is completely
      pointless and just makes input/output channel specification confusing.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    Transposed N-d convolution, with output padding following the conventions of
    keras.layers.Conv2DTranspose.
  """
  assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
  ndims = len(lhs.shape)
  one = (1,) * (ndims - 2)
  # Set dimensional layout defaults if not specified.
  if dimension_numbers is None:
    if ndims == 2:
      dimension_numbers = ('NC', 'IO', 'NC')
    elif ndims == 3:
      dimension_numbers = ('NHC', 'HIO', 'NHC')
    elif ndims == 4:
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    elif ndims == 5:
      dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
    else:
      raise ValueError('No 4+ dimensional dimension_number defaults.')
  dn = conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
  k_shape = np.take(rhs.shape, dn.rhs_spec)
  k_sdims = k_shape[2:]  # type: ignore[index]
  # Calculate correct output shape given padding and strides.
  pads: Union[str, Sequence[Tuple[int, int]]]
  if padding in {'SAME', 'VALID'}:
    if rhs_dilation is None:
      rhs_dilation = (1,) * (rhs.ndim - 2)
    effective_k_size = map(lambda k, r: (k-1) * r + 1, k_sdims, rhs_dilation)
    pads = [_conv_transpose_padding(k, s, padding)
            for k,s in zip(effective_k_size, strides)]
  else:
    pads = padding
  if transpose_kernel:
    # flip spatial dims and swap input / output channel axes
    rhs = _flip_axes(rhs, np.array(dn.rhs_spec)[2:])
    rhs = np.swapaxes(rhs, dn.rhs_spec[0], dn.rhs_spec[1])
  return conv_general_dilated(lhs, rhs, one, pads, strides, rhs_dilation, dn,
                              precision=precision,
                              preferred_element_type=preferred_element_type)


def full_like(x: Array, fill_value: Array, dtype: Optional[DType] = None,
              shape: Optional[Shape] = None) -> Array:
  """Create a full array like np.full based on the example array `x`.

  Args:
    x: example array-like, used for shape and dtype information.
    fill_value: a scalar value to fill the entries of the output array.
    dtype: optional, a dtype parameter for the output ndarray.
    shape: optional, a shape parameter for the output ndarray.

  Returns:
    An ndarray with the same shape as `x` with its entries set equal to
    `fill_value`, similar to the output of np.full.
  """
  fill_shape = np.shape(x) if shape is None else canonicalize_shape(shape)
  weak_type = dtype is None and dtypes.is_weakly_typed(x)
  dtype = dtype or _dtype(x)
  return full(fill_shape, _convert_element_type(fill_value, dtype, weak_type))


def collapse(operand: Array, start_dimension: int,
             stop_dimension: int) -> Array:
  """Collapses dimensions of an array into a single dimension.

  For example, if ``operand`` is an array with shape ``[2, 3, 4]``,
  ``collapse(operand, 0, 2).shape == [6, 4]``. The elements of the collapsed
  dimension are laid out major-to-minor, i.e., with the lowest-numbered
  dimension as the slowest varying dimension.

  Args:
    operand: an input array.
    start_dimension: the start of the dimensions to collapse (inclusive).
    stop_dimension: the end of the dimensions to collapse (exclusive).

  Returns:
    An array where dimensions ``[start_dimension, stop_dimension)`` have been
    collapsed (raveled) into a single dimension.
  """
  lo, hi = start_dimension, stop_dimension
  size = prod(operand.shape[lo:hi])
  new_shape = operand.shape[:lo] + (size,) + operand.shape[hi:]
  return reshape(operand, new_shape)


def slice_in_dim(operand: Array, start_index: Optional[int],
                 limit_index: Optional[int],
                 stride: int = 1, axis: int = 0)-> Array:
  """Convenience wrapper around slice applying to only one dimension."""
  start_indices = [0] * operand.ndim
  limit_indices = list(operand.shape)
  strides = [1] * operand.ndim

  # translate `None`
  len_axis = operand.shape[axis]
  start_index_int = _canonicalize_dimension(start_index) if start_index is not None else 0
  limit_index_int = _canonicalize_dimension(limit_index) if limit_index is not None else len_axis

  # translate negative indices
  if start_index_int < 0:
    start_index_int = start_index_int + len_axis
  if limit_index_int < 0:
    limit_index_int = limit_index_int + len_axis

  axis = int(axis)
  start_indices[axis] = start_index_int
  limit_indices[axis] = limit_index_int
  strides[axis] = int(stride)

  return slice(operand, start_indices, limit_indices, strides)


def index_in_dim(operand: Array, index: int, axis: int = 0,
                 keepdims: bool = True) -> Array:
  """Convenience wrapper around slice to perform int indexing."""
  index, axis = int(index), int(axis)
  axis_size = operand.shape[axis]
  wrapped_index = index + axis_size if index < 0 else index
  if not 0 <= wrapped_index < axis_size:
    msg = 'index {} is out of bounds for axis {} with size {}'
    raise IndexError(msg.format(index, axis, axis_size))
  result = slice_in_dim(operand, wrapped_index, wrapped_index + 1, 1, axis)
  if keepdims:
    return result
  else:
    return squeeze(result, (axis,))


def dynamic_slice_in_dim(operand: Array, start_index: Array,
                         slice_size: int, axis: int = 0) -> Array:
  """Convenience wrapper around dynamic_slice applying to one dimension."""
  start_indices = [_zero(start_index)] * operand.ndim
  slice_sizes = list(operand.shape)

  axis = int(axis)
  start_indices[axis] = start_index
  slice_sizes[axis] = int(slice_size)
  return dynamic_slice(operand, start_indices, slice_sizes)


def dynamic_index_in_dim(operand: Array, index: Array, axis: int = 0,
                         keepdims: bool = True) -> Array:
  """Convenience wrapper around dynamic_slice to perform int indexing."""
  result = dynamic_slice_in_dim(operand, index, 1, axis)
  if keepdims:
    return result
  else:
    return squeeze(result, (axis,))


def dynamic_update_slice_in_dim(operand: Array, update: Array,
                                start_index: Array, axis: int) -> Array:
  """Convenience wrapper around :func:`dynamic_update_slice` to update a slice
     in a single ``axis``.
  """
  axis = int(axis)
  start_indices = [_zero(start_index)] * _ndim(operand)
  start_indices[axis] = start_index
  return dynamic_update_slice(operand, update, start_indices)


def dynamic_update_index_in_dim(operand: Array, update: Array, index: Array,
                                axis: int) -> Array:
  """Convenience wrapper around :func:`dynamic_update_slice` to update a slice
     of size 1 in a single ``axis``.
  """
  axis = int(axis)
  if _ndim(update) != _ndim(operand):
    assert _ndim(update) + 1 == _ndim(operand)
    update = expand_dims(update, (axis,))
  return dynamic_update_slice_in_dim(operand, update, index, axis)


def batch_matmul(lhs: Array, rhs: Array,
                 precision: PrecisionLike = None) -> Array:
  """Batch matrix multiplication."""
  if _min(lhs.ndim, rhs.ndim) < 2:
    raise ValueError('Arguments to batch_matmul must be at least 2D, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  if lhs.ndim != rhs.ndim:
    raise ValueError('Arguments to batch_matmul must have same ndim, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  lhs_contract = (lhs.ndim - 1,)
  rhs_contract = (rhs.ndim - 2,)
  batch = tuple(range(lhs.ndim - 2))
  return dot_general(lhs, rhs, ((lhs_contract, rhs_contract), (batch, batch)),
                     precision=precision)


# These functions also exist in the XLA client library, but we treat them
# as non-primitive to maintain a smaller set of autodiff primitives.

def square(x: Array) -> Array:
  r"""Elementwise square: :math:`x^2`."""
  return integer_pow(x, 2)

def reciprocal(x: Array) -> Array:
  r"""Elementwise reciprocal: :math:`1 \over x`."""
  return integer_pow(x, -1)

def _upcast_fp16_for_computation(f):
  @functools.wraps(f)
  def f_wrapped(x):
    dtype = _dtype(x)
    if dtype == np.float16 or dtype == dtypes.bfloat16:
      return convert_element_type(
        f(convert_element_type(x, np.float32)), dtype)
    return f(x)

  return f_wrapped

def tan(x: Array) -> Array:
  r"""Elementwise tangent: :math:`\mathrm{tan}(x)`."""
  return tan_p.bind(x)

def asin(x: Array) -> Array:
  r"""Elementwise arc sine: :math:`\mathrm{asin}(x)`."""
  return asin_p.bind(x)

def acos(x: Array) -> Array:
  r"""Elementwise arc cosine: :math:`\mathrm{acos}(x)`."""
  return acos_p.bind(x)

def atan(x: Array) -> Array:
  r"""Elementwise arc tangent: :math:`\mathrm{atan}(x)`."""
  return atan_p.bind(x)

def sinh(x: Array) -> Array:
  r"""Elementwise hyperbolic sine: :math:`\mathrm{sinh}(x)`."""
  return sinh_p.bind(x)

def cosh(x: Array) -> Array:
  r"""Elementwise hyperbolic cosine: :math:`\mathrm{cosh}(x)`."""
  return cosh_p.bind(x)

def asinh(x: Array) -> Array:
  r"""Elementwise inverse hyperbolic sine: :math:`\mathrm{asinh}(x)`."""
  return asinh_p.bind(x)

def acosh(x: Array) -> Array:
  r"""Elementwise inverse hyperbolic cosine: :math:`\mathrm{acosh}(x)`."""
  return acosh_p.bind(x)

def atanh(x: Array) -> Array:
  r"""Elementwise inverse hyperbolic tangent: :math:`\mathrm{atanh}(x)`."""
  return atanh_p.bind(x)


# Add some methods to ShapedArray that rely on lax primitives

ShapedArray.broadcast = core.aval_method(broadcast)
ShapedArray.transpose = core.aval_method(transpose)  # clobbered by lax_numpy
ShapedArray.reshape = core.aval_method(reshape)      # clobbered by lax_numpy

def _iter(tracer):
  if tracer.ndim == 0:
    raise TypeError("iteration over a 0-d array")  # same as numpy error
  else:
    n = int(tracer.shape[0])
    # return (index_in_dim(tracer, i, keepdims=False) for i in range(n))
    return iter([index_in_dim(tracer, i, keepdims=False) for i in range(n)])
ShapedArray._iter = staticmethod(_iter)

# Add some ad handlers that use (or could use) lax primitives

def zeros_like_array(x):
  return full_like(x, 0)

for t in itertools.chain(
    dtypes.python_scalar_dtypes.keys(), array_types,
    [xla._CppDeviceArray, xla._DeviceArray, pxla.ShardedDeviceArray]):
  ad_util.jaxval_adders[t] = add
ad_util.jaxval_zeros_likers[xla._DeviceArray] = zeros_like_array
ad_util.jaxval_zeros_likers[xla._CppDeviceArray] = zeros_like_array
ad_util.jaxval_zeros_likers[pxla.ShardedDeviceArray] = zeros_like_array


### primitives


_input_dtype = lambda *args, **_: dtypes.canonicalize_dtype(args[0].dtype)
_fixed_dtype = lambda dtype: lambda *args, **kwargs: dtypes.canonicalize_dtype(dtype)
_complex_basetype = lambda dtype: np.abs(np.zeros((), dtype)).dtype

_strip_weak_type = lambda *args, **_: False
def _argnum_weak_type(*argnums):
  return lambda *args, **_: all(args[i].weak_type for i in argnums)

def standard_primitive(shape_rule, dtype_rule, name, translation_rule=None,
                       weak_type_rule=None, named_shape_rule=None):
  weak_type_rule = weak_type_rule or _standard_weak_type_rule
  named_shape_rule = named_shape_rule or standard_named_shape_rule
  prim = Primitive(name)
  prim.def_impl(partial(xla.apply_primitive, prim))
  prim.def_abstract_eval(
      partial(standard_abstract_eval, prim, shape_rule, dtype_rule,
              weak_type_rule, named_shape_rule))
  xla.translations[prim] = translation_rule or partial(standard_translate, name)
  return prim

def standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule,
                           named_shape_rule, *avals, **kwargs):
  assert all(isinstance(aval, UnshapedArray) for aval in avals), avals
  assert not prim.multiple_results
  weak_type = weak_type_rule(*avals, **kwargs)
  least_specialized = _max(map(type, avals),
                           key=operator.attrgetter('array_abstraction_level'))
  if least_specialized is ConcreteArray:
    return ConcreteArray(prim.impl(*[x.val for x in avals], **kwargs),
                         weak_type=weak_type)
  elif least_specialized is ShapedArray:
    return ShapedArray(shape_rule(*avals, **kwargs), dtype_rule(*avals, **kwargs),
                       weak_type=weak_type,
                       named_shape=named_shape_rule(*avals, **kwargs))
  elif least_specialized is UnshapedArray:
    return UnshapedArray(dtype_rule(*avals, **kwargs), weak_type=weak_type)
  else:
    raise TypeError(avals, least_specialized)

def standard_multi_result_abstract_eval(
    prim, shape_rule, dtype_rule, weak_type_rule,
    named_shape_rule, *avals, **kwargs):
  assert prim.multiple_results
  assert all(isinstance(aval, UnshapedArray) for aval in avals), avals
  least_specialized = _max(map(type, avals),
                           key=operator.attrgetter('array_abstraction_level'))
  weak_types = weak_type_rule(*avals, **kwargs)
  if least_specialized is ConcreteArray:
    out_vals = prim.impl(*[x.val for x in avals], **kwargs)
    return [ConcreteArray(val, weak_type=weak_type)
            for val, weak_type in safe_zip(out_vals, weak_types)]
  elif least_specialized is ShapedArray:
    out_shapes = shape_rule(*avals, **kwargs)
    out_dtypes = dtype_rule(*avals, **kwargs)
    out_named_shapes = named_shape_rule(*avals, **kwargs)
    return [ShapedArray(s, d, weak_type=weak_type, named_shape=named_shape)
            for s, d, weak_type, named_shape
            in safe_zip(out_shapes, out_dtypes, weak_types, out_named_shapes)]
  elif least_specialized is UnshapedArray:
    out_dtypes = dtype_rule(*avals, **kwargs)
    return [UnshapedArray(dtype, weak_type=weak_type)
            for dtype, weak_type in safe_zip(out_dtypes, weak_types)]
  else:
    raise TypeError(avals, least_specialized)

def standard_translate(name, c, *args, **kwargs):
  xla_opname = ''.join(term.capitalize() for term in name.split('_'))
  return getattr(xops, xla_opname)(*args, **kwargs)

def standard_named_shape_rule(*avals, **kwargs):
  return core.join_named_shapes(*(a.named_shape for a in avals))


def unop_dtype_rule(result_dtype, accepted_dtypes, name, aval, **kwargs):
  if not any(dtypes.issubdtype(aval.dtype, t) for t in accepted_dtypes):
    msg = '{} does not accept dtype {}. Accepted dtypes are subtypes of {}.'
    typename = str(np.dtype(aval.dtype).name)
    accepted_typenames = (t.__name__ for t in accepted_dtypes)
    raise TypeError(msg.format(name, typename, ', '.join(accepted_typenames)))
  return result_dtype(aval.dtype)


def unop(result_dtype, accepted_dtypes, name, translation_rule=None):
  dtype_rule = partial(unop_dtype_rule, result_dtype, accepted_dtypes, name)
  weak_type_rule = partial(_naryop_weak_type_rule, name)
  prim = standard_primitive(_attrgetter('shape'), dtype_rule, name,
                            translation_rule=translation_rule, weak_type_rule=weak_type_rule)
  batching.defvectorized(prim)
  masking.defvectorized(prim)
  return prim
standard_unop = partial(unop, _identity)
_attrgetter = lambda name: lambda x, **kwargs: getattr(x, name)


def naryop_dtype_rule(result_dtype, accepted_dtypes, name, *avals, **kwargs):
  aval_dtypes = [aval.dtype for aval in avals]
  for i, (aval_dtype, types) in enumerate(zip(aval_dtypes, accepted_dtypes)):
    if not any(dtypes.issubdtype(aval_dtype, t) for t in types):
      if aval_dtype is dtypes.float0:
        raise TypeError(
            f"Called {name} with a float0 at position {i}. "
            "float0s do not support any operations by design, because they "
            "are not compatible with non-trivial vector spaces. No implicit dtype "
            "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
            "to cast a float0 array to a regular zeros array. \n"
            "If you didn't expect to get a float0 you might have accidentally "
            "taken a gradient with respect to an integer argument.")
      else:
        msg = ('{} does not accept dtype {} at position {}. '
               'Accepted dtypes at position {} are subtypes of {}.')
        typename = str(np.dtype(aval_dtype).name)
        typenames = ', '.join(t.__name__ for t in types)
        raise TypeError(msg.format(name, typename, i, i, typenames))
  _check_same_dtypes(name, False, *aval_dtypes)
  return result_dtype(*avals)


def _broadcasting_shape_rule(name, *avals):
  shapes = [aval.shape for aval in avals if aval.shape]
  if not shapes:
    return ()
  if len({len(shape) for shape in shapes}) != 1:
    msg = '{} got arrays of different rank: {}.'
    raise TypeError(msg.format(name, ', '.join(map(str, map(tuple, shapes)))))
  result_shape = _try_broadcast_shapes(shapes)
  if result_shape is None:
    msg = '{} got incompatible shapes for broadcasting: {}.'
    raise TypeError(msg.format(name, ', '.join(map(str, map(tuple, shapes)))))
  return result_shape

def _standard_weak_type_rule(*avals, **kwargs):
  return all(aval.weak_type for aval in avals)

def _naryop_weak_type_rule(name, *avals, **kwargs):
  if any(aval.dtype is dtypes.float0 for aval in avals):
    pos = next(i for i, aval in enumerate(avals) if aval.dtype is dtypes.float0)
    raise TypeError(
        f"Called {name} with a float0 at position {pos}. "
        "float0s do not support any operations by design, because they "
        "are not compatible with non-trivial vector spaces. No implicit dtype "
        "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
        "to cast a float0 array to a regular zeros array. \n"
        "If you didn't expect to get a float0 you might have accidentally "
        "taken a gradient with respect to an integer argument.")
  return all(aval.weak_type for aval in avals)

def naryop(result_dtype, accepted_dtypes, name, translation_rule=None):
  # TODO(frostig,mattjj): only used with arity > 2 once, simplify
  dtype_rule = partial(naryop_dtype_rule, result_dtype, accepted_dtypes, name)
  shape_rule = partial(_broadcasting_shape_rule, name)
  weak_type_rule = partial(_naryop_weak_type_rule, name)
  prim = standard_primitive(shape_rule, dtype_rule, name,
                            translation_rule=translation_rule,
                            weak_type_rule=weak_type_rule)
  batching.defbroadcasting(prim)
  masking.defnaryop(prim)
  return prim
standard_naryop = partial(naryop, _input_dtype)

# Decorator for translation rules which adds explicit broadcasting of positional
# arguments. This is necessary only for a handful of primitives whose XLA
# implementations do not support broadcasting.
def _broadcast_translate(translate: Callable):
  def _broadcast_array(x, shape, result_shape):
    if shape == result_shape:
      return x
    bcast_dims = tuple(range(len(result_shape) - len(shape), len(result_shape)))
    result = xops.BroadcastInDim(x, result_shape, bcast_dims)
    return result

  def _broadcasted_translation_rule(c, *args, **kwargs):
    shapes = [c.get_shape(x).dimensions() for x in args]
    result_shape = broadcast_shapes(*shapes)
    args = [_broadcast_array(x, s, result_shape) for x, s in zip(args, shapes)]
    return translate(c, *args, **kwargs)
  return _broadcasted_translation_rule

# Like autograd.numpy.numpy_vjps.unbroadcast, this utility handles transposition
# involving linear primitives with implicit broadcasting.
def _unbroadcast(aval, x):
  if not isinstance(aval, ShapedArray):
    raise TypeError("transpose with implicit broadcasting of unshaped values")
  x_shape = np.shape(x)
  if aval.shape == x_shape:
    return x
  assert not aval.shape or len(x_shape) == len(aval.shape)
  if not aval.shape:
    return _reduce_sum(x, list(range(len(x_shape))))
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, aval.shape)) if a != b]
    if config.jax_enable_checks: assert all(aval.shape[i] == 1 for i in dims)
    return reshape(_reduce_sum(x, dims), aval.shape)

def _maybe_broadcast(target_shape, x):
  x_shape = np.shape(x)
  if x_shape == target_shape:
    return x
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, target_shape)) if a == b]
    squeeze_shape = [x_shape[i] for i in dims]
    return broadcast_in_dim(reshape(x, squeeze_shape), target_shape, dims)


_float = {np.floating}
_complex = {np.complexfloating}
_complex_elem_types = {np.float32, np.float64}
_int = {np.integer}
_bool = {np.bool_}

_num = _int | _float | _complex
_any = _int | _float | _complex | _bool
_bool_or_int = _int | _bool

neg_p = standard_unop(_num, 'neg')
ad.deflinear2(neg_p, lambda t, operand: [neg(t)])

def _sign_translation_rule(c, x):
  shape = c.get_shape(x)
  dtype = shape.numpy_dtype()
  if dtypes.issubdtype(dtype, np.unsignedinteger):
    zero = xb.constant(c, np.array(0, dtype=dtype))
    dims = c.get_shape(x).dimensions()
    return xops.Select(xops.Eq(x, zero), xops.Broadcast(zero, dims),
                       xops.Broadcast(xb.constant(c, np.array(1, dtype=dtype)),
                                      dims))
  return xops.Sign(x)

sign_p = standard_unop(_num, 'sign', translation_rule=_sign_translation_rule)
ad.defjvp_zero(sign_p)

_nextafter_translation_rule = \
    _broadcast_translate(partial(standard_translate, 'next_after'))
nextafter_p = standard_naryop([_float, _float], 'nextafter',
                              translation_rule=_nextafter_translation_rule)

floor_p = standard_unop(_float, 'floor')
ad.defjvp_zero(floor_p)

ceil_p = standard_unop(_float, 'ceil')
ad.defjvp_zero(ceil_p)

def _round_to_nearest_even(x):
  half = _const(x, 0.5)
  one = _const(x, 1)
  round_val = floor(x)
  fraction = x - round_val
  nearest_even_int = sub(
    round_val, mul(_const(x, 2), floor(mul(half, x))))
  is_odd = eq(nearest_even_int, one)
  return select(
    bitwise_or(gt(fraction, half),
               bitwise_and(eq(fraction, half), is_odd)),
    add(round_val, one), round_val)

def _round_translation_rule(c, x, *, rounding_method):
  if rounding_method is RoundingMethod.AWAY_FROM_ZERO:
    return xops.Round(x)
  else: # rounding_method is RoundingMethod.TO_NEAREST_EVEN
    rounding_fun = xla.lower_fun(_round_to_nearest_even, multiple_results=False)
    return rounding_fun(c, x)

round_p = standard_unop(_float, 'round')
xla.translations[round_p] = _round_translation_rule
ad.defjvp_zero(round_p)

is_finite_p = unop(_fixed_dtype(np.bool_), _float, 'is_finite')
ad.defjvp_zero(is_finite_p)

exp_p = standard_unop(_float | _complex, 'exp')
ad.defjvp2(exp_p, lambda g, ans, x: mul(g, ans))
iad.definverse(exp_p, lambda r, x: log(r))
# For exp_p it is more efficient to use the reconstructed output for the vjp
# rule instead of computing it again from the input.
iad.primitive_ivjps[exp_p] = lambda x, y, ct: [[log(y[0])], [ct[0] * y[0]]]

log_p = standard_unop(_float | _complex, 'log')
ad.defjvp(log_p, lambda g, x: div(g, x))
iad.definverse(log_p, lambda r, x: exp(r))

expm1_p = standard_unop(_float | _complex, 'expm1')
ad.defjvp2(expm1_p, lambda g, ans, x: mul(g, add(ans, _one(ans))))

log1p_p = standard_unop(_float | _complex, 'log1p')
ad.defjvp(log1p_p, lambda g, x: div(g, add(x, _one(x))))

tanh_p = standard_unop(_float | _complex, 'tanh')
ad.defjvp2(tanh_p, lambda g, ans, x: mul(add(g, mul(g, ans)),
                                         sub(_one(x), ans)))

sin_p = standard_unop(_float | _complex, 'sin')
ad.defjvp(sin_p, lambda g, x: mul(g, cos(x)))

cos_p = standard_unop(_float | _complex, 'cos')
ad.defjvp(cos_p, lambda g, x: neg(mul(g, sin(x))))

@partial(xla.lower_fun, multiple_results=False)
@_upcast_fp16_for_computation
def tan_translation_rule(x):
  return div(sin(x), cos(x))

tan_p = standard_unop(_float | _complex, 'tan',
                       translation_rule=tan_translation_rule)
ad.defjvp(tan_p, lambda g, x: mul(g, _const(x, 1) + square(tan(x))))


@partial(xla.lower_fun, multiple_results=False)
def asin_translation_rule(x):
  if dtypes.issubdtype(_dtype(x), np.complexfloating):
    return mul(_const(x, -1j), asinh(mul(_const(x, 1j), x)))
  else:
    return mul(_const(x, 2),
               atan2(x, add(_const(x, 1), sqrt(sub(_const(x, 1), square(x))))))

asin_p = standard_unop(_float | _complex, 'asin',
                       translation_rule=asin_translation_rule)
ad.defjvp(asin_p, lambda g, x: mul(g, rsqrt(_const(x, 1) - square(x))))


@partial(xla.lower_fun, multiple_results=False)
def acos_translation_rule(x):
  if dtypes.issubdtype(_dtype(x), np.complexfloating):
    result = mul(_const(x, 1j), acosh(x))
    # By convention, numpy chooses the branch with positive real part.
    rpart = real(result)
    return select(
      gt(rpart, _const(rpart, 0)),
      result,
      neg(result)
    )
  else:
    return select(
        ne(x, _const(x, -1.0)),
        mul(_const(x, 2),
            atan2(sqrt(sub(_const(x, 1), square(x))), add(_const(x, 1), x))),
        full_like(x, np.pi))

acos_p = standard_unop(_float | _complex, 'acos',
                       translation_rule=acos_translation_rule)
ad.defjvp(acos_p, lambda g, x: mul(g, -rsqrt(_const(x, 1) - square(x))))

@partial(xla.lower_fun, multiple_results=False)
def atan_translation_rule(x):
  if dtypes.issubdtype(_dtype(x), np.complexfloating):
    return mul(_const(x, -1j), atanh(mul(_const(x, 1j), x)))
  else:
    return atan2(x, _const(x, 1))

atan_p = standard_unop(_float | _complex, 'atan',
                       translation_rule=atan_translation_rule)
ad.defjvp(atan_p, lambda g, x: div(g, _const(x, 1) + square(x)))

atan2_p = standard_naryop([_float, _float], 'atan2')
ad.defjvp(atan2_p,
          lambda g, x, y: g * (y / (square(x) + square(y))),
          lambda g, x, y: g * -x / (square(x) + square(y)))

sinh_p = standard_unop(_float | _complex, 'sinh')
ad.defjvp(sinh_p, lambda g, x: mul(g, cosh(x)))

cosh_p = standard_unop(_float | _complex, 'cosh')
ad.defjvp(cosh_p, lambda g, x: mul(g, sinh(x)))

asinh_p = standard_unop(_float | _complex, 'asinh')
ad.defjvp(asinh_p, lambda g, x: mul(g, rsqrt(square(x) + _one(x))))

acosh_p = standard_unop(_float | _complex, 'acosh')
ad.defjvp(acosh_p,
          lambda g, x: mul(g, rsqrt((x - _one(x)) * (x + _one(x)))))

atanh_p = standard_unop(_float | _complex, 'atanh')
ad.defjvp(atanh_p,
          lambda g, x: mul(reciprocal(_one(x) + x), div(g, (_one(x) - x))))

regularized_incomplete_beta_p = standard_naryop(
    [_float, _float, _float], 'regularized_incomplete_beta',
    translation_rule=_broadcast_translate(
      partial(standard_translate, 'regularized_incomplete_beta')))

def betainc_gradx(g, a, b, x):
  lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
  partial_x = exp((b - 1) * log1p(-x) +
                  (a - 1) * log(x) - lbeta)
  return partial_x * g

def betainc_grad_not_implemented(g, a, b, x):
  raise ValueError("Betainc gradient with respect to a and b not supported.")

ad.defjvp(regularized_incomplete_beta_p,
  betainc_grad_not_implemented,
  betainc_grad_not_implemented,
  betainc_gradx)

lgamma_p = standard_unop(_float, 'lgamma')
ad.defjvp(lgamma_p, lambda g, x: mul(g, digamma(x)))

digamma_p = standard_unop(_float, 'digamma')

igamma_p = standard_naryop(
  [_float, _float], 'igamma',
  translation_rule=_broadcast_translate(partial(standard_translate, 'igamma')))
igamma_grad_a_p = standard_naryop([_float, _float], 'igamma_grad_a',
  translation_rule=_broadcast_translate(partial(standard_translate,
                                               'igamma_grad_a')))

def igamma_gradx(g, a, x):
  return g * exp(-x + (a - _ones(a)) * log(x) - lgamma(a))

def igamma_grada(g, a, x):
  return g * igamma_grad_a(a, x)

ad.defjvp(igamma_p, igamma_grada, igamma_gradx)

igammac_p = standard_naryop(
  [_float, _float], 'igammac',
  translation_rule=_broadcast_translate(partial(standard_translate, 'igammac')))

def igammac_gradx(g, a, x):
  return -igamma_gradx(g, a, x)

def igammac_grada(g, a, x):
  return -igamma_grada(g, a, x)

ad.defjvp(igammac_p, igammac_grada, igammac_gradx)

random_gamma_grad_p = standard_naryop([_float, _float], 'random_gamma_grad',
  translation_rule=_broadcast_translate(partial(standard_translate,
                                               'random_gamma_grad')))

bessel_i0e_p = standard_unop(_float, 'bessel_i0e')
ad.defjvp2(bessel_i0e_p, lambda g, y, x: g * (bessel_i1e(x) - sign(x) * y))

bessel_i1e_p = standard_unop(_float, 'bessel_i1e')
def _bessel_i1e_jvp(g, y, x):
  eps = dtypes.finfo(_dtype(x)).eps
  x_is_not_tiny = abs(x) > eps
  safe_x = select(x_is_not_tiny, x, full_like(x, eps))
  dy_dx = bessel_i0e(safe_x) - y * (sign(safe_x) + reciprocal(safe_x))
  dy_dx = select(x_is_not_tiny, dy_dx, full_like(x, 0.5))
  return g * dy_dx
ad.defjvp2(bessel_i1e_p, _bessel_i1e_jvp)

erf_p = standard_unop(_float, 'erf')
ad.defjvp(erf_p, lambda g, x: mul(_const(x, 2. / np.sqrt(np.pi)),
                                  mul(g, exp(neg(square(x))))))

erfc_p = standard_unop(_float, 'erfc')
ad.defjvp(erfc_p, lambda g, x: mul(_const(x, 2. / np.sqrt(np.pi)),
                                   mul(neg(g), exp(neg(square(x))))))

erf_inv_p = standard_unop(_float, 'erf_inv')
ad.defjvp2(erf_inv_p, lambda g, ans, x: mul(_const(x, np.sqrt(np.pi) / 2.),
                                            mul(g, exp(square(ans)))))

real_p = unop(_complex_basetype, _complex, 'real')
ad.deflinear2(real_p, lambda t, _: [complex(t, np.zeros((), _dtype(t)))])

imag_p = unop(_complex_basetype, _complex, 'imag')
ad.deflinear2(imag_p, lambda t, _: [complex(np.zeros((), _dtype(t)), neg(t))])


def _complex_transpose_rule(t, x, y):
  assert ad.is_undefined_primal(x) or ad.is_undefined_primal(y)
  if ad.is_undefined_primal(x) and ad.is_undefined_primal(y):
    if type(t) is ad_util.Zero:
      return [ad_util.Zero(x.aval), ad_util.Zero(y.aval)]
    else:
      return [_unbroadcast(x.aval, real(t)), _unbroadcast(y.aval, imag(neg(t)))]
  elif ad.is_undefined_primal(x):
    if type(t) is ad_util.Zero:
      return [ad_util.Zero(x.aval), None]
    else:
      return [_unbroadcast(x.aval, real(t)), None]
  else:
    if type(t) is ad_util.Zero:
      return [None, ad_util.Zero(y.aval)]
    else:
      return [None, _unbroadcast(y.aval, imag(neg(t)))]

_complex_dtype = lambda dtype, *args: (np.zeros((), dtype) + np.zeros((), np.complex64)).dtype
complex_p = naryop(_complex_dtype, [_complex_elem_types, _complex_elem_types],
                  'complex')
ad.deflinear2(complex_p, _complex_transpose_rule)

conj_p = unop(_complex_dtype, _complex_elem_types | _complex, 'conj')

def _conj_transpose_rule(t, x, *, input_dtype):
  assert ad.is_undefined_primal(x)
  if dtypes.issubdtype(input_dtype, np.complexfloating):
    return [conj(t)]
  else:
    return [real(t)]

xla.translations[conj_p] = lambda c, x, **kwargs: xops.Conj(x)
ad.primitive_jvps[conj_p] = partial(ad.linear_jvp, conj_p)
ad.primitive_transposes[conj_p] = _conj_transpose_rule

abs_p = unop(_complex_basetype, _num, 'abs')

def _abs_jvp_rule(g, ans, x):
  if _iscomplex(x):
    return _maybe_real(mul(g, div(_maybe_conj(x),
           _replace_zero(convert_element_type(ans, _dtype(x))))))
  else:
    return select(ge(x, _zero(x)), g, neg(g))
ad.defjvp2(abs_p, _abs_jvp_rule)
_maybe_conj = lambda x: conj(x) if _iscomplex(x) else x
_maybe_real = lambda x: real(x) if _iscomplex(x) else x

sqrt_p = standard_unop(_float | _complex, 'sqrt')
ad.defjvp2(sqrt_p, lambda g, ans, x: mul(g, div(_const(x, 0.5), ans)))

rsqrt_p = standard_unop(_float | _complex, 'rsqrt')
ad.defjvp2(rsqrt_p,
           lambda g, ans, x:
           mul(g, mul(_const(x, -0.5), pow(x, _const(x, -1.5)))))

pow_p = standard_naryop([_float | _complex, _float | _complex], 'pow')

def _pow_jvp_lhs(g, ans, x, y):
  jac = mul(y, pow(x, select(eq(y, _zeros(y)), _ones(y), sub(y, _ones(y)))))
  return mul(g, jac)

def _pow_jvp_rhs(g, ans, x, y):
  return mul(g, mul(log(_replace_zero(x)), ans))

ad.defjvp2(pow_p, _pow_jvp_lhs, _pow_jvp_rhs)


def _integer_pow_dtype_rule(x, *, y):
  dtype = unop_dtype_rule(_identity, _int | _float | _complex, 'integer_pow', x)
  if y < 0 and dtypes.issubdtype(dtype, np.integer):
    raise TypeError("Integers cannot be raised to negative powers, got "
                    f"integer_pow({x}, {y})")
  return dtype

def _integer_pow_translation_rule(c, x, *, y):
  if y == 0:
    shape = c.get_shape(x)
    one = xb.constant(c, np.array(1, dtype=shape.numpy_dtype()))
    return xops.Broadcast(one, shape.dimensions())
  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y
  acc = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else xops.Mul(acc, x)
    y >>= 1
    if y > 0:
      x = xops.Mul(x, x)
  return xops.Reciprocal(acc) if is_reciprocal else acc

def _integer_pow_jvp(g, x, *, y):
  return _zeros(g) if y == 0 else mul(g, mul(_const(x, y), integer_pow(x, y - 1)))

integer_pow_p = standard_primitive(
  _attrgetter('shape'), _integer_pow_dtype_rule, 'integer_pow',
  translation_rule=_integer_pow_translation_rule)
batching.defvectorized(integer_pow_p)
masking.defvectorized(integer_pow_p)
ad.defjvp(integer_pow_p, _integer_pow_jvp)

_replace_zero = lambda x: select(eq(x, _const(x, 0)), _ones(x), x)

not_p = standard_unop(_bool_or_int, 'not')
ad.defjvp_zero(not_p)

and_p = standard_naryop([_bool_or_int, _bool_or_int], 'and')
ad.defjvp_zero(and_p)

or_p = standard_naryop([_bool_or_int, _bool_or_int], 'or')
ad.defjvp_zero(or_p)

xor_p = standard_naryop([_bool_or_int, _bool_or_int], 'xor')
ad.defjvp_zero(xor_p)

population_count_p = standard_unop(_int, 'population_count')

clz_p = standard_unop(_int, 'clz')

def _add_jvp(primals, tangents):
  x, y = primals
  xdot, ydot = tangents
  primal_out = add(x, y)
  if type(xdot) is type(ydot) is ad_util.Zero:
    return primal_out, ad_util.Zero.from_value(primal_out)
  if type(xdot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, ydot)
  elif type(ydot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, xdot)
  else:
    return primal_out, add(xdot, ydot)

def _add_transpose(t, x, y):
  # Morally the following assertion is true, but because we instantiate zeros in
  # some places (e.g. in custom_jvp) it may not always hold. For example, see
  # api_test.py's CustomJVPTest.test_jaxpr_zeros.
  # assert ad.is_undefined_primal(x) and ad.is_undefined_primal(y)
  x_aval = x.aval if ad.is_undefined_primal(x) else _abstractify(x)
  y_aval = y.aval if ad.is_undefined_primal(y) else _abstractify(y)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(x_aval), ad_util.Zero(y_aval)]
  else:
    return [_unbroadcast(x_aval, t), _unbroadcast(y_aval, t)]

def _add_inverse(r, x, y):
  xr = r - y
  yr = r - x
  return xr, yr

add_p = standard_naryop([_num, _num], 'add')
ad.primitive_jvps[add_p] = _add_jvp
ad.primitive_transposes[add_p] = _add_transpose
iad.definverse(add_p, _add_inverse)

def _sub_jvp(primals, tangents):
  x, y = primals
  xdot, ydot = tangents
  primal_out = sub(x, y)
  if type(xdot) is type(ydot) is ad_util.Zero:
    return primal_out, ad_util.Zero.from_value(primal_out)
  if type(xdot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, neg(ydot))
  elif type(ydot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, xdot)
  else:
    return primal_out, sub(xdot, ydot)

def _sub_transpose(t, x, y):
  # Morally the following assertion is true, but see the comment in add_p's
  # transpose rule.
  # assert ad.is_undefined_primal(x) and ad.is_undefined_primal(y)
  x_aval = x.aval if ad.is_undefined_primal(x) else _abstractify(x)
  y_aval = y.aval if ad.is_undefined_primal(y) else _abstractify(y)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(x_aval), ad_util.Zero(y_aval)]
  else:
    return [_unbroadcast(x_aval, t), _unbroadcast(y_aval, neg(t))]

sub_p = standard_naryop([_num, _num], 'sub')
ad.primitive_jvps[sub_p] = _sub_jvp
ad.primitive_transposes[sub_p] = _sub_transpose


def _mul_transpose(ct, x, y):
  assert ad.is_undefined_primal(x) ^ ad.is_undefined_primal(y)
  if ad.is_undefined_primal(x):
    if type(ct) is ad_util.Zero:
      return [ad_util.Zero(x.aval), None]
    else:
      return [_unbroadcast(x.aval, mul(ct, y)), None]
  else:
    if type(ct) is ad_util.Zero:
      return [None, ad_util.Zero(y.aval)]
    else:
      return [None, _unbroadcast(y.aval, mul(x, ct))]

def _mul_inverse(r, x, y):
  xr = r / y
  yr = r / x
  return xr, yr

mul_p = standard_naryop([_num, _num], 'mul')
ad.defjvp(mul_p,
          lambda xdot, x, y: mul(xdot, y),
          lambda ydot, x, y: mul(x, ydot))
ad.primitive_transposes[mul_p] = _mul_transpose
iad.definverse(mul_p, _mul_inverse)

def _div_transpose_rule(cotangent, x, y):
  assert ad.is_undefined_primal(x) and not ad.is_undefined_primal(y)
  if type(cotangent) is ad_util.Zero:
    return [ad_util.Zero(x.aval), None]
  else:
    return [_unbroadcast(x.aval, div(cotangent, y)), None]
div_p = standard_naryop([_num, _num], 'div')
ad.defjvp(div_p,
          lambda g, x, y: div(g, y),
          lambda g, x, y: mul(mul(neg(g), x), integer_pow(y, -2)))
ad.primitive_transposes[div_p] = _div_transpose_rule

rem_p = standard_naryop([_num, _num], 'rem')
ad.defjvp(
    rem_p,
    lambda g, x, y: _maybe_broadcast(broadcast_shapes(np.shape(x), np.shape(y)), g),
    lambda g, x, y: mul(neg(g), floor(div(x, y))))


def _broadcasting_select(c, which, x, y):
  """Wrapper around XLA `Select` that broadcasts its arguments."""
  which_shape, x_shape, y_shape = (
    c.get_shape(t).dimensions() for t in (which, x, y))
  out_shape = broadcast_shapes(which_shape, x_shape, y_shape)
  bcast_dims = lambda shape: tuple(range(len(out_shape) - len(shape),
                                         len(out_shape)))
  which = xops.BroadcastInDim(which, out_shape, bcast_dims(which_shape))
  x = xops.BroadcastInDim(x, out_shape, bcast_dims(x_shape))
  y = xops.BroadcastInDim(y, out_shape, bcast_dims(y_shape))
  return xops.Select(which, x, y)


def _minmax_translation_rule(c, x, y, *, minmax=None, cmp=None):
  dtype = c.get_shape(x).numpy_dtype()
  if dtypes.issubdtype(dtype, np.complexfloating):
    rx = xops.Real(x)
    ry = xops.Real(y)
    return _broadcasting_select(
        c, xops.Select(xops.Eq(rx, ry), cmp(xops.Imag(x), xops.Imag(y)),
                       cmp(rx, ry)),
        x, y)
  return minmax(x, y)

max_p: core.Primitive = standard_naryop(
  [_any, _any], 'max', translation_rule=partial(
    _minmax_translation_rule, minmax=xops.Max, cmp=xops.Gt))
ad.defjvp2(max_p,
           lambda g, ans, x, y: mul(g, _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(g, _balanced_eq(y, ans, x)))

min_p: core.Primitive = standard_naryop(
  [_any, _any], 'min', translation_rule=partial(
    _minmax_translation_rule, minmax=xops.Min, cmp=xops.Lt))
ad.defjvp2(min_p,
           lambda g, ans, x, y: mul(g, _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(g, _balanced_eq(y, ans, x)))

shift_left_p = standard_naryop([_int, _int], 'shift_left')
ad.defjvp_zero(shift_left_p)

shift_right_arithmetic_p = standard_naryop([_int, _int], 'shift_right_arithmetic')
ad.defjvp_zero(shift_right_arithmetic_p)

shift_right_logical_p = standard_naryop([_int, _int], 'shift_right_logical')
ad.defjvp_zero(shift_right_logical_p)

eq_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'eq')
ad.defjvp_zero(eq_p)

ne_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'ne')
ad.defjvp_zero(ne_p)

ge_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'ge')
ad.defjvp_zero(ge_p)

gt_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'gt')
ad.defjvp_zero(gt_p)

le_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'le')
ad.defjvp_zero(le_p)

lt_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'lt')
ad.defjvp_zero(lt_p)


def _convert_element_type_shape_rule(operand, *, new_dtype, weak_type):
  return operand.shape

def _convert_element_type_dtype_rule(operand, *, new_dtype, weak_type):
  return new_dtype

def _convert_element_type_weak_type_rule(operand, *, new_dtype, weak_type):
  return weak_type

def _convert_element_type_translation_rule(c, operand, *, new_dtype, weak_type):
  old_dtype = c.get_shape(operand).numpy_dtype()
  if (dtypes.issubdtype(old_dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    operand = xops.Real(operand)
  new_etype = xla_client.dtype_to_etype(new_dtype)
  return xops.ConvertElementType(operand, new_element_type=new_etype)

def _convert_element_type_transpose_rule(ct, operand, *, new_dtype, weak_type):
  assert ad.is_undefined_primal(operand)
  old_dtype = operand.aval.dtype
  old_weak_type = dtypes.is_weakly_typed(operand)
  if type(ct) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)]
  elif core.primal_dtype_to_tangent_dtype(old_dtype) is dtypes.float0:
    return [ad_util.Zero(operand.aval.update(dtype=dtypes.float0, weak_type=False))]
  else:
    return [convert_element_type_p.bind(ct, new_dtype=old_dtype,
                                        weak_type=old_weak_type)]

def _convert_element_type_jvp_rule(tangent, operand , *, new_dtype, weak_type):
  if core.primal_dtype_to_tangent_dtype(new_dtype) is dtypes.float0:
    return ad_util.Zero(tangent.aval.update(dtype=dtypes.float0, weak_type=False))
  else:
    return convert_element_type_p.bind(tangent, new_dtype=new_dtype,
                                       weak_type=weak_type)

convert_element_type_p = core.convert_element_type_p
convert_element_type_p.def_impl(partial(xla.apply_primitive, convert_element_type_p))
convert_element_type_p.def_abstract_eval(
    partial(standard_abstract_eval, convert_element_type_p,
            _convert_element_type_shape_rule, _convert_element_type_dtype_rule,
            _convert_element_type_weak_type_rule, standard_named_shape_rule))
xla.translations[convert_element_type_p] = _convert_element_type_translation_rule
ad.defjvp(convert_element_type_p, _convert_element_type_jvp_rule)
ad.primitive_transposes[convert_element_type_p] = _convert_element_type_transpose_rule
batching.defvectorized(convert_element_type_p)
masking.defvectorized(convert_element_type_p)


def _bitcast_convert_type_shape_rule(operand, *, new_dtype):
  return operand.shape

def _bitcast_convert_type_dtype_rule(operand, *, new_dtype):
  return new_dtype

def _bitcast_convert_type_translation_rule(c, operand, *, new_dtype):
  new_etype = xla_bridge.dtype_to_etype(new_dtype)
  return xops.BitcastConvertType(operand, new_element_type=new_etype)

bitcast_convert_type_p = standard_primitive(
    _bitcast_convert_type_shape_rule, _bitcast_convert_type_dtype_rule,
    'bitcast_convert_type', _bitcast_convert_type_translation_rule,
    weak_type_rule=_strip_weak_type)
ad.defjvp_zero(bitcast_convert_type_p)
batching.defvectorized(bitcast_convert_type_p)
masking.defvectorized(bitcast_convert_type_p)


def _conv_general_dilated_shape_rule(
    lhs: ShapedArray, rhs: ShapedArray, *, window_strides, padding,
    lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count,
    batch_group_count, **unused_kwargs) -> Tuple[int, ...]:
  assert type(dimension_numbers) is ConvDimensionNumbers
  if len(lhs.shape) != len(rhs.shape):
    msg = ("conv_general_dilated lhs and rhs must have the same number of "
           "dimensions, but got {} and {}.")
    raise ValueError(msg.format(lhs.shape, rhs.shape))
  if not feature_group_count > 0:
    msg = ("conv_general_dilated feature_group_count "
           "must be a positive integer, got {}.")
    raise ValueError(msg.format(feature_group_count))
  lhs_feature_count = lhs.shape[dimension_numbers.lhs_spec[1]]
  quot, rem = divmod(lhs_feature_count, feature_group_count)
  if rem:
    msg = ("conv_general_dilated feature_group_count must divide lhs feature "
           "dimension size, but {} does not divide {}.")
    raise ValueError(msg.format(feature_group_count, lhs_feature_count))
  if quot != rhs.shape[dimension_numbers.rhs_spec[1]]:
    msg = ("conv_general_dilated lhs feature dimension size divided by "
           "feature_group_count must equal the rhs input feature dimension "
           "size, but {} // {} != {}.")
    raise ValueError(msg.format(lhs_feature_count, feature_group_count,
                                rhs.shape[dimension_numbers.rhs_spec[1]]))
  if rhs.shape[dimension_numbers.rhs_spec[0]] % feature_group_count:
    msg = ("conv_general_dilated rhs output feature dimension size must be a "
           "multiple of feature_group_count, but {} is not a multiple of {}.")
    raise ValueError(msg.format(rhs.shape[dimension_numbers.rhs_spec[0]],
                                feature_group_count))

  if not batch_group_count > 0:
    msg = ("conv_general_dilated batch_group_count "
           "must be a positive integer, got {}.")
    raise ValueError(msg.format(batch_group_count))
  lhs_batch_count = lhs.shape[dimension_numbers.lhs_spec[0]]
  if batch_group_count > 1 and lhs_batch_count % batch_group_count != 0:
    msg = ("conv_general_dilated batch_group_count must divide lhs batch "
           "dimension size, but {} does not divide {}.")
    raise ValueError(msg.format(batch_group_count, lhs_batch_count))

  if rhs.shape[dimension_numbers.rhs_spec[0]] % batch_group_count:
    msg = ("conv_general_dilated rhs output feature dimension size must be a "
           "multiple of batch_group_count, but {} is not a multiple of {}.")
    raise ValueError(msg.format(rhs.shape[dimension_numbers.rhs_spec[0]],
                                batch_group_count))

  if batch_group_count > 1 and feature_group_count > 1:
    msg = ("At most one of batch_group_count and feature_group_count may be > "
           "1, got batch_group_count={} and feature_group_count={}")
    raise ValueError(msg.format(batch_group_count, feature_group_count))

  if len(_conv_sdims(dimension_numbers.rhs_spec)) != len(window_strides):
    msg = ("conv_general_dilated window and window_strides must have "
           "the same number of dimensions, but got {} and {}")
    raise ValueError(
        msg.format(len(_conv_sdims(dimension_numbers.rhs_spec)), len(window_strides)))

  lhs_perm, rhs_perm, out_perm = dimension_numbers
  lhs_trans = _dilate_shape(np.take(lhs.shape, lhs_perm), lhs_dilation)
  rhs_trans = _dilate_shape(np.take(rhs.shape, rhs_perm), rhs_dilation)
  out_trans = conv_shape_tuple(lhs_trans, rhs_trans, window_strides, padding,
                               batch_group_count)
  return tuple(np.take(out_trans, np.argsort(out_perm)))  # type: ignore[arg-type]

def _validate_preferred_element_type(input_dtype, preferred_element_type):
  allowed_types = (np.integer, np.floating, np.complexfloating)
  if any(dtypes.issubdtype(input_dtype, t) and not dtypes.issubdtype(preferred_element_type, t) for t in allowed_types):
    raise TypeError("`preferred_element_type` and the original type must both be integral, both be floating point, or both complex.")
  if dtypes.issubdtype(input_dtype, np.signedinteger) and not dtypes.issubdtype(preferred_element_type, np.signedinteger):
    raise TypeError("`preferred_element_type` must have the same signedness as the original type.")
  input_bitwidth = np.dtype(input_dtype).itemsize
  preferred_bitwidth = np.dtype(preferred_element_type).itemsize
  if preferred_bitwidth < input_bitwidth:
    raise TypeError("`preferred_element_type` must not be narrower than the original type.")

def _conv_general_dilated_dtype_rule(
    lhs, rhs, *, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, preferred_element_type, **unused_kwargs):
  input_dtype = naryop_dtype_rule(_input_dtype, [_any, _any],
                                  'conv_general_dilated', lhs, rhs)
  if preferred_element_type is None:
    return input_dtype
  _validate_preferred_element_type(input_dtype, preferred_element_type)
  return preferred_element_type

_conv_spec_transpose = lambda spec: (spec[1], spec[0]) + spec[2:]
_conv_sdims = lambda spec: spec[2:]

# Understanding the convolution transpose rules:
# Ignoring the spatial dimensions, let m = batch, j = input feature,
# k = output feature.
#
# Convolution computes the following contraction:
# Forward: [m, j] [j, k] -> [m, k]
#
# The transposes are similar to the rules for transposing a matmul:
# LHS transpose: [m, k] [k, j] -> [m, j]
# RHS transpose: [j, m] [m, k] -> [j, k]
#
# With feature grouping, we have the following signatures:
# Forward: [m, gj] [j, gk] -> [m, gk]
# LHS transpose: [m, gk] [k, gj] -> [m, gj]
# --> implemented as feature grouping after transposing the group from the
#     kernel input features to the kernel output features.
# RHS transpose: [gj, m] [m, gk] -> [j, gk]
# --> which is batch grouping.
#
# With batch grouping, we have the following signatures:
# Forward: [gm,j] [j,gk]->[m,gk]
# LHS transpose: [m, gk][gk, j] -> [gm, j]
# --> implemented as feature grouping with transposing the group on the kernel
#     and the output.
# RHS transpose: [j, gm][m, gk] -> [j, gk]
# --> which is feature grouping.

def _conv_general_dilated_transpose_lhs(
    g, rhs, *, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, batch_group_count,
    lhs_shape, rhs_shape, precision, preferred_element_type):
  assert type(dimension_numbers) is ConvDimensionNumbers
  assert batch_group_count == 1 or feature_group_count == 1
  lhs_sdims, rhs_sdims, out_sdims = map(_conv_sdims, dimension_numbers)
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  t_rhs_spec = _conv_spec_transpose(rhs_spec)
  if feature_group_count > 1:
    # in addition to switching the dims in the spec, need to move the feature
    # group axis into the transposed rhs's output feature dim
    rhs = _reshape_axis_out_of(rhs_spec[0], feature_group_count, rhs)
    rhs = _reshape_axis_into(rhs_spec[0], rhs_spec[1], rhs)
  elif batch_group_count > 1:
    rhs = _reshape_axis_out_of(rhs_spec[0], batch_group_count, rhs)
    rhs = _reshape_axis_into(rhs_spec[0], rhs_spec[1], rhs)
    feature_group_count = batch_group_count
  trans_dimension_numbers = ConvDimensionNumbers(out_spec, t_rhs_spec, lhs_spec)
  padding = _conv_general_vjp_lhs_padding(
      np.take(lhs_shape, lhs_sdims), np.take(rhs_shape, rhs_sdims),
      window_strides, np.take(g.shape, out_sdims), padding, lhs_dilation,
      rhs_dilation)
  revd_weights = rev(rhs, rhs_sdims)
  out = conv_general_dilated(
      g, revd_weights, window_strides=lhs_dilation, padding=padding,
      lhs_dilation=window_strides, rhs_dilation=rhs_dilation,
      dimension_numbers=trans_dimension_numbers,
      feature_group_count=feature_group_count,
      batch_group_count=1, precision=precision,
      preferred_element_type=preferred_element_type)
  if batch_group_count > 1:
    out = _reshape_axis_out_of(lhs_spec[1], batch_group_count, out)
    out = _reshape_axis_into(lhs_spec[1], lhs_spec[0], out)
  return out

def _conv_general_dilated_transpose_rhs(
    g, lhs, *, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers: ConvDimensionNumbers, feature_group_count: int,
    batch_group_count: int, lhs_shape, rhs_shape, precision,
    preferred_element_type):
  assert type(dimension_numbers) is ConvDimensionNumbers
  if np.size(g) == 0:
    # Avoids forming degenerate convolutions where the RHS has spatial size 0.
    # Awkwardly, we don't have an aval for the rhs readily available, so instead
    # of returning an ad_util.Zero instance here, representing a symbolic zero
    # value, we instead return a None, which is meant to represent having no
    # cotangent at all (and is thus incorrect for this situation), since the two
    # are treated the same operationally.
    # TODO(mattjj): adjust defbilinear so that the rhs aval is available here
    return None
  lhs_sdims, rhs_sdims, out_sdims = map(_conv_sdims, dimension_numbers)
  lhs_trans, rhs_trans, out_trans = map(_conv_spec_transpose, dimension_numbers)
  assert batch_group_count == 1 or feature_group_count == 1
  if batch_group_count > 1:
    feature_group_count = batch_group_count
    batch_group_count = 1
  elif feature_group_count > 1:
    batch_group_count = feature_group_count
    feature_group_count = 1
  trans_dimension_numbers = ConvDimensionNumbers(lhs_trans, out_trans, rhs_trans)
  padding = _conv_general_vjp_rhs_padding(
      np.take(lhs_shape, lhs_sdims), np.take(rhs_shape, rhs_sdims),
      window_strides, np.take(g.shape, out_sdims), padding, lhs_dilation,
      rhs_dilation)
  return conv_general_dilated(
      lhs, g, window_strides=rhs_dilation, padding=padding,
      lhs_dilation=lhs_dilation, rhs_dilation=window_strides,
      dimension_numbers=trans_dimension_numbers,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count, precision=precision,
      preferred_element_type=preferred_element_type)


def _conv_general_dilated_translation_rule(
    c, lhs, rhs, *, window_strides, padding,
    lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count,
    batch_group_count, precision, expand_complex_convolutions,
    preferred_element_type, **unused_kwargs):
  assert type(dimension_numbers) is ConvDimensionNumbers
  dimension_numbers = _conv_general_proto(dimension_numbers)
  precision_config = _precision_config(precision)
  dtype = c.get_shape(lhs).numpy_dtype()
  if expand_complex_convolutions and np.issubdtype(dtype, np.complexfloating):
    # We use a trick for complex multiplication due to Gauss which uses three
    # multiplications and five additions; instead of the naive method of four
    # multiplications and two additions.
    # https://en.wikipedia.org/wiki/Multiplication_algorithm#Complex_multiplication_algorithm
    #
    # This performance win comes with a trade-off in accuracy; especially in
    # cases when the real and imaginary differ hugely in magnitude. The relative
    # error bound (e.g. 1p-24 in case of float32) would be relative to the
    # maximum of real and imaginary parts of the result instead of being
    # satisfied by the real and imaginary parts independently of each other.
    if preferred_element_type is not None:
      # Convert complex dtype to types used for real and imaginary parts
      assert np.issubdtype(preferred_element_type, np.complexfloating)
      preferred_element_type = xla_client.dtype_to_etype(
          np.float64 if preferred_element_type == np.complex128 else np.float32)

    conv = lambda x, y: xops.ConvGeneralDilated(
        x, y, window_strides, padding, lhs_dilation, rhs_dilation,
        dimension_numbers, feature_group_count, batch_group_count,
        precision_config=precision_config,
        preferred_element_type=preferred_element_type)
    lhs_real, lhs_imag = xops.Real(lhs), xops.Imag(lhs)
    rhs_real, rhs_imag = xops.Real(rhs), xops.Imag(rhs)
    k1 = conv(xops.Add(lhs_real, lhs_imag), rhs_real)
    k2 = conv(lhs_real, xops.Sub(rhs_imag, rhs_real))
    k3 = conv(lhs_imag, xops.Add(rhs_real, rhs_imag))
    return xops.Complex(xops.Sub(k1, k3), xops.Add(k1, k2))

  if preferred_element_type is not None:
    preferred_element_type = xla_client.dtype_to_etype(preferred_element_type)

  return xops.ConvGeneralDilated(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dimension_numbers, feature_group_count, batch_group_count,
      precision_config=precision_config,
      preferred_element_type=preferred_element_type)

def _conv_general_dilated_batch_rule(
    batched_args, batch_dims, *, window_strides, padding,
    lhs_dilation, rhs_dilation, dimension_numbers,
    feature_group_count, batch_group_count, precision,
    preferred_element_type, **unused_kwargs):
  assert batch_group_count == 1 or feature_group_count == 1
  lhs, rhs = batched_args
  lhs_bdim, rhs_bdim = batch_dims
  lhs_spec, rhs_spec, out_spec = dimension_numbers

  if lhs_bdim is not None and rhs_bdim is not None:
    assert lhs.shape[lhs_bdim] == rhs.shape[rhs_bdim]
    if batch_group_count > 1:
      new_lhs = _reshape_axis_into(lhs_bdim, lhs_spec[0], lhs)
      batch_group_count *= lhs.shape[lhs_bdim]
    else:
      new_lhs = _reshape_axis_into(lhs_bdim, lhs_spec[1], lhs)
      feature_group_count *= lhs.shape[lhs_bdim]
    new_rhs = _reshape_axis_into(rhs_bdim, rhs_spec[0], rhs)
    out = conv_general_dilated(
      new_lhs, new_rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dimension_numbers, feature_group_count=feature_group_count,
      batch_group_count=batch_group_count, precision=precision,
      preferred_element_type=preferred_element_type)
    out = _reshape_axis_out_of(out_spec[1], lhs.shape[lhs_bdim], out)
    return out, out_spec[1]

  elif lhs_bdim is not None:
    if batch_group_count == 1:
      new_lhs = _reshape_axis_into(lhs_bdim, lhs_spec[0], lhs)
      out = conv_general_dilated(new_lhs, rhs, window_strides, padding,
                                 lhs_dilation, rhs_dilation, dimension_numbers,
                                 feature_group_count, precision=precision,
                                 preferred_element_type=preferred_element_type)
      out = _reshape_axis_out_of(out_spec[0], lhs.shape[lhs_bdim], out)
      return out, out_spec[0]
    else:
      new_lhs = _reshape_axis_out_of(lhs_spec[0] + int(lhs_bdim <= lhs_spec[0]),
                                     batch_group_count, lhs)
      new_lhs = _reshape_axis_into(lhs_bdim + int(lhs_spec[0] < lhs_bdim),
                                   lhs_spec[0] + 1,
                                   new_lhs)
      new_lhs = _reshape_axis_into(lhs_spec[0], lhs_spec[0], new_lhs)
      out = conv_general_dilated(new_lhs, rhs, window_strides, padding,
                                 lhs_dilation, rhs_dilation, dimension_numbers,
                                 feature_group_count, batch_group_count,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)
      out = _reshape_axis_out_of(out_spec[0], lhs.shape[lhs_bdim], out)
      return out, out_spec[0]

  elif rhs_bdim is not None:
    if feature_group_count == 1 and batch_group_count == 1:
      new_rhs = _reshape_axis_into(rhs_bdim, rhs_spec[0], rhs)
      out = conv_general_dilated(lhs, new_rhs, window_strides, padding,
                                 lhs_dilation, rhs_dilation, dimension_numbers,
                                 feature_group_count, batch_group_count,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)
      out = _reshape_axis_out_of(out_spec[1], rhs.shape[rhs_bdim], out)
      return out, out_spec[1]
    else:
      # groups need to be outermost, so we need to factor them out of the
      # rhs output feature dim, then factor the batch dim into the remaining rhs
      # output feature dim, then put groups back in. We do something
      # similar on the output. An alternative which would require more FLOPs but
      # fewer reshapes would be to broadcast lhs.
      group_count = (feature_group_count if feature_group_count > 1
                     else batch_group_count)
      new_rhs = _reshape_axis_out_of(rhs_spec[0] + int(rhs_bdim <= rhs_spec[0]),
                                     group_count, rhs)
      new_rhs = _reshape_axis_into(rhs_bdim + int(rhs_spec[0] < rhs_bdim),
                                   rhs_spec[0] + 1,
                                   new_rhs)
      new_rhs = _reshape_axis_into(rhs_spec[0], rhs_spec[0], new_rhs)
      out = conv_general_dilated(lhs, new_rhs, window_strides, padding,
                                 lhs_dilation, rhs_dilation, dimension_numbers,
                                 feature_group_count, batch_group_count,
                                 precision=precision,
                                 preferred_element_type=preferred_element_type)
      out = _reshape_axis_out_of(out_spec[1], group_count, out)
      out = _reshape_axis_out_of(out_spec[1] + 1, rhs.shape[rhs_bdim], out)
      out = _reshape_axis_into(out_spec[1], out_spec[1] + 1, out)
      return out, out_spec[1]

def _masked(padded_value, logical_shape, dimensions, value=0):
  """
  Sets all padding to the given value (default is 0) in the given dimensions.
  All values outside the logical shape are considered padding.
  """
  if len(dimensions) == 0:
    return padded_value

  masks = [broadcasted_iota(np.int32, padded_value.shape, d) < logical_shape[d]
           for d in dimensions]
  mask_intersection = masks[0]
  for mask in masks[1:]:
    mask_intersection &= mask
  return select(mask_intersection, padded_value, full_like(padded_value, value))

def _conv_general_dilated_masking_rule(
        padded_vals, logical_shapes, window_strides, padding, lhs_dilation,
        rhs_dilation, dimension_numbers, feature_group_count, batch_group_count,
        lhs_shape, rhs_shape, precision, preferred_element_type):
  lhs, rhs = padded_vals
  logical_lhs_shape, logical_rhs_shape = logical_shapes

  o, i, *window_dimensions = dimension_numbers.rhs_spec
  assert (np.all(np.take(rhs.shape, window_dimensions)
                  == np.take(logical_rhs_shape, window_dimensions))), \
              "Conv filter masking not yet implemented."

  n, c, *padded_dimensions = dimension_numbers.lhs_spec

  return conv_general_dilated(
    _masked(lhs, logical_lhs_shape, padded_dimensions),
    _masked(rhs, logical_rhs_shape, (i,)),
    window_strides=window_strides, padding=padding,
    lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
    dimension_numbers=dimension_numbers,
    feature_group_count=feature_group_count,
    batch_group_count=batch_group_count,
    precision=precision,
    preferred_element_type=preferred_element_type)

conv_general_dilated_p = standard_primitive(
    _conv_general_dilated_shape_rule, _conv_general_dilated_dtype_rule,
    'conv_general_dilated', partial(_conv_general_dilated_translation_rule,
                                    expand_complex_convolutions=False))

# TODO(b/161124619, b/161126248): XLA does not support complex convolution on
# CPU or GPU; on these backends, lower complex convolutions away.
xla.backend_specific_translations['cpu'][conv_general_dilated_p] = partial(
    _conv_general_dilated_translation_rule, expand_complex_convolutions=True)
xla.backend_specific_translations['gpu'][conv_general_dilated_p] = partial(
    _conv_general_dilated_translation_rule, expand_complex_convolutions=True)

ad.defbilinear(conv_general_dilated_p,
               _conv_general_dilated_transpose_lhs,
               _conv_general_dilated_transpose_rhs)
batching.primitive_batchers[conv_general_dilated_p] = \
    _conv_general_dilated_batch_rule
masking.masking_rules[conv_general_dilated_p] = \
  _conv_general_dilated_masking_rule

def _reshape_axis_into(src, dst, x):
  perm = [i for i in range(x.ndim) if i != src]
  perm.insert(dst, src)
  new_shape = list(np.delete(x.shape, src))
  new_shape[dst] *= x.shape[src]
  return reshape(x, new_shape, perm)

def _reshape_axis_out_of(src, size1, x):
  shape = list(x.shape)
  size2, ragged = divmod(shape[src], size1)
  assert not ragged
  shape[src:src+1] = [size1, size2]
  return reshape(x, shape)

def _precision_config(precision):
  if precision is not None:
    config = xla_client.PrecisionConfig()
    if isinstance(precision, tuple):
      config.operand_precision.extend(precision)
    else:
      config.operand_precision.extend((precision, precision))
    return config
  return None


def _dot_general_shape_rule(lhs, rhs, *, dimension_numbers, precision,
                            preferred_element_type: Optional[DType]):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  if not all(np.all(np.greater_equal(d, 0)) and np.all(np.less(d, lhs.ndim))
             for d in (lhs_contracting, lhs_batch)):
    msg = ("dot_general requires lhs dimension numbers to be nonnegative and "
           "less than the number of axes of the lhs value, got "
           f"lhs_batch of {lhs_batch} and lhs_contracting of {lhs_contracting} "
           f"for lhs of rank {lhs.ndim}")
    raise TypeError(msg)
  if not all(np.all(np.greater_equal(d, 0)) and np.all(np.less(d, rhs.ndim))
             for d in (rhs_contracting, rhs_batch)):
    msg = ("dot_general requires rhs dimension numbers to be nonnegative and "
           "less than the number of axes of the rhs value, got "
           f"rhs_batch of {rhs_batch} and rhs_contracting of {rhs_contracting} "
           f"for rhs of rank {rhs.ndim}")
    raise TypeError(msg)
  if len(lhs_batch) != len(rhs_batch):
    msg = ("dot_general requires equal numbers of lhs_batch and rhs_batch "
           "dimensions, got lhs_batch {} and rhs_batch {}.")
    raise TypeError(msg.format(lhs_batch, rhs_batch))
  lhs_contracting_set, lhs_batch_set = set(lhs_contracting), set(lhs_batch)
  rhs_contracting_set, rhs_batch_set = set(rhs_contracting), set(rhs_batch)
  if len(lhs_batch_set) != len(lhs_batch):
    msg = ("dot_general requires lhs batch dimensions to be distinct, got "
           f"lhs_batch {lhs_batch}.")
    raise TypeError(msg)
  if len(rhs_batch_set) != len(rhs_batch):
    msg = ("dot_general requires rhs batch dimensions to be distinct, got "
           f"rhs_batch {rhs_batch}.")
    raise TypeError(msg)
  if len(lhs_contracting_set) != len(lhs_contracting):
    msg = ("dot_general requires lhs contracting dimensions to be distinct, "
           f"got lhs_contracting {lhs_contracting}.")
    raise TypeError(msg)
  if len(rhs_contracting_set) != len(rhs_contracting):
    msg = ("dot_general requires rhs contracting dimensions to be distinct, "
           f"got rhs_contracting {rhs_contracting}.")
    raise TypeError(msg)
  if lhs_contracting_set & lhs_batch_set:
    msg = ("dot_general requires lhs batch dimensions to be disjoint from "
           "contracting dimensions, got lhs_batch {} and lhs_contracting {}.")
    raise TypeError(msg.format(lhs_batch, lhs_contracting))
  if rhs_contracting_set & rhs_batch_set:
    msg = ("dot_general requires rhs batch dimensions to be disjoint from "
           "contracting dimensions, got rhs_batch {} and rhs_contracting {}.")
    raise TypeError(msg.format(rhs_batch, rhs_contracting))
  lhs_batch_shape = np.take(lhs.shape, lhs_batch)
  rhs_batch_shape = np.take(rhs.shape, rhs_batch)
  if not np.all(np.equal(lhs_batch_shape, rhs_batch_shape)):
    msg = ("dot_general requires lhs batch dimensions and rhs batch dimensions "
           "to have the same shape, got {} and {}.")
    raise TypeError(msg.format(lhs_batch_shape, rhs_batch_shape))
  lhs_contracting_shape = np.take(lhs.shape, lhs_contracting)
  rhs_contracting_shape = np.take(rhs.shape, rhs_contracting)
  if not np.all(np.equal(lhs_contracting_shape, rhs_contracting_shape)):
    msg = ("dot_general requires contracting dimensions to have the same "
           "shape, got {} and {}.")
    raise TypeError(msg.format(lhs_contracting_shape, rhs_contracting_shape))

  return _dot_general_shape_computation(lhs.shape, rhs.shape, dimension_numbers)

def _dot_general_shape_computation(lhs_shape, rhs_shape, dimension_numbers):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  batch_shape = tuple(np.take(lhs_shape, lhs_batch))
  lhs_contract_or_batch = tuple(sorted(tuple(lhs_contracting) + tuple(lhs_batch)))
  lhs_tensored_shape = tuple(np.delete(lhs_shape, lhs_contract_or_batch))
  rhs_contract_or_batch = tuple(sorted(tuple(rhs_contracting) + tuple(rhs_batch)))
  rhs_tensored_shape = tuple(np.delete(rhs_shape, rhs_contract_or_batch))
  return batch_shape + lhs_tensored_shape + rhs_tensored_shape

def _dot_general_dtype_rule(lhs, rhs, *, dimension_numbers, precision,
                            preferred_element_type: Optional[DType]):
  input_dtype = naryop_dtype_rule(_input_dtype, [_any, _any], 'dot_general', lhs, rhs)
  if preferred_element_type is None:
    return input_dtype
  _validate_preferred_element_type(input_dtype, preferred_element_type)
  return preferred_element_type

def _dot_general_transpose_lhs(g, y, *, dimension_numbers, precision,
                               preferred_element_type: Optional[DType],
                               swap_ans=False):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim = g.ndim - y.ndim + len(x_batch) + 2 * len(x_contract)
  x_kept = remaining(range(x_ndim), x_contract, x_batch)
  y_kept = remaining(range(y.ndim), y_contract, y_batch)
  if swap_ans:
    ans_batch, ans_y, _ = ranges_like(x_batch, y_kept, x_kept)
  else:
    ans_batch, _, ans_y = ranges_like(x_batch, x_kept, y_kept)
  dims = ((ans_y, y_kept), (ans_batch, y_batch))
  x_contract_sorted_by_y = list(np.take(x_contract, np.argsort(y_contract)))  # type: ignore[arg-type]
  out_axes = np.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)
  return transpose(dot_general(g, y, dims, precision=precision, preferred_element_type=preferred_element_type),
                   tuple(out_axes))

def _dot_general_transpose_rhs(g, x, *, dimension_numbers, precision,
                               preferred_element_type: Optional[DType]):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
  return _dot_general_transpose_lhs(
    g, x, dimension_numbers=swapped_dimension_numbers, precision=precision,
    preferred_element_type=preferred_element_type,
    swap_ans=True)


def _dot_general_batch_rule(batched_args, batch_dims, *, dimension_numbers,
                            precision,
                            preferred_element_type: Optional[DType]):
  lhs, rhs = batched_args
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (lhs.ndim, rhs.ndim), batch_dims, dimension_numbers)
  batched_out = dot_general(lhs, rhs, new_dimension_numbers,
                            precision=precision,
                            preferred_element_type=preferred_element_type)
  return batched_out, result_batch_dim

def _dot_general_batch_dim_nums(ndims, batch_dims, dimension_numbers):
  # there are three kinds of dimensions in a dot_general:
  # - contraction dimensions appear in lhs and rhs but not the result
  # - batch dimensions appear in lhs, rhs, and result
  # - tensor product dimensions appear in the result and one of lhs or rhs
  lhs_ndim, rhs_ndim = ndims
  lbd, rbd = batch_dims
  assert lbd is not None or rbd is not None
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  def bump_dims(dims, b):
    return tuple(np.add(dims, np.greater_equal(dims, b)))

  if lbd is not None and rbd is not None:
    # adding a batch dimension
    lhs_batch = (lbd,) + bump_dims(lhs_batch, lbd)
    rhs_batch = (rbd,) + bump_dims(rhs_batch, rbd)
    lhs_contract = bump_dims(lhs_contract, lbd)
    rhs_contract = bump_dims(rhs_contract, rbd)
    result_batch_dim = 0
  else:
    # adding a tensor product dimension
    if lbd is not None:
      other = tuple(d for d in range(lhs_ndim)
                    if d not in lhs_batch and d not in lhs_contract)
      result_batch_dim = (len(lhs_batch) + sum(np.less(other, lbd)))
      lhs_batch = bump_dims(lhs_batch, lbd)
      lhs_contract = bump_dims(lhs_contract, lbd)
    else:
      other = tuple(d for d in range(rhs_ndim)
                    if d not in rhs_batch and d not in rhs_contract)
      result_batch_dim = (lhs_ndim - len(lhs_contract) +
                          sum(np.less(other, rbd)))
      rhs_batch = bump_dims(rhs_batch, rbd)
      rhs_contract = bump_dims(rhs_contract, rbd)

  new_dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
  return new_dimension_numbers, int(result_batch_dim)

def _dot_general_translation_rule(c, lhs, rhs, *, dimension_numbers, precision,
                                  preferred_element_type: Optional[DType]):
  if preferred_element_type is not None:
    preferred_element_type = xla_client.dtype_to_etype(preferred_element_type)
  return xops.DotGeneral(lhs, rhs,
                         xc.make_dot_dimension_numbers(dimension_numbers),
                         precision_config=_precision_config(precision),
                         preferred_element_type=preferred_element_type)

def _dot_general_masking_rule(padded_vals, logical_shapes, *, dimension_numbers,
                              precision,
                              preferred_element_type: Optional[DType]):
  lhs, rhs = padded_vals
  # Only need to mask off contraction dims of one side - we mask the lhs here
  # but this is arbitrary. Could check the sizes of lhs and rhs and mask
  # whichever is smallest.
  lhs_shape, _ = logical_shapes
  (lhs_contract, _), _ = dimension_numbers
  return dot_general(_masked(lhs, lhs_shape, lhs_contract),
                     rhs, dimension_numbers, precision=precision,
                     preferred_element_type=preferred_element_type)

dot_general_p = standard_primitive(_dot_general_shape_rule,
                                   _dot_general_dtype_rule, 'dot_general',
                                   _dot_general_translation_rule)
ad.defbilinear(dot_general_p,
               _dot_general_transpose_lhs, _dot_general_transpose_rhs)
batching.primitive_batchers[dot_general_p] = _dot_general_batch_rule
masking.masking_rules[dot_general_p] = _dot_general_masking_rule

def _broadcast_shape_rule(operand, sizes):
  _check_shapelike('broadcast', 'sizes', sizes)
  return tuple(sizes) + operand.shape

def _broadcast_batch_rule(batched_args, batch_dims, *, sizes):
  operand, = batched_args
  bdim, = batch_dims
  new_bdim = None if bdim is None else bdim + len(sizes)
  return broadcast(operand, sizes), new_bdim

broadcast_p = standard_primitive(
    _broadcast_shape_rule, _input_dtype, 'broadcast')
ad.deflinear2(broadcast_p, lambda t, _, sizes: [_reduce_sum(t, range(len(sizes)))])
batching.primitive_batchers[broadcast_p] = _broadcast_batch_rule

def _broadcast_in_dim_shape_rule(operand, *, shape, broadcast_dimensions):
  _check_shapelike('broadcast_in_dim', 'shape', shape)
  _check_shapelike('broadcast_in_dim', 'broadcast_dimensions',
                   broadcast_dimensions)
  operand_ndim = np.ndim(operand)
  if operand_ndim != len(broadcast_dimensions):
    msg = ('broadcast_in_dim broadcast_dimensions must have length equal to '
           'operand ndim; got broadcast_dimensions {} for operand ndim {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim))
  if len(shape) < operand_ndim:
    msg = ('broadcast_in_dim target broadcast shape must have equal or higher rank '
           'to the operand shape; got operand ndim {} and target broadcast ndim {}.')
    raise TypeError(msg.format(operand_ndim, len(shape)))
  if not set(broadcast_dimensions).issubset(set(range(len(shape)))):
    msg = ('broadcast_in_dim broadcast_dimensions must be a subset of output '
           'dimensions, got {} for operand ndim {} and shape {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim, shape))
  if not all(core.symbolic_equal_one_of_dim(operand.shape[i],
                                            [1, shape[broadcast_dimensions[i]]])
             for i in range(operand_ndim)):
    msg = (
        "broadcast_in_dim operand dimension sizes must either be 1, or be "
        "equal to their corresponding dimensions in the target broadcast "
        "shape; got operand of shape {}, target broadcast shape {}, "
        "broadcast_dimensions {} ")
    raise TypeError(msg.format(operand.shape, shape, broadcast_dimensions))
  if (len(broadcast_dimensions) != len(set(broadcast_dimensions)) or
      tuple(broadcast_dimensions) != tuple(sorted(broadcast_dimensions))):
    msg = ("broadcast_in_dim broadcast_dimensions must be strictly increasing; "
           "got broadcast_dimensions {}")
    raise TypeError(msg.format(broadcast_dimensions))

  return shape

def _broadcast_in_dim_transpose_rule(ct, operand, *, shape, broadcast_dimensions):
  shape_in = operand.aval.shape
  unit_dimensions = tuple(i for i, s in enumerate(shape_in) if s == 1)
  bdims = tuple(np.delete(broadcast_dimensions, unit_dimensions))
  axes = tuple(np.delete(range(len(shape)), bdims))
  return [expand_dims(_reduce_sum(ct, axes), unit_dimensions)]

def _broadcast_in_dim_batch_rule(batched_args, batch_dims, *, shape,
                                 broadcast_dimensions):
  operand, = batched_args
  bdim, = batch_dims
  new_operand = batching.moveaxis(operand, bdim, 0)
  new_shape = (operand.shape[bdim],) + shape
  new_broadcast_dimensions = (0,) + tuple(np.add(1, broadcast_dimensions))
  return broadcast_in_dim(new_operand, new_shape, new_broadcast_dimensions), 0


broadcast_in_dim_p = standard_primitive(
    _broadcast_in_dim_shape_rule, _input_dtype, 'broadcast_in_dim')
ad.deflinear2(broadcast_in_dim_p, _broadcast_in_dim_transpose_rule)
batching.primitive_batchers[broadcast_in_dim_p] = _broadcast_in_dim_batch_rule


def _clamp_shape_rule(min, operand, max):
  if min.shape and min.shape != operand.shape:
    m = "clamp requires min.shape == operand.shape or min.shape == (), got {}."
    raise TypeError(m.format(min.shape))
  if max.shape and max.shape != operand.shape:
    m = "clamp requires max.shape == operand.shape or max.shape == (), got {}."
    raise TypeError(m.format(max.shape))
  return operand.shape

_clamp_dtype_rule = partial(naryop_dtype_rule, _input_dtype, [_any, _any, _any],
                            'clamp')

clamp_p = standard_primitive(_clamp_shape_rule, _clamp_dtype_rule, 'clamp')
ad.defjvp(clamp_p,
          lambda g, min, operand, max:
          select(bitwise_and(gt(min, operand), lt(min, max)),
                 g, _zeros(operand)),
          lambda g, min, operand, max:
          select(bitwise_and(gt(operand, min), lt(operand, max)),
                 g, _zeros(operand)),
          lambda g, min, operand, max:
          select(lt(max, operand), g, _zeros(operand)))
batching.defbroadcasting(clamp_p)


def _concatenate_shape_rule(*operands, **kwargs):
  dimension = kwargs.pop('dimension')
  if not operands:
    msg = "concatenate expects at least one operand, got 0."
    raise TypeError(msg)
  if not all(isinstance(operand, UnshapedArray) for operand in operands):
    msg = "All objects to concatenate must be arrays, got {}."
    op = next(op for op in operands if not isinstance(op, UnshapedArray))
    raise TypeError(msg.format(type(op)))
  if len({operand.ndim for operand in operands}) != 1:
    msg = "Cannot concatenate arrays with different ranks, got {}."
    raise TypeError(msg.format(", ".join(str(o.ndim) for o in operands)))
  if not 0 <= dimension < operands[0].ndim:
    msg = "concatenate dimension out of bounds: dimension {} for shapes {}."
    raise TypeError(msg.format(dimension, ", ".join([str(o.shape) for o in operands])))
  shapes = [operand.shape[:dimension] + operand.shape[dimension+1:]
            for operand in operands]
  if not shapes[:-1] == shapes[1:]:
    msg = ("Cannot concatenate arrays with shapes that differ in dimensions "
           "other than the one being concatenated: concatenating along "
           "dimension {} for shapes {}.")
    shapes = [operand.shape for operand in operands]
    raise TypeError(msg.format(dimension, ", ".join(map(str, shapes))))

  concat_size = sum(o.shape[dimension] for o in operands)
  ex_shape = operands[0].shape
  return ex_shape[:dimension] + (concat_size,) + ex_shape[dimension+1:]

def _concatenate_dtype_rule(*operands, **kwargs):
  _check_same_dtypes('concatenate', False, *(o.dtype for o in operands))
  return operands[0].dtype

def _concatenate_translation_rule(c, *operands, **kwargs):
  dimension = kwargs.pop('dimension')
  return xops.ConcatInDim(c, operands, dimension)

def _concatenate_transpose_rule(t, *operands, dimension):
  operand_shapes = [o.aval.shape if ad.is_undefined_primal(o) else o.shape
                    for o in operands]
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(o.aval) if ad.is_undefined_primal(o) else None
            for o in operands]
  else:
    limit_points = np.cumsum([shape[dimension] for shape in operand_shapes])
    starts = np.zeros((len(operands), t.ndim), dtype=int)
    starts[1:, dimension] = limit_points[:-1]
    limits = np.tile(t.shape, (len(operands), 1))
    limits[:, dimension] = limit_points

    return [slice(t, start, limit) if ad.is_undefined_primal(o) else None
            for o, start, limit in zip(operands, starts, limits)]

def _concatenate_batch_rule(batched_args, batch_dims, *, dimension):
  size = next(op.shape[bdim] for op, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  operands = [batching.moveaxis(op, bdim, 0) if bdim is not None
              else broadcast(op, (size,))
              for op, bdim in zip(batched_args, batch_dims)]
  return concatenate(operands, dimension + 1), 0

# The concatenate_p masking rule requires use of a while-loop construct and so
# is defined in lax_control_flow.py

concatenate_p = standard_primitive(
    _concatenate_shape_rule, _concatenate_dtype_rule, 'concatenate',
    _concatenate_translation_rule)
ad.deflinear2(concatenate_p, _concatenate_transpose_rule)
ad.primitive_transposes[concatenate_p] = _concatenate_transpose_rule
batching.primitive_batchers[concatenate_p] = _concatenate_batch_rule


def _pad_dtype_rule(operand, padding_value, *, padding_config):
  if operand.dtype != padding_value.dtype:
    msg = "pad operand and padding_value must be same dtype: got {} and {}."
    raise TypeError(msg.format(operand.dtype, padding_value.dtype))

  return _input_dtype(operand, padding_value)

def _pad_shape_rule(operand, padding_value, *, padding_config):
  del padding_value
  op_shape = np.shape(operand)
  if not len(padding_config) == np.ndim(operand):
    raise ValueError("length of padding_config must equal the number of axes "
                     f"of operand, got padding_config {padding_config} "
                     f"for operand shape {op_shape}")
  if not all(i >= 0 for _, _, i in padding_config):
    raise ValueError("interior padding in padding_config must be nonnegative, "
                     f"got padding_config {padding_config}")
  result = tuple(core.sum_dim(l, h, core.dilate_dim(d, i + 1))
                 for (l, h, i), d in zip(padding_config, op_shape))
  if not all(core.greater_equal_dim(d, 0) for d in result):
    msg = (f"Dimension size after padding is not at least 0, "
           f"got result shape {result}, for padding_config {padding_config}"
           f" and operand shape {op_shape}")
    raise ValueError(msg)
  return result

def _pad_transpose(t, operand, padding_value, *, padding_config):
  if type(t) is ad_util.Zero:
    t_operand = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    t_padv = ad_util.Zero(padding_value.aval) if ad.is_undefined_primal(padding_value) else None
  else:
    lo, hi, interior = zip(*padding_config)
    total = lambda x: _reduce_sum(x, list(range(t.ndim)))

    def t_op():
      unpad_config = safe_zip(np.negative(lo), np.negative(hi),
                              np.zeros_like(interior))
      unpadded = pad(t, np.array(0., t.dtype), unpad_config)
      return slice(unpadded, np.zeros_like(lo), unpadded.shape, np.add(interior, 1))

    t_operand = t_op() if ad.is_undefined_primal(operand) else None
    t_padv = sub(total(t), total(t_operand)) if ad.is_undefined_primal(padding_value) else None
  return [t_operand, t_padv]

def _pad_batch_rule(batched_args, batch_dims, *, padding_config):
  operand, padding_value = batched_args
  operand_bdim, padding_value_bdim = batch_dims
  if operand_bdim is None:
    operand_bdim = 0
    operand = broadcast(operand, (padding_value.shape[padding_value_bdim],))

  padding_config = list(padding_config)
  padding_config.insert(operand_bdim, (0, 0, 0))
  if padding_value_bdim is None:
    return pad(operand, padding_value, padding_config), operand_bdim

  assert padding_value_bdim == 0, padding_value_bdim

  x = pad(operand, _zero(operand), padding_config)
  mask = pad(full_like(operand, True, np.bool_), False, padding_config)
  broadcasted_padding = broadcast_in_dim(padding_value, x.shape,
                                         (operand_bdim,))
  return select(mask, x, broadcasted_padding), operand_bdim

def _pad_translation_rule(c, operand, padding_value, *, padding_config):
  return xops.Pad(operand, padding_value,
                  xc.make_padding_config(padding_config))

def _pad_masking_rule(padded_vals, logical_shapes, padding_config):
  operand, padding_value = padded_vals
  shape, _ = logical_shapes

  out = pad(operand, padding_value, padding_config)
  out_shape = [lo + shape[i] * (interior + 1)
               for i, (lo, hi, interior) in enumerate(padding_config)]
  padded_dims = [i for i, config in enumerate(padding_config)
                 if config != (0, 0, 0)]
  return _masked(out, out_shape, padded_dims, padding_value)

pad_p = standard_primitive(_pad_shape_rule, _pad_dtype_rule, 'pad',
                           translation_rule=_pad_translation_rule)
ad.deflinear2(pad_p, _pad_transpose)
batching.primitive_batchers[pad_p] = _pad_batch_rule
masking.masking_rules[pad_p] = _pad_masking_rule


# The squeeze primitive exists for the benefit of masking and other
# transformations that need to keep track of axis identity.
# For example, consider reshaping a 2D array with shape (1, N) into a 1D array
# with shape (N,). This results in the following JAXpr:
#   reshape[ dimension=None new_sizes=(N,) ]
# For N > 1, we can match up the output array axis with the second axis of the
# input. But for N = 1, it is not clear how axes match up: all we know from the
# JAXpr is that we are reshaping from (1, 1) to (1,).
# In constrast, squeeze[ dimensions=(0,) ] is unambiguous.

def squeeze(array: Array, dimensions: Tuple[int, ...]) -> Array:
  """Squeeze any number of size 1 dimensions from an array."""
  ndim = np.ndim(array)
  dimensions = tuple(sorted(canonicalize_axis(i, ndim) for i in dimensions))
  if not dimensions:
    return array
  return squeeze_p.bind(array, dimensions=dimensions)

def _squeeze_dtype_rule(operand, *, dimensions):
  return operand.dtype

def _squeeze_shape_rule(operand, *, dimensions):
  return _compute_squeeze_shape(np.shape(operand), dimensions)

def _compute_squeeze_shape(shape, dimensions):
  dims_set = set(dimensions)
  if len(dims_set) != len(dimensions):
    raise ValueError(f"dimensions are not unique: {dimensions}")
  if not all(0 <= d < len(shape) for d in dims_set):
    raise ValueError(f"dimensions outside range [0, ndim): {dimensions}")
  if any(shape[d] != 1 for d in dimensions):
    raise ValueError(
        "cannot select an axis to squeeze out which has size not equal to "
        f"one, got shape={shape} and dimensions={dimensions}")
  return tuple(s for i, s in enumerate(shape) if i not in dims_set)

def _squeeze_translation_rule(c, arg, *, dimensions):
  new_shape = _compute_squeeze_shape(c.get_shape(arg).dimensions(), dimensions)
  return xops.Reshape(arg, new_shape)

def _squeeze_transpose_rule(t, operand, *, dimensions):
  assert ad.is_undefined_primal(operand)
  return [expand_dims(t, dimensions)]

def _squeeze_batch_rule(batched_args, batch_dims, *, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  operand = batching.moveaxis(operand, bdim, 0)
  dimensions = tuple(np.add(1, dimensions))
  return squeeze(operand, dimensions=dimensions), 0

squeeze_p = standard_primitive(_squeeze_shape_rule, _squeeze_dtype_rule,
                               'squeeze', _squeeze_translation_rule)
ad.deflinear2(squeeze_p, _squeeze_transpose_rule)
batching.primitive_batchers[squeeze_p] = _squeeze_batch_rule


def expand_dims(array: Array, dimensions: Tuple[int, ...]) -> Array:
  """Insert any number of size 1 dimensions into an array."""
  ndim_out = np.ndim(array) + len(dimensions)
  dims_set = frozenset(canonicalize_axis(i, ndim_out) for i in dimensions)
  result_shape = list(np.shape(array))
  for i in sorted(dims_set):
    result_shape.insert(i, 1)
  broadcast_dims = [i for i in range(ndim_out) if i not in dims_set]
  return broadcast_in_dim(array, result_shape, broadcast_dims)


def _is_singleton_reshape(old, new):
  # A singleton reshape is one where only singleton dimensions are added. We
  # want to detect them because they can be expressed as (lazy) broadcasts.
  old, new = iter(old), iter(new)
  d1, d2 = next(old, None), next(new, None)
  bcast_dims = []
  i = 0
  while True:
    if d1 is d2 is None:
      return bcast_dims
    elif d1 == d2:
      bcast_dims.append(i)
      i += 1
      d1, d2 = next(old, None), next(new, None)
    elif d2 == 1:
      i += 1
      d2 = next(new, None)
    else:
      return None

def _reshape_shape_rule(operand, *, new_sizes, dimensions):
  if not all(core.greater_equal_dim(d, 0) for d in new_sizes):
    msg = 'reshape new_sizes must all be positive, got {}.'
    raise TypeError(msg.format(new_sizes))
  if not core.same_shape_sizes(np.shape(operand), new_sizes):
    msg = 'reshape total size must be unchanged, got new_sizes {} for shape {}.'
    raise TypeError(msg.format(new_sizes, np.shape(operand)))
  if dimensions is not None:
    if set(dimensions) != set(range(np.ndim(operand))):
      msg = ('reshape dimensions must be a permutation of operand dimensions, '
             'got dimensions {} for shape {}.')
      raise TypeError(msg.format(dimensions, np.shape(operand)))
  return tuple(new_sizes)

def _reshape_dtype_rule(operand, *, new_sizes, dimensions):
  return operand.dtype

def _reshape_translation_rule(c, operand, *, new_sizes, dimensions):
  if dimensions is None:
    return xops.Reshape(operand, new_sizes)
  else:
    return xops.Reshape(operand, dimensions, new_sizes)

def _reshape_transpose_rule(t, operand, *, new_sizes, dimensions):
  assert ad.is_undefined_primal(operand)
  if dimensions is None:
    return [reshape(t, operand.aval.shape)]
  else:
    return [transpose(reshape(t, np.take(operand.aval.shape, dimensions)),
                      np.argsort(dimensions))]

def _reshape_batch_rule(batched_args, batch_dims, *, new_sizes, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  operand = batching.moveaxis(operand, bdim, 0)
  if dimensions is not None:
    dimensions = (0,) + tuple(np.add(1, dimensions))
  return reshape(operand, operand.shape[:1] + new_sizes, dimensions), 0

def _reshape_masking_rule(padded_args, logical_shapes, polymorphic_shapes,
                          new_sizes, dimensions):
  operand, = padded_args
  old_shape, = polymorphic_shapes
  def is_poly(size): return type(size) is masking.Poly and not size.is_constant
  def merge_const_sizes(shape):
    """Merges all nonpolymorphic sizes into the previous polymorphic size."""
    poly_dims = [i for i, size in enumerate(shape) if is_poly(size)]
    return [prod(shape[start:stop])
            for start, stop in zip([0] + poly_dims, poly_dims + [len(shape)])]
  if merge_const_sizes(old_shape) != merge_const_sizes(new_sizes):
    raise NotImplementedError(
      "Reshape on padded dimensions causing fragmentation is not supported.")

  return reshape(operand,
                 new_sizes=masking.padded_shape_as_value(new_sizes),
                 dimensions=dimensions)

reshape_p = standard_primitive(_reshape_shape_rule, _reshape_dtype_rule,
                               'reshape', _reshape_translation_rule)
ad.deflinear2(reshape_p, _reshape_transpose_rule)
batching.primitive_batchers[reshape_p] = _reshape_batch_rule
masking.masking_rules[reshape_p] = _reshape_masking_rule

def _rev_shape_rule(operand, *, dimensions):
  _check_shapelike('rev', 'dimensions', dimensions)
  if len(set(dimensions)) != len(dimensions):
    msg = 'rev dimensions must be unique, got {}.'
    raise TypeError(msg.format(dimensions))
  if dimensions and not _max(dimensions) < operand.ndim:
    msg = ('rev dimensions must all be less than operand ndim, got dimensions '
           '{} for operand ndim {}.')
    raise TypeError(msg.format(dimensions, operand.ndim))
  return operand.shape

def _rev_batch_rule(batched_args, batch_dims, *, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  new_dimensions = [i + 1 if i >= bdim else i for i in dimensions]
  return rev(operand, new_dimensions), bdim

rev_p = standard_primitive(_rev_shape_rule, _input_dtype, 'rev')
ad.deflinear2(rev_p, lambda t, _, dimensions: [rev(t, dimensions)])
batching.primitive_batchers[rev_p] = _rev_batch_rule


def _transpose_shape_rule(operand, *, permutation):
  if not isinstance(permutation, (tuple, list, np.ndarray)):
    msg = "transpose permutation must be a tuple/list/ndarray, got {}."
    raise TypeError(msg.format(type(permutation)))
  if tuple(sorted(permutation)) != tuple(range(operand.ndim)):
    msg = ("transpose permutation isn't a permutation of operand dimensions, "
           "got permutation {} for operand shape {}.")
    raise TypeError(msg.format(permutation, operand.shape))
  return tuple(np.take(operand.shape, permutation))

def _transpose_batch_rule(batched_args, batch_dims, *, permutation):
  operand, = batched_args
  bdim, = batch_dims
  perm = (bdim,) + tuple(i if i < bdim else i+1 for i in permutation)
  return transpose(operand, perm), 0

def _transpose_masking_rule(padded_vals, logical_shapes, permutation):
  return transpose(*padded_vals, permutation=permutation)

transpose_p = standard_primitive(_transpose_shape_rule, _input_dtype,
                                 'transpose')
ad.deflinear2(transpose_p,
              lambda t, _, permutation: [transpose(t, np.argsort(permutation))])  # type: ignore[arg-type]
batching.primitive_batchers[transpose_p] = _transpose_batch_rule
masking.masking_rules[transpose_p] = _transpose_masking_rule


def _select_shape_rule(pred, on_true, on_false):
  if on_true.shape != on_false.shape:
    msg = "select on_true and on_false must have the same shape, got {} and {}."
    raise TypeError(msg.format(on_true.shape, on_false.shape))
  if pred.shape and pred.shape != on_true.shape:
    msg = ("select pred must be scalar or have the same shape as on_true and "
           "on_false, got pred shape {} for on_true and on_false of shape {}.")
    raise TypeError(msg.format(pred.shape, on_true.shape))
  return on_true.shape

def _select_dtype_rule(pred, on_true, on_false):
  _check_same_dtypes("select", False, on_true.dtype, on_false.dtype)
  if not dtypes.issubdtype(pred.dtype, np.bool_):
    msg = "select pred must be boolean type, got {}."
    raise TypeError(msg.format(pred.dtype))
  return on_true.dtype

def _select_transpose_rule(t, pred, on_true, on_false):
  assert not ad.is_undefined_primal(pred)
  if type(t) is ad_util.Zero:
    return [None,
            ad_util.Zero(on_true.aval) if ad.is_undefined_primal(on_true) else None,
            ad_util.Zero(on_false.aval) if ad.is_undefined_primal(on_false) else None]
  else:
    zeros = full_like(t, 0)
    return [None,
            select(pred, t, zeros) if ad.is_undefined_primal(on_true) else None,
            select(pred, zeros, t) if ad.is_undefined_primal(on_false) else None]

def _select_batch_rule(batched_args, batch_dims, **unused_kwargs):
  pred, on_true, on_false, = batched_args
  pred_bdim, ot_bdim, of_bdim = batch_dims
  size = next(x.shape[i] for x, i in zip(batched_args, batch_dims)
              if i is not None)

  # avoid transposes and some broadcasts in special cases
  if pred_bdim == ot_bdim == of_bdim:
    if np.shape(pred) == np.shape(on_true):
      return select(pred, on_true, on_false), pred_bdim
    else:
      # vmapped function had a scalar pred with nonscalar args
      assert np.ndim(pred) == 1
      pred = broadcast_in_dim(pred, on_true.shape, [pred_bdim])
      return select(pred, on_true, on_false), pred_bdim
  elif np.ndim(pred) == 0 and ot_bdim is not None and of_bdim is not None:
    if ot_bdim == of_bdim:
      return select(pred, on_true, on_false), ot_bdim
    elif np.shape(on_true) == np.shape(on_false):
      on_false = batching.moveaxis(on_false, of_bdim, ot_bdim)
      return select(pred, on_true, on_false), ot_bdim

  pred = batching.bdim_at_front(pred, pred_bdim, size) if np.shape(pred) else pred
  if not () == np.shape(on_true) == np.shape(on_false):
    on_true = batching.bdim_at_front(on_true, ot_bdim, size)
    on_false = batching.bdim_at_front(on_false, of_bdim, size)
  assert np.shape(on_true) == np.shape(on_false)
  if 0 < np.ndim(pred) < np.ndim(on_true):
    # vmapped function had a scalar pred with nonscalar args
    assert np.ndim(pred) == 1
    pred = broadcast_in_dim(pred, on_true.shape, [0])
  if np.ndim(pred) > np.ndim(on_true):
    assert np.ndim(on_true) == 0
    on_true = broadcast(on_true, pred.shape)
    on_false = broadcast(on_false, pred.shape)
  return select(pred, on_true, on_false), 0

def _select_masking_rule(padded_vals, logical_shapes):
  pred_shape, true_shape, false_shape = [
      masking.padded_shape_as_value(val.shape) for val in padded_vals]
  assert np.array_equal(pred_shape, true_shape)
  assert np.array_equal(pred_shape, false_shape)
  return select(*padded_vals)

def _select_jvp(primals, tangents):
  pred, on_true, on_false = primals
  _, on_true_dot, on_false_dot = tangents
  out = select(pred, on_true, on_false)
  if type(on_true_dot) is ad_util.Zero:
    if type(on_false_dot) is ad_util.Zero:
      out_dot = ad_util.Zero(on_true_dot.aval)
    else:
      out_dot = select(pred, _zeros(on_false_dot), on_false_dot)
  elif type(on_false_dot) is ad_util.Zero:
    out_dot = select(pred, on_true_dot, _zeros(on_true_dot))
  else:
    out_dot = select(pred, on_true_dot, on_false_dot)
  return out, out_dot

select_p = standard_primitive(_select_shape_rule, _select_dtype_rule, 'select',
                              weak_type_rule=_argnum_weak_type(1, 2))
ad.primitive_jvps[select_p] = _select_jvp
ad.primitive_transposes[select_p] = _select_transpose_rule
batching.primitive_batchers[select_p] = _select_batch_rule
masking.masking_rules[select_p] = _select_masking_rule


def _slice_shape_rule(operand, *, start_indices, limit_indices, strides):
  _check_shapelike("slice", "start_indices", start_indices)
  _check_shapelike("slice", "limit_indices", limit_indices)
  if operand.ndim != len(start_indices):
    msg = ("slice start_indices must have length equal to the number of "
           "dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(limit_indices):
    msg = ("slice limit_indices must have the same length as start_indices, "
           "got start_indices {} and limit_indices {}.")
    raise TypeError(msg.format(start_indices, limit_indices))
  if not core.greater_equal_shape(operand.shape, limit_indices):
    msg = ("slice limit_indices must be less than or equal to operand shape, "
           "got limit_indices {} for operand shape {}.")
    raise TypeError(msg.format(limit_indices, operand.shape))
  if not all(core.greater_equal_dim(si, 0) for si in start_indices):
    msg = ("slice start_indices must be greater than or equal to zero, "
           "got start_indices of {}.")
    raise TypeError(msg.format(start_indices))
  if not core.greater_equal_shape(limit_indices, start_indices):
    msg = ("slice limit_indices must be greater than or equal to start_indices,"
           " got start_indices {} and limit_indices {}.")
    raise TypeError(msg.format(start_indices, limit_indices))
  if strides is None:
    strides = np.ones(operand.ndim, np.int32)
  else:
    _check_shapelike("slice", "strides", strides)
    if len(strides) != operand.ndim:
      msg = ("slice strides must have length equal to the number of dimensions "
             "of the operand, got strides {} for operand shape {}.")
      raise TypeError(msg.format(strides, operand.shape))
    if not core.greater_equal_shape(strides, (0,) * len(strides)):
      msg = "slice strides must be positive, got {}"
      raise TypeError(msg.format(strides))

  diff = core.diff_shape(limit_indices, start_indices)
  return core.stride_shape(diff, (1,) * len(diff), strides)

def _slice_translation_rule(c, operand, *, start_indices, limit_indices,
                            strides):
  return xops.Slice(operand, start_indices, limit_indices,
                    strides or [1] * len(start_indices))

def _slice_transpose_rule(t, operand, *, start_indices, limit_indices, strides):
  assert ad.is_undefined_primal(operand)
  operand_shape = operand.aval.shape
  if strides is None or np.all(np.equal(strides, 1)):
    pads = zip(start_indices, np.subtract(operand_shape, limit_indices),
               (0,) * len(start_indices))
  else:
    real_limits = np.add(
      start_indices,
      np.where(np.array(t.shape) == 0, 0,
               np.add(1, np.multiply(np.subtract(t.shape, 1), strides))))
    pads = safe_zip(start_indices, np.subtract(operand_shape, real_limits),
                    np.subtract(strides, 1))
  result = pad(t, _const(t, 0), pads)
  assert result.shape == operand_shape, (
    f"result.shape={result.shape} operand_shape={operand_shape}")
  return [result]


def _slice_batching_rule(batched_args, batch_dims, *, start_indices,
                         limit_indices, strides):
  operand, = batched_args
  bdim, = batch_dims

  new_start_indices = list(start_indices)
  new_start_indices.insert(bdim, 0)

  new_limit_indices = list(limit_indices)
  new_limit_indices.insert(bdim, operand.shape[bdim])

  if strides is None:
    new_strides = None
  else:
    new_strides = list(strides)
    new_strides.insert(bdim, 1)

  out = slice(operand, new_start_indices, new_limit_indices, new_strides)
  return out, bdim

def _slice_masking_rule(
    padded_vals, logical_shapes, start_indices, limit_indices, strides):
  operand, = padded_vals
  strides = masking.padded_shape_as_value(strides) if strides else None
  return slice(operand,
               start_indices=masking.padded_shape_as_value(start_indices),
               limit_indices=masking.padded_shape_as_value(limit_indices),
               strides=strides)

slice_p = standard_primitive(_slice_shape_rule, _input_dtype, 'slice',
                             _slice_translation_rule)
ad.deflinear2(slice_p, _slice_transpose_rule)
batching.primitive_batchers[slice_p] = _slice_batching_rule
masking.masking_rules[slice_p] = _slice_masking_rule


def _dynamic_slice_shape_rule(operand, *start_indices, slice_sizes):
  if operand.ndim != len(start_indices):
    msg = ("dynamic_slice start_indices must have length equal to the number "
           "of dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(slice_sizes):
    msg = ("dynamic_slice slice_sizes must have the same length as "
           "start_indices, got start_indices length {} and slice_sizes {}.")
    raise TypeError(msg.format(len(start_indices), slice_sizes))
  if not core.greater_equal_shape(operand.shape, slice_sizes):
    msg = ("slice slice_sizes must be less than or equal to operand shape, "
           "got slice_sizes {} for operand shape {}.")
    raise TypeError(msg.format(slice_sizes, operand.shape))
  if not all(core.greater_equal_dim(ssz, 0) for ssz in slice_sizes):
    msg = ("slice slice_sizes must be greater than or equal to zero, "
           "got slice_sizes of {}.")
    raise TypeError(msg.format(slice_sizes))
  return tuple(slice_sizes)

def _dynamic_slice_dtype_rule(operand, *start_indices, slice_sizes):
  if any(i.dtype != start_indices[0].dtype or
         not dtypes.issubdtype(i.dtype, np.integer) for i in start_indices):
    msg = ("index arguments to dynamic_slice must be integers of the same "
           "type, got: {}")
    raise TypeError(msg.format(", ".join(i.dtype.name for i in start_indices)))
  return operand.dtype

def _dynamic_slice_translation_rule(c, operand, *start_indices, slice_sizes):
  return xops.DynamicSlice(operand, start_indices, slice_sizes)

def _dynamic_slice_jvp(primals, tangents, *, slice_sizes):
  tangent_out = tangents[0]
  if type(tangent_out) is not ad_util.Zero:
    tangent_out = dynamic_slice(tangent_out, primals[1:], slice_sizes)
  return dynamic_slice(primals[0], primals[1:], slice_sizes), tangent_out

def _dynamic_slice_transpose_rule(t, operand, *start_indices, slice_sizes):
  assert ad.is_undefined_primal(operand)
  assert all(not ad.is_undefined_primal(s) for s in start_indices)
  operand_shape, operand_dtype = operand.aval.shape, operand.aval.dtype
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)] + [None] * len(start_indices)
  else:
    zeros = full(operand_shape, 0, operand_dtype)
    return ([dynamic_update_slice(zeros, t, start_indices)] +
            [None] * len(start_indices))

def _batch_dynamic_slice_indices(indices, bdims):
  if len(indices) == 0:
    return np.array([], 'int32'), None
  empty_marker = object()
  size = next((x.shape[i] for x, i in zip(indices, bdims) if i is not None),
              empty_marker)
  if size is empty_marker:
    return concatenate([broadcast(i, (1,)) for i in indices], 0), None
  indices = concatenate(
    [broadcast_in_dim(x, (size, 1),
                      broadcast_dimensions=((0,) if i is not None else ()))
     for x, i in zip(indices, bdims)],
    dimension=1)
  return indices, 0

def _dynamic_slice_batching_rule(batched_args, batch_dims, *, slice_sizes):
  # A dynamic slice is a special case of gather; we can delegate to the gather
  # batching rule.
  # TODO(phawkins): consider removing dynamic_slice entirely and using gather
  # always.
  operand, *start_indices = batched_args
  operand_bd, *start_idx_bds = batch_dims
  operand_shape = (operand.shape if operand_bd is batching.not_mapped
                   else tuple(np.delete(operand.shape, operand_bd)))
  dims = tuple(range(len(operand_shape)))
  dnums = GatherDimensionNumbers(offset_dims=dims, collapsed_slice_dims=(),
                                 start_index_map=dims)
  index, index_bdim = _batch_dynamic_slice_indices(start_indices, start_idx_bds)
  return _gather_batching_rule(
    [operand, index], [operand_bd, index_bdim], dimension_numbers=dnums,
    slice_sizes=slice_sizes)


dynamic_slice_p = standard_primitive(
    _dynamic_slice_shape_rule, _dynamic_slice_dtype_rule, 'dynamic_slice',
    _dynamic_slice_translation_rule, weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[dynamic_slice_p] = _dynamic_slice_jvp  # TODO
ad.primitive_transposes[dynamic_slice_p] = _dynamic_slice_transpose_rule
batching.primitive_batchers[dynamic_slice_p] = _dynamic_slice_batching_rule


def _dynamic_update_slice_shape_rule(operand, update, *start_indices):
  if operand.ndim != update.ndim:
    msg = ("dynamic_update_slice update must have the same rank as operand, "
           "got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  if operand.ndim != len(start_indices):
    msg = ("dynamic_update_slice start_indices must have length equal to the "
           "rank of operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if not np.all(np.less_equal(update.shape, operand.shape)):
    msg = ("dynamic_update_slice update shape must be smaller than operand "
           "shape, got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  return operand.shape

def _dynamic_update_slice_dtype_rule(operand, update, *start_indices):
  _check_same_dtypes("dynamic_update_slice", False, operand.dtype, update.dtype)
  if any(i.dtype != start_indices[0].dtype or
         not dtypes.issubdtype(i.dtype, np.integer) for i in start_indices):
    msg = ("index arguments to dynamic_update_slice must be integers of the "
           "same type, got {}")
    raise TypeError(msg.format(", ".join(i.dtype.name for i in start_indices)))
  return operand.dtype

def _dynamic_update_slice_jvp(primals, tangents):
  operand, update = primals[:2]
  start_indices = primals[2:]
  g_operand, g_update = tangents[:2]
  val_out = dynamic_update_slice(operand, update, start_indices)
  if type(g_operand) is ad_util.Zero and type(g_update) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_update = ad.instantiate_zeros(g_update)
    tangent_out = dynamic_update_slice(g_operand, g_update, start_indices)
  return val_out, tangent_out

def _dynamic_update_slice_transpose_rule(t, operand, update, *start_indices):
  assert all(not ad.is_undefined_primal(x) for x in start_indices)
  if ad.is_undefined_primal(update):
    update_shape = update.aval.shape
  else:
    update_shape = update.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(update.aval) if ad.is_undefined_primal(update) else None
  else:
    dus = dynamic_update_slice
    ds = dynamic_slice
    zeros = _zeros(t, shape=update_shape)
    operand_t = dus(t, zeros, start_indices) if ad.is_undefined_primal(operand) else None
    update_t = ds(t, start_indices, update_shape) if ad.is_undefined_primal(update) else None
  return [operand_t, update_t] + [None] * len(start_indices)

def _dynamic_update_slice_translation_rule(c, operand, update, *start_indices):
  return xops.DynamicUpdateSlice(operand, update, start_indices)

def _dynamic_update_slice_batching_rule(batched_args, batch_dims):
  # A dynamic update slice is a special case of scatter; we can delegate to the
  # scatter batching rule.
  # TODO(phawkins): consider removing dynamic_update_slice entirely and using
  # scatter always.
  operand, update, *start_idx = batched_args
  operand_bd, update_bd, *start_idx_bd = batch_dims
  update_shape = (np.shape(update) if update_bd is batching.not_mapped
                  else tuple(np.delete(np.shape(update), update_bd)))
  dims = tuple(range(len(update_shape)))
  dnums = ScatterDimensionNumbers(update_window_dims=dims,
                                  inserted_window_dims=(),
                                  scatter_dims_to_operand_dims=dims)
  index, index_bdim = _batch_dynamic_slice_indices(start_idx, start_idx_bd)
  return _scatter_batching_rule(
    scatter, (operand, index, update), (operand_bd, index_bdim, update_bd),
    update_jaxpr=None, update_consts=None, dimension_numbers=dnums,
    indices_are_sorted=True, unique_indices=True)


dynamic_update_slice_p = standard_primitive(
    _dynamic_update_slice_shape_rule, _dynamic_update_slice_dtype_rule,
    'dynamic_update_slice', _dynamic_update_slice_translation_rule)
ad.primitive_jvps[dynamic_update_slice_p] = _dynamic_update_slice_jvp
ad.primitive_transposes[dynamic_update_slice_p] = \
    _dynamic_update_slice_transpose_rule
batching.primitive_batchers[dynamic_update_slice_p] = \
    _dynamic_update_slice_batching_rule


def _gather_dimensions_proto(indices_shape, dimension_numbers):
  assert type(dimension_numbers) is GatherDimensionNumbers
  proto = xla_client.GatherDimensionNumbers()
  proto.offset_dims.extend(dimension_numbers.offset_dims)
  proto.collapsed_slice_dims.extend(dimension_numbers.collapsed_slice_dims)
  proto.start_index_map.extend(dimension_numbers.start_index_map)
  assert indices_shape.rank() > 0
  proto.index_vector_dim = indices_shape.rank() - 1
  return proto

def _gather_dtype_rule(operand, start_indices, **kwargs):
  if not dtypes.issubdtype(start_indices.dtype, np.integer):
    raise ValueError("start_indices must have an integer type")
  return dtypes.canonicalize_dtype(operand.dtype)

_rank = lambda arr: len(arr.shape)

def _is_sorted(dims, op_name, name):
  for i in range(1, len(dims)):
    if dims[i] < dims[i - 1]:
      raise TypeError(f"{name} in {op_name} op must be sorted; got {dims}")

def _sorted_dims_in_range(dims, rank, op_name, name):
  if len(dims) == 0:
    return
  invalid_dim = None
  if dims[0] < 0:
    invalid_dim = dims[0]
  elif dims[-1] >= rank:
    invalid_dim = dims[-1]
  if invalid_dim:
    raise TypeError(f"Invalid {name} set in {op_name} op; valid range is "
                    f"[0, {rank}); got: {invalid_dim}.")

def _no_duplicate_dims(dims, op_name, name):
  if len(set(dims)) != len(dims):
    raise TypeError(f"{name} in {op_name} op must not repeat; got: {dims}.")

def _gather_shape_rule(operand, start_indices, *, dimension_numbers,
                       slice_sizes):
  """Validates the well-formedness of the arguments to Gather.

  The code implements the checks based on the detailed operation semantics of
  XLA's `Gather <https://www.tensorflow.org/xla/operation_semantics#gather>`_
  operator and following the outline of the implementation of
  ShapeInference::InferGatherShape in TensorFlow.
  """

  offset_dims = dimension_numbers.offset_dims
  collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
  start_index_map = dimension_numbers.start_index_map

  # Note: in JAX, index_vector_dim is always computed as below, cf. the
  # documentation of the GatherDimensionNumbers class.
  index_vector_dim = _rank(start_indices) - 1

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if _rank(start_indices) < index_vector_dim or index_vector_dim < 0:
    raise TypeError(f"Gather index leaf dimension must be within [0, rank("
                    f"start_indices) + 1). rank(start_indices) is "
                    f"{_rank(start_indices)} and gather index leaf dimension "
                    f"is {index_vector_dim}.")

  expanded_start_indices_shape = list(start_indices.shape)

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if len(expanded_start_indices_shape) == index_vector_dim:
    expanded_start_indices_shape.append(1)

  # Start ValidateGatherDimensions
  # In the error messages output by XLA, "offset_dims" is called "Output window
  # dimensions" in error messages. For consistency's sake, our error messages
  # stick to "offset_dims".
  _is_sorted(offset_dims, "gather", "offset_dims")
  _no_duplicate_dims(offset_dims, "gather", "offset_dims")

  output_offset_dim_count = len(offset_dims)
  output_shape_rank = len(offset_dims) + _rank(start_indices) - 1

  for i in range(output_offset_dim_count):
    offset_dim = offset_dims[i]
    if offset_dim < 0 or offset_dim >= output_shape_rank:
      raise TypeError(f"Offset dimension {i} in gather op is out of bounds; "
                      f"got {offset_dim}, but should have been in "
                      f"[0, {output_shape_rank})")

  if len(start_index_map) != start_indices.shape[index_vector_dim]:
    raise TypeError(f"Gather op has {len(start_index_map)} elements in "
                    f"start_index_map and the bound of dimension "
                    f"index_vector_dim={index_vector_dim} of start_indices is "
                    f"{start_indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal.")

  for i in range(len(start_index_map)):
    operand_dim_for_start_index_i = start_index_map[i]
    if (operand_dim_for_start_index_i < 0 or
        operand_dim_for_start_index_i >= _rank(operand)):
      raise TypeError(f"Invalid start_index_map; domain is "
                      f"[0, {_rank(operand)}), got: "
                      f"{i}->{operand_dim_for_start_index_i}.")

  _no_duplicate_dims(start_index_map, "gather", "start_index_map")

  # _is_sorted and _sorted_dims_in_range are checked in the opposite order
  # compared to the XLA implementation. In cases when the input is not sorted
  # AND there are problematic collapsed_slice_dims, the error message will thus
  # be different.
  _is_sorted(collapsed_slice_dims, "gather", "collapsed_slice_dims")
  _sorted_dims_in_range(collapsed_slice_dims, _rank(operand), "gather",
                        "collapsed_slice_dims")
  _no_duplicate_dims(collapsed_slice_dims, "gather", "collapsed_slice_dims")
  # End ValidateGatherDimensions

  if _rank(operand) != len(slice_sizes):
    raise TypeError(f"Gather op must have one slice size for every input "
                    f"dimension; got: len(slice_sizes)={len(slice_sizes)}, "
                    f"input_shape.rank={_rank(operand)}")

  if len(slice_sizes) != len(offset_dims) + len(collapsed_slice_dims):
    raise TypeError(f"All components of the offset index in a gather op must "
                    f"either be a offset dimension or explicitly collapsed; "
                    f"got len(slice_sizes)={len(slice_sizes)}, "
                    f"output_slice_sizes={offset_dims}, collapsed_slice_dims="
                    f"{collapsed_slice_dims}.")

  for i in range(len(slice_sizes)):
    slice_size = slice_sizes[i]
    corresponding_input_size = operand.shape[i]

    if not (core.greater_equal_dim(slice_size, 0) and
            core.greater_equal_dim(corresponding_input_size, slice_size)):
      raise TypeError(f"Slice size at index {i} in gather op is out of range, "
                      f"must be within [0, {corresponding_input_size} + 1), "
                      f"got {slice_size}.")

  for i in range(len(collapsed_slice_dims)):
    bound = slice_sizes[collapsed_slice_dims[i]]
    if bound > 1:
      raise TypeError(f"Gather op can only collapse slice dims with bound 1 "
                      f"or 0, but bound is {bound} for index "
                      f"{collapsed_slice_dims[i]} at position {i}.")

  expanded_start_indices_shape.pop(index_vector_dim)
  start_indices_shape = iter(expanded_start_indices_shape)

  slice_sizes = iter(np.delete(slice_sizes, collapsed_slice_dims))
  return tuple(next(slice_sizes) if i in offset_dims
               else next(start_indices_shape) for i in range(output_shape_rank))

def _gather_translation_rule(c, operand, start_indices, *, dimension_numbers,
                             slice_sizes):
  indices_shape = c.get_shape(start_indices)
  return xops.Gather(
    operand, start_indices,
    _gather_dimensions_proto(indices_shape, dimension_numbers), slice_sizes,
    indices_are_sorted=False)

def _gather_jvp_rule(g, operand, start_indices, *, dimension_numbers,
                     slice_sizes):
  return gather(g, start_indices, dimension_numbers, slice_sizes)

def _gather_transpose_rule(t, operand, start_indices, *, dimension_numbers,
                          slice_sizes):
  assert ad.is_undefined_primal(operand)
  operand_shape = operand.aval.shape
  if type(t) is ad_util.Zero:
    out = ad_util.Zero(operand.aval)
  else:
    zeros = full(operand_shape, _zero(t))
    scatter_dnums = ScatterDimensionNumbers(
      update_window_dims=dimension_numbers.offset_dims,
      inserted_window_dims=dimension_numbers.collapsed_slice_dims,
      scatter_dims_to_operand_dims=dimension_numbers.start_index_map)
    out = scatter_add(zeros, start_indices, t, scatter_dnums,
                      indices_are_sorted=False,
                      unique_indices=False)
  return [out, None]

def _gather_batching_rule(batched_args, batch_dims, *, dimension_numbers,
                          slice_sizes):
  operand, start_indices = batched_args
  operand_bdim, start_indices_bdim = batch_dims

  if operand_bdim is not None and start_indices_bdim is None:
    operand = batching.moveaxis(operand, operand_bdim, 0)
    slice_sizes = (operand.shape[0],) + slice_sizes
    offset_dims = (0,) + tuple(np.add(1, dimension_numbers.offset_dims))
    collapsed_slice_dims = tuple(np.add(1, dimension_numbers.collapsed_slice_dims))
    start_index_map = tuple(np.add(1, dimension_numbers.start_index_map))
    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map)
    return gather(operand, start_indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes), 0

  elif operand_bdim is None and start_indices_bdim is not None:
    start_indices = batching.moveaxis(start_indices, start_indices_bdim, 0)
    offset_dims = tuple(np.add(1, dimension_numbers.offset_dims))
    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=dimension_numbers.collapsed_slice_dims,
        start_index_map=dimension_numbers.start_index_map)
    return gather(operand, start_indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes), 0

  else:
    # move batch dimensions to the front to simplify logic
    operand = batching.moveaxis(operand, operand_bdim, 0)
    start_indices = batching.moveaxis(start_indices, start_indices_bdim, 0)

    # Example: user code had start_indices shape (3, 4, 5), and we have to deal
    # with start_indices shape (7, 3, 4, 5). We transform that to a
    # start_indices of shape (7, 3, 4, 6) where we concatenated an iota that
    # counts along our batch dimension to the front of the ndindex.
    count_shape = list(start_indices.shape)
    count_shape[-1] = 1
    counts = broadcasted_iota(start_indices.dtype, tuple(count_shape), 0)
    start_indices = concatenate([counts, start_indices], len(count_shape) - 1)

    batch_slice_size = 1 if core.greater_equal_dim(operand.shape[0], 1) else 0
    slice_sizes = (batch_slice_size,) + slice_sizes
    collapsed_slice_dims = (0,) + tuple(np.add(1, dimension_numbers.collapsed_slice_dims))
    offset_dims = tuple(np.add(1, dimension_numbers.offset_dims))
    start_index_map = (0,) + tuple(np.add(1, dimension_numbers.start_index_map))

    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map)
    return gather(operand, start_indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes), 0

gather_p = standard_primitive(
    _gather_shape_rule, _gather_dtype_rule, 'gather',
    _gather_translation_rule, weak_type_rule=_argnum_weak_type(0))
ad.defjvp(gather_p, _gather_jvp_rule, None)

ad.primitive_transposes[gather_p] = _gather_transpose_rule
batching.primitive_batchers[gather_p] = _gather_batching_rule


def _scatter_dimensions_proto(indices_shape, dimension_numbers):
  assert type(dimension_numbers) is ScatterDimensionNumbers
  proto = xla_client.ScatterDimensionNumbers()
  proto.update_window_dims.extend(dimension_numbers.update_window_dims)
  proto.inserted_window_dims.extend(dimension_numbers.inserted_window_dims)
  proto.scatter_dims_to_operand_dims.extend(
      dimension_numbers.scatter_dims_to_operand_dims)
  assert indices_shape.rank() > 0
  proto.index_vector_dim = indices_shape.rank() - 1
  return proto

def _scatter_dtype_rule(operand, scatter_indices, updates, **kwargs):
  if not dtypes.issubdtype(scatter_indices.dtype, np.integer):
    raise ValueError("scatter_indices must have an integer type")
  _check_same_dtypes("scatter", False, operand.dtype, updates.dtype)
  return dtypes.canonicalize_dtype(operand.dtype)

def _scatter_shape_rule(operand, scatter_indices, updates, *, update_jaxpr,
                        update_consts, dimension_numbers, indices_are_sorted,
                        unique_indices):
  """Validates the well-formedness of the ``dimension_numbers`` argument to
  Scatter.

  The code implements the checks based on the detailed operation semantics of
  XLA's `Scatter <https://www.tensorflow.org/xla/operation_semantics#scatter>`_
  operator and following the outline of the implementation of
  ShapeInference::InferScatterShape in TensorFlow.
  """

  update_window_dims = dimension_numbers.update_window_dims
  inserted_window_dims = dimension_numbers.inserted_window_dims
  scatter_dims_to_operand_dims = dimension_numbers.scatter_dims_to_operand_dims
  # Note: in JAX, index_vector_dim is always computed as below, cf. the
  # documentation of the ScatterDimensionNumbers class.
  index_vector_dim = _rank(scatter_indices) - 1

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if _rank(scatter_indices) < index_vector_dim or index_vector_dim < 0:
    raise TypeError(f"Scatter index leaf dimension must be within [0, "
                    f"rank(scatter_indices) + 1). rank(scatter_indices) is "
                    f"{_rank(scatter_indices)} and scatter index leaf "
                    f"dimension is {index_vector_dim}.")

  expanded_scatter_indices_shape = list(scatter_indices.shape)
  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if len(expanded_scatter_indices_shape) == index_vector_dim:
    expanded_scatter_indices_shape.append(1)

  expected_updates_rank = (len(expanded_scatter_indices_shape) - 1 +
                           len(update_window_dims))

  if _rank(updates) != expected_updates_rank:
    raise TypeError(f"Updates tensor must be of rank {expected_updates_rank}; "
                    f"got {_rank(updates)}.")

  # Validate update_window_dims
  _is_sorted(update_window_dims, "scatter", "update_window_dims")
  _no_duplicate_dims(update_window_dims, "scatter", "update_window_dims")
  _sorted_dims_in_range(update_window_dims, _rank(updates), "scatter",
                        "update_window_dims")

  # Validate inserted_window_dims
  _is_sorted(inserted_window_dims, "scatter", "inserted_window_dims")
  _no_duplicate_dims(inserted_window_dims, "scatter", "inserted_window_dims")
  _sorted_dims_in_range(inserted_window_dims, _rank(operand), "scatter",
                        "inserted_window_dims")

  # Validate window_size
  window_size = len(update_window_dims) + len(inserted_window_dims)
  if _rank(operand) != window_size:
    raise TypeError(f"Scatter op has window of size {window_size}; doesn't "
                    f"match operand of rank {_rank(operand)}.")

  # Validate scatter_dims_to_operand_dims
  if (len(scatter_dims_to_operand_dims) !=
      scatter_indices.shape[index_vector_dim]):
    raise TypeError(f"Scatter op has {len(scatter_dims_to_operand_dims)} "
                    f"elements in scatter_dims_to_operand_dims and the bound "
                    f"of dimension index_vector_dim={index_vector_dim} of "
                    f"scatter_indices is "
                    f"{scatter_indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal")

  for i in range(len(scatter_dims_to_operand_dims)):
    dim = scatter_dims_to_operand_dims[i]
    if dim < 0 or dim >= _rank(operand):
      raise TypeError(f"Invalid scatter_dims_to_operand_dims mapping; domain "
                      f"is [0, {_rank(operand)}), got: {i}->{dim}.")

  _no_duplicate_dims(scatter_dims_to_operand_dims, "scatter",
                     "scatter_dims_to_operand_dims")

  max_update_slice_sizes = [operand.shape[i] for i in range(len(operand.shape))
                            if not i in set(inserted_window_dims)]

  for i in range(len(update_window_dims)):
    update_window_dim = update_window_dims[i]
    if not core.greater_equal_dim(max_update_slice_sizes[i], updates.shape[update_window_dim]):
      raise TypeError(f"Bounds of the window dimensions of updates must not "
                      f"exceed the bounds of the corresponding dimensions of "
                      f"operand. For dimension {update_window_dim}, updates "
                      f"bound is {updates.shape[update_window_dim]}, operand "
                      f"bound is {max_update_slice_sizes[i]}.")

  update_scatter_dims = [dim for dim in range(_rank(updates)) if dim not in
                         set(update_window_dims)]

  scatter_dims_seen = 0
  for i in update_scatter_dims:
    if scatter_dims_seen == index_vector_dim:
      scatter_dims_seen += 1
    if updates.shape[i] != expanded_scatter_indices_shape[scatter_dims_seen]:
      raise TypeError(f"Bounds of the scatter dimensions of updates must be "
                      f"the same as the bounds of the corresponding dimensions "
                      f"of scatter indices. For scatter dimension {i}, updates "
                      f"bound is {updates.shape[i]}, scatter_indices bound is "
                      f"{expanded_scatter_indices_shape[scatter_dims_seen]}.")
    scatter_dims_seen += 1

  return operand.shape

def _scatter_translation_rule(c, operand, scatter_indices, updates, *,
                              update_jaxpr, update_consts, dimension_numbers,
                              indices_are_sorted, unique_indices):
  dtype = c.get_shape(operand).numpy_dtype()
  init_value = xb.constant(c, np.array(0, dtype))
  update_computation = _reduction_computation(
      c, update_jaxpr, update_consts, init_value)
  indices_shape = c.get_shape(scatter_indices)
  return xops.Scatter(operand, scatter_indices, updates, update_computation,
                      _scatter_dimensions_proto(indices_shape, dimension_numbers),
                      indices_are_sorted, unique_indices)

def _scatter_add_translation_rule(
    c, operand, scatter_indices, updates, *, update_jaxpr, update_consts,
    dimension_numbers, indices_are_sorted, unique_indices,
    expand_complex128=False):
  dtype = c.get_shape(operand).numpy_dtype()
  scatter_dims = _scatter_dimensions_proto(c.get_shape(scatter_indices),
                                           dimension_numbers)

  def _make_reducer(dtype):
    subc = xla_bridge.make_computation_builder("scatter_add_reducer")
    shape = xc.Shape.array_shape(np.dtype(dtype), ())
    args = [xb.parameter(subc, 0, shape), xb.parameter(subc, 1, shape)]
    out = xops.Add(args[0], args[1])
    return subc.build(out)

  if expand_complex128 and dtype == np.complex128:
    update_computation = _make_reducer(np.float64)
    re = xops.Scatter(xops.Real(operand), scatter_indices, xops.Real(updates),
                      update_computation, scatter_dims, indices_are_sorted,
                      unique_indices)
    im = xops.Scatter(xops.Imag(operand), scatter_indices, xops.Imag(updates),
                      update_computation, scatter_dims, indices_are_sorted,
                      unique_indices)
    return xops.Complex(re, im)
  else:
    update_computation = _make_reducer(dtype)
    return xops.Scatter(operand, scatter_indices, updates, update_computation,
                        scatter_dims, indices_are_sorted, unique_indices)

def _scatter_add_jvp(primals, tangents, *, update_jaxpr, update_consts,
                     dimension_numbers, indices_are_sorted, unique_indices):
  operand, scatter_indices, updates = primals
  g_operand, g_scatter_indices, g_updates = tangents
  del g_scatter_indices  # ignored
  val_out = scatter_add_p.bind(
      operand, scatter_indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)
  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_updates = ad.instantiate_zeros(g_updates)
    tangent_out = scatter_add_p.bind(
        g_operand, scatter_indices, g_updates, update_jaxpr=update_jaxpr,
        update_consts=update_consts, dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)
  return val_out, tangent_out

def _scatter_add_transpose_rule(t, operand, scatter_indices, updates, *,
                                update_jaxpr, update_consts, dimension_numbers,
                                indices_are_sorted, unique_indices):
  assert not ad.is_undefined_primal(scatter_indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      operand_t = t

    if ad.is_undefined_primal(updates):
      gather_dnums = GatherDimensionNumbers(
        offset_dims=dimension_numbers.update_window_dims,
        collapsed_slice_dims=dimension_numbers.inserted_window_dims,
        start_index_map=dimension_numbers.scatter_dims_to_operand_dims)
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if i in dimension_numbers.inserted_window_dims:
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(t, scatter_indices, dimension_numbers=gather_dnums,
                        slice_sizes=slice_sizes)
  return [operand_t, None, update_t]

def _scatter_mul_transpose_rule(t, operand, scatter_indices, updates, *,
                                update_jaxpr, update_consts, dimension_numbers,
                                indices_are_sorted, unique_indices):
  assert not ad.is_undefined_primal(scatter_indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      operand_t = scatter_mul(
          t, scatter_indices, updates, dimension_numbers=dimension_numbers,
          indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)
    if ad.is_undefined_primal(updates):
      gather_dnums = GatherDimensionNumbers(
        offset_dims=dimension_numbers.update_window_dims,
        collapsed_slice_dims=dimension_numbers.inserted_window_dims,
        start_index_map=dimension_numbers.scatter_dims_to_operand_dims)
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if i in dimension_numbers.inserted_window_dims:
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(mul(t, operand), scatter_indices,
                        dimension_numbers=gather_dnums, slice_sizes=slice_sizes)
  return [operand_t, None, update_t]


def _scatter_batching_rule(scatter_op, batched_args, batch_dims, *,
                           update_jaxpr, update_consts, dimension_numbers,
                           indices_are_sorted, unique_indices):
  operand, scatter_indices, updates = batched_args
  operand_bdim, scatter_indices_bdim, updates_bdim = batch_dims
  del update_jaxpr, update_consts  # Unused.

  # move the operand batch dim to the front if it is not None, otherwise create
  # it at the front (so that we can scatter into it)
  size = next(x.shape[ax] for x, ax in zip(batched_args, batch_dims)
              if ax is not None)
  operand = batching.bdim_at_front(operand, operand_bdim, size)
  operand_bdim = 0

  updates = batching.bdim_at_front(updates, updates_bdim, size)

  if scatter_indices_bdim is None:
    inserted_window_dims = tuple(np.add(1, dimension_numbers.inserted_window_dims))
    update_window_dims = (0,) + tuple(np.add(1, dimension_numbers.update_window_dims))
    scatter_dims_to_operand_dims = tuple(np.add(1, dimension_numbers.scatter_dims_to_operand_dims))
    dnums = ScatterDimensionNumbers(
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        scatter_dims_to_operand_dims=scatter_dims_to_operand_dims)
    return scatter_op(
      operand, scatter_indices, updates, dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices), 0


  # see the third case in _gather_batching_rule for comparison and comments
  scatter_indices = batching.bdim_at_front(
    scatter_indices, scatter_indices_bdim, size)

  count_shape = list(scatter_indices.shape)
  count_shape[-1] = 1
  counts = broadcasted_iota(scatter_indices.dtype, tuple(count_shape), 0)
  scatter_indices = concatenate([counts, scatter_indices],
                                len(count_shape) - 1)

  update_window_dims = tuple(np.add(1, dimension_numbers.update_window_dims))
  inserted_window_dims = (0,) + tuple(np.add(1, dimension_numbers.inserted_window_dims))
  scatter_dims_to_operand_dims = (0,) + tuple(np.add(1, dimension_numbers.scatter_dims_to_operand_dims))

  dnums = ScatterDimensionNumbers(
      update_window_dims=update_window_dims,
      inserted_window_dims=inserted_window_dims,
      scatter_dims_to_operand_dims=scatter_dims_to_operand_dims)
  return scatter_op(
      operand, scatter_indices, updates, dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices), 0

scatter_add_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-add',
    _scatter_add_translation_rule, weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[scatter_add_p] = _scatter_add_jvp
ad.primitive_transposes[scatter_add_p] = _scatter_add_transpose_rule
batching.primitive_batchers[scatter_add_p] = (
  partial(_scatter_batching_rule, scatter_add))

xla.backend_specific_translations['gpu'][scatter_add_p] = partial(
    _scatter_add_translation_rule, expand_complex128=True)

scatter_mul_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-mul',
    _scatter_translation_rule, weak_type_rule=_argnum_weak_type(0))

def _scatter_mul_jvp_rhs(g, x, i, y, *, dimension_numbers,
                         indices_are_sorted, unique_indices, **kw):
  return mul(x, scatter_add(
      zeros_like_array(x), i, g, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices))

ad.defjvp(scatter_mul_p,
          lambda g, x, i, y, **kw: scatter_mul_p.bind(g, i, y, **kw),
          None,
          _scatter_mul_jvp_rhs)
ad.primitive_transposes[scatter_mul_p] = _scatter_mul_transpose_rule
batching.primitive_batchers[scatter_mul_p] = (
  partial(_scatter_batching_rule, scatter_mul))

def _scatter_extremal_jvp(scatter_op, primals, tangents, update_jaxpr,
                          update_consts, dimension_numbers,
                          indices_are_sorted, unique_indices):
  operand, scatter_indices, updates = primals
  g_operand, g_scatter_indices, g_updates = tangents

  scatter_dnums = dimension_numbers
  updates_shape = updates.shape

  val_out = scatter_op.bind(
      operand, scatter_indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=scatter_dnums,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)

  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_updates = ad.instantiate_zeros(g_updates)

    # gather_dnums and slice_sizes define the gather op that is the inverse of
    # the scatter op specified by scatter_dnums
    gather_dnums = GatherDimensionNumbers(
        offset_dims=scatter_dnums.update_window_dims,
        collapsed_slice_dims=scatter_dnums.inserted_window_dims,
        start_index_map=scatter_dnums.scatter_dims_to_operand_dims)

    slice_sizes = []
    pos = 0
    for i in range(len(operand.shape)):
      if i in scatter_dnums.inserted_window_dims:
        slice_sizes.append(1)
      else:
        slice_sizes.append(updates_shape[scatter_dnums.update_window_dims[pos]])
        pos += 1

    # For consistency with other max operations, if there are two or more values
    # in updates that are contending to replace the same index location, the
    # resulting tangent at that location will be the average of the associated
    # tangents for the values in updates.

    initial_vals = gather(
        operand, scatter_indices, gather_dnums, np.array(slice_sizes))

    target_vals = gather(
        val_out, scatter_indices, gather_dnums, np.array(slice_sizes))

    successful_updates = (updates == target_vals)
    retained_values = (initial_vals == target_vals)

    num_updates = gather(
        scatter_add(_zeros(operand),
                    scatter_indices,
                    select(successful_updates, _ones(updates), _zeros(updates)),
                    scatter_dnums),
        scatter_indices,
        gather_dnums,
        np.array(slice_sizes))

    num_refs = gather(
        scatter_add(_zeros(operand),
                    scatter_indices,
                    _ones(updates),
                    scatter_dnums),
        scatter_indices,
        gather_dnums,
        np.array(slice_sizes))

    updates_normalizer = select(retained_values,
                                1.0 / (num_updates + 1),
                                1.0 / num_updates)

    updates_coef = select(successful_updates,
                          updates_normalizer,
                          _zeros(updates))

    operand_normalizer = select(retained_values,
                                1.0 / (num_updates + 1),
                                _zeros(num_updates))

    operand_coef = (-1.0 + operand_normalizer) / num_refs

    # This can be simplified once scatter has transpose implemented
    target_tangents = gather(
        g_operand, scatter_indices, gather_dnums, np.array(slice_sizes))

    tangent_updates = (target_tangents * operand_coef +
                       g_updates * updates_coef)

    tangent_out = scatter_add(g_operand,
                              scatter_indices,
                              tangent_updates,
                              scatter_dnums,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices)

  return val_out, tangent_out

scatter_min_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-min',
    _scatter_translation_rule, weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_min_p] = (
  partial(_scatter_batching_rule, scatter_min))
ad.primitive_jvps[scatter_min_p] = partial(_scatter_extremal_jvp, scatter_min_p)

scatter_max_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-max',
    _scatter_translation_rule, weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_max_p] = (
  partial(_scatter_batching_rule, scatter_max))
ad.primitive_jvps[scatter_max_p] = partial(_scatter_extremal_jvp, scatter_max_p)

def _scatter_jvp(primals, tangents, *, update_jaxpr, update_consts,
                 dimension_numbers, indices_are_sorted, unique_indices):
  operand, scatter_indices, updates = primals
  g_operand, g_scatter_indices, g_updates = tangents
  dnums = dimension_numbers

  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    val_out = scatter_p.bind(
      operand, scatter_indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)
    return val_out, ad_util.Zero.from_value(val_out)

  g_operand = ad.instantiate_zeros(g_operand)
  g_updates = ad.instantiate_zeros(g_updates)

  # If there are overlapping indices in the scatter, it is unspecified which
  # update "wins". So we use the following perhaps surprising scheme:
  # a) attach a positive ID to each update in updates, and perform the scatter
  #    on the IDs
  # b) perform the inverse gather on the scattered IDs (similar to
  #    _scatter_add_transpose).
  # c) use the gathered IDs to mask the primal and tangent values.
  # d) perform a scatter-add on the masked primal and tangent values. A benefit
  #    of using scatter-add here is that we don't need a `scatter` transpose
  #    rule.


  # a) attach a positive ID to each update in `updates`, and perform a scatter
  #    on the IDs.
  ids_shape = np.array(updates.shape, dtype=np.int64)
  ids_shape[dnums.update_window_dims,] = 1
  num_ids = np.prod(ids_shape)
  id_dtype = np.uint32 if (num_ids + 1) < np.iinfo(np.uint32).max else np.uint64
  update_ids = add(reshape(iota(id_dtype, num_ids), ids_shape),
                   _ones(updates, dtype=id_dtype))

  scattered_ids = scatter(full(operand.shape, 0, id_dtype),
                          scatter_indices, update_ids, dnums,
                          indices_are_sorted=indices_are_sorted,
                          unique_indices=unique_indices)

  # b) compute the inverse gather that "undoes" the scatter on the id values.
  gather_dnums = GatherDimensionNumbers(
    offset_dims=dnums.update_window_dims,
    collapsed_slice_dims=dnums.inserted_window_dims,
    start_index_map=dnums.scatter_dims_to_operand_dims)
  slice_sizes = []
  pos = 0
  for i in range(len(scattered_ids.shape)):
    if i in dnums.inserted_window_dims:
      slice_sizes.append(1)
    else:
      slice_sizes.append(updates.shape[dnums.update_window_dims[pos]])
      pos += 1
  gathered_update_ids = gather(scattered_ids, scatter_indices,
                               dimension_numbers=gather_dnums,
                               slice_sizes=slice_sizes)

  # c) mask off input elements that do not correspond to a primal output.
  masked_operand = select(eq(scattered_ids, _zeros(scattered_ids)),
                          operand, _zeros(operand))
  masked_updates = select(eq(update_ids,  gathered_update_ids),
                          updates, _zeros(updates))
  masked_g_operand = select(eq(scattered_ids, _zeros(scattered_ids)),
                            g_operand, _zeros(g_operand))
  masked_g_updates = select(eq(update_ids, gathered_update_ids),
                            g_updates, _zeros(g_updates))

  # d) perform scatter-adds to compute the primal and tangent outputs.
  val_out = scatter_add(masked_operand, scatter_indices, masked_updates,
                        dimension_numbers=dnums,
                        indices_are_sorted=indices_are_sorted,
                        unique_indices=unique_indices)
  tangent_out = scatter_add(masked_g_operand, scatter_indices, masked_g_updates,
                            dimension_numbers=dnums,
                            indices_are_sorted=indices_are_sorted,
                            unique_indices=unique_indices)
  return val_out, tangent_out


scatter_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter',
    _scatter_translation_rule, weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[scatter_p] = _scatter_jvp
batching.primitive_batchers[scatter_p] = (
  partial(_scatter_batching_rule, scatter))


def _reduce_shape_rule(*avals, computation, jaxpr, consts, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  if any(arg.shape != () for arg in init_val_avals):
    init_val_shapes = [a.shape for a in init_val_avals]
    raise ValueError(f'reduce found non-scalar initial value: {init_val_shapes}')
  return [tuple(np.delete(op.shape, dimensions)) for op in operand_avals]

def _reduce_dtype_rule(*avals, computation, jaxpr, consts, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  operand_dtypes = [dtypes.canonicalize_dtype(op.dtype) for op in operand_avals]
  init_val_dtypes = [dtypes.canonicalize_dtype(init.dtype) for init in init_val_avals]
  if operand_dtypes != init_val_dtypes:
    raise TypeError(
        "reduce operand dtypes should match corresponding initial value dtypes, "
        f"got operands={operand_avals} and initial_values={init_val_avals}")
  return operand_dtypes

def _reduce_weak_type_rule(*avals, computation, jaxpr, consts, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  return [op.weak_type and init_val.weak_type
          for op, init_val in safe_zip(operand_avals, init_val_avals)]

def _reduce_translation_rule(c, *values, computation, jaxpr,
                             consts, dimensions):
  operands, init_values = split_list(values, [len(values) // 2])
  if len(operands) == 1:
    init_value = init_values[0]
    xla_computation = _reduction_computation(c, jaxpr, consts, init_value)
    out = xops.Reduce(c, operands, init_values, xla_computation, dimensions)
    return xops.Tuple(c, (out,))
  xla_computation = _reduction_computation(c, jaxpr, consts, init_values, singleton=False)
  return xops.Reduce(c, operands, init_values, xla_computation, dimensions)

def _reduce_batch_rule(batched_args, batch_dims, *, computation, jaxpr,
                       consts, dimensions):
  # TODO(mattjj,frostig): use batch_jaxpr, delete computation (assumes poly??)
  num_operands = len(batched_args) // 2
  operands, init_values = split_list(batched_args, [num_operands])
  operand_bdims, init_value_bdims = split_list(batch_dims, [num_operands])
  if all(init_value_bdim is batching.not_mapped
         for init_value_bdim in init_value_bdims):
    # Assume all batch dims are the same for each of the operands
    # TODO(sharadmv): handle the case when batch dims are different across
    # operands or when some are unbatched
    if not all(operand_bdim is not batching.not_mapped for operand_bdim in operand_bdims):
      raise NotImplementedError
    if not all(operand_bdim == operand_bdims[0] for operand_bdim in operand_bdims):
      raise NotImplementedError
    operand_bdim = operand_bdims[0]
    new_dimensions = [d + bool(d >= operand_bdim) for d in dimensions]
    new_operand_bdim = operand_bdim - int(np.sum(np.less(dimensions, operand_bdim)))
    new_operand_bdims = [new_operand_bdim] * num_operands
    return reduce_p.bind(*(operands + init_values),
                         computation=computation, dimensions=tuple(new_dimensions),
                         consts=consts,
                         jaxpr=jaxpr), new_operand_bdims
  else:
    raise NotImplementedError  # loop and stack

def _reduction_computation(c, jaxpr, consts, init_values, singleton=True):
  if singleton:
    init_values = [init_values]
  shapes = safe_map(c.get_shape, init_values + init_values)
  axis_env = xla.AxisEnv(1, (), ())  # no parallel primitives inside reductions
  subc = xla_bridge.make_computation_builder("reduction_computation")
  assert len(consts) == 0, "Reduction computations cannot have constants"
  args = [xb.parameter(subc, i, shape) for i, shape in enumerate(shapes)]
  out_nodes = xla.jaxpr_subcomp(subc, jaxpr, None, axis_env, consts, '', *args)
  if singleton:
    return subc.build(out_nodes[0])
  out_nodes = xops.Tuple(subc, out_nodes)
  return subc.build(out_nodes)

def _reduce_jvp(reducer, init_values, primals, tangents, axes):
  input_shape = np.array(primals[0].shape)

  n = np.prod(input_shape[list(axes)])
  non_axes = np.delete(np.arange(len(input_shape)), axes)

  # Move the reduced axes to the front, and flatten them to 1D.
  permutation = axes + tuple(non_axes)
  new_shape = (n,) + tuple(input_shape[non_axes])
  primals = tuple(reshape(x, new_shape, permutation) for x in primals)
  tangents = tuple(reshape(t, new_shape, permutation) for t in tangents)

  for d in range(len(non_axes) + 1):
    reducer = api.vmap(reducer)
  def _reduce_tree(*xs, axis=0):
    """Reduce by repeatedly splitting the array and multiplying."""
    while xs[0].shape[axis] > 1:
      n = xs[0].shape[axis]
      n1 = (n + 1) // 2
      n2 = n - n1
      xs1 = [slice_in_dim(x, 0, n1) for x in xs]
      xs2 = [slice_in_dim(x, n1, None) for x in xs]
      if n2 != n1:
        paddings = [(0, 0, 0)] * len(xs[0].shape)
        paddings[axis] = (0, 1, 0)
        xs2 = [pad(x2, i, paddings) for x2, i in zip(xs2, init_values)]
      xs = reducer(*(xs1 + xs2))
    if xs[0].shape[axis] == 0:
      return [full(input_shape[non_axes], i) for i in init_values]
    return tuple(squeeze(x, (axis,)) for x in xs)

  return api.jvp(_reduce_tree, primals, tangents)

def _reduce_jvp_rule(primals, tangents, *, computation, jaxpr,
                     consts, dimensions):
  primal_xs, init_values = split_list(primals, [len(primals) // 2])
  tangent_xs, tangent_init = split_list(tangents, [len(tangents) // 2])
  # This test may be too strict, if a value is actually zero but we cannot prove
  # it is symbolically zero.
  if any(type(t) is not ad_util.Zero for t in tangent_init):
    raise NotImplementedError(
      "Gradient of general lax.reduce with non-zero tangents for "
      "initial values to reduction not implemented")
  reducer = core.jaxpr_as_fun(core.ClosedJaxpr(jaxpr, consts))
  return _reduce_jvp(reducer, init_values, primal_xs, tangent_xs, dimensions)

def _masking_defreducer(prim, identity):
  masking.masking_rules[prim] = partial(_reducer_masking_rule, prim, identity)

def _reducer_masking_rule(prim, identity, padded_vals, logical_shapes,
                          axes, input_shape=None, **reduce_kwargs):
  (padded_val,), (logical_shape,) = padded_vals, logical_shapes
  padded_shape = masking.padded_shape_as_value(padded_val.shape)
  masks = [broadcasted_iota(np.int32, padded_shape, i) < d
           for i, d in enumerate(logical_shape) if i in axes]
  mask = _reduce(operator.and_, masks)
  masked_val = select(mask, padded_val, identity(padded_shape, padded_val.dtype))
  prim_bind = partial(prim.bind, **reduce_kwargs)
  bind = prim_bind if input_shape is None else partial(prim_bind, input_shape=padded_shape)
  return bind(masked_val, axes=axes)

def _reduce_named_shape_rule(*avals, computation, jaxpr, consts, dimensions):
  # TODO(mattjj,frostig): see the TODOs noting limitations/assumptions in
  # _reduce_batching_rule. We're making the same assumptions here for now.
  num_operands = len(avals) // 2
  operand_avals, init_avals = split_list(avals, [num_operands])
  if any(a.named_shape for a in init_avals):
    raise NotImplementedError
  named_shapes = [a.named_shape for a in operand_avals]
  if not all(named_shapes[0] == named_shape for named_shape in named_shapes):
    raise NotImplementedError
  return named_shapes


reduce_p = core.Primitive('reduce')
reduce_p.multiple_results = True
reduce_p.def_impl(partial(xla.apply_primitive, reduce_p))
reduce_p.def_abstract_eval(
    partial(standard_multi_result_abstract_eval, reduce_p, _reduce_shape_rule,
            _reduce_dtype_rule, _reduce_weak_type_rule,
            _reduce_named_shape_rule))
xla.translations[reduce_p] = _reduce_translation_rule
batching.primitive_batchers[reduce_p] = _reduce_batch_rule
ad.primitive_jvps[reduce_p] = _reduce_jvp_rule

def _reduce_number_dtype_rule(name, operand, *args, **kw):
  if not dtypes.issubdtype(operand.dtype, np.number):
    raise TypeError("{} does not accept dtype {}. Accepted dtypes are subtypes "
                    "of number.".format(name, np.dtype(operand.dtype).name))
  return dtypes.canonicalize_dtype(operand.dtype)

def _reduce_sum_shape_rule(operand, *, axes):
  return _reduce_op_shape_rule(operand, axes=axes)

def _reduce_sum_translation_rule(c, operand, *, axes):
  shape = c.get_shape(operand)
  dtype = shape.numpy_dtype()
  scalar = ShapedArray((), dtype)
  return xops.Reduce(c, [operand], [xb.constant(c, np.array(0, dtype))],
                     xla.primitive_subcomputation(add_p, scalar, scalar),
                     axes)

def _reduce_sum_transpose_rule(cotangent, operand, *, axes):
  assert ad.is_undefined_primal(operand)
  input_shape = operand.aval.shape
  broadcast_dimensions = tuple(np.delete(np.arange(len(input_shape)), axes))
  result = broadcast_in_dim(cotangent, input_shape, broadcast_dimensions)
  assert result.shape == input_shape
  return [result]

reduce_sum_p = standard_primitive(
  _reduce_sum_shape_rule, partial(_reduce_number_dtype_rule, 'reduce_sum'),
  'reduce_sum', _reduce_sum_translation_rule)
ad.deflinear2(reduce_sum_p, _reduce_sum_transpose_rule)
batching.defreducer(reduce_sum_p)
_masking_defreducer(reduce_sum_p,
                    lambda shape, dtype: np.broadcast_to(np.array(0, dtype), shape))


def _reduce_op_shape_rule(operand, *, axes, input_shape=None):
  del input_shape  # Unused.
  if len(axes) != len(set(axes)):
    raise ValueError(f"duplicate value in 'axes' of reduction: {axes}")
  if not all(0 <= a < operand.ndim for a in axes):
    raise ValueError(f"reduction axes {axes} contains out-of-bounds indices for {operand}.")
  return tuple(np.delete(operand.shape, axes))

def _reduce_prod_translation_rule(c, operand, *, axes):
  dtype = c.get_shape(operand).numpy_dtype()
  scalar = ShapedArray((), dtype)
  return xops.Reduce(c, [operand], [xb.constant(c, np.array(1, dtype))],
                     xla.primitive_subcomputation(mul_p, scalar, scalar), axes)

def _reduce_prod_jvp_rule(primals, tangents, *, axes):
  reducer = lambda x, y: [mul(x, y)]
  primals_out, tangents_out = _reduce_jvp(reducer, [_const(primals[0], 1)],
                                          primals, tangents, axes)
  return primals_out[0], tangents_out[0]

reduce_prod_p = standard_primitive(
  _reduce_op_shape_rule, partial(_reduce_number_dtype_rule, 'reduce_prod'),
  'reduce_prod', _reduce_prod_translation_rule)
ad.primitive_jvps[reduce_prod_p] = _reduce_prod_jvp_rule
batching.defreducer(reduce_prod_p)
_masking_defreducer(reduce_prod_p,
                    lambda shape, dtype: np.broadcast_to(np.array(1, dtype), shape))


def _reduce_chooser_shape_rule(operand, *, axes):
  return tuple(np.delete(operand.shape, axes))

def _reduce_chooser_translation_rule(prim, identity, c, operand, *, axes):
  dtype = c.get_shape(operand).numpy_dtype()
  scalar = ShapedArray((), dtype)
  return xops.Reduce(c, [operand], [xb.constant(c, identity(dtype))],
                     xla.primitive_subcomputation(prim, scalar, scalar), axes)

def _reduce_chooser_jvp_rule(g, ans, operand, *, axes):
  # TODO(mattjj): an alternative is to use variadic reduce to compute the chosen
  # locations in a single pass (rather than comparing equality) and use a
  # gather, and/or even push along the chosen elements of g (b/112040122)
  shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
  location_indicators = convert_element_type(
      _eq_meet(operand, reshape(ans, shape)), g.dtype)
  counts = _reduce_sum(location_indicators, axes)
  return div(_reduce_sum(mul(g, location_indicators), axes), counts)

_reduce_max_translation_rule = partial(_reduce_chooser_translation_rule, max_p,
                                       _get_max_identity)
reduce_max_p = standard_primitive(_reduce_op_shape_rule, _input_dtype,
                                  'reduce_max', _reduce_max_translation_rule)
ad.defjvp2(reduce_max_p, _reduce_chooser_jvp_rule)
batching.defreducer(reduce_max_p)
_masking_defreducer(reduce_max_p,
                    lambda shape, dtype: np.broadcast_to(np.array(-np.inf, dtype), shape))


_reduce_min_translation_rule = partial(
    _reduce_chooser_translation_rule, min_p, _get_min_identity)
reduce_min_p = standard_primitive(_reduce_op_shape_rule, _input_dtype,
                                  'reduce_min', _reduce_min_translation_rule)
ad.defjvp2(reduce_min_p, _reduce_chooser_jvp_rule)
batching.defreducer(reduce_min_p)
_masking_defreducer(reduce_min_p,
                    lambda shape, dtype: np.broadcast_to(np.array(np.inf, dtype), shape))



def _argminmax_shape_rule(operand, *, axes, index_dtype):
  axis, = axes
  return tuple(np.delete(operand.shape, axis))

def _argminmax_dtype_rule(operand, *, axes, index_dtype):
  if not dtypes.issubdtype(index_dtype, np.integer):
    raise TypeError("index_dtype must be an integer type, but got {}"
                    .format(np.dtype(index_dtype).name))
  return index_dtype

def _argminmax_translation_rule(value_comparator, identity,
                                c, operand, *, axes, index_dtype):
  axis, = axes
  shape = c.get_shape(operand)
  dtype = shape.numpy_dtype()

  subc = xb.make_computation_builder("argminmax_comparator")
  value_shape = xc.Shape.array_shape(shape.xla_element_type(), ())
  index_shape = xc.Shape.array_shape(index_dtype, ())
  x_value = xb.parameter(subc, 0, value_shape)
  x_index = xb.parameter(subc, 1, index_shape)
  y_value = xb.parameter(subc, 2, value_shape)
  y_index = xb.parameter(subc, 3, index_shape)
  which_value = value_comparator(x_value, y_value)
  which_index = xops.Or(which_value, xops.And(xops.Eq(x_value, y_value),
                                              xops.Lt(x_index, y_index)))
  xops.Tuple(subc, [xops.Select(which_value, x_value, y_value),
                    xops.Select(which_index, x_index, y_index)])
  comparator = subc.build()

  iota_shape = xc.Shape.array_shape(index_dtype, shape.dimensions())
  iota = xc.ops.Iota(c, iota_shape, axis)
  out = xops.Reduce(
    c, [operand, iota],
    [xb.constant(c, identity(dtype)),
     xb.constant(c, np.array(0, index_dtype))], comparator, [axis])
  return xops.GetTupleElement(out, 1)

def _argminmax_gpu_translation_rule(op, a, *, axes, index_dtype):
  axis, = axes
  idxs = tie_in(a, broadcasted_iota(index_dtype, a.shape, axis))
  maxval = np.array(dtypes.iinfo(index_dtype).max, dtype=index_dtype)
  maxval = broadcast(tie_in(a, maxval), a.shape)
  mask_idxs = select(eq(a, expand_dims(op(a, (axis,)), (axis,))), idxs,
                     maxval)
  return _reduce_min(mask_idxs, (axis,))

_argmin_translation_rule = partial(_argminmax_translation_rule, xops.Lt,
                                   _get_min_identity)
_argmax_translation_rule = partial(_argminmax_translation_rule, xops.Gt,
                                   _get_max_identity)

argmin_p = standard_primitive(_argminmax_shape_rule, _argminmax_dtype_rule,
                              'argmin', _argmin_translation_rule,
                              weak_type_rule=_strip_weak_type)
batching.defreducer(argmin_p)
ad.defjvp_zero(argmin_p)
xla.backend_specific_translations['gpu'][argmin_p] = xla.lower_fun(
  partial(_argminmax_gpu_translation_rule, _reduce_min),
  multiple_results=False)

argmax_p = standard_primitive(_argminmax_shape_rule, _argminmax_dtype_rule,
                              'argmax', _argmax_translation_rule,
                              weak_type_rule=_strip_weak_type)
batching.defreducer(argmax_p)
ad.defjvp_zero(argmax_p)
xla.backend_specific_translations['gpu'][argmax_p] = xla.lower_fun(
  partial(_argminmax_gpu_translation_rule, _reduce_max),
  multiple_results=False)


def _reduce_logical_shape_rule(operand, *, axes):
  if operand.dtype != np.bool_:
    msg = "logical reduction requires operand dtype bool, got {}."
    raise TypeError(msg.format(operand.dtype))
  return tuple(np.delete(operand.shape, axes))

def _reduce_logical_translation_rule(prim, identity, c, operand, *, axes):
  scalar = ShapedArray((), np.bool_)
  return xops.Reduce(c, [operand], [xb.constant(c, identity(np.bool_))],
                     xla.primitive_subcomputation(prim, scalar, scalar), axes)

_reduce_or_translation_rule = partial(_reduce_logical_translation_rule,
                                      or_p, _get_max_identity)
reduce_or_p = standard_primitive(_reduce_logical_shape_rule, _fixed_dtype(np.bool_),
                                 'reduce_or', _reduce_or_translation_rule,
                                 weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_or_p)


_reduce_and_translation_rule = partial(_reduce_logical_translation_rule,
                                       and_p, _get_min_identity)
reduce_and_p = standard_primitive(_reduce_logical_shape_rule, _fixed_dtype(np.bool_),
                                 'reduce_and', _reduce_and_translation_rule,
                                 weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_and_p)

def _reduce_window_shape_rule(operand, init_value, *, jaxpr, consts,
                              window_dimensions, window_strides, padding,
                              base_dilation, window_dilation):
  if operand.dtype != init_value.dtype:
    msg = ("reduce_window got inconsistent dtypes for operand and init_value: "
           " got operand dtype {} and init_value dtype {}.")
    raise TypeError(msg.format(operand.dtype, init_value.dtype))
  if init_value.shape != ():
    msg = ("reduce_window expected init_value to be a scalar but init_value "
           "has shape {}.")
    raise TypeError(msg.format(init_value.shape))
  return _common_reduce_window_shape_rule(
    operand, window_dimensions, window_strides, padding, base_dilation,
    window_dilation)

def _reduce_window_translation_rule(c, operand, init_value, *, jaxpr, consts,
                                    window_dimensions, window_strides, padding,
                                    base_dilation, window_dilation):
  xla_computation = _reduction_computation(c, jaxpr, consts, init_value)
  return xops.ReduceWindowWithGeneralPadding(
    operand, init_value, xla_computation, window_dimensions,
    window_strides, base_dilation, window_dilation, padding)

def _generic_reduce_window_batch_rule(
    batched_args, batch_dims, *, jaxpr, consts, window_dimensions,
    window_strides, padding, base_dilation, window_dilation):
  operand, init = batched_args
  bdim, init_bdim = batch_dims
  if init_bdim is not None:
    raise NotImplementedError("reduce_window batching is not implemented for "
                              "initial values")

  def reduce_window(x, window_dimensions, window_strides, padding, base_dilation,
                    window_dilation):
    return reduce_window_p.bind(
      x, init, jaxpr=jaxpr, consts=consts, window_dimensions=window_dimensions,
      window_strides=window_strides, padding=padding, base_dilation=base_dilation,
      window_dilation=window_dilation)
  return _reduce_window_batch_rule(
    reduce_window, (operand,), (bdim,), window_dimensions=window_dimensions,
    window_strides=window_strides, padding=padding, base_dilation=base_dilation,
    window_dilation=window_dilation)


reduce_window_p = standard_primitive(
    _reduce_window_shape_rule, _input_dtype, 'reduce_window',
    _reduce_window_translation_rule)
batching.primitive_batchers[reduce_window_p] = _generic_reduce_window_batch_rule


def _reduce_window_sum_shape_rule(operand, *, window_dimensions, window_strides,
                                  padding, base_dilation, window_dilation):
  if not dtypes.issubdtype(operand.dtype, np.number):
    msg = "operand to reduce_window_sum must have a number dtype, got {}"
    raise TypeError(msg.format(np.dtype(operand.dtype).name))
  return _common_reduce_window_shape_rule(operand, window_dimensions,
                                          window_strides, padding, base_dilation,
                                          window_dilation)

def _reduce_window_sum_translation_rule(c, operand, *, window_dimensions,
                                        window_strides, padding, base_dilation,
                                        window_dilation):
  dtype = c.get_shape(operand).numpy_dtype()
  scalar = ShapedArray((), dtype)
  return xops.ReduceWindowWithGeneralPadding(
    operand, xb.constant(c, np.array(0, dtype)),
    xla.primitive_subcomputation(add_p, scalar, scalar), window_dimensions,
    window_strides, base_dilation, window_dilation, padding)

def _reduce_window_sum_transpose_rule(cotangent, operand, *, window_dimensions,
                                      window_strides, padding, base_dilation,
                                      window_dilation):
  assert ad.is_undefined_primal(operand)
  input_shape = operand.aval.shape
  pads = _conv_general_vjp_lhs_padding(
      input_shape, window_dimensions, window_strides, cotangent.shape, padding,
      base_dilation, window_dilation)
  ones = [1] * len(input_shape)
  padding_config = [(lo, hi, stride - 1)
                    for (lo, hi), stride in zip(pads, window_strides)]
  pad_cotangent = pad(cotangent, _zero(cotangent), padding_config)
  result = _reduce_window_sum(pad_cotangent, window_dimensions, base_dilation,
                              [(0, 0)] * len(input_shape),
                              base_dilation=ones,
                              window_dilation=window_dilation)
  assert result.shape == input_shape, (result.shape, input_shape)
  return [result]

def _reduce_window_batch_rule(reduce_window, batched_args, bdims, *,
                              window_dimensions, window_strides, padding,
                              base_dilation, window_dilation):
  operand, = batched_args
  bdim, = bdims

  if bdim is not None:
    window_dimensions = \
        window_dimensions[:bdim] + (1,) + window_dimensions[bdim:]
    window_strides = window_strides[:bdim] + (1,) + window_strides[bdim:]
    padding = padding[:bdim] + ((0, 0),) + padding[bdim:]
    base_dilation = base_dilation[:bdim] + (1,) + base_dilation[bdim:]
    window_dilation = window_dilation[:bdim] + (1,) + window_dilation[bdim:]

  operand = reduce_window(operand, window_dimensions, window_strides, padding,
                          base_dilation, window_dilation)
  return operand, bdim

reduce_window_sum_p = standard_primitive(
    _reduce_window_sum_shape_rule, _input_dtype, 'reduce_window_sum',
    _reduce_window_sum_translation_rule)
ad.deflinear2(reduce_window_sum_p, _reduce_window_sum_transpose_rule)
batching.primitive_batchers[reduce_window_sum_p] = partial(
  _reduce_window_batch_rule, _reduce_window_sum)

def _reduce_window_chooser_translation_rule(
    prim, identity, c, operand, *, window_dimensions, window_strides, padding,
    base_dilation, window_dilation):
  dtype = c.get_shape(operand).numpy_dtype()
  scalar = ShapedArray((), dtype)
  return xops.ReduceWindowWithGeneralPadding(
    operand, xb.constant(c, identity(dtype)),
    xla.primitive_subcomputation(prim, scalar, scalar), window_dimensions,
    window_strides, base_dilation, window_dilation, padding)

def _reduce_window_chooser_jvp_rule(prim, g, operand, *, window_dimensions,
                                    window_strides, padding, base_dilation,
                                    window_dilation):
  assert prim is max_p or prim is min_p
  select_prim = ge_p if prim is max_p else le_p
  return _select_and_gather_add(g, operand, select_prim, window_dimensions,
                                window_strides, padding, base_dilation,
                                window_dilation)


def _common_reduce_window_shape_rule(operand, window_dimensions,
                                     window_strides, padding, base_dilation,
                                     window_dilation):
  _check_shapelike("reduce_window", "window_dimensions", window_dimensions,
                   non_zero_shape=True)
  _check_shapelike("reduce_window", "window_strides", window_strides,
                   non_zero_shape=True)
  _check_shapelike("reduce_window", "base_dilation", base_dilation)
  _check_shapelike("reduce_window", "window_dilation", window_dilation)
  if operand.ndim != len(window_dimensions):
    msg = ("reduce_window got the wrong number of window_dimensions for "
           "operand: got operand shape {} with window_dimensions {}.")
    raise TypeError(msg.format(operand.shape, window_dimensions))
  if len(window_strides) != len(window_dimensions):
    msg = ("reduce_window got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))
  if len(base_dilation) != len(window_dimensions):
    msg = ("reduce_window got inconsistent base_dilation and "
           "window_dimensions: got base_dilation {} and window_dimensions {}.")
    raise TypeError(msg.format(base_dilation, window_dimensions))
  if len(window_dilation) != len(window_dimensions):
    msg = ("reduce_window got inconsistent window_dilation and "
           "window_dimensions: got window_dilation {} and window_dimensions "
           "{}.")
    raise TypeError(msg.format(window_dilation, window_dimensions))

  return reduce_window_shape_tuple(operand.shape, window_dimensions,
                                   window_strides, padding, base_dilation,
                                   window_dilation)

def reduce_window_shape_tuple(operand_shape, window_dimensions, window_strides,
                              padding, base_dilation=None,
                              window_dilation=None):
  if base_dilation is not None:
    operand_shape = _dilate_shape(operand_shape, base_dilation)
  if window_dilation is not None:
    window_dimensions = _dilate_shape(window_dimensions, window_dilation)
  pads_lo, pads_hi = zip(*padding)
  operand_padded = core.sum_shapes(operand_shape, pads_lo, pads_hi)
  return core.stride_shape(operand_padded, window_dimensions, window_strides)

_reduce_window_max_translation_rule = partial(
    _reduce_window_chooser_translation_rule, max_p, _get_max_identity)
reduce_window_max_p = standard_primitive(
    _common_reduce_window_shape_rule, _input_dtype, 'reduce_window_max',
    _reduce_window_max_translation_rule)
ad.defjvp(reduce_window_max_p, partial(_reduce_window_chooser_jvp_rule, max_p))
batching.primitive_batchers[reduce_window_max_p] = partial(
  _reduce_window_batch_rule, _reduce_window_max)

_reduce_window_min_translation_rule = partial(
    _reduce_window_chooser_translation_rule, min_p, _get_min_identity)
reduce_window_min_p = standard_primitive(
    _common_reduce_window_shape_rule, _input_dtype, 'reduce_window_min',
    _reduce_window_min_translation_rule)
ad.defjvp(reduce_window_min_p, partial(_reduce_window_chooser_jvp_rule, min_p))

_reduce_window_min_batch_rule = partial(_reduce_window_batch_rule,
                                        _reduce_window_min)
batching.primitive_batchers[reduce_window_min_p] = partial(
  _reduce_window_batch_rule, _reduce_window_min)


def _reduce_precision_shape_rule(operand, *, exponent_bits, mantissa_bits):
  exponent_bits = operator.index(exponent_bits)
  mantissa_bits = operator.index(mantissa_bits)
  if exponent_bits < 1:
    raise ValueError(f"reduce_precision: exponent_bits must be positive; got {exponent_bits}")
  if mantissa_bits < 0:
    raise ValueError(f"reduce_precision: mantissa_bits must be non-negative; got {mantissa_bits}")
  return operand.shape


reduce_precision_p = standard_primitive(
    _reduce_precision_shape_rule,
    partial(unop_dtype_rule, _identity, _float, 'reduce_precision'),
    name='reduce_precision')
batching.defvectorized(reduce_precision_p)
masking.defvectorized(reduce_precision_p)


def reduce_precision(operand, exponent_bits, mantissa_bits):
  """Wraps XLA's `ReducePrecision
  <https://www.tensorflow.org/xla/operation_semantics#reduceprecision>`_
  operator.
  """
  exponent_bits = core.concrete_or_error(
    operator.index, exponent_bits, "exponent_bits argument of lax.reduce_precision")
  mantissa_bits = core.concrete_or_error(
    operator.index, mantissa_bits, "mantissa_bits argument of lax.reduce_precision")
  return reduce_precision_p.bind(operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)


def _select_and_scatter_shape_rule(
    operand, source, init_value, *, select_jaxpr, select_consts, scatter_jaxpr,
    scatter_consts, window_dimensions, window_strides, padding):
  _check_shapelike("select_and_scatter", "window_dimensions", window_dimensions)
  _check_shapelike("select_and_scatter", "window_strides", window_strides)
  if len(window_dimensions) != len(window_strides):
    msg = ("select_and_scatter got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))
  return operand.shape

def _select_and_scatter_translation(
  c, operand, source, init_value, *, select_jaxpr, select_consts, scatter_jaxpr,
  scatter_consts, window_dimensions, window_strides, padding):
  select = _reduction_computation(c, select_jaxpr, select_consts, init_value)
  scatter = _reduction_computation(c, scatter_jaxpr, scatter_consts, init_value)
  return xops.SelectAndScatterWithGeneralPadding(
    operand, select, window_dimensions, window_strides, padding, source,
    init_value, scatter)

select_and_scatter_p = standard_primitive(
    _select_and_scatter_shape_rule, _input_dtype, 'select_and_scatter',
    _select_and_scatter_translation)


def _select_and_scatter_add_shape_rule(
    source, operand, *, select_prim, window_dimensions, window_strides,
    padding):
  return operand.shape

def _select_and_scatter_add_translation(
    c, source, operand, *, select_prim, window_dimensions, window_strides,
    padding, expand_padding):
  shape = c.get_shape(operand)
  dtype = shape.numpy_dtype()
  scalar = ShapedArray((), dtype)
  select = xla.primitive_subcomputation(select_prim, scalar, scalar)
  scatter = xla.primitive_subcomputation(add_p, scalar, scalar)
  zero = xb.constant(c, np.array(0, dtype))
  # TODO(b/161704903): remove this workaround when XLA:CPU bug is fixed.
  expand_padding = (expand_padding and
                    not all(lo == 0 and hi == 0 for (lo, hi) in padding))
  if expand_padding:
    original_padding = padding
    identity = (_get_max_identity if select_prim is ge_p
                else _get_min_identity)
    pads = [(lo, hi, 0) for (lo, hi) in padding]
    operand = xops.Pad(operand, xb.constant(c, identity(dtype)),
                       xc.make_padding_config(pads))
    padding = [(0, 0) for _ in padding]
  output = xops.SelectAndScatterWithGeneralPadding(
    operand, select, window_dimensions, window_strides, padding, source, zero,
    scatter)
  if expand_padding:
    start_indices = [lo for (lo, hi) in original_padding]
    stop_indices = [lo + d for ((lo, hi), d) in zip(original_padding,
                                                    shape.dimensions())]
    output = xops.Slice(output, start_indices, stop_indices,
                        [1] * len(start_indices))
  return output

def _select_and_scatter_add_jvp(
    primals, tangents, *, select_prim, window_dimensions, window_strides,
    padding):
  source, operand = primals
  g_source, g_operand = tangents
  val_out = _select_and_scatter_add(
      source, operand, select_prim, window_dimensions, window_strides,
      padding)
  del g_operand
  if type(g_source) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    tangent_out = _select_and_scatter_add(
        g_source, operand, select_prim, window_dimensions,
        window_strides, padding)
  return val_out, tangent_out

def _select_and_scatter_add_transpose(
    t, source, operand, *, select_prim, window_dimensions, window_strides,
    padding):
  assert ad.is_undefined_primal(source) and not ad.is_undefined_primal(operand)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(source.aval), None]
  ones = (1,) * len(window_dimensions)
  source_t = _select_and_gather_add(t, operand, select_prim, window_dimensions,
                                    window_strides, padding, ones, ones)
  return [source_t, None]

def _select_and_scatter_add_batch_rule(
    batched_args, batch_dims, *, select_prim, window_dimensions, window_strides,
    padding):
  source, operand = batched_args
  s_bdim, o_bdim = batch_dims
  size = next(a.shape[bdim] for a, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  source = batching.bdim_at_front(source, s_bdim, size)
  operand = batching.bdim_at_front(operand, o_bdim, size)

  window_dimensions = (1,) + window_dimensions
  window_strides = (1,) + window_strides
  padding = ((0, 0),) + padding
  out = _select_and_scatter_add(source, operand, select_prim, window_dimensions,
                                window_strides, padding)
  return out, 0

select_and_scatter_add_p = standard_primitive(
    _select_and_scatter_add_shape_rule, _input_dtype, 'select_and_scatter_add',
    partial(_select_and_scatter_add_translation, expand_padding=False))

ad.primitive_transposes[select_and_scatter_add_p] = \
    _select_and_scatter_add_transpose
ad.primitive_jvps[select_and_scatter_add_p] = _select_and_scatter_add_jvp
batching.primitive_batchers[select_and_scatter_add_p] = \
    _select_and_scatter_add_batch_rule

# TODO(b/161704903): workaround for XLA/CPU crash.
xla.backend_specific_translations['cpu'][select_and_scatter_add_p] = partial(
    _select_and_scatter_add_translation, expand_padding=True)
# TODO(b/182390722): workaround for XLA/GPU crash.
xla.backend_specific_translations['gpu'][select_and_scatter_add_p] = partial(
    _select_and_scatter_add_translation, expand_padding=True)

def _select_and_gather_add_shape_rule(
    tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  if tangents.shape != operand.shape:
    msg = ("select_and_gather_add tangents and operand shapes must match, "
           "got {} and {}.")
    raise TypeError(msg.format(tangents.shape, operand.shape))
  return _common_reduce_window_shape_rule(
    operand, window_dimensions, window_strides, padding, base_dilation,
    window_dilation)

_UINT_DTYPES = {
  16: np.uint16,
  32: np.uint32,
  64: np.uint64,
}

_INT_DTYPES = {
  16: np.int16,
  32: np.int32,
  64: np.int64,
}

def _select_and_gather_add_translation(
    c, tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation, max_bits=64):
  shape = c.get_shape(operand)
  dtype = shape.numpy_dtype()
  etype = shape.xla_element_type()
  nbits = dtypes.finfo(dtype).bits

  assert nbits <= max_bits
  double_word_reduction = nbits * 2 <= max_bits

  const = lambda c, dtype, x: xb.constant(c, np.array(x, dtype=dtype),
                                          canonicalize_types=False)

  if double_word_reduction:
    # TODO(b/73062247): XLA doesn't yet implement ReduceWindow on tuples, so
    # we implement a pair-wise ReduceWindow by packing two k-bit values into
    # 2k-bit unsigned integer using bit tricks.
    word_dtype = _UINT_DTYPES[nbits]
    double_word_dtype = _UINT_DTYPES[nbits * 2]
    word_type = xla_client.dtype_to_etype(word_dtype)
    double_word_type = xla_client.dtype_to_etype(double_word_dtype)

    # Packs two values into a tuple.
    def pack(a, b):
      a = xops.BitcastConvertType(a, word_type)
      b = xops.BitcastConvertType(b, word_type)
      a = xops.ConvertElementType(a, double_word_type)
      b = xops.ConvertElementType(b, double_word_type)
      a = xops.ShiftLeft(a, const(c, double_word_dtype, nbits))
      return xops.Or(a, b)

    # Unpacks the first element of a tuple.
    def fst(c, t):
      st = xops.ShiftRightLogical(t, const(c, double_word_dtype, nbits))
      return xops.BitcastConvertType(xops.ConvertElementType(st, word_type), etype)

    # Unpacks the second element of a tuple.
    def snd(t):
      return xops.BitcastConvertType(xops.ConvertElementType(t, word_type), etype)

  else:
    # The double-word trick above only works if we have a sufficiently large
    # type. As an alternative, we can pack two half words into a single word,
    # at the cost of precision.
    # TODO(b/73062247): add support for tuple reductions and remove this case.
    warnings.warn("Using reduced precision for gradient of reduce-window "
                  "min/max operator to work around missing XLA support for "
                  "pair-reductions. This is likely from a second or "
                  "higher derivative of a max-pooling operation.")
    r_nbits = nbits // 2
    # Drop/round the bottom mantissa bits.
    nexp = dtypes.finfo(dtype).nexp
    nmant = r_nbits - nexp - 1

    double_word_dtype = word_dtype = _UINT_DTYPES[nbits]
    word_type = xla_client.dtype_to_etype(word_dtype)

    # Packs two values into a tuple.
    def pack(a, b):
      a = xops.ReducePrecision(a, exponent_bits=nexp, mantissa_bits=nmant)
      b = xops.ReducePrecision(b, exponent_bits=nexp, mantissa_bits=nmant)
      a = xops.BitcastConvertType(a, word_type)
      b = xops.BitcastConvertType(b, word_type)
      b = xops.ShiftRightLogical(b, const(c, word_dtype, r_nbits))
      return xops.Or(a, b)

    # Unpacks the first element of a tuple.
    def fst(c, t):
      st = xops.And(t, const(c, word_dtype, ((1 << r_nbits) - 1) << r_nbits))
      return xops.BitcastConvertType(st, etype)

    # Unpacks the second element of a tuple.
    def snd(t):
      return xops.BitcastConvertType(xops.ShiftLeft(t, const(c, word_dtype, r_nbits)),
                                  etype)

  def reducer():
    c = xla_bridge.make_computation_builder("select_and_gather_pair_reducer")
    x = xb.parameter(c, 0,
      xla_client.Shape.array_shape(np.dtype(double_word_dtype), ()))
    y = xb.parameter(c, 1,
      xla_client.Shape.array_shape(np.dtype(double_word_dtype), ()))
    assert select_prim is ge_p or select_prim is le_p
    which = xops.Ge if select_prim is ge_p else xops.Le
    xops.Select(which(fst(c, x), fst(c, y)), x, y)
    return c.build()


  assert select_prim is ge_p or select_prim is le_p, select_prim
  init = -np.inf if select_prim is ge_p else np.inf
  out = xops.ReduceWindowWithGeneralPadding(
    pack(operand, tangents), pack(const(c, dtype, init), const(c, dtype, 0)),
    reducer(), window_dimensions, window_strides, base_dilation,
    window_dilation, padding)
  return snd(out)

# TODO(phawkins): use this translation rule on all platforms.
def _select_and_gather_add_translation_using_variadic_reducewindow(
    c, tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  shape = c.get_shape(operand)
  dtype = shape.numpy_dtype()

  const = lambda c, dtype, x: xb.constant(c, np.array(x, dtype=dtype),
                                          canonicalize_types=False)

  def reducer():
    c = xla_bridge.make_computation_builder("select_and_gather_pair_reducer")
    shape = xla_client.Shape.array_shape(np.dtype(dtype), ())
    kx, vx, ky, vy = (xb.parameter(c, i, shape) for i in range(4))
    which = (xops.Ge if select_prim is ge_p else xops.Le)(kx, ky)
    xops.Tuple(c, [xops.Select(which, kx, ky), xops.Select(which, vx, vy)])
    return c.build()

  assert select_prim is ge_p or select_prim is le_p, select_prim
  init = -np.inf if select_prim is ge_p else np.inf
  out = xops.ReduceWindowWithGeneralPadding(
    [operand, tangents], [const(c, dtype, init), const(c, dtype, 0)],
    reducer(), window_dimensions, window_strides, base_dilation,
    window_dilation, padding)
  return xops.GetTupleElement(out, 1)

def _select_and_gather_add_jvp(
    primals, tangents, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  source, operand = primals
  g_source, g_operand = tangents
  val_out = _select_and_gather_add(
      source, operand, select_prim, window_dimensions, window_strides,
      padding, base_dilation, window_dilation)
  del g_operand
  if type(g_source) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    tangent_out = _select_and_gather_add(
        g_source, operand, select_prim, window_dimensions,
        window_strides, padding, base_dilation, window_dilation)
  return val_out, tangent_out

def _select_and_gather_add_transpose(
    t, tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  assert select_prim in (le_p, ge_p)
  assert ad.is_undefined_primal(tangents) and not ad.is_undefined_primal(operand)
  if any(d != 1 for d in window_dilation):
    msg = ("VJP not implemented for select_and_gather (MaxPool) with window "
           "dilation, got window_dilation={}.")
    raise NotImplementedError(msg.format(window_dilation))
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(tangents.aval), None]
  has_base_dilation = any(d != 1 for d in base_dilation)
  if has_base_dilation:
    select_identity = (_get_max_identity if select_prim is ge_p
                       else _get_min_identity)
    operand = pad(operand, select_identity(operand.dtype),
                  tuple((0, 0, d - 1) for d in base_dilation))
  result = _select_and_scatter_add(t, operand, select_prim, window_dimensions,
                                   window_strides, padding)
  if has_base_dilation:
    result = slice(operand, (0,) * len(operand.shape), operand.shape,
                   base_dilation)
  return [result, None]

def _select_and_gather_add_batching_rule(
    batched_args, batch_dims, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  t, x = batched_args
  t_bdim, x_bdim = batch_dims
  size = next(a.shape[bdim] for a, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  t = batching.bdim_at_front(t, t_bdim, size)
  x = batching.bdim_at_front(x, x_bdim, size)
  window_dimensions = (1,) + window_dimensions
  window_strides = (1,) + window_strides
  padding = ((0, 0),) + padding
  base_dilation = (1,) + base_dilation
  window_dilation = (1,) + window_dilation
  out = _select_and_gather_add(t, x, select_prim, window_dimensions,
                               window_strides, padding, base_dilation,
                               window_dilation)
  return (out, 0)


select_and_gather_add_p = standard_primitive(
    _select_and_gather_add_shape_rule, _input_dtype, 'select_and_gather_add',
    _select_and_gather_add_translation)
ad.primitive_jvps[select_and_gather_add_p] = _select_and_gather_add_jvp
ad.primitive_transposes[select_and_gather_add_p] = \
  _select_and_gather_add_transpose
batching.primitive_batchers[select_and_gather_add_p] = \
  _select_and_gather_add_batching_rule
# TODO(b/183233858): use variadic reducewindow on GPU, when implemented.
if jax.lib._xla_extension_version >= 15:
  xla.backend_specific_translations['cpu'][select_and_gather_add_p] = \
    _select_and_gather_add_translation_using_variadic_reducewindow
  xla.backend_specific_translations['tpu'][select_and_gather_add_p] = \
    _select_and_gather_add_translation_using_variadic_reducewindow
else:
  xla.backend_specific_translations['tpu'][select_and_gather_add_p] = partial(
    _select_and_gather_add_translation, max_bits=32)


def _sort_abstract_eval(*args, **kwargs):
  args = tuple(raise_to_shaped(arg) for arg in args)
  if any(arg.shape != args[0].shape for arg in args[1:]):
    shapes = " ".join(str(a.shape) for a in args)
    raise TypeError(f"Arguments to sort must have equal shapes, got: {shapes}")
  return args


def _float_to_int_for_sort(x):
  # Switch from a floating point value to a integer value in such a way that
  # when using the integer value to compare, we get the same result for normal
  # values, and -nan is treated as the smallest value, and nan is treated as
  # the largest value.
  # If f is a float, and
  # x = bit_cast<int32>(f);
  # y = x < 0 ? int32_max - x : x;
  # then y is ordered as an int32 such that finite values have the obvious
  # order, -0 is ordered before 0, and -NaN and NaN appear at the beginning
  # and end of the ordering.
  # Note that in order to avoid -x to overflow, we calculate
  # int32_max - x as unsigned, and then convert back to signed.
  if x.dtype == dtypes.bfloat16:
    x = convert_element_type(x, np.float32)
  nbits = np.finfo(x).bits
  signed_dtype = _INT_DTYPES[nbits]
  unsigned_dtype = _UINT_DTYPES[nbits]

  signed = bitcast_convert_type(x, signed_dtype)
  unsigned = bitcast_convert_type(x, unsigned_dtype)
  flipped = bitcast_convert_type(
    sub(unsigned_dtype(np.iinfo(signed_dtype).max), unsigned), signed_dtype)
  return select(lt(signed, _zero(signed)), flipped, signed)

# Default comparator that sorts the operands lexicographically on the
# first `num_keys` arguments.
# For floating point types, a total order is created where
# -NaN < -infinity < ... < -0 < 0 < ... < infinity < NaN.
# For complex types, the (real, imag) pairs are sorted lexicographically
# (following NumPy's semantics).
# This code adds complex-number support and lexicographic ordering to the algorithm from:
# https://github.com/tensorflow/tensorflow/blob/ba43780830f09da72081fe5061c436f1c6203a92/tensorflow/compiler/xla/client/lib/comparators.h#L33
def _sort_lt_comparator(*operands, num_keys=1):
  assert len(operands) >= 2 and len(operands) % 2 == 0, operands
  assert len(operands) // 2 >= num_keys, (operands, num_keys)
  x_keys, y_keys = [], []
  for x, y in zip(operands[:2*num_keys:2], operands[1:2*num_keys:2]):
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    if np.issubdtype(x.dtype, np.complexfloating):
      x_keys.extend([_float_to_int_for_sort(real(x)), _float_to_int_for_sort(imag(x))])
      y_keys.extend([_float_to_int_for_sort(real(y)), _float_to_int_for_sort(imag(y))])
    elif np.issubdtype(x.dtype, np.floating):
      x_keys.append(_float_to_int_for_sort(x))
      y_keys.append(_float_to_int_for_sort(y))
    else:
      x_keys.append(x)
      y_keys.append(y)

  p = None
  for xk, yk in zip(x_keys[::-1], y_keys[::-1]):
    p = (bitwise_or(lt(xk, yk), bitwise_and(eq(xk, yk), p)) if p is not None
         else lt(xk, yk))
  return p


def _sort_translation_rule(c, *operands, dimension, is_stable, num_keys):
  types = [c.get_shape(x).xla_element_type() for x in operands]
  subc = xla_bridge.make_computation_builder("sort_lt_comparator")
  params = [xb.parameter(subc, 2 * i + j, xc.Shape.array_shape(typ, ()))
            for i, typ in enumerate(types) for j in range(2)]
  result = xla.lower_fun(partial(_sort_lt_comparator, num_keys=num_keys),
                         multiple_results=False)(subc, *params)
  comparator = subc.build(result)
  out = xops.Sort(c, operands, dimension=dimension, is_stable=is_stable,
                  comparator=comparator)
  return out if len(operands) != 1 else xops.Tuple(c, [out])

def _sort_jvp(primals, tangents, *, dimension, is_stable, num_keys):
  shape = primals[0].shape
  iotas = []
  for dim, size in enumerate(shape):
    dtype = np.int32 if size < np.iinfo(np.int32).max else np.int64
    iotas.append(broadcasted_iota(dtype, shape, dim))
  primals = sort_p.bind(*(primals + (iotas[dimension],)), dimension=dimension,
                        is_stable=is_stable, num_keys=num_keys)
  idx = tuple(primals[-1] if i == dimension else iotas[i]
              for i in range(len(shape)))
  tangents_out = tuple(t if type(t) is ad_util.Zero else t[idx] for t in tangents)
  return tuple(primals[:-1]), tangents_out

def _sort_batch_rule(batched_args, batch_dims, *, dimension, is_stable, num_keys):
  prototype_arg, new_bdim = next(
    (a, b) for a, b in zip(batched_args, batch_dims) if b is not None)
  new_args = []
  for arg, bdim in zip(batched_args, batch_dims):
    if bdim is None:
      dims = np.delete(np.arange(prototype_arg.ndim), new_bdim)
      new_args.append(broadcast_in_dim(arg, prototype_arg.shape, dims))
    else:
      new_args.append(batching.moveaxis(arg, bdim, new_bdim))
  new_dimension = dimension + (new_bdim <= dimension)
  bdims = (new_bdim,) * len(new_args)
  return (sort_p.bind(*new_args, dimension=new_dimension, is_stable=is_stable, num_keys=num_keys),
          bdims)


sort_p = Primitive('sort')
sort_p.multiple_results = True
sort_p.def_impl(partial(xla.apply_primitive, sort_p))
sort_p.def_abstract_eval(_sort_abstract_eval)
xla.translations[sort_p] = _sort_translation_rule
ad.primitive_jvps[sort_p] = _sort_jvp
batching.primitive_batchers[sort_p] = _sort_batch_rule


def _top_k_abstract_eval(operand, *, k):
  if k < 0:
    raise ValueError("k argument to top_k must be nonnegative, got {}".format(k))
  if len(operand.shape) == 0:
    raise TypeError("top_k operand must have >= 1 dimension, got {}"
                    .format(operand.shape))
  shape = list(operand.shape)
  if shape[-1] < k:
    msg = "k argument to top_k must be no larger than minor dimension; {} vs {}"
    raise ValueError(msg.format(k, shape))
  shape[-1] = k
  return (operand.update(shape=shape, dtype=operand.dtype,
                         weak_type=operand.weak_type),
          operand.update(shape=shape, dtype=np.dtype(np.int32)))

def _top_k_jvp(primals, tangents, *, k):
  operand, = primals
  tangent, = tangents
  primals_out = top_k(operand, k)
  if type(tangent) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(primals_out[0])
  else:
    _, k_idxs = primals_out
    idx_shape = k_idxs.shape
    rank = len(idx_shape)
    gather_index_shape = idx_shape + (1,)
    gather_indices = []
    for i in range(rank-1):
      _iota = iota(k_idxs.dtype, idx_shape[i])
      _iota = broadcast_in_dim(_iota, gather_index_shape, (i,))
      gather_indices.append(_iota)
    gather_indices.append(reshape(k_idxs, gather_index_shape))
    gather_indices = concatenate(gather_indices, dimension=rank)
    slice_sizes = (1,) * rank
    dnums = GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=tuple(range(rank)),
      start_index_map=tuple(range(rank)))
    tangent_out = gather(tangent, gather_indices, dnums, slice_sizes)
  return primals_out, (tangent_out, ad_util.Zero.from_value(primals_out[1]))

def _top_k_batch_rule(batched_args, batch_dims, *, k):
  operand, = batched_args
  bdim, = batch_dims
  if bdim == operand.ndim-1:
    perm = np.arange(operand.ndim)
    perm[bdim-1], perm[bdim] = perm[bdim], perm[bdim-1]
    top_k_v, top_k_i = top_k(transpose(operand, perm), k=k)
    return (transpose(top_k_v, perm),
            transpose(top_k_i, perm)), (bdim, bdim)
  else:
    return top_k(operand, k=k), (bdim, bdim)

top_k_p = Primitive('top_k')
top_k_p.multiple_results = True
top_k_p.def_impl(partial(xla.apply_primitive, top_k_p))
top_k_p.def_abstract_eval(_top_k_abstract_eval)
xla.translations[top_k_p] = partial(standard_translate, 'top_k')
ad.primitive_jvps[top_k_p] = _top_k_jvp
batching.primitive_batchers[top_k_p] = _top_k_batch_rule

def _stop_gradient_jvp_rule(primals, tangents):
  # if we don't call stop_gradient here, we'd only peel off one autodiff tracer
  x, = primals
  return stop_gradient(x), ad_util.Zero.from_value(x)

def _stop_gradient_batch_rule(batched_args, batch_dims):
  x, = batched_args
  dim, = batch_dims
  return stop_gradient(x), dim

ad.primitive_jvps[ad_util.stop_gradient_p] = _stop_gradient_jvp_rule
batching.primitive_batchers[ad_util.stop_gradient_p] = _stop_gradient_batch_rule


def create_token(_=None):
  """Creates an XLA token value with no preconditions for sequencing effects.

  Experimental.

  The argument is ignored. It exists for backward compatibility.
  """
  return create_token_p.bind()

create_token_p = Primitive("create_token")
create_token_p.def_impl(partial(xla.apply_primitive, create_token_p))
create_token_p.def_abstract_eval(lambda *_: abstract_token)
xla.translations[create_token_p] = lambda c, *_: xops.CreateToken(c)

def after_all(*operands):
  """Merges one or more XLA token values. Experimental.

  Wraps the XLA AfterAll operator."""
  return after_all_p.bind(*operands)

def _after_all_abstract_eval(*operands):
  if any(x is not abstract_token for x in operands):
    raise TypeError("Arguments to after_all must be tokens")
  return abstract_token


def _after_all_translation_rule(c, *operands):
  return xops.AfterAll(c, operands)

after_all_p = Primitive("after_all")
after_all_p.def_impl(partial(xla.apply_primitive, after_all_p))
after_all_p.def_abstract_eval(_after_all_abstract_eval)
xla.translations[after_all_p] = _after_all_translation_rule


def infeed(token, shape=None, partitions=None):
  """Consumes an infeed value of `shape` from the host. Experimental.

  `token` is used to sequence infeed and outfeed effects.
  `partitions` may be specified inside a `sharded_jit` function.
  """
  flat_shapes, treedef = pytree.flatten(shape)
  for shape in flat_shapes:
    if not isinstance(shape, ShapedArray):
      raise TypeError("shape argument to infeed must be a pytree of "
                      "ShapedArray values, got {}".format(shape))
  if partitions is not None:
    # Always replicate token.
    # We specifically use type() to raise an error for PartitionSpecs.
    if type(partitions) != tuple:  # pylint: disable=unidiomatic-typecheck
      raise ValueError(f"'partitions' argument to infeed should be a tuple, "
                       f"got {partitions}")
    partitions = partitions + (None,)
  xs_and_token = infeed_p.bind(token, shapes=tuple(flat_shapes),
                               partitions=partitions)
  return (treedef.unflatten(xs_and_token[:-1]), xs_and_token[-1])

def _infeed_abstract_eval(token, *, shapes, partitions):
  if token is not abstract_token:
    raise TypeError("First argument to infeed must be a token")
  return shapes + (abstract_token,)


def _infeed_translation_rule(c, token, *, shapes, partitions):
  shape = tuple(shape.with_major_to_minor_layout_if_absent()
                for x in shapes for shape in xla.aval_to_xla_shapes(x))
  build_infeed = partial(xops.InfeedWithToken, token,
                         xla_client.Shape.tuple_shape(shape))
  if partitions:
    xs_and_token = xb.with_sharding(c, partitions, build_infeed)
  else:
    # Note that infeed will default to replication if inside a sharded
    # computation and no sharding is specified.
    xs_and_token = build_infeed()
  xs = xops.GetTupleElement(xs_and_token, 0)
  token = xops.GetTupleElement(xs_and_token, 1)
  outs = [xops.GetTupleElement(xs, i) for i in range(len(shapes))] + [token]
  return xops.Tuple(c, outs)

infeed_p = Primitive("infeed")
infeed_p.multiple_results = True
infeed_p.def_impl(partial(xla.apply_primitive, infeed_p))
infeed_p.def_abstract_eval(_infeed_abstract_eval)
xla.translations[infeed_p] = _infeed_translation_rule

def outfeed(token, xs):
  """Outfeeds value `xs` to the host. Experimental.

  `token` is used to sequence infeed and outfeed effects.
  """
  flat_xs, _ = pytree.flatten(xs)
  return outfeed_p.bind(token, *flat_xs)

def _outfeed_abstract_eval(token, *xs):
  if token is not abstract_token:
    raise TypeError("First argument to outfeed must be a token")
  return abstract_token


def _outfeed_translation_rule(c, token, *xs):
  t = xops.Tuple(c, xs)
  return xops.OutfeedWithToken(t, token, c.get_shape(t))

outfeed_p = Primitive("outfeed")
outfeed_p.def_impl(partial(xla.apply_primitive, outfeed_p))
outfeed_p.def_abstract_eval(_outfeed_abstract_eval)
xla.translations[outfeed_p] = _outfeed_translation_rule

def rng_uniform(a, b, shape):
  """Stateful PRNG generator. Experimental and its use is discouraged.

  Returns uniformly distributed random numbers in the range [a, b)

  You should use jax.random for most purposes; this function exists only for
  niche use cases with special performance requirements.

  This API may be removed at any time.
  """
  return rng_uniform_p.bind(a, b, shape=tuple(shape))

def _rng_uniform_abstract_eval(a, b, *, shape):
  if a.dtype != b.dtype:
    raise ValueError(
      "Arguments to rng_uniform must have identical dtypes, got {} "
      "and {}.".format(a.dtype, b.dtype))
  if a.shape != () or b.shape != ():
    raise ValueError(
      "Arguments to rng_uniform must be scalars; got shapes {} and {}."
      .format(a.shape, b.shape))
  return a.update(shape=shape, dtype=a.dtype,
                  weak_type=(a.weak_type and b.weak_type))

def _rng_uniform_translation_rule(c, a, b, *, shape):
  xla_shape = xc.Shape.array_shape(c.get_shape(a).xla_element_type(), shape)
  return xops.RngUniform(a, b, xla_shape)

rng_uniform_p = Primitive("rng_uniform")
rng_uniform_p.def_impl(partial(xla.apply_primitive, rng_uniform_p))
rng_uniform_p.def_abstract_eval(_rng_uniform_abstract_eval)
xla.translations[rng_uniform_p] = _rng_uniform_translation_rule


def _rng_bit_generator_shape_rule(key, *, shape, dtype, algorithm):
  _ = dtype, algorithm
  return (key.shape, tuple(shape))


def _rng_bit_generator_dtype_rule(key, *, shape, dtype, algorithm):
  _ = key, shape, algorithm
  return (key.dtype, dtype)


def _rng_bit_generator_weak_type_rule(key, *, shape, dtype, algorithm):
  _ = shape, dtype, algorithm
  return (key.weak_type, False)


def _rng_bit_generator_translation_rule(c, key, *, shape, dtype, algorithm):
  _ = c
  xla_shape = xc.Shape.array_shape(np.dtype(dtype), shape)
  return xops.RngBitGenerator(algorithm, key, xla_shape)


def _rng_bit_generator_named_shape_rule(key, *, shape, dtype, algorithm):
  return [key.named_shape, key.named_shape]

rng_bit_generator_p = Primitive("rng_bit_generator")
rng_bit_generator_p.multiple_results = True
rng_bit_generator_p.def_impl(
    partial(xla.apply_primitive, rng_bit_generator_p))
rng_bit_generator_p.def_abstract_eval(
    partial(standard_multi_result_abstract_eval, rng_bit_generator_p,
            _rng_bit_generator_shape_rule, _rng_bit_generator_dtype_rule,
            _rng_bit_generator_weak_type_rule,
            _rng_bit_generator_named_shape_rule))
xla.translations[rng_bit_generator_p] = _rng_bit_generator_translation_rule

RandomAlgorithm = xops.RandomAlgorithm
RandomAlgorithm.__str__ = lambda algorithm: algorithm.name


def rng_bit_generator(key,
                      shape,
                      dtype=np.uint32,
                      algorithm=RandomAlgorithm.RNG_DEFAULT):
  """Stateless PRNG bit generator. Experimental and its use is discouraged.

  Returns uniformly distributed random bits with the specified shape and dtype
  (what is required to be an integer type) using the platform specific
  default algorithm or the one specified.

  It provides direct acces to the RngBitGenerator primitive exposed by XLA
  (https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator) for low
  level API access.

  Most users should use `jax.random` instead for a stable and more user
  friendly API.
  """
  shape = jax.core.canonicalize_shape(shape)
  return tuple(
      rng_bit_generator_p.bind(
          key, shape=shape, dtype=dtype, algorithm=algorithm))


def _iota_abstract_eval(*, dtype, shape, dimension):
  _check_shapelike("iota", "shape", shape)
  if not any(dtypes.issubdtype(dtype, t) for t in _num):
    msg = 'iota does not accept dtype {}. Accepted dtypes are subtypes of {}.'
    typename = str(np.dtype(dtype).name)
    accepted_typenames = (t.__name__ for t in _num)
    raise TypeError(msg.format(typename, ', '.join(accepted_typenames)))
  if not 0 <= dimension < len(shape):
    raise ValueError("iota dimension must be between 0 and len(shape), got "
                     f"dimension={dimension} for shape {shape}")
  return ShapedArray(shape, dtype)

def _iota_translation_rule(c, dtype, shape, dimension):
  etype = xla_client.dtype_to_etype(dtype)
  xla_shape = xc.Shape.array_shape(etype, shape)
  return xops.Iota(c, xla_shape, dimension)

iota_p = Primitive('iota')
iota_p.def_impl(partial(xla.apply_primitive, iota_p))
iota_p.def_abstract_eval(_iota_abstract_eval)
xla.translations[iota_p] = _iota_translation_rule


### util

_ndim = np.ndim


def _dilate_shape(shape, dilation):
  """Utility function for computing the shape resulting from a dilation."""
  if not np.all(np.greater(dilation, 0)):
    msg = "All dilations must be positive, got {}."
    raise TypeError(msg.format(dilation))
  dilation = (1,) * (len(shape) - len(dilation)) + tuple(dilation)
  return core.dilate_shape(shape, dilation)

def _ceil_divide(x1, x2):
  return -np.floor_divide(np.negative(x1), x2)

def padtype_to_pads(in_shape, window_shape, window_strides, padding):
  """Convert padding string to list of pairs of pad values."""
  PaddingType = xla_client.PaddingType

  if isinstance(padding, str):
    mapping = {'VALID': PaddingType.VALID, 'SAME': PaddingType.SAME}
    try:
      padding = mapping[padding.upper()]
    except KeyError as err:
      msg = "Unrecognized padding type: expected 'VALID' or 'SAME', got {}."
      raise RuntimeError(msg.format(padding)) from err

  if padding == PaddingType.SAME:
    out_shape = _ceil_divide(in_shape, window_strides)
    pad_sizes = np.maximum(0, (out_shape - 1) * window_strides +
                                window_shape - in_shape)
    return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
  elif padding == PaddingType.VALID:
    return [(0, 0)] * len(in_shape)
  else:
    msg = "Unknown padding type: {}."
    raise TypeError(msg.format(padding))


def _check_same_dtypes(name, ignore_fp_precision, *ttypes):
  """Check that dtypes agree, possibly ignoring float precision."""
  # the `ignore_fp_precision` flag exists because the XLA shape inference logic
  # allows mixed floating point precision, but the HLO verifier often rejects it
  types = list(map(np.dtype, ttypes))  # canonicalize
  if ignore_fp_precision:
    types = [
        np.floating if dtypes.issubdtype(dtype, np.floating)
        else np.complexfloating if dtypes.issubdtype(dtype, np.complexfloating)
        else dtype for dtype in types]
  if len({dtypes.canonicalize_dtype(t) for t in types}) != 1:
    if ignore_fp_precision:
      msg = ("{} requires arguments to have same dtypes up to floating point "
             "precision, got {}.")
    else:
      msg = "{} requires arguments to have the same dtypes, got {}."
    raise TypeError(msg.format(name, ", ".join(map(str, types))))


def _check_conv_shapes(name, lhs_shape, rhs_shape, window_strides):
  """Check that conv shapes are valid and are consistent with window_strides."""
  if len(lhs_shape) != len(rhs_shape):
    msg = "Arguments to {} must have same rank, got {} and {}."
    raise TypeError(msg.format(name, len(lhs_shape), len(rhs_shape)))
  if len(lhs_shape) < 2:
    msg = "Arguments to {} must have rank at least 2, got {} and {}."
    raise TypeError(msg.format(name, len(lhs_shape), len(rhs_shape)))
  if lhs_shape[1] != rhs_shape[1]:
    msg = "Arguments to {} must agree on input feature size, got {} and {}."
    raise TypeError(msg.format(name, lhs_shape[1], rhs_shape[1]))
  _check_shapelike(name, "window_strides", window_strides)
  if not np.all(np.greater(window_strides, 0)):
    msg = "All elements of window_strides must be positive, got {}."
    raise TypeError(msg.format(window_strides))
  if len(window_strides) != len(lhs_shape) - 2:
    msg = "{} window_strides has wrong length: expected {}, got {}."
    expected_length = len(lhs_shape) - 2
    raise TypeError(msg.format(name, expected_length, len(window_strides)))


def conv_shape_tuple(lhs_shape, rhs_shape, strides, pads, batch_group_count=1):
  """Compute the shape tuple of a conv given input shapes in canonical order."""
  if isinstance(pads, str):
    pads = padtype_to_pads(lhs_shape[2:], rhs_shape[2:], strides, pads)
  if len(pads) != len(lhs_shape) - 2:
    msg = "Wrong number of explicit pads for convolution: expected {}, got {}."
    raise TypeError(msg.format(len(lhs_shape) - 2, len(pads)))

  lhs_padded = np.add(lhs_shape[2:], np.sum(np.array(pads).reshape(-1, 2),
                                              axis=1))
  out_space = core.stride_shape(lhs_padded, rhs_shape[2:], strides)
  out_space = np.maximum(0, out_space)
  if batch_group_count > 1:
    assert lhs_shape[0] % batch_group_count == 0
    out_shape_0 = lhs_shape[0] // batch_group_count
  else:
    out_shape_0 = lhs_shape[0]
  out_shape = (out_shape_0, rhs_shape[0])
  return tuple(out_shape + tuple(out_space))


def conv_general_shape_tuple(lhs_shape, rhs_shape, window_strides, padding,
                             dimension_numbers):
  lhs_perm, rhs_perm, out_perm = conv_general_permutations(dimension_numbers)
  lhs_trans = np.take(lhs_shape, lhs_perm)
  rhs_trans = np.take(rhs_shape, rhs_perm)
  out_trans = conv_shape_tuple(lhs_trans, rhs_trans, window_strides, padding)
  return tuple(np.take(out_trans, np.argsort(out_perm)))


def conv_transpose_shape_tuple(lhs_shape, rhs_shape, window_strides, padding,
                               dimension_numbers):
  lhs_perm, rhs_perm, out_perm = conv_general_permutations(dimension_numbers)
  lhs_trans = np.take(lhs_shape, lhs_perm)
  rhs_trans = np.take(rhs_shape, rhs_perm)
  if isinstance(padding, str):
    padding = [_conv_transpose_padding(k, s, padding)
               for k,s in zip(rhs_trans[2:], window_strides)]
  padding = list(map(np.sum, padding))
  unpad_out_space = [(i-1) * s - k + 2
                     for i, k, s in zip(lhs_trans[2:],
                                        rhs_trans[2:],
                                        window_strides)]
  out_space = np.sum([unpad_out_space, padding], axis=0).tolist()
  out_trans = tuple((lhs_trans[0], rhs_trans[0]) + tuple(out_space))
  return tuple(np.take(out_trans, np.argsort(out_perm)))


def _check_shapelike(fun_name, arg_name, obj, non_zero_shape=False):
  """Check that `obj` is a shape-like value (e.g. tuple of nonnegative ints)."""
  if not isinstance(obj, (tuple, list, np.ndarray)):
    msg = "{} {} must be of type tuple/list/ndarray, got {}."
    raise TypeError(msg.format(fun_name, arg_name, type(obj)))
  # bool(obj) for an ndarray raises an error, so we check len
  if not len(obj):  # pylint: disable=g-explicit-length-test
    return
  obj_arr = np.array(obj)
  if obj_arr.ndim != 1:
    msg = "{} {} must be rank 1, got {}."
    raise TypeError(msg.format(obj_arr.ndim))
  try:
    canonicalize_shape(obj_arr)
  except TypeError as err:
    msg = "{} {} must have every element be an integer type, got {}."
    raise TypeError(msg.format(fun_name, arg_name, tuple(map(type, obj)))) from err
  lower_bound, bound_error = (
      (1, "strictly positive") if non_zero_shape else (0, "nonnegative"))
  if not all(core.greater_equal_dim(d, lower_bound) for d in obj_arr):
    msg = "{} {} must have every element be {}, got {}."
    raise TypeError(msg.format(fun_name, arg_name, bound_error, obj))


def _dynamic_slice_indices(operand, start_indices):
  if len(start_indices) != operand.ndim:
    msg = ("Length of slice indices must match number of operand dimensions ({} "
          "vs {})")
    raise ValueError(msg.format(len(start_indices), operand.shape))
  if not isinstance(start_indices, (tuple, list)):
    if start_indices.ndim != 1:
      raise ValueError("Slice indices must be a 1D sequence, got {}"
                       .format(start_indices.shape))
    return select(lt(start_indices, _zeros(start_indices)),
                  add(start_indices, _const(start_indices, operand.shape)),
                  start_indices)
  else:
    return [np.asarray(i + d if i < 0 else i, getattr(i, 'dtype', dtypes.int_))
            if isinstance(i, (int, np.integer))
            else select(lt(i, _const(i, 0)), add(i, _const(i, d)), i)
            for i, d in zip(start_indices, operand.shape)]


def _const(example, val):
  dtype = _dtype(example)
  if dtypes.is_python_scalar(example):
    val = dtypes.scalar_type_of(example)(val)
    return val if dtype == _dtype(val) else np.array(val, dtype)
  return np.array(val, dtype)

_zeros: Callable = partial(full_like, fill_value=0)
_zero: Callable = partial(full_like, shape=(), fill_value=0)
_ones: Callable = partial(full_like, fill_value=1)
_one: Callable = partial(full_like, shape=(), fill_value=1)
_twos: Callable = partial(full_like, fill_value=2)
_two: Callable = partial(full_like, shape=(), fill_value=2)

dtype: Callable = dtypes.result_type
_dtype: Callable = dtypes.result_type

def _iscomplex(x) -> bool:
  return dtypes.issubdtype(_dtype(x), np.complexfloating)


def ranges_like(*xs):
  start = 0
  for x in xs:
    x_len = len(x)
    yield range(start, start + x_len)
    start += x_len


def remaining(original, *removed_lists):
  removed = set(itertools.chain(*removed_lists))
  return [i for i in original if i not in removed]


def _canonicalize_precision(precision):
  if precision is None:
    if config.jax_default_matmul_precision is None:
      return None
    try:
      return _precision_strings[config.jax_default_matmul_precision]
    except KeyError:
      raise ValueError(
          "jax_default_matmul_precision flag must be set to None or a value in "
          f"{_precision_strings}, but got {config.jax_default_matmul_precision}"
      ) from None
  elif isinstance(precision, str) and precision in _precision_strings:
    return _precision_strings.get(precision)
  elif isinstance(precision, Precision):
    return precision
  elif (isinstance(precision, (list, tuple)) and len(precision) == 2 and
        all(isinstance(p, Precision) for p in precision)):
    return precision
  elif (isinstance(precision, (list, tuple)) and len(precision) == 2 and
        all(isinstance(s, str) for s in precision)):
    s1, s2 = precision
    return (_canonicalize_precision(s1), _canonicalize_precision(s2))
  else:
    raise ValueError(
        f"Precision argument must be None, a string in {_precision_strings}, "
        "a lax.Precision value or a tuple of two lax.Precision values or "
        f"strings; got {precision}.")


def conv_dimension_numbers(lhs_shape, rhs_shape, dimension_numbers
                           ) -> ConvDimensionNumbers:
  """Converts convolution `dimension_numbers` to a `ConvDimensionNumbers`.

  Args:
    lhs_shape: tuple of nonnegative integers, shape of the convolution input.
    rhs_shape: tuple of nonnegative integers, shape of the convolution kernel.
    dimension_numbers: None or a tuple/list of strings or a ConvDimensionNumbers
      object following the convolution dimension number specification format in
      xla_client.py.

  Returns:
    A `ConvDimensionNumbers` object that represents `dimension_numbers` in the
    canonical form used by lax functions.
  """
  if isinstance(dimension_numbers, ConvDimensionNumbers):
    return dimension_numbers
  if len(lhs_shape) != len(rhs_shape):
    msg = "convolution requires lhs and rhs ndim to be equal, got {} and {}."
    raise TypeError(msg.format(len(lhs_shape), len(rhs_shape)))

  if dimension_numbers is None:
    iota = tuple(range(len(lhs_shape)))
    return ConvDimensionNumbers(iota, iota, iota)
  elif isinstance(dimension_numbers, (list, tuple)):
    if len(dimension_numbers) != 3:
      msg = "convolution dimension_numbers list/tuple must be length 3, got {}."
      raise TypeError(msg.format(len(dimension_numbers)))
    if not all(isinstance(elt, str) for elt in dimension_numbers):
      msg = "convolution dimension_numbers elements must be strings, got {}."
      raise TypeError(msg.format(tuple(map(type, dimension_numbers))))
    msg = ("convolution dimension_numbers[{}] must have len equal to the ndim "
           "of lhs and rhs, got {} for lhs and rhs shapes {} and {}.")
    for i, elt in enumerate(dimension_numbers):
      if len(elt) != len(lhs_shape):
        raise TypeError(msg.format(i, len(elt), lhs_shape, rhs_shape))

    lhs_spec, rhs_spec, out_spec = conv_general_permutations(dimension_numbers)
    return ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)
  else:
    msg = "convolution dimension_numbers must be tuple/list or None, got {}."
    raise TypeError(msg.format(type(dimension_numbers)))


def conv_general_permutations(dimension_numbers):
  """Utility for convolution dimension permutations relative to Conv HLO."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  lhs_char, rhs_char, out_char = charpairs = ("N", "C"), ("O", "I"), ("N", "C")
  for i, (a, b) in enumerate(charpairs):
    if not dimension_numbers[i].count(a) == dimension_numbers[i].count(b) == 1:
      msg = ("convolution dimension_numbers[{}] must contain the characters "
             "'{}' and '{}' exactly once, got {}.")
      raise TypeError(msg.format(i, a, b, dimension_numbers[i]))
    if len(dimension_numbers[i]) != len(set(dimension_numbers[i])):
      msg = ("convolution dimension_numbers[{}] cannot have duplicate "
             "characters, got {}.")
      raise TypeError(msg.format(i, dimension_numbers[i]))
  if not (set(lhs_spec) - set(lhs_char) == set(rhs_spec) - set(rhs_char) ==
          set(out_spec) - set(out_char)):
    msg = ("convolution dimension_numbers elements must each have the same "
           "set of spatial characters, got {}.")
    raise TypeError(msg.format(dimension_numbers))

  def getperm(spec, charpair):
    spatial = (i for i, c in enumerate(spec) if c not in charpair)
    if spec is not rhs_spec:
      spatial = sorted(spatial, key=lambda i: rhs_spec.index(spec[i]))
    return (spec.index(charpair[0]), spec.index(charpair[1])) + tuple(spatial)

  lhs_perm, rhs_perm, out_perm = map(getperm, dimension_numbers, charpairs)
  return lhs_perm, rhs_perm, out_perm


def _conv_general_proto(dimension_numbers):
  assert type(dimension_numbers) is ConvDimensionNumbers
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  proto = xla_client.ConvolutionDimensionNumbers()
  proto.input_batch_dimension = lhs_spec[0]
  proto.input_feature_dimension = lhs_spec[1]
  proto.output_batch_dimension = out_spec[0]
  proto.output_feature_dimension = out_spec[1]
  proto.kernel_output_feature_dimension = rhs_spec[0]
  proto.kernel_input_feature_dimension = rhs_spec[1]
  proto.input_spatial_dimensions.extend(lhs_spec[2:])
  proto.kernel_spatial_dimensions.extend(rhs_spec[2:])
  proto.output_spatial_dimensions.extend(out_spec[2:])
  return proto


def _conv_general_vjp_lhs_padding(
    in_shape, window_dimensions, window_strides, out_shape, padding,
    lhs_dilation, rhs_dilation) -> List[Tuple[int, int]]:
  lhs_dilated_shape = _dilate_shape(in_shape, lhs_dilation)
  rhs_dilated_shape = _dilate_shape(window_dimensions, rhs_dilation)
  out_dilated_shape = _dilate_shape(out_shape, window_strides)
  pad_before = np.subtract(rhs_dilated_shape, [lo for lo, _ in padding]) - 1
  pad_after = (np.add(lhs_dilated_shape, rhs_dilated_shape) - 1
               - out_dilated_shape - pad_before)
  return safe_zip(pad_before, pad_after)


def _conv_general_vjp_rhs_padding(
    in_shape, window_dimensions, window_strides, out_shape, padding,
    lhs_dilation, rhs_dilation):
  lhs_dilated_shape = _dilate_shape(in_shape, lhs_dilation)
  rhs_dilated_shape = _dilate_shape(window_dimensions, rhs_dilation)
  out_dilated_shape = _dilate_shape(out_shape, window_strides)
  pads_lo, _ = zip(*padding)
  pads_from_lhs = core.diff_shape(out_dilated_shape, lhs_dilated_shape)
  pads_from_rhs = core.diff_shape(core.diff_shape(rhs_dilated_shape, pads_lo),
                                  (1,) * len(pads_lo))
  pads_hi = core.sum_shapes(pads_from_lhs, pads_from_rhs)
  return list(zip(pads_lo, pads_hi))


def _balanced_eq(x, z, y):
  return div(select(_eq_meet(x, z), _ones(z), _zeros(z)),
             select(_eq_meet(y, z), _twos(z), _ones(z)))


def _eq_meet(a, b):
  a_dtype, b_dtype = _dtype(a), _dtype(b)
  if a_dtype != b_dtype:
    higher_dtype = dtypes.promote_types(a_dtype, b_dtype)
    if higher_dtype == a_dtype:
      a = convert_element_type(a, b_dtype)
    else:
      b = convert_element_type(b, a_dtype)
  return eq(a, b)


def _abstractify(x):
  return raise_to_shaped(core.get_aval(x))


def _check_user_dtype_supported(dtype, fun_name=None):
  # Avoid using `dtype in [...]` becuase of numpy dtype equality overloading.
  if isinstance(dtype, type) and dtype in {bool, int, float, complex}:
    return
  np_dtype = np.dtype(dtype)
  if np_dtype.kind not in "biufc" and np_dtype.type != dtypes.bfloat16:
    msg = f"JAX only supports number and bool dtypes, got dtype {dtype}"
    msg += f" in {fun_name}" if fun_name else ""
    raise TypeError(msg)
  if dtype is not None and np_dtype != dtypes.canonicalize_dtype(dtype):
    msg = ("Explicitly requested dtype {} {} is not available, "
           "and will be truncated to dtype {}. To enable more dtypes, set the "
           "jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell "
           "environment variable. "
           "See https://github.com/google/jax#current-gotchas for more.")
    fun_name = f"requested in {fun_name}" if fun_name else ""
    truncated_dtype = dtypes.canonicalize_dtype(dtype).name
    warnings.warn(msg.format(dtype, fun_name , truncated_dtype), stacklevel=2)


def _canonicalize_axis(axis, num_dims):
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(
        "axis {} is out of bounds for array of dimension {}".format(
            axis, num_dims))
  if axis < 0:
    axis = axis + num_dims
  return axis
