# Copyright 2023 The JAX Authors.
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
# limitations under the License

""" Special functions

LAX decompositions for special functions into their StableHLO counterparts.
"""

from enum import Enum
import numpy as np
from functools import partial, reduce as _reduce

from jax._src import config
from jax._src import core
from jax._src.custom_partitioning_sharding_rule import (
    sdy_sharding_rule_to_mlir,
    str_to_sdy_sharding_rule,
)
from jax._src import ffi
from jax._src.lax.bessel import bessel_i0e_impl, bessel_i1e_impl
from jax._src.lax.erf import erf_inv_core, erfc_impl, exp_neg_sq
from jax._src.lax.lax import (AccuracyMode, Tolerance, add, bitwise_and,
                              bitwise_not, bitwise_or, broadcast_in_dim,
                              broadcast_shapes, convert_element_type, div, eq,
                              exp, full_like, ge, gt, le, log, log1p, lt, mul,
                              ne, neg, reciprocal, reduce, select, sign,
                              square, standard_naryop, standard_unop, sub,
                              _const, _dtype, _float, _nary_lower_hlo, _ones,
                              _isnan, _unary_with_accuracy_pp_rule)
from jax._src.lax.control_flow.loops import while_loop
from jax._src.lax.digamma import digamma_impl
from jax._src.lax.lgamma import lgamma_impl

from jax._src import dtypes
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.lib import cpu_math, cuda_math, hip_math
from jax._src.lib.mlir.dialects import chlo
from jax._src.typing import Array, ArrayLike

if cpu_math is not None:
  for _name, _value in cpu_math.registrations().items():
    ffi.register_ffi_target(_name, _value, platform='cpu', api_version=1)
    ffi.register_ffi_target_as_batch_partitionable(_name)
for _platform, _plugin_math in (('CUDA', cuda_math), ('ROCM', hip_math)):
  if _plugin_math is not None:
    for _name, _value in _plugin_math.registrations().items():
      ffi.register_ffi_target(_name, _value, platform=_platform, api_version=1)
      ffi.register_ffi_target_as_batch_partitionable(_name)

# TODO(mattjj): this function sucks, delete it
def _up_and_broadcast(doit):
  def up_and_broadcast(*args):
    broadcasted_shape = broadcast_shapes(*(a.shape for a in args))
    args = [broadcast_in_dim(a, broadcasted_shape, list(range(a.ndim))) for a in args]

    a_dtype = args[0].dtype
    needs_upcast = a_dtype == dtypes.bfloat16 or a_dtype == np.float16
    if needs_upcast:
      args = [convert_element_type(a, np.float32) for a in args]
      a_x_type = np.float32
    else:
      a_x_type = a_dtype
    result = doit(*args, dtype=a_x_type)
    if needs_upcast:
      result = convert_element_type(result, a_dtype)
    return result
  return up_and_broadcast

def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise regularized incomplete beta integral."""
  a, b, x = core.auto_insert_reshard(a, b, x)
  return regularized_incomplete_beta_p.bind(a, b, x)

def lgamma(x: ArrayLike) -> Array:
  r"""Elementwise log gamma: :math:`\mathrm{log}(\Gamma(x))`."""
  return lgamma_p.bind(x)

def digamma(x: ArrayLike) -> Array:
  r"""Elementwise digamma: :math:`\psi(x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.
  """
  return digamma_p.bind(x)

def polygamma(m: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise polygamma: :math:`\psi^{(m)}(x)`."""
  m, x = core.auto_insert_reshard(m, x)
  return polygamma_p.bind(m, x)

def igamma(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise regularized incomplete gamma function."""
  a, x = core.auto_insert_reshard(a, x)
  return igamma_p.bind(a, x)

def igammac(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise complementary regularized incomplete gamma function."""
  a, x = core.auto_insert_reshard(a, x)
  return igammac_p.bind(a, x)

def igamma_grad_a(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise derivative of the regularized incomplete gamma function."""
  a, x = core.auto_insert_reshard(a, x)
  return igamma_grad_a_p.bind(a, x)

@_up_and_broadcast
def random_gamma_grad(a: ArrayLike, x: ArrayLike, *, dtype) -> Array:
  r"""Elementwise derivative of samples from `Gamma(a, 1)`."""
  a, x = core.auto_insert_reshard(a, x)
  return random_gamma_grad_impl(a, x, dtype=dtype)

def zeta(x: ArrayLike, q: ArrayLike) -> Array:
  r"""Elementwise Hurwitz zeta function: :math:`\zeta(x, q)`"""
  x, q = core.auto_insert_reshard(x, q)
  return zeta_p.bind(x, q)

def bessel_i0e(x: ArrayLike) -> Array:
  r"""Exponentially scaled modified Bessel function of order 0:

  :math:`\mathrm{i0e}(x) = e^{-|x|} \mathrm{i0}(x)`

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.
  """
  return bessel_i0e_p.bind(x)

def bessel_i1e(x: ArrayLike) -> Array:
  r"""Exponentially scaled modified Bessel function of order 1:

  :math:`\mathrm{i1e}(x) = e^{-|x|} \mathrm{i1}(x)`

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.
  """
  return bessel_i1e_p.bind(x)

def erf(x: ArrayLike) -> Array:
  r"""Elementwise error function: :math:`\mathrm{erf}(x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.
  """
  return erf_p.bind(x)

def erfc(
    x: ArrayLike, *, accuracy: Tolerance | AccuracyMode | None = None
) -> Array:
  r"""Elementwise complementary error function:

    :math:`\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)`.

  Args:
    x: input array. Must have floating-point type.
    accuracy: Optional `lax.Tolerance` or `lax.AccuracyMode` object that
      selects the implementation of the op based on the requested accuracy. If
      the implementation cannot satisfy the requested tolerance, the
      compiler will return an error. If mode is specified and there are no
      multiple implementations available, the default implementation will be
      used.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.
  """
  return erfc_p.bind(x, accuracy=accuracy)

def erf_inv(x: ArrayLike) -> Array:
  r"""Elementwise inverse error function: :math:`\mathrm{erf}^{-1}(x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.
  """
  return erf_inv_p.bind(x)

def betainc_gradx(g, a, b, x):
  lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
  partial_x = exp((b - 1) * log1p(-x) +
                  (a - 1) * log(x) - lbeta)
  return partial_x * g

def betainc_grad_not_implemented(g, a, b, x):
  raise ValueError("Betainc gradient with respect to a and b not supported.")

def igamma_gradx(g, a, x):
  return g * exp(-x + (a - _ones(a)) * log(x) - lgamma(a))

def igamma_grada(g, a, x):
  return g * igamma_grad_a(a, x)

def igammac_gradx(g, a, x):
  return -igamma_gradx(g, a, x)

def igammac_grada(g, a, x):
  return -igamma_grada(g, a, x)

def polygamma_gradm(g, m, x):
  raise ValueError("polygamma gradient with respect to m is not supported")

def polygamma_gradx(g, m, x):
  return g * polygamma(add(m, _const(m, 1)), x)

# The below is directly ported from tensorflow/compiler/xla/client/lib/math.cc
# We try to follow the corresponding functions as closely as possible, so that
# we can quickly incorporate changes.
def lentz_thompson_barnett_algorithm(*,num_iterations, small, threshold, nth_partial_numerator, nth_partial_denominator, inputs):
  # Loop state:
  #   iteration: position in the evaluation.
  #   unconverged: per-element mask of values that have not reached the
  #     desired tolerance.
  #   c: ratio between nth canonical numerator and the nth-1 canonical
  #     numerator.
  #   d: ratio between nth-1 canonical denominator and the nth canonical
  #     denominator.
  #   h: computed approximant in the evaluation.

  def while_cond_fn(state):
    iterations_remain_cond = lt(state["iteration"], num_iterations)
    values_unconverged_cond = _any(state["unconverged"])
    return bitwise_and(iterations_remain_cond, values_unconverged_cond)

  def while_body_fn(state):
    iteration = state["iteration"]
    unconverged = state["unconverged"]
    partial_numerator = nth_partial_numerator(iteration, *inputs)
    partial_denominator = nth_partial_denominator(iteration, *inputs)

    c = add(partial_denominator, div(partial_numerator, state["c"]))
    small_constant = full_like(c, small)
    c = select(lt(abs(c), small_constant), small_constant, c)
    d = add(partial_denominator, mul(partial_numerator, state["d"]))
    d = select(lt(abs(d), small_constant), small_constant, d)
    d = reciprocal(d)
    delta = mul(c, d)
    h = mul(state["h"], delta)

    # Elements that have already converged keep their state, so the result
    # for each element does not depend on the other elements.
    # An element stays unconverged while its delta is outside the tolerance.
    tolerance_comparison = ge(abs(sub(delta, _const(delta, 1.0))), threshold)
    return {
        "iteration": iteration + 1,
        "unconverged": bitwise_and(unconverged, tolerance_comparison),
        "c": select(unconverged, c, state["c"]),
        "d": select(unconverged, d, state["d"]),
        "h": select(unconverged, h, state["h"]),
    }

  partial_denominator = nth_partial_denominator(0, *inputs)
  h = select(lt(abs(partial_denominator), small),
             broadcast_in_dim(small, partial_denominator.shape, ()),
             partial_denominator)
  state = {
      "iteration": 1,
      "unconverged": full_like(h, True, dtype=bool),
      "c": h,
      "d": full_like(h, 0),
      "h": h,
  }
  state = while_loop(while_cond_fn, while_body_fn, state)
  return state["h"]


def regularized_incomplete_beta_impl(a, b, x, *, dtype):
  shape = a.shape

  def nth_partial_betainc_numerator(iteration, a, b, x):
    """
    The partial numerator for the incomplete beta function is given
    here: http://dlmf.nist.gov/8.17.E23 Note that there is a special
    case: the partial numerator for the first iteration is one.
    """
    iteration_bcast = broadcast_in_dim(iteration, shape, [])
    iteration_is_even = eq(iteration_bcast % full_like(iteration_bcast, 2),
                           full_like(iteration_bcast, 0))
    iteration_is_one = eq(iteration_bcast, full_like(iteration_bcast, 1))
    iteration_minus_one = iteration_bcast - full_like(iteration_bcast, 1)
    m = iteration_minus_one // full_like(iteration_minus_one, 2)
    m_is_zero = eq(m, full_like(m, 0))
    m = convert_element_type(m, dtype)
    one = full_like(a, 1)
    two = full_like(a, 2.0)
    # Partial numerator terms

    # When a is close to zero and m == 0, using zero_numerator avoids
    # inaccuracies when FTZ or DAZ is enabled:
    zero_numerator = -(a + b) * x / (a + one)
    even_numerator = select(m_is_zero, zero_numerator,
                            -(a + m) * (a + b + m) * x / (
                              (a + two * m) * (a + two * m + one)))
    odd_numerator = m * (b - m) * x / ((a + two * m - one) * (a + two * m))
    one_numerator = full_like(x, 1.0)
    numerator = select(iteration_is_even, even_numerator, odd_numerator)
    return select(iteration_is_one, one_numerator, numerator)

  def nth_partial_betainc_denominator(iteration, a, b, x):
    iteration_bcast = broadcast_in_dim(iteration, shape, [])
    return select(eq(iteration_bcast, full_like(iteration_bcast, 0)),
                  full_like(x, 0), full_like(x, 1))

  a_is_zero = bitwise_or(eq(a, full_like(a, 0)), eq(b, full_like(b, float('inf'))))
  b_is_zero = bitwise_or(eq(b, full_like(b, 0)), eq(a, full_like(a, float('inf'))))
  x_is_zero = eq(x, full_like(x, 0))
  x_is_one = eq(x, full_like(x, 1))
  x_is_not_zero = bitwise_not(x_is_zero)
  x_is_not_one = bitwise_not(x_is_one)
  is_nan = bitwise_or(bitwise_or(_isnan(a), _isnan(b)), _isnan(x))

  result_is_zero = bitwise_or(bitwise_and(b_is_zero, x_is_not_one), bitwise_and(a_is_zero, x_is_zero))
  result_is_one = bitwise_or(bitwise_and(a_is_zero, x_is_not_zero), bitwise_and(b_is_zero, x_is_one))

  result_is_nan = bitwise_or(bitwise_or(bitwise_or(
    lt(a, full_like(a, 0)), lt(b, full_like(b, 0))),
    lt(x, full_like(x, 0))), gt(x, full_like(x, 1)))
  result_is_nan = bitwise_or(result_is_nan, bitwise_or(bitwise_and(a_is_zero, b_is_zero), is_nan))

  # The continued fraction will converge rapidly when x <
  # (a+1)/(a+b+2) as per: http://dlmf.nist.gov/8.17.E23.
  #
  # Otherwise, we can rewrite using the symmetry relation as per:
  # http://dlmf.nist.gov/8.17.E4
  converges_rapidly = lt(x, (a + full_like(a, 1)) / (a + b + full_like(b, 2.0)))
  a_orig = a
  a = select(converges_rapidly, a, b)
  b = select(converges_rapidly, b, a_orig)
  x = select(converges_rapidly, x, sub(full_like(x, 1), x))

  continued_fraction = lentz_thompson_barnett_algorithm(
    num_iterations=200 if dtype == np.float32 else 600,
    small=(dtypes.finfo(dtype).eps / 2).astype(dtype),
    threshold=(dtypes.finfo(dtype).eps / 2).astype(dtype),
    nth_partial_numerator=nth_partial_betainc_numerator,
    nth_partial_denominator=nth_partial_betainc_denominator,
    inputs=[a, b, x]
  )

  # For very small a and to avoid division by zero, we'll use
  # a * gamma(a) = gamma(a + 1) -> 1 as a -> 0+.
  very_small = (dtypes.finfo(dtype).tiny * 2).astype(dtype)
  lbeta_ab_small_a = lgamma(b) - lgamma(a + b)
  lbeta_ab = lgamma(a) + lbeta_ab_small_a
  factor = select(lt(a, full_like(a, very_small)),
                  exp(log1p(-x) * b - lbeta_ab_small_a),
                  exp(log(x) * a + log1p(-x) * b - lbeta_ab) / a)
  result = continued_fraction * factor
  result = select(converges_rapidly, result, sub(full_like(result, 1), result))

  result = select(result_is_zero, full_like(a, 0), result)
  result = select(result_is_one, full_like(a, 1), result)
  result = select(result_is_nan, full_like(a, float('nan')), result)
  return result

class IgammaMode(Enum):
  VALUE = 1
  DERIVATIVE = 2
  SAMPLE_DERIVATIVE = 3

def _any(predicates: Array) -> Array:
  f = _const(predicates, False)
  predicates_shape = predicates.shape
  all_dimensions = tuple(range(len(predicates_shape)))
  return reduce(predicates, f, bitwise_or, all_dimensions)

def _igamma_series(ax, x, a, enabled, dtype, mode, dlogax_da):
  def cond_fn(vals):
    return _any(vals[0])

  def body_fn(vals):
    enabled, r, c, ans, x, dc_da, dans_da = vals

    r = r + _const(r, 1.)
    dc_da = dc_da * (x / r) - (c * x) / (r * r)
    dans_da = dans_da + dc_da
    c = c * (x / r)
    ans = ans + c

    if mode == IgammaMode.VALUE:
      conditional = bitwise_and(enabled, c / ans > dtypes.finfo(dtype).eps)
    else:
      conditional = bitwise_and(enabled,
                                abs(dc_da / dans_da) >  dtypes.finfo(dtype).eps)

    # TODO: Make this a vmap. Might be tricky with the imports.
    return (
      conditional,
      select(enabled, r, vals[1]),
      select(enabled, c, vals[2]),
      select(enabled, ans, vals[3]),
      select(enabled, x, vals[4]),
      select(enabled, dc_da, vals[5]),
      select(enabled, dans_da, vals[6]),
    )

  init_vals = (
    enabled, a, full_like(a, 1), full_like(a, 1), x, full_like(a, 0),
    full_like(a, 0),
  )

  vals = while_loop(cond_fn, body_fn, init_vals)
  ans = vals[3]
  dans_da = vals[6]

  if mode == IgammaMode.VALUE:
    return (ans * ax) / a

  assert dlogax_da is not None

  if mode == IgammaMode.DERIVATIVE:
    return ax * (ans * dlogax_da + dans_da) / a
  elif mode == IgammaMode.SAMPLE_DERIVATIVE:
    return -(dans_da + ans * dlogax_da) * x / a
  else:
    raise ValueError("Invalid IgammaMode")

def igamma_impl(a, x, *, dtype):
  is_nan = bitwise_or(_isnan(a), _isnan(x))
  x_is_infinity = eq(x, _const(x, float('inf')))
  a_is_infinity = eq(a, _const(a, float('inf')))
  a_is_zero = eq(a, _const(a, 0))
  x_is_zero = eq(x, _const(x, 0))
  domain_error = _reduce(
      bitwise_or,
      [
          lt(x, _const(x, 0)),
          lt(a, _const(a, 0)),
          bitwise_and(a_is_zero, x_is_zero),
          bitwise_and(a_is_infinity, x_is_infinity),
          is_nan,
      ],
  )

  use_igammac = bitwise_and(ge(x, _const(x, 1)), gt(x, a))
  ax = a * log(x) - x - lgamma(a)
  underflow = lt(ax, -log(dtypes.finfo(dtype).max))
  ax = exp(ax)
  enabled = bitwise_not(
      _reduce(
          bitwise_or,
          [x_is_zero, domain_error, underflow, x_is_infinity, a_is_infinity],
      )
  )

  output = select(
    use_igammac,
    _const(a, 1) -
      _igammac_continued_fraction(ax, x, a, bitwise_and(enabled, use_igammac),
                                  dtype, IgammaMode.VALUE, None),
    _igamma_series(ax, x, a, bitwise_and(enabled, bitwise_not(use_igammac)),
                   dtype, IgammaMode.VALUE, None)
  )
  output = select(a_is_infinity, full_like(a, 0), output)
  output = select(x_is_zero, full_like(a, 0), output)
  output = select(x_is_infinity, full_like(a, 1), output)
  output = select(domain_error, full_like(a, float('nan')), output)
  return output

def _igammac_continued_fraction(ax, x, a, enabled, dtype, mode, dlogax_da):
  eps = dtypes.finfo(dtype).eps

  def cond_fn(vals):
    enabled, _ans, _t, _y, _x, c, *_ = vals
    return bitwise_and(c < _const(c, 2000), _any(enabled))

  def body_fn(vals):
    (enabled, ans, t, y, z, c, pkm1, qkm1, pkm2, qkm2,
       dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da) = vals

    c = c + _const(c, 1)
    y = y + _const(y, 1)
    z = z + _const(z, 2)
    yc = y * c
    pk = pkm1 * z - pkm2 * yc
    qk = qkm1 * z - qkm2 * yc
    qk_is_nonzero = ne(qk, _const(qk, 0))
    r = pk / qk

    t = select(qk_is_nonzero, abs(div(sub(ans, r), r)), full_like(r, 1))
    ans = select(qk_is_nonzero, r, ans)

    dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c
    dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c
    dans_da_new = select(qk_is_nonzero, div(dpk_da - ans * dqk_da, qk), dans_da)
    grad_conditional = select(qk_is_nonzero,
                              abs(dans_da_new - dans_da),
                              full_like(dans_da, 1))

    pkm2 = pkm1
    pkm1 = pk
    qkm2 = qkm1
    qkm1 = qk

    dpkm2_da = dpkm1_da
    dqkm2_da = dqkm1_da
    dpkm1_da = dpk_da
    dqkm1_da = dqk_da

    rescale = gt(abs(pk), reciprocal(_const(pk, eps)))
    pkm2 = select(rescale, mul(pkm2, _const(pkm2, eps)), pkm2)
    pkm1 = select(rescale, mul(pkm1, _const(pkm1, eps)), pkm1)
    qkm2 = select(rescale, mul(qkm2, _const(qkm2, eps)), qkm2)
    qkm1 = select(rescale, mul(qkm1, _const(qkm1, eps)), qkm1)

    dpkm2_da = select(rescale, mul(dpkm2_da, _const(dpkm2_da, eps)), dpkm2_da)
    dqkm2_da = select(rescale, mul(dqkm2_da, _const(dqkm2_da, eps)), dqkm2_da)
    dpkm1_da = select(rescale, mul(dpkm1_da, _const(dpkm1_da, eps)), dpkm1_da)
    dqkm1_da = select(rescale, mul(dqkm1_da, _const(dqkm1_da, eps)), dqkm1_da)

    if mode == IgammaMode.VALUE:
      conditional = bitwise_and(enabled, t > eps)
    else:
      conditional = bitwise_and(enabled,
        grad_conditional > _const(grad_conditional, eps))

    return (conditional,
         select(enabled, ans, vals[1]),
         select(enabled, t, vals[2]),
         select(enabled, y, vals[3]),
         select(enabled, z, vals[4]),
         c,
         select(enabled, pkm1, vals[6]),
         select(enabled, qkm1, vals[7]),
         select(enabled, pkm2, vals[8]),
         select(enabled, qkm2, vals[9]),
         select(enabled, dpkm2_da, vals[10]),
         select(enabled, dqkm2_da, vals[11]),
         select(enabled, dpkm1_da, vals[12]),
         select(enabled, dqkm1_da, vals[13]),
         select(enabled, dans_da_new, vals[14]))

  y = _const(a, 1) - a
  z = x + y + _const(x, 1)
  c = _const(x, 0)
  pkm2 = full_like(x, 1)
  qkm2 = x
  pkm1 = x + _const(x, 1)
  qkm1 = z * x
  ans = pkm1 / qkm1
  t = full_like(x, 1)
  dpkm2_da = full_like(x, 0)
  dqkm2_da = full_like(x, 0)
  dpkm1_da = full_like(x, 0)
  dqkm1_da = -x
  dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1
  init_vals = (enabled,  ans,    t,    y,    z,
               c,    pkm1,   qkm1,   pkm2,   qkm2,
               dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da)

  vals = while_loop(cond_fn, body_fn, init_vals)
  ans = vals[1]
  if mode == IgammaMode.VALUE:
    return ans *  ax
  assert dlogax_da is not None
  dans_da = vals[14]

  if mode == IgammaMode.DERIVATIVE:
    return mul(ax, add(mul(ans, dlogax_da), dans_da))
  elif mode == IgammaMode.SAMPLE_DERIVATIVE:
    return neg(add(dans_da, mul(ans, dlogax_da)) * x)
  else:
    raise ValueError(f"Invalid mode: {mode}")

def igammac_impl(a, x, *, dtype):
  is_nan = bitwise_or(_isnan(a), _isnan(x))
  a_is_zero = eq(a, _const(a, 0))
  x_is_zero = eq(x, _const(x, 0))
  x_is_infinity = eq(x, _const(x, float('inf')))
  a_is_infinity = eq(a, _const(a, float('inf')))
  domain_error = _reduce(
      bitwise_or,
      [
          lt(x, _const(x, 0)),
          lt(a, _const(a, 0)),
          bitwise_and(a_is_zero, x_is_zero),
          bitwise_and(a_is_infinity, x_is_infinity),
          is_nan,
      ],
  )
  use_igamma = bitwise_or(lt(x, _const(x, 1)), lt(x, a))
  ax = a * log(x) - x - lgamma(a)
  underflow = lt(ax, -log(dtypes.finfo(dtype).max))
  enabled = bitwise_not(
      _reduce(
          bitwise_or,
          [domain_error, underflow, x_is_infinity, a_is_zero, a_is_infinity],
      )
  )
  ax = exp(ax)

  igamma_call = _igamma_series(ax, x, a, bitwise_and(enabled, use_igamma),
                               dtype, IgammaMode.VALUE, None)
  igammac_cf_call = _igammac_continued_fraction(ax, x, a,
    bitwise_and(enabled, bitwise_not(use_igamma)), dtype, IgammaMode.VALUE,
    None)

  output = select(use_igamma, _const(a, 1) - igamma_call, igammac_cf_call)
  output = select(a_is_infinity, full_like(a, 1), output)
  output = select(bitwise_or(x_is_infinity, a_is_zero), full_like(output, 0), output)
  output = select(domain_error, full_like(a, float('nan')), output)
  return output

def igamma_grad_a_impl(a, x, *, dtype):
  is_nan = bitwise_or(_isnan(a), _isnan(x))
  x_is_zero = eq(x, full_like(x,0))
  domain_error = bitwise_or(lt(x, full_like(x, 0)), le(a, full_like(a, 0)))
  use_igammac = bitwise_and(gt(x, full_like(x,1)), gt(x, a))
  log_x = log(x)
  ax = a * log_x - x - lgamma(a)
  underflow = lt(ax, -log(dtypes.finfo(dtype).max))
  ax = exp(ax)
  enabled = bitwise_not(bitwise_or(bitwise_or(bitwise_or(
      x_is_zero, domain_error), underflow), is_nan))
  dlogax_da = log_x - digamma(select(use_igammac, a, a + _const(a, 1)))
  output = select(use_igammac,
    -_igammac_continued_fraction(ax, x, a, bitwise_and(enabled, use_igammac),
                                 dtype, IgammaMode.DERIVATIVE, dlogax_da),
    _igamma_series(ax, x, a, bitwise_and(enabled, bitwise_not(use_igammac)),
                   dtype, IgammaMode.DERIVATIVE, dlogax_da))
  output = select(x_is_zero, full_like(output,0), output)
  output = select(bitwise_or(domain_error, is_nan),
                  full_like(a, float('nan')), output)
  return output

def random_gamma_grad_impl(a, x, *, dtype):
  is_nan = bitwise_or(_isnan(a), _isnan(x))
  x_is_zero = eq(x, full_like(x,0))
  domain_error = bitwise_or(lt(x, full_like(x,0)), le(a, full_like(a,0)))
  use_igammac = bitwise_and(gt(x, full_like(x,1)), gt(x, a))
  log_x = log(x)
  ax = a * log_x - x - lgamma(a)
  underflow = lt(ax, -log(dtypes.finfo(a.dtype).max))
  ax = exp(ax)
  enabled = bitwise_not(bitwise_or(bitwise_or(bitwise_or
    (x_is_zero, domain_error), underflow), is_nan))
  dlogax_da = log_x - digamma(select(use_igammac, a, a + _const(a, 1)))
  output = select(use_igammac,
    -_igammac_continued_fraction(ax, x, a, bitwise_and(enabled, use_igammac),
                                 dtype, IgammaMode.SAMPLE_DERIVATIVE, dlogax_da),
    _igamma_series(ax, x, a, bitwise_and(enabled, bitwise_not(use_igammac)),
                                         dtype, IgammaMode.SAMPLE_DERIVATIVE,
                                         dlogax_da))
  output = select(x_is_zero, full_like(output,0), output)
  output = select(bitwise_or(domain_error, is_nan),
                  full_like(a, float('nan')), output)
  return output


regularized_incomplete_beta_p = standard_naryop(
    [_float, _float, _float], 'regularized_incomplete_beta')
mlir.register_lowering(regularized_incomplete_beta_p,
  mlir.lower_fun(_up_and_broadcast(regularized_incomplete_beta_impl),
                 multiple_results=False))

ad.defjvp(regularized_incomplete_beta_p,
  betainc_grad_not_implemented,
  betainc_grad_not_implemented,
  betainc_gradx)


def _lgamma_lowering(ctx, x, *, use_f64_for_f32=False):
  aval = ctx.avals_in[0]
  def _impl(v):
    if aval.dtype not in (np.float32, np.float64):
      return convert_element_type(
          lgamma_impl(
              convert_element_type(v, np.float32),
              use_f64_for_f32=use_f64_for_f32,
          ),
          aval.dtype,
      )
    return lgamma_impl(v, use_f64_for_f32=use_f64_for_f32)
  return mlir.lower_fun(_impl, multiple_results=False)(ctx, x)


def _unary_math_ffi_lowering(target_name: str, fallback_lowering):
  def rule(ctx, x):
    if ctx.is_forward_compat():
      return fallback_lowering(ctx, x)
    x_aval, = ctx.avals_in
    num_batch_dims = len(x_aval.shape)
    frontend_attrs = mlir.ir_attribute({'num_batch_dims': str(num_batch_dims)})
    extra_attributes = {'mhlo.frontend_attributes': frontend_attrs}
    # For polymorphic shapes, ffi_lowering appends shape operands
    # (indices_of_shape_operands) that StablehloCanonicalizeDynamism later
    # drops without updating sdy.sharding_rule, so no single operand mapping
    # count can pass verification both before and after shape refinement.
    if core.is_constant_shape(x_aval.shape):
      in_types = list(mlir.aval_to_ir_types(ctx.module_context, x_aval))
      out_types = list(mlir.aval_to_ir_types(ctx.module_context, ctx.avals_out[0]))
      extra_attributes['sdy.sharding_rule'] = sdy_sharding_rule_to_mlir(
          str_to_sdy_sharding_rule('... -> ...'),
          in_types,
          out_types,
      )
    return ffi.ffi_lowering(
        target_name, extra_attributes=extra_attributes
    )(ctx, x)
  return rule


_cpu_lgamma_lowering = partial(_lgamma_lowering, use_f64_for_f32=True)
lgamma_p = standard_unop(_float, 'lgamma')
ad.defjvp(lgamma_p, lambda g, x: mul(g, digamma(x)))
mlir.register_lowering(lgamma_p, _lgamma_lowering, inline=False)
# TODO(phawkins): Make this unconditional after jax 0.12.1 is released.
if cpu_math is not None:
  mlir.register_lowering(
      lgamma_p,
      _unary_math_ffi_lowering('cpu_lgamma_ffi', _cpu_lgamma_lowering),
      platform='cpu',
  )
else:
  # Only CPU uses XLA_LATE; XLA:GPU does not inline XLA_LATE calls (turning each
  # op inside into a separate kernel), and XLA:TPU's late CallInliner hits
  # tile-layout mismatches inside fused loops.
  mlir.register_lowering(
      lgamma_p,
      _cpu_lgamma_lowering,
      platform='cpu',
      inline=core.Inline.XLA_LATE,
  )
if cuda_math is not None:
  mlir.register_lowering(
      lgamma_p,
      _unary_math_ffi_lowering('cu_lgamma_ffi', _lgamma_lowering),
      platform='cuda',
  )
if hip_math is not None:
  mlir.register_lowering(
      lgamma_p,
      _unary_math_ffi_lowering('hip_lgamma_ffi', _lgamma_lowering),
      platform='rocm',
  )

def _digamma_lowering(
    ctx, x, *, use_f64_for_f32=False, use_optimization_barrier=False
):
  aval = ctx.avals_in[0]
  def _impl(v):
    if aval.dtype not in (np.float32, np.float64):
      return convert_element_type(
          digamma_impl(
              convert_element_type(v, np.float32),
              use_f64_for_f32=use_f64_for_f32,
              use_optimization_barrier=use_optimization_barrier,
          ),
          aval.dtype,
      )
    return digamma_impl(
        v,
        use_f64_for_f32=use_f64_for_f32,
        use_optimization_barrier=use_optimization_barrier,
    )
  return mlir.lower_fun(_impl, multiple_results=False)(ctx, x)


_cpu_digamma_lowering = partial(_digamma_lowering, use_f64_for_f32=True)
digamma_p = standard_unop(_float, 'digamma')
ad.defjvp(digamma_p, lambda g, x: mul(g, polygamma(_const(x, 1), x)))
mlir.register_lowering(digamma_p, _digamma_lowering, inline=False)
# TODO(phawkins): Make this unconditional after jax 0.12.1 is released.
if cpu_math is not None:
  mlir.register_lowering(
      digamma_p,
      _unary_math_ffi_lowering('cpu_digamma_ffi', _cpu_digamma_lowering),
      platform='cpu',
  )
else:
  mlir.register_lowering(
      digamma_p,
      _cpu_digamma_lowering,
      platform='cpu',
      inline=core.Inline.XLA_LATE,
  )
if cuda_math is not None:
  mlir.register_lowering(
      digamma_p,
      _unary_math_ffi_lowering('cu_digamma_ffi', _digamma_lowering),
      platform='cuda',
  )
if hip_math is not None:
  mlir.register_lowering(
      digamma_p,
      _unary_math_ffi_lowering('hip_digamma_ffi', _digamma_lowering),
      platform='rocm',
  )
mlir.register_lowering(
    digamma_p,
    partial(_digamma_lowering, use_optimization_barrier=True),
    platform='tpu',
    inline=False,
)

polygamma_p = standard_naryop([_float, _float], 'polygamma')
mlir.register_lowering(polygamma_p, partial(_nary_lower_hlo, chlo.polygamma))
ad.defjvp(polygamma_p, polygamma_gradm, polygamma_gradx)

igamma_p = standard_naryop([_float, _float], 'igamma')
mlir.register_lowering(igamma_p, mlir.lower_fun(_up_and_broadcast(igamma_impl),
                                                multiple_results=False))

igamma_grad_a_p = standard_naryop([_float, _float], 'igamma_grad_a')
mlir.register_lowering(igamma_grad_a_p,
                       mlir.lower_fun(_up_and_broadcast(igamma_grad_a_impl),
                                      multiple_results=False))

ad.defjvp(igamma_p, igamma_grada, igamma_gradx)

igammac_p = standard_naryop([_float, _float], 'igammac')
mlir.register_lowering(igammac_p,
                       mlir.lower_fun(_up_and_broadcast(igammac_impl),
                                      multiple_results=False))

ad.defjvp(igammac_p, igammac_grada, igammac_gradx)

zeta_p = standard_naryop([_float, _float], 'zeta')
mlir.register_lowering(zeta_p, partial(_nary_lower_hlo, chlo.zeta))

bessel_i0e_p = standard_unop(_float, 'bessel_i0e')
mlir.register_lowering(bessel_i0e_p,
                       mlir.lower_fun(bessel_i0e_impl,
                                      multiple_results=False))
ad.defjvp2(bessel_i0e_p, lambda g, y, x: g * (bessel_i1e(x) - sign(x) * y))

bessel_i1e_p = standard_unop(_float, 'bessel_i1e')
mlir.register_lowering(bessel_i1e_p,
                       mlir.lower_fun(bessel_i1e_impl,
                                      multiple_results=False))

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
                                  mul(g, exp_neg_sq(x))))
mlir.register_lowering(erf_p, partial(_nary_lower_hlo, chlo.erf))

erfc_p = standard_unop(_float, 'erfc')
ad.defjvp(
    erfc_p,
    lambda g, x, **kwargs: mul(
        _const(x, -2.0 / np.sqrt(np.pi)), mul(g, exp_neg_sq(x))
    ),
)
def _erfc_lowering(ctx, x, *, accuracy=None):
  if (
      accuracy is AccuracyMode.HIGHEST
      and ctx.avals_in[0].dtype in (np.float32, np.float64)
  ):
    return mlir.lower_fun(erfc_impl, multiple_results=False)(ctx, x)
  return _nary_lower_hlo(chlo.erfc, ctx, x)
mlir.register_lowering(erfc_p, _erfc_lowering)
core.pp_eqn_rules[erfc_p] = _unary_with_accuracy_pp_rule

erf_inv_p = standard_unop(_float, 'erf_inv')
ad.defjvp2(erf_inv_p, lambda g, ans, x: mul(_const(x, np.sqrt(np.pi) / 2.),
                                            mul(g, exp(square(ans)))))

def erf_inv_impl(x: Array) -> Array:
  if x.dtype not in (np.float32, np.float64):
    return convert_element_type(
        erf_inv_impl(convert_element_type(x, np.float32)), x.dtype
    )
  ax = abs(x)
  one = _const(x, 1.0)
  out_of_bounds = bitwise_or(gt(ax, one), _isnan(x))
  is_one = eq(ax, one)
  x_safe = select(bitwise_or(out_of_bounds, is_one), full_like(x, 0.0), x)
  q_safe = sub(one, abs(x_safe))
  res = erf_inv_core(x_safe, q_safe, has_far_tail=False)
  return select(
      out_of_bounds,
      full_like(x, np.nan),
      select(is_one, mul(x, _const(x, np.inf)), res),
  )

def _erf_inv_lowering(ctx, x):
  if config.jax_accurate_erf_inv.value:
    return mlir.lower_fun(erf_inv_impl, multiple_results=False)(ctx, x)
  return _nary_lower_hlo(chlo.erf_inv, ctx, x)

mlir.register_lowering(erf_inv_p, _erf_inv_lowering)
