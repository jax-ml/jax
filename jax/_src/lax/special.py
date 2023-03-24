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
from functools import partial

from jax._src.lax.lax import (bitwise_and, bitwise_not, bitwise_or,
                              broadcast_in_dim, broadcast_shapes,
                              convert_element_type, eq, exp, full_like,
                              gt, le, log, log1p, lt, mul, neg, reciprocal,
                              reduce, select, sign, square, standard_naryop,
                              standard_unop, xla, xops, ne, div, sub, add,
                              _broadcast_translate, _const, _dtype, _float,
                              _nary_lower_hlo, _ones, _isnan, _reduce)
from jax._src.lax.control_flow import while_loop
from jax._src.lax.utils import (standard_translate)

from jax._src import dtypes
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.lib.mlir.dialects import chlo
from jax._src.typing import Array, ArrayLike

def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise regularized incomplete beta integral."""
  return regularized_incomplete_beta_p.bind(a, b, x)

def lgamma(x: ArrayLike) -> Array:
  r"""Elementwise log gamma: :math:`\mathrm{log}(\Gamma(x))`."""
  return lgamma_p.bind(x)

def digamma(x: ArrayLike) -> Array:
  r"""Elementwise digamma: :math:`\psi(x)`."""
  return digamma_p.bind(x)

def igamma(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise regularized incomplete gamma function."""
  return igamma_p.bind(a, x)

def igammac(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise complementary regularized incomplete gamma function."""
  return igammac_p.bind(a, x)

def igamma_grad_a(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise derivative of the regularized incomplete gamma function."""
  return igamma_grad_a_p.bind(a, x)

def random_gamma_grad(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise derivative of samples from `Gamma(a, 1)`."""
  return random_gamma_grad_p.bind(a, x)

def bessel_i0e(x: ArrayLike) -> Array:
  r"""Exponentially scaled modified Bessel function of order 0:
  :math:`\mathrm{i0e}(x) = e^{-|x|} \mathrm{i0}(x)`
  """
  return bessel_i0e_p.bind(x)

def bessel_i1e(x: ArrayLike) -> Array:
  r"""Exponentially scaled modified Bessel function of order 1:
  :math:`\mathrm{i1e}(x) = e^{-|x|} \mathrm{i1}(x)`
  """
  return bessel_i1e_p.bind(x)

def erf(x: ArrayLike) -> Array:
  r"""Elementwise error function: :math:`\mathrm{erf}(x)`."""
  return erf_p.bind(x)

def erfc(x: ArrayLike) -> Array:
  r"""Elementwise complementary error function:
    :math:`\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)`."""
  return erfc_p.bind(x)

def erf_inv(x: ArrayLike) -> Array:
  r"""Elementwise inverse error function: :math:`\mathrm{erf}^{-1}(x)`."""
  return erf_inv_p.bind(x)


regularized_incomplete_beta_p = standard_naryop(
    [_float, _float, _float], 'regularized_incomplete_beta')
xla.register_translation(
    regularized_incomplete_beta_p,
    partial(_broadcast_translate, xops.RegularizedIncompleteBeta))

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

def igamma_gradx(g, a, x):
  return g * exp(-x + (a - _ones(a)) * log(x) - lgamma(a))

def igamma_grada(g, a, x):
  return g * igamma_grad_a(a, x)

def igammac_gradx(g, a, x):
  return -igamma_gradx(g, a, x)

def igammac_grada(g, a, x):
  return -igamma_grada(g, a, x)

# The below is directly ported from tensorflow/compiler/xla/client/lib/math.cc
# We try to follow the corresponding functions as closely as possible, so that
# we can quickly incorporate changes.
class IgammaMode(Enum):
  VALUE = 1
  DERIVATIVE = 2
  SAMPLE_DERIVATIVE = 3

def _any(predicates: Array) -> Array:
  f = _const(predicates, False)
  predicates_shape = predicates.shape
  all_dimensions = tuple(range(len(predicates_shape)))
  return reduce(predicates, f, bitwise_or, all_dimensions)

def _igamma_series(ax, x, a, enabled, dtype, mode):
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

  dlogax_da = log(x) - digamma(a + _const(a, 1))

  if mode == IgammaMode.DERIVATIVE:
    return ax * (ans * dlogax_da + dans_da) / a
  elif mode == IgammaMode.SAMPLE_DERIVATIVE:
    return -(dans_da + ans * dlogax_da) * x / a
  else:
    raise ValueError("Invalid IgammaMode")

def igamma_impl(a, x, dtype):
  is_nan = bitwise_or(_isnan(a), _isnan(x))
  x_is_zero = eq(x, _const(x, 0))
  x_is_infinity = eq(x, _const(x, float('inf')))
  domain_error = bitwise_or(lt(x, _const(x, 0)), le(a, _const(a, 0)))
  use_igammac = bitwise_and(gt(x, _const(x, 1)), gt(x, a))
  ax = a * log(x) - x - lgamma(a)
  underflow = lt(ax, -log(dtypes.finfo(dtype).max))
  ax = exp(ax)
  enabled = bitwise_not(
      _reduce(bitwise_or,[x_is_zero, domain_error, underflow, is_nan]))

  output = select(
    use_igammac,
    _const(a, 1) -
      _igammac_continued_fraction(ax, x, a, bitwise_and(enabled, use_igammac),
                                  dtype, IgammaMode.VALUE),
    _igamma_series(ax, x, a, bitwise_and(enabled, bitwise_not(use_igammac)),
                   dtype, IgammaMode.VALUE)
  )
  output = select(x_is_zero, full_like(a, 0), output)
  output = select(x_is_infinity, full_like(a, 1), output)
  output = select(bitwise_or(domain_error, is_nan),
                  full_like(a, float('nan')), output)
  return output

def _igammac_continued_fraction(ax, x, a, enabled, dtype, mode):
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
  dans_da = vals[14]
  dlogax_da = log(x) -  digamma(a)

  if mode == IgammaMode.DERIVATIVE:
    return mul(ax, add(mul(ans, dlogax_da), dans_da))
  elif mode == IgammaMode.SAMPLE_DERIVATIVE:
    return neg(add(dans_da, mul(ans, dlogax_da)) * x)
  else:
    raise ValueError(f"Invalid mode: {mode}")


def igammac_impl(a, x, dtype):
  out_of_range = bitwise_or(le(x, _const(x, 0)), le(a, _const(a, 0)))
  use_igamma = bitwise_or(lt(x, _const(x, 1)), lt(x, a))
  ax = a * log(x) - x - lgamma(a)
  underflow = lt(ax, -log(dtypes.finfo(dtype).max))
  enabled = bitwise_not(bitwise_or(out_of_range, underflow))
  ax = exp(ax)

  igamma_call = _igamma_series(ax, x, a, bitwise_and(enabled, use_igamma),
                               dtype, IgammaMode.VALUE)
  igammac_cf_call = _igammac_continued_fraction(ax, x, a,
    bitwise_and(enabled, bitwise_not(use_igamma)), dtype, IgammaMode.VALUE)

  result = select(use_igamma, _const(a, 1) - igamma_call, igammac_cf_call)
  x_is_infinity = eq(x, _const(x, float('inf')))
  result = select(x_is_infinity, full_like(result, 0), result)
  return select(out_of_range, full_like(a, 1), result)

def igamma_grad_a_impl(a, x, dtype):
  is_nan = bitwise_or(_isnan(a), _isnan(x))
  x_is_zero = eq(x, full_like(x,0))
  domain_error = bitwise_or(lt(x, full_like(x, 0)), le(a, full_like(a, 0)))
  use_igammac = bitwise_and(gt(x, full_like(x,1)), gt(x, a))
  ax = a * log(x) - x - lgamma(a)
  underflow = lt(ax, -log(dtypes.finfo(dtype).max))
  ax = exp(ax)
  enabled = bitwise_not(bitwise_or(bitwise_or(bitwise_or(
      x_is_zero, domain_error), underflow), is_nan))
  output = select(use_igammac,
    -_igammac_continued_fraction(ax, x, a, bitwise_and(enabled, use_igammac),
                                 dtype, IgammaMode.DERIVATIVE),
    _igamma_series(ax, x, a, bitwise_and(enabled, bitwise_not(use_igammac)),
                   dtype, IgammaMode.DERIVATIVE))
  output = select(x_is_zero, full_like(output,0), output)
  output = select(bitwise_or(domain_error, is_nan),
                  full_like(a, float('nan')), output)
  return output

def _up_and_broadcast(doit):
  def up_and_broadcast(a, x):
    broadcasted_shape = broadcast_shapes(a.shape, x.shape)
    a = broadcast_in_dim(a, broadcasted_shape, list(range(a.ndim)))
    x = broadcast_in_dim(x, broadcasted_shape, list(range(x.ndim)))

    needs_upcast = a.dtype == dtypes.bfloat16 or a.dtype == np.float16
    if needs_upcast:
      a_dtype = a.dtype
      a = convert_element_type(a, np.float32)
      x = convert_element_type(x, np.float32)
      a_x_type = np.float32
    else:
      a_x_type = a.dtype

    result = doit(a, x, a_x_type)
    if needs_upcast:
      result = convert_element_type(result, a_dtype)
    return result
  return up_and_broadcast

lgamma_p = standard_unop(_float, 'lgamma')
ad.defjvp(lgamma_p, lambda g, x: mul(g, digamma(x)))
mlir.register_lowering(lgamma_p, partial(_nary_lower_hlo, chlo.LgammaOp))

digamma_p = standard_unop(_float, 'digamma')
mlir.register_lowering(digamma_p, partial(_nary_lower_hlo, chlo.DigammaOp))

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

random_gamma_grad_p = standard_naryop([_float, _float], 'random_gamma_grad')
xla.register_translation(random_gamma_grad_p,
                         partial(_broadcast_translate, xops.RandomGammaGrad))

bessel_i0e_p = standard_unop(_float, 'bessel_i0e')
xla.register_translation(bessel_i0e_p, standard_translate(bessel_i0e_p))
ad.defjvp2(bessel_i0e_p, lambda g, y, x: g * (bessel_i1e(x) - sign(x) * y))

bessel_i1e_p = standard_unop(_float, 'bessel_i1e')
mlir.register_lowering(bessel_i1e_p,
                        partial(_nary_lower_hlo, chlo.BesselI1eOp))

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
mlir.register_lowering(erf_p, partial(_nary_lower_hlo, chlo.ErfOp))

erfc_p = standard_unop(_float, 'erfc')
ad.defjvp(erfc_p, lambda g, x: mul(_const(x, -2. / np.sqrt(np.pi)),
                                   mul(g, exp(neg(square(x))))))
mlir.register_lowering(erfc_p, partial(_nary_lower_hlo, chlo.ErfcOp))

erf_inv_p = standard_unop(_float, 'erf_inv')
ad.defjvp2(erf_inv_p, lambda g, ans, x: mul(_const(x, np.sqrt(np.pi) / 2.),
                                            mul(g, exp(square(ans)))))
mlir.register_lowering(erf_inv_p, partial(_nary_lower_hlo, chlo.ErfInvOp))
