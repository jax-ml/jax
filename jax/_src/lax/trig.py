# Copyright 2026 The JAX Authors.
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

"""Trigonometric functions scaled by pi."""

from functools import partial

from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.lax.lax import (
    RoundingMethod,
    _float,
    bitcast_convert_type,
    convert_element_type,
    full_like,
    polynomial,
    round as lax_round,
    select,
    select_n,
    standard_unop,
)
from jax._src.typing import Array, ArrayLike
import numpy as np

export = util.set_module("jax.lax")

# Polynomial coefficients for sin(pi * r), cos(pi * r), and tan(pi * r) on
# r in [-0.25, 0.25] (u = r * r in [0, 0.0625]), in ascending degree order for
# lax.polynomial.
#
# For sinpi and tanpi, pi is split into pi_hi + pi_lo (with pi_lo stored as the
# constant term of the u-polynomial) and evaluated as:
#   pi_hi * r + r * polynomial(r * r, coeffs)
# so the leading pi * r term keeps full precision when r is small.
#
# Generated with Sollya 8.0:
#   prec = 350!;
#   display = hexadecimal!;
#   u_lo = 1b-120; u_hi = 0.0625;
#   pi_hi = single(pi); pi_lo = single(pi - pi_hi);  /* or double(...) */
#   p_sin = fpminimax(sin(pi * sqrt(u)) / sqrt(u), [|1, ..., deg|],
#                     [|SG...|], [u_lo; u_hi], floating, relative,
#                     pi_hi + pi_lo);
#   p_cos = fpminimax(cos(pi * sqrt(u)), [|1, ..., deg|],
#                     [|SG...|], [u_lo; u_hi], floating, relative, 1);
#   p_tan = fpminimax(tan(pi * sqrt(u)) / sqrt(u), [|1, ..., deg|],
#                     [|SG...|], [u_lo; u_hi], floating, relative,
#                     pi_hi + pi_lo);

_PI_HI_32 = float.fromhex("0x1.921fb6p1")

_SINPI_32 = (
    float.fromhex("-0x1.777a5cp-24"),
    float.fromhex("-0x1.4abbcep2"),
    float.fromhex("0x1.466b84p1"),
    float.fromhex("-0x1.32b5ap-1"),
    float.fromhex("0x1.46c086p-4"),
)

_COSPI_32 = (
    float.fromhex("0x1p0"),
    float.fromhex("-0x1.3bd3ccp2"),
    float.fromhex("0x1.03c1b8p2"),
    float.fromhex("-0x1.55b7cep0"),
    float.fromhex("0x1.d684aap-3"),
)

_TANPI_32 = (
    float.fromhex("-0x1.777a5cp-24"),
    float.fromhex("0x1.4abbdap3"),
    float.fromhex("0x1.466758p5"),
    float.fromhex("0x1.4693p7"),
    float.fromhex("0x1.3cfa48p9"),
    float.fromhex("0x1.8fbfcep11"),
    float.fromhex("0x1.2649cp7"),
    float.fromhex("0x1.ebdc6cp16"),
)

_PI_HI_64 = float.fromhex("0x1.921fb54442d18p1")

_SINPI_64 = (
    float.fromhex("0x1.1a62633145c07p-53"),
    float.fromhex("-0x1.4abbce625be53p2"),
    float.fromhex("0x1.466bc6775ab5bp1"),
    float.fromhex("-0x1.32d2cce635e46p-1"),
    float.fromhex("0x1.5078349480d19p-4"),
    float.fromhex("-0x1.e3075f86a10fdp-8"),
    float.fromhex("0x1.e8fcbf60ee7cfp-12"),
    float.fromhex("-0x1.70298f665471fp-16"),
)

_COSPI_64 = (
    float.fromhex("0x1p0"),
    float.fromhex("-0x1.3bd3cc9be45dep2"),
    float.fromhex("0x1.03c1f081b5a67p2"),
    float.fromhex("-0x1.55d3c7e3c325cp0"),
    float.fromhex("0x1.e1f5067b90e22p-3"),
    float.fromhex("-0x1.a6d1e7297d362p-6"),
    float.fromhex("0x1.f9c89cce8e272p-10"),
    float.fromhex("-0x1.b1673f09c144cp-14"),
)

_TANPI_64 = (
    float.fromhex("0x1.1a62633145c07p-53"),
    float.fromhex("0x1.4abbce625be5bp3"),
    float.fromhex("0x1.466bc67759dd3p5"),
    float.fromhex("0x1.45fff9b50329dp7"),
    float.fromhex("0x1.45f472c2ac0d5p9"),
    float.fromhex("0x1.45f334b562128p11"),
    float.fromhex("0x1.45f26f68ec321p13"),
    float.fromhex("0x1.45febbc45c794p15"),
    float.fromhex("0x1.4553a4c27db68p17"),
    float.fromhex("0x1.4c1213a3785a3p19"),
    float.fromhex("0x1.1a2e32b762664p21"),
    float.fromhex("0x1.1448d89654eb7p24"),
    float.fromhex("-0x1.fc2b28dda3b8bp25"),
    float.fromhex("0x1.2ec954a74a288p30"),
    float.fromhex("-0x1.74d898db81474p32"),
    float.fromhex("0x1.75c0b233b2e6fp34"),
)


def _xor_signbit(y: Array, x: Array) -> Array:
  """Flips the sign bit of `y` wherever `x` has its sign bit set."""
  if y.dtype == np.float64:
    uint_dtype = np.uint64
    sign_mask = np.uint64(1 << 63)
  else:
    uint_dtype = np.uint32
    sign_mask = np.uint32(1 << 31)
  y_u = bitcast_convert_type(y, uint_dtype)
  x_u = bitcast_convert_type(x, uint_dtype)
  return bitcast_convert_type(y_u ^ (x_u & sign_mask), y.dtype)


def _reduce_half_turns(x: Array) -> tuple[Array, Array, Array]:
  """Reduces `|x|` into quarter-turn index `q` in `{0, 1, 2, 3}` and remainder `r` in `[-0.25, 0.25]`."""
  even_int_threshold = 2.0**53 if x.dtype == np.float64 else 2.0**24

  ax = abs(x)
  is_fin = ax < np.inf

  # Every finite float with |x| >= 2^24 (float32) or 2^53 (float64) is already
  # an even integer. Replace those and non-finite inputs with 0.0, then subtract
  # the nearest multiple of 2 so ax_mod2 is in [-1, 1] with no rounding error.
  ax_clamped = select(is_fin & (ax < even_int_threshold), ax, full_like(x, 0.0))
  two_turns = lax_round(0.5 * ax_clamped, RoundingMethod.TO_NEAREST_EVEN)
  ax_mod2 = ax_clamped - 2.0 * two_turns

  # Round 2 * ax_mod2 to the nearest integer k in {-2, -1, 0, 1, 2}, so
  # ax_mod2 = 0.5 * k + r with r in [-0.25, 0.25]. Subtracting 0.5 * k has no
  # rounding error because 0.5 * k is an exact half-integer within 0.25 of
  # ax_mod2, and for |x| < 0.25 both two_turns and k are 0 so subnormal inputs
  # pass through unchanged.
  k = lax_round(2.0 * ax_mod2, RoundingMethod.TO_NEAREST_EVEN)
  r = ax_mod2 - 0.5 * k
  q = convert_element_type(k, np.int32) & 3
  return r, q, is_fin


def _sinpi_poly(r: Array, u: Array) -> Array:
  pi_hi, coeffs = (
      (_PI_HI_64, _SINPI_64)
      if r.dtype == np.float64
      else (_PI_HI_32, _SINPI_32)
  )
  return pi_hi * r + r * polynomial(u, coeffs, unroll=len(coeffs))


def _cospi_poly(u: Array) -> Array:
  coeffs = _COSPI_64 if u.dtype == np.float64 else _COSPI_32
  return polynomial(u, coeffs, unroll=len(coeffs))


def _tanpi_poly(r: Array, u: Array) -> Array:
  pi_hi, coeffs = (
      (_PI_HI_64, _TANPI_64)
      if r.dtype == np.float64
      else (_PI_HI_32, _TANPI_32)
  )
  return pi_hi * r + r * polynomial(u, coeffs, unroll=len(coeffs))


def _sinpi_impl(x: Array) -> Array:
  r, q, is_fin = _reduce_half_turns(x)
  u = r * r
  s = _sinpi_poly(r, u)
  c = _cospi_poly(u)

  # Quarter-turns on |x| >= 0: 0 -> +s, 1 -> +c, 2 -> -s, 3 -> -c.
  res = select_n(q, s, c, -s, -c)
  # Normalize zeros on |x| >= 0 to +0.0 (at odd integers q == 2, -s(+0.0) is
  # -0.0), then apply the sign of x so sinpi(n) has the same sign as n.
  res = select(res == 0.0, full_like(x, 0.0), res)
  res = _xor_signbit(res, x)
  return select(is_fin, res, full_like(x, np.nan))


def _cospi_impl(x: Array) -> Array:
  r, q, is_fin = _reduce_half_turns(x)
  u = r * r
  s = _sinpi_poly(r, u)
  c = _cospi_poly(u)

  # Quarter-turns on |x| >= 0: 0 -> +c, 1 -> -s, 2 -> -c, 3 -> +s.
  res = select_n(q, c, -s, -c, s)
  # At half-integers with q == 1, -s(+0.0) is -0.0; IEEE 754 requires +0.0 at
  # all half-integers.
  res = select(res == 0.0, full_like(x, 0.0), res)
  return select(is_fin, res, full_like(x, np.nan))


def _tanpi_impl(x: Array) -> Array:
  r, q, is_fin = _reduce_half_turns(x)
  u = r * r
  t = _tanpi_poly(r, u)

  # For odd q (near half-integers), tan(pi * (0.5 * k + r)) = -1 / tan(pi * r).
  # Replace t with 1.0 when q is even before dividing, so small inputs near 0 do
  # not divide by 0 in the unused -1 / t branch.
  odd_q = (q & 1) != 0
  safe_t = select(odd_q, t, full_like(x, 1.0))
  res = select(odd_q, -1.0 / safe_t, t)

  # Exact zeros at integers and signed infinities at half-integers on |x| >= 0:
  #   q == 0 (even |n|):       +0.0
  #   q == 1 (even |n| + 0.5): +inf
  #   q == 2 (odd |n|):        -0.0
  #   q == 3 (odd |n| + 0.5):  -inf
  # Flipping the sign bit when x < 0 preserves tanpi(-x) = -tanpi(x) everywhere.
  zero_or_pole = select_n(
      q,
      full_like(x, 0.0),
      full_like(x, np.inf),
      full_like(x, -0.0),
      full_like(x, -np.inf),
  )
  res = select(r == 0.0, zero_or_pole, res)
  res = _xor_signbit(res, x)
  return select(is_fin, res, full_like(x, np.nan))


def _trig_unary_lowering(impl, ctx, x):
  aval = ctx.avals_in[0]

  def _lower(v):
    if aval.dtype not in (np.float32, np.float64):
      return convert_element_type(
          impl(convert_element_type(v, np.float32)), aval.dtype
      )
    return impl(v)

  return mlir.lower_fun(_lower, multiple_results=False)(ctx, x)


@export
def sinpi(x: ArrayLike) -> Array:
  r"""Elementwise sine of :math:`\pi x`: :math:`\mathrm{sin}(\pi x)`.

  Implements the IEEE 754-2019 ``sinPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{sin}(\pi x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.sin`: elementwise sine in radians.
    - :func:`jax.lax.cospi`: elementwise cosine of :math:`\pi x`.
    - :func:`jax.lax.tanpi`: elementwise tangent of :math:`\pi x`.
  """
  return sinpi_p.bind(x)


@export
def cospi(x: ArrayLike) -> Array:
  r"""Elementwise cosine of :math:`\pi x`: :math:`\mathrm{cos}(\pi x)`.

  Implements the IEEE 754-2019 ``cosPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{cos}(\pi x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.cos`: elementwise cosine in radians.
    - :func:`jax.lax.sinpi`: elementwise sine of :math:`\pi x`.
    - :func:`jax.lax.tanpi`: elementwise tangent of :math:`\pi x`.
  """
  return cospi_p.bind(x)


@export
def tanpi(x: ArrayLike) -> Array:
  r"""Elementwise tangent of :math:`\pi x`: :math:`\mathrm{tan}(\pi x)`.

  Implements the IEEE 754-2019 ``tanPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{tan}(\pi x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.tan`: elementwise tangent in radians.
    - :func:`jax.lax.sinpi`: elementwise sine of :math:`\pi x`.
    - :func:`jax.lax.cospi`: elementwise cosine of :math:`\pi x`.
  """
  return tanpi_p.bind(x)


sinpi_p = standard_unop(_float, "sinpi")
ad.defjvp(sinpi_p, lambda g, x: g * (np.pi * cospi(x)))
mlir.register_lowering(
    sinpi_p, partial(_trig_unary_lowering, _sinpi_impl), inline=False
)

cospi_p = standard_unop(_float, "cospi")
ad.defjvp(cospi_p, lambda g, x: -g * (np.pi * sinpi(x)))
mlir.register_lowering(
    cospi_p, partial(_trig_unary_lowering, _cospi_impl), inline=False
)

tanpi_p = standard_unop(_float, "tanpi")
ad.defjvp2(tanpi_p, lambda g, ans, x: g * (np.pi * (1.0 + ans * ans)))
mlir.register_lowering(
    tanpi_p, partial(_trig_unary_lowering, _tanpi_impl), inline=False
)
