from functools import partial

import scipy.special as osp_special

from ... import api
from ...numpy import lax_numpy as jnp
from ...numpy.lax_numpy import _wraps


@_wraps(osp_special.ellipk)
@api.custom_jvp
def ellipk(m):
  machep = jnp.finfo(m.dtype).eps
  # Fixed polynomial coefficients
  P = [
      1.37982864606273237150E-4,
      2.28025724005875567385E-3,
      7.97404013220415179367E-3,
      9.85821379021226008714E-3,
      6.87489687449949877925E-3,
      6.18901033637687613229E-3,
      8.79078273952743772254E-3,
      1.49380448916805252718E-2,
      3.08851465246711995998E-2,
      9.65735902811690126535E-2,
      1.38629436111989062502E0
  ]

  Q = [
      2.94078955048598507511E-5,
      9.14184723865917226571E-4,
      5.94058303753167793257E-3,
      1.54850516649762399335E-2,
      2.39089602715924892727E-2,
      3.01204715227604046988E-2,
      3.73774314173823228969E-2,
      4.88280347570998239232E-2,
      7.03124996963957469739E-2,
      1.24999999999870820058E-1,
      4.99999999999999999821E-1
  ]
  C1 = 1.3862943611198906188E0  # log(4)

  x = 1 - m

  large_mask = x > 1
  x = large_mask * jnp.reciprocal(x) + (1 - large_mask) * x
  div_factor = large_mask * jnp.sqrt(x) + (1 - large_mask) * 1

  small_mask = x <= machep
  small_value = C1 - 0.5 * jnp.log(x)

  value = jnp.polyval(P, x) - jnp.log(x) * jnp.polyval(Q, x)
  value = small_mask * small_value + (1 - small_mask) * value
  value = value / div_factor

  is_nan = x < 0
  nans = jnp.sqrt((1 - is_nan * 2).astype(dtype=m.dtype))

  return value * nans


@ellipk.defjvp
def ellipk_jvp(primals, tangents):
  m, = primals
  m_dot, = tangents
  k = ellipk(m)
  e = ellipe(m)
  k_dot = e / (m - m ** 3) - k / m
  return k, k_dot * m_dot


@_wraps(osp_special.ellipe)
@api.custom_jvp
def ellipe(m):
  # Fixed polynomial coefficients
  P = [
      1.53552577301013293365E-4,
      2.50888492163602060990E-3,
      8.68786816565889628429E-3,
      1.07350949056076193403E-2,
      7.77395492516787092951E-3,
      7.58395289413514708519E-3,
      1.15688436810574127319E-2,
      2.18317996015557253103E-2,
      5.68051945617860553470E-2,
      4.43147180560990850618E-1,
      1.00000000000000000299E0
  ]
  Q = [
      3.27954898576485872656E-5,
      1.00962792679356715133E-3,
      6.50609489976927491433E-3,
      1.68862163993311317300E-2,
      2.61769742454493659583E-2,
      3.34833904888224918614E-2,
      4.27180926518931511717E-2,
      5.85936634471101055642E-2,
      9.37499997197644278445E-2,
      2.49999999999888314361E-1
  ]

  x = 1 - m

  large_mask = x > 1
  x = large_mask * (1 - jnp.reciprocal(x)) + (1 - large_mask) * x
  mul_factor = large_mask * jnp.sqrt(x) + (1 - large_mask) * 1

  value = jnp.polyval(P, x) - jnp.log(x) * (x * jnp.polyval(Q, x))
  value = value * mul_factor

  is_nan = x < 0
  nans = jnp.sqrt((1 - is_nan * 2).astype(dtype=m.dtype))

  return value * nans


@ellipe.defjvp
def ellipe_jvp(primals, tangents):
  m, = primals
  m_dot, = tangents
  k = ellipk(m)
  e = ellipe(m)
  e_dot = (e - k) / m
  return e, e_dot * m_dot


@_wraps(osp_special.ellipj)
@partial(api.custom_jvp, nondiff_argnums=[1])
def ellipj(u, m):
  machep = jnp.finfo(u.dtype).eps
  # First if statement
  t = jnp.sin(u)
  b = jnp.cos(u)
  ai = 0.25 * m * (u - t * b)
  sn = t - ai * b
  cn = b + ai * t
  ph = u - ai
  dn = 1.0 - 0.5 * m * t * t
  out_1 = (sn, cn, dn, ph)
  mask_1 = m < 1.0e-9

  # Second if statement
  ai = 0.25 * (1.0 - m)
  b = jnp.cosh(u)
  t = jnp.tanh(u)
  phi = 1.0 / b
  twon = b * jnp.sinh(u)
  sn = t + ai * (twon - u) / (b * b)
  ph = 2.0 * jnp.arctan(jnp.exp(u)) - jnp.pi / 2 + ai * (twon - u) / b
  ai = ai * t * phi
  cn = phi - ai * (twon - u)
  dn = phi + ai * (twon + u)
  out_2 = (sn, cn, dn, ph)
  mask_2 = m >= 0.9999999999

  # Main computation
  a = [jnp.ones_like(u)]
  c = [jnp.sqrt(m)]
  done_mask = [jnp.fabs(c[0] / a[0]) < machep]
  b = jnp.sqrt(1.0 - m)
  twon = 1.0

  for i in range(8):
    ai = a[-1]
    c.append((ai - b) / 2)
    t = jnp.sqrt(ai * b)
    a.append((ai + b) / 2)
    b = t
    twon = twon * done_mask[-1] + (1 - done_mask[-1]) * (twon * 2)
    done_mask.append(jnp.fabs(c[-1] / a[-1]) < machep)
    done_mask[-1] = jnp.logical_or(done_mask[-1], done_mask[-2])

  assert len(a) == 9 and len(c) == 9

  # backward recurrence
  phi = 0.0
  for i in range(8, 0, -1):
    # Initial value
    phi_start = twon * a[i] * u
    is_start = done_mask[i] if i == 0 else done_mask[i] != done_mask[i - 1]
    phi = is_start * phi_start + (1 - is_start) * phi
    # Standard recursion
    t = c[i] * jnp.sin(phi) / a[i]
    b = phi
    phi = (jnp.arcsin(t) + phi) / 2

  sn = jnp.sin(phi)
  t = jnp.cos(phi)
  cn = t
  dnfac = jnp.cos(phi - b)

  # See discussion after DLMF 22.20.5
  cond = jnp.fabs(dnfac) < 0.1
  dn_true = jnp.sqrt(1 - m * sn * sn)
  dn_false = t / dnfac
  dn = cond * dn_true + (1 - cond) * dn_false
  ph = phi
  out_3 = (sn, cn, dn, ph)

  # Produce NaNs instead of error
  not_nan = jnp.logical_or(mask_1, mask_2)
  not_nan = jnp.logical_or(not_nan, done_mask[-1])
  is_nan = jnp.logical_or(m < 0, m > 1)
  is_nan = jnp.logical_or(is_nan, jnp.logical_not(not_nan))
  nans = jnp.sqrt((1 - is_nan * 2).astype(dtype=u.dtype))

  def mask_output(x1, x2, x3):
    y = mask_1 * x1 + (1 - mask_1) * (mask_2 * x2 + (1 - mask_2) * x3)
    return nans * y

  return [mask_output(x1, x2, x3) for x1, x2, x3 in zip(out_1, out_2, out_3)]


@ellipj.defjvp
def ellipj_jvp(m, primals, tangents):
  u,  = primals
  u_dot, = tangents
  sn, cn, dn, ph = ellipj(u, m)
  t_sn = cn * dn * u_dot
  t_cn = - sn * dn * u_dot
  t_dn = - m * sn * cn
  t_ph = dn * u_dot
  return (sn, cn, dn, ph), (t_sn, t_cn, t_dn, t_ph)
