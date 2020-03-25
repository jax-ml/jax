from functools import partial

import scipy.special as osp_special

from ... import api
from ...numpy import lax_numpy as jnp
from ...lax import broadcast_shapes, while_loop
from ...numpy.lax_numpy import _wraps


@_wraps(osp_special.cosm1)
@api.custom_jvp
def cosm1(x):
  coeffs = [
    4.7377507964246204691685E-14,
    -1.1470284843425359765671E-11,
    2.0876754287081521758361E-9,
    -2.7557319214999787979814E-7,
    2.4801587301570552304991E-5,
    -1.3888888888888872993737E-3,
    4.1666666666666666609054E-2,
  ]
  normal_mask = jnp.logical_or(x < -jnp.pi / 4, x > jnp.pi / 4)
  normal_value = jnp.cos(x) - 1

  xx = x * x
  xx = -0.5 * xx + xx * xx * jnp.polyval(coeffs, xx)
  return normal_mask * normal_value + (1 - normal_mask) * xx


@cosm1.defjvp
def cosm1_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  y = cosm1(x)
  y_dot = - jnp.sin(x)
  return y, y_dot * x_dot


def _elem_if_else(conditions, values):
  assert len(values) == len(conditions) + 1
  is_nan = jnp.isnan(values[-1])
  result = jnp.nan_to_num(values[-1])
  for cond, value in zip(reversed(conditions), reversed(values[:-1])):
    result = cond * jnp.nan_to_num(value) + (1 - cond) * result
    is_nan_new = jnp.logical_and(cond, jnp.isnan(value))
    is_nan_old = jnp.logical_and(jnp.logical_not(cond), is_nan)
    is_nan = jnp.logical_or(is_nan_new, is_nan_old)
  return _make_nans(is_nan, result)


def _make_nans(is_nan, value):
  nans = jnp.sqrt((1 - is_nan * 2).astype(dtype=value.dtype))
  return value * nans


def _elem_while_loop(cond_fun, body_fun, init_val, init_not_done=None):
  if init_not_done is None:
    init_not_done = cond_fun(init_val)
  _init_val = list(init_val) + [init_not_done]

  def _cond_fun(values):
    not_done = values[-1]
    return jnp.any(not_done)

  def _body_fun(values):
    values, not_done = values[:-1], values[-1]
    values_next = body_fun(values)
    values = [(1 - not_done.astype(v.dtype)) * v + not_done * v_n for v, v_n in zip(values, values_next)]
    not_done_next = jnp.logical_and(not_done, cond_fun(values))
    return values + [not_done_next]

  return while_loop(_cond_fun, _body_fun, _init_val)[:-1]


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

  # Note that ellipk(m) = ellpk(1 - m)
  x = 1 - m
  x_is_zero = x == 0

  large_mask = (x > 1).astype(x.dtype)
  div_factor = large_mask * jnp.sqrt(x) + (1 - large_mask) * 1
  x = large_mask * jnp.nan_to_num(jnp.reciprocal(x)) + (1 - large_mask) * jnp.nan_to_num(x)

  small_mask = (x <= machep).astype(x.dtype)
  small_value = C1 - 0.5 * jnp.nan_to_num(jnp.log(x))

  value = jnp.polyval(P, x) - jnp.nan_to_num(jnp.log(x)) * jnp.polyval(Q, x)
  value = small_mask * small_value + (1 - small_mask) * value
  value = value / div_factor

  return _make_nans(x < 0, value / (1 - x_is_zero.astype(m.dtype)))


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
  # By using the nan_to_num we take care of the case where x == 0 correctly

  large_mask = (x > 1).astype(x.dtype)
  mul_factor = large_mask * jnp.sqrt(x) + (1 - large_mask) * 1
  x = large_mask * jnp.nan_to_num(jnp.reciprocal(x)) + (1 - large_mask) * jnp.nan_to_num(x)

  value = jnp.polyval(P, x) - jnp.nan_to_num(jnp.log(x)) * (x * jnp.polyval(Q, x))
  value = value * mul_factor

  return _make_nans(x < 0, value)


@ellipe.defjvp
def ellipe_jvp(primals, tangents):
  m, = primals
  m_dot, = tangents
  k = ellipk(m)
  e = ellipe(m)
  e_dot = (e - k) / m
  return e, e_dot * m_dot


@_wraps(osp_special.ellipj)
def ellipj(u, m):
  return _ellipj(u, m, 8, True)


@partial(api.custom_jvp, nondiff_argnums=[1, 2, 3])
def _ellipj(u, m, agm_iters, agm_non_term_nans):
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
  mask_1 = (m < 1.0e-9).astype(u.dtype)

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
  mask_2 = (m >= 0.9999999999).astype(u.dtype)

  # Main computation
  a = [jnp.ones_like(u)]
  c = [jnp.sqrt(m)]
  done_mask = [jnp.fabs(c[0] / a[0]) < machep]
  b = jnp.sqrt(1.0 - m)
  twon = 1.0

  for i in range(agm_iters):
    ai = a[-1]
    c.append((ai - b) / 2)
    t = jnp.sqrt(ai * b)
    a.append((ai + b) / 2)
    b = t
    twon = twon * done_mask[-1] + (1 - done_mask[-1]) * (twon * 2)
    done_mask.append(jnp.fabs(c[-1] / a[-1]) < machep)
    done_mask[-1] = jnp.logical_or(done_mask[-1], done_mask[-2])

  twon = jnp.asarray(twon, dtype=u.dtype)
  assert len(a) == 9 and len(c) == 9

  # backward recurrence
  phi = 0.0
  for i in range(8, 0, -1):
    # Initial value
    phi_start = twon * a[i] * u
    is_start = done_mask[i] if i == 0 else done_mask[i] != done_mask[i - 1]
    is_start = is_start.astype(u.dtype)
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
  cond = (jnp.fabs(dnfac) < 0.1).astype(u.dtype)
  dn_true = jnp.sqrt(1 - m * sn * sn)
  dn_false = t / dnfac
  dn = cond * dn_true + (1 - cond) * dn_false
  ph = phi
  out_3 = (sn, cn, dn, ph)

  # Produce NaNs instead of error
  not_nan = jnp.logical_or(mask_1, mask_2)
  if agm_non_term_nans:
    not_nan = jnp.logical_or(not_nan, done_mask[-1])
  is_nan = jnp.logical_or(m < 0, m > 1)
  is_nan = jnp.logical_or(is_nan, jnp.logical_not(not_nan))

  def mask_output(x1, x2, x3):
    return _make_nans(is_nan, _elem_if_else([mask_1, mask_2], [x1, x2, x3]))

  return [mask_output(x1, x2, x3) for x1, x2, x3 in zip(out_1, out_2, out_3)]


@_ellipj.defjvp
def _ellipj_jvp(m, agm_iters, agm_non_term_nans, primals, tangents):
  u, = primals
  u_dot, = tangents
  sn, cn, dn, ph = _ellipj(u, m, agm_iters, agm_non_term_nans)
  t_sn = cn * dn * u_dot
  t_cn = - sn * dn * u_dot
  t_dn = - m * sn * cn
  t_ph = dn * u_dot
  return (sn, cn, dn, ph), (t_sn, t_cn, t_dn, t_ph)


@_wraps(osp_special.ellipkinc)
@partial(api.custom_jvp, nondiff_argnums=[1])
def ellipkinc(phi, m):
  return _ellipkinc(phi, m, can_recurse=True)


def _ellipkinc(phi_input, m, can_recurse=True):
  phi = phi_input
  shape = broadcast_shapes(phi.shape, m.shape)
  dtype = jnp.promote_types(phi.dtype, m.dtype)
  machep = jnp.finfo(phi.dtype).eps

  m_inf = jnp.logical_and(jnp.isinf(m), jnp.isfinite(phi))
  m_inf_value = jnp.zeros(shape, dtype=dtype)

  phi_inf = jnp.logical_and(jnp.isinf(phi), jnp.isfinite(m))

  m_is_zero = m == 0
  m_is_zero_value = phi

  m_is_one = m == 1
  phi_more_than_pi_2 = jnp.abs(phi) >= jnp.pi / 2
  # This makes all values where phi >= pi / 2 to be inf
  m_is_one_value = jnp.arcsinh(jnp.tan(phi))

  # Standard computation
  a = 1 - m
  npio2 = jnp.floor(2 * phi / jnp.pi)
  npio2 = npio2 + (jnp.fmod(jnp.abs(npio2), 2.0) == 1)
  npio2_not_zero = npio2 != 0
  # Note that ellipk(x) = ellpk(1 - x)
  k = npio2_not_zero * jnp.nan_to_num(ellipk(1 - a))
  phi = phi - npio2_not_zero * (npio2 * np.pi / 2)

  sign = jnp.sign(phi)
  phi = jnp.abs(phi)

  check_1 = a > 1
  temp_1 = _ellik_neg_m(phi, m)

  b = jnp.sqrt(a)
  t = jnp.tan(phi)

  if can_recurse:
    e_ = 1.0 / (b * t)
    check_2 = jnp.logical_and(jnp.abs(t) > 10, jnp.abs(e_) < 10)
    npio2_is_zero_ = npio2 == 0
    e_ = jnp.arctan(e_)
    k_ = npio2_is_zero_ * ellipk(a) + (1 - npio2_is_zero_) * k
    temp_2 = k_ - _ellipkinc(e_, m, can_recurse=False)
  else:
    check_2 = jnp.full_like(check_1, False)
    temp_2 = jnp.zeros_like(temp_1)

  loop_init = (
    jnp.ones(shape, dtype=dtype) * (b != 0).astype(dtype),  # a
    b,  # b
    jnp.sqrt(m),  # c
    jnp.ones(shape, dtype=jnp.int32),  # d
    t,  # t
    phi,  # phi
    jnp.zeros(shape, dtype=jnp.int32),  # mod
  )

  def loop_cond(values):
    a_, b_, c_, d_, t_, phi_, mod_ = values
    # Eliminate the cases where a_i = 0 as we also have b_i = 0
    return jnp.nan_to_num(jnp.abs(c_ / a_)) * (a_ != 0) > machep

  def loop_body(values):
    a_, b_, c_, d_, t_, phi_, mod_ = values
    temp_ = b_ / a_
    phi_ = phi_ + jnp.arctan(t_ * temp_) + mod_ * jnp.pi
    denom = 1.0 - temp_ * t_ * t_
    check_ = jnp.abs(denom) > 10 * machep
    # if
    t_1 = t_ * (1 + temp_) / denom
    mod_1 = phi_ + jnp.pi / 2
    # else
    t_2 = jnp.tan(phi_)
    mod_2 = phi_ - jnp.arctan(t)

    t_ = check_ * t_1 + (1 - check_) * t_2
    mod_ = check_ * mod_1 + (1 - check_) * mod_2
    mod_ = jnp.floor_divide(mod_, jnp.pi).astype(jnp.int32)

    c_ = (a_ - b_) / 2
    temp_ = jnp.sqrt(a_ * b_)
    a_ = (a_ + b_) / 2
    b_ = temp_

    d_ = d_ + d_

    return a_, b_, c_, d_, t_, phi_, mod_

  outs = _elem_while_loop(loop_cond, loop_body, loop_init)
  a, b, c, d, t, phi, mod = outs
  temp_3 = (jnp.arctan(t) + mod * jnp.pi) / (d * a)

  # Done
  temp = _elem_if_else([check_1, check_2], [temp_1, temp_2, temp_3])
  temp = sign * temp
  temp = temp + npio2 * k

  value = _elem_if_else([m_inf, m_is_zero, m_is_one],
                        [m_inf_value, m_is_zero_value, m_is_one_value, temp])

  # Make infinite
  is_inf = jnp.logical_or(phi_inf, jnp.logical_and(m_is_one, phi_more_than_pi_2))
  value = value / (1 - is_inf.astype(dtype))

  is_nan = jnp.logical_or(jnp.isnan(m), jnp.isnan(phi_input))
  is_nan = jnp.logical_or(is_nan, m > 1)
  is_nan = jnp.logical_or(is_nan, jnp.logical_and(jnp.isinf(m), jnp.isinf(phi)))

  return _make_nans(is_nan, value)


def _ellik_neg_m(phi, m):
  shape = broadcast_shapes(phi.shape, m.shape)
  dtype = jnp.promote_types(phi.dtype, m.dtype)

  mpp = (m * phi) * phi

  check_1 = jnp.logical_and(-mpp < 1e-6, phi < -m)
  value_1 = phi + (-mpp * phi * phi / 30 + 3 * mpp * mpp / 40 + mpp / 6) * phi

  check_2 = - mpp > 4e7

  def check2_value_(phi_, m_):
    sm = jnp.sqrt(-m_)
    sp = jnp.sin(phi_)
    cp = jnp.cos(phi_)
    a_ = jnp.log(4 * sp * sm / (1 + cp))
    b_ = -(1 + cp / sp / sp - a_) / 4 / m_
    return (a_ + b_) / sm

  value_2 = check2_value_(phi, m)

  cond = jnp.logical_and(phi > 1e-153, m > -1e305)
  s = jnp.sin(phi)
  csc2 = 1.0 / (s * s)
  scale = cond * 1 + (1 - cond) * phi
  x = cond / (jnp.tan(phi) * jnp.tan(phi)) + (1 - cond) * 1
  y = cond * (csc2 - m) + (1 - cond) * (1 - m * phi * phi)
  z = cond * csc2 + (1 - cond) * 1

  check_3 = jnp.logical_and(x == y, x == z)
  value_3 = scale / jnp.sqrt(x)

  a0 = (x + y + z) / 3
  max3 = jnp.maximum(jnp.maximum(jnp.abs(a0 - x), jnp.abs(a0 - y)), jnp.abs(a0 - z))
  loop_init = (x, y, z, 400 * max3, jnp.zeros(shape, dtype=jnp.int32))

  def loop_cond(values):
    x_, y_, z_, q_, n_ = values
    a_ = (x_ + y_ + z_) / 3
    return jnp.logical_and(q_ > jnp.abs(a_), n_ <= 100)

  def loop_body(values):
    x_, y_, z_, q_, n_ = values
    sx = jnp.sqrt(x_)
    sy = jnp.sqrt(y_)
    sz = jnp.sqrt(z_)
    lam = sx * sy + sx * sz + sy * sz
    x_ = (x_ + lam) / 4
    y_ = (y_ + lam) / 4
    z_ = (z_ + lam) / 4
    q_ = q_ / 4
    n_ = n_ + 1
    return x_, y_, z_, q_, n_

  x, y, z, q, n = _elem_while_loop(loop_cond, loop_body, loop_init)
  a = (x + y + z) / 3
  x = (a0 - x) / a / jnp.power(2, 2 * n)
  y = (a0 - y) / a / jnp.power(2, 2 * n)
  z = - (x + y)
  e2 = x * y - z * z
  e3 = x * y * z
  value_4 = scale * (1 - e2 / 10 + e3 / 14 + e2 * e2 / 24 - 3 * e2 * e3 / 44) / jnp.sqrt(a)

  return _elem_if_else([check_1, check_2, check_3],
                       [value_1, value_2, value_3, value_4])


@ellipkinc.defjvp
def ellipkinc_jvp(m, primals, tangents):
  phi, = primals
  phi_dot, = tangents
  u = ellipkinc(phi, m)
  u_dot = 1 / jnp.sqrt(1 - m * jnp.sin(phi) ** 2)
  return u, u_dot * phi_dot


@_wraps(osp_special.ellipeinc)
@partial(api.custom_jvp, nondiff_argnums=[1])
def ellipeinc(phi, m):
  return _ellipeinc(phi, m, True)


def _ellipeinc(phi_input, m, can_recurse):
  phi = phi_input
  shape = broadcast_shapes(phi.shape, m.shape)
  dtype = jnp.promote_types(phi.dtype, m.dtype)
  machep = jnp.finfo(phi.dtype).eps

  m_is_zero = m == 0
  m_is_zero_value = phi

  phi_is_inf = jnp.isinf(phi)
  m_is_inf = jnp.isinf(m)
  is_inf = jnp.logical_or(phi_is_inf, m_is_inf)
  inf_sign = phi_is_inf * jnp.sign(phi) - (1 - phi_is_inf) * jnp.sign(m)

  # Standard computation
  lphi = phi
  npio2 = jnp.floor(2 * lphi / jnp.pi)
  cond = jnp.fmod(jnp.abs(npio2), 2.0) == 1
  npio2 = npio2 + cond
  lphi = lphi - npio2 * jnp.pi / 2
  sign = jnp.sign(lphi)
  lphi = jnp.abs(lphi)

  a = 1 - m
  big_e = ellipe(m)
  check_1 = a == 0
  temp_1 = jnp.sin(lphi)
  check_2 = a > 1
  temp_2 = _ellie_neg_m(lphi, m)

  check_3 = lphi < 0.135
  m11 = (((((-7.0 / 2816.0) * m + (5.0 / 1056.0)) * m - (7.0 / 2640.0)) * m +
          (17.0 / 41580.0)) * m - (1.0 / 155925.0)) * m
  m9 = ((((-5.0 / 1152.0) * m + (1.0 / 144.0)) * m - (1.0 / 360.0)) * m + (1.0 / 5670.0)) * m
  m7 = ((-m / 112.0 + (1.0 / 84.0)) * m - (1.0 / 315.0)) * m
  m5 = (-m / 40.0 + (1.0 / 30)) * m
  m3 = -m / 6.0
  p2 = lphi * lphi
  temp_3 = ((((m11 * p2 + m9) * p2 + m7) * p2 + m5) * p2 + m3) * p2 * lphi + lphi

  t = jnp.tan(lphi)
  b = jnp.sqrt(a)

  if can_recurse:
    e = 1 / (b * t)
    check_4 = jnp.logical_and(jnp.abs(t) > 10, jnp.abs(e) < 10)
    e = jnp.arctan(e)
    temp_4 = big_e + m * jnp.sin(lphi) * jnp.sin(e) - _ellipeinc(e, m, can_recurse=False)
  else:
    check_4 = jnp.full_like(check_1, False)
    temp_4 = jnp.zeros_like(temp_1)

  loop_init = (
    jnp.ones(shape, dtype=dtype) * (b != 0).astype(dtype),  # a
    b,  # b
    jnp.sqrt(m),  # c
    jnp.ones(shape, dtype=jnp.int32),  # d
    jnp.zeros(shape, dtype=dtype),  # e
    t,  # t
    lphi,  # lphi
    jnp.zeros(shape, dtype=jnp.int32),  # mod
  )

  def loop_cond(values):
    a_, b_, c_, d_, e_, t_, lphi_, mod_ = values
    # Eliminate the cases where a_i = 0 as we also have b_i = 0
    return jnp.nan_to_num(jnp.abs(c_ / a_)) * (a_ != 0) > machep

  def loop_body(values):
    a_, b_, c_, d_, e_, t_, lphi_, mod_ = values
    temp_ = b_ / a_
    lphi_ = lphi_ + jnp.arctan(t_ * temp_) + mod_ * jnp.pi
    denom = 1 - temp_ * t_ * t_

    check_ = jnp.abs(denom) > 10 * machep
    t_1 = t_ * (1 + temp_) / denom
    mod_1 = lphi_ + jnp.pi / 2
    t_2 = jnp.tan(lphi_)
    mod_2 = lphi_ - jnp.arctan(t_)

    t_ = check_ * t_1 + (1 - check_) * t_2
    mod_ = check_ * mod_1 + (1 - check_) * mod_2
    mod_ = jnp.floor_divide(mod_, jnp.pi).astype(jnp.int32)

    c_ = (a_ - b_) / 2
    temp_ = jnp.sqrt(a_ * b_)
    a_ = (a_ + b_) / 2
    b_ = temp_

    d_ = d_ + d_
    e_ = e_ + c_ * jnp.sin(lphi_)

    return a_, b_, c_, d_, e_, t_, lphi_, mod_

  outs = _elem_while_loop(loop_cond, loop_body, loop_init)
  a, b, c, d, e, t, lphi, mod = outs
  # Note that ins sicpy ellipk(m) = cephes::ellpk(1 - m)
  # hence rather than ellpk(1.0 - m) we sue ellipk(m)
  temp_5 = big_e * (jnp.arctan(t) + mod * jnp.pi) / (d * a * ellipk(m)) + e

  # Done
  temp = _elem_if_else([check_1, check_2, check_3, check_4],
                       [temp_1, temp_2, temp_3, temp_4, temp_5])
  temp = sign * temp
  temp = temp + npio2 * big_e

  value = _elem_if_else([m_is_zero, is_inf], [m_is_zero_value, inf_sign, temp])
  # Make any infs
  value = jnp.nan_to_num(value) / (1 - is_inf)

  is_nan = jnp.logical_or(jnp.isnan(m), jnp.isnan(phi_input))
  is_nan = jnp.logical_or(is_nan, m > 1)

  return _make_nans(is_nan, value)


def _ellie_neg_m(phi, m):
  shape = broadcast_shapes(phi.shape, m.shape)
  dtype = jnp.promote_types(phi.dtype, m.dtype)

  mpp = (m * phi) * phi

  check_1 = jnp.logical_and(-mpp < 1e-6, phi < -m)
  value_1 = phi + (mpp * phi * phi / 30 - mpp * mpp / 40 - mpp / 6) * phi

  check_2 = - mpp > 1e6

  def check2_value_(phi_, m_):
    sm = jnp.sqrt(-m_)
    sp = jnp.sin(phi_)
    cp = jnp.cos(phi_)
    a_ = cosm1(phi_)
    b1 = jnp.log(4 * sp * sm / (1 + cp))
    b_ = - (0.5 + b1) / (2 * m)
    c_ = (0.75 + cp / (sp * sp) - b1) / (16 * m * m)
    return (a_ + b_ + c_) * sm

  value_2 = check2_value_(phi, m)

  cond = jnp.logical_and(phi > 1e-153, m > -1e200).astype(dtype)
  s = jnp.sin(phi)
  csc2 = 1 / (s * s)
  scalef = cond * 1 + (1 - cond) * phi
  scaled = cond * m / 3 + (1 - cond) * mpp * phi / 3
  x = cond / (jnp.tan(phi) ** 2) + (1 - cond) * 1
  y = cond * (csc2 - m) + (1 - cond) * (1 - mpp)
  z = cond * csc2 + (1 - cond) * 1

  check_3 = jnp.logical_and(x == y, x == z)
  value_3 = (scalef + scaled / x) / jnp.sqrt(x)

  a0f = (x + y + z) / 3
  a0d = (x + y + 3 * z) / 5
  max3 = jnp.maximum(jnp.maximum(jnp.abs(a0f - x), jnp.abs(a0f - y)), jnp.abs(a0f - z))
  # x, y, z, q, n, ad, series_d, series_n
  loop_init = (x, y, z, 400 * max3, jnp.zeros(shape, dtype=jnp.int32), a0d, jnp.zeros_like(x), jnp.ones_like(y))

  def loop_cond(values):
    x_, y_, z_, q_, n_, ad_, series_d_, series_n_ = values
    af = (x_ + y_ + z_) / 3
    return jnp.logical_and(jnp.logical_and(q_ > jnp.abs(af), q_ > jnp.abs(ad_)), n_ <= 100)

  def loop_body(values):
    x_, y_, z_, q_, n_, ad_, series_d_, series_n_ = values
    sx = jnp.sqrt(x_)
    sy = jnp.sqrt(y_)
    sz = jnp.sqrt(z_)
    lam = sx * sy + sx * sz + sy * sz
    series_d_ = series_d_ + series_n_ / (sz * (z_ + lam))
    series_n_ = series_n_ / 4
    x_ = (x_ + lam) / 4
    y_ = (y_ + lam) / 4
    z_ = (z_ + lam) / 4
    q_ = q_ / 4
    n_ = n_ + 1
    ad_ = (ad_ + lam) / 4
    return x_, y_, z_, q_, n_, ad_, series_d_, series_n_

  x1, y1, z1, q, n, ad, series_d, series_n = _elem_while_loop(loop_cond, loop_body, loop_init)
  af = (x1 + y1 + z1) / 3
  two_to_2n = jnp.power(2, 2 * n)

  xf = (a0f - x) / (af * two_to_2n)
  yf = (a0f - y) / (af * two_to_2n)
  zf = - (xf + yf)
  e2f = xf * yf - zf * zf
  e3f = xf * yf * zf

  value_4 = scalef * (1 - e2f / 10 + e3f / 14 + e2f * e2f / 24 - 3 * e2f * e3f / 44) / jnp.sqrt(af)
  xd = (a0d - x) / (ad * two_to_2n)
  yd = (a0d - y) / (ad * two_to_2n)
  zd = -(xd + yd) / 3
  e2d = xd * yd - 6 * zd * zd
  e3d = (3 * xd * yd - 8 * zd * zd) * zd
  e4d = 3 * (xd * yd - zd * zd) * zd * zd
  e5d = xd * yd * zd * zd * zd
  v1 = scaled * (
        1.0 - 3.0 * e2d / 14.0 + e3d / 6.0 + 9.0 * e2d * e2d / 88.0 - 3.0 * e4d / 22.0 - 9.0 * e2d * e3d / 52.0 + 3.0 * e5d / 26.0)
  value_4 = value_4 - v1 / (ad * jnp.sqrt(ad) * two_to_2n)
  value_4 = value_4 - 3 * scaled * series_d

  return _elem_if_else([check_1, check_2, check_3],
                       [value_1, value_2, value_3, value_4])


@ellipeinc.defjvp
def ellipeinc_jvp(m, primals, tangents):
  # This uses the fact that ellipkinc(phi, m) is inverse of sn(u, m)
  phi, = primals
  phi_dot, = tangents
  u = ellipkinc(phi, m)
  u_dot = jnp.sqrt(1 - m * jnp.sin(phi) ** 2)
  return u, u_dot * phi_dot
