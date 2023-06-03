# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import operator
import numpy as np

import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from jax.numpy.linalg import eigvals
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.numpy.polynomials import polyutils as pu

__all__ = [
  'chebline', 'chebadd', 'chebsub', 'chebmulx', 'chebmul', 'chebdiv',
  'chebpow', 'chebval', 'chebder', 'chebint', 'cheb2poly', 'poly2cheb',
  'chebfromroots', 'chebvander', 'chebfit', 'chebroots', 'chebpts1',
  'chebpts2', 'chebval2d', 'chebval3d', 'chebgrid2d', 'chebgrid3d',
  'chebvander2d', 'chebvander3d', 'chebcompanion', 'chebgauss',
  'chebweight', 'chebinterpolate'
]

@jit
def _cseries_to_zseries(c):
  n = c.size
  zs = jnp.zeros(2*n-1, dtype=c.dtype)
  zs = zs.at[n-1:].set(c/2)
  return zs + zs[::-1]

@jit
def _zseries_to_cseries(zs):
  n = (zs.size + 1)//2
  c = zs[n-1:]
  c = c.at[1:n].multiply(2)
  return c

@jit
def _zseries_mul(z1, z2):
  return jnp.convolve(z1, z2)

@jit
def _zseries_div(z1, z2):
  lc1 = len(z1)
  lc2 = len(z2)

  if lc2 == 1:
    z1 /= z2
    return z1, z1[:1]*0
  elif lc1 < lc2:
    return z1[:1]*0, z1
  else:
    dlen = lc1 - lc2
    scl = z2[0]
    z2 /= scl
    quo = jnp.zeros(dlen + 1, dtype=z1.dtype)

    i = 0
    j = dlen

    def cond_fun(val):
      return val[0] < val[1]

    def body_fun(val):
      i = val[0]
      j = val[1]
      quo = val[2]
      z1 = val[3]
      r = z1[i]
      quo = quo.at[i].set(r)
      quo = quo.at[j].set(r)
      tmp = r*z2
      z1 = _chebadd(jnp.roll(z1, -i), -tmp)
      z1 = _chebadd(jnp.roll(z1, i-j), -tmp)
      z1 = jnp.roll(z1, j)
      i += 1
      j -= 1
      return(i, j, quo, z1)

    i, j, quo, z1 = lax.while_loop(cond_fun, body_fun, (i, j, quo, z1))
    r = z1[i]
    quo = quo.at[i].set(r)
    tmp = r*z2
    z1 = _chebadd(jnp.roll(z1, -i), -tmp)
    quo /= scl
    rem = jnp.roll(z1, -1)[:lc2-2]
    return quo, rem

@jit
def _poly2cheb(pol):
  [pol] = pu._as_series([pol])
  pol, = promote_dtypes_inexact(pol)
  deg = len(pol) - 1
  res = jnp.zeros(deg+1, dtype=pol.dtype)
  pol = jnp.flip(pol)

  def body_fun(res, p):
    prd = jnp.zeros(len(res), dtype=res.dtype)
    prd = prd.at[1].set(res[0])
    if len(res) > 1:
      tmp = res[1:-1]/2
      prd = prd.at[2:].set(tmp)
      prd = prd.at[0:-2].add(tmp)
    prd = prd.at[0].add(p)
    return prd, p

  res, _ = lax.scan(body_fun, res, pol)
  return res

def poly2cheb(pol, trim=True):
  if trim:
    return _poly2cheb(pu.trimseq(pol))
  else:
    return _poly2cheb(pol)

@jit
def _cheb2poly(c):
  [c] = pu._as_series([c])
  c, = promote_dtypes_inexact(c)
  n = len(c)
  if n < 3:
    return c
  else:
    c0 = jnp.zeros(len(c), dtype=c.dtype)
    c0 = c0.at[0].set(c[-2])
    c1 = jnp.zeros(len(c), dtype=c.dtype)
    c1 = c1.at[0].set(c[-1])

    def body_fun(i, val):
      v0 = _chebsub(c[-i-2], val[1])
      v1 = jnp.zeros(len(val[1]), dtype=val[1].dtype)
      v1 = v1.at[1:].set(val[1][:-1])
      v1 = _chebadd(val[0], 2*v1)
      return(v0, v1)

    c0, c1 = lax.fori_loop(-(n-1),-1, body_fun, (c0, c1))

    tmp = jnp.zeros(len(c1), dtype=c1.dtype)
    c1 = tmp.at[1:].set(c1[:-1])

    return _chebadd(c0, c1)

def cheb2poly(c, trim=True):
  if trim:
    return _cheb2poly(pu.trimseq(c))
  else:
    return _cheb2poly(c)

@jit
def _chebline(off, scl):
  return jnp.array([off, scl])

def chebline(off, scl, trim=True):
  if trim:
    return pu.trimseq(_chebline(off, scl))
  else:
    return _chebline(off, scl)

@jit
def chebfromroots(roots):
  if len(roots) == 0:
    return jnp.ones(1, dtype=roots.dtype)
  else:
    [roots] = pu._as_series([roots])
    roots = roots.sort()
    retlen = len(roots)+1

    def p_scan_fun(carry, x):
      return carry, _chebadd(jnp.zeros(retlen, dtype=x.dtype), jnp.array([-x, 1], dtype=x.dtype))

    _, p = lax.scan(p_scan_fun, 0, roots)

    p = jnp.asarray(p)
    n = len(p)

    def cond_fun(val):
      return val[0] > 1

    def body_fun(val):
      m, r = divmod(val[0], 2)
      arr = val[1]
      tmp = jnp.array([jnp.zeros(retlen, dtype=p.dtype)]*len(p))

      def inner_body_fun(i, val):
        return val.at[i].set(_chebmul(arr[i], arr[i+m])[:retlen])

      tmp = lax.fori_loop(0, m, inner_body_fun, tmp)
      tmp = lax.cond(r, lambda x: x.at[0].set(_chebmul(x[0], arr[2*m])[:retlen]), lambda x: x, tmp)

      return(m, tmp)

    _, ret = lax.while_loop(cond_fun, body_fun, (n, p))
    return ret[0]

@jit
def _chebadd(c1, c2):
  [c1, c2] = pu._as_series([c1, c2])
  c1, c2 = promote_dtypes_inexact(c1, c2)
  if len(c1) > len(c2):
    c1 = c1.at[:c2.size].add(c2)
    ret = c1
  else:
    c2 = c2.at[:c1.size].add(c1)
    ret = c2
  return ret

def chebadd(c1, c2, trim=True):
  if trim:
    return pu.trimseq(_chebadd(c1, c2))
  else:
    return _chebadd(c1, c2)

@jit
def _chebsub(c1, c2):
  [c1, c2] = pu._as_series([c1, c2])
  c1, c2 = promote_dtypes_inexact(c1, c2)
  if len(c1) > len(c2):
    c1 = c1.at[:c2.size].add(-c2)
    ret = c1
  else:
    c2 = -c2
    c2 = c2.at[:c1.size].add(c1)
    ret = c2
  return ret

def chebsub(c1, c2, trim=True):
  if trim:
    return pu.trimseq(_chebsub(c1, c2))
  else:
    return _chebsub(c1, c2)

@jit
def _chebmulx(c):
  [c] = pu._as_series([c])
  c, = promote_dtypes_inexact(c)
  prd = jnp.zeros(len(c) + 1, dtype=c.dtype)
  prd = prd.at[1].set(c[0])
  if len(c) > 1:
    tmp = c[1:]/2
    prd = prd.at[2:].set(tmp)
    prd = prd.at[0:-2].add(tmp)
  return prd

def chebmulx(c, trim=True):
  if trim:
    return pu.trimseq(_chebmulx(pu.trimseq(c)))
  else:
    return _chebmulx(c)

@jit
def _chebmul(c1, c2):
  [c1, c2] = pu._as_series([c1, c2])
  c1, c2 = promote_dtypes_inexact(c1, c2)
  z1 = _cseries_to_zseries(c1)
  z2 = _cseries_to_zseries(c2)
  prd = _zseries_mul(z1, z2)
  ret = _zseries_to_cseries(prd)
  return ret

def chebmul(c1, c2, trim=True):
  if trim:
    return pu.trimseq(_chebmul(pu.trimseq(c1), pu.trimseq(c2)))
  else:
    return _chebmul(c1, c2)

@jit
def _chebdiv(c1, c2):
  [c1, c2] = pu._as_series([c1, c2])
  c1, c2 = promote_dtypes_inexact(c1, c2)
  lc1 = len(c1)
  lc2 = len(c2)
  if lc1 < lc2:
    return c1[:1]*0, c1
  elif lc2 == 1:
    return c1/c2[-1], c1[:1]*0
  else:
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    quo, rem = _zseries_div(z1, z2)
    quo = _zseries_to_cseries(quo)
    rem = _zseries_to_cseries(rem)
    return quo, rem

def chebdiv(c1, c2, trim=True):
  if trim:
    quo, rem = _chebdiv(pu.trimseq(c1), pu.trimseq(c2))
    return pu.trimseq(quo), pu.trimseq(rem)
  else:
    return _chebdiv(c1, c2)

@partial(jit, static_argnames=('pow', 'maxpower'))
def _chebpow(c, pow, maxpower=16):
  [c] = pu._as_series([c])
  c, = promote_dtypes_inexact(c)
  power = operator.index(pow)
  if power != pow or power < 0:
    raise ValueError("Power must be a non-negative integer.")
  elif maxpower is not None and power > maxpower:
    raise ValueError("Power is too large.")
  elif power == 0:
    return jnp.array([1], dtype=c.dtype)
  elif power == 1:
    return c
  else:
    zs = jnp.zeros(len(c)+(len(c)-1)*(pow-1), dtype=c.dtype)
    zs = _chebadd(zs, c)
    zs = _cseries_to_zseries(zs)
    prd = zs

    def body_fun(i, val):
      return jnp.convolve(val, zs, mode='same')

    prd = lax.fori_loop(2, power+1, body_fun, prd)
    return _zseries_to_cseries(prd)

def chebpow(c, pow, maxpower=16, trim=True):
  if trim:
    return _chebpow(pu.trimseq(c), pow, maxpower=maxpower)
  else:
    return _chebpow(c, pow, maxpower=maxpower)

@partial(jit, static_argnames=('m',))
def chebder(c, m=1):
  [c] = pu._as_series([c])
  c, = promote_dtypes_inexact(c)
  cnt = operator.index(m)
  if cnt < 0:
    raise ValueError("The order of derivation must be non-negative")
  if cnt == 0:
    return c

  n = len(c)
  if cnt >= n:
    c = c.at[:1].multiply(0)
    c = c[:1]
  else:

    def outer_body_fun(i, outer_val):
      inner_start = len(outer_val)-i-1
      tmp = jnp.zeros(len(outer_val), dtype=outer_val.dtype)

      def inner_body_fun(j, inner_val):
        tmp1 = inner_val[0]
        tmp2 = inner_val[1]
        tmp1 = tmp1.at[-j-1].set((2*-j)*tmp2[-j])
        tmp2 = tmp2.at[-j-2].add((-j*tmp2[-j])/(-j-2))
        return (tmp1, tmp2)

      tmp, outer_val = lax.fori_loop(-inner_start, -2, inner_body_fun, (tmp, outer_val))

      tmp = lax.cond(inner_start > 1, lambda x: x.at[1].set(4*outer_val[2]), lambda x: x, tmp)
      tmp = tmp.at[0].set(outer_val[1])
      return tmp

    c = lax.fori_loop(0, cnt, outer_body_fun, c)
    c = c[:n-cnt]
  return c

@partial(jit, static_argnames=('m',))
def _chebint(c, m=1, k=[], lbnd=0):
  [c] = pu._as_series([c])
  c, = promote_dtypes_inexact(c)
  if not np.iterable(k):
    k = [k]
  cnt = operator.index(m)
  if cnt < 0:
    raise ValueError("The order of integration must be non-negative")
  if len(k) > cnt:
    raise ValueError("Too many integration constants")
  if cnt == 0:
    return c

  k = jnp.asarray(list(k) + [0]*(cnt - len(k)), dtype=c.dtype)

  out = jnp.zeros(len(c)+cnt, dtype=c.dtype)
  out = _chebadd(out, c)

  def scan_fun(carry, k_val):
    n = len(carry)
    tmp = jnp.zeros(n, dtype=carry.dtype)
    tmp = tmp.at[1].set(carry[0])
    tmp = tmp.at[2].set(carry[1]/4)

    def inner_body_fun(j, inner_val):
      tmp = inner_val.at[j+1].set(carry[j]/(2*(j+1)))
      tmp = tmp.at[j-1].add(-carry[j]/(2*(j-1)))
      return tmp

    if n > 2:
      tmp = lax.fori_loop(2, n-1, inner_body_fun, tmp)
    tmp = tmp.at[0].add(k_val - chebval(lbnd, tmp))
    return tmp, k_val

  c, _ = lax.scan(scan_fun, out, k)
  return c

def chebint(c, m=1, k=[], lbnd=0, trim=True):
  if trim:
    if len(c) == 1 and c[0] == 0:
      return pu.trimseq(_chebint(c, m=m, k=k, lbnd=lbnd))
  return _chebint(c, m=m, k=k, lbnd=lbnd)

@partial(jit, static_argnames=('tensor',))
def chebval(x, c, tensor=True):
  c = jnp.asarray(c)
  x = jnp.asarray(x)
  c, = promote_dtypes_inexact(c)
  x, = promote_dtypes_inexact(x)
  if x.ndim != 0 and tensor:
    c = c.reshape(c.shape + (1,)*x.ndim)

  if len(c) == 1:
    c0 = c[0]
    c1 = 0
  elif len(c) == 2:
    c0 = c[0]
    c1 = c[1]
  else:
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]

    tmp = c0
    c0 = c[-3] - c1
    c1 = tmp + c1*x2
    if len(c) == 3:
      return c0 + c1*x

    tmp = c0
    c0 = c[-4] - c1
    c1 = tmp + c1*x2

    def body_fun(i, val):
      return (c[-i] - val[1], val[0] + val[1]*x2)

    c0, c1 = lax.fori_loop(5, len(c)+1, body_fun, (c0, c1))

  return c0 + c1*x

@jit
def chebval2d(x, y, c):
  x = jnp.asarray(x)
  y = jnp.asarray(y)
  if x.shape != y.shape:
    raise ValueError('x, y are incompatible')

  return chebval(y, chebval(x, c), tensor=False)

@jit
def chebgrid2d(x, y, c):
  return chebval(y, chebval(x, c))

@jit
def chebval3d(x, y, z, c):
  x = jnp.asarray(x)
  y = jnp.asarray(y)
  z = jnp.asarray(z)
  if x.shape != y.shape or x.shape != z.shape:
    raise ValueError('x, y, z are incompatible')

  return chebval(z, chebval(y, chebval(x, c), tensor=False), tensor=False)

@jit
def chebgrid3d(x, y, z, c):
  return chebval(z, chebval(y, chebval(x, c)))

@partial(jit, static_argnames=('deg',))
def chebvander(x, deg):
  if deg < 0 or deg != operator.index(deg):
    raise ValueError("deg must be a non-negative integer")
  deg = operator.index(deg)
  [x] = pu._as_series([x])
  x, = promote_dtypes_inexact(x)
  dims = (deg+1,) + x.shape
  dtyp = x.dtype
  v = jnp.zeros(dims, dtype=dtyp)
  v = v.at[0].set(x*0 + 1)
  if deg > 0:
    x2 = 2*x
    v = v.at[1].set(x)

    def body_fun(i, val):
      return val.at[i].set(val[i-1]*x2 - val[i-2])

    v = lax.fori_loop(2, deg+1, body_fun, v)

  return jnp.moveaxis(v, 0, -1)

@partial(jit, static_argnames=('deg_x', 'deg_y'))
def chebvander2d(x, y, deg_x, deg_y):
  [x, y] = pu._as_series([x, y])
  x, y = promote_dtypes_inexact(x, y)

  a = chebvander(x, deg_x)[(...,) + (slice(None), jnp.newaxis)]
  b = chebvander(y, deg_y)[(...,) + (jnp.newaxis, slice(None))]
  v = a * b

  return v.reshape(v.shape[:-2] + (-1,))

@partial(jit, static_argnames=('deg_x', 'deg_y', 'deg_z'))
def chebvander3d(x, y, z, deg_x, deg_y, deg_z):
  [x, y, z] = pu._as_series([x, y, z])
  x, y, z = promote_dtypes_inexact(x, y, z)

  a = chebvander(x, deg_x)[(...,) + (slice(None), jnp.newaxis, jnp.newaxis)]
  b = chebvander(y, deg_y)[(...,) + (jnp.newaxis, slice(None), jnp.newaxis)]
  c = chebvander(z, deg_z)[(...,) + (jnp.newaxis, jnp.newaxis, slice(None))]
  v = a * b * c

  return v.reshape(v.shape[:-3] + (-1,))

def _chebfit(x, y, deg, rcond=None, full=False, w=None, numpy_resid=True):
  [x] = pu._as_series([x])
  y = jnp.asarray(y)
  x, y = promote_dtypes_inexact(x, y)
  deg = operator.index(deg)

  if deg < 0:
    raise ValueError("expected deg >= 0")
  if x.ndim != 1:
    raise TypeError("expected 1D vector for x")
  if x.size == 0:
    raise TypeError("expected non-empty vector for x")
  if y.ndim < 1 or y.ndim > 2:
    raise TypeError("expected 1D or 2D array for y")
  if x.shape[0] != y.shape[0]:
    raise TypeError("expected x and y to have the same length")

  van = chebvander(x, deg)
  lhs = jnp.array(van.T, ndmin=2)
  rhs = jnp.array(y.T, ndmin=2)
  if w is not None:
    [w] = pu._as_series([w])
    w, = promote_dtypes_inexact(w)
    if len(x) != len(w):
      raise TypeError("expected x and w to have same length")

    def scan_fun(carry, x):
      return carry, x*carry

    _, lhs = lax.scan(scan_fun, w, lhs)
    _, rhs = lax.scan(scan_fun, w, rhs)
    lhs = jnp.array(lhs, dtype=x.dtype)
    rhs = jnp.array(rhs, dtype=y.dtype)

  if rcond is None:
    rcond = jnp.finfo(x.dtype).eps
    rcond = (rcond*len(x)).astype(rcond.dtype)

  if issubclass(lhs.dtype.type, jnp.complexfloating):
    scl = jnp.sqrt((jnp.square(lhs.real) + jnp.square(lhs.imag)).sum(1)).astype(lhs.dtype)
  else:
    scl = jnp.sqrt(jnp.square(lhs).sum(1)).astype(lhs.dtype)

  def scl_body_fun(i, val):
    return lax.cond(val[i] == 0, lambda x: x.at[i].set(1), lambda x: x, val)

  scl = lax.fori_loop(0, len(scl), scl_body_fun, scl)

  def lhs_body_fun(i, val):
    return val.at[i].divide(scl)

  lhs = lax.fori_loop(0, lhs.shape[1], lhs_body_fun, lhs.T)

  c, resids, rank, s = jnp.linalg.lstsq(lhs, rhs.T, rcond, numpy_resid=numpy_resid)
  c = c.T[0]/scl

  if full:
    return c, [resids, rank, s, rcond]
  else:
    return c

def chebfit(x, y, deg, rcond=None, full=False, w=None, numpy_resid=True):
  if numpy_resid:
    return _chebfit(x, y, deg, rcond=rcond, full=full, w=w, numpy_resid=True)
  else:
    return jit(partial(_chebfit, numpy_resid=False), static_argnames=('deg', 'full'))(x, y, deg, rcond=rcond, full=full, w=w)

@jit
def _chebcompanion(c):
  [c] = pu._as_series([c])
  c, = promote_dtypes_inexact(c)
  if len(c) < 2:
    raise ValueError("Series must have a maximum degree of at least 1")
  if len(c) == 2:
    return jnp.array([[-c[0]/c[1]]], dtype=c.dtype)
  n = len(c) - 1
  mat = jnp.zeros((n, n), dtype=c.dtype)
  scl = jnp.array([1.] + [jnp.sqrt(0.5)]*(n-1), dtype=c.dtype)
  top = mat.reshape(-1)[1::n+1]
  bot = mat.reshape(-1)[n::n+1]
  top = top.at[0].set(jnp.sqrt(0.5))
  top = top.at[1:].set(0.5)
  bot = bot.at[...].set(top)
  mat = mat.reshape(-1).at[1::n+1].set(top)
  mat = mat.at[n::n+1].set(bot)
  mat = mat.reshape((n,n))
  mat = mat.at[:, -1].add(-(c[:-1]/c[-1])*(scl/scl[-1])*0.5)
  return mat

def chebcompanion(c, trim=True):
  if trim:
    return _chebcompanion(pu.trimseq(c))
  else:
    return _chebcompanion(c)

@jit
def _chebroots(c):
  [c] = pu._as_series([c])
  c, = promote_dtypes_inexact(c)
  if len(c) < 2:
    return jnp.array([], dtype=c.dtype)
  if len(c) == 2:
    return jnp.array([-c[0]/c[1]], dtype=c.dtype)
  m = _chebcompanion(c)[::-1,::-1]
  r = eigvals(m)
  return r.sort()

def chebroots(c, trim=True):
  if trim:
    return _chebroots(pu.trimseq(c))
  else:
    return _chebroots(c)

def chebinterpolate(func, deg, args=()):
  deg = operator.index(deg)
  if deg < 0:
    raise ValueError("deg must be a non-negative integer")
  order = deg + 1
  xcheb = chebpts1(order)
  yfunc = func(xcheb, *args)
  m = chebvander(xcheb, deg)
  c = jnp.dot(m.T, yfunc)
  c = c.at[0].divide(order)
  c = c.at[1:].divide(0.5*order)
  return c

def chebgauss(deg):
  deg = operator.index(deg)
  if deg <= 0:
    raise ValueError("deg must be a positive integer")

  x = jnp.cos(jnp.pi * jnp.arange(1., 2*deg, 2.) / (2.0*deg))
  w = jnp.ones(deg)*(jnp.pi/deg)

  return x, w

@jit
def chebweight(x):
  [x] = pu._as_series([x])
  x, = promote_dtypes_inexact(x)
  w = 1./(jnp.sqrt(1. + x) * jnp.sqrt(1. - x))
  return w

def chebpts1(npts):
  _npts = operator.index(npts)
  if _npts < 1:
    raise ValueError("npts must be >= 1")

  x = 0.5 * jnp.pi / _npts * jnp.arange(-_npts+1, _npts+1, 2, dtype="float32")
  return jnp.sin(x)

def chebpts2(npts):
  _npts = operator.index(npts)
  if _npts < 2:
    raise ValueError("npts must be >= 2")

  x = jnp.linspace(-jnp.pi, 0, _npts)
  return jnp.cos(x)
