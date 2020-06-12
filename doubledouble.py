import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


class dfloat:
  def __init__(self, value, base_dtype=jnp.float64):
    self._base_dtype = jnp.dtype(base_dtype)
    if isinstance(value, dfloat):
      self._value = tuple(jnp.asarray(v, dtype=base_dtype) for v in value._value)
    else:
      v = jnp.asarray(value, dtype=base_dtype)
      self._value = (v, jnp.asarray(value - v, dtype=base_dtype))

  def _from_tuple(self, tup):
    assert len(tup) == 2
    result = self.__class__(0, self._base_dtype)
    result._value = tuple(jnp.asarray(v, dtype=self._base_dtype) for v in tup)
    return result

  def __neg__(self):
    return self._from_tuple([-v for v in self._value])

  def __add__(self, other):
    other = self.__class__(other, self._base_dtype)
    return self._from_tuple(_add2(self._value, other._value))

  def __sub__(self, other):
    other = self.__class__(other, self._base_dtype)
    return self._from_tuple(_sub2(self._value, other._value))

  def __mul__(self, other):
    other = self.__class__(other, self._base_dtype)
    return self._from_tuple(_mul2(self._value, other._value))

  def __truediv__(self, other):
    other = self.__class__(other, self._base_dtype)
    return self._from_tuple(_div2(self._value, other._value))

  def __radd__(self, other):
    return self.__class__(other, self._base_dtype) + self

  def __rsub__(self, other):
    return self.__class__(other, self._base_dtype) - self

  def __rmul__(self, other):
    return self.__class__(other, self._base_dtype) * self

  def __rtruediv__(self, other):
    return self.__class__(other, self._base_dtype) / self
    
  def __repr__(self):
    return "dfloat({0})".format(self._value[0] + self._value[1])

  def astype(self, dtype):
    dtype = jnp.dtype(dtype)
    return jnp.array(self._value[0], dtype=dtype) + jnp.array(self._value[1], dtype=dtype)


def _add2(x, y):
  # From http://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
  (x, xx), (y, yy) = x, y
  r = x + y
  s = jnp.where(abs(x) > abs(y), x - r + y + yy + xx, y - r + x + xx + yy)
  z = r + s
  zz = r - z + s
  return (z, zz)

def _sub2(x, y):
  (x, xx), (y, yy) = x, y
  r = x - y
  s = jnp.where(abs(x) > abs(y), x - r - y - yy + xx, -y - r + x + xx - yy)
  z = r + s
  zz = r - z + s
  return (z, zz)

_nmant = jnp.finfo(jnp.float64).nmant
_mul_const = (2 << (_nmant - _nmant // 2)) + 1

def _mul12(x, y):
  p = x * _mul_const
  hx = x - p + p
  tx = x - hx
  p = y * _mul_const
  hy = y - p + p
  ty = y - hy
  p = hx * hy
  q = hx * ty + tx * hy
  z = p + q
  zz = p - z + q + tx * ty
  return z, zz

def _mul2(x, y):
  (x, xx), (y, yy) = x, y
  c, cc = _mul12(x, y)
  cc = x * yy + xx * y + cc
  z = c + cc
  zz = c - z + cc
  return (z, zz)

def _div2(x, y):
  (x, xx), (y, yy) = x, y
  c = x / y
  u, uu = _mul12(c, y)
  cc = (x - u - uu + xx - c * yy) / y
  z = c + cc
  zz = c - z + cc
  return z, zz

def _sqrt2(x):
  x, xx = x
  c = jnp.sqrt(x)
  u, uu = mul12(c, c)
  cc = x - u - uu + xx * 0.5 / c
  y = c + cc
  yy = c - y + cc
  return y, yy


if __name__ == '__main__':
  A = jnp.arange(10) * 1E20
  B = jnp.arange(10)
  print(A.dtype)
  print(A + B - A)
  print((4 * (A + B) - 4 * A) / 4)
  print()

  A, B = dfloat(A), dfloat(B)
  print("dfloat")
  print(A + B - A)
  print((4 * (A + B) - 4 * A) / 4)

  print(jnp.pi)
  dpi = dfloat(jnp.pi, base_dtype='float32')
  print(dpi._value)
  print(dpi.astype('float64'))