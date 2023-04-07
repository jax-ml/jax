import functools
import typing

import jax
import jax.lax as lax
import jax.numpy as jnp


class Rotation(typing.NamedTuple):
  """Rotation in 3 dimensions."""

  quat: jax.Array

  @classmethod
  def from_quat(cls, quat: jax.Array):
    """Initialize from quaternions."""
    return cls(quat / _vector_norm(quat))

  @classmethod
  def from_matrix(cls, matrix: jax.Array):
    """Initialize from rotation matrix."""
    return cls(jnp.roll(_quaternion_from_matrix(matrix), -1))

  @classmethod
  def from_rotvec(cls, rotvec: jax.Array, degrees: bool = False):
    """Initialize from rotation vectors."""
    norm = _vector_norm(rotvec)
    axis = jnp.where(norm > 0, rotvec / norm, jnp.array([1., 0., 0.]))
    angle = jnp.where(degrees, jnp.radians(norm), norm)
    return cls(jnp.roll(_quaternion_about_axis(angle, axis), -1))

  @classmethod
  def from_euler(cls, seq: str, angles: jax.Array, degrees: bool = False):
    """Initialize from Euler angles."""
    assert seq == 'xyz'
    assert angles.size == 3
    ai = angles[0]
    aj = angles[1]
    ak = angles[2]
    return cls(jnp.roll(_quaternion_from_euler(ai, aj, ak, axes='sxyz'), -1))

  @jax.jit
  def as_quat(self) -> jax.Array:
    """Represent as quaternions."""
    return self.quat

  @jax.jit
  def as_matrix(self) -> jax.Array:
    """Represent as rotation matrix."""
    return _quaternion_matrix(jnp.roll(self.quat, 1))[:3, :3]

  @functools.partial(jax.jit, static_argnames=['degrees'])
  def as_rotvec(self, degrees: bool = False) -> jax.Array:
    """Represent as rotation vectors."""
    angle = 2 * jnp.arccos(self.quat[-1])
    wrapped_angle = jnp.where(angle >= jnp.pi, angle - 2*jnp.pi, angle)
    norm = _vector_norm(self.quat[0:3])
    axis = self.quat[0:3]
    return jnp.where(norm > 0, wrapped_angle * axis / norm, jnp.zeros(3))

  @functools.partial(jax.jit, static_argnames=['seq', 'degrees'])
  def as_euler(self, seq: str, degrees: bool = False):
    """Represent as Euler angles."""
    assert seq == 'xyz'
    ai, aj, ak = _euler_from_quaternion(jnp.roll(self.quat, 1), axes='sxyz')
    angles = jnp.array([ai, aj, ak])
    return jnp.where(degrees, jnp.degrees(angles), angles)

  @functools.partial(jax.jit, static_argnames=['inverse'])
  def apply(self, vector: jax.Array, inverse: bool = False) -> jax.Array:
    """Apply this rotation to a single vector."""
    matrix = self.as_matrix()
    if inverse:
      result = jnp.einsum('kj,k->j', matrix, vector)
    else:
      result = jnp.einsum('jk,k->j', matrix, vector)
    return result

  @jax.jit
  def __mul__(self, other):
    """Compose this rotation with the other."""
    quat = _quaternion_multiply(jnp.roll(self.quat, 1), jnp.roll(other.quat, 1))
    return Rotation(jnp.roll(quat, -1))

  @jax.jit
  def inv(self):
    """Invert this rotation."""
    return Rotation(jnp.array([self.quat[0], self.quat[1], self.quat[2], -self.quat[3]]))


_EPS = jnp.finfo(float).eps * 4.0

_NEXT_AXIS = [1, 2, 0, 1]

_AXES2TUPLE = {
  'sxyz': (0, 0, 0, 0),
  'sxyx': (0, 0, 1, 0),
  'sxzy': (0, 1, 0, 0),
  'sxzx': (0, 1, 1, 0),
  'syzx': (1, 0, 0, 0),
  'syzy': (1, 0, 1, 0),
  'syxz': (1, 1, 0, 0),
  'syxy': (1, 1, 1, 0),
  'szxy': (2, 0, 0, 0),
  'szxz': (2, 0, 1, 0),
  'szyx': (2, 1, 0, 0),
  'szyz': (2, 1, 1, 0),
  'rzyx': (0, 0, 0, 1),
  'rxyx': (0, 0, 1, 1),
  'ryzx': (0, 1, 0, 1),
  'rxzx': (0, 1, 1, 1),
  'rxzy': (1, 0, 0, 1),
  'ryzy': (1, 0, 1, 1),
  'rzxy': (1, 1, 0, 1),
  'ryxy': (1, 1, 1, 1),
  'ryxz': (2, 0, 0, 1),
  'rzxz': (2, 0, 1, 1),
  'rxyz': (2, 1, 0, 1),
  'rzyz': (2, 1, 1, 1)
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def _euler_from_matrix(matrix, axes='sxyz'):
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis
  j = _NEXT_AXIS[i + parity]
  k = _NEXT_AXIS[i - parity + 1]

  M = jnp.array(matrix, dtype=jnp.float64, copy=False)[:3, :3]
  if repetition:
    sy = jnp.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
    ax, ay, az = lax.cond(
        sy > _EPS, (jnp.arctan2(M[i, j], M[i, k]), jnp.arctan2(
            sy, M[i, i]), jnp.arctan2(M[j, i], -M[k, i])), lambda x: x,
        (jnp.arctan2(-M[j, k], M[j, j]), jnp.arctan2(sy, M[i, i]), 0.0),
        lambda x: x)
  else:
    cy = jnp.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
    ax, ay, az = lax.cond(
        cy > _EPS, (jnp.arctan2(M[k, j], M[k, k]), jnp.arctan2(
            -M[k, i], cy), jnp.arctan2(M[j, i], M[i, i])), lambda x: x,
        (jnp.arctan2(-M[j, k], M[j, j]), jnp.arctan2(-M[k, i], cy), 0.0),
        lambda x: x)

  if parity:
    ax, ay, az = -ax, -ay, -az
  if frame:
    ax, az = az, ax
  return ax, ay, az


def _euler_from_quaternion(quaternion, axes='sxyz'):
  return _euler_from_matrix(_quaternion_matrix(quaternion), axes)


def _quaternion_about_axis(angle, axis):
  q = jnp.array([0.0, axis[0], axis[1], axis[2]])
  qlen = _vector_norm(q)
  q = lax.cond(qlen > _EPS, q * jnp.sin(angle / 2.0) / qlen, lambda x: x, q, lambda x: x)
  q = q.at[0].set(jnp.cos(angle / 2.0))
  return q


def _quaternion_from_euler(ai, aj, ak, axes='sxyz'):
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
  firstaxis, parity, repetition, frame = axes

  i = firstaxis + 1
  j = _NEXT_AXIS[i + parity - 1] + 1
  k = _NEXT_AXIS[i - parity] + 1

  if frame:
    ai, ak = ak, ai
  if parity:
    aj = -aj

  ai /= 2.0
  aj /= 2.0
  ak /= 2.0
  ci = jnp.cos(ai)
  si = jnp.sin(ai)
  cj = jnp.cos(aj)
  sj = jnp.sin(aj)
  ck = jnp.cos(ak)
  sk = jnp.sin(ak)
  cc = ci * ck
  cs = ci * sk
  sc = si * ck
  ss = si * sk

  q = jnp.empty((4,))
  if repetition:
    q = q.at[0].set(cj * (cc - ss))
    q = q.at[i].set(cj * (cs + sc))
    q = q.at[j].set(sj * (cc + ss))
    q = q.at[k].set(sj * (cs - sc))
  else:
    q = q.at[0].set(cj * cc + sj * ss)
    q = q.at[i].set(cj * sc - sj * cs)
    q = q.at[j].set(cj * ss + sj * cc)
    q = q.at[k].set(cj * cs - sj * sc)
  if parity:
    q = q.at[j].multiply(-1.0)
  return q


def _quaternion_from_matrix(matrix):
  m00 = matrix[0, 0]
  m01 = matrix[0, 1]
  m02 = matrix[0, 2]
  m10 = matrix[1, 0]
  m11 = matrix[1, 1]
  m12 = matrix[1, 2]
  m20 = matrix[2, 0]
  m21 = matrix[2, 1]
  m22 = matrix[2, 2]
  K = jnp.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
           [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
           [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
           [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
  K /= 3.0
  w, V = jnp.linalg.eigh(K, UPLO='L', symmetrize_input=False)
  q = V[[3, 0, 1, 2], jnp.argmax(w)]
  q = lax.cond(q[0] < 0.0, -1.0 * q, lambda x: x, q, lambda x: x)
  return q


def _quaternion_matrix(quaternion):
  q = jnp.array(quaternion, dtype=jnp.float64, copy=True)
  n = jnp.dot(q, q)
  def calc_mat_posn(qn):
    q, n = qn
    q *= jnp.sqrt(2.0 / n)
    q = jnp.outer(q, q)
    return jnp.array(
      [[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
       [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
       [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
       [0.0, 0.0, 0.0, 1.0]])
  return lax.cond(n < _EPS, jnp.identity(4), lambda x: x, (q, n), calc_mat_posn)


def _quaternion_multiply(quaternion1, quaternion0):
  w0, x0, y0, z0 = quaternion0
  w1, x1, y1, z1 = quaternion1
  return jnp.array([
    -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, x1 * w0 + y1 * z0 - z1 * y0 +
    w1 * x0, -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])


def _quaternion_conjugate(quaternion):
  q = quaternion.at[1:].set(-quaternion[1:])
  return q


def _quaternion_inverse(quaternion):
  q = _quaternion_conjugate(quaternion)
  return q / jnp.dot(q, q)


def _vector_norm(data):
  return jnp.sqrt(jnp.dot(data, data))
