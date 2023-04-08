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
# limitations under the License.

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
    if quat.ndim == 1:
      return cls(_normalize_quaternion(quat))
    elif quat.ndim == 2:
      return cls(jax.vmap(_normalize_quaternion)(quat))

  @classmethod
  def from_matrix(cls, matrix: jax.Array):
    """Initialize from rotation matrix."""
    if matrix.ndim == 2:
      return cls(_from_matrix(matrix))
    elif matrix.ndim == 3:
      return cls(jax.vmap(_from_matrix)(matrix))

  @classmethod
  def from_rotvec(cls, rotvec: jax.Array, degrees: bool = False):
    """Initialize from rotation vectors."""
    if rotvec.ndim == 1:
      return cls(_from_rotvec(rotvec, degrees))
    else:
      return cls(jax.vmap(_from_rotvec, in_axes=[0, None])(rotvec, degrees))

  @classmethod
  def from_euler(cls, seq: str, angles: jax.Array, degrees: bool = False):
    """Initialize from Euler angles."""
    assert seq == 'xyz'
    assert angles.size == 3
    return cls(jnp.roll(_quaternion_from_euler(angles[0], angles[1], angles[2], axes='sxyz'), -1))

  @jax.jit
  def as_quat(self) -> jax.Array:
    """Represent as quaternions."""
    return self.quat

  @jax.jit
  def as_matrix(self) -> jax.Array:
    """Represent as rotation matrix."""
    if self.quat.ndim == 1:
      return _as_matrix(self.quat)
    else:
      return jax.vmap(_as_matrix)(self.quat)

  @functools.partial(jax.jit, static_argnames=['degrees'])
  def as_rotvec(self, degrees: bool = False) -> jax.Array:
    """Represent as rotation vectors."""
    if self.quat.ndim == 1:
      return _as_rotvec(self.quat, degrees)
    else:
      return jax.vmap(_as_rotvec, in_axes=[0, None])(self.quat, degrees)

  @functools.partial(jax.jit, static_argnames=['seq', 'degrees'])
  def as_euler(self, seq: str, degrees: bool = False):
    """Represent as Euler angles."""
    if self.quat.ndim == 1:
      return _as_euler(self.quat, seq, degrees)
    else:
      return jax.vmap(_as_euler, in_axes=[0, None, None])(self.quat, seq, degrees)

  @functools.partial(jax.jit, static_argnames=['inverse'])
  def apply(self, vectors: jax.Array, inverse: bool = False) -> jax.Array:
    """Apply this rotation to one or more vectors."""
    if self.quat.ndim == 1 and vectors.ndim == 1:
      return _apply(self, vectors, inverse)
    else:
      self_axis = None if self.quat.ndim == 1 else 0
      vector_axis = None if vectors.ndim == 1 else 0
      return jax.vmap(_apply, in_axes=[self_axis, vector_axis, None])(self, vectors, inverse)

  def __mul__(self, other):
    """Compose this rotation with the other."""
    if self.quat.ndim == 1 and other.quat.ndim == 1:
      return Rotation.from_quat(_mul(self.quat, other.quat))
    else:
      self_axis = None if self.quat.ndim == 1 else 0
      other_axis = None if other.quat.ndim == 1 else 0
      return Rotation.from_quat(jax.vmap(_mul, in_axes=[self_axis, other_axis])(self.quat, other.quat))

  def inv(self):
    """Invert this rotation."""
    if self.quat.ndim == 1:
      return Rotation(_inv(self.quat))
    else:
      return Rotation(jax.vmap(_inv)(self.quat))

def _apply(rotation, vector, inverse):
  matrix = rotation.as_matrix()
  if inverse:
    result = jnp.einsum('kj,k->j', matrix, vector)
  else:
    result = jnp.einsum('jk,k->j', matrix, vector)
  return result

def _as_euler(quat, seq, degrees):
  assert seq == 'xyz'
  ai, aj, ak = _euler_from_quaternion(jnp.roll(quat, 1), axes='sxyz')
  angles = jnp.array([ai, aj, ak])
  return jnp.where(degrees, jnp.degrees(angles), angles)

def _as_matrix(quat):
  x = quat[0]
  y = quat[1]
  z = quat[2]
  w = quat[3]
  x2 = x * x
  y2 = y * y
  z2 = z * z
  w2 = w * w
  xy = x * y
  zw = z * w
  xz = x * z
  yw = y * w
  yz = y * z
  xw = x * w
  return jnp.array([[+ x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
                    [2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw)],
                    [2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]])

def _as_rotvec(quat, degrees):
  angle = 2 * jnp.arccos(quat[-1])
  wrapped_angle = jnp.where(angle >= jnp.pi, angle - 2*jnp.pi, angle)
  norm = _vector_norm(quat[0:3])
  axis = quat[0:3]
  magnitude = jnp.where(degrees, jnp.degrees(wrapped_angle), wrapped_angle) / norm
  return jnp.where(norm > 0, magnitude * axis, jnp.zeros(3))

def _from_rotvec(rotvec, degrees):
  norm = _vector_norm(rotvec)
  axis = jnp.where(norm > 0, rotvec / norm, jnp.array([1., 0., 0.]))
  angle = jnp.where(degrees, jnp.radians(norm), norm)
  return jnp.roll(_quaternion_about_axis(angle, axis), -1)

def _from_matrix(matrix):
  M = jnp.array(matrix, copy=False)[:3, :3]
  m00 = M[0, 0]
  m01 = M[0, 1]
  m02 = M[0, 2]
  m10 = M[1, 0]
  m11 = M[1, 1]
  m12 = M[1, 2]
  m20 = M[2, 0]
  m21 = M[2, 1]
  m22 = M[2, 2]
  tr = m00 + m11 + m22
  s1 = jnp.sqrt(tr + 1.) * 2.
  q1 = jnp.array([(m21 - m12) / s1, (m02 - m20) / s1, (m10 - m01) / s1, s1 / 4.])
  s2 = jnp.sqrt(1. + m00 - m11 - m22) * 2.
  q2 = jnp.array([s2 / 4., (m01 + m10) / s2, (m02 + m20) / s2, (m21 - m12) / s2])
  s3 = jnp.sqrt(1. + m11 - m00 - m22) * 2.
  q3 = jnp.array([(m01 + m10) / s3, s3 / 4., (m12 + m21) / s3, (m02 - m20) / s3])
  s4 = jnp.sqrt(1. + m22 - m00 - m11) * 2.
  q4 = jnp.array([(m02 + m20) / s4, (m12 + m21) / s4, s4 / 4., (m10 - m01) / s4])
  quat = jnp.where(m11 > m22, q3, q4)
  quat = jnp.where((m00 > m11) & (m00 > m22), q2, quat)
  quat = jnp.where(tr > 0, q1, quat)
  return _normalize_quaternion(quat)

def _mul(quat, other):
  return jnp.roll(_quaternion_multiply(jnp.roll(quat, 1), jnp.roll(other, 1)), -1)

def _inv(quat):
  return jnp.array([quat[0], quat[1], quat[2], -quat[3]])

def _normalize_quaternion(quat):
  return quat / _vector_norm(quat)


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

  M = jnp.array(matrix, copy=False)[:3, :3]
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
  q = jnp.where(qlen > _EPS, q * jnp.sin(angle / 2.0) / qlen, q)
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
  M = jnp.array(matrix, copy=False)[:3, :3]
  m00 = M[0, 0]
  m01 = M[0, 1]
  m02 = M[0, 2]
  m10 = M[1, 0]
  m11 = M[1, 1]
  m12 = M[1, 2]
  m20 = M[2, 0]
  m21 = M[2, 1]
  m22 = M[2, 2]
  # symmetric matrix K
  K = jnp.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                 [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                 [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                 [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
  K /= 3.0
  # quaternion is eigenvector of K that corresponds to largest eigenvalue
  w, V = jnp.linalg.eigh(K, UPLO='L', symmetrize_input=False)
  q = V[[3, 0, 1, 2], jnp.argmax(w)]
  q = lax.cond(q[0] < 0.0, -1.0 * q, lambda x: x, q, lambda x: x)
  return q


def _quaternion_matrix(quaternion):
  q = jnp.array(quaternion, copy=True)
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
