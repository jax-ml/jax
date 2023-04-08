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
  def concatenate(cls, rotations: typing.Sequence):
    return cls(jnp.vstack([rotation.quat for rotation in rotations]))

  @classmethod
  def from_euler(cls, seq: str, angles: jax.Array, degrees: bool = False):
    """Initialize from Euler angles."""
    assert angles.ndim in [1, 2]
    if angles.ndim == 1:
      return cls(_from_euler(seq, angles, degrees))
    else:
      return cls(jax.vmap(_from_euler, in_axes=[None, 0, None])(seq, angles, degrees))

  @classmethod
  def from_matrix(cls, matrix: jax.Array):
    """Initialize from rotation matrix."""
    assert matrix.ndim in [2, 3]
    if matrix.ndim == 2:
      return cls(_from_matrix(matrix))
    else:
      return cls(jax.vmap(_from_matrix)(matrix))

  @classmethod
  def from_mrp(cls, mrp: jax.Array):
    """Initialize from Modified Rodrigues Parameters (MRPs)."""
    assert mrp.ndim in [1, 2]
    if mrp.ndim == 1:
      return cls(_from_mrp(mrp))
    else:
      return cls(jax.vmap(_from_mrp)(mrp))

  @classmethod
  def from_quat(cls, quat: jax.Array):
    """Initialize from quaternions."""
    assert quat.ndim in [1, 2]
    if quat.ndim == 1:
      return cls(_normalize_quaternion(quat))
    else:
      return cls(jax.vmap(_normalize_quaternion)(quat))

  @classmethod
  def from_rotvec(cls, rotvec: jax.Array, degrees: bool = False):
    """Initialize from rotation vectors."""
    assert rotvec.ndim in [1, 2]
    if rotvec.ndim == 1:
      return cls(_from_rotvec(rotvec, degrees))
    else:
      return cls(jax.vmap(_from_rotvec, in_axes=[0, None])(rotvec, degrees))

  @classmethod
  def identity(cls, num: typing.Optional[int] = None):
    """Get identity rotation(s)."""
    assert num is None
    quat = jnp.array([0, 0, 0, 1])
    return cls(quat)

  def __bool__(self):
    """Comply with Python convention for objects to be True."""
    return True

  def __getitem__(self, indexer):
    """Extract rotation(s) at given index(es) from object."""
    if self.quat.ndim == 1:
      raise TypeError("Single rotation is not subscriptable.")
    return self.__class__(self.quat[indexer])

  def __len__(self):
    """Number of rotations contained in this object."""
    if self.quat.ndim == 1:
      raise TypeError('Single rotation has no len().')
    else:
      return self.quat.shape[0]

  def __mul__(self, other):
    """Compose this rotation with the other."""
    if self.quat.ndim == 1 and other.quat.ndim == 1:
      return Rotation.from_quat(_compose_quat(self.quat, other.quat))
    else:
      self_axis = None if self.quat.ndim == 1 else 0
      other_axis = None if other.quat.ndim == 1 else 0
      return Rotation.from_quat(jax.vmap(_compose_quat, in_axes=[self_axis, other_axis])(self.quat, other.quat))

  @functools.partial(jax.jit, static_argnames=['inverse'])
  def apply(self, vectors: jax.Array, inverse: bool = False) -> jax.Array:
    """Apply this rotation to one or more vectors."""
    if self.quat.ndim == 1 and vectors.ndim == 1:
      return _apply(self, vectors, inverse)
    else:
      self_axis = None if self.quat.ndim == 1 else 0
      vector_axis = None if vectors.ndim == 1 else 0
      return jax.vmap(_apply, in_axes=[self_axis, vector_axis, None])(self, vectors, inverse)

  @functools.partial(jax.jit, static_argnames=['seq', 'degrees'])
  def as_euler(self, seq: str, degrees: bool = False):
    """Represent as Euler angles."""
    if self.quat.ndim == 1:
      return _as_euler(self.quat, seq, degrees)
    else:
      return jax.vmap(_as_euler, in_axes=[0, None, None])(self.quat, seq, degrees)

  def as_matrix(self) -> jax.Array:
    """Represent as rotation matrix."""
    if self.quat.ndim == 1:
      return _as_matrix(self.quat)
    else:
      return jax.vmap(_as_matrix)(self.quat)

  def as_mrp(self) -> jax.Array:
    """Represent as Modified Rodrigues Parameters (MRPs)."""
    if self.quat.ndim == 1:
      return _as_mrp(self.quat)
    else:
      return jax.vmap(_as_mrp)(self.quat)

  @functools.partial(jax.jit, static_argnames=['degrees'])
  def as_rotvec(self, degrees: bool = False) -> jax.Array:
    """Represent as rotation vectors."""
    if self.quat.ndim == 1:
      return _as_rotvec(self.quat, degrees)
    else:
      return jax.vmap(_as_rotvec, in_axes=[0, None])(self.quat, degrees)

  def as_quat(self) -> jax.Array:
    """Represent as quaternions."""
    return self.quat

  def inv(self):
    """Invert this rotation."""
    if self.quat.ndim == 1:
      return Rotation(_inv(self.quat))
    else:
      return Rotation(jax.vmap(_inv)(self.quat))

  def magnitude(self) -> jax.Array:
    """Get the magnitude(s) of the rotation(s)."""
    if self.quat.ndim == 1:
      return _magnitude(self.quat)
    else:
      return jax.vmap(_magnitude)(self.quat)

  # def mean(self) -> jax.Array:
  #   """Get the mean of the rotations."""
  #   if self.quat.ndim == 1:
  #     return self.__class__(_mean(self.quat))
  #   else:
  #     return self.__class__(jax.vmap(_mean)(self.quat))

  @property
  def single(self) -> bool:
    """Whether this instance represents a single rotation."""
    return self.quat.ndim == 1


class Slerp(typing.NamedTuple):
  """Spherical Linear Interpolation of Rotations."""

  times: jnp.ndarray
  timedelta: jnp.ndarray
  rotations: Rotation
  rotvecs: jnp.ndarray

  @classmethod
  def init(cls, times: jax.Array, rotations: Rotation):
    if not isinstance(rotations, Rotation):
      raise TypeError("`rotations` must be a `Rotation` instance.")
    if rotations.single or len(rotations) == 1:
      raise ValueError("`rotations` must be a sequence of at least 2 rotations.")
    times = jnp.asarray(times)
    if times.ndim != 1:
      raise ValueError("Expected times to be specified in a 1 "
                       "dimensional array, got {} "
                       "dimensions.".format(times.ndim))
    if times.shape[0] != len(rotations):
      raise ValueError("Expected number of rotations to be equal to "
                       "number of timestamps given, got {} rotations "
                       "and {} timestamps.".format(len(rotations), times.shape[0]))
    timedelta = jnp.diff(times)
    if jnp.any(timedelta <= 0):
      raise ValueError("Times must be in strictly increasing order.")
    new_rotations = rotations[:-1]
    return cls(
      times=times,
      timedelta=jnp.diff(times),
      rotations=new_rotations,
      rotvecs=(new_rotations.inv() * rotations[1:]).as_rotvec())

  def __call__(self, times: jax.Array):
    """Interpolate rotations."""
    times = jnp.asarray(times)
    if times.ndim > 1:
      raise ValueError("`times` must be at most 1-dimensional.")
    compute_times = jnp.atleast_1d(times)
    ind = jnp.searchsorted(self.times, compute_times) - 1
    ind = ind.at[compute_times <= self.times[0]].set(0)
    alpha = (compute_times - self.times[ind]) / self.timedelta[ind]
    result = (self.rotations[ind] * Rotation.from_rotvec(self.rotvecs[ind] * alpha[:, None]))
    if times.ndim == 0:
      return result[0]
    return result


def _apply(rotation: Rotation, vector: jax.Array, inverse: bool) -> jax.Array:
  matrix = rotation.as_matrix()
  if inverse:
    result = jnp.einsum('kj,k->j', matrix, vector)
  else:
    result = jnp.einsum('jk,k->j', matrix, vector)
  return result

def _as_euler(quat: jax.Array, seq: str, degrees: bool) -> jax.Array:
  assert seq == 'xyz'
  ai, aj, ak = _euler_from_quaternion(jnp.roll(quat, 1), axes='sxyz')
  angles = jnp.array([ai, aj, ak])
  return jnp.where(degrees, jnp.degrees(angles), angles)

def _as_matrix(quat: jax.Array) -> jax.Array:
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

def _as_mrp(quat: jax.Array) -> jax.Array:
  sign = jnp.where(quat[3] < 0, -1., 1.)
  denominator = 1. + sign * quat[3]
  return sign * quat[:3] / denominator

def _as_rotvec(quat: jax.Array, degrees: bool) -> jax.Array:
  angle = 2 * jnp.arccos(quat[-1])
  wrapped_angle = jnp.where(angle >= jnp.pi, angle - 2*jnp.pi, angle)
  norm = _vector_norm(quat[0:3])
  axis = quat[0:3]
  scale = jnp.where(degrees, jnp.degrees(wrapped_angle), wrapped_angle) / norm
  return jnp.where(norm > 0, scale * axis, jnp.zeros(3))

def _compose_quat(p: jax.Array, q: jax.Array) -> jax.Array:
  cross = jnp.cross(p[:3], q[:3])
  return jnp.array([p[3]*q[0] + q[3]*p[0] + cross[0],
                    p[3]*q[1] + q[3]*p[1] + cross[1],
                    p[3]*q[2] + q[3]*p[2] + cross[2],
                    p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]])

def _from_euler(seq: str, angles: jax.Array, degrees: bool) -> jax.Array:
  a = jnp.where(degrees, jnp.radians(angles), angles)
  return jnp.roll(_quaternion_from_euler(a[0], a[1], a[2], axes='sxyz'), -1)

def _from_rotvec(rotvec: jax.Array, degrees: bool) -> jax.Array:
  norm = _vector_norm(rotvec)
  axis = jnp.where(norm > 0, rotvec / norm, jnp.array([1., 0., 0.]))
  angle = jnp.where(degrees, jnp.radians(norm), norm)
  return jnp.roll(_quaternion_about_axis(angle, axis), -1)

def _from_matrix(matrix: jax.Array) -> jax.Array:
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

def _from_mrp(mrp: jax.Array) -> jax.Array:
  mrp_squared_plus_1 = jnp.dot(mrp, mrp) + 1
  return jnp.hstack([2 * mrp[:3], (2 - mrp_squared_plus_1)]) / mrp_squared_plus_1

def _inv(quat: jax.Array) -> jax.Array:
  return jnp.array([quat[0], quat[1], quat[2], -quat[3]])

def _magnitude(quat: jax.Array) -> jax.Array:
  return 2. * jnp.arctan2(_vector_norm(quat[:3]), jnp.abs(quat[3]))

def _mean(quat: jax.Array) -> jax.Array:
  K = jnp.dot(quat.T, quat)
  _, v = jnp.linalg.eigh(K)
  return v[:, -1]

def _normalize_quaternion(quat: jax.Array) -> jax.Array:
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
  firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]

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


def _vector_norm(data):
  return jnp.sqrt(jnp.dot(data, data))


# if __name__ == '__main__':
#   import numpy as onp
#   times = 0.
#   num = 5
#   rotations = Rotation.from_quat(onp.random.randn(num, 4))
#   slerp = Slerp.init(list(range(num)), rotations)
#   result = slerp(times)
#   import pdb; pdb.set_trace()
