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
import re
import typing

import jax
import jax.numpy as jnp


class Rotation(typing.NamedTuple):
  """Rotation in 3 dimensions."""

  quat: jax.Array

  @classmethod
  def concatenate(cls, rotations: typing.Sequence):
    """Concatenate a sequence of `Rotation` objects."""
    return cls(jnp.concatenate([rotation.quat for rotation in rotations]))

  @classmethod
  def from_euler(cls, seq: str, angles: jax.Array, degrees: bool = False):
    """Initialize from Euler angles."""
    assert angles.ndim in [1, 2]
    num_axes = len(seq)
    if num_axes < 1 or num_axes > 3:
      raise ValueError("Expected axis specification to be a non-empty "
                       "string of upto 3 characters, got {}".format(seq))
    intrinsic = (re.match(r'^[XYZ]{1,3}$', seq) is not None)
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    if not (intrinsic or extrinsic):
      raise ValueError("Expected axes from `seq` to be from ['x', 'y', "
                       "'z'] or ['X', 'Y', 'Z'], got {}".format(seq))
    if any(seq[i] == seq[i+1] for i in range(num_axes - 1)):
      raise ValueError("Expected consecutive axes to be different, "
                       "got {}".format(seq))
    seq = seq.lower()
    _from_euler = lambda a: _elementary_quat_compose(a, seq, intrinsic, degrees)
    if angles.ndim == 1:
      return cls(_from_euler(angles))
    else:
      return cls(jax.vmap(_from_euler)(angles))

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

  # @classmethod
  # def random(cls, random_key: jax.random.PRNGKey, num: typing.Optional[int] = None):
  #   """Generate uniformly distributed rotations."""
  #   # Need to implement scipy.stats.special_ortho_group for this to work...

  def __bool__(self):
    """Comply with Python convention for objects to be True."""
    return True

  def __getitem__(self, indexer):
    """Extract rotation(s) at given index(es) from object."""
    if self.single:
      raise TypeError("Single rotation is not subscriptable.")
    return Rotation(self.quat[indexer])

  def __len__(self):
    """Number of rotations contained in this object."""
    if self.single:
      raise TypeError('Single rotation has no len().')
    else:
      return self.quat.shape[0]

  def __mul__(self, other):
    """Compose this rotation with the other."""
    if self.single and other.single:
      return Rotation.from_quat(_compose_quat(self.quat, other.quat))
    else:
      self_axis = None if self.single else 0
      other_axis = None if other.quat.ndim == 1 else 0
      return Rotation.from_quat(jax.vmap(_compose_quat, in_axes=[self_axis, other_axis])(self.quat, other.quat))

  @functools.partial(jax.jit, static_argnames=['inverse'])
  def apply(self, vectors: jax.Array, inverse: bool = False) -> jax.Array:
    """Apply this rotation to one or more vectors."""
    if self.single and vectors.ndim == 1:
      return _apply(self, vectors, inverse)
    else:
      self_axis = None if self.single else 0
      vector_axis = None if vectors.ndim == 1 else 0
      return jax.vmap(_apply, in_axes=[self_axis, vector_axis, None])(self, vectors, inverse)

  @functools.partial(jax.jit, static_argnames=['seq', 'degrees'])
  def as_euler(self, seq: str, degrees: bool = False):
    """Represent as Euler angles."""
    if len(seq) != 3:
      raise ValueError("Expected 3 axes, got {}.".format(seq))
    intrinsic = (re.match(r'^[XYZ]{1,3}$', seq) is not None)
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    if not (intrinsic or extrinsic):
      raise ValueError("Expected axes from `seq` to be from "
                       "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                       "got {}".format(seq))
    if any(seq[i] == seq[i+1] for i in range(2)):
      raise ValueError("Expected consecutive axes to be different, "
                       "got {}".format(seq))
    seq = seq.lower()
    _as_euler = lambda q: _compute_euler_from_quat(q, seq, extrinsic, degrees)
    if self.single:
      return _as_euler(self.quat)
    else:
      return jax.vmap(_as_euler)(self.quat)

  def as_matrix(self) -> jax.Array:
    """Represent as rotation matrix."""
    if self.single:
      return _as_matrix(self.quat)
    else:
      return jax.vmap(_as_matrix)(self.quat)

  def as_mrp(self) -> jax.Array:
    """Represent as Modified Rodrigues Parameters (MRPs)."""
    if self.single:
      return _as_mrp(self.quat)
    else:
      return jax.vmap(_as_mrp)(self.quat)

  @functools.partial(jax.jit, static_argnames=['degrees'])
  def as_rotvec(self, degrees: bool = False) -> jax.Array:
    """Represent as rotation vectors."""
    if self.single:
      return _as_rotvec(self.quat, degrees)
    else:
      return jax.vmap(_as_rotvec, in_axes=[0, None])(self.quat, degrees)

  def as_quat(self) -> jax.Array:
    """Represent as quaternions."""
    return self.quat

  def inv(self):
    """Invert this rotation."""
    if self.single:
      return Rotation(_inv(self.quat))
    else:
      return Rotation(jax.vmap(_inv)(self.quat))

  def magnitude(self) -> jax.Array:
    """Get the magnitude(s) of the rotation(s)."""
    if self.single:
      return _magnitude(self.quat)
    else:
      return jax.vmap(_magnitude)(self.quat)

  def mean(self, weights: typing.Optional[jax.Array] = None) -> jax.Array:
    """Get the mean of the rotations."""
    weights = jnp.where(weights is None, jnp.ones(self.quat.shape[0]), jnp.asarray(weights))
    if weights.ndim != 1:
      raise ValueError("Expected `weights` to be 1 dimensional, got "
                       "shape {}.".format(weights.shape))
    if weights.shape[0] != len(self):
      raise ValueError("Expected `weights` to have number of values "
                       "equal to number of rotations, got "
                       "{} values and {} rotations.".format(weights.shape[0], len(self)))
    K = jnp.dot(weights[jnp.newaxis, :] * self.quat.T, self.quat)
    _, v = jnp.linalg.eigh(K)
    return Rotation(v[:, -1])

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
    # if jnp.any(timedelta <= 0):  # this causes a concretization error...
    #   raise ValueError("Times must be in strictly increasing order.")
    new_rotations = rotations[:-1]
    return cls(
      times=times,
      timedelta=timedelta,
      rotations=new_rotations,
      rotvecs=(new_rotations.inv() * rotations[1:]).as_rotvec())

  def __call__(self, times: jax.Array):
    """Interpolate rotations."""
    compute_times = jnp.asarray(times)
    if compute_times.ndim > 1:
      raise ValueError("`times` must be at most 1-dimensional.")
    single_time = compute_times.ndim == 0
    compute_times = jnp.atleast_1d(compute_times)
    ind = jnp.maximum(jnp.searchsorted(self.times, compute_times) - 1, 0)
    alpha = (compute_times - self.times[ind]) / self.timedelta[ind]
    result = (self.rotations[ind] * Rotation.from_rotvec(self.rotvecs[ind] * alpha[:, None]))
    if single_time:
      return result[0]
    return result


def _apply(rotation: Rotation, vector: jax.Array, inverse: bool) -> jax.Array:
  matrix = rotation.as_matrix()
  if inverse:
    result = jnp.einsum('kj,k->j', matrix, vector)
  else:
    result = jnp.einsum('jk,k->j', matrix, vector)
  return result


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
  quat = jnp.where(quat[3] < 0, -quat, quat)  # w > 0 to ensure 0 <= angle <= pi
  angle = 2. * jnp.arctan2(_vector_norm(quat[:3]), quat[3])
  angle2 = angle * angle
  small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
  large_scale = angle / jnp.sin(angle / 2)
  scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
  scale = jnp.where(degrees, jnp.rad2deg(scale), scale)
  return scale * jnp.array(quat[:3])


def _compose_quat(p: jax.Array, q: jax.Array) -> jax.Array:
  cross = jnp.cross(p[:3], q[:3])
  return jnp.array([p[3]*q[0] + q[3]*p[0] + cross[0],
                    p[3]*q[1] + q[3]*p[1] + cross[1],
                    p[3]*q[2] + q[3]*p[2] + cross[2],
                    p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]])


def _compute_euler_from_quat(quat: jax.Array, seq: str, extrinsic: bool, degrees: bool) -> jax.Array:
  if extrinsic:
    angle_first = 0
    angle_third = 2
  else:
    seq = seq[::-1]
    angle_first = 2
    angle_third = 0
  i = _elementary_basis_index(seq[0])
  j = _elementary_basis_index(seq[1])
  k = _elementary_basis_index(seq[2])
  symmetric = i == k
  k = jnp.where(symmetric, 3 - i - j, k)
  sign = (i - j) * (j - k) * (k - i) // 2
  eps = 1e-7
  a = jnp.where(symmetric, quat[3], quat[3] - quat[j])
  b = jnp.where(symmetric, quat[i], quat[i] + quat[k] * sign)
  c = jnp.where(symmetric, quat[j], quat[j] + quat[3])
  d = jnp.where(symmetric, quat[k] * sign, quat[k] * sign - quat[i])
  angles = jnp.empty(3)
  angles = angles.at[1].set(2 * jnp.arctan2(jnp.hypot(c, d), jnp.hypot(a, b)))
  case = jnp.where(jnp.abs(angles[1] - jnp.pi) <= eps, 2, 0)
  case = jnp.where(jnp.abs(angles[1]) <= eps, 1, case)
  half_sum = jnp.arctan2(b, a)
  half_diff = jnp.arctan2(d, c)
  angles = angles.at[0].set(jnp.where(case == 1, 2 * half_sum, 2 * half_diff * jnp.where(extrinsic, -1, 1)))  # any degenerate case
  angles = angles.at[angle_first].set(jnp.where(case == 0, half_sum - half_diff, angles[angle_first]))
  angles = angles.at[angle_third].set(jnp.where(case == 0, half_sum + half_diff, angles[angle_third]))
  angles = angles.at[angle_third].set(jnp.where(not symmetric, angles[angle_third] * sign, angles[angle_third]))
  angles = angles.at[1].set(jnp.where(not symmetric, angles[1] - jnp.pi / 2, angles[1]))
  angles = (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
  return jnp.where(degrees, jnp.rad2deg(angles), angles)
  return angles


def _elementary_basis_index(axis: str):
  if axis == 'x':
    return 0
  elif axis == 'y':
    return 1
  elif axis == 'z':
    return 2


def _elementary_quat_compose(angles: jax.Array, seq: str, intrinsic: bool, degrees: bool):
  if degrees:
    angles = jnp.deg2rad(angles)
  result = _make_elementary_quat(seq[0], angles[0])
  for idx in range(1, len(seq)):
    if intrinsic:
      result = _compose_quat(result, _make_elementary_quat(seq[idx], angles[idx]))
    else:
      result = _compose_quat(_make_elementary_quat(seq[idx], angles[idx]), result)
  return result


def _from_rotvec(rotvec: jax.Array, degrees: bool) -> jax.Array:
  rotvec = jnp.where(degrees, jnp.deg2rad(rotvec), rotvec)
  angle = _vector_norm(rotvec)
  angle2 = angle * angle
  small_scale = scale = 0.5 - angle2 / 48 + angle2 * angle2 / 3840
  large_scale = jnp.sin(angle / 2) / angle
  scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
  return jnp.hstack([scale * rotvec, jnp.cos(angle / 2)])


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
  return quat.at[3].set(-quat[3])


def _magnitude(quat: jax.Array) -> jax.Array:
  return 2. * jnp.arctan2(_vector_norm(quat[:3]), jnp.abs(quat[3]))


def _make_elementary_quat(axis: jax.Array, angle: jax.Array):
  axis_ind = _elementary_basis_index(axis)
  quat = jnp.zeros(4)
  quat = quat.at[3].set(jnp.cos(angle / 2.))
  quat = quat.at[axis_ind].set(jnp.sin(angle / 2.))
  return quat

def _normalize_quaternion(quat: jax.Array) -> jax.Array:
  return quat / _vector_norm(quat)


def _vector_norm(vector):
  return jnp.sqrt(jnp.dot(vector, vector))
