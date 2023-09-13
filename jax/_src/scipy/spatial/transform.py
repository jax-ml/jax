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

import scipy.spatial.transform

import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from scipy.constants import golden


@_wraps(scipy.spatial.transform.Rotation)
class Rotation(typing.NamedTuple):
  """Rotation in 3 dimensions."""

  quat: jax.Array

  @classmethod
  def align_vectors(cls, a: jax.Array, b: jax.Array, weights: typing.Optional[jax.Array] = None, return_sensitivity: bool = False):
    """Estimate a rotation to optimally align two sets of vectors."""
    a = jnp.asarray(a)
    if a.ndim < 2 or a.shape[-1] != 3:
      raise ValueError("Expected input `a` to have shape (..., 3), "
                       "got {}".format(a.shape))
    b = jnp.asarray(b)
    if b.ndim < 2 or b.shape[-1] != 3:
      raise ValueError("Expected input `b` to have shape (..., 3), "
                       "got {}.".format(b.shape))
    if weights is None:
      weights = jnp.ones(b.shape[-2], dtype=b.dtype)
    else:
      weights = jnp.asarray(weights, dtype=b.dtype)
      # if jnp.any(weights < 0):  # this causes a concretization error...
      #   raise ValueError("`weights` may not contain negative values")
    matrix, rssd, sensitivity = _align_vectors(a, b, weights)
    if return_sensitivity:
      return cls.from_matrix(matrix), rssd, sensitivity
    else:
      return cls.from_matrix(matrix), rssd

  @classmethod
  def create_group(cls, group: str, axis: str = 'Z', dtype=float):
    """Create a 3D rotation group."""
    if not isinstance(group, str):
      raise ValueError("`group` argument must be a string")
    permitted_axes = ['x', 'y', 'z', 'X', 'Y', 'Z']
    if axis not in permitted_axes:
      raise ValueError("`axis` must be one of " + ", ".join(permitted_axes))
    if group in ['I', 'O', 'T']:
      symbol = group
      order = 1
    elif group[:1] in ['C', 'D'] and group[1:].isdigit():
      symbol = group[:1]
      order = int(group[1:])
    else:
      raise ValueError("`group` must be one of 'I', 'O', 'T', 'Dn', 'Cn'")
    axis_index = _elementary_basis_index(axis.lower())
    if order < 1:
      raise ValueError("Group order must be positive")
    if symbol == 'I':
      quat = _create_icosahedral_group()
    elif symbol == 'O':
      quat = _create_octahedral_group()
    elif symbol == 'T':
      quat = _create_tetrahedral_group()
    elif symbol == 'D':
      quat = _create_dicyclic_group(order, axis=axis_index)
    elif symbol == 'C':
      quat = _create_cyclic_group(order, axis=axis_index)
    else:
      assert False
    return cls.from_quat(quat)

  @classmethod
  def concatenate(cls, rotations: typing.Sequence):
    """Concatenate a sequence of `Rotation` objects."""
    return cls(jnp.concatenate([rotation.quat for rotation in rotations]))

  @classmethod
  def from_euler(cls, seq: str, angles: jax.Array, degrees: bool = False):
    """Initialize from Euler angles."""
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
    angles = jnp.atleast_1d(angles)
    if len(seq) == 1 and angles.ndim == 1:
      angles = angles[:, jnp.newaxis]
    axes = jnp.array([_elementary_basis_index(x) for x in seq.lower()])
    quat = _elementary_quat_compose(angles, axes, intrinsic, degrees)
    return cls(quat)

  @classmethod
  def from_matrix(cls, matrix: jax.Array):
    """Initialize from rotation matrix."""
    return cls(_from_matrix(matrix))

  @classmethod
  def from_mrp(cls, mrp: jax.Array):
    """Initialize from Modified Rodrigues Parameters (MRPs)."""
    return cls(_from_mrp(mrp))

  @classmethod
  def from_quat(cls, quat: jax.Array):
    """Initialize from quaternions."""
    return cls(_normalize_quaternion(quat))

  @classmethod
  def from_rotvec(cls, rotvec: jax.Array, degrees: bool = False):
    """Initialize from rotation vectors."""
    return cls(_from_rotvec(rotvec, degrees))

  @classmethod
  def identity(cls, num: typing.Optional[int] = None, dtype=float):
    """Get identity rotation(s)."""
    assert num is None
    quat = jnp.array([0., 0., 0., 1.], dtype=dtype)
    return cls(quat)

  @classmethod
  def random(cls, random_key: jax.Array, num: typing.Optional[int] = None, dtype=float):
    """Generate uniformly distributed rotations."""
    quat = _random_quaternion(random_key=random_key, num=num, dtype=dtype)
    return cls(quat)

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
    return Rotation.from_quat(_compose_quat(self.quat, other.quat))

  def apply(self, vectors: jax.Array, inverse: bool = False) -> jax.Array:
    """Apply this rotation to one or more vectors."""
    return _apply(self.as_matrix(), vectors, inverse)

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
    axes = jnp.array([_elementary_basis_index(x) for x in seq.lower()])
    return _compute_euler_from_quat(self.quat, axes, extrinsic, degrees)

  def as_matrix(self) -> jax.Array:
    """Represent as rotation matrix."""
    return _as_matrix(self.quat)

  def as_mrp(self) -> jax.Array:
    """Represent as Modified Rodrigues Parameters (MRPs)."""
    return _as_mrp(self.quat)

  def as_rotvec(self, degrees: bool = False) -> jax.Array:
    """Represent as rotation vectors."""
    return _as_rotvec(self.quat, degrees)

  def as_quat(self) -> jax.Array:
    """Represent as quaternions."""
    return self.quat

  def inv(self):
    """Invert this rotation."""
    return Rotation(_inv(self.quat))

  def magnitude(self) -> jax.Array:
    """Get the magnitude(s) of the rotation(s)."""
    return _magnitude(self.quat)

  def mean(self, weights: typing.Optional[jax.Array] = None):
    """Get the mean of the rotations."""
    weights = jnp.where(weights is None, jnp.ones(self.quat.shape[0], dtype=self.quat.dtype), jnp.asarray(weights, dtype=self.quat.dtype))
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

  def reduce(self, left=None, right=None, return_indices=False):
    """Reduce this rotation with the provided rotation groups."""
    p = self.as_quat()
    l = (Rotation.identity(dtype=p.dtype) if left is None else left).as_quat()
    r = (Rotation.identity(dtype=p.dtype) if right is None else right).as_quat()
    q, left_best, right_best = _reduce(p, l, r)
    reduced = Rotation(jnp.atleast_2d(q))
    if return_indices:
      if left is None:
        left_best = None
      if right is None:
        right_best = None
      return reduced, left_best, right_best
    else:
      return reduced

  @property
  def single(self) -> bool:
    """Whether this instance represents a single rotation."""
    return self.quat.ndim == 1


@_wraps(scipy.spatial.transform.Slerp)
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
    times = jnp.asarray(times, dtype=rotations.quat.dtype)
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
    new_rotations = Rotation(rotations.as_quat()[:-1])
    return cls(
      times=times,
      timedelta=timedelta,
      rotations=new_rotations,
      rotvecs=(new_rotations.inv() * Rotation(rotations.as_quat()[1:])).as_rotvec())

  def __call__(self, times: jax.Array):
    """Interpolate rotations."""
    compute_times = jnp.asarray(times, dtype=self.times.dtype)
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


@functools.partial(jnp.vectorize, signature='(m,n),(m,n),(m)->(n,n),(),(n,n)')
def _align_vectors(a: jax.Array, b: jax.Array, weights: jax.Array) -> typing.Tuple[jax.Array, jax.Array, jax.Array]:
  B = jnp.einsum('ji,jk->ik', weights[:, jnp.newaxis] * a, b)
  u, s, vh = jnp.linalg.svd(B)
  is_neg_det = jnp.linalg.det(u @ vh) < 0
  s = s.at[-1].set(jnp.where(is_neg_det, -s[-1], s[-1]))
  u = u.at[:, -1].set(jnp.where(is_neg_det, -u[:, -1], u[:, -1]))
  C = jnp.dot(u, vh)
  # if s[1] + s[2] < 1e-16 * s[0]:
  #   warnings.warn("Optimal rotation is not uniquely or poorly defined for the given sets of vectors.")
  rssd = jnp.sqrt(jnp.maximum(jnp.sum(weights * jnp.sum(b*b + a*a, axis=1)) - 2 * jnp.sum(s), 0.))
  zeta = (s[0] + s[1]) * (s[1] + s[2]) * (s[2] + s[0])
  kappa = s[0] * s[1] + s[1] * s[2] + s[2] * s[0]
  # with jnp.errstate(divide='ignore', invalid='ignore'):
  sensitivity = jnp.mean(weights) / zeta * (kappa * jnp.eye(3, dtype=B.dtype) + jnp.dot(B, B.T))
  return C, rssd, sensitivity


@functools.partial(jnp.vectorize, signature='(m,m),(m),()->(m)')
def _apply(matrix: jax.Array, vector: jax.Array, inverse: bool) -> jax.Array:
  return jnp.where(inverse, matrix.T, matrix) @ vector


@functools.partial(jnp.vectorize, signature='(m)->(n,n)')
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


@functools.partial(jnp.vectorize, signature='(m)->(n)')
def _as_mrp(quat: jax.Array) -> jax.Array:
  sign = jnp.where(quat[3] < 0, -1., 1.)
  denominator = 1. + sign * quat[3]
  return sign * quat[:3] / denominator


@functools.partial(jnp.vectorize, signature='(m),()->(n)')
def _as_rotvec(quat: jax.Array, degrees: bool) -> jax.Array:
  quat = jnp.where(quat[3] < 0, -quat, quat)  # w > 0 to ensure 0 <= angle <= pi
  angle = 2. * jnp.arctan2(_vector_norm(quat[:3]), quat[3])
  angle2 = angle * angle
  small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
  large_scale = angle / jnp.sin(angle / 2)
  scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
  scale = jnp.where(degrees, jnp.rad2deg(scale), scale)
  return scale * jnp.array(quat[:3])


@functools.partial(jnp.vectorize, signature='(n),(n)->(n)')
def _compose_quat(p: jax.Array, q: jax.Array) -> jax.Array:
  cross = jnp.cross(p[:3], q[:3])
  return jnp.array([p[3]*q[0] + q[3]*p[0] + cross[0],
                    p[3]*q[1] + q[3]*p[1] + cross[1],
                    p[3]*q[2] + q[3]*p[2] + cross[2],
                    p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]])


@functools.partial(jnp.vectorize, signature='(m),(l),(),()->(n)')
def _compute_euler_from_quat(quat: jax.Array, axes: jax.Array, extrinsic: bool, degrees: bool) -> jax.Array:
  angle_first = jnp.where(extrinsic, 0, 2)
  angle_third = jnp.where(extrinsic, 2, 0)
  axes = jnp.where(extrinsic, axes, axes[::-1])
  i = axes[0]
  j = axes[1]
  k = axes[2]
  symmetric = i == k
  k = jnp.where(symmetric, 3 - i - j, k)
  sign = jnp.array((i - j) * (j - k) * (k - i) // 2, dtype=quat.dtype)
  eps = 1e-7
  a = jnp.where(symmetric, quat[3], quat[3] - quat[j])
  b = jnp.where(symmetric, quat[i], quat[i] + quat[k] * sign)
  c = jnp.where(symmetric, quat[j], quat[j] + quat[3])
  d = jnp.where(symmetric, quat[k] * sign, quat[k] * sign - quat[i])
  angles = jnp.empty(3, dtype=quat.dtype)
  angles = angles.at[1].set(2 * jnp.arctan2(jnp.hypot(c, d), jnp.hypot(a, b)))
  case = jnp.where(jnp.abs(angles[1] - jnp.pi) <= eps, 2, 0)
  case = jnp.where(jnp.abs(angles[1]) <= eps, 1, case)
  half_sum = jnp.arctan2(b, a)
  half_diff = jnp.arctan2(d, c)
  angles = angles.at[0].set(jnp.where(case == 1, 2 * half_sum, 2 * half_diff * jnp.where(extrinsic, -1, 1)))  # any degenerate case
  angles = angles.at[angle_first].set(jnp.where(case == 0, half_sum - half_diff, angles[angle_first]))
  angles = angles.at[angle_third].set(jnp.where(case == 0, half_sum + half_diff, angles[angle_third]))
  angles = angles.at[angle_third].set(jnp.where(symmetric, angles[angle_third], angles[angle_third] * sign))
  angles = angles.at[1].set(jnp.where(symmetric, angles[1], angles[1] - jnp.pi / 2))
  angles = (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
  return jnp.where(degrees, jnp.rad2deg(angles), angles)


def _create_cyclic_group(n: int, axis: int = 2) -> jax.Array:
  thetas = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
  rv = jnp.vstack([thetas, jnp.zeros(n), jnp.zeros(n)]).T
  return _from_rotvec(jnp.roll(rv, axis, axis=1), False)


def _create_dicyclic_group(n: int, axis: int = 2) -> jax.Array:
  g1 = _as_rotvec(_create_cyclic_group(n, axis), False)
  thetas = jnp.linspace(0, jnp.pi, n, endpoint=False)
  rv = jnp.pi * jnp.vstack([jnp.zeros(n), jnp.cos(thetas), jnp.sin(thetas)]).T
  g2 = jnp.roll(rv, axis, axis=1)
  return _from_rotvec(jnp.concatenate((g1, g2)), False)


def _create_icosahedral_group() -> jax.Array:
  g1 = _create_tetrahedral_group()
  a = 0.5
  b = 0.5 / golden
  c = golden / 2
  g2 = jnp.array([[+a, +b, +c, 0],
                  [+a, +b, -c, 0],
                  [+a, +c, 0, +b],
                  [+a, +c, 0, -b],
                  [+a, -b, +c, 0],
                  [+a, -b, -c, 0],
                  [+a, -c, 0, +b],
                  [+a, -c, 0, -b],
                  [+a, 0, +b, +c],
                  [+a, 0, +b, -c],
                  [+a, 0, -b, +c],
                  [+a, 0, -b, -c],
                  [+b, +a, 0, +c],
                  [+b, +a, 0, -c],
                  [+b, +c, +a, 0],
                  [+b, +c, -a, 0],
                  [+b, -a, 0, +c],
                  [+b, -a, 0, -c],
                  [+b, -c, +a, 0],
                  [+b, -c, -a, 0],
                  [+b, 0, +c, +a],
                  [+b, 0, +c, -a],
                  [+b, 0, -c, +a],
                  [+b, 0, -c, -a],
                  [+c, +a, +b, 0],
                  [+c, +a, -b, 0],
                  [+c, +b, 0, +a],
                  [+c, +b, 0, -a],
                  [+c, -a, +b, 0],
                  [+c, -a, -b, 0],
                  [+c, -b, 0, +a],
                  [+c, -b, 0, -a],
                  [+c, 0, +a, +b],
                  [+c, 0, +a, -b],
                  [+c, 0, -a, +b],
                  [+c, 0, -a, -b],
                  [0, +a, +c, +b],
                  [0, +a, +c, -b],
                  [0, +a, -c, +b],
                  [0, +a, -c, -b],
                  [0, +b, +a, +c],
                  [0, +b, +a, -c],
                  [0, +b, -a, +c],
                  [0, +b, -a, -c],
                  [0, +c, +b, +a],
                  [0, +c, +b, -a],
                  [0, +c, -b, +a],
                  [0, +c, -b, -a]])
  return jnp.concatenate((g1, g2))


def _create_octahedral_group() -> jax.Array:
  g1 = _create_tetrahedral_group()
  c = jnp.sqrt(2) / 2
  g2 = jnp.array([[+c, 0, 0, +c],
                  [0, +c, 0, +c],
                  [0, 0, +c, +c],
                  [0, 0, -c, +c],
                  [0, -c, 0, +c],
                  [-c, 0, 0, +c],
                  [0, +c, +c, 0],
                  [0, -c, +c, 0],
                  [+c, 0, +c, 0],
                  [-c, 0, +c, 0],
                  [+c, +c, 0, 0],
                  [-c, +c, 0, 0]])
  return jnp.concatenate((g1, g2))


def _create_tetrahedral_group() -> jax.Array:
  g1 = jnp.eye(4)
  c = 0.5
  g2 = jnp.array([[c, -c, -c, +c],
                  [c, -c, +c, +c],
                  [c, +c, -c, +c],
                  [c, +c, +c, +c],
                  [c, -c, -c, -c],
                  [c, -c, +c, -c],
                  [c, +c, -c, -c],
                  [c, +c, +c, -c]])
  return jnp.concatenate((g1, g2))


def _elementary_basis_index(axis: str) -> int:
  if axis == 'x':
    return 0
  elif axis == 'y':
    return 1
  elif axis == 'z':
    return 2
  raise ValueError("Expected axis to be from ['x', 'y', 'z'], got {}".format(axis))


@functools.partial(jnp.vectorize, signature=('(m),(m),(),()->(n)'))
def _elementary_quat_compose(angles: jax.Array, axes: jax.Array, intrinsic: bool, degrees: bool) -> jax.Array:
  angles = jnp.where(degrees, jnp.deg2rad(angles), angles)
  result = _make_elementary_quat(axes[0], angles[0])
  for idx in range(1, len(axes)):
    quat = _make_elementary_quat(axes[idx], angles[idx])
    result = jnp.where(intrinsic, _compose_quat(result, quat), _compose_quat(quat, result))
  return result


@functools.partial(jnp.vectorize, signature=('(m),()->(n)'))
def _from_rotvec(rotvec: jax.Array, degrees: bool) -> jax.Array:
  rotvec = jnp.where(degrees, jnp.deg2rad(rotvec), rotvec)
  angle = _vector_norm(rotvec)
  angle2 = angle * angle
  small_scale = scale = 0.5 - angle2 / 48 + angle2 * angle2 / 3840
  large_scale = jnp.sin(angle / 2) / angle
  scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
  return jnp.hstack([scale * rotvec, jnp.cos(angle / 2)])


@functools.partial(jnp.vectorize, signature=('(m,m)->(n)'))
def _from_matrix(matrix: jax.Array) -> jax.Array:
  matrix_trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
  decision = jnp.array([matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix_trace], dtype=matrix.dtype)
  choice = jnp.argmax(decision)
  i = choice
  j = (i + 1) % 3
  k = (j + 1) % 3
  quat_012 = jnp.empty(4, dtype=matrix.dtype)
  quat_012 = quat_012.at[i].set(1 - decision[3] + 2 * matrix[i, i])
  quat_012 = quat_012.at[j].set(matrix[j, i] + matrix[i, j])
  quat_012 = quat_012.at[k].set(matrix[k, i] + matrix[i, k])
  quat_012 = quat_012.at[3].set(matrix[k, j] - matrix[j, k])
  quat_3 = jnp.empty(4, dtype=matrix.dtype)
  quat_3 = quat_3.at[0].set(matrix[2, 1] - matrix[1, 2])
  quat_3 = quat_3.at[1].set(matrix[0, 2] - matrix[2, 0])
  quat_3 = quat_3.at[2].set(matrix[1, 0] - matrix[0, 1])
  quat_3 = quat_3.at[3].set(1 + decision[3])
  quat = jnp.where(choice != 3, quat_012, quat_3)
  return _normalize_quaternion(quat)


@functools.partial(jnp.vectorize, signature='(m)->(n)')
def _from_mrp(mrp: jax.Array) -> jax.Array:
  mrp_squared_plus_1 = jnp.dot(mrp, mrp) + 1
  return jnp.hstack([2 * mrp[:3], (2 - mrp_squared_plus_1)]) / mrp_squared_plus_1


@functools.partial(jnp.vectorize, signature='(n)->(n)')
def _inv(quat: jax.Array) -> jax.Array:
  return quat.at[3].set(-quat[3])


@functools.partial(jnp.vectorize, signature='(n)->()')
def _magnitude(quat: jax.Array) -> jax.Array:
  return 2. * jnp.arctan2(_vector_norm(quat[:3]), jnp.abs(quat[3]))


@functools.partial(jnp.vectorize, signature='(),()->(n)')
def _make_elementary_quat(axis: int, angle: jax.Array) -> jax.Array:
  quat = jnp.zeros(4, dtype=angle.dtype)
  quat = quat.at[3].set(jnp.cos(angle / 2.))
  quat = quat.at[axis].set(jnp.sin(angle / 2.))
  return quat


@functools.partial(jnp.vectorize, signature='(n)->(n)')
def _normalize_quaternion(quat: jax.Array) -> jax.Array:
  return quat / _vector_norm(quat)


@functools.partial(jax.jit, static_argnames=['num', 'dtype'])
def _random_quaternion(random_key: jax.Array, num: typing.Optional[int], dtype):
  if num is None:
    sample = jax.random.normal(key=random_key, shape=(4,), dtype=dtype)
  else:
    sample = jax.random.normal(key=random_key, shape=(num, 4), dtype=dtype)
  return _normalize_quaternion(sample)


@functools.partial(jnp.vectorize, signature='(n),(n),(n)->(n),(),()')
def _reduce(p: jax.Array, l: jax.Array, r: jax.Array):
  e = jnp.zeros((3, 3, 3), dtype=p.dtype)
  e = e.at[[0, 1, 2], [1, 2, 0], [2, 0, 1]].set(1)
  e = e.at[[0, 2, 1], [2, 1, 0], [1, 0, 2]].set(-1)
  ps, pv = _split_quaternion(p)
  ls, lv = _split_quaternion(l)
  rs, rv = _split_quaternion(r)
  qs = jnp.abs(jnp.einsum('i,j,k', ls, ps, rs) -
               jnp.einsum('i,jx,kx', ls, pv, rv) -
               jnp.einsum('ix,j,kx', lv, ps, rv) -
               jnp.einsum('ix,jx,k', lv, pv, rs) -
               jnp.einsum('xyz,ix,jy,kz', e, lv, pv, rv))
  qs = jnp.reshape(jnp.moveaxis(qs, 1, 0), (qs.shape[1], -1))
  max_ind = jnp.argmax(jnp.reshape(qs, (len(qs), -1)), axis=1)
  left_best = max_ind // len(rv)
  right_best = max_ind % len(rv)
  if l.ndim > 1:
    l = l[left_best]
  if r.ndim > 1:
    r = r[right_best]
  reduced = _compose_quat(l, _compose_quat(p, r))
  if p.ndim == 1:
    reduced = p
    left_best = left_best[0]
    right_best = right_best[0]
  return reduced, left_best, right_best


def _split_quaternion(q):
  q = jnp.atleast_2d(q)
  return q[:, -1], q[:, :-1]


@functools.partial(jnp.vectorize, signature='(n)->()')
def _vector_norm(vector: jax.Array) -> jax.Array:
  return jnp.sqrt(jnp.dot(vector, vector))
