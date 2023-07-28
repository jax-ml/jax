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

from __future__ import annotations

from functools import partial
from typing import Callable

import scipy.spatial.distance

import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax._src.scipy.special import rel_entr
from jax._src.typing import Array, ArrayLike, DTypeLike


def _validate_vector(u: ArrayLike, dtype: DTypeLike | None = None) -> Array:
  u = jnp.asarray(u, dtype=dtype)
  if u.ndim != 1:
    raise ValueError("Input vector should be 1-D.")
  return u


def _validate_weights(w: ArrayLike, dtype: DTypeLike | None = None) -> Array:
  w = jnp.asarray(w, dtype=dtype)
  if jnp.any(w < 0):
    raise ValueError("Input weights should be all non-negative.")
  return w


def _nbool_correspond_all(
  u: Array, v: Array, w: Array | None = None
) -> tuple[Array, Array, Array, Array]:
  if u.dtype == v.dtype == bool and w is None:
    not_u = ~u
    not_v = ~v
    nff = jnp.sum(not_u & not_v)
    nft = jnp.sum(not_u & v)
    ntf = jnp.sum(u & not_v)
    ntt = jnp.sum(u & v)
  else:
    dtype = jnp.result_type(int, u.dtype, v.dtype)
    u = u.astype(dtype)
    v = v.astype(dtype)
    not_u = 1.0 - u
    not_v = 1.0 - v

    if w is not None:
      w = _validate_weights(w)
      not_u = w * not_u
      u = w * u

    nff = jnp.sum(not_u * not_v)
    nft = jnp.sum(not_u * v)
    ntf = jnp.sum(u * not_v)
    ntt = jnp.sum(u * v)

  return nff, nft, ntf, ntt


def _nbool_correspond_ft_tf(
  u: Array, v: Array, w: Array | None = None
) -> tuple[Array, Array]:
  if u.dtype == v.dtype == bool and w is None:
    not_u = ~u
    not_v = ~v
    nft = jnp.sum(not_u & v)
    ntf = jnp.sum(u & not_v)
  else:
    dtype = jnp.result_type(int, u.dtype, v.dtype)
    u = u.astype(dtype)
    v = v.astype(dtype)
    not_u = 1.0 - u
    not_v = 1.0 - v

    if w is not None:
      w = _validate_weights(w)
      not_u = w * not_u
      u = w * u

    nft = jnp.sum(not_u * v)
    ntf = jnp.sum(u * not_v)

  return nft, ntf


@_wraps(scipy.spatial.distance.braycurtis)
def braycurtis(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  l1_diff = jnp.abs(u - v)
  l1_sum = jnp.abs(u + v)

  if w is not None:
    w = _validate_weights(w)
    l1_diff = w * l1_diff
    l1_sum = w * l1_sum

  return jnp.sum(l1_diff) / jnp.sum(l1_sum)


@_wraps(scipy.spatial.distance.canberra)
def canberra(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  abs_uv = jnp.abs(u - v)
  abs_u = jnp.abs(u)
  abs_v = jnp.abs(v)
  d = abs_uv / (abs_u + abs_v)

  if w is not None:
    w = _validate_weights(w)
    d = d * w

  return jnp.nansum(d)


@_wraps(scipy.spatial.distance.chebyshev)
def chebyshev(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)

  if w is not None:
    w = _validate_weights(w)
    has_weight = w > 0
    if jnp.sum(has_weight) < w.size:
      u = u[has_weight]
      v = v[has_weight]

  return jnp.max(jnp.abs(u - v))


@_wraps(scipy.spatial.distance.cityblock)
def cityblock(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)

  l1_diff = jnp.abs(u - v)
  if w is not None:
    w = _validate_weights(w)
    l1_diff = w * l1_diff

  return jnp.sum(l1_diff)


@_wraps(scipy.spatial.distance.correlation)
def correlation(
  u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None, centered: bool = True,
) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  if w is not None:
    w = _validate_weights(w)

  if centered:
    umu = jnp.average(u, weights=w)
    vmu = jnp.average(v, weights=w)
    u = u - umu
    v = v - vmu

  uv = jnp.average(u * v, weights=w)
  uu = jnp.average(jnp.square(u), weights=w)
  vv = jnp.average(jnp.square(v), weights=w)
  dist = 1.0 - uv / jnp.sqrt(uu * vv)

  # Return absolute value to avoid small negative value due to rounding
  return jnp.abs(dist)


@_wraps(scipy.spatial.distance.cosine)
def cosine(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  corr = correlation(u, v, w=w, centered=False)
  return jnp.clip(corr, 0.0, 2.0)


@_wraps(scipy.spatial.distance.euclidean)
def euclidean(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  return minkowski(u, v, p=2, w=w)


@_wraps(scipy.spatial.distance.jensenshannon)
def jensenshannon(
  p: ArrayLike,
  q: ArrayLike,
  base: int | None = None,
  *,
  axis: int = 0,
  keepdims: bool = False,
) -> Array:
  p = jnp.asarray(p)
  q = jnp.asarray(q)

  p = p / jnp.sum(p, axis=axis, keepdims=True)
  q = q / jnp.sum(q, axis=axis, keepdims=True)
  m = (p + q) / 2.0
  left = rel_entr(p, m)
  right = rel_entr(q, m)
  left_sum = jnp.sum(left, axis=axis, keepdims=keepdims)
  right_sum = jnp.sum(right, axis=axis, keepdims=keepdims)
  js = left_sum + right_sum
  if base is not None:
    js = js / jnp.log(base)

  return jnp.sqrt(js / 2.0)


@_wraps(scipy.spatial.distance.euclidean)
def minkowski(
  u: ArrayLike, v: ArrayLike, p: int = 2, w: ArrayLike | None = None
) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)

  if p <= 0:
    raise ValueError("p must be greater than 0")

  uv_diff = u - v

  if w is not None:
    w = _validate_weights(w)
    if p == 1:
      root_w = w
    elif p == 2:
      root_w = jnp.sqrt(w)
    else:
      root_w = jnp.power(w, 1 / p)

    uv_diff = root_w * uv_diff

  return jnp.linalg.norm(uv_diff, ord=p)


@_wraps(scipy.spatial.distance.mahalanobis)
def mahalanobis(u: ArrayLike, v: ArrayLike, VI: ArrayLike) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  VI = jnp.atleast_2d(VI)

  uv_diff = u - v
  m = jnp.dot(jnp.dot(uv_diff, VI), uv_diff)

  return jnp.sqrt(m)


@_wraps(scipy.spatial.distance.seuclidean)
def seuclidean(u: ArrayLike, v: ArrayLike, V: ArrayLike) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  V = _validate_vector(V, dtype=jnp.float64)

  if V.shape[0] != u.shape[0] or u.shape[0] != v.shape[0]:
    raise TypeError("V must be a 1-D array of the same dimension "
                    "as u and v.")

  return euclidean(u, v, w=1/V)


@_wraps(scipy.spatial.distance.sqeuclidean)
def sqeuclidean(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  utype, vtype = None, None
  if not (hasattr(u, "dtype") and jnp.issubdtype(u.dtype, jnp.inexact)):
    utype = jnp.float64
  if not (hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.inexact)):
    vtype = jnp.float64

  u = _validate_vector(u, dtype=utype)
  v = _validate_vector(v, dtype=vtype)

  uv_diff = u - v
  w_uv_diff = uv_diff  # only want weights applied once

  if w is not None:
    w = _validate_weights(w)
    w_uv_diff = w * uv_diff

  return jnp.dot(uv_diff, w_uv_diff)


@_wraps(scipy.spatial.distance.dice)
def dice(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)

  if u.dtype == v.dtype == bool and w is None:
    ntt = jnp.sum(u & v)
  else:
    dtype = jnp.result_type(int, u.dtype, v.dtype)
    u = u.astype(dtype)
    v = v.astype(dtype)

    if w is None:
      ntt = jnp.sum(u * v)
    else:
      w = _validate_weights(w)
      ntt = jnp.sum(u * v * w)

  (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
  return (ntf + nft) / jnp.array(2.0 * ntt + ntf + nft)


@_wraps(scipy.spatial.distance.hamming)
def hamming(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  if w is not None:
    w = _validate_weights(w)

  return jnp.average(u != v, weights=w)


@_wraps(scipy.spatial.distance.jaccard)
def jaccard(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)

  nonzero = jnp.bitwise_or(u != 0, v != 0)
  unequal_nonzero = jnp.bitwise_and((u != v), nonzero)

  if w is not None:
    w = _validate_weights(w)
    nonzero = w * nonzero
    unequal_nonzero = w * unequal_nonzero

  a = jnp.sum(unequal_nonzero)
  b = jnp.sum(nonzero)

  if b != 0:
    return a / b
  else:
    return b


@_wraps(scipy.spatial.distance.kulczynski1)
def kulczynski1(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  if w is not None:
      w = _validate_weights(w)

  _, nft, ntf, ntt = _nbool_correspond_all(u, v, w=w)

  return ntt / (ntf + nft)


@_wraps(scipy.spatial.distance.rogerstanimoto)
def rogerstanimoto(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  if w is not None:
      w = _validate_weights(w)

  (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v, w=w)
  num = 2.0 * (ntf + nft)
  denom = num + nff + ntt

  return num / denom


@_wraps(scipy.spatial.distance.russellrao)
def russellrao(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)

  if u.dtype == v.dtype == bool and w is None:
    ntt = jnp.sum(u & v)
    n = float(len(u))
  elif w is None:
    ntt = jnp.sum(u * v)
    n = float(len(u))
  else:
    w = _validate_weights(w)
    ntt = jnp.sum(u * v * w)
    n = float(jnp.sum(w))

  return (n - ntt) / n


@_wraps(scipy.spatial.distance.sokalmichener)
def sokalmichener(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  if w is not None:
      w = _validate_weights(w)

  nff, nft, ntf, ntt = _nbool_correspond_all(u, v, w=w)
  num = 2.0 * (ntf + nft)
  denom = num + nff + ntt

  return num / denom


@_wraps(scipy.spatial.distance.sokalsneath)
def sokalsneath(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)

  if u.dtype == v.dtype == bool and w is None:
    ntt = jnp.sum(u & v)
  elif w is None:
    ntt = jnp.sum(u * v)
  else:
    w = _validate_weights(w)
    ntt = jnp.sum(u * v * w)

  nft, ntf = _nbool_correspond_ft_tf(u, v, w=w)
  num = 2.0 * (ntf + nft)
  denom = num + ntt

  if not jnp.any(denom):
    raise ValueError('Sokal-Sneath dissimilarity is not defined for '
                     'vectors that are entirely false.')

  return num / denom


@_wraps(scipy.spatial.distance.yule)
def yule(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = None) -> Array:
  u = _validate_vector(u)
  v = _validate_vector(v)
  if w is not None:
    w = _validate_weights(w)

  nff, nft, ntf, ntt = _nbool_correspond_all(u, v, w=w)
  half_R = ntf * nft

  if half_R == 0:
    return half_R
  else:
    return 2.0 * half_R / (ntt * nff + half_R)


@_wraps(scipy.spatial.distance.directed_hausdorff)
def directed_hausdorff(u: ArrayLike, v: ArrayLike, seed: int = 0) -> Array:
  # TODO
  return jnp.double(0.0)


@_wraps(scipy.spatial.distance.cdist)
def cdist(
  XA: ArrayLike,
  XB: ArrayLike,
  metric: Callable | str = "euclidean",
  *,
  out: ArrayLike | None = None,
  **kwargs,
) -> Array:
  # TODO: validate inputs
  # TODO: use out parameter

  if callable(metric):
    # TODO: validate callable
    metric_fn = metric
  elif isinstance(metric, str):
    metric_fn = locals()[metric]
  else:
    raise TypeError("metric must be a callable or a string.")

  metric_fn_partial = partial(metric_fn, **kwargs)

  return jax.vmap(lambda xa: jax.vmap(lambda xb: metric_fn_partial(xa, xb))(XB))(XA)


@_wraps(scipy.spatial.distance.pdist)
def pdist(
  X: ArrayLike,
  metric: Callable | str = "euclidean",
  *,
  out: ArrayLike | None = None,
  **kwargs
) -> Array:
  # TODO
  return jnp.double(0.0)


@_wraps(scipy.spatial.distance.squareform)
def squareform(X: ArrayLike, force: str = "no", checks: bool = True) -> Array:
  # TODO
  return jnp.double(0.0)
