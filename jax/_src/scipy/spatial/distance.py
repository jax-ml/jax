# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License as , Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing as , software
# distributed under the License is distributed on an "AS IS" BASIS as ,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND as , either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial
from typing import Union, Optional

import scipy.spatial.distance

import jax.numpy as jnp
from jax import jit
from jax import vmap
from jax._src.numpy.linalg import norm
from jax._src.numpy.util import (
    _wraps, promote_dtypes_inexact)
from jax._src.typing import Array, ArrayLike


# TODO degan validate the input shapes in functions

def _validate_vector(u, dtype=None):

    u = jnp.asarray(u, dtype=dtype)
    if u.ndim == 1:
        return u
    return ValueError("Input vector should be 1-D.")

@partial(jit, static_argnames=('p',))
def _minkowski(u: ArrayLike, v: ArrayLike, p: int, w: ArrayLike) -> Union[float, Array]:

    if p <= 0:
        raise ValueError("p must be >= 0")

    # TODO degan
    #  Use promote_dtypes or promote_dtypes_inexact or neither?
    #  Use check_arraylike?
    u, v, = promote_dtypes_inexact(jnp.asarray(u), jnp.asarray(v))

    if w is not None:
        w, = promote_dtypes_inexact(jnp.asarray(w))
        if p == 1:
            r_w = w
        elif p == 2:
            r_w = jnp.sqrt(w)   # jnp.sqrt ~6x faster than jnp.power
        elif p == jnp.inf:
            r_w = (w != 0)
        else:
            r_w = jnp.power(w, 1/p)
    else:
        r_w = 1

    return norm((u-v) * r_w, ord=p)


@_wraps(scipy.spatial.distance.minkowski)
def minkowski(u: ArrayLike, v: ArrayLike, p: int = 2, w: Optional[ArrayLike] = None):

    if p <= 0:
        raise ValueError("p must be greater than 0")

    u_v = u - v

    if w is not None:
        if p == 1:
            root_w = w
        elif p == 2:
            root_w = jnp.sqrt(w)
        elif p == jnp.inf:
            root_w = (w != 0)
        else:
            root_w = jnp.power(w, 1/p)

        u_v = root_w * u_v

    return jnp.linalg.norm(u_v, ord=p)


@_wraps(scipy.spatial.distance.euclidean)
def euclidean(u, v, w: Optional[ArrayLike] = None) -> float:
    return _minkowski(u, v, p=2, w=w)


@partial(jit, static_argnames=('w',))
def _sqeuclidean(u: ArrayLike, v: ArrayLike, w: ArrayLike):
    u_v = u - v
    return jnp.dot(u_v, u_v * (w if w is not None else 1))

@_wraps(scipy.spatial.distance.sqeuclidean)
def sqeuclidean(u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = None) -> float:
    return _sqeuclidean(u, v, w=w)


@partial(jit, static_argnames=('w', 'centered'))
def _correlation(u: ArrayLike, v: ArrayLike, w: ArrayLike, centered: ArrayLike) -> float:
    u, v, = promote_dtypes_inexact(jnp.asarray(u), jnp.asarray(v))
    if centered:
        w, = promote_dtypes_inexact(jnp.asarray(w))
        u = u - jnp.average(u, weights=w)
        v = v - jnp.average(v, weights=w)

    uv = jnp.average(u * v, weights=w)
    uu = jnp.average(jnp.square(u), weights=w)
    vv = jnp.average(jnp.square(v), weights=w)

    return jnp.abs(1.0 - uv / jnp.sqrt(uu * vv))


@_wraps(scipy.spatial.distance.correlation)
def correlation(u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = None, centered: Optional[ArrayLike] = True):
    return _correlation(u, v, w=w, centered=centered)


@jit
@_wraps(scipy.spatial.distance.cosine)
def cosine(u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = None) -> float:
    return jnp.maximum(0, jnp.minimum(correlation(u, v, w=w, centered=False), 2))


@_wraps(scipy.spatial.distance.braycurtis)
def braycurtis(x: ArrayLike, y: ArrayLike) -> Array:
    return jnp.sum(jnp.abs(x - y), axis=-1) / jnp.sum(jnp.abs(x + y), axis=-1)


@_wraps(scipy.spatial.distance.canberra)
def canberra(x: ArrayLike, y: ArrayLike) -> float:
    return jnp.sum(jnp.abs(x - y) / (jnp.abs(x) + jnp.abs(y)), axis=-1)

@_wraps(scipy.spatial.distance.cityblock)
def cityblock(x: ArrayLike, y: ArrayLike) -> float:
    return jnp.sum(jnp.abs(x - y), axis=-1)


@_wraps(scipy.spatial.distance.chebyshev)
def chebyshev(x: ArrayLike, y: ArrayLike) -> float:
    return jnp.max(jnp.abs(x - y), axis=-1)


@_wraps(scipy.spatial.distance.dice)
def dice(x: ArrayLike, y: ArrayLike) -> float:
    return jnp.sum(x != y, axis=-1) / (jnp.sum(x != 0, axis=-1) + jnp.sum(y != 0, axis=-1))


@_wraps(scipy.spatial.distance.hamming)
def hamming(x: ArrayLike, y: ArrayLike) -> float:
    return jnp.sum(x != y, axis=-1) / x.shape[-1]


@_wraps(scipy.spatial.distance.jaccard)
def jaccard(x: ArrayLike, y: ArrayLike) -> float:
    return jnp.sum(x != y, axis=-1) / jnp.sum((x != y) | (x != 0) | (y != 0), axis=-1)


@_wraps(scipy.spatial.distance.kulczynski)
def kulczynski(x: ArrayLike, y: ArrayLike) -> float:
    return (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1)) / (jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1))


@_wraps(scipy.spatial.distance.mahalanobis)
def mahalanobis(x: ArrayLike, y: ArrayLike, v: ArrayLike) -> float:
    return jnp.sqrt(jnp.sum(((x - y) / v) ** 2, axis=-1))


@_wraps(scipy.spatial.distance.matching)
def matching(x: ArrayLike, y: ArrayLike) -> float:
    return hamming(x, y)


@_wraps(scipy.spatial.distance.rogerstanimoto)
def rogerstanimoto(x: ArrayLike, y: ArrayLike) -> float:
    return (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1)) / (2 * jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1))


@_wraps(scipy.spatial.distance.russellrao)
def russellrao(x: ArrayLike, y: ArrayLike) -> float:
    return jnp.sum(x != y, axis=-1) / x.shape[-1]


@_wraps(scipy.spatial.distance.seuclidean)
def seuclidean(x: ArrayLike, y: ArrayLike, V: ArrayLike) -> float:
    return euclidean(x, y, w=1/V)


@_wraps(scipy.spatial.distance.sokalmichener)
def sokalmichener(x: ArrayLike, y: ArrayLike) -> float:
    return (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1))


@_wraps(scipy.spatial.distance.sokalsneath)
def sokalsneath(x: ArrayLike, y: ArrayLike) -> float:
    return (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1)) / (jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1))


@_wraps(scipy.spatial.distance.yule)
def yule(x: ArrayLike, y: ArrayLike) -> float:
    return (jnp.sum(x != y, axis=-1) - jnp.sum(x == y, axis=-1)) / (jnp.sum(x != y, axis=-1) + jnp.sum(x == y, axis=-1))


# define primitives for distance functions according to scipy.spatial.distance
# which can be batched over the first dimension and then used inside jit and vmap
# as well as used for implementing cdist and pdist versions
primitives = {
    braycurtis,
    canberra,
    chebyshev,
    cityblock,
    correlation,
    cosine,
    dice,
    euclidean,
    hamming,
    jaccard,
    kulczynski,
    mahalanobis,
    matching,
    minkowski,
    rogerstanimoto,
    russellrao,
    seuclidean,
    sokalmichener,
    sokalsneath,
    sqeuclidean,
    yule
}

# vectorize primitives for batched, jit'ed evaluation. Take special care for the primitives that
# take additional arguments (p, V)
_METRICS = {}
for fun in primitives:
    name = fun.__name__
    if name in ("minkowski", "mahalanobis", "seuclidean"):
        _METRICS[name] = jit(vmap(fun, in_axes=(0, 0, None)))
    else:
        _METRICS[name] = jit(vmap(fun))


@_wraps(scipy.spatial.disance.pdist)
def pdist(X, metric="euclidean", p=2, V=None):
    if metric not in _METRICS:
        raise ValueError(f"Unknown distance metric {metric}.")

    if metric in ("mahalanobis", "seuclidean"):
        if V is None:
            raise ValueError(
                "Variance vector V must be specified for Mahalanobis distance."
            )
        V = jnp.asarray(V)
        if V.ndim != 1:
            raise ValueError("Variance vector V must be one-dimensional.")

    if callable(metric):
        return metric(X)

    if metric == "minkowski":
        return _METRICS[metric](X, p=p)

    return _METRICS[metric](X, X, V=V)
