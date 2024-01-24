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

from collections import namedtuple
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax import jit
from jax._src import dtypes
from jax._src.api import vmap
from jax._src.numpy.util import check_arraylike, implements, promote_args_inexact
from jax._src.typing import ArrayLike, Array
from jax._src.util import canonicalize_axis

import scipy

ModeResult = namedtuple('ModeResult', ('mode', 'count'))

@implements(scipy.stats.mode, lax_description="""\
Currently the only supported nan_policy is 'propagate'
""")
@partial(jit, static_argnames=['axis', 'nan_policy', 'keepdims'])
def mode(a: ArrayLike, axis: int | None = 0, nan_policy: str = "propagate", keepdims: bool = False) -> ModeResult:
  check_arraylike("mode", a)
  x = jnp.atleast_1d(a)

  if nan_policy not in ["propagate", "omit", "raise"]:
    raise ValueError(
      f"Illegal nan_policy value {nan_policy!r}; expected one of "
      "{'propagate', 'omit', 'raise'}"
    )
  if nan_policy == "omit":
    # TODO: return answer without nans included.
    raise NotImplementedError(
      f"Logic for `nan_policy` of {nan_policy} is not implemented"
    )
  if nan_policy == "raise":
    raise NotImplementedError(
      "In order to best JIT compile `mode`, we cannot know whether `x` contains nans. "
      "Please check if nans exist in `x` outside of the `mode` function."
    )

  input_shape = x.shape
  if keepdims:
    if axis is None:
      output_shape = tuple(1 for i in input_shape)
    else:
      output_shape = tuple(1 if i == axis else s for i, s in enumerate(input_shape))
  else:
    if axis is None:
      output_shape = ()
    else:
      output_shape = tuple(s for i, s in enumerate(input_shape) if i != axis)

  if axis is None:
    axis = 0
    x = x.ravel()

  def _mode_helper(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Helper function to return mode and count of a given array."""
    if x.size == 0:
      return (jnp.array(jnp.nan, dtype=dtypes.canonicalize_dtype(jnp.float_)),
              jnp.array(0, dtype=dtypes.canonicalize_dtype(jnp.float_)))
    else:
      vals, counts = jnp.unique(x, return_counts=True, size=x.size)
      return vals[jnp.argmax(counts)], counts.max()

  axis = canonicalize_axis(axis, x.ndim)
  x = jnp.moveaxis(x, axis, 0)
  x = x.reshape(x.shape[0], math.prod(x.shape[1:]))
  vals, counts = vmap(_mode_helper, in_axes=1)(x)
  return ModeResult(vals.reshape(output_shape), counts.reshape(output_shape))

def invert_permutation(i: Array) -> Array:
  """Helper function that inverts a permutation array."""
  return jnp.empty_like(i).at[i].set(jnp.arange(i.size, dtype=i.dtype))

@implements(scipy.stats.rankdata, lax_description="""\
Currently the only supported nan_policy is 'propagate'
""")
@partial(jit, static_argnames=["method", "axis", "nan_policy"])
def rankdata(
  a: ArrayLike,
  method: str = "average",
  *,
  axis: int | None = None,
  nan_policy: str = "propagate",
) -> Array:

  check_arraylike("rankdata", a)

  if nan_policy not in ["propagate", "omit", "raise"]:
    raise ValueError(
      f"Illegal nan_policy value {nan_policy!r}; expected one of "
      "{'propoagate', 'omit', 'raise'}"
    )
  if nan_policy == "omit":
    raise NotImplementedError(
      f"Logic for `nan_policy` of {nan_policy} is not implemented"
    )
  if nan_policy == "raise":
    raise NotImplementedError(
      "In order to best JIT compile `mode`, we cannot know whether `x` "
      "contains nans. Please check if nans exist in `x` outside of the "
      "`rankdata` function."
    )

  if method not in ("average", "min", "max", "dense", "ordinal"):
    raise ValueError(f"unknown method '{method}'")

  a = jnp.asarray(a)

  if axis is not None:
    return jnp.apply_along_axis(rankdata, axis, a, method)

  arr = jnp.ravel(a)
  sorter = jnp.argsort(arr)
  inv = invert_permutation(sorter)

  if method == "ordinal":
    return inv + 1
  arr = arr[sorter]
  obs = jnp.insert(arr[1:] != arr[:-1], 0, True)
  dense = obs.cumsum()[inv]
  if method == "dense":
    return dense
  count = jnp.nonzero(obs, size=arr.size + 1, fill_value=len(obs))[0]
  if method == "max":
    return count[dense]
  if method == "min":
    return count[dense - 1] + 1
  if method == "average":
    return .5 * (count[dense] + count[dense - 1] + 1).astype(dtypes.canonicalize_dtype(jnp.float_))
  raise ValueError(f"unknown method '{method}'")

@implements(scipy.stats.sem, lax_description="""\
Currently the only supported nan_policies are 'propagate' and 'omit'
""")
@partial(jit, static_argnames=['axis', 'nan_policy', 'keepdims'])
def sem(a: ArrayLike, axis: int | None = 0, ddof: int = 1, nan_policy: str = "propagate", *, keepdims: bool = False) -> Array:
  b, = promote_args_inexact("sem", a)
  if axis is None:
    b = b.ravel()
    axis = 0
  if nan_policy == "propagate":
    return b.std(axis, ddof=ddof) / jnp.sqrt(b.shape[axis]).astype(b.dtype)
  elif nan_policy == "omit":
    count = (~jnp.isnan(b)).sum(axis)
    return jnp.nanstd(b, axis, ddof=ddof) / jnp.sqrt(count).astype(b.dtype)
  else:
    raise ValueError(f"{nan_policy} is not supported")
