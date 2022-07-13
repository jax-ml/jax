# Copyright 2018 Google LLC
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
from collections import namedtuple
from functools import partial
from typing import Optional, List, Tuple, Iterable, Any

import scipy

from jax import jit, vmap
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _check_arraylike

ModeResult = namedtuple("ModeResult", ("mode", "count"))


@partial(jit, static_argnums=(1,))
def _mode(x: Iterable[Any], axis: Optional[int]) -> ModeResult:
  """
  Hidden jit-compliable function accessed through `jax.scipy.stats.mode`.
  """
  def _mode_helper(x: Iterable[Any]) -> Tuple[Iterable[Any], Iterable[Any]]:
    vals, counts = jnp.unique(x, return_counts=True, size=x.size)
    return vals[jnp.argmax(counts)], jnp.max(counts)

  if x.size == 0:
    return ModeResult(jnp.array([]), jnp.array([]))
  elif axis is None:
    vals, counts = _mode_helper(x)
    return ModeResult(vals.reshape(-1), counts.reshape(-1))
  else:
    new_shape: List[int] = list(x.shape)
    new_shape[axis] = 1
    x = jnp.moveaxis(x, axis, 0).reshape(x.shape[axis], -1)
    vals, counts = vmap(_mode_helper, in_axes=(1,))(x)
    return ModeResult(vals.reshape(new_shape), counts.reshape(new_shape))


@_wraps(scipy.stats.mode)
def mode(x: Iterable[Any], axis: int = 0, nan_policy: str = "propagate") -> ModeResult:
  _check_arraylike("mode", x)
  if nan_policy not in {"propagate", "omit", "raise"}:
    raise ValueError("Illegal nan_policy value")
  if nan_policy == "omit":
    raise NotImplementedError()
  if nan_policy == "raise":
    contains_nan = jnp.isnan(jnp.sum(x))
    if contains_nan:
      raise ValueError("x contains nans.")
  return _mode(x=x, axis=axis)
