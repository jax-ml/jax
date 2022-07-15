# Copyright 2022 Google LLC
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
from typing import Any, List, Optional, Tuple

import scipy
import numpy as np

from jax import core, jit, lax, vmap
import jax.numpy as jnp
from jax._src.numpy.lax_numpy import _check_arraylike
from jax._src.numpy.reductions import _isscalar
from jax._src.numpy.util import _wraps
from jax._src.util import canonicalize_axis

ModeResult = namedtuple("ModeResult", ("mode", "count"))


@_wraps(scipy.stats.mode)
@partial(jit, static_argnames=['axis', 'nan_policy'])
def mode(x: jnp.ndarray, axis: int = 0, nan_policy: str = "propagate") -> ModeResult:

  def _mode_helper(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    "Helper function to return mode and count of a given array."""
    vals, counts = jnp.unique(x, return_counts=True, size=x.size)
    return vals[jnp.argmax(counts)], jnp.max(counts)

  _check_arraylike("mode", x)
  x = jnp.asarray(x)
  if nan_policy not in {"propagate", "omit", "raise"}:
    raise ValueError(
      f"Illegal nan_policy value {nan_policy!r}; expected one of "
      "{'propoagate', 'omit', 'raise'}"
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

  if x.size == 0:
    return ModeResult(jnp.array([]), jnp.array([]))
  if axis is None or _isscalar(x):
    x = lax.reshape(x, (np.size(x),))
    axis = 0
  x_shape = list(np.shape(x))
  x_shape[axis] = 1
  x = jnp.moveaxis(x, axis, 0).reshape(x.shape[axis], -1)
  vals, counts = vmap(_mode_helper, in_axes=(1,))(x)
  return ModeResult(vals.reshape(x_shape), counts.reshape(x_shape))
