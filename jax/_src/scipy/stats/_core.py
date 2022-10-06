# Copyright 2022 The JAX Authors.
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
from typing import Optional, Tuple

import jax.numpy as jnp
import scipy
from jax import jit
from jax._src import dtypes
from jax._src.api import vmap
from jax._src.numpy.lax_numpy import _check_arraylike
from jax._src.numpy.util import _wraps
from jax._src.typing import ArrayLike
from jax._src.util import canonicalize_axis, prod

ModeResult = namedtuple('ModeResult', ('mode', 'count'))

@_wraps(scipy.stats.mode, lax_description="""\
Currently the only supported nan_policy is 'propagate'
""")
@partial(jit, static_argnames=['axis', 'nan_policy', 'keepdims'])
def mode(a: ArrayLike, axis: Optional[int] = 0, nan_policy: str = "propagate", keepdims: bool = False) -> ModeResult:
  _check_arraylike("mode", a)
  x = jnp.atleast_1d(a)

  if nan_policy not in ["propagate", "omit", "raise"]:
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

  def _mode_helper(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Helper function to return mode and count of a given array."""
    if x.size == 0:
      return jnp.array(jnp.nan, dtype=dtypes.canonicalize_dtype(jnp.float_)), jnp.array(jnp.nan, dtype=dtypes.canonicalize_dtype(jnp.float_))
    else:
      vals, counts = jnp.unique(x, return_counts=True, size=x.size)
      return vals[jnp.argmax(counts)], counts.max()

  axis = canonicalize_axis(axis, x.ndim)
  x = jnp.moveaxis(x, axis, 0)
  x = x.reshape(x.shape[0], prod(x.shape[1:]))
  vals, counts = vmap(_mode_helper, in_axes=1)(x)
  return ModeResult(vals.reshape(output_shape), counts.reshape(output_shape))
