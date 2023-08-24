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

import scipy.integrate

from jax import jit
from jax._src.numpy import util
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp

@util._wraps(scipy.integrate.trapezoid)
@partial(jit, static_argnames=('axis',))
def trapezoid(y: ArrayLike, x: ArrayLike | None = None, dx: ArrayLike = 1.0,
              axis: int = -1) -> Array:
  # TODO(phawkins): remove this annotation after fixing jnp types.
  dx_array: Array
  if x is None:
    util.check_arraylike('trapz', y)
    y_arr, = util.promote_dtypes_inexact(y)
    dx_array = jnp.asarray(dx)
  else:
    util.check_arraylike('trapz', y, x)
    y_arr, x_arr = util.promote_dtypes_inexact(y, x)
    if x_arr.ndim == 1:
      dx_array = jnp.diff(x_arr)
    else:
      dx_array = jnp.moveaxis(jnp.diff(x_arr, axis=axis), axis, -1)
  y_arr = jnp.moveaxis(y_arr, axis, -1)
  return 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).sum(-1)
