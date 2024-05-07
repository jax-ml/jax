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

import jax
from jax.numpy import isdtype
from jax._src.dtypes import issubdtype
from jax._src.numpy.util import promote_args


# TODO(micky774): Update jnp.ceil to preserve integral dtype
def ceil(x, /):
  """Rounds each element x_i of the input array x to the smallest (i.e., closest to -infinity) integer-valued number that is not less than x_i."""
  x, = promote_args("ceil", x)
  if isdtype(x.dtype, "integral"):
    return x
  return jax.numpy.ceil(x)


# TODO(micky774): Remove when jnp.clip deprecation is completed
# (began 2024-4-2) and default behavior is Array API 2023 compliant
def clip(x, /, min=None, max=None):
  """Returns the complex conjugate for each element x_i of the input array x."""
  x, = promote_args("clip", x)

  if any(jax.numpy.iscomplexobj(t) for t in (x, min, max)):
    raise ValueError(
      "Clip received a complex value either through the input or the min/max "
      "keywords. Complex values have no ordering and cannot be clipped. "
      "Please convert to a real value or array by taking the real or "
      "imaginary components via jax.numpy.real/imag respectively."
    )
  return jax.numpy.clip(x, min=min, max=max)


# TODO(micky774): Update jnp.floor to preserve integral dtype
def floor(x, /):
  """Rounds each element x_i of the input array x to the greatest (i.e., closest to +infinity) integer-valued number that is not greater than x_i."""
  x, = promote_args("floor", x)
  if isdtype(x.dtype, "integral"):
    return x
  return jax.numpy.floor(x)


# TODO(micky774): Remove when jnp.hypot deprecation is completed
# (began 2024-4-14) and default behavior is Array API 2023 compliant
def hypot(x1, x2, /):
  """Computes the square root of the sum of squares for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = promote_args("hypot", x1, x2)

  if issubdtype(x1.dtype, jax.numpy.complexfloating):
    raise ValueError(
      "hypot does not support complex-valued inputs. Please convert to real "
      "values first, such as by using jnp.real or jnp.imag to take the real "
      "or imaginary components respectively.")
  return jax.numpy.hypot(x1, x2)


# TODO(micky774): Update jnp.trunc to preserve integral dtype
def trunc(x, /):
  """Rounds each element x_i of the input array x to the nearest integer-valued number that is closer to zero than x_i."""
  x, = promote_args("trunc", x)
  if isdtype(x.dtype, "integral"):
    return x
  return jax.numpy.trunc(x)
