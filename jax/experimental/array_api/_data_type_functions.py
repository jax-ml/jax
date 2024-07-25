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

import jax.numpy as jnp

from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding
from jax._src import dtypes as _dtypes


# TODO(micky774): Remove when jax.numpy.astype is deprecation is completed
def astype(x, dtype, /, *, copy: bool = True, device: xc.Device | Sharding | None = None):
  src_dtype = x.dtype if hasattr(x, "dtype") else _dtypes.dtype(x)
  if (
    src_dtype is not None
    and _dtypes.isdtype(src_dtype, "complex floating")
    and _dtypes.isdtype(dtype, ("integral", "real floating"))
  ):
    raise ValueError(
      "Casting from complex to non-complex dtypes is not permitted. Please "
      "first use jnp.real or jnp.imag to take the real/imaginary component of "
      "your input."
    )
  return jnp.astype(x, dtype, copy=copy, device=device)
