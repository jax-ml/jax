# Copyright 2018 The JAX Authors.
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

from jax._src.interpreters.xla import (
  abstractify as abstractify,
  canonicalize_dtype as canonicalize_dtype,
  canonicalize_dtype_handlers as canonicalize_dtype_handlers,
  pytype_aval_mappings as pytype_aval_mappings,
)

from jax._src.dispatch import (
  apply_primitive as apply_primitive,
)

from jax._src.lib import xla_client as _xc
Backend = _xc._xla.Client
del _xc

# Deprecations
_deprecations = {
    # Finalized 2024-10-24; remove after 2025-01-24
    "xb": (
        ("jax.interpreters.xla.xb was removed in JAX v0.4.36. "
         "Use jax.lib.xla_bridge instead."), None
    ),
    "xc": (
        ("jax.interpreters.xla.xc was removed in JAX v0.4.36. "
         "Use jax.lib.xla_client instead."), None
    ),
    "xe": (
        ("jax.interpreters.xla.xe was removed in JAX v0.4.36. "
         "Use jax.lib.xla_extension instead."), None
    ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
