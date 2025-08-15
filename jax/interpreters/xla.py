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

from jax._src.dtypes import (
  canonicalize_value as _deprecated_canonicalize_dtype,
  canonicalize_value_handlers as _canonicalize_value_handlers,
)

canonicalize_dtype_handlers = _canonicalize_value_handlers

from jax._src.dispatch import (
  apply_primitive as apply_primitive,
)

from jax._src.lib import xla_client as _xc
Backend = _xc._xla.Client
del _xc

# Deprecations
_deprecations = {
    # Finalized in JAX v0.7.0
    "abstractify": (
        (
            "jax.interpreters.xla.abstractify was deprecated in JAX v0.5.0"
            " and removed in JAX v0.7.0. jax.core.get_aval can be used as"
            " a replacement in most cases."
        ),
        None,
    ),
    "pytype_aval_mappings": (
        (
            "jax.interpreters.xla.pytype_aval_mappings was deprecated in JAX"
            " v0.5.0 and removed in JAX v0.7.0. jax.core.pytype_aval_mappings"
            " can be used as a replacement in most cases."
        ),
        None,
    ),
    # Added in JAX v0.7.0
    "canonicalize_dtype": (
        (
            "jax.interpreters.xla.canonicalize_dtype was deprecated in JAX"
            " v0.7.0 and will be removed in JAX v0.8.0. For canonicalizing"
            " dtypes, prefer jax.dtypes.canonicalize_dtype. For checking whether"
            " an object is a valid jax input, prefer jax.core.valid_jaxtype."
        ),
        _deprecated_canonicalize_dtype,
    )
}

import typing as _typing
if _typing.TYPE_CHECKING:
  canonicalize_dtype = _deprecated_canonicalize_dtype
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
